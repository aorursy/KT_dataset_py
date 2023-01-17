import numpy as np 

from scipy import interpolate

import pandas as pd 

import os

import shutil

import warnings



import pydicom

import cv2



from tqdm import tqdm_notebook



!pip install swifter

import swifter

from joblib import Parallel, delayed

import multiprocessing



from sklearn.model_selection import train_test_split



from matplotlib import pyplot as plt

import matplotlib.patches as patches

from imageio import imwrite

%matplotlib inline 
pd.set_option('display.max_columns', None)
# Paths of raw data

BASE_PATH = '/kaggle/input/rsna-intracranial-hemorrhage-detection/'

TRAIN_DIR = BASE_PATH + 'stage_1_train_images/'

SUB_DIR = BASE_PATH + 'stage_1_test_images/'



# Paths to save images after pre-processing and to be used for training

TRAIN_PNG = '/kaggle/tmp/train/'

TEST_PNG = '/kaggle/tmp/test/'

SUB_PNG = '/kaggle/tmp/sub/'



# Path to save CSV files corresponents to the images saved

CSV_DIR = './csv/'



# Classes to classify in the multi-label training

CLASSES = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']



# As the training dataset is huge we could only train on a part of the whole training directory

FRACTION_TRAINING = 0.2



# Image size which Xception model was trained

img_size = (299, 299)



# Minimum area that a training/validation image need to contain of bran

USE_MIN_AREA = False # Currently disabled

MIN_AREA = 200*200 # From an image of 512*512 



NUM_CORES = multiprocessing.cpu_count()



SEED = 42

np.random.seed(seed=SEED)
def load_csv(path):

    df = pd.read_csv(path)

    df['filename'] = df['ID'].apply(lambda st: "ID_" + st.split('_')[1] + ".png")

    df['type'] = df['ID'].apply(lambda st: st.split('_')[2])

    return df
data_df = load_csv(BASE_PATH + 'stage_1_train.csv')

data_df.drop_duplicates(['filename','type'], inplace=True)

data_df = data_df.pivot('filename', 'type', 'Label').reset_index() # Extract Labels

sub_df = load_csv(BASE_PATH + 'stage_1_sample_submission.csv')

sub_df = pd.DataFrame(sub_df.filename.unique(), columns=['filename'])
print(f"Data shape: {data_df.shape}")

data_df.head()
print(f"Submission shape: {sub_df.shape}")

sub_df.head()
# There are 3 groups, but we will create a fourth for Others:

# 1) Bits Stored 16bits

# 2) Bits Stored 12bits - Pixel Representation 0

# 3) Bits Stored 12bits - Pixel Representation 1

# -1) Others (in case new data appears)

def _subgroup(res):

    if res['BitsStored'] == 16:

        res['SubGroup'] = 1

    elif res['BitsStored'] == 12 and res['PixelRepresentation'] == 0:

        res['SubGroup'] = 2

    elif res['BitsStored'] == 12 and res['PixelRepresentation'] == 1:

        res['SubGroup'] = 3

    else:

        res['SubGroup'] = -1



def _cast_dicom_special(x):

    cls = type(x)

    if not cls.__module__.startswith('pydicom'): return x

    return cls.__base__(x)



# Extract data from values

def _split_elem(res,k,v):

    if not isinstance(v,pydicom.multival.MultiValue): return

    res[f'Multi{k}'] = 1

    for i,o in enumerate(v): res[f'{k}{"" if i==0 else i}']=o



# Transform DICOM data to dictionary

def as_dict(dcm, px_summ=True):

    pxdata = (0x7fe0,0x0010)

    vals = [dcm[o] for o in dcm.keys() if o != pxdata]

    its = [(v.keyword,v.value) for v in vals]

    res = dict(its)

    res['fname'] = dcm.filename

    for k,v in its: _split_elem(res,k,v)

    _subgroup(res)

    if not px_summ: return res

    stats = 'min','max','mean','std'

    try:

        pxs = dcm.pixel_array

        for f in stats: res['img_'+f] = getattr(pxs,f)()

    except Exception as e:

        for f in stats: res['img_'+f] = 0

        print(res,e)

    for k in res: res[k] = _cast_dicom_special(res[k])

    return res



# Function used in apply function to fill row with DICOM data

def fill_row(row, px_summ = True):

    row_dict = as_dict(pydicom.dcmread(row.paths), px_summ)

    for key in row_dict:

        row[key] = row_dict[key]



def get_all_metadata(load_dir, px_summ = True, n_sample = None, dcm_dir = None):

    if dcm_dir is None:

        dcm_dir = load_dir

    filenames = os.listdir(load_dir)

    if not n_sample is None:

        filenames = np.random.choice(filenames, size = n_sample, replace = False)

    dcm_paths = [dcm_dir + filename[:-3] + 'dcm' for filename in filenames]

    filenames = [filename[:-3] + 'png' for filename in filenames.copy()]

    sample_dict = as_dict(pydicom.dcmread(dcm_paths[0]), px_summ = True)

    sample_dict['paths'] = dcm_paths[0]

    columns = list(sample_dict.keys())

    metadata_df = pd.DataFrame({'paths' : dcm_paths},columns = columns, index = filenames)

    metadata_df.swifter.apply(fill_row, axis=1, px_summ = px_summ)

    return metadata_df



def get_file_metadata(load_dir, filename, px_summ = True):

    dcm_path = load_dir + filename[:-3] + 'dcm'

    filename = filename[:-3] + 'png'

    sample_dict = as_dict(pydicom.dcmread(dcm_path), px_summ)

    sample_dict['paths'] = dcm_path

    columns = list(sample_dict.keys())

    metadata_df = pd.DataFrame(sample_dict, columns = columns, index = [filename])

    return metadata_df
# To big to be stored all in-memory (we only could stored 50.000 files)

# train_meta = get_all_metadata(TRAIN_DIR, px_summ = False, n_sample = 50000)
# From https://radiopaedia.org/articles/windowing-ct

dicom_windows = {

    'brain' : (80,40),

    'subdural':(200,80),

    'stroke':(8,32),

    'brain_bone':(2800,600),

    'brain_soft':(375,40),

    'lungs':(1500,-600),

    'mediastinum':(350,50),

    'abdomen_soft':(400,50),

    'liver':(150,30),

    'spine_soft':(250,50),

    'spine_bone':(1800,400)

}
def crop_brain(image, n_top_areas = 5, max_scale_diff = 3, plot = False):

    image = normalize_img(image, use_min_max = True)

    if (image.max() - image.min()) == 0 or np.isnan(image.max()) or np.isnan(image.min()):

        raise ValueError('Empty image')

    gray = np.uint8(image * 255)

    blur = cv2.blur(gray, (5, 5)) # blur the image

    # Detect edges using Canny

    #canny_output = cv2.Canny(blur, threshold, threshold * 2)

    # Find contours

    contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    # Cycle through contours and add area to array

    areas = []

    for c in contours:

        areas.append(cv2.contourArea(c))



    # Sort array of areas by size

    sorted_areas = sorted(zip(areas, contours), key=lambda x: x[0], reverse=True)

    biggest_area = sorted_areas[0][0]



    # Approximate contours to polygons + get bounding rects and circles

    contours_poly = []

    boundRect = []

    min_dist_to_center = np.inf

    best_contour_idx = 0

    for i, c in enumerate(sorted_areas):

        # Only treat contours which are in top 5 and less than 'max_scale_diff' times smaller than the biggest one

        if c[0] > 0 and i < n_top_areas and biggest_area/c[0] < max_scale_diff:

            contour_poly = cv2.approxPolyDP(c[1], 3, True)

            contours_poly.append(contour_poly)

            boundRect.append(cv2.boundingRect(contour_poly))

            center, _ = cv2.minEnclosingCircle(contour_poly)



            # Calculate distance from contour center to center of image

            dist = (gray.shape[0]//2 - center[0])**2 + (gray.shape[1]//2 - center[1])**2

            if min_dist_to_center > dist:

                best_contour_idx = i

                min_dist_to_center = dist

        else:

            break



    # Get boundaries of the Rectangle which includes the contour

    x,y,w,h = boundRect[best_contour_idx]

    # Crop the image

    cropped = image[y:y+h,x:x+w]

    # Pad needed pixels

    final_image = pad_square(cropped)

    

    # Show three images (original, cropped, final)

    if plot:

        fig=plt.figure(figsize  = (10,30))    

        ax = fig.add_subplot(1, 3, 1)

        plt.imshow(image)

        ax.add_patch(patches.Rectangle(

            (x, y),

            w,

            h,

            fill=False      # remove background

         )) 

        fig.add_subplot(1, 3, 2)

        plt.imshow(cropped)

        fig.add_subplot(1, 3, 3)

        plt.imshow(final_image)

        plt.show()

    return final_image, sorted_areas[best_contour_idx]



def pad_square(x):

    r,c = x.shape

    d = (c-r)/2

    pl,pr,pt,pb = 0,0,0,0

    if d>0: pt,pd = int(np.floor( d)),int(np.ceil( d))        

    else:   pl,pr = int(np.floor(-d)),int(np.ceil(-d))

    return np.pad(x, ((pt,pb),(pl,pr)), 'minimum')
def get_first_of_dicom_field_as_int(x):

    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)



def get_windowing(data):

    dicom_fields = [data[('0028','1050')].value, #window center

                    data[('0028','1051')].value, #window width

                    data[('0028','1052')].value, #intercept

                    data[('0028','1053')].value] #slope

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]



def get_subgroups(data):

    # There are 3 groups, but we will create a fourth for Others:

    # 1) Bits Stored 16bits

    # 2) Bits Stored 12bits - Pixel Representation 0

    # 3) Bits Stored 12bits - Pixel Representation 1

    # -1) Others (in case new data appears)

    dicom_fields = [data[('0028', '0101')].value, #Bits Stored

                    data[('0028', '0103')].value] #Pixel Representation

    dicom_values = [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

    if dicom_values[0] == 16:

        return 1

    elif dicom_values[0] == 12 and dicom_values[1] == 0:

        return 2

    elif dicom_values[0] == 12 and dicom_values[1] == 1:

        return 3

    else:

        return -1



# According to https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai

def correct_dcm(dcm):

    x = dcm.pixel_array + 1000

    px_mode = 4096

    x[x>=px_mode] = x[x>=px_mode] - px_mode

    dcm.PixelData = x.tobytes()

    dcm.RescaleIntercept = -1000

    return dcm.pixel_array, dcm.RescaleIntercept



def get_freqhist_bins(dcm_img, n_bins = 100):

    imsd = np.sort(dcm_img.reshape(-1))

    t = np.concatenate([[0.001],

                       np.arange(n_bins).astype(np.float64)/n_bins+(1/2/n_bins),

                       [0.999]])

    t = (len(imsd)*t).astype(np.int64)

    return np.unique(imsd[t])



def get_dcm_img(path, window_type = 'brain'):

    # Read and scale of DICOM images according to its metadata

    dcm = pydicom.dcmread(path)

    window_center, window_width, intercept, slope = get_windowing(dcm)

    group = get_subgroups(dcm)   

    

    if group == 2 and (int(intercept) > -100):

        dcm_img, intercept = correct_dcm(dcm)

    

    dcm_img = dcm.pixel_array.astype(np.float64) 

    dcm_img = dcm_img * slope + intercept

    

    min_px = dicom_windows[window_type][1] - dicom_windows[window_type][0]//2

    max_px = dicom_windows[window_type][1] + dicom_windows[window_type][0]//2

    if min_px is not None: dcm_img[dcm_img<min_px] = min_px

    if max_px is not None: dcm_img[dcm_img>max_px] = max_px

    if (dcm_img.max() - dcm_img.min()) == 0:

        dcm_img[:, :] = 0

        warnings.warn('Empty image from path: ' + path, UserWarning)

    

    return dcm_img, group



def interpolate_img(dcm_img, bins = None, n_bins = 100):

    # Equal distribution of intensity

    if bins is None: 

        bins = get_freqhist_bins(dcm_img, n_bins)

    

    return np.clip(interpolate.interp1d(bins, np.linspace(0., 1., len(bins)), fill_value="extrapolate")(dcm_img.flatten()).reshape(dcm_img.shape), 0., 1.)



def normalize_img(dcm_img, mean = None, std = None, use_min_max = False):

    # Normalization to zero mean and unit variance

    if use_min_max:

        if (dcm_img.max() - dcm_img.min()) != 0:

            return (dcm_img - dcm_img.min()) / (dcm_img.max() - dcm_img.min())

        else:

            return dcm_img

    else:

        if mean is None: 

            mean = dcm_img.mean()



        if std is None: 

            std = dcm_img.std()

        return (dcm_img - mean) / std

    



def preprocess_dicom(path, x, y, bins = None, n_bins = 100, mean = None, std = None, use_min_max = False, remove_empty = False): 

    dcm_img, group = get_dcm_img(path)

    

    # Crop image to show only the brain part (only posible if the image is not empyt)

    try:

        isEmpty = False

        dcm_img, area = crop_brain(dcm_img)

    except ValueError as e:

        isEmpty = True

        area = 0

        print("DICOM image from ", path, " is not treated because gave the following error: ", e)

    finally:

        if isEmpty and remove_empty:

            return None, None

        else:

            # If distributed by groups (different than -1) then use only the values of the group

            if group != -1:

                if type(bins) == dict:

                    bins = bins[group]

                if type(mean) == dict:

                    mean = mean[group]

                if type(std) == dict:

                    std = std[group]



            dcm_img = interpolate_img(dcm_img, bins, n_bins)

            dcm_img = normalize_img(dcm_img, mean, std, use_min_max)



            # Rescale to the defined image size

            if dcm_img.shape != (x, y):

                dcm_img = cv2.resize(dcm_img, (x, y), interpolation=cv2.INTER_NEAREST)



            return dcm_img, area
n_samples = 10

filenames = data_df.sample(n_samples)['filename']

png_paths = [TRAIN_DIR + filename for filename in filenames]

for path in png_paths:

    path = path.replace('.png', '.dcm')

    image, _ = get_dcm_img(path)

    print(path)

    _,_ = crop_brain(image, plot = True)
filename = 'ID_9d9cc6b01.dcm'

sample_path = TRAIN_DIR + filename



print('Before preprocessing')

img, _ = get_dcm_img(sample_path)

px = img.flatten()

plt.hist(px, bins=40)

plt.title('Histogram pixel values')

plt.show()

plt.imshow(img)

plt.show()



print('After preprocessing')

img, area = preprocess_dicom(sample_path, img_size[0], img_size[1], use_min_max = True)

px = img.flatten()

plt.hist(px, bins=40)

plt.title('Histogram pixel values')

plt.show()

plt.imshow(img)

plt.show()



# Labels of example

display(data_df[data_df['filename'] == filename[:-3] + 'png'])



# Metadata from example

get_file_metadata(TRAIN_DIR, filename)
# Samples from each group are extracted by trying to find an specific number of samples per group

# This is done due to memory limitations of have all training metadata in a DataFrame

def sample_groups(load_dir, samples_per_group = 5, max_trys = 1000):

    filenames = os.listdir(load_dir)

    filenames_groups = {1 : [], 2 : [], 3 : []}

    for group in tqdm_notebook([1,2,3], desc = 'Group sample'):

        count_samples = 0

        for _ in tqdm_notebook(range(max_trys), desc = 'Try'):

            filename = np.random.choice(filenames, size = 1, replace = False)[0]

            sample_group = get_subgroups(pydicom.dcmread(load_dir + filename))

            if sample_group == group:

                filenames_groups[group].append(load_dir + filename)

                count_samples += 1

                if count_samples >= samples_per_group:

                    break

    return filenames_groups



# For each group it is computed:

# Firstly mean of equally distributed bin

# Secondly mean of mean pixels values and mean of std pixel values using the previous bin mean

def sample_bins_mean_std(load_dir, samples_per_group = 5, max_trys = 1000, n_bins = 100):

    bins_mean = {}

    mean = {}

    std = {}

    groups_paths = sample_groups(TRAIN_DIR, samples_per_group = samples_per_group, max_trys = max_trys)

    for group in [1,2,3]:    

        # Do not proceed if there is no images

        if len(groups_paths[group]) == 0:

            bins_mean[group] = []

            mean[group] = np.nan

            std[group] = np.nan

            continue

                    

        filenames = groups_paths[group]

    

        dcm_img_array = []

        for filename in tqdm_notebook(filenames, desc = 'Calc bins'):

            dcm_img, _ = get_dcm_img(filename)

            dcm_img_array.append(dcm_img)

            #bins_array.append(get_freqhist_bins(dcm_img, n_bins))



        #bins_mean[group] = np.array(bins_array).mean(axis = 0)

        bins_mean[group] = get_freqhist_bins(np.array(dcm_img_array).reshape(-1), n_bins)

    

        dcm_img_array = []

        for filename in tqdm_notebook(filenames, desc = 'Calc mean & std'):

            dcm_img, _ = get_dcm_img(filename)

            dcm_img_array.append(interpolate_img(dcm_img, bins_mean[group], n_bins))

    

        mean[group] = np.array(dcm_img_array).flatten().mean()

        std[group] = np.array(dcm_img_array).flatten().std()

    

    return bins_mean, mean, std
bins_mean, mean, std = sample_bins_mean_std(TRAIN_DIR, samples_per_group = 5, max_trys = 5000, n_bins = 100)
train_df, _ = train_test_split(data_df, train_size = FRACTION_TRAINING, stratify = data_df[CLASSES], random_state = SEED)

train_df, test_df = train_test_split(train_df, test_size = 0.1, stratify = train_df[CLASSES], random_state = SEED)
def save_and_resize(load_dir, filenames, img_size, bins = None, mean = None, std = None, use_min_max = False, remove_small = False, remove_empty = False, save_dir = '', save_on_zip = True, zip_name = 'output'):

    if not os.path.exists(save_dir):

        os.makedirs(save_dir)    

    

    png_paths = [load_dir + filename for filename in filenames]

    #for png_path in tqdm_notebook(png_paths):

    #    print(png_path)

    #    process_save_and_resize(png_path, img_size, bins, mean, std, use_min_max, remove_small, remove_empty, save_dir)

    Parallel(n_jobs=NUM_CORES)(delayed(process_save_and_resize)(png_path, img_size, bins, mean, std, use_min_max, remove_small, remove_empty, save_dir) for png_path in tqdm_notebook(png_paths))

    

    # Save images in ZIP file

    if save_on_zip:

        shutil.make_archive(zip_name, 'zip', save_dir)

    

def process_save_and_resize(png_path, img_size, bins, mean, std, use_min_max, remove_small, remove_empty, save_dir):

    path = png_path.replace('.png', '.dcm')

    filename = png_path[len(png_path) - png_path[::-1].find("/") : ]

    new_path = save_dir + filename

    img, area = preprocess_dicom(path, img_size[0], img_size[1], bins = bins, mean = mean, std = std, use_min_max = use_min_max)

    # Do not save image in case that:

    # 1 - it is empty and want to remove empty

    # 2 - It has an small brain area and want to remove these images

    if (remove_empty and img is None) or (remove_small and USE_MIN_AREA and area < MIN_AREA):

        pass # Do not save image

    else:

        #image.imsave(new_path, img)

        imwrite(new_path, img)
#save_and_resize(TRAIN_DIR, train_df['filename'], img_size, bins = bins_mean, mean = mean, std = std, save_dir = TRAIN_PNG)

save_and_resize(TRAIN_DIR, train_df['filename'], img_size, bins = bins_mean, use_min_max = True, remove_small = True, remove_empty = True, save_dir = TRAIN_PNG, save_on_zip = True, zip_name = 'train')



#save_and_resize(TRAIN_DIR, test_df['filename'], img_size, bins = bins_mean, mean = mean, std = std, save_dir = TEST_PNG)

save_and_resize(TRAIN_DIR, test_df['filename'], img_size, bins = bins_mean, use_min_max = True, remove_small = True, remove_empty = True, save_dir = TEST_PNG, save_on_zip = True, zip_name = 'test')



#save_and_resize(SUB_DIR, sub_df['filename'], img_size, bins = bins_mean, mean = mean, std = std, save_dir = SUB_PNG)

save_and_resize(SUB_DIR, sub_df['filename'], img_size, bins = bins_mean, use_min_max = True, remove_small = False, remove_empty = False, save_dir = SUB_PNG, save_on_zip = True, zip_name = 'sub')
def check_files(df, png_dir):

    df['check_in_dir'] = df['filename'].apply(lambda filename : os.path.exists(png_dir + filename))

    return df[df['check_in_dir'] == True].drop('check_in_dir', axis = 1)
if not os.path.exists(CSV_DIR):

    os.makedirs(CSV_DIR)



train_df = check_files(train_df, TRAIN_PNG)

test_df = check_files(test_df, TEST_PNG)

sub_df = check_files(sub_df, SUB_PNG)

    

train_df.to_csv(CSV_DIR + 'train_df.csv', index = None, header=True)

test_df.to_csv(CSV_DIR + 'test_df.csv', index = None, header=True)

sub_df.to_csv(CSV_DIR + 'sub_df.csv', index = None, header=True)



train_meta_df = get_all_metadata(TRAIN_PNG, dcm_dir = TRAIN_DIR)

train_meta_df.to_csv(CSV_DIR + 'train_meta_df.csv', header=True)



test_meta_df = get_all_metadata(TEST_PNG, dcm_dir = TRAIN_DIR)

test_meta_df.to_csv(CSV_DIR + 'test_meta_df.csv', header=True)



sub_meta_df = get_all_metadata(SUB_PNG, dcm_dir = SUB_DIR)

sub_meta_df.to_csv(CSV_DIR + 'sub_meta_df.csv', header=True)
shutil.rmtree(TRAIN_PNG)

shutil.rmtree(TEST_PNG)

shutil.rmtree(SUB_PNG)