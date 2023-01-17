import pydicom

import os

import numpy

import pandas as pd

from matplotlib import pyplot, cm

from fastai.vision import *

import fastai
DATA_DIR = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection'

TRAIN_IMAGES_DIR = DATA_DIR + '/stage_2_train/'

TRAIN_CSV_DIR = DATA_DIR + '/stage_2_train.csv'

TEST_IMAGES_DIR = DATA_DIR + '/stage_2_test/'

TEST_CSV_DIR = DATA_DIR + '/stage_2_sample_submission.csv'
initial_df = pd.read_csv(TRAIN_CSV_DIR)

initial_table = initial_df.copy()

print(initial_table.shape)

initial_table.head(n=5)
#gets rid of the ID at the beginning, now we have the name of the sample and the ICH type together as label

new1 = initial_table["ID"].str.split("_", n = 1, expand = True)

#further splits between name of sample and ICH type

new2 = new1[1].str.split("_", n = 1, expand = True)



#add values in the new2 table to the initial_table we were working with

initial_table['Image_ID'] = new2[0]

initial_table['Sub_type'] = new2[1]

initial_table.head(n=5)
#now extract the image IDs, we have the column set up so we can extract these

image_ids = initial_table.Image_ID.unique()

#make an empty list for each unique image id

typelabel = ["" for _ in range(len(image_ids))]

#manipulate to a dataframe with each row being a array with the id and the list of labels, not filled yet

cleaned_df = pd.DataFrame(np.array([image_ids, typelabel]).transpose(), columns=["case", "label"])

cleaned_df.head(n=5)
#make an object type dictionary for each image id

lbls = {i : "" for i in image_ids}

#reduces initial_table to only be values where label is 1, ie there is an ICH

initial_table = initial_table[initial_table.Label == 1]

#reduces initial_table, gets rid of cases where any = 1, which is redundant, but we could also edit this

initial_table = initial_table[initial_table.Sub_type != "any"]



#fill the lbls dictionary object with the ICH if there is one, a blank if there isnt

i = 0

for name, group in initial_table.groupby("Image_ID"):

    lbls[name] = " ".join(group.Sub_type)

    i += 1
#now fill the labels column in the cleaned df to be blank if no ICH, have the type if there is

cleaned_df = pd.DataFrame(np.array([list(lbls.keys()), list(lbls.values())]).transpose(), 

                          columns=["case", "label"])

cleaned_df.head(n=5)
#loads 5 random images of the different types and 5 images without ICHs

def load_random_images():

    image_names = [list(train[train[h_type] == 1].sample(1)['filename'])[0] for h_type in hem_types]

    image_names += list(train[train['any'] == 0].sample(5)['filename'])

    return [pydicom.read_file(os.path.join(TRAIN_IMG_PATH, img_name)) for img_name in image_names]



#function to view the images we sampled

def view_images(images):

    width = 5

    height = 2

    fig, axs = plt.subplots(height, width, figsize=(15,5))

    

    for im in range(0, height * width):

        image = images[im]

        i = im // width

        j = im % width

        axs[i,j].imshow(image, cmap=plt.cm.bone) 

        axs[i,j].axis('off')

        title = hem_types[im] if im < len(hem_types) else 'normal'

        axs[i,j].set_title(title)



    plt.show()



def get_first_of_dicom_field_as_int(x):

       #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)



#gets the window center, width, intercept from dicom data    

def get_windowing(data):

    dicom_fields = [data[('0028','1050')].value, #window center

                    data[('0028','1051')].value, #window width

                    data[('0028','1052')].value, #intercept

                    data[('0028','1053')].value] #slope

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
#what is the purpose of rescale??? Setting to false for now

def window_image_matteo(img, window_center = None, window_width = None, rescale = False, exclusive = False):

    if window_center is None and window_width is None:

        window_center, window_width , intercept, slope = get_windowing(img)

    else:

        _, _, intercept, slope = get_windowing(img)

    #img = img.pixel_array

    img = (img*slope +intercept)

    img_min = window_center - window_width//2

    img_max = window_center + window_width//2

    img[img<img_min] = img_min

    if exclusive:

        img[img>img_max] = img_min

    else:

        img[img>img_max] = img_max



    if rescale:

        # Extra rescaling to 0-1, not in the original notebook

        img = (img - img_min) / (img_max - img_min)

    

    return img



def window_image(img, window_center,window_width, intercept, slope):



    img = (img*slope +intercept)

    img_min = window_center - window_width//2

    img_max = window_center + window_width//2

    img[img<img_min] = img_min

    img[img>img_max] = img_max

    return img
#we need IDs in our dataframe to match up with the image names, so lets convert them to be such

cleaned_df.case = "ID_" + cleaned_df.case + ".dcm"

cleaned_df.head(n=10)
#this windows the image that is opened as needed to be passed into the training framework

def new_open_image(path, convert_mode=None, after_open=None):

    dcm = pydicom.dcmread(str(path))

    window_center, window_width, intercept, slope = get_windowing(dcm)

    im = window_image(dcm.pixel_array, window_center, window_width, intercept, slope)

    im = np.stack((im,)*3, axis=-1)

    im -= im.min()

    im_max = im.max()

    if im_max != 0: im = im / im.max()

    x = Image(pil2tensor(im, dtype=np.float32))

    #if div: x.div_(2048)  # ??

    return x



vision.data.open_image = new_open_image
#sampling a 50/50 split of 5,000 total images, half have labels, half do not

df_train = pd.concat([cleaned_df[cleaned_df.label == ""][:2500], cleaned_df[cleaned_df.label != ""][:2500]])



df_train.head(n=5)

df_train.shape
#this will make a list of images based on the subsetted training dataframe, type: fastai.vision.data.ImageList

im_list = ImageList.from_df(df_train, path=TRAIN_IMAGES_DIR)

#prepares a list of test names, and takes test images based on that

test_fnames = pd.DataFrame("ID_" + pd.read_csv(TEST_CSV_DIR)["ID"].str.split("_", n=2, expand = True)[1].unique() + ".dcm")

test_im_list = ImageList.from_df(test_fnames, TEST_IMAGES_DIR)



tfms = get_transforms(do_flip=False)

bs = 128

data = (im_list.split_by_rand_pct(0.2)

               .label_from_df(label_delim=" ")

               .transform(tfms, size=512)

               .add_test(test_im_list)

               .databunch(bs=bs, num_workers=0)

               .normalize())
data.show_batch(3)
learn = cnn_learner(data, models.resnet18)



models_path = Path("/kaggle/working/models")

if not models_path.exists(): models_path.mkdir()

    

learn.model_dir = models_path

learn.metrics = [accuracy_thresh]
#learn.fit_one_cycle(1)

#learn.recorder.plot()