from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
import gc
import time
from IPython.display import clear_output
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint as MC
from tensorflow.keras import backend as K

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


root = '/kaggle/input/rsna-str-pulmonary-embolism-detection'
for item in os.listdir(root):
    path = os.path.join(root, item)
    if os.path.isfile(path):
        print(path)


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print('Reading train data...')
train = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")
print(train.shape)
train.head()
print('Reading test data...')
test = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/test.csv")
print(test.shape)
test.head()
print('Reading sample data...')
ss = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/sample_submission.csv")
print(ss.shape)
ss.head()
import vtk
from vtk.util import numpy_support
import cv2

reader = vtk.vtkDICOMImageReader()
def get_img(path):
    reader.SetFileName(path)
    reader.Update()
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

    ConstPixelSpacing = reader.GetPixelSpacing()
    imageData = reader.GetOutput()
    pointData = imageData.GetPointData()
    arrayData = pointData.GetArray(0)
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
    ArrayDicom = cv2.resize(ArrayDicom,(512,512))
    return ArrayDicom
import pydicom
dpath = "../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf"
def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

id=0
patient = load_scan(dpath)
imgs = get_pixels_hu(patient)
output_path = working_path = "../working/"
np.save(output_path + "fullimages_%d.npy" % (id), imgs)
import matplotlib.pyplot as plt
file_used=output_path+"fullimages_%d.npy" % id
imgs_to_process = np.load(file_used).astype(np.float64) 

plt.hist(imgs_to_process.flatten(), bins=50, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()
id = 0
imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))

def sample_stack(stack, rows=6, cols=6, start_with=10, show_every=3):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    for i in range(rows*cols):
        ind = start_with + i*show_every
        ax[int(i/rows),int(i % rows)].set_title('slice %d' % ind)
        ax[int(i/rows),int(i % rows)].imshow(stack[ind],cmap='gray')
        ax[int(i/rows),int(i % rows)].axis('off')
    plt.show()

sample_stack(imgs_to_process)
print("Slice Thickness: %f" % patient[0].SliceThickness)
print("Pixel Spacing (row, col): (%f, %f) " % (patient[0].PixelSpacing[0], patient[0].PixelSpacing[1]))
id = 0
imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))
def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    #spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

print("Shape before resampling\t", imgs_to_process.shape)
imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])
print("Shape after resampling\t", imgs_after_resamp.shape)
def make_mesh(image, threshold=-300, step_size=1):

    print( "Transposing surface")
    p = image.transpose(2,1,0)
    
    print("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes_lewiner(p, threshold, step_size=step_size, allow_degenerate=True) 
    
    return verts, faces

def plotly_3d(verts, faces):
    x,y,z = zip(*verts) 
    
    print("Drawing") 
    
    # Make the colormap single color since the axes are positional not intensity. 
#    colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
    colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
    
    fig = FF.create_trisurf(x=x,
                        y=y, 
                        z=z, 
                        plot_edges=False,
                        colormap=colormap,
                        simplices=faces,
                        backgroundcolor='rgb(64, 64, 64)',
                        title="Interactive Visualization")
    iplot(fig)

def plt_3d(verts, faces):
    print("Drawing")
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_facecolor((0.7, 0.6, 0.4))
    plt.show()
v, f = make_mesh(imgs_after_resamp, 350)
plt_3d(v, f)
v, f = make_mesh(imgs_after_resamp, 350, 2)
plotly_3d(v, f)
def make_lungmask(img, display=False):
    row_size= img.shape[0]
    col_size = img.shape[1]
    
    mean = np.mean(img)
    std = np.std(img)
    img = img-mean
    img = img/std
    # Find the average pixel value near the lungs
    # to renormalize washed out images
    middle = img[int(col_size/5):int(col_size/5*4),int(row_size/5):int(row_size/5*4)] 
    mean = np.mean(middle)  
    max = np.max(img)
    min = np.min(img)
    # To improve threshold finding, I'm moving the 
    # underflow and overflow on the pixel spectrum
    img[img==max]=mean
    img[img==min]=mean
    #
    # Using Kmeans to separate foreground (soft tissue / bone) and background (lung/air)
    #
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle,[np.prod(middle.shape),1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img<threshold,1.5,0.0)  # threshold the image

    # First erode away the finer elements, then dilate to include some of the pixels surrounding the lung.  
    # We don't want to accidentally clip the lung.

    eroded = morphology.erosion(thresh_img,np.ones([3,3]))
    dilation = morphology.dilation(eroded,np.ones([8,8]))

    labels = measure.label(dilation) # Different labels are displayed in different colors
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2]-B[0]<row_size/10*9 and B[3]-B[1]<col_size/10*9 and B[0]>row_size/5 and B[2]<col_size/5*4:
            good_labels.append(prop.label)
    mask = np.ndarray([row_size,col_size],dtype=np.int8)
    mask[:] = 0

    #
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask 
    #
    for N in good_labels:
        mask = mask + np.where(labels==N,1,0)
    mask = morphology.dilation(mask,np.ones([10,10])) # one last dilation

    if (display):
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask*img, cmap='gray')
        ax[2, 1].axis('off')
        
        plt.show()
    return mask*img
img = imgs_after_resamp[260]
make_lungmask(img, display=True)
masked_lung = []

for img in imgs_after_resamp:
    masked_lung.append(make_lungmask(img))

sample_stack(masked_lung, show_every=5)
np.save(output_path + "maskedimages_%d.npy" % (id), imgs)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D

inputs = Input((512, 512, 3))
#x = Conv2D(3, (1, 1), activation='relu')(inputs)
base_model = keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

outputs = base_model(inputs, training=False)
outputs = keras.layers.GlobalAveragePooling2D()(outputs)
outputs = Dropout(0.25)(outputs)
outputs = Dense(1024, activation='relu')(outputs)
outputs = Dense(512, activation='relu')(outputs)
outputs = Dense(256, activation='relu')(outputs)
outputs = Dense(128, activation='relu')(outputs)
outputs = Dense(64, activation='relu')(outputs)
ppoi = Dense(1, activation='sigmoid', name='pe_present_on_image')(outputs)
rlrg1 = Dense(1, activation='sigmoid', name='rv_lv_ratio_gte_1')(outputs)
rlrl1 = Dense(1, activation='sigmoid', name='rv_lv_ratio_lt_1')(outputs) 
lspe = Dense(1, activation='sigmoid', name='leftsided_pe')(outputs)
cpe = Dense(1, activation='sigmoid', name='chronic_pe')(outputs)
rspe = Dense(1, activation='sigmoid', name='rightsided_pe')(outputs)
aacpe = Dense(1, activation='sigmoid', name='acute_and_chronic_pe')(outputs)
cnpe = Dense(1, activation='sigmoid', name='central_pe')(outputs)
indt = Dense(1, activation='sigmoid', name='indeterminate')(outputs)

model = Model(inputs=inputs, outputs={'pe_present_on_image':ppoi,
                                      'rv_lv_ratio_gte_1':rlrg1,
                                      'rv_lv_ratio_lt_1':rlrl1,
                                      'leftsided_pe':lspe,
                                      'chronic_pe':cpe,
                                      'rightsided_pe':rspe,
                                      'acute_and_chronic_pe':aacpe,
                                      'central_pe':cnpe,
                                      'indeterminate':indt})

opt = keras.optimizers.Adam(lr=0.0001)

model.compile(optimizer=opt,
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()
model.save('pe_detection_model.h5')
del model
K.clear_session()
gc.collect()
def convert_to_rgb(array):
    array = array.reshape((512, 512, 1))
    return np.stack([array, array, array], axis=2).reshape((512, 512, 3))
    
def custom_dcom_image_generator(batch_size, dataset, test=False, debug=False):
    
    fnames = dataset[['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']]
    
    if not test:
        Y = dataset[['pe_present_on_image', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1', 'leftsided_pe',
                     'chronic_pe', 'rightsided_pe', 'acute_and_chronic_pe', 'central_pe', 'indeterminate'
                    ]]
        prefix = 'input/rsna-str-pulmonary-embolism-detection/train'
        
    else:
        prefix = 'input/rsna-str-pulmonary-embolism-detection/test'
    
    X = []
    batch = 0
    for st, sr, so in fnames.values:
        if debug:
            print(f"Current file: ../{prefix}/{st}/{sr}/{so}.dcm")

        dicom = get_img(f"../{prefix}/{st}/{sr}/{so}.dcm")
        image = convert_to_rgb(dicom)
        X.append(image)
        
        del st, sr, so
        
        if len(X) == batch_size:
            if test:
                yield np.array(X)
                del X
            else:
                yield np.array(X), Y[batch*batch_size:(batch+1)*batch_size].values
                del X
                
            gc.collect()
            X = []
            batch += 1
        
    if test:
        yield np.array(X)
    else:
        yield np.array(X), Y[batch*batch_size:(batch+1)*batch_size].values
        del Y
    del X
    gc.collect()
    return
history = {}
start = time.time()
debug = 0
batch_size = 1000
train_size = int(batch_size*0.9)

max_train_time = 3600*3  #hours to seconds of training

checkpoint = MC(filepath='../working/pe_detection_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
#Train loop
for n, (x, y) in enumerate(custom_dcom_image_generator(batch_size, train.sample(frac=1), False, debug)):
    
    if len(x) < 10: #Tries to filter out empty or short data
        break
        
    clear_output(wait=True)
    print("Training batch: %i - %i" %(batch_size*n, batch_size*(n+1)))
    
    model = load_model('../working/pe_detection_model.h5')
    hist = model.fit(
        x[:train_size], #Y values are in a dict as there's more than one target for training output
        {'pe_present_on_image':y[:train_size, 0],
         'rv_lv_ratio_gte_1':y[:train_size, 1],
         'rv_lv_ratio_lt_1':y[:train_size, 2],
         'leftsided_pe':y[:train_size, 3],
         'chronic_pe':y[:train_size, 4],
         'rightsided_pe':y[:train_size, 5],
         'acute_and_chronic_pe':y[:train_size, 6],
         'central_pe':y[:train_size, 7],
         'indeterminate':y[:train_size, 8]},

        callbacks = checkpoint,

        validation_split=0.2,
        epochs=3,
        batch_size=8,
        verbose=debug
    )
    
    print("Metrics for batch validation:")
    model.evaluate(x[train_size:],
                   {'pe_present_on_image':y[train_size:, 0],
                    'rv_lv_ratio_gte_1':y[train_size:, 1],
                    'rv_lv_ratio_lt_1':y[train_size:, 2],
                    'leftsided_pe':y[train_size:, 3],
                    'chronic_pe':y[train_size:, 4],
                    'rightsided_pe':y[train_size:, 5],
                    'acute_and_chronic_pe':y[train_size:, 6],
                    'central_pe':y[train_size:, 7],
                    'indeterminate':y[train_size:, 8]
                   }
                  )
    
    try:
        for key in hist.history.keys():
            history[key] = np.concatenate([history[key], hist.history[key]], axis=0)
    except:
        for key in hist.history.keys():
            history[key] = hist.history[key]
            
    #To make sure that our model don't train overtime
    if time.time() - start >= max_train_time:
        print("Time's up!")
        break
        
    model.save('pe_detection_model.h5')
    del model, x, y, hist
    K.clear_session()
    gc.collect()
for key in history.keys():
    if key.startswith('val'):
        continue
    else:
        epoch = range(len(history[key]))
        plt.plot(epoch, history[key]) #X=epoch, Y=value
        plt.plot(epoch, history['val_'+key])
        plt.title(key)
        if 'accuracy' in key:
            plt.axis([0, len(history[key]), -0.1, 1.1]) #Xmin, Xmax, Ymin, Ymax
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
predictions = {}
stopper = 3600*4  #4 hours limit for prediction
pred_start_time = time.time()

p, c = time.time(), time.time()
batch_size = 500
    
l = 0
n = test.shape[0]

for x in custom_dcom_image_generator(batch_size, test, True, False):
    clear_output(wait=True)
    model = load_model("../working/pe_detection_model.h5")
    preds = model.predict(x, batch_size=8, verbose=1)
    
    try:
        for key in preds.keys():
            predictions[key] += preds[key].flatten().tolist()
            
    except Exception as e:
        print(e)
        for key in preds.keys():
            predictions[key] = preds[key].flatten().tolist()
            
    l = (l+batch_size)%n
    print('Total predicted:', len(predictions['indeterminate']),'/', n)
    p, c = c, time.time()
    print("One batch time: %.2f seconds" %(c-p))
    print("ETA: %.2f" %((n-l)*(c-p)/batch_size))
    
    if c - pred_start_time >= stopper:
        print("Time's up!")
        break
    
    del model
    K.clear_session()
    
    del x, preds
    gc.collect()
test_ids = []
for v in test.StudyInstanceUID:
    if v not in test_ids:
        test_ids.append(v)
        
test_preds = test.copy()
test_preds = pd.concat([test_preds, pd.DataFrame(predictions)], axis=1)
test_preds.to_csv('test_predictions.csv', index=False)
test_preds
from scipy.special import softmax

label_agg = {key:[] for key in 
             ['id', 'negative_exam_for_pe', 'rv_lv_ratio_gte_1',
              'rv_lv_ratio_lt_1', 'leftsided_pe', 'chronic_pe',
              'rightsided_pe', 'acute_and_chronic_pe',
              'central_pe', 'indeterminate']
            }

for uid in test_ids:
    temp = test_preds.loc[test_preds.StudyInstanceUID ==uid]
    label_agg['id'].append(uid)
    
    n = temp.shape[0]
    #Check for any image level presence of PE of high confidence
    positive = any(temp.pe_present_on_image >= 0.5) #50% threshhold
    
    #Only one from positive, negative and indeterminate should have value>0.5
    #per exam
    if positive: 
        label_agg['indeterminate'].append(temp.indeterminate.min()/2)
        label_agg['negative_exam_for_pe'].append(0)
    else:
        if any(temp.indeterminate >= 0.5):
            label_agg['indeterminate'].append(temp.indeterminate.max())
            label_agg['negative_exam_for_pe'].append(1)
        else:
            label_agg['indeterminate'].append(temp.indeterminate.min()/2)
            label_agg['negative_exam_for_pe'].append(1)
    
    #I decided that the total ratio should be equal to 1, so I used softmax
    #We modify the weights by multiplying the bigger by 2 and dividing the smaller by 2
    a, b = temp[['rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1']].mean().values
    if a > b:
        a, b = a*2, b/2
    elif a < b:
        a, b = a/2, b*2
    a, b = softmax([a, b])
    if positive:
        label_agg['rv_lv_ratio_gte_1'].append(a)
        label_agg['rv_lv_ratio_lt_1'].append(b)
    else:
        label_agg['rv_lv_ratio_gte_1'].append(a/2)
        label_agg['rv_lv_ratio_lt_1'].append(b/2)
    
    #Next is for Chronic (C), Acute-Chronic (AC) and Acute (A) PE
    #We need to see if we got a high confidence value from either C or AC
    #If there is, we add it to a 50% based score for high confidence
    #and half weight for low confidence score
    if any(temp['acute_and_chronic_pe'] > 0.5): #50% confidence level
        label_agg['acute_and_chronic_pe'].append(0.5 + temp['acute_and_chronic_pe'].mean()/2)
        label_agg['chronic_pe'].append(temp['chronic_pe'].mean()/2)
        
    elif any(temp['chronic_pe'] > 0.5):
        label_agg['acute_and_chronic_pe'].append(temp['acute_and_chronic_pe'].mean()/2)
        label_agg['chronic_pe'].append(0.5 + temp['chronic_pe'].mean()/2)
        
    else: #Else, we set both to half values, as we declare the A as the value
        label_agg['acute_and_chronic_pe'].append(temp['acute_and_chronic_pe'].mean()/2)
        label_agg['chronic_pe'].append(temp['chronic_pe'].mean()/2)
    
    #for right, left, central, we use the same metric above
    for key in ['leftsided_pe', 'rightsided_pe', 'central_pe']:
        if positive:
            label_agg[key].append(0.5 + temp[key].mean()/2)
        else:
            label_agg[key].append(temp[key].mean()/2)
uid = []
labels = []
df = pd.DataFrame(label_agg)
for key in ['negative_exam_for_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1', 'leftsided_pe', 'chronic_pe',
            'rightsided_pe', 'acute_and_chronic_pe', 'central_pe', 'indeterminate']:
    for i in df.id:
        uid.append('_'.join([i, key]))
        labels.append(df.loc[df.id==i][key].values[0])
del df
gc.collect()

uid += test_preds.SOPInstanceUID.tolist()
labels += test_preds['pe_present_on_image'].tolist()

sub = pd.DataFrame({"id":uid, 'label':labels})
sub
sub.fillna(0.2, inplace=True)
sub.to_csv('submission.csv', index=False)