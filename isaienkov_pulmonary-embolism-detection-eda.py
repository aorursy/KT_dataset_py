!conda install -c conda-forge gdcm -y



import os

import numpy as np

import pandas as pd

import pydicom as dcm

import matplotlib

import plotly.express as px

import matplotlib.pyplot as plt

import seaborn as sns

import glob

import gdcm

from matplotlib import animation, rc

from plotly.subplots import make_subplots

import plotly.graph_objs as go



TRAIN_DIR = "../input/rsna-str-pulmonary-embolism-detection/train/"

files = glob.glob('../input/rsna-str-pulmonary-embolism-detection/train/*/*/*.dcm')



rc('animation', html='jshtml')



np.random.seed(666)
train = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")

test = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/test.csv")
train
cols = [

    'pe_present_on_image', 'negative_exam_for_pe', 'qa_motion', 

    'qa_contrast', 'flow_artifact', 'rv_lv_ratio_gte_1', 

    'rv_lv_ratio_lt_1', 'leftsided_pe', 'chronic_pe', 

    'true_filling_defect_not_pe', 'rightsided_pe', 

    'acute_and_chronic_pe', 'central_pe', 'indeterminate'

]



fig = make_subplots(rows=5, cols=3)



traces = [

    go.Bar(

        x=[0, 1], 

        y=[

            len(train[train[col]==0]),

            len(train[train[col]==1])

        ], 

        name=col,

        text = [

            str(round(100 * len(train[train[col]==0]) / len(train), 2)) + '%',

            str(round(100 * len(train[train[col]==1]) / len(train), 2)) + '%'

        ],

        textposition='auto'

    ) for col in cols

]



for i in range(len(traces)):

    fig.append_trace(traces[i], (i // 3) + 1, (i % 3)  +1)



fig.update_layout(

    title_text='Train columns',

    height=1200,

    width=1000

)



fig.show()
x = train.drop(['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], axis=1).sum(axis=0).sort_values().reset_index()

x.columns = ['column', 'nonzero_records']



fig = px.bar(

    x, 

    x='nonzero_records', 

    y='column', 

    orientation='h', 

    title='Columns and non zero samples', 

    height=800, 

    width=800

)



fig.show()
data = train.drop(['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], axis=1).astype(bool).sum(axis=1).reset_index()

data.columns = ['row', 'count']

data = data.groupby(['count'])['row'].count().reset_index()



fig = px.bar(

    data, 

    y=data['row'], 

    x="count", 

    title='Number of activations in for every sample in training set', 

    width=800, 

    height=500

)



fig.show()
data = train.drop(['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID'], axis=1).astype(bool).sum(axis=1).reset_index()

data.columns = ['row', 'count']

data = data.groupby(['count'])['row'].count().reset_index()



fig = px.pie(

    data, 

    values=round((100 * data['row'] / len(train)), 2), 

    names="count", 

    title='Number of activations for every sample (Percent)', 

    width=800, 

    height=500

)



fig.show()
data = train[[

    'pe_present_on_image', 'negative_exam_for_pe', 'qa_motion', 

    'qa_contrast', 'flow_artifact', 'rv_lv_ratio_gte_1', 

    'rv_lv_ratio_lt_1', 'leftsided_pe', 'chronic_pe', 

    'true_filling_defect_not_pe', 'rightsided_pe', 

    'acute_and_chronic_pe', 'central_pe', 'indeterminate'

]]



f = plt.figure(figsize=(16, 16))

plt.matshow(data.corr(), fignum=f.number)

plt.xticks(range(data.shape[1]), data.columns, fontsize=13, rotation=70)

plt.yticks(range(data.shape[1]), data.columns, fontsize=13)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=13)
print('Total number (dirictories) in training set {}'.format(len(os.listdir(TRAIN_DIR))))
test.head()
fig, ax = plt.subplots(figsize=(8, 8))

ax.imshow(

    dcm.dcmread("../input/rsna-str-pulmonary-embolism-detection/train/4833c9b6a5d0/57e3e3c5f910/f4fdc88f2ace.dcm").pixel_array

)
test_image = dcm.dcmread("../input/rsna-str-pulmonary-embolism-detection/train/4833c9b6a5d0/57e3e3c5f910/f4fdc88f2ace.dcm").pixel_array

print('Image shape: ', test_image.shape)
dcm.dcmread("../input/rsna-str-pulmonary-embolism-detection/train/4833c9b6a5d0/57e3e3c5f910/f4fdc88f2ace.dcm")
f, plots = plt.subplots(6, 6, sharex='col', sharey='row', figsize=(17, 17))



for i in range(36):

    plots[i // 6, i % 6].axis('off')

    plots[i // 6, i % 6].imshow(dcm.dcmread(np.random.choice(files[:10000])).pixel_array)
def load_slice(path):

    slices = [dcm.read_file(path + '/' + s) for s in os.listdir(path)]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

    

    try:

        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])

    except:

        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

        

    for s in slices:

        s.SliceThickness = slice_thickness

        

    return slices



### Source: https://www.kaggle.com/allunia/pulmonary-fibrosis-dicom-preprocessing

def transform_to_hu(slices):

    images = np.stack([file.pixel_array for file in slices])

    images = images.astype(np.int16)

    images[images <= -1000] = 0



    for n in range(len(slices)):

        intercept = slices[n].RescaleIntercept

        slope = slices[n].RescaleSlope

        

        if slope != 1:

            images[n] = slope * images[n].astype(np.float64)

            images[n] = images[n].astype(np.int16)

            

        images[n] += np.int16(intercept)

    

    return np.array(images, dtype=np.int16)
first_patient = load_slice('../input/rsna-str-pulmonary-embolism-detection/train/eac9014cea52/90cc14605905')

first_patient_pixels = transform_to_hu(first_patient)



fig, plots = plt.subplots(16, 10, sharex='col', sharey='row', figsize=(20, 25))



for i in range(160):

    plots[i // 10, i % 10].axis('off')

    plots[i // 10, i % 10].imshow(first_patient_pixels[i], cmap=plt.cm.viridis) 
scans = glob.glob('/kaggle/input/rsna-str-pulmonary-embolism-detection/train/*/*/')

print('Total number of scans: ', len(scans))
def read_scan(path):

    fragments = glob.glob(path + '/*')

    

    slices = []

    for f in fragments:

        img = dcm.dcmread(f)

        img_data = img.pixel_array

        length = int(img.InstanceNumber)

        slices.append((length, img_data))

    slices.sort()

    return [s[1] for s in slices]





def animate(ims):

    fig = plt.figure(figsize=(11, 11))

    plt.axis('off')

    im = plt.imshow(ims[0])



    def animate_func(i):

        im.set_array(ims[i])

        return [im]



    return animation.FuncAnimation(fig, animate_func, frames = len(ims), interval = 1000//24)
movie = animate(read_scan(scans[666]))
movie
plt.figure(figsize=(15,8))



for n in range(100):

    loaded = dcm.dcmread(np.random.choice(files[:]))

    image = loaded.pixel_array.flatten()

    rescaled_image = image * loaded.RescaleSlope + loaded.RescaleIntercept

    sns.distplot(image.flatten())



plt.title("HU unit distributions for 100 examples")
scans = [dcm.dcmread(files[i]) for i in range(500)]
hu_scans = transform_to_hu(scans)
img = hu_scans[13]

a = img.reshape((512, 512, 1))

a = np.concatenate([a, a, a], axis=2)



fig = make_subplots(1, 2)

img = hu_scans[0]



fig.add_trace(go.Image(z=a), 1, 1)

fig.add_trace(go.Histogram(x=img.ravel(), opacity=1), 1, 2)



fig.update_layout(

    height=600, 

    width=800,

    title='Image in HU and HU values distribution'

)



fig.show()
img = hu_scans[55]

a = img.reshape((512, 512, 1))

a = np.concatenate([a, a, a], axis=2)



fig = make_subplots(1, 2)

img = hu_scans[0]



fig.add_trace(go.Image(z=a), 1, 1)

fig.add_trace(go.Histogram(x=img.ravel(), opacity=1), 1, 2)

fig.update_layout(

    height=600, 

    width=800,

    title='Image in HU and HU values distribution'

)



fig.show()
img = hu_scans[90]

a = img.reshape((512, 512, 1))

a = np.concatenate([a, a, a], axis=2)



fig = make_subplots(1, 2)

img = hu_scans[0]



fig.add_trace(go.Image(z=a), 1, 1)

fig.add_trace(go.Histogram(x=img.ravel(), opacity=1), 1, 2)



fig.update_layout(

    height=600, 

    width=800,

    title='Image in HU and HU values distribution'

)



fig.show()
im_path = list()



for i in os.listdir(TRAIN_DIR): 

    for j in os.listdir(TRAIN_DIR + i):

        x = i+'/'+j

        im_path.append(x)
pixelspacing_r = []

pixelspacing_c = []

rows = []

columns = []

ids = []

slice_thicknesses = []

kvp = []

modality = []

table_height = []

x_ray = []

exposure = []

patient_position = []

detector_tilt = []

bits_allocated = []

rescale_intercept = []

rescale_slope = []

photometric_interpretation = []

convolution_kernel = [] 



for i in im_path:

    ids.append(i.split('/')[0]+'_'+i.split('/')[1])

    example_dcm = os.listdir(TRAIN_DIR  + i + "/")[0]

    dataset = dcm.dcmread(TRAIN_DIR + i + "/" + example_dcm)



    spacing = dataset.PixelSpacing

    pixelspacing_r.append(spacing[0])

    pixelspacing_c.append(spacing[1])

    rows.append(dataset.Rows)

    columns.append(dataset.Columns)

    slice_thicknesses.append(dataset.SliceThickness)

    kvp.append(dataset.KVP)

    modality.append(dataset.Modality)

    table_height.append(dataset.TableHeight)

    x_ray.append(dataset.XRayTubeCurrent)

    exposure.append(dataset.Exposure)

    patient_position.append(dataset.PatientPosition)

    detector_tilt.append(dataset.GantryDetectorTilt)

    bits_allocated.append(dataset.BitsAllocated)

    rescale_intercept.append(dataset.RescaleIntercept)

    rescale_slope.append(dataset.RescaleSlope)

    photometric_interpretation.append(dataset.PhotometricInterpretation)

    convolution_kernel.append(dataset.ConvolutionKernel)

    

scan_properties = pd.DataFrame(data=ids, columns=["ID"])

scan_properties.loc[:, "pixelspacing_r"] = pixelspacing_r

scan_properties.loc[:, "pixelspacing_c"] = pixelspacing_c

scan_properties.loc[:, "rows"] = rows

scan_properties.loc[:, "columns"] = columns

scan_properties.loc[:, "slice_thicknesses"] = slice_thicknesses

scan_properties.loc[:, "kvp"] = kvp

scan_properties.loc[:, "modality"] = modality

scan_properties.loc[:, "table_height"] = table_height

scan_properties.loc[:, "x_ray_tube_current"] = x_ray

scan_properties.loc[:, "exposure"] = exposure

scan_properties.loc[:, "patient_position"] = patient_position

scan_properties.loc[:, "gantry/detector_tilt"] = detector_tilt

scan_properties.loc[:, "bits_allocated"] = bits_allocated

scan_properties.loc[:, "rescale_intercept"] = rescale_intercept

scan_properties.loc[:, "rescale_slope"] = rescale_slope

scan_properties.loc[:, "photometric_interpretation"] = photometric_interpretation

scan_properties.loc[:, "convolution_kernel"] = convolution_kernel



scan_properties
print('Unique rows number: ', scan_properties['rows'].unique().tolist())

print('Unique columns number: ', scan_properties['columns'].unique().tolist())
print('Number of inconsistencies in pixel spacing: ', len(scan_properties[scan_properties['pixelspacing_r'] != scan_properties['pixelspacing_c']]))
fig = px.histogram(

    scan_properties, 

    "pixelspacing_r", 

    nbins=100, 

    title='Pixel spacing distribution', 

    width=700,

    height=500

)



fig.show()
data = scan_properties['slice_thicknesses'].value_counts().reset_index()

data.columns = ['slice_thicknesses', 'count']

data['slice_thicknesses'] = 'st: ' + data['slice_thicknesses'].astype(str)



fig = px.bar(

    data, 

    x="slice_thicknesses", 

    y="count", 

    title='slice_thicknesses distribution', 

    width=700,

    height=500,

)



fig.show()
data = scan_properties['kvp'].value_counts().reset_index()

data.columns = ['kvp', 'count']

data['kvp'] = 'kvp: ' + data['kvp'].astype(str)

fig = px.bar(

    data, 

    x="kvp", 

    y="count", 

    title='Peak kilovoltage distribution', 

    width=700,

    height=500

)

fig.show()
fig = px.histogram(

    scan_properties, 

    "table_height", 

    nbins=100, 

    title='Table_height distribution', 

    width=700,

    height=500

)



fig.show()
fig = px.histogram(

    scan_properties, 

    "x_ray_tube_current", 

    nbins=100, 

    title='x_ray_tube_current distribution', 

    width=700,

    height=500

)



fig.show()
fig = px.histogram(

    scan_properties, 

    "exposure", 

    nbins=100, 

    title='exposure distribution', 

    width=700,

    height=500

)

fig.show()
data = scan_properties['patient_position'].value_counts().reset_index()

data.columns = ['patient_position', 'count']



fig = px.bar(

    data, 

    x="patient_position", 

    y="count", 

    title='patient_position distribution', 

    width=700,

    height=500

)



fig.show()
data = scan_properties["rescale_intercept"].value_counts().reset_index()

data.columns = ["rescale_intercept", 'count']

fig = px.bar(

    data, 

    x="rescale_intercept", 

    y="count", 

    title='"rescale_intercept" distribution', 

    width=700,

    height=500

)

fig.show()