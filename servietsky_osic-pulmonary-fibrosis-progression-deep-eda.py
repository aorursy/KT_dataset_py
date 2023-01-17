import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pydicom

from pydicom import dcmread

import glob, os

from collections import defaultdict

import tqdm

import gc

import seaborn as sns

import ast

import plotly.express as px

from pandas_profiling import ProfileReport 

pd.options.display.max_columns = None

import cv2



from collections import defaultdict

import collections

import imageio

from IPython.display import HTML



import plotly.offline as pyo

from scipy import ndimage, misc

import warnings

warnings.filterwarnings('ignore')



pyo.init_notebook_mode()
pydicom.read_file("../input/osic-pulmonary-fibrosis-progression/train/ID00213637202257692916109/29.dcm")
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

Data = pd.read_pickle('../input/osic-transform-dicom-into-dataframe/output_data.pkl')
print('Head :')



display(Data.head())



print('Info :')



Data.info()
type_dict_all = ['ORIGINAL', 'PRIMARY', 'AXIAL', 'CT_SOM5 SPI', 'HELIX', 'CT_SOM5 SEQ', 'SECONDARY', 'DERIVED', 'JP2K LOSSY 6:1', 'VOLUME', 'OTHER', 'CSA MPR', 'CSAPARALLEL', 

                'CSA RESAMPLED', 'REFORMATTED', 'AVERAGE', 'CT_SOM7 SPI DUAL', 'STD', 'SNRG', 'DET_AB']

sns.set(rc={'figure.figsize':(15,7.5)})

plt.xticks(rotation=45)

ax = sns.barplot(y=0, x = Data[type_dict_all].sum().to_frame().sort_values(0,  ascending=False).index,  data=Data[type_dict_all].sum().to_frame().sort_values(0,  ascending=False))
tmp = Data.groupby('Manufacturer')[type_dict_all].sum()

tmp = pd.melt(tmp.reset_index(), id_vars=['Manufacturer'])

tmp.columns = ['Manufacturer','ImageType', 'Value']

sns.factorplot(x='Manufacturer', y='Value', data=tmp, kind='bar' , hue = 'ImageType', size=10, aspect=3)
plt.figure(figsize=(30,10))



plt.subplot(1,2,1)



sns.set(rc={'figure.figsize':(15,7.5)})

tmp = Data['Manufacturer'].value_counts(ascending=False).to_frame().reset_index()

tmp

ax1 = sns.barplot(y='index', x = 'Manufacturer',  data=tmp)



plt.subplot(1,2,2)



sns.set(rc={'figure.figsize':(15,7.5)})

tmp = Data['ManufacturerModelName'].value_counts(ascending=False).to_frame().reset_index()

tmp

ax2 = sns.barplot(y='index', x = 'ManufacturerModelName',  data=tmp)
tmp = Data.groupby(['Manufacturer','ManufacturerModelName']).count()['PatientID'].to_frame().reset_index()

sns.factorplot(x='Manufacturer', y='PatientID', data=tmp, kind='bar' , hue = 'ManufacturerModelName', size=10 , aspect=3 )#, palette=tmp['ManufacturerModelName'])
tmp = Data['SliceThickness'].astype('float').value_counts().to_frame().reset_index().sort_values(by = 'index')

tmp.columns = ['SliceThickness','Count']

ax = sns.barplot(y='Count', x = 'SliceThickness',  data=tmp)
tmp = Data['KVP'].astype('float').value_counts().to_frame().reset_index().sort_values(by = 'index')

tmp.columns = ['KVP','Count']

ax = sns.barplot(y='Count', x = 'KVP',  data=tmp)
tmp = Data['SpacingBetweenSlices'].astype('float').value_counts().to_frame().reset_index().sort_values(by = 'index')

tmp.columns = ['SpacingBetweenSlices','Count']

ax = sns.barplot(y='Count', x = 'SpacingBetweenSlices',  data=tmp)
ig, ax = plt.subplots()



tmp = Data['TableHeight'].astype('float').value_counts().to_frame().reset_index().sort_values(by = 'index')

sns.set(rc={'figure.figsize':(20,10)})

sns.distplot(tmp["TableHeight"])



ax2 = plt.axes([0.7, 0.5, .15, .3], facecolor='y')

ax2 = sns.violinplot(y=tmp["TableHeight"],  ax=ax2)
ig, ax = plt.subplots()



tmp = Data['XRayTubeCurrent'].astype('float').value_counts().to_frame().reset_index().sort_values(by = 'index')

sns.set(rc={'figure.figsize':(20,10)})

sns.distplot(tmp["XRayTubeCurrent"])



ax2 = plt.axes([0.7, 0.5, .15, .3], facecolor='y')

ax2 = sns.violinplot(y=tmp["XRayTubeCurrent"],  ax=ax2)
tmp = Data['ConvolutionKernel'].value_counts().to_frame().reset_index().sort_values(by = 'ConvolutionKernel',ascending = False)

tmp.columns = ['Convolution Kernel','Count']

plt.xticks(rotation=45)

ax = sns.barplot(y='Count', x = 'Convolution Kernel',  data=tmp)
tmp = Data['PatientPosition'].value_counts().to_frame().reset_index().sort_values(by = 'index')

tmp.columns = ['PatientPosition','Count']

plt.xticks(rotation=45)

ax = sns.barplot(y='Count', x = 'PatientPosition',  data=tmp)
ig, ax = plt.subplots()



tmp = Data['InstanceNumber'].astype('float').value_counts().to_frame().reset_index().sort_values(by = 'index')

sns.set(rc={'figure.figsize':(20,10)})

sns.distplot(tmp["InstanceNumber"])



ax2 = plt.axes([0.7, 0.5, .15, .3], facecolor='y')

ax2 = sns.violinplot(y=tmp["InstanceNumber"],  ax=ax2)
fig = px.scatter_3d(Data, x='ImagePositionPatient_x', y='ImagePositionPatient_y', z='ImagePositionPatient_z', color='PatientID')

# fig.update_layout(autosize=False,

#                   scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),

#                   width=500, height=500,

#                   margin=dict(l=65, r=50, b=65, t=90)

# )

fig.update_traces(marker=dict(size=5,

                              line=dict(width=0,

                                        color='DarkSlateGrey')),

                  selector=dict(mode='markers'))

fig.update_layout(showlegend=False) 

fig.show()

tmp1 = Data[['ImagePositionPatient_x','ImagePositionPatient_y', 'ImagePositionPatient_z', 'ImageOrientationPatient_a','ImageOrientationPatient_b', 'ImageOrientationPatient_c']]

tmp1.columns = ['x','y','z','a','b','c']



tmp1['Cos'] = 'red'

tmp2 = Data[['ImagePositionPatient_x','ImagePositionPatient_y', 'ImagePositionPatient_z', 'ImageOrientationPatient_d','ImageOrientationPatient_e', 'ImageOrientationPatient_f']]

tmp2.columns = ['x','y','z','a','b','c']

tmp2['Cos'] = 'blue'



cos = pd.concat([tmp1, tmp2], ignore_index = True)

cos['width'] = 10



cos[['a','b','c']] = cos[['a','b','c']] * 200



fig = plt.figure()

ax = fig.gca(projection='3d')

ax.view_init(60, 35)

ax.quiver(cos['x'], cos['y'], cos['z'], cos['a'], cos['b'], cos['c'], length=0.1, colors = cos['Cos'])



plt.show()



fig = plt.figure()

ax = fig.gca(projection='3d')



ax.quiver(cos['x'], cos['y'], cos['z'], cos['a'], cos['b'], cos['c'], length=0.1, colors = cos['Cos'])



plt.show()
fig, axes = plt.subplots(nrows=2, ncols=3)

Data['ImageOrientationPatient_a'].plot.hist(title  = 'Alpha 1', ax=axes[0,0])

Data['ImageOrientationPatient_b'].plot.hist(title  = 'Beta 1', ax=axes[0,1])

Data['ImageOrientationPatient_c'].plot.hist(title  = 'Gamma 1', ax=axes[0,2])

Data['ImageOrientationPatient_d'].plot.hist(title  = 'Alpha 2', ax=axes[1,0])

Data['ImageOrientationPatient_e'].plot.hist(title  = 'Beta 2', ax=axes[1,1])

Data['ImageOrientationPatient_f'].plot.hist(title  = 'Gamma 2', ax=axes[1,2])
tmp = Data.PositionReferenceIndicator.value_counts().to_frame().reset_index()

tmp.columns = ['Position Reference Indicator','Count']



sns.barplot(x="Position Reference Indicator", y="Count", data=tmp)
# Data.SliceLocation.astype(float).value_counts()



tmp = Data['SliceLocation'].astype('float').value_counts().to_frame().reset_index().sort_values(by = 'index')

sns.set(rc={'figure.figsize':(20,10)})

sns.distplot(tmp["SliceLocation"])



ax2 = plt.axes([0.7, 0.5, .15, .3], facecolor='y')

ax2 = sns.violinplot(y=tmp["SliceLocation"],  ax=ax2)
sns.jointplot(x="Rows", y="Columns", data=Data[['Rows','Columns']].astype('int'), kind='reg',joint_kws={'color':'green'})
sns.jointplot(x="PixelSpacing_row", y="PixelSpacing_column", data=Data[['PixelSpacing_row','PixelSpacing_column']].astype('float'),kind='reg',joint_kws={'color':'green'})
plt.figure(figsize=(30,10))



plt.subplot(1,2,1)

tmp = Data.BitsStored.value_counts().to_frame().reset_index()

tmp.columns = ['Bits Stored','Count']

sns.barplot(x="Bits Stored", y="Count", data=tmp)



plt.subplot(1,2,2)

tmp = Data.HighBit.value_counts().to_frame().reset_index()

tmp.columns = ['High Bit','Count']

sns.barplot(x="High Bit", y="Count", data=tmp)
sns.jointplot(x="HighBit", y="BitsStored", data=Data[['HighBit','BitsStored']].astype('float'), kind='reg',joint_kws={'color':'green'})
tmp = Data.PixelRepresentation.value_counts().to_frame().reset_index()

tmp.columns = ['Pixel Representation','Count']

sns.barplot(x="Pixel Representation", y="Count", data=tmp)
plt.figure(figsize=(30,10))



plt.subplot(1,2,1)



tmp = Data.WindowCenter.value_counts().to_frame().reset_index()

tmp.columns = ['Window Center','Count']

sns.barplot(x="Window Center", y="Count", data=tmp)



plt.subplot(1,2,2)



tmp = Data.WindowWidth.value_counts().to_frame().reset_index()

tmp.columns = ['Window Width','Count']

sns.barplot(x="Window Width", y="Count", data=tmp)
sns.jointplot(x="WindowCenter", y="WindowWidth", data=Data[Data['WindowCenter'] != '[-500, 40]'][['WindowCenter','WindowWidth']].astype('float'), )
tmp = Data.groupby(['WindowCenter','WindowWidth']).count().reset_index()[['WindowCenter','WindowWidth','ImageType']]

tmp.columns = ['WindowCenter','WindowWidth','Count']

tmp['Window Center and Width'] = tmp['WindowCenter'] + ' | ' + tmp['WindowWidth']

sns.barplot(x="Window Center and Width", y="Count", data=tmp)
plt.figure(figsize=(30,10))



plt.subplot(1,2,1)



tmp = Data.RescaleIntercept.astype('float').value_counts().to_frame().reset_index()

tmp.columns = ['Rescale Intercept','Count']

sns.barplot(x="Rescale Intercept", y="Count", data=tmp)



plt.subplot(1,2,2)



tmp = Data.RescaleSlope.astype('float').value_counts().to_frame().reset_index()

tmp.columns = ['Rescale Slope','Count']

sns.barplot(x="Rescale Slope", y="Count", data=tmp)
img_array = []



for filename in glob.glob('../input/osic-pulmonary-fibrosis-progression/train/ID00061637202188184085559/*.dcm'):

    

    img = pydicom.dcmread(filename)

    img_array.append(img.pixel_array)

imageio.mimsave('movie.gif', img_array)



HTML('<img src="./movie.gif">')
plt.figure(figsize=(20,15))

i=1

for filename in glob.glob('../input/osic-pulmonary-fibrosis-progression/train/ID00048637202185016727717/*.dcm'): 

    plt.subplot(5,6,i)

    plt.grid(False)

    plt.imshow(pydicom.dcmread(filename).pixel_array, cmap=plt.cm.bone)

    i = i + 1
plt.figure(figsize=(30,15))

i=1

for filename in glob.glob('../input/osic-pulmonary-fibrosis-progression/test/ID00421637202311550012437/*.dcm'): 

    plt.subplot(6,11,i)

    plt.grid(False)

    plt.imshow(pydicom.dcmread(filename).pixel_array, cmap=plt.cm.bone)

    i = i + 1
plt.figure(figsize=(20,10))

img = '../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/18.dcm'



plt.subplot(1,2,1)

plt.grid(False)

plt.imshow(pydicom.dcmread(img).pixel_array, cmap=plt.cm.bone)

plt.title("Original")



plt.subplot(1,2,2)

plt.grid(False)

test = cv2.bitwise_not(pydicom.dcmread(img).pixel_array)

plt.title("invert the image")



plt.imshow(test, cmap=plt.cm.bone)
image = pydicom.dcmread(img).pixel_array





imageio.imwrite('img.jpg', image)

image = imageio.imread('./img.jpg')



plt.figure(figsize=(30, 30))

plt.subplot(3, 2, 1)

plt.grid(False)

plt.title("Original")

plt.imshow(image, cmap=plt.cm.bone)



ret,thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)



plt.subplot(3, 2, 2)

plt.grid(False)

plt.title("Threshold Binary")

plt.imshow(thresh1, cmap=plt.cm.bone)



# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# image = np.array(image, dtype=np.uint8)

# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image = cv2.GaussianBlur(image, (3, 3), 0)

# print(image)

# image = image.reshape(768, 768, 1)

thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5) 



plt.subplot(3, 2, 3)

plt.grid(False)

plt.title("Adaptive Mean Thresholding")

plt.imshow(thresh, cmap=plt.cm.bone)





_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



plt.subplot(3, 2, 4)

plt.grid(False)

plt.title("Otsu's Thresholding")

plt.imshow(th2, cmap=plt.cm.bone)





plt.subplot(3, 2, 5)

plt.grid(False)

blur = cv2.GaussianBlur(image, (5,5), 0)

_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.title("Guassian Otsu's Thresholding")

plt.imshow(th3, cmap=plt.cm.bone)

plt.show()
image = pydicom.dcmread(img).pixel_array



plt.figure(figsize=(20, 20))

plt.subplot(3, 2, 1)

plt.grid(False)

plt.title("Original")

plt.imshow(image, cmap=plt.cm.bone)





# Let's define our kernel size

kernel = np.ones((5,5), np.uint8)



# Now we erode

erosion = cv2.erode(image, kernel, iterations = 1)



plt.subplot(3, 2, 2)

plt.grid(False)

plt.title("Erosion")

plt.imshow(erosion, cmap=plt.cm.bone)



# 

dilation = cv2.dilate(image, kernel, iterations = 1)

plt.subplot(3, 2, 3)

plt.grid(False)

plt.title("Dilation")

plt.imshow(dilation, cmap=plt.cm.bone)





# Opening - Good for removing noise

opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

plt.subplot(3, 2, 4)

plt.grid(False)

plt.title("Opening")

plt.imshow(opening, cmap=plt.cm.bone)





# Closing - Good for removing noise

closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

plt.subplot(3, 2, 5)

plt.grid(False)

plt.title("Closing")

plt.imshow(closing, cmap=plt.cm.bone)

# image = pydicom.dcmread(img).pixel_array

image = imageio.imread('./img.jpg')



height, width = image.shape



# Extract Sobel Edges

sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)



plt.figure(figsize=(20, 20))



plt.subplot(3, 2, 1)

plt.grid(False)

plt.title("Original")

plt.imshow(image, cmap=plt.cm.bone)



plt.subplot(3, 2, 2)

plt.grid(False)

plt.title("Sobel X")

plt.imshow(sobel_x, cmap=plt.cm.bone)





plt.subplot(3, 2, 3)

plt.grid(False)

plt.title("Sobel Y")

plt.imshow(sobel_y, cmap=plt.cm.bone)



sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)



plt.subplot(3, 2, 4)

plt.grid(False)

plt.title("sobel_OR")

plt.imshow(sobel_OR, cmap=plt.cm.bone)



laplacian = cv2.Laplacian(image, cv2.CV_64F)



plt.subplot(3, 2, 5)

plt.grid(False)

plt.title("Laplacian")

plt.imshow(laplacian, cmap=plt.cm.bone)



# image = np.array(image*255, dtype=np.uint8)

canny = cv2.Canny(image, 50, 120)



plt.subplot(3, 2, 6)

plt.grid(False)

plt.title("Canny")

plt.imshow(canny, cmap=plt.cm.bone)

# image = pydicom.dcmread(img).pixel_array

image = imageio.imread('./img.jpg')



plt.figure(figsize=(20, 20))



plt.subplot(2, 2, 1)

plt.grid(False)

plt.title("Original")

plt.imshow(image, cmap=plt.cm.bone)





# Grayscale

# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)



# Find Canny edges

edged = cv2.Canny(image, 30, 200)



plt.subplot(2, 2, 2)

plt.grid(False)

plt.title("Canny Edges")

plt.imshow(edged, cmap=plt.cm.bone)





# Finding Contours

# Use a copy of your image e.g. edged.copy(), since findContours alters the image

contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)



plt.subplot(2, 2, 3)

plt.grid(False)

plt.title("Canny Edges After Contouring")

plt.imshow(edged, cmap=plt.cm.bone)



print("Number of Contours found = " + str(len(contours)))



# Draw all contours

# Use '-1' as the 3rd parameter to draw all

cv2.drawContours(image, contours, -1, (0,255,0), 3)



plt.subplot(2, 2, 4)

plt.grid(False)

plt.title("Contours")

plt.imshow(image, cmap=plt.cm.bone)
train = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

data = pd.concat([train, test], ignore_index = True)



ProfileReport(data)
f, axes = plt.subplots(2 ,figsize=(30, 10), sharex=True)

# plt.subplot(1,2,1);

sns.lineplot(hue="SmokingStatus", x="Weeks", y = 'FVC', data = data, ax=axes[0])

# subplot(1,2,2);

sns.lineplot(hue="SmokingStatus", x="Weeks", y = 'Percent',  data = data, ax=axes[1])

f, axes = plt.subplots(3, figsize=(10, 20), sharex=True)

sns.violinplot(x="SmokingStatus", y="Age", data=data, ax=axes[0])





tmp = data.groupby(['SmokingStatus', 'Sex']).count()['Patient'].reset_index()

tmp.columns= ['Smoking Status', 'Sex', 'Count']



sns.barplot(x="Smoking Status", y="Count", hue="Sex", data=tmp, ax=axes[1])



tmp = data.groupby('SmokingStatus').count()['Patient'].reset_index()

tmp.columns= ['Smoking Status', 'Count']

sns.barplot(x="Smoking Status", y="Count", data= tmp, ax=axes[2])
f, axes = plt.subplots(2 ,figsize=(30, 10), sharex=True)

# plt.subplot(1,2,1);

sns.lineplot(hue="Sex", x="Weeks", y = 'FVC', data = data, ax=axes[0])

# subplot(1,2,2);

sns.lineplot(hue="Sex", x="Weeks", y = 'Percent',  data = data, ax=axes[1])
f, axes = plt.subplots(3, figsize=(10, 20), sharex=True)

sns.violinplot(x="Sex", y="Age", data=data, ax=axes[0])





tmp = data.groupby(['Sex', 'SmokingStatus']).count()['Patient'].reset_index()

tmp.columns= ['Sex', 'Smoking Status', 'Count']



sns.barplot(x="Sex", y="Count", hue="Smoking Status", data=tmp, ax=axes[1])



tmp = data.groupby('Sex').count()['Patient'].reset_index()

tmp.columns= ['Sex', 'Count']

sns.barplot(x="Sex", y="Count", data= tmp, ax=axes[2])
tmp = pd.cut(data.Age, 3).to_frame().merge(data,left_index=True, right_index=True)

tmp.columns = ['Age_Range', 'Patient', 'Weeks', 'FVC', 'Percent','Age', 'Sex', 'SmokingStatus']



f, axes = plt.subplots(2 ,figsize=(30, 10), sharex=True)

# plt.subplot(1,2,1);

sns.lineplot(hue="Age_Range", x="Weeks", y = 'FVC', data = tmp, ax=axes[0])

# subplot(1,2,2);

sns.lineplot(hue="Age_Range", x="Weeks", y = 'Percent',  data = tmp, ax=axes[1])



# sns.distplot(data.Age)
sns.distplot(data.Age)
# sns.lineplot( x="FVC", y = 'Percent',  data = tmp)



f, axes = plt.subplots(2 ,figsize=(30, 10), sharex=True)

# plt.subplot(1,2,1);

sns.lineplot( x="Weeks", y = 'FVC', data = tmp, ax=axes[0])

# subplot(1,2,2);

sns.lineplot( x="Weeks", y = 'Percent',  data = tmp, ax=axes[1])



# sns.distplot(data.Age)
sns.jointplot(x="FVC", y="Percent", data=data, kind='reg',

                  joint_kws={'line_kws':{'color':'green'}})
corr = data[['Weeks','FVC','Percent','Age','Sex','SmokingStatus']].corr()



mask = np.triu(np.ones_like(corr, dtype=np.bool))



f, ax = plt.subplots(figsize=(11, 9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
sns.pairplot(data[['Weeks','FVC','Percent','Age','Sex','SmokingStatus']])