import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

import os
print(os.listdir("../input"))

patient_classes = pd.read_csv('../input/stage_2_detailed_class_info.csv')
patient_classes.head()
print("Number of classes in the dataset :: %i" %  len(patient_classes["class"].unique()))
print("Classes' names are :: %s" % patient_classes["class"].unique())
class_count = patient_classes['class'].value_counts()
class_count.plot.bar( ec="orange")
print(class_count)

train_labels = pd.read_csv('../input/stage_2_train_labels.csv')
print(train_labels.iloc[0])

dcm_file = '../input/stage_2_train_images/%s.dcm' % train_labels.patientId.tolist()[0]
dcm_data = pydicom.read_file(dcm_file)
print(dcm_data)  
patientIds = train_labels.drop_duplicates('patientId', keep = 'first').patientId.tolist()
Sex = []
Age = []
for patientId in patientIds:
    dcm_file = '../input/stage_2_train_images/%s.dcm' % patientId
    dcm_data = pydicom.read_file(dcm_file)
    Sex.append(dcm_data.PatientSex)
    Age.append(int(dcm_data.PatientAge))
patientInfo = pd.DataFrame({'patientId': patientIds, 'patientSex': Sex, 'patientAge': Age})
patientInfo.dtypes
patientAge_count = patientInfo['patientAge'].value_counts().sum()
patientSex_count = patientInfo['patientSex'].value_counts().sum()
patient_count = patientInfo['patientId'].value_counts().sum()

print("total number of patientId :: %i" % patient_count )
print("Total number of patients with Non null patientSex :: %i " % patientSex_count )
print("Total number of patients with Non null patientAge :: %i " %  patientAge_count )
print("Number of missing values to be imputed for the first field :: %i " % (patient_count - patientSex_count) )
print("Number of missing values to be imputed for the second field :: %i " % (patient_count - patientAge_count) )
patientInfo = patientInfo.set_index('patientId').join(train_labels.set_index('patientId'))[['patientSex', 'patientAge', 'Target']]
patientInfo.reset_index(inplace=True)
patientInfo.head()
patientInfo.describe()

patientInfo['patientAge'].unique()

patientInfo['patientAge'].hist()

patientInfo[patientInfo['patientAge']>=90]['patientAge'].hist(bins=50)
patientInfo[patientInfo['patientAge']>=85]['patientAge'].value_counts()

def draw_img(patient_id, title=None):
    dcm_file = '../input/stage_2_train_images/%s.dcm' % patient_id
    dcm_data = pydicom.read_file(dcm_file)
    plt.imshow(dcm_data.pixel_array)
    if title is not None:
        plt.title(title)

patients_greater_100 = patientInfo[patientInfo['patientAge']>=100]
patients_less_5 = patientInfo[patientInfo['patientAge']<=5]
patients_mid_age = patientInfo[(patientInfo['patientAge']>=30) & (patientInfo['patientAge']<= 50)]

def draw_grid(arr_patients, rows=5, columns=4, titles=None, figsize=(15, 15)):
    fig=plt.figure(figsize=figsize)
    for i in range(1, columns*rows + 1):
        if(i <= len(arr_patients)):
            fig.add_subplot(rows, columns, i)
            if titles is None:
                    draw_img(arr_patients[i - 1])
            else:
                    draw_img(arr_patients[i - 1], title=titles[i - 1])
    plt.show()
draw_grid(patients_mid_age['patientId'].tolist())
draw_grid(patients_greater_100['patientId'].tolist(), 2, 2)
draw_grid(patients_less_5['patientId'].tolist())
patientInfo['age_category'] = (patientInfo['patientAge'] // 10) * 10
ax = sns.countplot(x="age_category", hue="Target", data=patientInfo)
ax.set_title('Disease per age category')
ax.legend(title='Disease')
ax.legend()
patientInfo['patientSex'].value_counts()

draw_grid(patientInfo[patientInfo['patientSex'] == 'M']['patientId'].tolist(), 3, 3, 
          titles=patientInfo[patientInfo['patientSex'] == 'M']['patientAge'].tolist(), figsize=(15, 15))

draw_grid(patientInfo[patientInfo['patientSex'] == 'F']['patientId'].tolist(), 3, 3, 
          titles=patientInfo[patientInfo['patientSex'] == 'F']['patientAge'].tolist(), figsize=(15, 15))

ax = sns.countplot(x="patientSex", hue="Target", data=patientInfo)
ax.set_title('Disease per gender')
ax.legend(title='Disease')
ax.legend()

z = {'F': 1, 'M': 0}
patientInfo['Sex'] = patientInfo['patientSex'].map(z)

sns.pairplot(patientInfo[['Sex', 'age_category', 'Target']]);
corr = patientInfo.corr()
corr
import seaborn as sns


import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(25, 25))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmax= .3,vmin = - 0.3, center=0,
            square=True, linewidths=.5)
print("The number instances for the 0 class:: %i and for the 1 class = %i " % (train_labels.Target.value_counts()[0], train_labels.Target.value_counts()[1]) )
train_ones = train_labels[train_labels.Target == 1]
train_ones.head()
print("The number of NaN values for bounding box dimentions columns for class 1 data = %s " % train_ones.isna().sum() )

train_labels.Target.value_counts()

train_labels.iloc[110]


print('all images:', train_labels.shape[0])
print('unique images:', np.unique(train_labels.patientId.tolist()).shape[0])

def draw_mult_rects(patientIds, rows=3, cols=3, figsize=(15, 15)):
    
    fig=plt.figure(figsize=figsize)
    
    for i in range(1, len(patientIds)+1):
        fig.add_subplot(rows, cols, i)
        records = train_labels[train_labels.patientId == patientIds[i-1]]
        class_label = patient_classes[patient_classes.patientId == patientIds[i-1]]['class'].tolist()[0]
        dcm_file = '../input/stage_2_train_images/%s.dcm' % patientIds[i-1]
        dcm_data = pydicom.read_file(dcm_file)
        plt.imshow(dcm_data.pixel_array)
        plt.title(class_label)
        for j in range(records.shape[0]):
            record = records.iloc[j]                
            x = record.x
            y = record.y
            width = record.width
            height = record.height
            if x is not None:
                rect = patches.Rectangle((x,y),width,height,linewidth=1,edgecolor='r',facecolor='none')
                plt.gca().add_patch(rect)
    plt.show()

draw_mult_rects(train_labels.patientId.unique()[:9])
patientId = patientInfo.patientId
tmp_cluster = patientInfo.drop(['patientId', "Target", 'patientSex', 'age_category'], axis = 1)
tmp_cluster['patientAge'].max()
from sklearn.cluster import KMeans

def kmeans(n, tmp_cluster) :
    cluster = KMeans(n_clusters=n, max_iter=300, tol=0.0001, verbose=0, random_state = 0, n_jobs=-1).fit(tmp_cluster)
    return cluster.labels_
import matplotlib.pyplot as plt
tmp_cluster_Id = pd.DataFrame()
rows = 0
for i in range(2,10):
    columns = 3
    tmp_cluster["clusters"] = kmeans(i, tmp_cluster)
    n, bins, patches = plt.hist(tmp_cluster["clusters"], facecolor='b')
    tmp_cluster_Id = tmp_cluster.copy()
    tmp_cluster_Id["patientId"] = patientId
    pics = ((tmp_cluster_Id.drop_duplicates('clusters', keep = 'first'))['patientId']).tolist()
    
    if( len( pics ) >= 3) :
        rows = (((len(pics) - 3) / 3) + (len(pics) - 3) % 3) + 1
    else : 
        columns = 2
        rows = 1
    draw_grid(pics, columns = columns, rows = int(rows) )
    plt.show()

import cv2

def remove_borders(img_data, threshold = 10):
    img_data = img_data[:, np.max(img_data, axis=0) > threshold]
    img_data = img_data[np.max(img_data, axis=1) > threshold]
    img_data = cv2.resize(img_data, (1024, 1024))
    return img_data
def read_one_img(patientId):
    dcm_file = '../input/stage_2_train_images/%s.dcm' % patientId
    dcm_data = pydicom.read_file(dcm_file)
    img = dcm_data.pixel_array
    img = remove_borders(img)
    img = cv2.resize(img, (224, 224))
    return img
figure=plt.figure(figsize=(15,15))

for (i, j) in enumerate(pics) :
    image = read_one_img(j)
    image_pixels = remove_borders(image)
    figure.add_subplot(3, 3, i+1)
    plt.imshow(image_pixels)
    
plt.show()