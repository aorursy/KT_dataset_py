# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', -1)

path ='/kaggle/input/coronahack-chest-xraydataset/'

meta_data = pd.read_csv(path+'Chest_xray_Corona_Metadata.csv')
meta_data.head()
meta_data.drop(['Unnamed: 0'],inplace=True,axis=1)

meta_data.head()
print(meta_data['Label'].value_counts())
print(meta_data['Label_2_Virus_category'].value_counts())
print(meta_data['Label_1_Virus_category'].value_counts())
meta_data[(meta_data['Label']=='Normal') &( meta_data['Dataset_type']=='TRAIN') ].sample(10)
meta_data[meta_data['Label_2_Virus_category']=='COVID-19'].sample(10)
COVID_19_train = meta_data[(meta_data['Dataset_type']=='TRAIN') & 
                        ((meta_data['Label']=='Normal')|(meta_data['Label']=='Pnemonia')
                         & (meta_data['Label_1_Virus_category']=='Virus'))]


COVID_19_test = meta_data[(meta_data['Dataset_type']=='TEST') & 
                        ((meta_data['Label']=='Normal')|(meta_data['Label']=='Pnemonia') 
                         & (meta_data['Label_1_Virus_category']=='Virus'))]
print(COVID_19_train.shape)
print(COVID_19_train['Label'].value_counts())
Train_path='/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/'
COVID_19_train['X-ray_Path']=Train_path +COVID_19_train.loc[:,'X_ray_image_name']

print(COVID_19_test.shape)
print(COVID_19_test['Label'].value_counts())
Test_path='/kaggle/input/coronahack-chest-xraydataset/Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/'
COVID_19_test['X-ray_Path']=Test_path +COVID_19_test.loc[:,'X_ray_image_name']

import cv2

print("[INFO] loading Train images...")
imagePaths = COVID_19_train['X-ray_Path']
data = []
Trainlabels = []

Patient_df=pd.DataFrame()


for imagePath in imagePaths:

  
  column_name=imagePath.split('/')[-1]
  label=COVID_19_train.loc[COVID_19_train['X_ray_image_name']==column_name,'Label'].tolist()

  image=cv2.imread(imagePath)
  if image.any():
#       print(image.shape)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      image=cv2.resize(image, (300, 300),interpolation=cv2.INTER_AREA)
      Patient_df[column_name]=image.flatten()

      Trainlabels.append(label[0])
  else:
    print('failed to load',imagePath)
print(len(Trainlabels))
print(Patient_df.shape)
print("[INFO] loading Test images...")

TestimagePaths =COVID_19_test['X-ray_Path']

testPatient_df = pd.DataFrame()
Testlabels = []

for imagePath in TestimagePaths:

  column_name=imagePath.split('/')[-1]
  label=COVID_19_test.loc[COVID_19_test['X_ray_image_name']==column_name,'Label'].tolist()
  image=cv2.imread(imagePath)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  image=cv2.resize(image, (300, 300),interpolation=cv2.INTER_AREA)
  testPatient_df[column_name]=image.flatten()
  Testlabels.append(label[0])
  
print(len(Testlabels))
print(testPatient_df.shape)
Trainlabels = np.array(Trainlabels)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ytrain=le.fit_transform(Trainlabels)
test_Patient_df=pd.DataFrame()
test_Patient_df = testPatient_df.T
Traindata_df=Patient_df.T
# creating a dict file  
targets = {'Pnemonia':1 ,'Normal': 0,'covid19': 2}  
y_test = [targets[item] for item in Testlabels] 
# print(y_test,Testlabels) 
import umap
import matplotlib.pyplot as plt

# Patient_df data_df
trans = umap.UMAP(n_neighbors=50,
                      min_dist=0.5,n_components = 3,random_state=123).fit(Traindata_df.values)
test_embedding = trans.transform(test_Patient_df)
import plotly.express as px
df=pd.DataFrame(data=test_embedding,columns=['comp1','comp2','comp3'])
df['label']=y_test
fig = px.scatter_3d(df, x='comp1', y='comp2', z='comp3',color='label')
fig.show()
import umap
# Patient_df data_df
trans_1 = umap.UMAP(n_neighbors=50,
                      min_dist=0.5,n_components = 3,random_state=123).fit(Traindata_df.values,y=ytrain)
test_embedding_1 = trans_1.transform(test_Patient_df)
df2=pd.DataFrame(data=test_embedding_1,columns=['comp1','comp2','comp3'])
df2['label']=y_test
import plotly.express as px
fig = px.scatter_3d(df2, x='comp1', y='comp2', z='comp3',color='label')
fig.show()
embedding_1 = umap.UMAP(n_neighbors=10,
                      min_dist=0.05,n_components = 3,metric='manhattan',random_state=123).fit(Traindata_df.values,y=ytrain)
train_embeddings = embedding_1.transform(Traindata_df.values)
plt.scatter(train_embeddings[:, 0], train_embeddings[:, 1], c=ytrain, cmap='Spectral')
test_embedding_2 = embedding_1.transform(test_Patient_df)
import matplotlib.pyplot as plt
plt.scatter(test_embedding_2[:, 0], test_embedding_2[:, 1], c=y_test, cmap='Spectral')
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
%time knn.fit(embedding_1.embedding_, ytrain)
test_knn=embedding_1.transform(test_Patient_df)
%time knn.score(test_knn, y_test)
y_pred = knn.predict(test_knn)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
