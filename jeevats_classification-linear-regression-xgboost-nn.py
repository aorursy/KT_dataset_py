from skimage import color
import pandas as pd
import re
import random
import numpy as np
import os
import keras
!pip install metrics
import metrics
import keras.backend as k
from cv2 import imread,resize
from time import time
#from scipy.misc import imread
from sklearn.metrics import accuracy_score,normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
df_train=pd.read_csv('/content/final_train.csv')
df_test=pd.read_csv('/content/test.csv')

df_train.head()
df_test.head()
df_test.columns
print(len(df_train['anatom_site_general_challenge'].unique()))
print(len(df_train['sex'].unique()))
np.random.seed(1234)
boxplot = df_train.boxplot(column=['age_approx'])
Q1 = df_train.quantile(0.25)
Q3 = df_train.quantile(0.75)
IQR = Q3 - Q1
IQR
df_train[df_train['age_approx']==0]
for i in df_train.columns:
  column=df_train[i]
  print(i,column.isna().sum())
df_train['target'][df_train['sex'].isna()]
df_train['target'][df_train['age_approx'].isna()]
def max_count(df,col_1):
  return max(set(df[col_1]), key = list(df[col_1]).count) #.unique()

max(set(df_train['target'][df_train['anatom_site_general_challenge'].isna()]), key = list(df_train['target'][df_train['anatom_site_general_challenge'].isna()]).count) #.unique()
max(set(df_train['target']), key = list(df_train['target']).count)
anatom_mx=max_count(df_train,'anatom_site_general_challenge')

sex_mx=max_count(df_train,'sex')
#index = df_train['age_approx'].index[df_train['age_approx'].apply(np.isnan)]
def replace_val(df_train,column,val):
  index = df_train[column].index[df_train[column].isna()]
  for Index in index:
    df_train[column][Index]=val
    df_train[column][Index]=val
  return df_train

sex_le = LabelEncoder()
anatom_le=LabelEncoder()


def preprocess(df_train):
  global sex_le
  global anatom_le
  df_train=replace_val(df_train,'anatom_site_general_challenge',anatom_mx)
  df_train=replace_val(df_train,'sex',sex_mx)
  q_low = df_train["age_approx"].quantile(0.01)
  q_hi  = df_train["age_approx"].quantile(0.99)

  df_filtered = df_train[(df_train["age_approx"] < q_hi) & (df_train["age_approx"] > q_low)]
  
  try:
      df_filtered=df_filtered.drop(['diagnosis','benign_malignant'],axis=1)
      sex_le.fit(df_filtered['sex'])
      anatom_le.fit(df_filtered['anatom_site_general_challenge'])
      df_filtered['sex']=sex_le.transform(df_filtered['sex'])
      df_filtered['anatom_site_general_challenge']=anatom_le.transform(df_filtered['anatom_site_general_challenge'])
  except Exception as e:
      print(e)
      df_filtered['sex']=sex_le.transform(df_filtered['sex'])
      df_filtered['anatom_site_general_challenge']=anatom_le.transform(df_filtered['anatom_site_general_challenge'])
  return df_filtered
df_train1=preprocess(df_train)
df_train1.head()
import cv2
import glob
import random
len(list(df_train.index[df_train["target"]==0]))


def get_images(directory,df_train):
    Images=[]
    label=0
    os.chdir(directory)
    indx1=list(df_train["image_name"].index[df_train["target"]==1][:584])
    indx1.extend(list(df_train["image_name"].index[df_train["target"]==0])[:2000])
    indexes=random.sample(indx1,len(indx1))
    #print(type(indexes))
    print(len(indexes))
    for image_file in indexes:
        image=imread(df_train1["image_name"][image_file]+".jpg")
        image = color.rgb2gray(image) 
        image=cv2.resize(image,(150,150))
        Images.append(image)
        label=label+1
        if label%100==0:
            print(label)
            print(np.array(Images).shape)
    return Images
Images=get_images("/kaggle/input/siim-isic-melanoma-classification/jpeg/train",df_train1)

#Images=np.array(Images)
#Images.shape
len(Images)
for i in range(len(Images)):
    Images[i]=Images[i].astype('float32')
from sklearn.cluster import KMeans

train_x = np.stack(Images)
train_x /= 255.0
train_x = train_x.reshape(-1, 22500).astype('float32')
train_x.shape
km = KMeans(n_jobs=-1, n_clusters=2, n_init=20)
km.fit(train_x)
#pred = km.predict(val_x)

km
import pickle
os.chdir("/kaggle/working")

pickle.dump(km, open("kmeans_cluster.pkl", "wb"))


kmeans = pickle.load(open("/kaggle/input/kmeans-model/kmeans_cluster1.pkl", "rb"))

km=kmeans
from IPython.display import FileLink
os.chdir("/kaggle/working")
FileLink("kmeans_cluster.pkl")
index = ['Row'+str(i) for i in range(1, len(train_x)+1)]

df = pd.DataFrame(train_x, index=index)
df.to_csv("Images.csv",index=False)

len(df)
len(df_train1)
df_train.index
df_train2=df_train1.reset_index()
len(df_train2)
df_train2.index
Images=[]

def get_images_cluster(directory,df):
    global Images
    label=0
    os.chdir(directory)
    #print(len(indexes))
    for image_file in range(len(Images),len(df)):
        try:
            os.chdir(directory)
            image=imread(df["image_name"][image_file]+".jpg")
            image = color.rgb2gray(image) 
            image=cv2.resize(image,(150,150))
            image=[image]
            test = np.stack(image)
            test /= 255.0
            test = test.reshape(-1, 22500).astype('float32')
            pred = km.predict(test)
            Images.append(pred[0])
            label=label+1
            if label%100==0:
                print(label)
                os.chdir("/kaggle/working")
                with open("prediction.txt","w") as f:
                    for item in Images:
                        f.write("{}\n".format(item))
        except Exception as e:
            print(e)
            Images.append(0)
    return Images
Images=get_images_cluster("/kaggle/input/siim-isic-melanoma-classification/jpeg/train",df_train2)

Images
df_train2["cluster"]=Images
os.chdir("/kaggle/working")
df_train2.to_csv("final_train.csv",index=False)
df_test1=preprocess(df_test)
df_test1.head()
df_test2=df_test1.reset_index()
df_test2.index
Images=get_images_cluster("/kaggle/input/siim-isic-melanoma-classification/jpeg/test",df_test2)

#os.chdir("/kaggle/input/siim-isic-melanoma-classification/jpeg/test")
Images
df_test1["cluster"]=Images
os.chdir("/kaggle/working")
df_test1.to_csv("final_test.csv",index=False)
test.shape
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier)
!pip install catboost
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
import re
#len(df_train['patient_id'].unique())
df_train=pd.read_csv('/content/final_train.csv')
df_test=pd.read_csv('/content/final_test1.csv')
type(df_train['sex'][0])
patient=LabelEncoder()
vals=list(df_train['patient_id'])
vals.extend(df_test['patient_id'])
patient.fit(vals)


df_train['patient_id']=patient.transform(df_train['patient_id'])
'''patient_se=LabelEncoder()
vals1=list(df_test['sex'])
vals1.extend(df_test['sex'])
patient_se.fit(vals1)


df_train['sex']=patient_se.transform(df_train['sex'])'''
'''patient_an=LabelEncoder()
vals2=list(df_train['anatom_site_general_challenge'])
vals2.extend(df_test['anatom_site_general_challenge'])
patient_an.fit(vals2)


df_train['anatom_site_general_challenge']=patient_an.transform(df_train['anatom_site_general_challenge'])'''
df_test['patient_id']=patient.transform(df_test['patient_id'])
#df_test['sex']=patient_se.transform(df_test['sex'])
#df_test['anatom_site_general_challenge']=patient_an.transform(df_test['anatom_site_general_challenge'])
type(df_test['patient_id'][0])
df_test.head()
import seaborn as sns
sns.distplot(df_train['cluster'])
Y=df_train['target']
X=df_train[['patient_id', 'sex', 'age_approx','anatom_site_general_challenge', 'cluster']]
#X["A"] = X["A"] / X["A"].max()'''
seed = 7
test_size = 0.33
Y=df_train['target']
X=df_train[['patient_id', 'sex', 'age_approx','anatom_site_general_challenge', 'cluster']]
X_train=X
y_train=Y
X_test=df_test[['patient_id', 'sex', 'age_approx','anatom_site_general_challenge', 'cluster']]

train_DMatrix = xgb.DMatrix(X_train, label= y_train)
test_DMatrix = xgb.DMatrix(X_test)
param = {
    'booster':'gbtree', 
    'eta': 0.3,
    'num_class': 2,
    'max_depth': 100
}

epochs = 100
clf = xgb.XGBClassifier(n_estimators=2000, 
                        max_depth=8, 
                        objective='multi:softprob',
                        seed=0,  
                        nthread=-1, 
                        learning_rate=0.15, 
                        num_class = 2, 
                        scale_pos_weight = (len(X_train)/584))
clf.fit(X_train, y_train)
#clf.predict_proba(X_test)[:,1]
# clf.predict(x_test)


df=pd.DataFrame()
df['image_name']=df_test['image_name']

target = clf.predict_proba(X_test)[:,1]
df['target']=target
df.head()
df.to_csv('Submission_xgboost_cluster.csv',index=False)
#sub_tabular = sub.copy()
Y=df_train['target']
X=df_train[['patient_id', 'sex', 'age_approx','anatom_site_general_challenge']]
#X["A"] = X["A"] / X["A"].max()'''
seed = 7
test_size = 0.33
Y=df_train['target']
X=df_train[['patient_id', 'sex', 'age_approx','anatom_site_general_challenge']]
X_train=X
y_train=Y
X_test=df_test[['patient_id', 'sex', 'age_approx','anatom_site_general_challenge']]
train_DMatrix = xgb.DMatrix(X_train, label= y_train)
test_DMatrix = xgb.DMatrix(X_test)
param = {
    'booster':'gbtree', 
    'eta': 0.3,
    'num_class': 2,
    'max_depth': 100
}

epochs = 100
clf = xgb.XGBClassifier(n_estimators=2000, 
                        max_depth=8, 
                        objective='multi:softprob',
                        seed=0,  
                        nthread=-1, 
                        learning_rate=0.15, 
                        num_class = 2, 
                        scale_pos_weight = (len(X_train)/584))
clf.fit(X_train, y_train)
#clf.predict_proba(X_test)[:,1]
# clf.predict(x_test)

df=pd.DataFrame()
df['image_name']=df_test['image_name']

target = clf.predict_proba(X_test)[:,1]
df['target']=target
df.head()
df.to_csv('Submission_xgboost_.csv',index=False)
#sub_tabular = sub.copy()
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0).fit(X_train,y_train)
clf.predict(X_test)
df=pd.DataFrame()
df['image_name']=df_test['image_name']

target = clf.predict_proba(X_test)[:,1]
df['target']=target
df.head()
df.to_csv('Submission_Logistic_.csv',index=False)
df.head()
#prediction of the test set
#importing keras and other required library's
#importing and splitting data into training and testing 
import keras
from sklearn.model_selection import train_test_split

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler # Used for scaling of data
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import metrics
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
#creating new neural network model
classifier=Sequential()
classifier.add(Dense(output_dim=100,init='uniform',activation='relu',input_dim=X.shape[1]))
classifier.add(Dropout(p=0.2))


classifier.add(Dense(output_dim=50,init='uniform',activation='relu'))
classifier.add(Dropout(p=0.2))

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])




#trining or fitting the model
classifier.fit(X_train,y_train,batch_size=100,nb_epoch=100)
y_pred = classifier.predict(X_test)
y_pred

df=pd.DataFrame()
df['image_name']=df_test['image_name']

target = clf.predict_proba(X_test)[:,1]
df['target']=y_pred
df.to_csv('Submission_NN.csv',index=False)
df.head()
