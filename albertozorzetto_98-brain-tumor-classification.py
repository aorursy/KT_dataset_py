!pip install natsort
!pip install videofig
from natsort import natsorted 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import os
%matplotlib inline

data=pd.read_csv('/kaggle/input/brain-tumor/Brain Tumor.csv')
data
scalable=['Mean', 'Variance', 'Standard Deviation', 'Entropy',
       'Skewness', 'Kurtosis', 'Contrast', 'Energy', 'ASM', 'Homogeneity',
       'Dissimilarity', 'Correlation', 'Coarseness']


data[scalable]=StandardScaler().fit_transform(data[scalable])
data
sns.swarmplot(x=y, y= data['Homogeneity'])
plt.title("Distribution of image Homogenity, by Class")


class1=data['Class']== 1
class0=data['Class']== 0
_data=data.copy()
_data=data.drop('Image',axis=1,inplace=False)
sns.distplot(a= _data[class1]['Energy'], label="Tumor")
sns.distplot(a = _data[class0]['Energy'], label="No tumor" )

plt.title("Distribution of image Energy, by Class")
plt.legend()
sns.distplot(a= _data[class1]['Entropy'], label="Tumor")
sns.distplot(a = _data[class0]['Entropy'], label="No tumor" )
plt.title("Distribution of image Entropy, by Class")
plt.legend()
fig = plt.figure()  
folder='/kaggle/input/brain-tumor/Brain Tumor/Brain Tumor/'
imgs=[os.path.join(folder,img) for img in os.listdir(folder) if img.endswith('.jpg')]
imgs=natsorted(imgs)

img=cv2.imread(imgs[3760],cv2.IMREAD_GRAYSCALE)

im = plt.imshow(img,  interpolation='none', aspect='auto',cmap ='gray', vmin=0, vmax=255)   
plt.title('No Tumor')
img=cv2.imread(imgs[3],cv2.IMREAD_GRAYSCALE)

im = plt.imshow(img,  interpolation='none', aspect='auto',cmap ='gray', vmin=0, vmax=255)   
plt.title('Tumor')
y=data.Class
y
from collections import OrderedDict

model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 
model.fit(data.drop(['Image','Class'],axis=1,inplace=False),y)
OrderedDict(sorted(model.get_booster().get_fscore().items(),key=lambda t: t[1], reverse=True))
logr= LogisticRegression(dual=False, verbose=1, random_state=  4)
logr.fit(X_train , y_train )
logr.score(X_valid,y_valid)
RFclf = RandomForestClassifier(n_estimators = 2000, random_state= 4 ,verbose=1)
RFclf.fit( X_train, y_train  )
RFclf.score(X_valid,y_valid)
knn=KNeighborsClassifier( algorithm='auto' ,leaf_size= 50,n_neighbors= 5)
knn.fit(X_train,y_train )
knn.score ( X_valid,y_valid)