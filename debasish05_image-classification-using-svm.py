! pip install mahotas
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import mahotas
files=[]
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if '.db' not in filename:
            files.append(os.path.join(dirname,filename))
            #print(os.path.join(dirname, filename))
## total files
len(files)
Parasitized_Dir='../input/cell-images-for-detecting-malaria/null'
Uninfected_Dir='../input/cell-images-for-detecting-malaria/null'
pd.DataFrame(files).sample(frac=1).reset_index(drop=True)
from sklearn.model_selection import train_test_split
class DetectMalaria:
    def __init__(self,para_dir,uninfect_dir):
        self.parasitized_dir=para_dir
        self.uninfected_dir=uninfect_dir
    def dataset(self,ratio,files):
        Dataset=pd.DataFrame(files,columns=['Path'])
        Dataset=Dataset.sample(frac=1).reset_index(drop=True)  
        trainfiles,testfiles=train_test_split(Dataset,test_size=ratio,random_state=None)
        return(trainfiles,testfiles)
    
x=DetectMalaria(Parasitized_Dir,Uninfected_Dir)
train_data,test_data=x.dataset(ratio=0.3,files=files)
def label(df):
    if 'Uninfected' in df:
        return 0
    else:
        return 1


train_data['label']=train_data['Path'].apply(label)
test_data['label']=test_data['Path'].apply(label)
train_data.iloc[0,0]
image=cv2.imread('/kaggle/input/cell-images-for-detecting-malaria/cell_images/Uninfected/C65P26N_ThinF_IMG_20150818_154050_cell_160.png')
image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
import random
## plt random 4 pics
fig,ax=plt.subplots(2,2)

for i,axes in enumerate(ax.flatten()):
    image_path=random.choice(train_data['Path'].reset_index(drop=True))
    image=cv2.imread(image_path)
    image_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    axes.imshow(image_rgb)
    if 'Uninfected' in image_path:
        axes.set_title('Uninfected')
    else:
        axes.set_title('parasite')
plt.show()
## read_image
image_gray= cv2.cvtColor(image,
                         cv2.COLOR_BGR2GRAY)
feature=cv2.HuMoments(cv2.moments(image_gray)).flatten()
print(feature)
print(mahotas.features.haralick(image_gray).mean(axis=0))
## extract the features
 
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
def fd_haralick(image): 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
 
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return(hist.flatten())
feature=[]
def dataframe(df):
        
    image=cv2.imread(df['Path'])
    print(df['Path'])
    global_feature = np.hstack([ fd_haralick(image), fd_hu_moments(image),df['label']]) 
    feature.append(global_feature)
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #Normalize The feature vectors...
    #rescaled_features = scaler.fit_transform(global_features)

train_data.apply(dataframe,axis=1)

X_train=pd.DataFrame(feature).drop(columns=[20])
y_train=train_data['label']
feature=[]
test_data.apply(dataframe,axis=1)
X_test=pd.DataFrame(feature).drop(columns=[20])
y_test=test_data['label']
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)
pred=svc.predict(X_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
## accuracy score
accuracy_score(y_test,pred)
## confusion matrix
plot_confusion_matrix(svc,X_test,y_test)
plt.show()
## here 0 means uninfected and 1 means parasite
