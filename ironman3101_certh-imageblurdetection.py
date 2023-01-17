# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import cv2
import sklearn
import seaborn as sb

from skimage.color import rgb2gray
from skimage.filters import laplace, sobel, roberts

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print('Sobel operator:\n',np.matrix([[1,0,-1],[2,0,-2],[1,0,-1]]))
print('Laplacian operator:\n',np.matrix([[0,-1,0],[-1,4,-1],[0,-1,0]]))
ab_path ='../input/artificially-blurred/Artificially-Blurred/'
nb_path='../input/naturally-blurred/Naturally-Blurred/'
ud_path ='../input/undistorted/Undistorted/'



img_paths = ['../input/artificially-blurred/Artificially-Blurred/DiskR10_DSC02106.JPG','../input/naturally-blurred/Naturally-Blurred/15-08-07_1512.jpg','../input/undistorted/Undistorted/100_2088.JPG']
 
    
    
def show_images(path):
    plt.figure(figsize=(10,10))
    for i in range(len(path)):
        x=plt.imread(path[i])
        plt.subplot(1, 3, i+1)
        plt.imshow(x)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()
show_images(img_paths)
def show_images(path):
    plt.figure(figsize=(10,10))
    for i in range(len(path)):
        x=cv2.imread(path[i],0)
        l = laplace(x)
        plt.subplot(1, 3, i+1)
        plt.imshow(l,cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()
show_images(img_paths)
def show_images(path):
    plt.figure(figsize=(10,10))
    for i in range(len(path)):
        x=cv2.imread(path[i],0)
        l = sobel(x)
        plt.subplot(1, 3, i+1)
        plt.imshow(l,cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()
show_images(img_paths)
def show_images(path):
    plt.figure(figsize=(10,10))
    for i in range(len(path)):
        x=cv2.imread(path[i],0)
        l = roberts(x)
        plt.subplot(1, 3, i+1)
        plt.imshow(l,cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()
show_images(img_paths)
ab_images = os.listdir(ab_path)
nb_images = os.listdir(nb_path)
undistorted = os.listdir(ud_path)
def get_data(path,images):
    features=[]
    for img in images:
        feature=[]
        image_gray = cv2.imread(path+img,0)
        lap_feat = laplace(image_gray)
        sob_feat = sobel(image_gray)
        rob_feat = roberts(image_gray)
        feature.extend([img,lap_feat.mean(),lap_feat.var(),np.amax(lap_feat),
                        sob_feat.mean(),sob_feat.var(),np.max(sob_feat),
                        rob_feat.mean(),rob_feat.var(),np.max(rob_feat)])
        
        features.append(feature)
    return features
ab_images_features = get_data(ab_path,ab_images)
nb_images_features = get_data(nb_path,nb_images)
undistorted_features = get_data(ud_path,undistorted)
# #evaluation set

# db_ES_features= get_data(db_ES_path, db_ES)
# nb_ES1_features= get_data(nb_ES1_path, nb_ES1)
# nb_ES2_features= get_data(nb_ES2_path, nb_ES2)

ab_df = pd.DataFrame(ab_images_features)
ab_df.drop(0,axis=1,inplace=True)
ab_df.head()
nb_df = pd.DataFrame(nb_images_features)
nb_df.drop(0,axis=1,inplace=True)
nb_df.head()
undistorted_df = pd.DataFrame(undistorted_features)
undistorted_df.drop(0,axis=1,inplace=True)
undistorted_df.head()
# #evaluation set

# db_ES_df= pd.DataFrame(db_ES_features)
# db_ES_df.drop(0, axis= 1, inplace= True)
# print(db_ES_df.head())

# nb_ES1_df= pd.DataFrame(nb_ES1_features)
# nb_ES1_df.drop(0, axis= 1, inplace= True)
# print(nb_ES1_df.head())

# nb_ES2_df= pd.DataFrame(nb_ES2_features)
# nb_ES2_df.drop(0, axis= 1, inplace= True)
# print(nb_ES2_df.head())


# dbY=  pd.read_excel('../input/evalset/DigitalBlurSet.xlsx', sheetname='Sheet1')
# dbY.drop(0, axis= 1, inplace= True)

# nbY=  pd.read_excel('../input/evalset/NaturalBlurSet.xlsx', sheetname= 'Sheet1')
# nbY.drop(0, axis= 1, inplace= True)
label = ['Artificially_Blurred','Naturally_Blurred','Undistorted']
no_images=[len(ab_images_features),len(nb_images_features),len(undistorted_features)]
def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.bar(index, no_images)
    plt.xlabel('Image_type', fontsize=10)
    plt.ylabel('No of Images', fontsize=10)
    plt.xticks(index, label, fontsize=10, rotation=0)
    plt.title('Data Visualization')
    plt.show()
plot_bar_x()
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score, classification_report
images=pd.DataFrame()

images = images.append(undistorted_df)
images = images.append(nb_df)
all_features = np.array(images)
y_f = np.concatenate((np.zeros((undistorted_df.shape[0], ))-1, np.ones((nb_df.shape[0], ))), axis=0)

x_train,x_valid,y_train,y_valid = train_test_split(all_features,y_f,test_size=0.33,stratify=y_f)

svm_model = svm.SVC(C=100,kernel='linear')
svm_model.fit(x_train,y_train)
pred =svm_model.predict(x_valid)
print('Accuracy:',accuracy_score(y_valid,pred))
print('Confusion matrix:\n',confusion_matrix(y_valid,pred))
print('F1_score:',f1_score(y_valid,pred))
print('Classification_report:\n',classification_report(y_valid,pred))

print(pred)
svm_model = svm.SVC(C=100,kernel='rbf')
svm_model.fit(x_train,y_train)
pred =svm_model.predict(x_valid)
print('Accuracy:',accuracy_score(y_valid,pred))
print('Confusion matrix:\n',confusion_matrix(y_valid,pred))
print('F1_score:',f1_score(y_valid,pred))
print('Classification_report:\n',classification_report(y_valid,pred))
images=pd.DataFrame()

images = images.append(undistorted_df)
images = images.append(nb_df)
images = images.append(ab_df)
all_features = np.array(images)
y_f = np.concatenate((np.zeros((undistorted_df.shape[0], ))-1, np.ones((nb_df.shape[0]+ab_df.shape[0], ))), axis=0)

x_train,x_valid,y_train,y_valid = train_test_split(all_features,y_f,test_size=0.33,stratify=y_f)

svm_model = svm.SVC(C=100,kernel='rbf')
svm_model.fit(x_train,y_train)
pred =svm_model.predict(x_valid)
print('Accuracy:',accuracy_score(y_valid,pred))
print('Confusion matrix:\n',confusion_matrix(y_valid,pred))
print('F1_score:',f1_score(y_valid,pred))
print('Classification_report:\n',classification_report(y_valid,pred))
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score, classification_report
from keras.utils import to_categorical
images=pd.DataFrame()

images = images.append(undistorted_df)
images = images.append(nb_df)
images = images.append(ab_df)
all_features = np.array(images)
y_f = np.concatenate((np.zeros((undistorted_df.shape[0], ))-1, np.ones((nb_df.shape[0], )), 2*np.ones((ab_df.shape[0], ))-1), axis=0)

x_train,x_valid,y_train,y_valid = train_test_split(all_features,y_f,test_size=0.3,stratify=y_f)



svm_model = svm.SVC(C=100,kernel='rbf')
svm_model.fit(x_train,y_train)
pred =svm_model.predict(x_valid)
print('Accuracy:',accuracy_score(y_valid,pred))
print('Confusion matrix:\n',confusion_matrix(y_valid,pred))
# y_valid_cat = to_categorical(y_valid, num_classes=2)
# pred_cat = to_categorical(pred, num_classes=2)
# print('F1_score:',f1_score(y_valid_cat,pred_cat, average='weighted'))
# print('Classification_report:\n',classification_report(y_valid_cat,pred_cat))

print(pred)






#evaluation_set

db_ES_path= '../input/digitalblurset-evaluationset/DigitalBlurSet/'
nb_ES1_path= '../input/nb-es-1/NaturalBlurSet_1/'
nb_ES2_path= '../input/nb-es-2/NaturalBlurSet_2/'

# evalutaion_Set

db_ES = os.listdir(db_ES_path)
nb_ES1= os.listdir(nb_ES1_path)
nb_ES2= os.listdir(nb_ES2_path)


#evaluation set

db_ES_features= get_data(db_ES_path, db_ES)
nb_ES1_features= get_data(nb_ES1_path, nb_ES1)
nb_ES2_features= get_data(nb_ES2_path, nb_ES2)

#evaluation set

db_ES_df= pd.DataFrame(db_ES_features)

# db_ES_df.drop(0, axis= 1, inplace= True)
print(db_ES_df.head())

nb_ES1_df= pd.DataFrame(nb_ES1_features)
# nb_ES1_df.drop(0, axis= 1, inplace= True)
print(nb_ES1_df.head())

nb_ES2_df= pd.DataFrame(nb_ES2_features)
# nb_ES2_df.drop(0, axis= 1, inplace= True)
print(nb_ES2_df.head())




#make the correction here for the datasets, chage the heading for the original dataset, otherwise NaN values will occur

dbY=  pd.read_csv('../input/digitalblures/DigitalBlur_modified.csv')
# dbY.drop('Image Name', axis= 1, inplace= True)
print(dbY.shape)

nbY=  pd.read_csv('../input/esmodified-123/NaturalBlurSet_modified.csv')
# nbY.drop('Image Name', axis= 1, inplace= True)
print(nbY.shape)
ES= pd.DataFrame()
ES= ES.append(db_ES_df)
ES= ES.append(nb_ES1_df)
ES= ES.append(nb_ES2_df)

ES.sort_values(by=0, inplace=True)
filename= ES[0].values

ES.drop(0, axis=1, inplace=True)

Y_ES= pd.DataFrame()
Y_ES= Y_ES.append(dbY)
Y_ES= Y_ES.append(nbY)

Y_ES.sort_values(by=['Image Name'], inplace=True)

Y_ES.drop('Image Name', axis= 1, inplace= True)


original_label= Y_ES['Blur Label'].values

x_ES_feat= np.array(ES)
y_ES= Y_ES.values



print(filename)
print(x_ES_feat)
pred_ES =svm_model.predict(x_ES_feat)
print('Accuracy:',accuracy_score(y_ES,pred_ES))
print('Confusion matrix:\n',confusion_matrix(y_ES,pred_ES))
# y_valid_cat_ES = to_categorical(y_ES, num_classes=2)
# pred_cat_ES = to_categorical(pred_ES, num_classes=2)
# print('F1_score:',f1_score(y_valid_cat_ES,pred_cat_ES, average='weighted'))
# print('Classification_report:\n',classification_report(y_valid_cat_ES,pred_cat_ES))

# np.savetxt("foo.csv", pred, delimiter=",")
# print(y_ES)
















filename.reshape(1480, 1)
pred_ES.reshape(1480,1), 
original_label.reshape(1480, 1)
d1= pd.DataFrame({'name':filename,'original label':original_label, 'predictions':pred_ES})
d1.head()
d1.to_csv('results')
# import the modules we'll need
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))

# create a link to download the dataframe
create_download_link(d1)















