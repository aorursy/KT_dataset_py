# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
3# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from os import listdir

pathtest = '/kaggle/input/thai-mnist-classification/test/'
pathtrain = '/kaggle/input/thai-mnist-classification/train/'

def find_pic_name(patht, suffix=".png"):
    filenames = listdir(patht)
    names = [name for name in filenames if name.endswith(suffix)]
    return [str(filename) for filename in names]

pictest = find_pic_name(pathtest, ".png")
pictrain = find_pic_name(pathtrain, ".png" )
len(pictest)
# load file label
Y = pd.read_csv(r'/kaggle/input/thai-mnist-classification/mnist.train.map.csv')
Y.head()


#function get picture :thanks to @22p21c0022-Lay

lebelpath  = r'/kaggle/input/thai-mnist-classification/mnist.train.map.csv'

class getdata():
    def __init__(self,data_path,label_path):
        self.dataPath = data_path
        self.labelPath = label_path
        self.label_df = pd.read_csv(label_path)
        self.dataFile = self.label_df['id'].values
        self.label = self.label_df['category'].values
        self.n_index = len(self.dataFile)
        
    
    def get1img(self,img_index,mode='rgb',label = False):
        img = cv2.imread(os.path.join(self.dataPath,self.label_df.iloc[img_index]['id']) )
        if mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == 'gray':
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if label:
            return img,self.label_df.iloc[img_index]['category']
        return img
gdt = getdata(pathtrain,lebelpath)
#function resize picture and increase picture thresholding :thanks to @22p21c0022-Lay

import matplotlib.pyplot as plt
from PIL import Image
from IPython import display
import os.path
import cv2
from skimage import feature
from skimage import measure
 
from skimage.morphology import convex_hull_image
from skimage.util import invert

temp_img = invert(gdt.get1img(1000,'gray'))
fig, [ax1,ax2] = plt.subplots(1, 2)
ax1.imshow(temp_img)
cvh =  convex_hull_image(temp_img)
ax2.imshow(cvh)

def convex_crop(img,pad=20):
    convex = convex_hull_image(img)
    r,c = np.where(convex)
    while (min(r)-pad < 0) or (max(r)+pad > img.shape[0]) or (min(c)-pad < 0) or (max(c)+pad > img.shape[1]):
        pad = pad - 1
    return img[min(r)-pad:max(r)+pad,min(c)-pad:max(c)+pad]

def thes_resize(img,thes=40):
    img = invert(img)
    img = convex_crop(img,pad=20)
    img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 300):
        img = cv2.resize(img,(300,300))
        img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 150):
        img = cv2.resize(img,(150,150))
        img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(80,80))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(50,50))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(32,32))
    img = ((img > thes)*255).astype(np.uint8)
    return img
X = []
for i in range(gdt.n_index):
    X.append(thes_resize(gdt.get1img(i,'gray')))
    if (i+1) % 1000 == 0:
        print(i)
X = np.array(X)
y = Y['category'].values
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, Input, layers, losses, optimizers, datasets
from matplotlib import pyplot as plt

X = X.astype(np.float32) / 255.0
# Xtest = Xtest.astype(np.float32) / 255.0
X = X.reshape(-1, 32*32)
# Xtest = Xtest.reshape(-1, 32*32)
idx = np.random.permutation(np.arange(len(X)))
split = 8255 * 8 // 10
train_idx = idx[:split]
test_idx = idx[split:]
Xtrain, Ytrain = X[train_idx], y[train_idx].astype(np.int)
Xtest, Ytest = X[test_idx], y[test_idx].astype(np.int)
Xtrain.shape
Y.iloc[test_idx]
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
clf = SVC()
clf.fit(X, y.astype(np.int))
Z = clf.predict(Xtest)
print('accuracy rate=', accuracy_score(Ytest, Z))
print('confusion matrix:')
print(confusion_matrix(Ytest, Z))
import joblib
joblib.dump(clf, 'svc.mod')
# manage input data
Zall = clf.predict(X)
print('accuracy rate=', accuracy_score(y, Zall))
print('confusion matrix:')
print(confusion_matrix(y, Zall))
print('accuracy rate=', accuracy_score(y, Zall))
print('confusion matrix:')
print(confusion_matrix(y, Zall))
Y['predict'] = Zall
outlier = Y[Y['category'] != Y['predict']]
outlier.predict.value_counts()
# may be need to do Augmentation on 4,9,8,5 to improve learning
import matplotlib.image as mpimg

# to see if model make a mistake prediction or it actually outlier
_X = []
for index in (outlier.index):
    display.clear_output(wait=True)
    picture_path = pathtrain+'/'+str(outlier.loc[index, 'id'])
    picshow = mpimg.imread(picture_path)
#     _X.append(resize(picshow, (224, 224)))
    # print(picshow.shape)
    print('cat:', outlier.loc[index, 'category'], 'predict:', outlier.loc[index, 'predict'], '---', index)
    plt.imshow(picshow)
    plt.axis('On')    
    plt.show()
    input()

# X = np.array(_X)
# y = Y['category']
# del _X
# print(X[0].shape)
train2 = pd.read_csv(r'/kaggle/input/thai-mnist-classification/train.rules.csv')
test2 = pd.read_csv(r'/kaggle/input/thai-mnist-classification/test.rules.csv')
pathtest = '/kaggle/input/thai-mnist-classification/test/'
train2
X2name = train2['feature2']
X2name
import cv2
import joblib

svcmod = joblib.load(r'../input/svc-model/svc.mod')
import matplotlib.pyplot as plt
from PIL import Image
from IPython import display
import os.path
import cv2
from skimage import feature
from skimage import measure
import matplotlib.image as mpimg
from skimage.morphology import convex_hull_image
from skimage.util import invert


def getimg(imgindex, train2, f='feature2',  mode='gray' ):
    img = cv2.imread(os.path.join(pathtrain, train2.loc[imgindex, f]))
    if mode == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == 'gray':
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def convex_crop(img,pad=20):
    convex = convex_hull_image(img)
    r,c = np.where(convex)
    while (min(r)-pad < 0) or (max(r)+pad > img.shape[0]) or (min(c)-pad < 0) or (max(c)+pad > img.shape[1]):
        pad = pad - 1
    return img[min(r)-pad:max(r)+pad,min(c)-pad:max(c)+pad]

def thes_resize(img,thes=40):
    img = invert(img)
    img = convex_crop(img,pad=20)
    img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 300):
        img = cv2.resize(img,(300,300))
        img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 150):
        img = cv2.resize(img,(150,150))
        img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(80,80))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(50,50))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(32,32))
    img = ((img > thes)*255).astype(np.uint8)
    return img




train2.index
pf2 = []
pf3 = []

for i in (train2.index):
    pf2.append(thes_resize(getimg(i, train2, 'feature2','gray')))
    pf3.append(thes_resize(getimg(i, train2, 'feature3','gray')))
    if (i+1) % 1000 == 0:
        print(i)
pf2 = np.array(pf2)
pf3 = np.array(pf3)
pf1 = []

for i in (train2.index):
    if train2.loc[i, 'feature1'] is np.nan:
        pf1.append(np.nan)
    else:
        pic = thes_resize(getimg(i, train2, 'feature1','gray'))
        pic = pic.astype(np.float32) / 255.0
        pic = pic.reshape(32*32)
        predict_f1 = svcmod.predict(pic.reshape(1,-1))
        
        pf1.append(predict_f1)
        
    if (i+1) % 1000 == 0:
        print(i)
pf1 = np.array(pf1)
pf2 = pf2.astype(np.float32) / 255.0
pf2 = pf2.reshape(-1, 32*32)
predict_f2 = svcmod.predict(pf2)

pf3 = pf3.astype(np.float32) / 255.0
pf3 = pf3.reshape(-1, 32*32)
predict_f3 = svcmod.predict(pf3)
newpf1 =[]
for item in pf1:
    if item is np.nan:
        newpf1.append(np.nan)
    else:
        newpf1.append(item[0])
newpf1
train2['predict_f2'] = predict_f2
train2['predict_f3'] = predict_f3
train2['pf_1'] = newpf1
# save predicted file to csv and download to further process
train2.to_csv(r'./train_regression.csv')
# load predicted file after save and download last night

import pandas as pd
import numpy as np

path_predicted_csv = r'../input/for-regreesion/train_regression.csv'
train_s2 = pd.read_csv(path_predicted_csv)
train_s2
train_s2.columns
T_1nonan = train_s2[['predict','predict_f2', 'predict_f3', 'pf_1']][~train_s2['pf_1'].isna()]
T_nan = train_s2[['predict','predict_f2', 'predict_f3']][train_s2['pf_1'].isna()]
T_all = train_s2[['predict','predict_f2', 'predict_f3', 'pf_1']].fillna(-1)
len(T_nan.predict.unique())
284/17
def input_df(test_set):
    data_name= 'all'


    if test_set is T_nan:    
        test_column = ['predict_f3', 'predict_f2']
        data_name = 'T_nan'
    elif test_set is T_1nonan:
        test_column = ['predict_f3', 'predict_f2', 'pf_1']
        data_name = 'T_1nonan'
    elif test_set is T_all:
        test_column = ['predict_f3', 'predict_f2', 'pf_1']
        data_name = 'T_all'

    ttest = test_set[test_column]
    yall = test_set['predict'].to_numpy()
    ttest = ttest.to_numpy()
    return ttest, yall, data_name
def split_train_test(ttest, yall):
    idx_all = np.random.permutation(np.arange(len(ttest)))
    split = len(ttest) * 8 // 10
    trainall = idx_all[:split]
    testall = idx_all[split:]
    X_train, Y_train = ttest[trainall], yall[trainall]
    X_test, Y_test = ttest[testall], yall[testall]
    
    return X_train,Y_train,X_test,Y_test
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
# sklearn.metrics.mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
scale = StandardScaler()

# df_info...T_all ... 3 features replace nan with -1
# df_info...T_1nonan ... 3 features drop row nan
# df_info...T_nan ... 2 features drop column pf_1

ttest, yall, data_name = input_df(T_1nonan)
X_train,Y_train,X_test,Y_test = split_train_test(ttest, yall)
t2_clf = SVC()
t2_clf.fit(scale.fit_transform(X_train), Y_train)
Z_t2 = t2_clf.predict(scale.fit_transform(X_test))
print(data_name,'...model...','SVC')
print('accuracy rate=', accuracy_score(Y_test, Z_t2))
print('mae=', mean_absolute_error(Y_test, Z_t2))
#all dataframe
Z2_all = t2_clf.predict(scale.fit_transform(ttest))
print(data_name, 'all','...model...','SVC')
print('accuracy rate=', accuracy_score(yall, Z2_all))
print('mae=', mean_absolute_error(yall, Z2_all))
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# df_info...T_all ... 3 features replace nan with -1
# df_info...T_1nonan ... 3 features drop row nan
# df_info...T_nan ... 2 features drop column pf_1

ttest, yall, data_name = input_df(T_all)
X_train,Y_train,X_test,Y_test = split_train_test(ttest, yall)


#--- model----
Re_model = MLPClassifier(solver='lbfgs', alpha=5,hidden_layer_sizes=(1000,200,70), random_state=1, max_iter=3000, verbose=True)
Re_model.fit(scale.fit_transform(X_train), Y_train)
Z_t2 = Re_model.predict(scale.fit_transform(X_test))
print(data_name,'...model...','linear')
# print('accuracy rate=', accuracy_score(Y_test, Z_t2))
print('mae=', mean_absolute_error(Y_test, Z_t2))
np.unique(Z_t2)
len(np.unique(Z_t2))
#all dataframe
Z2_all = Re_model.predict(scale.fit_transform(ttest))
print(data_name, 'all','...model...','linear')
print('mae=', mean_absolute_error(yall, Z2_all))
import joblib
joblib.dump(Re_model, 'mlp2.mod')
# T_1nonan['z'] = Z2_all
Z2_all
from os import listdir
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython import display
import os.path
import cv2
from skimage import feature
from skimage import measure
import matplotlib.image as mpimg
from skimage.morphology import convex_hull_image
from skimage.util import invert

def getimg_from_fold(picname, pathtest,  mode='gray' ):
    img = cv2.imread(os.path.join(pathtest, picname))
    if mode == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == 'gray':
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img


def convex_crop(img,pad=20):
    convex = convex_hull_image(img)
    r,c = np.where(convex)
    while (min(r)-pad < 0) or (max(r)+pad > img.shape[0]) or (min(c)-pad < 0) or (max(c)+pad > img.shape[1]):
        pad = pad - 1
    return img[min(r)-pad:max(r)+pad,min(c)-pad:max(c)+pad]

def thes_resize(img,thes=40):
    img = invert(img)
    img = convex_crop(img,pad=20)
    img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 300):
        img = cv2.resize(img,(300,300))
        img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 150):
        img = cv2.resize(img,(150,150))
        img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(80,80))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(50,50))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(32,32))
    img = ((img > thes)*255).astype(np.uint8)
    return img


pathtest = '/kaggle/input/thai-mnist-classification/test/'


def find_pic_name(patht, suffix=".png"):
    filenames = listdir(patht)
    names = [name for name in filenames if name.endswith(suffix)]
    return [str(filename) for filename in names]

pictest = find_pic_name(pathtest, ".png")
len(pictest)
test2 = pd.read_csv(r'/kaggle/input/thai-mnist-classification/test.rules.csv')
import cv2
import joblib

svcmod = joblib.load(r'../input/svc-model/svc.mod')
mlpmod = joblib.load(r'../input/mlp-predict/mlp2.mod')
picmap = {}
picnum = 0

for picname in pictest:
    pic = thes_resize(getimg_from_fold(picname, pathtest,'gray'))
    pic = pic.astype(np.float32) / 255.0
    pic = pic.reshape(-1, 32*32)
    picmap[str(picname)] = svcmod.predict(pic.reshape(1,-1))
    picnum += 1
    if picnum % 500 == 0:
        print(picnum)
picmap
# convert picture to feature


pf2 = []
pf3 = []
pf1 = []

for i in (test2.index):
    f2 = picmap[test2.loc[i, 'feature2']][0]
    f3 = picmap[test2.loc[i, 'feature3']][0]
    pf2.append(f2)
    pf3.append(f3)
    
    
    if test2.loc[i, 'feature1'] is np.nan:
        pf1.append(np.nan)
    else:
        
        pf1.append(picmap[test2.loc[i, 'feature1']][0])
    if (i+1) % 1000 == 0:
        print(i)

test2['f2'] = pf2
test2['f3'] = pf3
test2['f1'] = pf1

Test_all = test2[['f2', 'f3', 'f1']].fillna(-1)
Test_all = Test_all.to_numpy()
Test_all
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
final = mlpmod.predict(scale.fit_transform(Test_all))
final
test2['predict'] = final
presubmit = test2[['id', 'predict']]
presubmit
submit = pd.read_csv(r'../input/thai-mnist-classification/submit.csv')
for i in range(len(submit.index)):
    if presubmit.loc[i, 'id'] != submit.loc[i, 'id']:
        print(i)
submit['predict'] = final
submit = submit.set_index(['id'])
submit
submit.to_csv(r'./Submit.csv')