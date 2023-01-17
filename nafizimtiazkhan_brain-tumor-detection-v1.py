from IPython.display import clear_output

!pip install imutils

!pip install -U imbalanced-learn

!pip install -U scikit-learn

clear_output()

import numpy as np 

from tqdm import tqdm

import cv2

import os

import shutil

import itertools



import imutils

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

from plotly import tools

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16, preprocess_input

from keras import layers

from keras.models import Model, Sequential

from keras.optimizers import Adam, RMSprop

from keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix

init_notebook_mode(connected=True)

import numpy as np

from sklearn import linear_model

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import AdaBoostClassifier

!pip install catboost

from catboost import CatBoostClassifier

import xgboost as xgb

from sklearn.ensemble import GradientBoostingClassifier

!pip install scikit-plot

import scikitplot as skplt

RANDOM_SEED = 43



!apt-get install tree

clear_output()

# create new folders

!mkdir TRAIN TEST VAL TRAIN/YES TRAIN/NO TEST/YES TEST/NO VAL/YES VAL/NO

!tree -d



IMG_PATH = '../input/brain-mri-images-for-brain-tumor-detection/brain_tumor_dataset/'

# split the data by train/val/test

for CLASS in os.listdir(IMG_PATH):

    if not CLASS.startswith('.'):

        IMG_NUM = len(os.listdir(IMG_PATH + CLASS))

        for (n, FILE_NAME) in enumerate(os.listdir(IMG_PATH + CLASS)):

            img = IMG_PATH + CLASS + '/' + FILE_NAME

            if n < 5:

                shutil.copy(img, 'TEST/' + CLASS.upper() + '/' + FILE_NAME)

            elif n < 0.8*IMG_NUM:

                shutil.copy(img, 'TRAIN/'+ CLASS.upper() + '/' + FILE_NAME)

            else:

                shutil.copy(img, 'VAL/'+ CLASS.upper() + '/' + FILE_NAME)
def load_data(dir_path, img_size=(100,100)):

    """

    Load resized images as np.arrays to workspace

    """

    X = []

    y = []

    i = 0

    labels = dict()

    for path in tqdm(sorted(os.listdir(dir_path))):

        if not path.startswith('.'):

            labels[i] = path

            for file in os.listdir(dir_path + path):

                if not file.startswith('.'):

                    img = cv2.imread(dir_path + path + '/' + file)

                    X.append(img)

                    y.append(i)

            i += 1

    X = np.array(X)

    y = np.array(y)

    

    

    

    print(f'{len(X)} images loaded from {dir_path} directory.')

    return X, y, labels







def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize = (6,6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    cm = np.round(cm,2)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
TRAIN_DIR = 'TRAIN/'

TEST_DIR = 'TEST/'

VAL_DIR = 'VAL/'

IMG_SIZE = (224,224)



# use predefined function to load the image data into workspace

X_train, y_train, labels = load_data(TRAIN_DIR, IMG_SIZE)

X_test, y_test, _ = load_data(TEST_DIR, IMG_SIZE)

X_val, y_val, _ = load_data(VAL_DIR, IMG_SIZE)
def crop_imgs(set_name, add_pixels_value=0):

    set_new = []

    for img in set_name:

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        gray = cv2.GaussianBlur(gray, (5, 5), 0)



        # threshold the image, then perform a series of erosions +

        # dilations to remove any small regions of noise

        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]

        thresh = cv2.erode(thresh, None, iterations=2)

        thresh = cv2.dilate(thresh, None, iterations=2)



        # find contours in thresholded image, then grab the largest one

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = imutils.grab_contours(cnts)

        c = max(cnts, key=cv2.contourArea)



        # find the extreme points

        extLeft = tuple(c[c[:, :, 0].argmin()][0])

        extRight = tuple(c[c[:, :, 0].argmax()][0])

        extTop = tuple(c[c[:, :, 1].argmin()][0])

        extBot = tuple(c[c[:, :, 1].argmax()][0])



        ADD_PIXELS = add_pixels_value

        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()

        set_new.append(new_img)



    return np.array(set_new)
X_train_crop = crop_imgs(set_name=X_train)

X_val_crop = crop_imgs(set_name=X_val)

X_test_crop = crop_imgs(set_name=X_test)

# plot_samples(X_train_crop, y_train, labels, 40)
def save_new_images(x_set, y_set, folder_name):

    i = 0

    for (img, imclass) in zip(x_set, y_set):

        if imclass == 0:

            cv2.imwrite(folder_name+'NO/'+str(i)+'.jpg', img)

        else:

            cv2.imwrite(folder_name+'YES/'+str(i)+'.jpg', img)

        i += 1

# saving new images to the folder

!mkdir TRAIN_CROP TEST_CROP VAL_CROP TRAIN_CROP/YES TRAIN_CROP/NO TEST_CROP/YES TEST_CROP/NO VAL_CROP/YES VAL_CROP/NO

save_new_images(X_train_crop, y_train, folder_name='TRAIN_CROP/')

save_new_images(X_val_crop, y_val, folder_name='VAL_CROP/')

save_new_images(X_test_crop, y_test, folder_name='TEST_CROP/')

def preprocess_imgs(set_name, img_size):

    """

    Resize and apply VGG-15 preprocessing

    """

    set_new = []

    for img in set_name:

        img = cv2.resize(

            img,

            dsize=img_size,

            interpolation=cv2.INTER_CUBIC

        )

        set_new.append(preprocess_input(img))

    return np.array(set_new)

X_train_prep = preprocess_imgs(set_name=X_train_crop, img_size=IMG_SIZE)

X_test_prep = preprocess_imgs(set_name=X_test_crop, img_size=IMG_SIZE)

X_val_prep = preprocess_imgs(set_name=X_val_crop, img_size=IMG_SIZE)
import pandas as pd

evaluation = pd.DataFrame({'Model': [],

                           'Accuracy(train)':[],

                           'Precision(train)':[],

                           'Recall(train)':[],

                           'F1_score(train)':[],

                           'Specificity(train)':[],

                           'Accuracy(test)':[],

                           'Precision(test)':[],

                           'Recalll(test)':[],

                           'F1_score(test)':[],

                           'Specificity(test)':[],

                          })

print(X_train.shape)
X_train = X_train_prep

X_test = X_test_prep

X_val = X_val_prep

py_test=np.concatenate((y_test, y_val))

py_train=y_train
p = X_train[0].flatten()

q = pd.Series(p)

q



cols = np.arange(q.shape[0])

df = pd.DataFrame(columns = cols)

df



x=0

for i in range(X_train.shape[0]):

    df.loc[x] = X_train[i].flatten()

    x = df.shape[0] + 1

    

for i in range(X_test.shape[0]):

    df.loc[x] = X_test[i].flatten()

    x = df.shape[0] + 1

    

    

for i in range(X_val.shape[0]):

    df.loc[x] = X_val[i].flatten()

    x = df.shape[0] + 1

    

df['label']=0

print(df)
df
y=np.concatenate((y_train,y_test, y_val))

# y=np.array(y)

X = df.loc[:, df.columns != 'label']

print(X.shape)

print(y.shape)

y


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

from imblearn.over_sampling import SMOTE



print('Before SMOTE')

print(X_train.shape)

print(y_train.shape)

asome = pd.DataFrame(y_train)

# print(asome.value_counts())





X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)



sm = SMOTE()

X_train, y_train = sm.fit_resample(X_train, y_train)





print('After SMOTE')

print(X_train.shape)

print(y_train.shape)

asome = pd.DataFrame(y_train)

# print(asome.value_counts())
clf =RandomForestClassifier(n_estimators=10000, random_state=500)

clf.fit(X_train, y_train)

acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='binary'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='binary'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(clf.predict(X_train), y_train).ravel()

specificity = tn / (tn+fp)

specificity_train=format(specificity,'.3f')



acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='binary'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='binary'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(clf.predict(X_test), y_test).ravel()

specificity = tn / (tn+fp)

specificity_test=format(specificity,'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['Random Forest',acc_train,precision_train,recall_train,f1_train,specificity_train,acc_test,precision_test,recall_test,f1_test,specificity_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test



y_train = pd.DataFrame(y_train)

y_train=y_train.replace([0,1], ["Negative","Positive"])



pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)

pred_train=pred_train.replace([0,1], ["Negative","Positive"])





pred_test=clf.predict(X_test)

y_test = pd.DataFrame(y_test)

y_test=y_test.replace([0,1], ["Negative","Positive"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1], ["Negative","Positive"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )

y_train=p

y_test=q
clf =svm.SVC(kernel='rbf',degree=5000)

clf.fit(X_train, y_train)

acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='binary'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='binary'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(clf.predict(X_train), y_train).ravel()

specificity = tn / (tn+fp)

specificity_train=format(specificity,'.3f')



acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='binary'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='binary'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(clf.predict(X_test), y_test).ravel()

specificity = tn / (tn+fp)

specificity_test=format(specificity,'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['SVM',acc_train,precision_train,recall_train,f1_train,specificity_train,acc_test,precision_test,recall_test,f1_test,specificity_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test



y_train = pd.DataFrame(y_train)

y_train=y_train.replace([0,1], ["Negative","Positive"])



pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)

pred_train=pred_train.replace([0,1], ["Negative","Positive"])





pred_test=clf.predict(X_test)

y_test = pd.DataFrame(y_test)

y_test=y_test.replace([0,1], ["Negative","Positive"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1], ["Negative","Positive"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )

y_train=p

y_test=q
clf =tree.DecisionTreeClassifier(random_state=500)

clf.fit(X_train, y_train)

acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='binary'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='binary'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(clf.predict(X_train), y_train).ravel()

specificity = tn / (tn+fp)

specificity_train=format(specificity,'.3f')



acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='binary'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='binary'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(clf.predict(X_test), y_test).ravel()

specificity = tn / (tn+fp)

specificity_test=format(specificity,'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['Decision Tree',acc_train,precision_train,recall_train,f1_train,specificity_train,acc_test,precision_test,recall_test,f1_test,specificity_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test



y_train = pd.DataFrame(y_train)

y_train=y_train.replace([0,1], ["Negative","Positive"])



pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)

pred_train=pred_train.replace([0,1], ["Negative","Positive"])





pred_test=clf.predict(X_test)

y_test = pd.DataFrame(y_test)

y_test=y_test.replace([0,1], ["Negative","Positive"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1], ["Negative","Positive"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )

y_train=p

y_test=q
clf =KNeighborsClassifier(n_neighbors=1)

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='binary'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='binary'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(clf.predict(X_train), y_train).ravel()

specificity = tn / (tn+fp)

specificity_train=format(specificity,'.3f')



acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='binary'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='binary'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(clf.predict(X_test), y_test).ravel()

specificity = tn / (tn+fp)

specificity_test=format(specificity,'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['KNN',acc_train,precision_train,recall_train,f1_train,specificity_train,acc_test,precision_test,recall_test,f1_test,specificity_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test



y_train = pd.DataFrame(y_train)

y_train=y_train.replace([0,1], ["Negative","Positive"])



pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)

pred_train=pred_train.replace([0,1], ["Negative","Positive"])





pred_test=clf.predict(X_test)

y_test = pd.DataFrame(y_test)

y_test=y_test.replace([0,1], ["Negative","Positive"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1], ["Negative","Positive"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )

y_train=p

y_test=q
clf =MLPClassifier(solver='lbfgs', random_state=100)

clf.fit(X_train, y_train)

acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='binary'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='binary'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(clf.predict(X_train), y_train).ravel()

specificity = tn / (tn+fp)

specificity_train=format(specificity,'.3f')



acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='binary'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='binary'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(clf.predict(X_test), y_test).ravel()

specificity = tn / (tn+fp)

specificity_test=format(specificity,'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['ANN',acc_train,precision_train,recall_train,f1_train,specificity_train,acc_test,precision_test,recall_test,f1_test,specificity_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test



y_train = pd.DataFrame(y_train)

y_train=y_train.replace([0,1], ["Negative","Positive"])



pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)

pred_train=pred_train.replace([0,1], ["Negative","Positive"])





pred_test=clf.predict(X_test)

y_test = pd.DataFrame(y_test)

y_test=y_test.replace([0,1], ["Negative","Positive"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1], ["Negative","Positive"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )

y_train=p

y_test=q
clf =GaussianNB()

clf.fit(X_train, y_train)

acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='binary'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='binary'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(clf.predict(X_train), y_train).ravel()

specificity = tn / (tn+fp)

specificity_train=format(specificity,'.3f')



acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='binary'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='binary'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(clf.predict(X_test), y_test).ravel()

specificity = tn / (tn+fp)

specificity_test=format(specificity,'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['Naive Bayes',acc_train,precision_train,recall_train,f1_train,specificity_train,acc_test,precision_test,recall_test,f1_test,specificity_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test



y_train = pd.DataFrame(y_train)

y_train=y_train.replace([0,1], ["Negative","Positive"])



pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)

pred_train=pred_train.replace([0,1], ["Negative","Positive"])





pred_test=clf.predict(X_test)

y_test = pd.DataFrame(y_test)

y_test=y_test.replace([0,1], ["Negative","Positive"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1], ["Negative","Positive"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )

y_train=p

y_test=q
clf = AdaBoostClassifier(n_estimators=1000,random_state=500)

clf.fit(X_train, y_train)

acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='binary'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='binary'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(clf.predict(X_train), y_train).ravel()

specificity = tn / (tn+fp)

specificity_train=format(specificity,'.3f')



acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='binary'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='binary'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(clf.predict(X_test), y_test).ravel()

specificity = tn / (tn+fp)

specificity_test=format(specificity,'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['AdaBoost',acc_train,precision_train,recall_train,f1_train,specificity_train,acc_test,precision_test,recall_test,f1_test,specificity_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test



y_train = pd.DataFrame(y_train)

y_train=y_train.replace([0,1], ["Negative","Positive"])



pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)

pred_train=pred_train.replace([0,1], ["Negative","Positive"])





pred_test=clf.predict(X_test)

y_test = pd.DataFrame(y_test)

y_test=y_test.replace([0,1], ["Negative","Positive"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1], ["Negative","Positive"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )

y_train=p

y_test=q
# clf = CatBoostClassifier(

# #     iterations=1000, 

# #     learning_rate=0.1, 

#     random_state=500,

#     #verbose=5,

#     #loss_function='CrossEntropy'

# )

# clf.fit(X_train, y_train)

# acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

# precision_train=format(precision_score(y_train, clf.predict(X_train), average='binary'),'.3f')

# recall_train=format(recall_score(y_train,clf.predict(X_train), average='binary'),'.3f')

# f1_train=format(f1_score(y_train,clf.predict(X_train), average='binary'),'.3f')

# tn, fp, fn, tp = confusion_matrix(clf.predict(X_train), y_train).ravel()

# specificity = tn / (tn+fp)

# specificity_train=format(specificity,'.3f')



# acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

# precision_test=format(precision_score(y_test, clf.predict(X_test), average='binary'),'.3f')

# recall_test=format(recall_score(y_test,clf.predict(X_test), average='binary'),'.3f')

# f1_test=format(f1_score(y_test,clf.predict(X_test), average='binary'),'.3f')

# tn, fp, fn, tp = confusion_matrix(clf.predict(X_test), y_test).ravel()

# specificity = tn / (tn+fp)

# specificity_test=format(specificity,'.3f')



# r = evaluation.shape[0]

# evaluation.loc[r] = ['CatBoost',acc_train,precision_train,recall_train,f1_train,specificity_train,acc_test,precision_test,recall_test,f1_test,specificity_test]

# evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
# p=y_train

# q=y_test



# y_train = pd.DataFrame(y_train)

# y_train=y_train.replace([0,1], ["Negative","Positive"])



# pred_train=clf.predict(X_train)

# pred_train=pd.DataFrame(pred_train)

# pred_train=pred_train.replace([0,1], ["Negative","Positive"])





# pred_test=clf.predict(X_test)

# y_test = pd.DataFrame(y_test)

# y_test=y_test.replace([0,1], ["Negative","Positive"])

# pred_test=pd.DataFrame(pred_test)



# pred_test=pred_test.replace([0,1], ["Negative","Positive"])



# skplt.metrics.plot_confusion_matrix(

#     y_train, 

#     pred_train,

#     figsize=(7,4),

#     title_fontsize='18',

#     text_fontsize='16',

#     title =' ',

#     cmap='BuGn'

#     )



# skplt.metrics.plot_confusion_matrix(

#     y_test, 

#     pred_test,

#     figsize=(7,4),

#     title_fontsize='18',

#     text_fontsize='16',

#     title =' ',

#     cmap='BuGn'

#     )

# y_train=p

# y_test=q
clf = xgb.XGBClassifier(n_estimators=1000,random_state=700)

clf.fit(X_train, y_train)

acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='binary'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='binary'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(clf.predict(X_train), y_train).ravel()

specificity = tn / (tn+fp)

specificity_train=format(specificity,'.3f')



acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='binary'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='binary'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(clf.predict(X_test), y_test).ravel()

specificity = tn / (tn+fp)

specificity_test=format(specificity,'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['XGBoost',acc_train,precision_train,recall_train,f1_train,specificity_train,acc_test,precision_test,recall_test,f1_test,specificity_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test



y_train = pd.DataFrame(y_train)

y_train=y_train.replace([0,1], ["Negative","Positive"])



pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)

pred_train=pred_train.replace([0,1], ["Negative","Positive"])





pred_test=clf.predict(X_test)

y_test = pd.DataFrame(y_test)

y_test=y_test.replace([0,1], ["Negative","Positive"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1], ["Negative","Positive"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )

y_train=p

y_test=q
# clf = GradientBoostingClassifier()

# clf.fit(X_train, y_train)

# acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

# precision_train=format(precision_score(y_train, clf.predict(X_train), average='binary'),'.3f')

# recall_train=format(recall_score(y_train,clf.predict(X_train), average='binary'),'.3f')

# f1_train=format(f1_score(y_train,clf.predict(X_train), average='binary'),'.3f')

# tn, fp, fn, tp = confusion_matrix(clf.predict(X_train), y_train).ravel()

# specificity = tn / (tn+fp)

# specificity_train=format(specificity,'.3f')



# acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

# precision_test=format(precision_score(y_test, clf.predict(X_test), average='binary'),'.3f')

# recall_test=format(recall_score(y_test,clf.predict(X_test), average='binary'),'.3f')

# f1_test=format(f1_score(y_test,clf.predict(X_test), average='binary'),'.3f')

# tn, fp, fn, tp = confusion_matrix(clf.predict(X_test), y_test).ravel()

# specificity = tn / (tn+fp)

# specificity_test=format(specificity,'.3f')



# r = evaluation.shape[0]

# evaluation.loc[r] = ['GradientBoost',acc_train,precision_train,recall_train,f1_train,specificity_train,acc_test,precision_test,recall_test,f1_test,specificity_test]

# evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
# p=y_train

# q=y_test



# y_train = pd.DataFrame(y_train)

# y_train=y_train.replace([0,1], ["Negative","Positive"])



# pred_train=clf.predict(X_train)

# pred_train=pd.DataFrame(pred_train)

# pred_train=pred_train.replace([0,1], ["Negative","Positive"])





# pred_test=clf.predict(X_test)

# y_test = pd.DataFrame(y_test)

# y_test=y_test.replace([0,1], ["Negative","Positive"])

# pred_test=pd.DataFrame(pred_test)



# pred_test=pred_test.replace([0,1], ["Negative","Positive"])



# skplt.metrics.plot_confusion_matrix(

#     y_train, 

#     pred_train,

#     figsize=(7,4),

#     title_fontsize='18',

#     text_fontsize='16',

#     title =' ',

#     cmap='BuGn'

#     )



# skplt.metrics.plot_confusion_matrix(

#     y_test, 

#     pred_test,

#     figsize=(7,4),

#     title_fontsize='18',

#     text_fontsize='16',

#     title =' ',

#     cmap='BuGn'

#     )

# y_train=p

# y_test=q
demo_datagen = ImageDataGenerator(

    rotation_range=15,

    width_shift_range=0.05,

    height_shift_range=0.05,

    rescale=1./255,

    shear_range=0.05,

    brightness_range=[0.1, 1.5],

    horizontal_flip=True,

    vertical_flip=True

)



os.mkdir('preview')

x = X_train_crop[0]  

x = x.reshape((1,) + x.shape) 



i = 0

for batch in demo_datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='aug_img', save_format='jpg'):

    i += 1

    if i > 20:

        break 

        

plt.imshow(X_train_crop[0])

plt.xticks([])

plt.yticks([])

plt.title('Original Image')

plt.show()



plt.figure(figsize=(15,6))

i = 1

for img in os.listdir('preview/'):

    img = cv2.cv2.imread('preview/' + img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(3,7,i)

    plt.imshow(img)

    plt.xticks([])

    plt.yticks([])

    i += 1

    if i > 3*7:

        break

plt.suptitle('Augemented Images')

plt.show()

!rm -rf preview/
TRAIN_DIR = 'TRAIN_CROP/'

VAL_DIR = 'VAL_CROP/'



train_datagen = ImageDataGenerator(

    rotation_range=15,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=0.1,

    brightness_range=[0.5, 1.5],

    horizontal_flip=True,

    vertical_flip=True,

    preprocessing_function=preprocess_input

)



test_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input

)





train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    color_mode='rgb',

    target_size=IMG_SIZE,

    batch_size=32,

    class_mode='binary',

    seed=RANDOM_SEED

)





validation_generator = test_datagen.flow_from_directory(

    VAL_DIR,

    color_mode='rgb',

    target_size=IMG_SIZE,

    batch_size=16,

    class_mode='binary',

    seed=RANDOM_SEED

)
# load base model

vgg16_weight_path = '../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

base_model = VGG16(

    weights=vgg16_weight_path,

    include_top=False, 

    input_shape=IMG_SIZE + (3,)

)



NUM_CLASSES = 1

model = Sequential()

model.add(base_model)

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

model.layers[0].trainable = False

model.compile(

    loss='binary_crossentropy',

    optimizer=RMSprop(lr=1e-4),

    metrics=['accuracy']

)





model.summary()
X_test = np.concatenate((X_test_prep, X_val_prep))

y_test = py_test 

X_train=X_train_prep

y_train = py_train



print(X_test.shape)

print(y_test.shape)

print(X_train.shape)

print(y_train.shape)
EPOCHS = 30

es = EarlyStopping(

    monitor='val_acc', 

    mode='max',

    patience=10

)

history = model.fit_generator(

    train_generator,

    steps_per_epoch=50,

    epochs=EPOCHS,

    validation_data=validation_generator,

    validation_steps=25,

    callbacks=[es]

)









clf=model



prediction_train = clf.predict(X_train)

prediction_train = [1 if x>0.5 else 0 for x in prediction_train]



prediction_test = clf.predict(X_test)

prediction_test = [1 if x>0.5 else 0 for x in prediction_test]



acc_train=format(accuracy_score(prediction_train, y_train),'.3f')

precision_train=format(precision_score(y_train, prediction_train, average='binary'),'.3f')

recall_train=format(recall_score(y_train,prediction_train, average='binary'),'.3f')

f1_train=format(f1_score(y_train,prediction_train, average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(prediction_train, y_train).ravel()

specificity = tn / (tn+fp)

specificity_train=format(specificity,'.3f')



acc_test=format(accuracy_score(prediction_test, y_test),'.3f')

precision_test=format(precision_score(y_test, prediction_test, average='binary'),'.3f')

recall_test=format(recall_score(y_test,prediction_test, average='binary'),'.3f')

f1_test=format(f1_score(y_test,prediction_test, average='binary'),'.3f')

tn, fp, fn, tp = confusion_matrix(prediction_test, y_test).ravel()

specificity = tn / (tn+fp)

specificity_test=format(specificity,'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['Neural Network',acc_train,precision_train,recall_train,f1_train,specificity_train,acc_test,precision_test,recall_test,f1_test,specificity_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)

p=y_train

q=y_test



y_train = pd.DataFrame(y_train)

y_train=y_train.replace([0,1], ["Negative","Positive"])



pred_train=pd.DataFrame(prediction_train)

pred_train=pred_train.replace([0,1], ["Negative","Positive"])



y_test = pd.DataFrame(y_test)

y_test=y_test.replace([0,1], ["Negative","Positive"])

pred_test=pd.DataFrame(prediction_test)



pred_test=pred_test.replace([0,1], ["Negative","Positive"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )

y_train=p

y_test=q
evaluation.to_csv('eval.csv')
# clf=model

# acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

# precision_train=format(precision_score(y_train, clf.predict(X_train), average='binary'),'.3f')

# recall_train=format(recall_score(y_train,clf.predict(X_train), average='binary'),'.3f')

# f1_train=format(f1_score(y_train,clf.predict(X_train), average='binary'),'.3f')

# tn, fp, fn, tp = confusion_matrix(clf.predict(X_train), y_train).ravel()

# specificity = tn / (tn+fp)

# specificity_train=format(specificity,'.3f')



# acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

# precision_test=format(precision_score(y_test, clf.predict(X_test), average='binary'),'.3f')

# recall_test=format(recall_score(y_test,clf.predict(X_test), average='binary'),'.3f')

# f1_test=format(f1_score(y_test,clf.predict(X_test), average='binary'),'.3f')

# tn, fp, fn, tp = confusion_matrix(clf.predict(X_test), y_test).ravel()

# specificity = tn / (tn+fp)

# specificity_test=format(specificity,'.3f')



# r = evaluation.shape[0]

# evaluation.loc[r] = ['CNN',acc_train,precision_train,recall_train,f1_train,specificity_train,acc_test,precision_test,recall_test,f1_test,specificity_test]

# evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
# # plot model performance

# acc = history.history['acc']

# val_acc = history.history['val_acc']

# loss = history.history['loss']

# val_loss = history.history['val_loss']

# epochs_range = range(1, len(history.epoch) + 1)



# plt.figure(figsize=(15,5))



# plt.subplot(1, 2, 1)

# plt.plot(epochs_range, acc, label='Train Set')

# plt.plot(epochs_range, val_acc, label='Val Set')

# plt.legend(loc="best")

# plt.xlabel('Epochs')

# plt.ylabel('Accuracy')

# plt.title('Model Accuracy')



# plt.subplot(1, 2, 2)

# plt.plot(epochs_range, loss, label='Train Set')

# plt.plot(epochs_range, val_loss, label='Val Set')

# plt.legend(loc="best")

# plt.xlabel('Epochs')

# plt.ylabel('Loss')

# plt.title('Model Loss')



# plt.tight_layout()

# plt.show()



# # validate on val set

# predictions = model.predict(X_val_prep)

# predictions = [1 if x>0.5 else 0 for x in predictions]



# accuracy = accuracy_score(y_val, predictions)

# print('Val Accuracy = %.2f' % accuracy)



# confusion_mtx = confusion_matrix(y_val, predictions) 

# cm = plot_confusion_matrix(confusion_mtx, classes = list(labels.items()), normalize=False)





# # validate on test set

# predictions = model.predict(X_test_prep)

# predictions = [1 if x>0.5 else 0 for x in predictions]



# accuracy = accuracy_score(y_test, predictions)

# print('Test Accuracy = %.2f' % accuracy)



# confusion_mtx = confusion_matrix(y_test, predictions) 

# cm = plot_confusion_matrix(confusion_mtx, classes = list(labels.items()), normalize=False)





# ind_list = np.argwhere((y_test == predictions) == False)[:, -1]

# if ind_list.size == 0:

#     print('There are no missclassified images.')

# else:

#     for i in ind_list:

#         plt.figure()

#         plt.imshow(X_test_crop[i])

#         plt.xticks([])

#         plt.yticks([])

#         plt.title(f'Actual class: {y_val[i]}\nPredicted class: {predictions[i]}')

#         plt.show()