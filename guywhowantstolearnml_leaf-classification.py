import numpy as np

import pandas as pd

from scipy import ndimage

import matplotlib.pyplot as plt

from os import listdir

%matplotlib inline

from sklearn.linear_model import LogisticRegression

from sklearn.grid_search import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from PIL import Image
path = '../input/images/'

train = pd.read_csv('../input/train.csv')

image_paths = [path + f for f in listdir(path)]
def plotGalery(classes, n_col=10, scale_x = 1.5, scale_y = 1.7):

    

    def pathsBySpecies(classes):

        paths = {}

        for row in train.values:

            if row[1] in classes:

                if row[1] in paths:

                    paths[row[1]].append('../input/images/' + str(row[0]) + '.jpg')

                else:

                    paths[row[1]] = ['../input/images/' + str(row[0]) + '.jpg']

        return paths

    

    dic = pathsBySpecies(classes)

    

    n_row = len(dic.keys())

    plt.figure(figsize=(scale_x * n_col, scale_y * n_row))

    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

    for i in range(n_row * n_col):

        key = list(dic.keys())[i // n_col]

        path = dic[key][i % n_col]

        image = Image.open(path)

        image = np.array(image)

        plt.subplot(n_row, n_col, i + 1)

        plt.imshow(image, cmap=plt.cm.gray, interpolation='none')

        plt.xticks(())

        plt.yticks(())

        plt.tight_layout

        

def imageById(id_):

    img = Image.open('../input/images/'+str(id_)+'.jpg')

    #img = img.resize((50, 50), Image.ANTIALIAS)

    return np.array(img)

'''

def imageByIdPCA(id_):

    img = Image.open('../input/images/'+str(id_)+'.jpg')

    img = img.resize((50, 50), Image.ANTIALIAS)

    from sklearn.decomposition import PCA

    # Make an instance of the Model

    pca = PCA(35)

    pcm = pca.fit_transform(img)

    return pcm

'''
classes = train.species.value_counts().keys()

plotGalery(classes)
import cv2



def imageFeatures(source, filepath):

    df = pd.read_csv(source)

    ids = df.values[:,0].astype(np.int)

    images = [imageById(id_) for id_ in ids]

    height = [image.shape[0] for image in images]

    width = [image.shape[1] for image in images]

    orientation = [int(h > w) for h, w in zip(height, width)]

    perimeters = [cv2.Canny(im,100,200).sum() / 255.0 for im in images]

    square = [image.sum() / 255.0/ image.size for image in images]

    square_r = [im.sum() / sq for im, sq in zip(images, square)]

    sums = [im.sum() for im in images]

    pd.DataFrame({

            'height': height,

            'width': width,

            'orientation': orientation,

            'square': square,

            'square_r': square_r,

            'sum': sums

        }).to_csv(filepath, index=False)

#Principal Component Analysis

import matplotlib

df = pd.read_csv('../input/train.csv')

df2 = pd.read_csv('../input/test.csv')

ids = df.values[:,0].astype(np.int)

testids = df2.values[:,0].astype(np.int)

pca_train = []

pca_test = []

for id_ in ids:

    img=Image.open('../input/images/'+str(id_)+'.jpg')

    img = img.resize((50, 50), Image.ANTIALIAS)

    img= np.array(img)

    pca_train.append(img)

for id_ in testids:

    img=Image.open('../input/images/'+str(id_)+'.jpg')

    img = img.resize((50, 50), Image.ANTIALIAS)

    img= np.array(img)

    pca_test.append(img)

pca_train=np.array(pca_train)

pca_test=np.array(pca_test)

pca_train = pca_train.reshape(990,2500)

pca_test = pca_test.reshape(594,2500)

from sklearn.decomposition import PCA

# Make an instance of the Model

pca = PCA(35)

pca.fit(pca_train)

pca_train = pca.transform(pca_train)

pca_test = pca.transform(pca_test)

pca_train=pd.DataFrame(pca_train)

pca_test = pd.DataFrame(pca_test)
imageFeatures('../input/train.csv', 'train_f1.csv')

imageFeatures('../input/test.csv', 'test_f1.csv')

#pcaImage('../input/train.csv','train_pca.csv')

#pcaImage('../input/test.csv','test_pca.csv')
def M(im):

    ret,thresh = cv2.threshold(im,127,255,0)

    contours,hierarchy, _ = cv2.findContours(thresh, 1, 2)

    cnt = contours[0]

    x = np.fromiter(iter(cv2.moments(cnt).values()), dtype=float)

    return x



def imageMoments(source, filepath):

    df = pd.read_csv(source)

    ids = df.values[:,0].astype(np.int)

    images = [imageById(id_) for id_ in ids]

        

    moments = [1.0*M(im)/im.size for im in images]

    pd.DataFrame(moments).to_csv(filepath, index=False)
imageMoments('../input/train.csv', 'train_M1.csv')

imageMoments('../input/test.csv', 'test_M1.csv')
train_f1 = pd.read_csv('train_f1.csv')

train_M1 = pd.read_csv('train_M1.csv')

test_f1 = pd.read_csv('test_f1.csv')

test_M1 = pd.read_csv('test_M1.csv')



test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")



# train

x_train = train.drop(['id', 'species'], axis=1)

y_train = train['species']

le = LabelEncoder()

y_train = le.fit_transform(y_train)

test_ids = test.pop('id')

x_test = test

x_train = pd.concat([train_f1,x_train,pca_train,train_M1], axis=1)

x_test = pd.concat([test_f1,x_test,pca_test,test_M1], axis =1)
scaler = StandardScaler().fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)
params = {'C':[1000], 'tol': [0.0008, 0.0007]}

log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial')

clf = GridSearchCV(log_reg, params, scoring='log_loss', refit='False', cv=3)

clf.fit(x_train, y_train)



y_pred = clf.predict_proba(x_test)



submission = pd.DataFrame(y_pred, index=test_ids, columns=le.classes_)

#submission.to_csv('test.csv')
colsub=[]

for i in submission.columns:

    colsub.append(i)

finalcol=[]

from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold()

selector.fit_transform(submission)     

j=0

for i in selector.variances_:

    if(i>0.009999):

        finalcol.append(colsub[j])

    j=j+1

#finalcol is the nparray of columns that don't perform well using LR
train_f1 = pd.read_csv('train_f1.csv')

train_M1 = pd.read_csv('train_M1.csv')

test_f1 = pd.read_csv('test_f1.csv')

test_M1 = pd.read_csv('test_M1.csv')



test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")

#Training df

x_train = train

x_rf_train = train



le = LabelEncoder()

y_train = train['species']

y_rf_train =train['species']

#labels

test_ids = test.pop('id')

x_test = test

#merging extracted features with given features

x_train = pd.concat([train_f1,x_train,pca_train,train_M1], axis=1)

x_rf_train = pd.concat([train_f1,x_rf_train,pca_train,train_M1], axis=1)

for index, row in x_train.iterrows():

    if(row['species'] in finalcol):

        x_train = x_train.drop(index)

        y_train = y_train.drop(index)

        print(index)

    else:

        x_rf_train = x_rf_train.drop(index)

        y_rf_train = y_rf_train.drop(index)



y_train = le.fit_transform(y_train)

x_train = x_train.drop(['id', 'species'], axis=1)

x_rf_train =x_rf_train.drop(['id', 'species'], axis=1)

x_test = pd.concat([test_f1,x_test,pca_test,test_M1], axis =1)

lerf=LabelEncoder()

y_rf_train =lerf.fit_transform(y_rf_train)
#Scaling for Logistic Regression

scaler = StandardScaler().fit(x_rf_train)

x_rf_train = scaler.transform(x_rf_train)

x_rf_test = scaler.transform(x_test)

#scaling for Random Forest

scaler = StandardScaler().fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification



clf = RandomForestClassifier()

clf.fit(x_rf_train, y_rf_train)

y_predrf = clf.predict_proba(x_rf_test)

submissionrf = pd.DataFrame(y_predrf, index=test_ids, columns=lerf.classes_)

params = {'C':[1000], 'tol': [0.0008, 0.0007]}

log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial')

clf = GridSearchCV(log_reg, params, scoring='log_loss', refit='False', cv=3)

clf.fit(x_train, y_train)



y_pred = clf.predict_proba(x_test)



submission = pd.DataFrame(y_pred, index=test_ids, columns=le.classes_)
submission = pd.concat([submission,submissionrf], axis =1)

submission.to_csv('final.csv')
for i in result.index:

    if(result[prediction[0][i]][i]>0.1):

        print(prediction[0][i])



train_f1 = pd.read_csv('train_f1.csv')

#train_M1 = pd.read_csv('train_M1.csv')

test_f1 = pd.read_csv('test_f1.csv')

#test_M1 = pd.read_csv('test_M1.csv')

test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")

train=pd.concat([train_f1,train,pca_train], axis=1)

test = pd.concat([test_f1,test,pca_test], axis =1)
from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedShuffleSplit
def encode(train, test):

    le = LabelEncoder().fit(train.species) 

    labels = le.transform(train.species)           # encode species strings

    classes = list(le.classes_)                    # save column names for submission

    test_ids = test.id                             # save test ids for submission

    

    train = train.drop(['species', 'id'], axis=1)  

    test = test.drop(['id'], axis=1)

    

    return train, labels, test, test_ids, classes



train, labels, test, test_ids, classes = encode(train, test)

train.head(1)
sss = StratifiedShuffleSplit(labels, 10, test_size=0.2, random_state=23)



for train_index, test_index in sss:

    X_train, X_test = train.values[train_index], train.values[test_index]

    y_train, y_test = labels[train_index], labels[test_index]
y_train.shape
from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    NuSVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis()]



# Logging for Visual Comparison

log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)



for clf in classifiers:

    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    print("Accuracy: {:.4%}".format(acc))

    

    train_predictions = clf.predict_proba(X_test)

    ll = log_loss(y_test, train_predictions)

    print("Log Loss: {}".format(ll))

    

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)
# Predict Test Set

favorite_clf =  LinearDiscriminantAnalysis(solver='lbfgs', multi_class='multinomial')

favorite_clf.fit(X_train, y_train)

y_pred = favorite_clf.predict_proba(test)



# Format DataFrame

#submission = pd.DataFrame(test_predictions, columns=classes)

#submission.insert(0, 'id', test_ids)

#ubmission.reset_index()



# Export Submission

#submission.to_csv('submissiondiff.csv', index = False)

#submission.tail()