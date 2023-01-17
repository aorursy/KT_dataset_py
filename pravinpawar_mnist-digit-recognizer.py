import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

pd.set_option('display.max_column', 100)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
trainDf = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

trainDf.head()
trainDf.shape
trainDf.describe()
trainDf.isnull().sum()[trainDf.isnull().sum() > 0]
trainDf[trainDf.duplicated()]
trainDf.info()
plt.figure(figsize=(20,25))

imgCount = 1

for d in range(0,10):

    for i, row in trainDf[trainDf['label'] == d].sample(10).iterrows():

        img = np.array(row)[1:]

        label = row[0]



        plottable_image = np.reshape(img, (28, 28))



        # Plot the image

        plt.subplot(10,10, imgCount)

        plt.imshow(plottable_image, cmap='gray_r')

        plt.title('Digit: {}'.format(label))

        imgCount = imgCount + 1



plt.show()
sns.countplot(data=trainDf, x='label')

plt.show()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
Y = trainDf["label"]

X = trainDf.drop(columns=['label'], axis=1)
#Split data in Train & Test set

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.4, stratify=Y)
# Check train & test data is balanced or not

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)

sns.countplot(x=Y_train)

plt.title('Train Data')



plt.subplot(1,2,2)

plt.title('Test data')

sns.countplot(x=Y_test)



plt.show()
#Scale data using StandardScaler

sc = StandardScaler()

scalledData = sc.fit_transform(X_train)

trainDf_scalled = pd.DataFrame(data=scalledData, columns=X_train.columns)

trainDf_scalled.head()
testDf_scalled = pd.DataFrame(data=sc.transform(X_test), columns=X_test.columns)

testDf_scalled.head()
pca = PCA(svd_solver='randomized', random_state=100)

pca.fit(trainDf_scalled)
pca.components_.shape
#Plot cumulative sum of variance explained by princepal component

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.yticks(np.arange(0,1.1,0.1))

plt.grid( linestyle='-', linewidth=0.5)

plt.show()
np.cumsum(pca.explained_variance_ratio_)[[100,200,300,400,500,600,700]]
from sklearn.decomposition import IncrementalPCA

final_pca = IncrementalPCA(n_components=300)

trainDf_pca = final_pca.fit_transform(trainDf_scalled)

testDf_pca = final_pca.transform(testDf_scalled)
trainDf_pca.shape
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

param = {'C' : [0.01, 0.1, 1, 10]}

svc = SVC(kernel='linear')

model_scv_li = GridSearchCV(svc,

                             param_grid=param, 

                             scoring='accuracy', 

                             cv=folds, 

                             verbose=1,

                             n_jobs = -1,

                             return_train_score=True)

model_scv_li.fit(trainDf_pca, Y_train)
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

param = {'C' : [0.01, 0.1, 1],

        'gamma': [0.001,0.01]}

svc = SVC(kernel = 'poly')

model_scv_poly = GridSearchCV(svc,

                             param_grid=param, 

                             scoring='accuracy', 

                             cv=folds, 

                             verbose=1,

                             n_jobs = -1,

                             return_train_score=True)

model_scv_poly.fit(trainDf_pca, Y_train)
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

param = {'C' : [1, 10, 100],

        'gamma': [0.1, 0.01, 0.001]}

svc = SVC(kernel = 'rbf')

model_scv_rbf = GridSearchCV(svc,

                             param_grid=param, 

                             scoring='accuracy', 

                             cv=folds, 

                             verbose=1,

                             n_jobs = -1,

                             return_train_score=True)

model_scv_rbf.fit(trainDf_pca, Y_train)
liDF = pd.DataFrame(data = model_scv_li.cv_results_)

polyDF = pd.DataFrame(data = model_scv_poly.cv_results_)

rbfDF = pd.DataFrame(data = model_scv_rbf.cv_results_)
#Plot Train and Test score for different parameter value of C

#Kernal = 'linear'

plt.plot(liDF['param_C'], liDF['mean_train_score'], 'g')

plt.plot(liDF['param_C'], liDF['mean_test_score'], 'r')

plt.legend(['Train Accuracy', 'Test Accuracy'])

plt.grid()

plt.show()
#Kernal = 'Poly'

plt.figure(figsize=(12,8))

for i,c in enumerate(np.unique(polyDF['param_C'])) :

    tempDF = polyDF[polyDF['param_C'] == c]

    plt.subplot(2,2, i+1)

    plt.plot(tempDF['param_gamma'], tempDF['mean_train_score'], 'g')

    plt.plot(tempDF['param_gamma'], tempDF['mean_test_score'], 'r')

    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1])

    plt.title("C = {}".format(c))

    plt.legend(['Train Accuracy', 'Test Accuracy'])

    plt.grid()

plt.show()
#Kernal = 'RBF'

plt.figure(figsize=(12,8))

for i,c in enumerate(np.unique(rbfDF['param_C'])) :

    tempDF = rbfDF[rbfDF['param_C'] == c]

    plt.subplot(2,2, i+1)

    plt.plot(tempDF['param_gamma'], tempDF['mean_train_score'], 'g')

    plt.plot(tempDF['param_gamma'], tempDF['mean_test_score'], 'r')

    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1])

    plt.title("C = {}".format(c))

    plt.legend(['Train Accuracy', 'Test Accuracy'])

    plt.grid()

plt.show()
print("Linear Model best Score : {} for Parameter : {}".format(model_scv_li.best_score_, model_scv_li.best_params_))

print("Non Linear Model (Kernal = Poly) best Score : {} for Parameter : {}".format(model_scv_poly.best_score_, model_scv_poly.best_params_))

print("Non Linear Model (Kernal = RBF) best Score : {} for Parameter : {}".format(model_scv_rbf.best_score_, model_scv_rbf.best_params_))
li_model = SVC(kernel='linear', C = 0.01)

li_model.fit(trainDf_pca, Y_train)
poly_model = SVC(kernel='poly', C = 1, gamma = 0.01)

poly_model.fit(trainDf_pca, Y_train)
rbf_model = SVC(kernel='rbf', C = 10, gamma = 0.001)

rbf_model.fit(trainDf_pca, Y_train)
y_train_pred = li_model.predict(trainDf_pca)

y_test_pred = li_model.predict(testDf_pca)

print("Linear : Train Accuracy : {}".format(metrics.accuracy_score(Y_train, y_train_pred)))

print("Linear : Test Accuracy : {}".format(metrics.accuracy_score(Y_test, y_test_pred)))
y_train_pred = poly_model.predict(trainDf_pca)

y_test_pred = poly_model.predict(testDf_pca)

print("Poly : Train Accuracy : {}".format(metrics.accuracy_score(Y_train, y_train_pred)))

print("Poly : Test Accuracy : {}".format(metrics.accuracy_score(Y_test, y_test_pred)))
y_train_pred = rbf_model.predict(trainDf_pca)

y_test_pred = rbf_model.predict(testDf_pca)

print("RBF : Train Accuracy : {}".format(metrics.accuracy_score(Y_train, y_train_pred)))

print("RBF : Test Accuracy : {}".format(metrics.accuracy_score(Y_test, y_test_pred)))
testDf = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

testDf_scalled = pd.DataFrame(data=sc.transform(testDf), columns=testDf.columns)

testDf_pca = final_pca.transform(testDf_scalled)
y_test_poly_pred = poly_model.predict(testDf_pca)

y_test_rbf_pred = rbf_model.predict(testDf_pca)
subDF = pd.DataFrame(data={'ImageId' : range (1, y_test_poly_pred.size + 1), 'Label' : y_test_poly_pred})

subDF.to_csv("submission_mist_poly.csv", index=False)



subDF = pd.DataFrame(data={'ImageId' : range (1, y_test_rbf_pred.size + 1), 'Label' : y_test_rbf_pred})

subDF.to_csv("submission_mist_rbf.csv", index=False)