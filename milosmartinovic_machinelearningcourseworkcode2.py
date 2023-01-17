# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

from sklearn.decomposition import PCA

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import PowerTransformer

from sklearn import preprocessing

from sklearn.neural_network import MLPClassifier

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Reading Testing and Training files from csv to pd

my_training_data = pd.read_csv('../input/brighton-a-memorable-city/training.csv')

my_testing_data = pd.read_csv('../input/brighton-a-memorable-city/testing.csv')

my_testing_IDs = my_testing_data['ID']

#create array of all the predictions from the training data set

trainingPredictions = []

for i in my_training_data['prediction']:

    trainingPredictions.append(i)

#get percentage of "memorable" and "not memorable" images

All = my_training_data.shape[0]

memorable = my_training_data[my_training_data['prediction'] == 1]

notMemorable = my_training_data[my_training_data['prediction'] == 0]



x = len(memorable)/All

y = len(notMemorable)/All



print('memorable :',x*100,'%')

print('not memorable :',y*100,'%')
#create an array with the names of all the features

gist = ["GIST"]

for i in range(1,512):

    gist.append("GIST." + str(i))

gist





cnns = ["CNNs"]

for j in range(1,4096):

    cnns.append("CNNs." + str(j))

cnns

gist

allFeatures = cnns+gist
#function that standardises the input data

def standardise(data):

    standard = data.loc[:,allFeatures].values

    standard = StandardScaler().fit_transform(standard)

    return standard
#function that scales the input data sets

def scale(data_train,data_test):

    min_max_scaler = preprocessing.MinMaxScaler()

    

    scale_train = data_train.loc[:,allFeatures].values

    scale_train =min_max_scaler.fit_transform(scale_train)

    

    scale_test = data_test.loc[:,allFeatures].values

    scale_test = min_max_scaler.transform(scale_test)

    return scale_train, scale_test
#removes unnecessary columns in the data sets

my_training_data = my_training_data.drop('ID',axis=1)

my_training_data = my_training_data.drop('prediction',axis=1)

my_testing_data = my_testing_data.drop('ID',axis=1)

#splits the data into validation, training sets (20%/80% split)

X_train,X_test,y_train,y_test = train_test_split(my_training_data,trainingPredictions,test_size=0.2,random_state = 42)
#standardises and power transforms the data

from sklearn.preprocessing import PowerTransformer

pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)

pt.fit(X_train)

st_train = pt.transform(X_train)

st_test = pt.transform(X_test)







from sklearn.neural_network import MLPClassifier

%matplotlib inline

import matplotlib.pyplot as plt

s = ['sgd','adam','lbfgs']

score=[]

for i in s:

    clf = MLPClassifier(solver=i,activation= 'relu',learning_rate='adaptive',

                    hidden_layer_sizes=(5), random_state=1, max_iter=10000)

    clf.fit(st_train,  y_train)

    score.append(clf.score(st_test, y_test))





plt.bar(s, score)

plt.xlabel('solver used')

plt.ylabel('score')

plt.title('Testing effect of Different Solvers on Classifier')
from sklearn.neural_network import MLPClassifier

%matplotlib inline

import matplotlib.pyplot as plt

s1 = [(5),(5,5),(5,5,5)]

s = [1,2,4,8,16,32,64,128]



score=[]

for i in s:

    clf = MLPClassifier(solver='sgd',activation= 'relu',learning_rate='adaptive',

                    hidden_layer_sizes=i, random_state=1, max_iter=10000)

    clf.fit(st_train,  y_train)

    score.append(clf.score(st_test, y_test))

    

plt.plot(s, score)

plt.xlabel('number of neurons')

plt.ylabel('score')

plt.title('Testing effect of Different neuron numbers on Classifier')
from sklearn.neural_network import MLPClassifier

%matplotlib inline

import matplotlib.pyplot as plt

s1 = [(5),(5,5),(5,5,5),(5,5,5,5),(5,5,5,5,5)]



score=[]

for i in s1:

    clf = MLPClassifier(solver='sgd',activation= 'relu',learning_rate='adaptive',

                    hidden_layer_sizes=i, random_state=1, max_iter=10000)

    clf.fit(st_train,  y_train)

    score.append(clf.score(st_test, y_test))

    

score

s = ["1 Layer","2 Layers","3 Layers","4 Layers","5 Layers"]

plt.bar(s, score)

plt.xlabel('number of hidden layers')

plt.ylabel('score')

plt.title('Testing effect of number of hidden neuron layers on Classifier')
from sklearn.neural_network import MLPClassifier

%matplotlib inline

import matplotlib.pyplot as plt

s = ['identity','logistic','tanh','relu']



score=[]

for i in s:

    clf = MLPClassifier(solver='sgd',activation= i,learning_rate='adaptive',

                    hidden_layer_sizes=(5), random_state=1, max_iter=10000)

    clf.fit(st_train,  y_train)

    score.append(clf.score(st_test, y_test))

    



plt.bar(s, score)

plt.xlabel('activation functions')

plt.ylabel('score')

plt.title('Testing effect different activation functions have on classifier')
from sklearn.neural_network import MLPClassifier

#Preprocessing the data - Power Transforming

theTransformer = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)

theTransformer.fit(my_training_data)

t_train = theTransformer.transform(my_training_data)

t_actual_test = theTransformer.transform(my_testing_data)
#Create Classifier with final hyperparameters

theClassifier = MLPClassifier(solver='sgd',activation= 'tanh',learning_rate='adaptive',

                    hidden_layer_sizes=(5), random_state=1, max_iter=1000)
#train the classifier with the training data 

theClassifier.fit(t_train,  trainingPredictions)

#Predict the classes of the London testing data and save the results to a csv file.

aqs=theClassifier.predict(t_actual_test)



dfx = pd.DataFrame(data = aqs

             , columns = ['prediction'], )

dfx = pd.concat([my_testing_IDs,dfx], axis = 1)

print(dfx)

dfx.to_csv('submission.csv', index = False)
sns.countplot(dfx['prediction'])
#Old code

pca = PCA(0.80)

standard = standardise(my_training_data)

principalComponents= pca.fit_transform(standard)

s = standardise(my_testing_data)

s = pca.transform(s)



principalDf = pd.DataFrame(data = principalComponents

             )

#print(principalDf)



from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression(solver='liblinear', class_weight = {1: 0.3848,0:0.6152})

logisticRegr.fit(principalComponents,trainingPredictions)





x= logisticRegr.predict(s)

dfx = pd.DataFrame(data = x

             , columns = ['prediction'], )

dfx = pd.concat([my_testing_IDs,dfx], axis = 1)

print(dfx)

dfx.to_csv('submission.csv', index = False)
#Old code

principalComponents

trainingPredictions
#Old code

pca = PCA(n_components=1)

principalComponentsCnns= pca.fit_transform(standard_cnns )

principalComponentsGist= pca.fit_transform(standard_gist)

principalDfCnns = pd.DataFrame(data = principalComponentsCnns

             , columns = ['principal component 2'])

principalDfGist = pd.DataFrame(data = principalComponentsGist

             , columns = ['principal component 1'])

principalDf = pd.concat([principalDfGist, principalDfCnns], axis = 1)

print(principalDf)