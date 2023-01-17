# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

ddata = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

ddata.head()
ddata.describe()
import seaborn as sns

#correlation

corr = ddata.corr()

print(corr)

sns.heatmap(corr,

           xticklabels = corr.columns,

           yticklabels=corr.columns)
%matplotlib inline

count0 = (ddata.Outcome =='0').count()

count1 = (ddata.Outcome =='1').count()

ddata.groupby('Outcome').hist(figsize=(9, 9))
print(ddata[ddata.BloodPressure == 0].groupby('Outcome')['Age'].count())
print(ddata[ddata.Glucose == 0].groupby('Outcome')['Age'].count())
print(ddata[ddata.SkinThickness == 0].groupby('Outcome')['Age'].count())
print(ddata[ddata.BMI == 0].groupby('Outcome')['Age'].count())
print(ddata[ddata.Insulin == 0].groupby('Outcome')['Age'].count())
#removing rows with zero values

new_data = ddata[(ddata.BloodPressure != 0) & (ddata.BMI != 0) & (ddata.Glucose!=0)]
new_data.describe()
new_data.groupby('Outcome').hist(figsize=(9, 9))
all_features = new_data[['Age', 'BMI',

                             'BloodPressure', 'DiabetesPedigreeFunction','Glucose','Insulin','SkinThickness']].values





feature_names =['Age', 'BMI','BloodPressure', 'DiabetesPedigreeFunction','Glucose','Insulin','SkinThickness']



classes = new_data['Outcome']

all_features

#Normalizing the data

from sklearn import preprocessing



scaler = preprocessing.StandardScaler()



all_features_scaled = scaler.fit_transform(all_features)

all_features_scaled

X = all_features_scaled

y = classes
#Using Decision Tree for Taining and Testing

import numpy

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 47, test_size = 0.25)
from sklearn.tree import DecisionTreeClassifier



clf= DecisionTreeClassifier(random_state=1)



# Train the classifier on the training set

clf.fit(X_train, y_train)

clf.score(X_test,y_test)
#Using k-fold cross Validation

from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(random_state=1)

cv_scores = cross_val_score(clf, all_features_scaled, classes, cv=10)



cv_scores.mean()
#Using RandomForestClassifier



from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(n_estimators=1000, random_state=42)

#cv_scores = cross_val_score(clf, all_features_scaled, classes, cv=10)

clf.fit(X_train,y_train)

predictions = clf.predict(X_test)

from sklearn import metrics

metrics.accuracy_score(y_test, predictions)
#Using svm.SVC with a linear Karnel

from sklearn import svm



C = 1.0

svc = svm.SVC(kernel='linear', C=C)

cv_scores = cross_val_score(svc, all_features_scaled, classes, cv=10)



cv_scores.mean()
#trying rbf kernel

C = 1.0

svc = svm.SVC(kernel='rbf', C=C)

cv_scores = cross_val_score(svc, all_features_scaled,classes, cv=10)

cv_scores.mean()
#using sigmoid

C = 1.0

svc = svm.SVC(kernel='sigmoid', C=C)

cv_scores = cross_val_score(svc, all_features_scaled, classes, cv=10)

cv_scores.mean()
C = 1.0

svc = svm.SVC(kernel='poly', C=C)

cv_scores = cross_val_score(svc, all_features_scaled, classes, cv=10)

cv_scores.mean()
#Using K-Nearest-Neighbors

from sklearn import neighbors



clf = neighbors.KNeighborsClassifier(n_neighbors=10)

cv_scores = cross_val_score(clf, all_features_scaled, classes, cv=10)



cv_scores.mean()
#Trying different k values

for n in range(1, 50):

    clf = neighbors.KNeighborsClassifier(n_neighbors=n)

    cv_scores = cross_val_score(clf, all_features_scaled, classes, cv=10)

    print (n, cv_scores.mean())
#Using Naive Bayes

from sklearn.naive_bayes import MultinomialNB



scaler = preprocessing.MinMaxScaler()

minmax = scaler.fit_transform(all_features)



clf = MultinomialNB()

cv_scores = cross_val_score(clf, minmax, classes, cv=10)



cv_scores.mean()
#Using logistic Regression

from sklearn.linear_model import LogisticRegression



clf = LogisticRegression()

cv_scores = cross_val_score(clf, all_features_scaled, classes, cv=10)

cv_scores.mean()
#Building neural networks and see how artificial neural network can do

import tensorflow as tf

from tensorflow.keras.backend import set_session

config = tf.ConfigProto()

config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

sess = tf.Session(config=config)

set_session(sess)  
from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Sequential



def create_model():

    model = Sequential()

    #4 feature inputs going into an 6-unit layer (more does not seem to help - in fact you can go down to 4)

    model.add(Dense(6, input_dim=7, kernel_initializer='normal', activation='relu'))

    # "Deep learning" turns out to be unnecessary - this additional hidden layer doesn't help either.

    #model.add(Dense(4, kernel_initializer='normal', activation='relu'))

    # Output layer with a binary classification (benign or malignant)

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    # Compile model; adam seemed to work best

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
Name=['Decision Tree','K-fold','Random Forest','SVM(Linear)','K-nearest','Naive Bayes','Logistic']

Accuracy = [ 0.6464,0.68,0.775,0.775,0.771,0.65,0.76]

lista = list(zip(Name,Accuracy))

df = pd.DataFrame(lista,columns=['Name','Accuracy'])

df
from matplotlib import pyplot as plt

import numpy as np

plt.figure(figsize=(12,12))

y_pos = np.arange(len(Name))

plt.bar(y_pos,Accuracy)

plt.xticks(y_pos,Name)

plt.tick_params(axis="x", labelsize=12)

plt.show()