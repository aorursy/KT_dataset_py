# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import VotingClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import cross_val_score



import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")



#Renaming variables to understand in easier manner

data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',

       'exercise_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']
data.head()
data.info()
data.shape
a1=data.target.value_counts()

print(a1)
a1.plot(kind="bar", color=["red", "blue"])
data.isna().sum()
#Correlation Map

x = data.corr()

pd.DataFrame(x['target']).sort_values(by='target',ascending = False).style.background_gradient(cmap = 'copper')
#Scaling data and encoding data for clearer picture and better correlation map



data.chest_pain_type = data.chest_pain_type.map({1:'angina pectoris', 2:'atypical angina', 3:'non-anginal pain', 4:'SMI', 0:'absent'})



data.st_slope = data.st_slope.map({1:'upsloping', 2:'horizontal', 3:'downsloping', 0:'absent'})



data.thalassemia = data.thalassemia.map({1:'normal', 2:'fixed defect', 3:'reversable defect', 0:'absent'})

data.head()
X = data.iloc[:, 0:13]



Y = data.iloc[:, -1]
X.head()
categorical_columns = ['chest_pain_type', 'thalassemia', 'st_slope']



for column in categorical_columns:

    dummies = pd.get_dummies(X[column], drop_first = True)

    X[dummies.columns] = dummies

    X.drop(column, axis =1, inplace = True)

    

    
X.head()


a2 = X.copy()

a2['target'] = Y



d = a2.corr()

pd.DataFrame(d['target']).sort_values(by='target',ascending = False).style.background_gradient(cmap = 'copper')
corr_matrix = data.corr()

fig, ax = plt.subplots(figsize=(15, 15))

ax = sns.heatmap(corr_matrix,

                 annot=True,

                 linewidths=0.5,

                 fmt=".2f",

                 cmap="YlGnBu");

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)
data.drop('target', axis=1).corrwith(data.target).plot(kind='bar', grid=True, figsize=(12, 8), 

                                                   title="Correlation with target")
#Splitting data into train and test data

X = a2.drop('target', axis=1)

y = a2.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)



print("X-Train:",X_train.shape)

print("X-Test:",X_test.shape)

print("Y-Train:",y_train.shape)

print("Y-Test:",y_test.shape)
#Scaling the data that isnt categorical data

num_columns =  ['resting_blood_pressure','serum_cholesterol', 'age', 'max_heart_rate', 'st_depression']



scaler = StandardScaler()



scaler.fit(X_train[num_columns])



X_train[num_columns] = scaler.transform(X_train[num_columns])



X_test[num_columns] = scaler.transform(X_test[num_columns])

X_train.head()
def print_score(clf, X_train, y_train, X_test, y_test, train=True):

    if train:

        pred = clf.predict(X_train)

        print("Train Result:\n==============================================")

        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")

        print("_____________________________________________")

        print("Classification Report:", end='')

        print(f"\tPrecision Score: {precision_score(y_train, pred) * 100:.2f}%")

        print(f"\t\t\tRecall Score: {recall_score(y_train, pred) * 100:.2f}%")

        print(f"\t\t\tF1 score: {f1_score(y_train, pred) * 100:.2f}%")

        print("_____________________________________________")

        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n\n")

        

    elif train==False:

        pred = clf.predict(X_test)

        print("Test Result:\n==============================================")        

        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")

        print("_____________________________________________")

        print("Classification Report:", end='')

        print(f"\tPrecision Score: {precision_score(y_test, pred) * 100:.2f}%")

        print(f"\t\t\tRecall Score: {recall_score(y_test, pred) * 100:.2f}%")

        print(f"\t\t\tF1 score: {f1_score(y_test, pred) * 100:.2f}%")

        print("_____________________________________________")

        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
from sklearn.svm import SVC





svm_model = SVC(kernel='rbf', gamma=0.1, C=1.0)

svm_model.fit(X_train, y_train)
print_score(svm_model, X_train, y_train, X_test, y_test, train=True)

print_score(svm_model, X_train, y_train, X_test, y_test, train=False)
from sklearn.tree import DecisionTreeClassifier





tree = DecisionTreeClassifier(random_state=42)

tree.fit(X_train, y_train)



print_score(tree, X_train, y_train, X_test, y_test, train=True)

print_score(tree, X_train, y_train, X_test, y_test, train=False)
# creating a list of our models

ensembles = [svm_model,tree]



# Train each of the model

for estimator in ensembles:

    print("Training the", estimator)

    estimator.fit(X_train,y_train)
# Find the scores of each estimator

scores = [estimator.score(X_test, y_test) for estimator in ensembles]



scores
#defining estimators in a list

named_estimators = [

    ('svm',svm_model),

    ('dt', tree),

]
# Creatinginstance for our Voting classifier



voting_clf = VotingClassifier(named_estimators)
# Fit the classifier



voting_clf.fit(X_train,y_train)
#accuracy

acc = voting_clf.score(X_test,y_test)



print(f"The accuracy on test set using voting classifier is {np.round(acc, 4)*100}%")