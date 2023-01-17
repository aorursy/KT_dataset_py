# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt



from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split

from pylab import plot, show, subplot, specgram, imshow, savefig

from sklearn import preprocessing

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import Normalizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score,f1_score, confusion_matrix, accuracy_score, classification_report

import pickle

%matplotlib inline
FILEPATH = '../input/loan-data-set/loan_data_set.csv'
df = pd.read_csv(FILEPATH)

df.sample(8)
df.drop(['Loan_ID', 'Gender', 'Dependents', 'Married', 'Property_Area', 'Education'], axis=1, inplace=True) #Using a small set of cols here

df.head()
df.isnull().sum().any()
df.isnull().sum()
df.Credit_History.value_counts()
df.Credit_History.fillna(0.0, inplace=True)

df.Loan_Amount_Term.fillna(df.Loan_Amount_Term.mean(), inplace=True)
df.LoanAmount.fillna(df.LoanAmount.mean(), inplace=True)

df.Self_Employed.fillna('No', inplace=True)

#df.Married.fillna('No', inplace=True)
df.isnull().sum().any()
df['Credit_History'] = df['Credit_History'].astype('int64')

df.dtypes
le = LabelEncoder()

cols = df.columns.tolist()

for column in cols:

    if df[column].dtype == 'object':

        df[column] = le.fit_transform(df[column])
df.dtypes
fig, ax = plt.subplots(figsize=(20, 15))

sns.heatmap(data=df.corr().round(2), annot=True, linewidths=0.7, cmap='Blues')

plt.show()
X = df.drop("Loan_Status", axis=1)

y = df["Loan_Status"]
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=30) 
classifiers = {

    'Multinomial NB': MultinomialNB(),

    'Gaussian NB': GaussianNB(),

    'Linear SVM': SVC(kernel='linear'),

    'RBF SVM': SVC(kernel='rbf'),

    'Sigmoid SVM': SVC(kernel='sigmoid'),

    #FOR SVM USE HYPERPARAMETER TUNING TO BETTER UNDERSTAND WHAT TO TAKE

    'MLP Classifier': MLPClassifier(),

    'MLP Hidden Layer': MLPClassifier(hidden_layer_sizes=[100,100]),

    'Ada Boost': AdaBoostClassifier(),

    'Decision Tree': DecisionTreeClassifier(),

    'Random Forest': RandomForestClassifier(),

    'Gradient Boosting': GradientBoostingClassifier(),

    'Logistic Regression': LogisticRegression()

}
acc_scores = dict()



for classifier in classifiers:

    

    clf = classifiers[classifier]

    clf.fit(Xtrain,ytrain)

    y_pred = clf.predict(Xtest)

    acc_scores[classifier] = accuracy_score(y_pred, ytest)

    print(classifier, acc_scores[classifier])
model = LogisticRegression()

model.fit(Xtrain, ytrain)



y_pred = model.predict(Xtest)

print("Accuracy: %s%%" % (100*accuracy_score(y_pred, ytest)))

print(confusion_matrix(ytest, y_pred))

print("Classification Report for Logistic Regression")

print(classification_report(ytest, y_pred))
filename = 'logistic_model.p'

pickle.dump(model, open('./'+filename, 'wb'))
filename = 'rand_forest_model.p'



model = RandomForestClassifier()

model.fit(Xtrain, ytrain)



y_pred = model.predict(Xtest)

print("Accuracy: %s%%" % (100*accuracy_score(y_pred, ytest)))

print(confusion_matrix(ytest, y_pred))

print("Classification Report for Random Forest Classifier")

print(classification_report(ytest, y_pred))



pickle.dump(model, open('./'+filename, 'wb'))
filename = 'dec_tree_model.p'



model = DecisionTreeClassifier()

model.fit(Xtrain, ytrain)



y_pred = model.predict(Xtest)

print("Accuracy: %s%%" % (100*accuracy_score(y_pred, ytest)))

print(confusion_matrix(ytest, y_pred))

print("Classification Report for Decision Tree Classifier")

print(classification_report(ytest, y_pred))



pickle.dump(model, open('./'+filename, 'wb'))
filename = 'gaussian_nb_model.p'



model = GaussianNB()

model.fit(Xtrain, ytrain)



y_pred = model.predict(Xtest)

print("Accuracy: %s%%" % (100*accuracy_score(y_pred, ytest)))

print(confusion_matrix(ytest, y_pred))

print("Classification Report for Gaussian Naïve Bayes")

print(classification_report(ytest, y_pred))



pickle.dump(model, open('./'+filename, 'wb'))
filename = 'linear_svm_model.p'



model = SVC(kernel='linear')

model.fit(Xtrain, ytrain)



y_pred = model.predict(Xtest)

print("Accuracy: %s%%" % (100*accuracy_score(y_pred, ytest)))

print(confusion_matrix(ytest, y_pred))

print("Classification Report for Linear Kernel SVM")

print(classification_report(ytest, y_pred))



pickle.dump(model, open('./'+filename, 'wb'))