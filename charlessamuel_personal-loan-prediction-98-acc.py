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

import missingno as msno

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

from matplotlib import pyplot as plt



from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split

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

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

%matplotlib inline
df = pd.read_csv('../input/personal-loan-modeling/Bank_Personal_Loan_Modelling.csv', index_col='ID')

df.head()
rows, cols = df.shape

print("Rows:", rows, "\nColumns:", cols)
df.info()
df.describe()
age_df = df.Age.value_counts().to_frame(name="Number in Age")



fig = px.bar(age_df, x=age_df.index, y='Number in Age', title="Number of people in each Age")

fig.update_layout(

    xaxis_title="Age",

)

fig.show()
fig = px.bar(df, x='Age', y='Income', title='Income by Age', color='Income')

fig.show()
fig = px.bar(df, x='Age', y='Experience', title='Experience by Age', color='Experience')

fig.show()
fig = px.bar(df.Education.value_counts().to_frame(), x=df.Education.value_counts().to_frame().index, y='Education', title='Education Count')

fig.show()
fig = px.bar(df, x='Age', y='Mortgage', title='Mortgage Amount by Age', color='Mortgage')

fig.show()
X = df.drop(['Personal Loan'], axis=1)

y = df['Personal Loan']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
classifiers = {

    #'Multinomial NB': MultinomialNB(), #Does not work for negative values

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

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    acc_scores[classifier] = accuracy_score(y_pred, y_test)

    print(classifier, acc_scores[classifier])
model = GradientBoostingClassifier()

model.fit(x_train, y_train)



y_pred = model.predict(x_test)

print("Accuracy: %s%%" % (100*accuracy_score(y_pred, y_test)))

print(confusion_matrix(y_test, y_pred))

print("Classification Report for Random Forest Classifier")

print(classification_report(y_test, y_pred))