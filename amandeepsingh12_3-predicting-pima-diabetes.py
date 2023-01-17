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

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

import plotly as pl

import plotly.express as px

import plotly.figure_factory as ff



import plotly.graph_objects as go



import plotly.offline as py

py.init_notebook_mode(connected=True)
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.info()
cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']



for i in range(len(cols)):

    mean = df[cols[i]].mean()

    df[cols[i]] = df[cols[i]].replace(0, mean)
df['Outcome'].value_counts()
vals = list(df['Outcome'].value_counts())

label = ['Non-diabetic', 'Diabetic']
fig = go.Figure(data=[go.Pie(labels=label, values=vals, title = 'Diabetic and Non-Diabetic', hole = 0.45)])

fig.show()
df['Pregnancies'].value_counts()
sns.set_style('whitegrid')
plt.figure(figsize=(20,8))

sns.countplot(x = 'Pregnancies', data = df, hue = 'Outcome')
corr_fea = df.corr()

plt.figure(figsize=(12,10))

sns.heatmap(corr_fea, annot = True)
df['Glucose'].value_counts()
diabetic_only = df.loc[df['Outcome'] == 1]
diabetic_only.head()
sns.jointplot(data=diabetic_only, y="Age", x="BloodPressure", kind="hex", height = 8)
sns.jointplot(data=diabetic_only, y ="Age", x ="BMI", kind="hex", height = 8)
X =  df.drop('Outcome',  axis = 1)

y = df['Outcome']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from xgboost import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
from sklearn.naive_bayes import GaussianNB



# create Gaussian Naive Bayes model object and train it with the data

nb_model = GaussianNB()



nb_model.fit(X_train, y_train.ravel())
# predict values using training data

nb_predict_train = nb_model.predict(X_train)



# import the performance metrics library from scikit learn

from sklearn import metrics



# check naive bayes model's accuracy

print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train,nb_predict_train)))

print()
nb_predict_test=nb_model.predict(X_test)



from sklearn import metrics



print("Accuracy:{0:.4f}".format(metrics.accuracy_score(y_test,nb_predict_test)))
print("Confusion Matrix")

print("{0}".format(metrics.confusion_matrix(y_test,nb_predict_test)))

print("")
print("Classification Report")

print("{0}".format(metrics.classification_report(y_test,nb_predict_test)))
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42) 

rf_model.fit(X_train,y_train.ravel())
rf_predict_train = rf_model.predict(X_train)

print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train,rf_predict_train)))

print()


rf_predict_test = rf_model.predict(X_test)

print("Accuracy:{0:.4f}".format(metrics.accuracy_score(y_test,rf_predict_test)))

print()



print("Confusion Matrix")

print(metrics.confusion_matrix(y_test, rf_predict_test) )

print("")
print("Classification Report")

print(metrics.classification_report(y_test, rf_predict_test))
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)