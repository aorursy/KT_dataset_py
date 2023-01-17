import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as snb
# read csv

df = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
df.head()
df.isnull().sum() # checking missing values
# get test preparation course values count

df['test preparation course'].value_counts()

mapping = {"none" : 0, "completed" : 1}

df['test preparation course'] = df['test preparation course'].map(mapping)

df.head()
import seaborn as sns

sns.pairplot(df,hue='test preparation course',palette='Set1')
df = pd.get_dummies(df, columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch'],drop_first = True)
df.head()
X= df.drop("test preparation course", axis = 1)

y = df["test preparation course"]
# split train test data set

from sklearn.model_selection import train_test_split

X_Train, X_Test, y_Train, y_Test = train_test_split(X, y)
from sklearn.ensemble import RandomForestClassifier

RandomForest = RandomForestClassifier(n_estimators = 100, max_features= 10) 
RandomForest.fit(X_Train, y_Train) 
RandomForest.score(X_Train, y_Train)
RandomForest.score(X_Test, y_Test)
predictions = RandomForest.predict(X_Test)
from sklearn.metrics import classification_report

print(classification_report(y_Test,predictions))
from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_Test,predictions)

fig, ax = plt.subplots(figsize=(5, 5))

sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 
n_features = X.shape[1]

plt.barh(range(n_features),RandomForest.feature_importances_)

plt.yticks(np.arange(n_features),df.columns[1:])
df.head()
df.columns
df["mean_grade"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3
df["math score_squared"] = df["math score"] * df["math score"]

df["reading score_squared"] = df["reading score"] * df["reading score"]

df["writing score_squared"] = df["writing score"] * df["writing score"]
df.columns
#X= df[['math score', 'reading score','writing score', 'gender_male','mean_grade', 'math score_squared', 'reading score_squared','writing score_squared']]

X = df.drop("test preparation course", 1)

y = df["test preparation course"]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

# split train test data set

from sklearn.model_selection import train_test_split

X_Train, X_Test, y_Train, y_Test = train_test_split(X, y)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression() 

model.fit(X_Train, y_Train) 

print (model.score(X_Train, y_Train))

print (model.score(X_Test, y_Test))