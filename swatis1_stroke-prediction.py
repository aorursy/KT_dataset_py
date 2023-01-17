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
stroke= pd.read_csv('/kaggle/input/healthcare-dataset-stroke-data/train_2v.csv')

stroke.head()
#checking the shape of our data

stroke.shape
stroke.columns
stroke= stroke.drop('id', axis=1)
#checking the null values

stroke.isnull().sum()
stroke[stroke==0].count()
import seaborn as sns

import matplotlib.pyplot as plt

sns.countplot(stroke['stroke'])
#filling null values with the mean

stroke['bmi'].fillna(stroke['bmi'].mean(), inplace= True)
#filling null values with mode

stroke['smoking_status'].fillna(stroke['smoking_status'].mode()[0], inplace=True)
#checking the data

stroke.isnull().sum()
stroke.describe()
stroke.info()
sns.distplot(stroke['avg_glucose_level'], bins=20)
sns.distplot(stroke['bmi'], bins=20)
sns.distplot(stroke['age'], bins=20)
#chances of stroke incraeses with incraese in age

stroke.loc[stroke['stroke'] == 0,

                 'age'].hist(label='No Stroke')

stroke.loc[stroke['stroke'] == 1,

                 'age'].hist(label='Heart Stroke')

plt.xlabel('Age')

plt.ylabel('Heart Stroke')

plt.legend()
#chances of stroke more with bmi 20-40



stroke.loc[stroke['stroke'] == 0,

                 'bmi'].hist(label='No Stroke')

stroke.loc[stroke['stroke'] == 1,

                 'bmi'].hist(label='Heart Stroke')

plt.xlabel('BMI')

plt.ylabel('Heart Stroke')

plt.legend()
#chances of stroke high with glucose levels in range of 70-100



stroke.loc[stroke['stroke'] == 0,

                 'avg_glucose_level'].hist(label='No Stroke')

stroke.loc[stroke['stroke'] == 1,

                 'avg_glucose_level'].hist(label='Heart Stroke')

plt.xlabel('Glucose Level')

plt.ylabel('Heart Stroke')

plt.legend()
#married females have more chances of heart stroke than married males

pd.pivot_table(stroke, index= 'stroke', columns='gender', values='ever_married', aggfunc= 'count')
#females with hypertension has more chance of heart stroke than males having hypertension problem

pd.pivot_table(stroke, index= 'stroke', columns='gender', values='hypertension', aggfunc= 'count')
#females with heart disease has more chances of stroke

pd.pivot_table(stroke, index= 'stroke', columns='gender', values='heart_disease', aggfunc= 'count')
#people having private jobs and has a habit of smoking has more chance of heart stroke 

pd.pivot_table(stroke, index= 'stroke', columns='work_type', values='smoking_status', aggfunc= 'count')
#as age incraeses gender does not play any role in heart stroke

sns.scatterplot(x= 'stroke', y='age', hue='gender', sizes= (15,200), data=stroke)

plt.xticks(rotation=90)
#can't say that marriage plays a role in heart stroke as people generally marry after the age of 25years

sns.relplot(x= 'stroke', y='age', hue= 'ever_married', sizes= (15,200), data=stroke)

plt.xticks(rotation=90)
#with age glucose level increases which increases the chances of stroke

plt.figure(figsize=(28,20))

sns.relplot(x= 'avg_glucose_level', y='age', hue= 'stroke', sizes= (15,200), data=stroke)

plt.xticks(rotation=90)
stroke['stroke'].value_counts()
#performing label encoding for the dataset

from sklearn import preprocessing 



encoder = preprocessing.LabelEncoder()



for i in stroke.columns:

    if isinstance(stroke[i][0], str):

            stroke[i] = encoder.fit_transform(stroke[i])
#standardizing the dataset with Standard Scaler

from sklearn.preprocessing import StandardScaler 

  

scalar = StandardScaler() 

  

scalar.fit(stroke) 

scaled_data = scalar.transform(stroke)
#checing the data for first 10 values

stroke.head(10)
#dropping the output label

X= stroke.drop('stroke', axis=1)

X.shape
y= stroke['stroke']

y.shape
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3, random_state = 1000)
log= LogisticRegression()
log.fit(X_train,y_train)
log.score(X_train, y_train)
stroke['stroke'].value_counts()
#to retain the original data, we craeted a copy of the dataset

stroke_copy= stroke.copy()
stroke_copy.head()
#creating a list of data values which is more in number

#to make a balance data

li = list(stroke_copy[stroke_copy.stroke == 0].sample(n=41800).index)
#dropping the values

stroke_copy = stroke_copy.drop(stroke_copy.index[li])



stroke_copy['stroke'].value_counts()
X_drop= stroke_copy.drop('stroke', axis=1)

X_drop.shape
y_drop= stroke_copy.stroke

y_drop.shape
X_droptr,X_dropts,y_droptr,y_dropts = train_test_split(X_drop,y_drop,test_size=.3, random_state = 1000)
#creating a Logistic Model for the new data

log.fit(X_droptr, y_droptr)
#the accuracy has dropped

log.score(X_droptr, y_droptr)
#predicting the output with Logistic

y_underlog= log.predict(X_dropts)
from sklearn.metrics import accuracy_score, f1_score, classification_report, recall_score, confusion_matrix

print('The accuracy score of the model is:', accuracy_score(y_dropts,y_underlog)*100)

print('The F1 score of the model is:', f1_score(y_dropts, y_underlog)*100)

print('The recall score of the model is:', recall_score(y_dropts, y_underlog)*100)

print('The confusion matrix of the model is:', confusion_matrix(y_dropts, y_underlog))

print('The classification report of logistic model is:', classification_report(y_dropts, y_underlog))
from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X_droptr, y_droptr)
#tuning the model using criterion and max_depth only



from sklearn.model_selection import GridSearchCV

param = {

    'criterion': ['entropy', 'gini'],

    'max_depth' :[2,3,4,5]

}

grid_svc = GridSearchCV(model, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(X_droptr, y_droptr)
grid_svc.best_params_
model= tree.DecisionTreeClassifier(criterion= 'gini', max_depth= 3)

model.fit(X_droptr, y_droptr)
model.score(X_droptr, y_droptr)
y_undersDT= model.predict(X_dropts)
print('The accuracy score of the model is:', accuracy_score(y_dropts,y_undersDT)*100)

print('The F1 score of the model is:', f1_score(y_dropts, y_undersDT)*100)

print('The recall score of the model is:', recall_score(y_dropts, y_undersDT)*100)

print('The confusion matrix of the model is:', confusion_matrix(y_dropts, y_undersDT))

print('The classification report of base model is:', classification_report(y_dropts, y_undersDT))
from sklearn import ensemble

rf= ensemble.RandomForestClassifier()
rf.fit(X_droptr, y_droptr)
#tuning the model using criterion, n_estimators, bootstrap and max_depth

param = {

    'criterion': ['entropy', 'gini'],

    'n_estimators': [10,20,30,40,50],

    'bootstrap': ['True', 'False'],

    'max_depth': [2,3,4,5]

}

grid_svc = GridSearchCV(rf, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(X_droptr, y_droptr)
grid_svc.best_params_
rf= ensemble.RandomForestClassifier(bootstrap= 'True', criterion= 'entropy', max_depth= 4, n_estimators=50)
rf.fit(X_droptr, y_droptr)
#checking the accuracy score of the Random Forest Model

rf.score(X_droptr, y_droptr)
#predicting the values through Random Forest

y_predRF= rf.predict(X_dropts)
print('The accuracy score of the model is:', accuracy_score(y_dropts,y_predRF)*100)

print('The F1 score of the model is:', f1_score(y_dropts, y_predRF)*100)

print('The recall score of the model is:', recall_score(y_dropts, y_predRF)*100)

print('The confusion matrix of the model is:', confusion_matrix(y_dropts, y_predRF))

print('The classification report of base model is:', classification_report(y_dropts, y_predRF))
cm_log= confusion_matrix(y_dropts, y_underlog)

cm_DT= confusion_matrix(y_dropts, y_undersDT)

cm_RF= confusion_matrix(y_dropts, y_predRF)
plt.figure(figsize=(24,12))



plt.suptitle("Confusion Matrixes After Undersampling",fontsize=24)

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)



plt.subplot(2,3,1)

plt.title("Logistic Regression Confusion Matrix")

sns.heatmap(cm_log,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,2)

plt.title("Decision Tree Confusion Matrix")

sns.heatmap(cm_DT,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(2,3,3)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_RF,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.show()