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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

df=pd.read_csv('/kaggle/input/indian-liver-patient-records/indian_liver_patient.csv')

df.head()
df['Gender']=df['Gender'].apply(lambda x:1 if x=='Male' else 0)#Convert categorical to numerical

df.isnull().sum()
df['Albumin_and_Globulin_Ratio'].mean()
df=df.fillna(.94)
df.isnull().sum()
#Let us compare the albumin and albumin and globulin ratio by a scatterplot.

f, ax = plt.subplots(figsize=(8, 6))

sns.scatterplot(x="Albumin", y="Albumin_and_Globulin_Ratio",color='mediumspringgreen',data=df)

plt.show()
#Let us compare the Gender based on the Protein Intake.

plt.figure(figsize=(8,6))

df.groupby('Gender').sum()["Total_Protiens"].plot.bar(color='coral')
#Let us compare male and female based on Albumin Level.



plt.figure(figsize=(8,6))

df.groupby('Gender').sum()['Albumin'].plot.bar(color='midnightblue')
#Albumin Level is higher in the case in the case of male compared to female.

#Finally Let us compare them based on the Bilirubin content.

plt.figure(figsize=(8,6))

df.groupby('Gender').sum()['Total_Bilirubin'].plot.bar(color='fuchsia')
#We can clearly see that males has more bilirubin content compared to females.

#Another point to be noted here is that higher the Bilirubin content, higher the case is prone to Liver disease.

#Train the model

X=df.drop('Dataset',axis=1)

X = StandardScaler().fit_transform(X)

y = df['Dataset']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
model = SVC()

parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5)

grid.fit(X_train, y_train)
roc_auc = np.around(np.mean(cross_val_score(grid, X_test, y_test, cv=5, scoring='roc_auc')), decimals=4)

print('Score: {}'.format(roc_auc))
model1= RandomForestClassifier(n_estimators=1000)

model1.fit(X_train, y_train)

predictions = cross_val_predict(model1, X_test, y_test, cv=5)

print(classification_report(y_test, predictions))
score1= np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc'))

np.around(score1, decimals=4)
model2=KNeighborsClassifier()

model2.fit(X_train,y_train)

predictions=cross_val_predict(model2,X_test,y_test,cv=5)

score2= np.around(np.mean(cross_val_score(model2, X_test, y_test, cv=5, scoring='roc_auc')),decimals=4)

print('Score : {}'.format(score2))
model3=LogisticRegression()

parameters={'C':[0.001,0.01,0.1,1,10,100]}

grid = GridSearchCV(estimator=model3, param_grid=parameters, cv=5)

grid.fit(X_train, y_train)

score3= np.around(np.mean(cross_val_score(model3, X_test, y_test, cv=5, scoring='roc_auc')),decimals=4)

print('Score : {}'.format(score3))
names=[]

scores=[]

names.extend(['SVC','RF','KNN','LR'])

scores.extend([roc_auc,score1,score2,score3])

alg=pd.DataFrame({'Score':scores},index=names)

print('Most Accurate : \n{}'.format(alg.loc[alg['Score'].idxmax()]))