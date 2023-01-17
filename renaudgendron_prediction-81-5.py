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
train=pd.read_csv("../input/titanic/train.csv")
train.head()
train.info()
#Drop the cabin, Passenger Id, Name and Ticket variables
train1=train.drop(columns=["Cabin","PassengerId","Name","Ticket"])

#Replace missing age values by its median
train1['Age'].fillna((train1['Age'].median()), inplace=True)

#Replace missing embarked values by S
train1["Embarked"].fillna("S", inplace = True)
train1["Family"]=train1["SibSp"]+train1["Parch"]
train2=train1.drop(columns=["SibSp","Parch"])
train2.head()
train2['Families'] = train2['Family'].apply(lambda x: '0' if x==0 else '1')
train3=train2.drop(columns='Family')
train3.head()
#Import the libraries first to see the correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure

#Pclass
figure(figsize=(20,20))
plt.subplot(2,2,1)
p1_data = train3.groupby('Pclass').Survived.mean()
p1 = sns.barplot(y=p1_data.values,
                 x=p1_data.index, 
                 palette=["lightgreen","green","black"])
p1.set(ylabel='Survival Rate', 
       xlabel=' Pclass');
#Embarked
figure(figsize=(20,20))
plt.subplot(2,2,2)
p1_data = train3.groupby('Embarked').Survived.mean()
p1 = sns.barplot(y=p1_data.values,
                 x=p1_data.index, 
                 palette=["blue","grey","black"])
p1.set(ylabel='Survival Rate', 
       xlabel=' Embarked');
#Sex
figure(figsize=(20,20))
plt.subplot(2,2,3)
p1_data = train3.groupby('Sex').Survived.mean()
p1 = sns.barplot(y=p1_data.values,
                 x=p1_data.index, 
                 palette=["blue","pink"])
p1.set(ylabel='Survival Rate', 
       xlabel=' Sex');
#Family size
figure(figsize=(20,20))
plt.subplot(2,2,4)
p1_data = train3.groupby('Families').Survived.mean()
p1 = sns.barplot(y=p1_data.values,
                 x=p1_data.index, 
                 palette=["red","orange"])
p1.set(ylabel='Survival Rate', 
       xlabel=' Family size');
#Fare
train3.boxplot( by='Survived',column=['Fare'],grid= False )
#Age
train3.boxplot( by='Survived',column=['Age'],grid= False )
#Import the libraries
from sklearn.preprocessing import LabelEncoder

#Label Encoder
label_encoder = LabelEncoder()
train4 = train3.apply(label_encoder.fit_transform)
train4

#Set the variables
y=train4[['Survived']]
x=train4[['Pclass','Embarked','Sex','Families','Fare']]
X = (x - np.min(x))/(np.max(x)-np.min(x)).values
#Import librairies
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

#Run the model without PCA
titanic_model = LogisticRegression()
titanic_model.fit(X,y)
scores=cross_val_score(titanic_model, X, y.values.ravel(),cv=10)
scores.mean()
i_range=[0.7,0.75,0.8,0.85,0.9,0.95]
scores1=[]

for i in i_range:
  pca=PCA(i) 
  pca.fit(X) 
  X1=pca.transform(X) 
  titanic_model1 = LogisticRegression()
  scores1=cross_val_score(titanic_model1, X1, y.values.ravel(),cv=10)
  print(scores1.mean())
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


lm = smf.ols(formula='Survived ~  Sex + Pclass + Embarked ', data=train4).fit()
print(lm.summary())

variables = lm.model.exog
vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif 

from sklearn.neighbors import KNeighborsClassifier

k_range = list(range(1, 30))
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y.values.ravel(), cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
    
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
knn = KNeighborsClassifier(n_neighbors=4)
print(cross_val_score(knn, X, y.values.ravel(), cv=10, scoring='accuracy').mean())
from sklearn.ensemble import RandomForestClassifier

j_range = [1,5,10,30,50,75,100,200,500]
j_score = []

for j in j_range:
    RF = RandomForestClassifier(n_estimators=j,random_state=2)
    score = cross_val_score(RF, X, y.values.ravel(), cv=50, scoring='accuracy')
    j_score.append(score.mean())
    
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(j_range, j_score)
plt.xlabel('Value of J for Random Forest')
plt.ylabel('Cross-Validated Accuracy')
import statistics

p_range = list(range(1, 40))
p_score = []

for p in p_range:
   RF = RandomForestClassifier(n_estimators=50,random_state=p)
   score=cross_val_score(RF, X, y.values.ravel(), cv=10, scoring='accuracy').mean()
   p_score.append(score.mean())

statistics.mean(p_score)
from xgboost import XGBClassifier

xgb_clf = XGBClassifier(n_estimators=300,random_state=1)
print(cross_val_score(xgb_clf, X, y.values.ravel(), cv=50, scoring='accuracy').mean())
from catboost import CatBoostClassifier

train4['Ages'] = train2['Age'].apply(lambda y: '0' if y<16 else '1')
train5=train4.drop(columns='Age')
train5




test_data=pd.read_csv("../input/titanic/test.csv")
test_data["Family"]=test_data["SibSp"]+test_data["Parch"]
test_data1=test_data.drop(columns=["SibSp","Parch"])
test_data1['Ages'] = test_data1['Age'].apply(lambda y: '0' if y<16 else '1')
test_data2=test_data1.drop(columns='Age')




features = [ "Sex", "Pclass","Ages","Embarked"]
X = (train5[features]).apply(pd.to_numeric)
X_test = test_data2[features]
X_test1 = (X_test.apply(label_encoder.fit_transform)).apply(pd.to_numeric)

model =XGBClassifier(n_estimators=100)
model.fit(X, y)
predictions = model.predict(X_test1)

output = pd.DataFrame({'PassengerId': test_data2.PassengerId, 'Survived': predictions})
output.to_csv('my_submission16.csv', index=False)
print("Your submission was successfully saved!")
