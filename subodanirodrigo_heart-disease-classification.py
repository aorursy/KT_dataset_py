import pandas as pd

import numpy as np

import os

print(os.listdir("../input"))
data=pd.read_csv("../input/heart.csv")

data.head()
print(data.shape)
def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    a = pd.concat([total], axis=1, keys=['Total'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    a['Types'] = types

    return(np.transpose(a))
%%time

missing_data(data)
#descriptive analysis

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

f,ax=plt.subplots(1,2,figsize=(18,8))

data['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('target')

ax[0].set_ylabel('')

sns.countplot('target',data=data,ax=ax[1])

ax[1].set_title('target')

plt.show()
data["trestbps"].hist();



plt.boxplot(x=data.trestbps[data.target==1])
data["thalach"].hist();
data["chol"].hist();
plt.boxplot(x=data.chol[data.target==1])
data['sex'].value_counts().plot.bar();
pd.crosstab(data.sex,data.target).plot(kind="bar",figsize=(15,6))

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency')

plt.show()
data['fbs'].value_counts().plot.bar();
pd.crosstab(data.fbs,data.target).plot(kind="bar",figsize=(15,6))

plt.title('Heart Disease Frequency According To FBS')

plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')

plt.xticks(rotation = 0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency of Disease or Not')

plt.show()
data["age"].hist();
plt.scatter(x=data.age[data.target==1], y=data.thalach[(data.target==1)], c="red")

plt.scatter(x=data.age[data.target==0], y=data.thalach[(data.target==0)])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
a=pd.get_dummies(data['sex'], prefix = "cp")

b=pd.get_dummies(data['cp'], prefix = "cp")

c=pd.get_dummies(data['thal'], prefix = "thal")

d=pd.get_dummies(data['slope'], prefix = "slope")
frames = [data, a, b, c,d]

data = pd.concat(frames, axis = 1)

data.head()
#remove existing categorical variable.(dummy variables are there)

data = data.drop(columns = ['sex','cp', 'thal', 'slope'])

data.head()
#seperating predictor & response variable

y = data.target.values

x = data.drop(['target'], axis = 1)

#spliting train and test data

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
#Random forest classifier

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=20000, max_depth=5, random_state=0)

clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
# Model Accuracy

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Gradient boosting classifier

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=100,max_depth=3,learning_rate=0.5,random_state=0)

model.fit(x_train, y_train)
y_pre = model.predict(x_test)
# Model Accuracy

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pre))