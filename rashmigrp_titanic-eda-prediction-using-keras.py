import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Dense, Activation



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
titanic = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic.head()
titanic = titanic.drop(labels = ["PassengerId","Name","Ticket","Cabin"],axis = 1)
titanic.isnull().sum()
titanic["Age"].fillna((titanic["Age"].mean()),inplace = True)

titanic["Embarked"].fillna((titanic["Embarked"].mode()[0]),inplace = True)
titanic.isnull().sum()
titanic.head()
plt.figure(figsize=(15,9))

plt.title('Survived v/s Age')

g = sns.violinplot(x = "Survived",y = "Age",data = titanic, hue = 'Sex')

g.set_xticklabels(['Yes','No'])
plt.figure(figsize=(15,9))

plt.title('Survived v/s Age by Pclass')

g = sns.boxplot(x = "Survived",y = "Age",data = titanic, hue = 'Pclass')

g.set_xticklabels(['Yes','No'])
sur = len(titanic[titanic["Survived"] == 1])

not_sur = len(titanic[titanic["Survived"] == 0])

labels = ["Yes","No"]

values  = [sur, not_sur]



plt.pie(values, colors = ['red','green'], labels = labels, autopct = '%1.1f%%', startangle = 90, 

        pctdistance = 0.85, explode = (0.05,0.05))

centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

plt.tight_layout()

plt.title("Percentage of survival")

plt.show()
le = LabelEncoder()

for feature in titanic.columns:

    titanic[feature]= le.fit_transform(titanic[feature])

    

titanic.head()
sns.heatmap(titanic.corr().loc[["Survived"],:],fmt = ".2f", annot = True)
bins = [0,13,20,60,100]

labels = ['Child', 'Teenagers', 'Adults', 'Old Age']

titanic['Age_Range'] = pd.cut(titanic.Age, bins, labels = labels, include_lowest = True)

titanic.head(3)
fg = sns.FacetGrid(titanic, col = "Age_Range",  hue = "Survived")

fg = (fg.map(plt.scatter, "Age","Fare", edgecolor = "w").add_legend())
X = titanic.drop(["Survived","Age_Range"],axis = 1)

y = titanic["Survived"]
model = Sequential([

    Dense(32,activation = "relu", input_dim = X.shape[1]),

    Dense(16,activation = "relu"),

    Dense(8,activation = "relu"),

    Dense(1,activation = "sigmoid")

    ])
model.compile(optimizer = "adam",

              loss = "mean_squared_error",

              metrics = ["accuracy"])
model.fit(X,y,epochs = 100, batch_size = 10)
test_titanic = pd.read_csv("/kaggle/input/titanic/test.csv")

target_titanic = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

test_titanic.head()
test_titanic.drop(labels = ["PassengerId","Name","Ticket","Cabin"],axis = 1,inplace = True)

test_titanic.head(3)
le = LabelEncoder()

for feature in test_titanic.columns:

    test_titanic[feature]= le.fit_transform(test_titanic[feature])
test_titanic.isnull().sum()
X_test = test_titanic

y_test = target_titanic["Survived"]
evalt = model.evaluate(X_test, y_test, batch_size = 10)
pred = model.predict(X_test,batch_size = 10)

final_survival = []

for val in pred:

    if val > 0.5:

        final_survival.append(1)

    else:

        final_survival.append(0)



target_titanic["final_survived"] = final_survival

target_titanic.head()
expec_sur = target_titanic["Survived"]

pred_sur = target_titanic["final_survived"]

acc = accuracy_score(expec_sur,pred_sur)

print("The accuracy is: {:.2%}".format(acc))
wrong = []

for sur,tar in zip(target_titanic["Survived"],target_titanic["final_survived"]):

    if sur != tar:

        wrong.append([sur,tar])

    

print("The first ten wrong predictions are:",wrong[:10])

print("The length of the array is:",len(wrong))
sub_file = pd.DataFrame({"PassengerId" : target_titanic['PassengerId'], "Survived":target_titanic["final_survived"] })

sub_file.to_csv("titanic_subm_file.csv",index = False)