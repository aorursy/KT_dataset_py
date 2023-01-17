#Import the libraries you will use



import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
#Import datasets



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train
#Describe them



train.describe()
train.mean()
train.mode()
train.mode().iloc[0]
train.median()
train.iloc[10]
train["Age"][0:10]
len(train)
len(test)
train.dtypes
train.max()
train.min()
train.std()
train.var()
train.kurt()
train.skew()
train.count()
train.count()!=len(train)
(train.count()!=len(train))*1
train.T
train.describe(include = ["O"])
train.describe(exclude = ["object"])
train["Age"].plot(kind="hist")
train["Age"].plot(kind="kde")
train["Fare"].plot(kind="box")
train["Survived"].plot(kind="area")
train["Survived"].plot(kind="hist")
train[["Age","Embarked"]].head()
train[["Age","Fare","Survived"]].tail(3)
train[["Age","Fare"]].sample(5)
train_2 = train[["Age","Fare","Parch","Survived"]]



train_2.head(3)
from pandas.plotting import scatter_matrix



scatter_matrix(train_2, alpha=0.2, figsize=(6, 6), diagonal='kde')
from pandas.plotting import andrews_curves



andrews_curves(train[["Pclass","Age","Survived"]], "Survived")
from pandas.plotting import radviz



radviz(train[["Pclass","Age","Survived"]], "Survived")
import seaborn as sb



sb.regplot(x = train["Survived"], y = train["Age"])
sb.regplot(x = train["Parch"], y = train["Survived"])
sb.regplot(x = train["Fare"], y = train["Age"])
sb.regplot(x = train["Pclass"], y = train["Fare"])
sb.countplot(train["Embarked"])
sb.countplot(train["Pclass"])
sb.countplot(train["Sex"])
sb.countplot(train["Sex"], hue = train["Survived"])
sb.countplot(train["Embarked"], hue = train["Survived"])
sb.violinplot(x = train["Age"])
train.sample(3).mean()
train["Embarked"].value_counts()
train["Sex"].value_counts()
train["Sex"].nunique()
train["Sex"].unique()
train.ix[train["Fare"]>500]
train.ix[(train["Age"]>70) & (train["Sex"]=="male")]
train.ix[(train["Age"]>=75) | (train["Cabin"]=="A5")]
train.count()
train["Age"].mean()
train["Age"].fillna(train["Age"].mean()).count()
train["Age"] = train["Age"].fillna(train["Age"].median())
x = train[["Age","Parch","SibSp","Fare","Survived"]]
x.count()
x.count()==len(x)
x.dtypes
y = x.pop("Survived")
l = []

l
l.append(2)

l.append(1)

l
l[0]
l[-1]
l[0:2]
l[::-1]
for x in [1,2,3]:

    print(x)
length = len(l)



while length > 0:

    print(l[length-1])

    length-= 1
from sklearn.tree import DecisionTreeClassifier as tree



model_1 = tree()



model_1
model_1.fit(x, y)
model_1.predict(x)[0:10]
prediction = model_1.predict(x)
(prediction==y).sum()
(prediction==y).sum()/len(y)
from sklearn.linear_model import LogisticRegression as lr



model_2 = lr()

model_2
model_2.fit(x, y)
prediction_2 = model_2.predict(x)
((prediction_2==y).sum())/len(y)
from sklearn.linear_model import LinearRegression as lin



model_3 = lin()



model_3.fit(x, y)



prediction_3 = model_3.predict(x)



(prediction_3.round()==y).sum()/len(y)
for model in [model_1, model_2, model_3]:

    model.fit(x,y)

    preds = model.predict(x)

    print(((preds.round()==y).sum()/len(y)*100).round())
benchmark = 0

for model in [model_1, model_2, model_3]:

    model.fit(x,y)

    preds = model.predict(x)

    score = (((preds.round()==y).sum()/len(y)*100).round())

    if score > benchmark:

        benchmark = score

        best_model = model
best_model
best_model.predict(x)[:10]