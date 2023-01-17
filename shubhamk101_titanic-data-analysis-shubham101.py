import pandas as pd

import numpy

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

train = pd.read_csv("../input/titanic/train.csv")

test  = pd.read_csv("../input/titanic/test.csv")

train.head()
test.head()
train.describe(include='all')
train.dtypes
train.isnull().sum().to_frame()
# display(train.isnull().sum())

# train["Embarked"].value_counts()

train["Age"].replace(numpy.nan,train["Age"].mean(),inplace=True)

train["Embarked"].replace(numpy.nan,"S",inplace=True) # As S is most occuring so nan is replaced by "S"

# train.isnull().sum()
plt.figure(figsize = (28,7))

sns.countplot(train["Age"])
display(train["Survived"].value_counts().to_frame())

sns.countplot(train["Survived"])
display(train["Pclass"].value_counts().to_frame())

sns.countplot(train["Pclass"])
display(train["Sex"].value_counts().to_frame())

sns.countplot(train["Sex"])
display(train["SibSp"].value_counts().to_frame())

sns.countplot(train["SibSp"])

px.pie(train,"SibSp")
display(train["Parch"].value_counts().to_frame())

# sns.countplot(train["Parch"])

px.pie(train,"Parch")
display(train["Embarked"].value_counts().to_frame())

sns.countplot(train["Embarked"])
bins = numpy.linspace(train["Age"].min() , train["Age"].max(),  4)

Names = ["Young", "Middle-Age","Elderly"]

train["BinnedAge"] = pd.cut(train["Age"], bins,labels = Names, include_lowest=True)

# train.head()



display(train["BinnedAge"].value_counts().to_frame())

px.pie(train,"BinnedAge")
train.loc[pd.isnull(train["Cabin"]), "Had Cabin"] = "No"

train["Had Cabin"].replace(numpy.nan, "Yes", inplace=True)
display(train["Had Cabin"].value_counts().to_frame())

sns.countplot(train["Had Cabin"])
binsFare=numpy.linspace(min(train["Fare"]), max(train["Fare"]), 4)

NameFares = ["Low Fare", "Moderate", "Expensive"]

train["BinnedFare"] = pd.cut(train["Fare"], binsFare,labels = NameFares, include_lowest=True)



display(train["BinnedFare"].value_counts().to_frame())

sns.countplot(train["BinnedFare"])
p1 = px.violin(train, "Survived","Age", color="Sex", box=True, points="all", hover_data=train.columns)

p2 = px.violin(train, "Sex","Age", color="Survived", box=True, points="all", hover_data=train.columns)

p1.show()

p2.show()
plt.figure(figsize = (24,8))

plt.plot(train[["Age","Survived"]].groupby(["Age"]).mean())
display(train[["Sex","Survived"]].groupby(["Sex"]).mean())

px.bar(train,"Sex","Survived")
display(train[["BinnedAge","Survived"]].groupby(["BinnedAge"]).mean()   ,   train[["BinnedAge","Sex","Survived"]].groupby(["BinnedAge","Sex"]).mean())

px.bar(train,"BinnedAge", "Survived", color = "Sex",barmode="group")
display(train[["Pclass","Survived"]].groupby(["Pclass"]).mean()   ,   train[["Pclass","Sex","Survived"]].groupby(["Pclass","Sex"]).mean())

px.bar(train,"Pclass","Survived",color="Sex")
display(train[["Embarked","Survived"]].groupby(["Embarked"]).mean()   ,   train[["Embarked","Sex","Survived"]].groupby(["Embarked","Sex"]).mean())

px.bar(train,"Embarked","Survived",color="Sex",barmode="group")
display(train[["SibSp","Survived"]].groupby(["SibSp"]).mean()   ,   train[["SibSp","Sex","Survived"]].groupby(["SibSp","Sex"]).mean())

px.bar(train,"SibSp","Survived",color = "Sex",barmode="group")
display(train[["Parch","Survived"]].groupby(["Parch"]).mean()  ,  train[["Parch","Sex","Survived"]].groupby(["Parch","Sex"]).mean())

px.bar(train,"Parch","Survived",color="Sex",barmode="group")
display(train[["Had Cabin","Survived"]].groupby(["Had Cabin"]).mean()  ,  train[["Had Cabin","Sex","Survived"]].groupby(["Had Cabin","Sex"]).mean())

px.bar(train,"Had Cabin","Survived",color="Sex",barmode="group")
display(train[["BinnedFare","Survived"]].groupby(["BinnedFare"]).mean()  ,  train[["BinnedFare","Sex","Survived"]].groupby(["BinnedFare","Sex"]).mean())

px.bar(train, "BinnedFare","Survived",color="Sex",barmode="group")
px.box(train,"Embarked","Fare")
px.box(train,"Survived","Age",color="Sex")
plt.figure(figsize = (20,9))

sns.boxplot("SibSp","Age",data=train)
plt.figure(figsize = (14,9))

sns.boxplot("Embarked","Parch",data=train)
px.box(train,"Survived","Fare")
testg = train[["Sex","Embarked","Pclass","Survived"]]

Group = testg.groupby(["Sex","Embarked","Pclass"],as_index=False).mean()

Group
train.head()
train.loc[train["Sex"] == "male" , "01Sex"] = 0

train.loc[train["Sex"] == "female" , "01Sex"] = 1



train.loc[train["Embarked"] == "S" , "123Embarked"] = 1

train.loc[train["Embarked"] == "C" , "123Embarked"] = 2

train.loc[train["Embarked"] == "Q" , "123Embarked"] = 3



train.loc[train["Had Cabin"] == "Yes" , "01 Had Cabin"] = 1

train.loc[train["Had Cabin"] == "No" , "01 Had Cabin"] = 0





train.loc[train["BinnedFare"] == "Low Fare" , "123BinnedFare"] = 1

train.loc[train["BinnedFare"] == "Moderate" , "123BinnedFare"] = 2

train.loc[train["BinnedFare"] == "Expensive" , "123BinnedFare"] = 3



# 

train.head()
plt.figure(figsize = (14,10))

testg = pd.DataFrame(train[["01Sex","123Embarked","Pclass","Survived"]])

sns.heatmap(testg.corr(),annot=True)
train.loc[train["BinnedAge"] == "Young" , "123BinnedAge"] = 1

train.loc[train["BinnedAge"] == "Middle-Age" , "123BinnedAge"] = 2

train.loc[train["BinnedAge"] == "Elderly" , "123BinnedAge"] = 3

train.head()
display(train.corr())

plt.figure(figsize = (24,12))

sns.heatmap(train.corr(),annot=True)
from sklearn.linear_model import LinearRegression 

from sklearn.pipeline import Pipeline 

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

train.head()
x = train[["Pclass","Age","SibSp","Parch","Fare","01Sex","123Embarked","123BinnedAge", "01 Had Cabin","123BinnedFare"]]

y = train["Survived"]

x.describe()



TrainX, TestX, TrainY, TestY = train_test_split(x,y,test_size=0.10,random_state=0)
lr = LinearRegression()

lr.fit(TrainX,TrainY)

# lr.fit(x,y)

print("Rsquared Score for Linear Regression Model = ", lr.score(TestX,TestY))
RsqTest = []

order = numpy.arange(1,7,1)



for n in order:

    pr = PolynomialFeatures(degree = n )

    Scale = StandardScaler()

    TrainXTrans = pr.fit_transform(TrainX) 

    TestXTrans = pr.fit_transform(TestX)

    

    

    lr.fit(TrainXTrans,TrainY)

    RsqTest.append(lr.score(TestXTrans,TestY))

        

display(RsqTest)
input = [["Scale",StandardScaler()],["Poly",PolynomialFeatures(degree=3)],["lr",LinearRegression()]]

Pipe = Pipeline(input)



# Pipe.fit(TrainX,TrainY)

Pipe.fit(x,y)



Pipe.score(TestX,TestY)
RidgeModel = Ridge(alpha=0.000000001)

RidgeModel.fit(TrainX,TrainY)

# RidgeModel.fit(x,y)



RidgeModel.score(TestX,TestY)




RsqTrain2 = []

RsqTest2 = []



ALFA = [0.000000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000,1000000,10000000]



for m in ALFA:

    RidgeModel1 = Ridge(alpha=m)

    

#     RidgeModel1.fit(TrainX,TrainY)

    RidgeModel1.fit(x,y)

    

    RsqTest2.append(RidgeModel1.score(TestX,TestY))

    RsqTrain2.append(RidgeModel1.score(TrainX,TrainY))



display(RsqTest2) 

# display(RsqTrain2)

# plt.plot(RsqTest2)



from sklearn.model_selection import GridSearchCV

parameters = ({"alpha" : [0.000000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000,1000000,10000000]})

RR = Ridge()

Grid = GridSearchCV(RR,parameters,cv=11)

Grid.fit(TestX,TestY)

# Grid.fit(x,y)

Grid.best_estimator_

Scr = Grid.cv_results_

Scr["mean_test_score"]
test.head()

test.isnull().sum()

test["Fare"].replace(numpy.nan,test["Fare"].mean(),inplace=True)

test["Age"].replace(numpy.nan,test["Age"].mean(),inplace=True)

test.isnull().sum()
test.loc[test["Sex"] == "male" , "01Sex"] = 0

test.loc[test["Sex"] == "female" , "01Sex"] = 1



test.loc[test["Embarked"] == "S" , "123Embarked"] = 1

test.loc[test["Embarked"] == "C" , "123Embarked"] = 2

test.loc[test["Embarked"] == "Q" , "123Embarked"] = 3



binsTEST = numpy.linspace(test["Age"].min() , test["Age"].max(),  4)

NamesTEST = ["Young", "Middle-Age","Elderly"]

test["BinnedAge"] = pd.cut(test["Age"], binsTEST,labels = NamesTEST, include_lowest=True)



test.loc[test["BinnedAge"] == "Young" , "123BinnedAge"] = 1

test.loc[test["BinnedAge"] == "Middle-Age" , "123BinnedAge"] = 2

test.loc[test["BinnedAge"] == "Elderly" , "123BinnedAge"] = 3



test.loc[pd.isnull(test["Cabin"]), "Had Cabin"] = "No"

test["Had Cabin"].replace(numpy.nan, "Yes", inplace=True)



test.loc[test["Had Cabin"] == "Yes" , "01 Had Cabin"] = 1

test.loc[test["Had Cabin"] == "No" , "01 Had Cabin"] = 0



binsFare=numpy.linspace(min(test["Fare"]), max(test["Fare"]), 4)

NameFares = ["Low Fare", "Moderate", "Expensive"]

test["BinnedFare"] = pd.cut(test["Fare"], binsFare,labels = NameFares, include_lowest=True)

test.loc[test["BinnedFare"] == "Low Fare" , "123BinnedFare"] = 1

test.loc[test["BinnedFare"] == "Moderate" , "123BinnedFare"] = 2

test.loc[test["BinnedFare"] == "Expensive" , "123BinnedFare"] = 3



test.head()
TestFeatures = test[["Pclass","Age","SibSp","Parch","Fare","01Sex","123Embarked","123BinnedAge", "01 Had Cabin","123BinnedFare"]]
input = [["Scale",StandardScaler()],["Poly",PolynomialFeatures(degree=3)],["lr",LinearRegression()]]

Pipe = Pipeline(input)

Pipe.fit(x,y)
Y = numpy.array(Pipe.predict(TestFeatures))

# Y = Y.reshape(418,1)

# Finaldf = test["PassengerId"].to_frame()



data = {"PassengerId":test["PassengerId"], "Surv":Y}

df = pd.DataFrame(data)

# data

df
df["Surv"].describe().to_frame()
df.loc[df["Surv"] < 0.50, "Survived"] = 0

df.loc[df["Surv"] >=0.50, "Survived"] = 1



df["Survived"] = df["Survived"].astype("int64")

df.head(11)
df.drop("Surv",axis=1,inplace = True)
df
df.to_csv("./Titanic-Final.csv", index = False)