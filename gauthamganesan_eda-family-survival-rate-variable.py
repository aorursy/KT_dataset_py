import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
%matplotlib inline
sns.set_style("dark")
trdf = pd.read_csv('../input/train.csv')
tstdf = pd.read_csv('../input/test.csv')

df= pd.concat([trdf,tstdf])
df.describe()
df.count()


df[['lastname','firstname']] = df['Name'].str.split(",",1,expand=True)
df[['title','fname']] = df['firstname'].str.split(".",1,expand=True)
df['Male'] = np.where(df['Sex'].str.lower()=='male', 1, 0)
df['Female'] = np.where(df['Sex'].str.lower()=='female', 1, 0)
df['Mister'] = np.where(df['title'].str.lower() == ' mr', 1, 0)
df['Class1'] = np.where(df['Pclass']==1, 1, 0)
df['Class2'] = np.where(df['Pclass']==2, 1, 0)
df['Class3'] = np.where(df['Pclass']==3, 1, 0)


# print df.describe()
# print traindf.describe()
# print testdf.describe()

# regression for age
regr = linear_model.LinearRegression()
agetraindf = df[df.Age.notnull()]
agetestdf = df[df.Age.isnull()]

ageX_train = agetraindf[[  'Class1',  'Class2',  'Class3']] 
ageX_test =  agetestdf[['Class1',  'Class2',  'Class3']] 
ageY_train = agetraindf["Age"]

regr.fit(ageX_train, ageY_train)
# df['Age'].fillna(df['Age'].mean(), inplace=True)

agetestdf['Age']=  regr.predict(ageX_test)
df= pd.concat([agetraindf,agetestdf])
df.describe()

df['Child'] = np.where(df['Age']<=5, 1, 0)
df['YA'] = np.where((df['Age']>5) & (df['Age']<=15), 1, 0)
df['Adult'] = np.where(df['Age']>=15, 1, 0)
df['Age'] = df['Age']
df['Agebkt'] = pd.cut(df['Age'], [0,5,14,28,32,50,100])
# print ageY_pred
# print regr.score(ageX_train, ageY_train)
# # The coefficients
# print('Coefficients: \n', regr.coef_)
# # The mean squared error
# print("Mean squared error: %.2f"
#       % mean_squared_error(ageY_train, ageY_pred))
# # Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(ageY_train, ageY_pred))

# Plot outputs
# plt.scatter(ageX_test, ageY_pred,  color='black')
# plt.plot(ageX_test, ageY_pred, color='blue', linewidth=3)

# plt.xticks(())
# plt.yticks(())


df.hist(column="Age",bins=50)


# # fammily survival rate - including just cause it took a lot of time :-P...not sure if useful..well see
df['famsize'] = df['SibSp']+df['Parch']+1
df['famtestdat'] = np.where(df.Survived.isnull() & df.famsize> 1, 1, 0)
df['famtestdatcnt'] = df.groupby(['lastname','famsize'])['famtestdat'].transform('sum')
df['famgp'] = df['famsize']/10

df['famwithmr'] = df.groupby(['lastname','famsize'])['Mister'].transform('sum')
df['famwithmr'] = np.where(df['famsize']==1, 1, df['famwithmr'] )
df['famsizbutemr'] = df['famsize']-df['famwithmr']-df['famtestdatcnt']
df['famsurvcnt'] = df.groupby(['lastname','famsize'])['Survived'].transform('sum')
df['famsurrate'] = df.famsurvcnt/df.famsizbutemr
df['famsurrate'] = np.where(df['famsurrate']==np.inf , 0, df['famsurrate'])
df['famsurrate'] = np.where(df['title'].str.lower() == ' mr', 0, df['famsurrate'])
df['famsurrate'].fillna(0, inplace=True)
df['nofam'] = np.where(df['famsize']==1, 1, 0)
df['smlfam'] = np.where((df['famsize']>1) & (df['famsize']<5), 1, 0)
df['bigfam'] = np.where(df['famsize']>=5, 1, 0)
# df.to_csv('C:/Users/gauth/Desktop/Titanic/df.csv')
traindf = df[df.Survived.notnull()]
testdf = df[df.Survived.isnull()]

sns.countplot(x="Survived", data=traindf, color="grey")

svd = traindf[traindf.Survived == 1]
title = pd.crosstab([svd.title],[svd.Pclass])
title.plot(kind="bar",stacked = False,width = .7, figsize = (12,7),color=["Orange","yellow","green"],grid = False,label="Survived : Title Bracket against Pclass")
svd = traindf[traindf.Survived == 0]
title = pd.crosstab([svd.title],[svd.Pclass])
title.plot(kind="bar",stacked = False,width = .7, figsize = (12,7),color=["Orange","yellow","green"],grid = False,label="Dead : Title Bracket against Pclass")

title = pd.crosstab([traindf.title],[traindf.Survived])
title.plot(kind="bar",stacked = False,width = .7, figsize = (12,7),color=["red","blue"],grid = False,label="Title Bracket against Survived")

agetitle = pd.crosstab([traindf.Agebkt],[traindf.title])
agetitle.plot(kind="bar",stacked = False,width = .7, figsize = (12,7),grid = False,label="Title Bracket against Survived")
lastname = pd.crosstab([traindf.lastname],[traindf.Survived])
lastname.plot(kind="bar",stacked = False,width = .7, figsize = (12,7),color=["red","blue"],grid = False,label="Title Bracket against Survived")
agebkt = pd.crosstab([traindf.Agebkt],[traindf.Survived])
agebkt.plot(kind="bar",stacked = False,color=["red","blue"],grid = False,label="AGE Bracket against Survived")
SibSp = pd.crosstab([traindf.SibSp],[traindf.Survived])
SibSp.plot(kind="bar",stacked = False,color=["red","blue"],grid = False,)
Parch = pd.crosstab([traindf.Parch],[traindf.Survived])
Parch.plot(kind="bar",stacked = False,color=["red","blue"],grid = False,)
famsize = pd.crosstab([df.famsize],[df.Survived])
famsize.plot(kind="bar",stacked = False,color=["red","blue"],grid = False)

sex = pd.crosstab([df.Sex],[df.Survived])
sex.plot(kind="bar",stacked = False,color=["red","blue"],grid = False,)


Pclass = pd.crosstab([df.Pclass],[df.Survived])
Pclass.plot(kind="bar",stacked = False,color=["red","blue"],grid = False,)


sns.pairplot(df, vars=["Age", "famsize","Pclass","famgp","famsurrate"], hue="Survived")
plt.show()

X_train = traindf[['Child','YA','Adult','Male',  'Female',  'Class1',  'Class2',  'Class3', 'famsurrate','nofam','smlfam','bigfam']] 
X_test = testdf[['Child','YA','Adult','Male',  'Female',  'Class1',  'Class2',  'Class3', 'famsurrate','nofam','smlfam','bigfam']]
Y_train = traindf["Survived"]

print ("Final Features\n")
print (X_train.head(10))
print (X_test.head(10))
print (Y_train.head(10))


logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

print (logreg.coef_)

Y_pred = logreg.predict(X_test)
print (logreg.score(X_train, Y_train))

pred = pd.DataFrame(data=Y_pred)

Accuracy = {}
scores = cross_val_score(logreg, X_train, Y_train, cv=5)
print (scores)
accuracy = scores.mean()
print("Logistic Regresion Accuracy :", accuracy)
Accuracy["logisticRegression"] = accuracy



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)
pred = pd.DataFrame(data=Y_pred)
print (random_forest.score(X_train, Y_train))

Accuracy = {}
scores = cross_val_score(random_forest, X_train, Y_train, cv=5)
print (scores)
accuracy = scores.mean()
print("random_forest Accuracy :", accuracy)
Accuracy["random_forest"] = accuracy
