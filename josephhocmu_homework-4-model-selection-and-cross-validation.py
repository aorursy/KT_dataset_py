!pip install regressors
import numpy as np

import pandas as pd 

import statsmodels.formula.api as smf

import statsmodels.api as sma

import seaborn as sns



from sklearn import linear_model as lm

from sklearn.linear_model import LinearRegression 

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, cross_val_score,cross_val_predict, LeaveOneOut

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

from statsmodels.formula.api import ols



import os

print(os.listdir("../input"))
#Data Preprocessing 

d = pd.read_csv("../input/cats.csv")

d = d.drop(d.columns[0], axis=1)

d['Sex'] = d['Sex'].map({'M': 1, 'F': 0})

d = d.dropna()

d.head()
main1 = sm.ols(formula="Hwt ~ Bwt+Sex",data=d).fit()

print(main1.summary())
main2 = sm.ols(formula="Hwt ~ Bwt*Sex",data=d).fit()

print(main2.summary())
anova_table = sma.stats.anova_lm(main1, main2)

print(anova_table)
inputDF = pd.DataFrame(columns = ['Bwt','Sex'])

inputDF['Bwt'] = [3.4]

inputDF['Sex'] = [0]



ypred = main1.predict(inputDF)

print(ypred)
ypred = main2.predict(inputDF)

print(ypred)
#Data Preprocessing 

dTree = pd.read_csv("../input/trees.csv")

dTree = dTree.drop(dTree.columns[0], axis=1)

dTree = dTree.dropna()

dTree.head()
treeMod = sm.ols(formula="Volume ~ Girth+Height",data=dTree).fit()

print(treeMod.summary())
treeMod = sm.ols(formula="Volume ~ Girth*Height",data=dTree).fit()

print(treeMod.summary())
#Logarithmic transformation

res = sm.ols(formula = "Volume ~ np.log(Girth)+np.log(Height)",data=dTree).fit()

print(res.summary())
#Data Preprocessing 

dCars = pd.read_csv("../input/mtcars.csv")

dCars = dCars.dropna()

dCars.head()
carMod = sm.ols(formula="mpg ~ cyl*hp + wt",data=dCars).fit()

print(carMod.summary())
inputDF = pd.DataFrame(columns = ['cyl','hp','wt'])

inputDF['cyl'] = [4, 8, 6]

inputDF['hp'] = [100, 210, 200]

inputDF['wt'] = [2.1, 3.9, 2.9]



ypredCar = carMod.predict(inputDF)

print(ypredCar)
#Data Preprocessing 

df = pd.read_csv("../input/diabetes.csv")

df = df.dropna()

#df['gender'] = df['gender'].map({'male': 1, 'female': 0})

#df['location'] = df['location'].map({'Buckingham': 1, 'Louisa': 0})

#df = pd.get_dummies(df)

df.head()
df.corr()
diaNull = sm.ols(formula="chol ~ 1",data=df).fit()

print(diaNull.summary())
diaFull = sm.ols(formula="chol ~ age*gender*weight*frame + waist*height*hip + location",data=df).fit()

print(diaFull.summary())
#inputDF = df[["age","gender", "weight", "frame_large", "frame_medium", "frame_small", "waist", "height", "hip", "location"]]

#outputDF = df[["chol"]]



#model = sfs(LinearRegression(),k_features=5,forward=True,verbose=2,cv=5,n_jobs=-1,scoring='r2')

#model.fit(inputDF,outputDF)

#model.k_feature_names_
#All Variables: age, gender, weight, frame, waist, height, hip, location

#Model K Features: age, gender, frame, height, hip

f1 = sm.ols(formula="chol ~ 1",data=df).fit()

f2 = sm.ols(formula="chol ~ age",data=df).fit()

f3 = sm.ols(formula="chol ~ age+gender",data=df).fit()

f4 = sm.ols(formula="chol ~ age+weight",data=df).fit()

f5 = sm.ols(formula="chol ~ age+frame",data=df).fit()

f6 = sm.ols(formula="chol ~ age+waist",data=df).fit()

f7 = sm.ols(formula="chol ~ age+height",data=df).fit()

f8 = sm.ols(formula="chol ~ age+hip",data=df).fit()

f9 = sm.ols(formula="chol ~ age+location",data=df).fit()



anova_table = sma.stats.anova_lm(f1, f2)

print("age:")

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f3)

print("age+gender:")

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f4)

print("age+weight")

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f5)

print("age+frame:")

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f6)

print("age+waist:")

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f7)

print("age+height:")

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f8)

print("age+hip:")

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f9)

print("age+location:")

print(anova_table)
#age, gender, weight, frame, waist, height, hip, location

f1 = sm.ols(formula="chol ~ age",data=df).fit()

f2 = sm.ols(formula="chol ~ age*gender",data=df).fit()

f3 = sm.ols(formula="chol ~ age*weight",data=df).fit()

f4 = sm.ols(formula="chol ~ age*frame",data=df).fit()

f5 = sm.ols(formula="chol ~ age*waist",data=df).fit()

f6 = sm.ols(formula="chol ~ age*height",data=df).fit()

f7 = sm.ols(formula="chol ~ age*hip",data=df).fit()

f8 = sm.ols(formula="chol ~ age*location",data=df).fit()



anova_table = sma.stats.anova_lm(f1, f2)

print("age*gender:")

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f3)

print("age*weight")

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f4)

print("age*frame:")

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f5)

print("age*waist:")

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f6)

print("age*height:")

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f7)

print("age*hip:")

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f8)

print("age*location:")

print(anova_table)
print(f2.summary())
#Model K Features: age, gender, frame, height, hip

f1 = sm.ols(formula="chol ~ age*gender",data=df).fit() 

f2 = sm.ols(formula="chol ~ age*gender+frame",data=df).fit()

f3 = sm.ols(formula="chol ~ age*gender+height",data=df).fit()

f4 = sm.ols(formula="chol ~ age*gender+hip",data=df).fit() 



anova_table = sma.stats.anova_lm(f1, f2)

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f3)

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f4)

print(anova_table)
#Model K Features: age, gender, frame, height, hip

f1 = sm.ols(formula="chol ~ age*gender",data=df).fit() 

f2 = sm.ols(formula="chol ~ age*gender*frame",data=df).fit()

f3 = sm.ols(formula="chol ~ age*gender*height",data=df).fit()

f4 = sm.ols(formula="chol ~ age*gender*hip",data=df).fit() 



anova_table = sma.stats.anova_lm(f1, f2)

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f3)

print(anova_table)

anova_table = sma.stats.anova_lm(f1, f4)

print(anova_table)
#Data Preprocessing 

df2 = pd.read_csv("../input/diabetes.csv")

df2 = df.dropna()

df2['gender'] = df2['gender'].map({'male': 1, 'female': 0})

df2['location'] = df2['location'].map({'Buckingham': 1, 'Louisa': 0})

df2 = pd.get_dummies(df2)

df2.head()
inputDF = df2[["age","gender", "weight", "frame_large", "frame_medium", "frame_small", "waist", "height", "hip", "location"]]

outputDF = df2[["chol"]]



model = sfs(LinearRegression(),k_features=5,forward=False,verbose=2,cv=5,n_jobs=-1,scoring='r2')

model.fit(inputDF,outputDF)

model.k_feature_names_
diaFull = sm.ols(formula="chol ~ age*gender*weight*frame + waist*height*hip + location",data=df).fit()

print(diaFull.summary())
#diafull = sm.ols(formula="chol ~ age*gender*weight*frame + waist*height*hip + location",data=df).fit()

f2 = sm.ols(formula="chol ~ age*gender*weight*frame + waist*height*hip",data=df).fit()

print(f2.summary())

f3 = sm.ols(formula="chol ~ age*gender*weight*frame + waist*hip",data=df).fit()

print(f3.summary())

f4 = sm.ols(formula="chol ~ age*gender*weight*frame + hip",data=df).fit()

print(f4.summary())

f5 = sm.ols(formula="chol ~ age*gender*frame",data=df).fit()

print(f5.summary())

f6 = sm.ols(formula="chol ~ age*gender",data=df).fit()

print(f6.summary())

f7 = sm.ols(formula="chol ~ age*frame",data=df).fit()

print(f7.summary())

f8 = sm.ols(formula="chol ~ frame*gender",data=df).fit()

print(f8.summary())

# f1 = sm.ols(formula="chol ~ age*gender*weight*frame + waist*height*hip + location",data=df).fit()

# f2 = sm.ols(formula="chol ~ age*gender*weight*frame + waist",data=df).fit()

# f3 = sm.ols(formula="chol ~ age*gender*weight*frame",data=df).fit()

# f4 = sm.ols(formula="chol ~ age*gender*weight",data=df).fit()

# f5 = sm.ols(formula="chol ~ age*gender*frame",data=df).fit()

# f6 = sm.ols(formula="chol ~ age*gender",data=df).fit()

# f7 = sm.ols(formula="chol ~ gender*frame",data=df).fit()

# f8 = sm.ols(formula="chol ~ weight*frame",data=df).fit()



# anova_table = sma.stats.anova_lm(f1, f2)

# print("f2:")

# print(anova_table)

# anova_table = sma.stats.anova_lm(f1, f3)

# print("f3:")

# print(anova_table)

# anova_table = sma.stats.anova_lm(f1, f4)

# print("f4:")

# print(anova_table)

# anova_table = sma.stats.anova_lm(f1, f5)

# print("f5:")

# print(anova_table)

# anova_table = sma.stats.anova_lm(f1, f6)

# print("f6:")

# print(anova_table)

# anova_table = sma.stats.anova_lm(f1, f7)

# print("f7:")

# print(anova_table)

# anova_table = sma.stats.anova_lm(f1, f8)

# print("f8:")

# print(anova_table)
# f1 = sm.ols(formula="chol ~ age*gender*frame*height*hip",data=df).fit()

# f2 = sm.ols(formula="chol ~ age*gender*frame*height",data=df).fit()

# f3 = sm.ols(formula="chol ~ age*gender*frame*height",data=df).fit()

# f4 = sm.ols(formula="chol ~ age*gender*frame",data=df).fit()

# f5 = sm.ols(formula="chol ~ age*gender",data=df).fit()



# anova_table = sma.stats.anova_lm(f1, f2)

# print("f2:")

# print(anova_table)

# anova_table = sma.stats.anova_lm(f1, f3)

# print("f3:")

# print(anova_table)

# anova_table = sma.stats.anova_lm(f1, f4)

# print("f4:")

# print(anova_table)

# anova_table = sma.stats.anova_lm(f1, f5)

# print("f5:")

# print(anova_table)