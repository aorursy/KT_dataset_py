

import numpy as np 

import pandas as pd  



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

import missingno as msno

import scipy.stats as stats

from sklearn.neighbors import LocalOutlierFactor

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.model_selection import train_test_split,cross_val_score,cross_val_predict

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV,cross_val_score,cross_val_predict

from numpy import log, log1p

from scipy.stats import boxcox

import pylab

from sklearn.linear_model import LinearRegression

#! pip install yellowbrick

from yellowbrick.regressor import residuals_plot

from scipy.stats import shapiro,boxcox,yeojohnson

from yellowbrick.regressor import prediction_error

!pip install dython

from dython import nominal

from mlxtend.plotting import plot_linear_regression,plot_learning_curves

%matplotlib inline
data=pd.read_csv("/kaggle/input/insurance/insurance.csv")

df=data.copy()



df.head()
print("row :",df.shape[0]," ","column :",df.shape[1])
df.describe().T
df.describe(include=["object"]).T
print("Sum of missing values :",df.isnull().sum().sum())
df.eq(0).sum()
nominal.associations(df,figsize=(20,10),mark_columns=True,cmap="rainbow");
plt.rcParams.update({'font.size': 12})

plt.figure(figsize=(10,5))

corr=df.corr()

mask=np.zeros_like(corr,dtype=np.bool)

mask[np.triu_indices_from(mask)]=True

sns.heatmap(corr.abs(),annot=True,cmap="coolwarm",mask=mask);
plt.figure(figsize=(12,5))

plt.subplot(121)

sns.distplot(df.charges,color="b");

plt.subplot(122)

sns.distplot(log(df.charges),color="b");



sns.pairplot(df,kind="reg",hue="smoker",aspect=2);
sns.pairplot(df,kind="reg",hue="sex",aspect=2);
sns.relplot(x="bmi",y="charges",hue="smoker",data=df,kind="scatter",aspect=2);
sns.relplot(x="bmi",y="charges",hue="children",data=df,kind="scatter",aspect=2,palette='coolwarm');
sns.catplot(x="age", y="charges", hue="smoker", data=df,aspect=3,kind="point");
sns.lmplot(x="bmi", y="charges", hue="smoker", data=df,aspect=2);
plt.figure(figsize=(12,5));

sns.jointplot(x="bmi", y="charges" ,data=df, kind="reg");
plt.figure(figsize=(12,5));

sns.jointplot(x="age", y="bmi" ,data=df);
plt.figure(figsize=(12,5));

sns.distplot(df.age);
plt.figure(figsize=(12,5));

stats.probplot(df.charges, dist="norm", plot=pylab) ;
plt.figure(figsize=(12,5));

df.groupby("smoker")["charges"].mean().plot.bar(color="r");
plt.figure(figsize=(12,5));

df.groupby("children")["charges"].mean().plot.bar(color="g");
print(sns.FacetGrid(df,hue="sex",height=5,aspect=2).map(sns.kdeplot,"charges",shade=True).add_legend());
print(sns.FacetGrid(df,hue="region",height=5,aspect=2).map(sns.kdeplot,"charges",shade=False).add_legend());
print(sns.catplot(x="sex",y="charges",hue="smoker",data=df,kind="bar",aspect=2));
print(sns.catplot(x="sex",y="charges",hue="region",data=df,kind="bar",aspect=2));
sns.catplot(x="smoker",y="charges",data=df,kind="box",aspect=2);
sns.catplot(x="sex",y="charges",data=df,kind="box",aspect=2);
sns.catplot(x="sex",y="charges",hue="smoker",data=df,kind="box",aspect=2);
sns.catplot(x="region",y="charges",data=df,kind="box",aspect=2);
sns.catplot(x="children",y="charges",data=df,kind="box",aspect=2);
labels=["too_weak","normal","heavy","too_heavy"]

ranges=[0,18.5,24.9,29.9,np.inf]

df["bmi"]=pd.cut(df["bmi"],bins=ranges,labels=labels)
print(sns.FacetGrid(df,hue="bmi",height=5,aspect=2).map(sns.kdeplot,"charges",shade=False).add_legend());
print(sns.catplot(x="bmi",y="charges",kind="bar",data=df,aspect=2));
print(sns.catplot(x="bmi",y="charges",hue="children",kind="bar",data=df,aspect=3));
print(sns.catplot(x="bmi",y="charges",hue="smoker",data=df,kind="bar",aspect=2));
plt.rcParams.update({'font.size': 12})

plt.figure(figsize=(10,5))

corr=df.corr()

mask=np.zeros_like(corr,dtype=np.bool)

mask[np.triu_indices_from(mask)]=True

sns.heatmap(corr.abs(),annot=True,cmap="coolwarm",mask=mask);
plt.figure(figsize=(15,5))

plt.subplot(121)

sns.boxplot(df["charges"],color="y");

plt.subplot(122)

sns.boxplot(df["age"],color="y");
pd.crosstab(df.age,df.children)[:10]
df[(df["age"]==18)&(df["sex"]=="female")&(df["children"]>0)]
df[(df["age"]==18)&(df["sex"]=="male")&(df["children"]>0)]
clf=LocalOutlierFactor(n_neighbors=50)

clf.fit_predict(df[["age","children"]])
clf_scores=clf.negative_outlier_factor_
np.sort(clf_scores)[0:20]
treshold=np.sort(clf_scores)[20]
df[clf_scores<treshold]
df[(df["age"]==18)&(df["children"]>1)]
df.drop(df[(df["age"]==18)&(df["children"]>0)].index,inplace=True)
df.corr()
print(sns.catplot(x="children",y="charges",hue="smoker",data=df,kind="bar",aspect=3));
df.shape
df.head()
df_new=df.copy()

df_new=pd.get_dummies(data=df,columns=["sex","smoker"],drop_first=True)
df_new.head()
df_new=pd.get_dummies(data=df_new,columns=["region","bmi"])
df_new.head()
df_new.charges=log(df_new.charges)



sc=StandardScaler()

df_scaled=pd.DataFrame(sc.fit_transform(df_new),columns=df_new.columns,index=df_new.index)



df_scaled.head()
X=df_scaled.drop("charges",axis=1)

y=df_scaled["charges"] 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
lm=sm.OLS(y_train,X_train)

model=lm.fit()

model.summary()
X=df_scaled.drop(["charges","region_northwest"],axis=1)

y=df_scaled["charges"] 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lm=sm.OLS(y_train,X_train)

model=lm.fit()

model.summary()
X=df_scaled.drop(["charges","region_northwest","bmi_heavy"],axis=1)

y=df_scaled["charges"] 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lm=sm.OLS(y_train,X_train)

model=lm.fit()

model.summary()
X=df_scaled.drop(["charges","region_northwest","bmi_heavy","bmi_too_weak"],axis=1)

y=df_scaled["charges"] 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lm=sm.OLS(y_train,X_train)

model=lm.fit()

model.summary()
X=df_scaled.drop(["charges","region_northwest","bmi_heavy","bmi_too_weak","bmi_normal"],axis=1)

y=df_scaled["charges"] 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lm=sm.OLS(y_train,X_train)

model=lm.fit()

model.summary()
X=df_scaled.drop(["charges","region_northwest","bmi_heavy","bmi_too_weak","bmi_normal","region_northeast"],axis=1)

y=df_scaled["charges"] 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

lm=sm.OLS(y_train,X_train)

model=lm.fit()

model.summary()
model.params
model=LinearRegression()

lin_mo=model.fit(X_train,y_train)

y_pred=lin_mo.predict(X_test)
lin_mo.score(X_train,y_train)
lin_mo.score(X_test,y_test)
r2_score(y_test,y_pred)
plt.figure(figsize=(12,5));

ax1=sns.distplot(y_test,hist=False)

sns.distplot(y_pred,ax=ax1,hist=False);
plt.figure(figsize=(12,8));

residuals_plot(model, X_train, y_train, X_test, y_test,line_color="red");
plt.figure(figsize=(12,8));

prediction_error(model, X_train, y_train, X_test, y_test);
model.coef_
model.intercept_