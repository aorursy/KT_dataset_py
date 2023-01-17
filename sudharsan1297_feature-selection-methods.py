# Importing the necessary libraries 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.datasets import load_boston
# Loading the Boston dataset

# Independent features

boston=load_boston()



df=pd.DataFrame(boston.data,columns=boston.feature_names)
# Dependent feature

df['Price']=boston.target
# Checking for null values

df.info()
# Splitting X and y

X=df.drop('Price',1)

y=df['Price']
plt.subplots(figsize=(20,7))

sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
corr=df.corr()

cor_target=abs(corr['Price'])

imp_features=cor_target[cor_target>=0.4]

imp_features
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model=LinearRegression()



rfe=RFE(model,10)

rfe.fit(X_train,y_train)
print(rfe.support_)

print(rfe.ranking_)
# Creating a dataframe to see which of the features are included and which ones are not

pd.DataFrame(list(zip(X.columns,rfe.support_,rfe.ranking_)),columns=['Features','Support','Rank']).T
# Builing a RFE model using the features included. This model is not the actual number of features to include but is a

# rough number lesser than the number of features

y_pred=rfe.predict(X)

print(r2_score(y,y_pred))

print(np.sqrt(mean_squared_error(y,y_pred)))
nof_cols=np.arange(1,14)



from sklearn.model_selection import train_test_split
# Getting the optimal number of features by getting the r-squared value for different number of features.

score_list=[]



model=LinearRegression()



for i in range(13):

    rfe=RFE(model,i+1)

    rfe.fit(X_train,y_train)

    y_pred=rfe.predict(X_test)

    score=r2_score(y_test,y_pred)

    score_list.append(score)
plt.plot(nof_cols,score_list)

plt.xticks(np.arange(0,14))

plt.grid()
# We again build a RFE model, but this time with 9 as the number of features.

rfe=RFE(model,9)

rfe.fit(X_train,y_train)

pd.DataFrame(list(zip(X.columns,rfe.support_,rfe.ranking_)),columns=['Features','Support','Rank']).T
y_pred_rfe=rfe.predict(X_test)

print(r2_score(y_test,y_pred_rfe))
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
model_1 = LinearRegression()
# We first opt for all the features to find out the optimal number of features

sfs1 = sfs(model_1 , k_features=13 , forward=True , scoring='r2')

sfs1 = sfs1.fit(X_train,y_train)

fig = plot_sfs(sfs1.get_metric_dict())

plt.grid(True)

plt.show()

sfs1 = sfs(model_1 , k_features=9 , forward=True , scoring='r2')

sfs1 = sfs1.fit(X_train,y_train)

sfs1.k_feature_names_
X_train1=X_train[['CRIM', 'CHAS', 'NOX', 'RM', 'DIS', 'RAD', 'PTRATIO', 'B', 'LSTAT']]

X_test1=X_test[['CRIM', 'CHAS', 'NOX', 'RM', 'DIS', 'RAD', 'PTRATIO', 'B', 'LSTAT']]
lr=LinearRegression().fit(X_train,y_train)

y_pred_fs=lr.predict(X_test)

r2_score(y_test,y_pred_fs)
sfs2=sfs(model_1,k_features=1,forward=False,scoring='r2')

sfs2 = sfs2.fit(X_train,y_train)

fig = plot_sfs(sfs2.get_metric_dict())

plt.grid(True)

plt.show()

lr=LinearRegression().fit(X_train,y_train)

y_pred_be=lr.predict(X_test)

r2_score(y_test,y_pred_be)