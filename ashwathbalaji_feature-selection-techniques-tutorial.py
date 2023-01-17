import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



from sklearn.model_selection import train_test_split                          # Train test split

from sklearn.feature_selection import RFE                                     # RFE

from sklearn.metrics import r2_score,mean_squared_error                       # Metrics

from sklearn.linear_model import LinearRegression

from sklearn.datasets import load_boston



from mlxtend.feature_selection import SequentialFeatureSelector as sfs         # Forward & backward selection

from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs     # Ploting for forward & backward selection 
boston = load_boston()



#IV

X = pd.DataFrame(boston.data , columns=boston.feature_names)

#DV

y = pd.DataFrame(boston.target , columns=['PRICE'])



df = pd.concat([X,y],1)

df.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
plt.figure(figsize=(12,7))

sns.heatmap(df.corr(),vmax=0.9,annot=True)

plt.show()
corr = df.corr()

cor_target = abs(corr['PRICE'])  

imp_features_corr = cor_target[cor_target>=0.4]

imp_features_corr
#Linear Regression model

model = LinearRegression()



#Initialize RFE

rfe = RFE(model,5)

rfe.fit(X_train,y_train)
pd.DataFrame(list(zip(X.columns,rfe.support_,rfe.ranking_)),columns=['Features','Support','Rank']).T
y_pred_rfe = rfe.predict(X_test)



print('RMSE : ',np.sqrt(mean_squared_error(y_test,y_pred_rfe)))

print('R2_Score : ',r2_score(y_test,y_pred_rfe))
#model to train

model_tune = LinearRegression()

#keep track of metric

score_list=[]



for n in range(13):

    rfe = RFE(model_tune,n+1)

    rfe.fit(X_train,y_train)

    y_pred = rfe.predict(X_test)

    score = r2_score(y_test,y_pred)

    score_list.append(score)

    

    

plt.figure(figsize=(10,5))

plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13],score_list)

plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13])

plt.xlabel('Number of Features')

plt.ylabel('R2_Score')

plt.grid(True)

plt.show()
model = LinearRegression()



#Initialize RFE

rfe = RFE(model,9)

rfe.fit(X_train,y_train)

y_pred_rfe = rfe.predict(X_test)



print('RMSE : ',np.sqrt(mean_squared_error(y_test,y_pred_rfe)))

print('R2_Score : ',r2_score(y_test,y_pred_rfe))
model_1 = LinearRegression()

# Forward Selection 

sfs1 = sfs(model_1 , k_features=12 , forward=True , scoring='r2')

sfs1 = sfs1.fit(X_train,y_train)
fig = plot_sfs(sfs1.get_metric_dict())

plt.grid(True)

plt.show()
sfs1.k_feature_names_
model_1 = LinearRegression()

#Backward Elimination

sfs2 = sfs(model_1 , k_features=8 , forward=False , scoring='r2')

sfs2 = sfs2.fit(X_train,y_train)
fig = plot_sfs(sfs2.get_metric_dict())

plt.grid(True)

plt.show()
sfs2.k_feature_names_