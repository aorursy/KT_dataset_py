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
df=pd.read_csv('/kaggle/input/nba-players-data/all_seasons.csv').drop('Unnamed: 0',axis=1)

df
df.shape
#checking for null values

df.dropna(inplace=True)

df.shape
df.info()
df_season_wise = df.set_index('season')



#Set undrafted to null

Undrafted= df_season_wise[df_season_wise['draft_year']=='Undrafted']

df_season_wise['draft_year']=df_season_wise['draft_year'].replace('Undrafted',np.NaN) 

df_season_wise['draft_round']=df_season_wise['draft_round'].replace('Undrafted',np.NaN)

df_season_wise['draft_number']=df_season_wise['draft_number'].replace('Undrafted',np.NaN)

df_season_wise
import matplotlib.pyplot as plt

import numpy as np



#selecting columns required for analysis

col_need=['age','player_height','player_weight','gp','pts','reb','ast','net_rating','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct']

ana_df=df[col_need]

ana_df
#checking for duplicate values; False=> NO duplicated values

ana_df.duplicated().values.any()
#co-relation matrix showing correlation between features

ana_df.corr()
#visualizing relation between features

pd.plotting.scatter_matrix(ana_df,figsize=(20,20),alpha=0.5);
#better visualization

import seaborn as sns

sns.pairplot(ana_df)
#plotting all other features correlation with net_rating

fig,((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8),(ax9,ax10,ax11,ax12))=plt.subplots(3,4,figsize=(30,10))

ax=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12]

k=0



for col in ana_df.columns:

    if col!='net_rating':

        ax[k].scatter(ana_df[col],ana_df['net_rating'])

        ax[k].set_xlabel(col)

        ax[k].set_ylabel("net_rating")

        k=k+1

#predicting net rating dependence on different individual features 



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split





X=['age','player_height','player_weight','gp','pts','reb','ast','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct']

y=['net_rating']



l=[]

for x in X:

    m=np.array(ana_df[x])

    n=np.array(ana_df[y])

    X_train,X_test,y_train,y_test=train_test_split(m.reshape(-1,1),n.reshape(-1,1),random_state=0)

    model=LinearRegression().fit(X_train,y_train)

    print(model.score(X_train,y_train),'  ',model.score(X_test,y_test))

    l.append((model.score(X_test,y_test),x))

    

# returns max model score of the feature that best predicts the net rating of the player

max(l)
#Multiple Linear Regression

#Polynomial Regression

#Decision Tree Regressor
#Multiple Linear Regression

from sklearn.metrics import r2_score

X_m=['age','player_height','player_weight','gp','pts','reb','ast','oreb_pct','dreb_pct','usg_pct','ts_pct','ast_pct']

y_m=['net_rating']



Xm_train,Xm_test,ym_train,ym_test=train_test_split(ana_df[X_m],ana_df[y_m])



mul_model=LinearRegression().fit(Xm_train,ym_train)

print('training multiple feature model score:',mul_model.score(Xm_train,ym_train),'\ntesting multiple feature model score:',mul_model.score(Xm_test,ym_test),

      '\ntraining multiple feature model r2 score:',r2_score(ym_train,mul_model.predict(Xm_train)),'\ntesting multiple feature model r2 score:',r2_score(ym_test,mul_model.predict(Xm_test)))



#cross validation

from sklearn.model_selection import cross_val_score

cv_score=cross_val_score(mul_model,ana_df[X_m],ana_df[y_m])

print('cross validation score:',cv_score)

print('mean cross validation score:',np.mean(cv_score))
#Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures



poly_X_m=PolynomialFeatures(degree=2).fit_transform(ana_df[X_m])

y_m=['net_rating']



poly_X_train,poly_X_test,y_trainn,y_testt=train_test_split(poly_X_m,ana_df[y_m])



poly_model= LinearRegression().fit(poly_X_train,y_trainn)



print('training polynomial model score:',poly_model.score(poly_X_train,y_trainn),'\ntesting polynomial model score:',poly_model.score(poly_X_test,y_testt),

      '\ntraining polynomial model r2 score:',r2_score(y_trainn,poly_model.predict(poly_X_train)),'\ntesting polynomial model r2 score:',r2_score(y_testt,poly_model.predict(poly_X_test)))



#cross validation

cv_score=cross_val_score(poly_model,poly_X_m,ana_df[y_m])

print('cross validation score:',cv_score)

print('mean cross validation score:',np.mean(cv_score))
#Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor



tree_model=  DecisionTreeRegressor(max_depth=1).fit(Xm_train,ym_train)



print('training decision tree model score:',tree_model.score(Xm_train,ym_train),'\ntesting decision tree model score:',tree_model.score(Xm_test,ym_test),

      '\ntraining decision tree model r2 score:',r2_score(ym_train,tree_model.predict(Xm_train)),'\ntesting decision tree model r2 score:',r2_score(ym_test,tree_model.predict(Xm_test)))



#cross validation

cv_score=cross_val_score(tree_model,ana_df[X_m],ana_df[y_m])

print('cross validation score:',cv_score)

print('mean cross validation score:',np.mean(cv_score))
#KNN Regression

from sklearn.neighbors import KNeighborsRegressor

knn_model=KNeighborsRegressor(n_neighbors=1000).fit(Xm_train,ym_train)

print('training knn model score:',knn_model.score(Xm_train,ym_train),'\ntesting knn model score:',knn_model.score(Xm_test,ym_test),

      '\ntraining knn model r2 score:',r2_score(ym_train,knn_model.predict(Xm_train)),'\ntesting knn model r2 score:',r2_score(ym_test,knn_model.predict(Xm_test)))



#cross validation

cv_score=cross_val_score(knn_model,ana_df[X_m],ana_df[y_m])

print('cross validation score:',cv_score)

print('mean cross validation score:',np.mean(cv_score))
#Hence, for this dataset, predicting the net_rating of a player depending on features available in dataset, best model is: 

#Polynomial Regression accompanied by linear regression

#However
#checking outliers in Measure of the player's shooting efficiency 

plt.figure()

_=plt.boxplot(ana_df['ts_pct'],whis='range')
#checking outliers in Average number of rebounds grabbed

plt.figure()

_=plt.boxplot(ana_df['reb'],whis='range')
#How does height and weight of a player can be used to predict his shooting efficiency

from sklearn.metrics import r2_score

from mpl_toolkits.mplot3d import Axes3D

%matplotlib notebook



col_shoot=['player_height','player_weight','ts_pct']

df_shoot= ana_df[col_shoot]



X_sht=['player_height','player_weight']

y_sht=['ts_pct']



X_sht_train,X_sht_test,y_sht_train,y_sht_test= train_test_split(df_shoot[X_sht],df_shoot[y_sht])



#Simple Linear Regression model

linear_model_sht=LinearRegression().fit(X_sht_train,y_sht_train)

print('training_set score:',linear_model_sht.score(X_sht_train,y_sht_train),'\ntest_set score:',linear_model_sht.score(X_sht_test,y_sht_test),

      '\ntraining_set r2_score:',r2_score(y_sht_train,linear_model_sht.predict(X_sht_train)),'\ntest_set r2_score:',r2_score(y_sht_test,linear_model_sht.predict(X_sht_test)))

predict=linear_model_sht.predict(X_sht_test)

plt.figure(figsize=(10,10))

plt.subplot(411)

sns.scatterplot(df_shoot['player_height'],df_shoot['player_weight'],alpha=0.5)



plt.subplot(412)

sns.scatterplot(df_shoot['player_height'],df_shoot['ts_pct'],alpha=0.5)

plt.plot(X_sht_test,linear_model_sht.predict(X_sht_test))

plt.legend(['height','weight'])

plt.subplot(413)

sns.scatterplot(df_shoot['player_weight'],df_shoot['ts_pct'],alpha=0.5)



plt.plot(X_sht_test,linear_model_sht.predict(X_sht_test))

plt.legend(['height','weight'])

sns.pairplot(df_shoot)
#Ridge Regression model

from sklearn.linear_model import Ridge

ridge_model_sht=Ridge(alpha=709999).fit(X_sht_train,y_sht_train)

print('training_set score:',ridge_model_sht.score(X_sht_train,y_sht_train),'\ntest_set score:',ridge_model_sht.score(X_sht_test,y_sht_test),

      '\ntraining_set r2_score:',r2_score(y_sht_train,ridge_model_sht.predict(X_sht_train)),'\ntest_set r2_score:',r2_score(y_sht_test,ridge_model_sht.predict(X_sht_test)))
#How does height and weight of a player can be used to predict whether he'll be able to take rebound

from sklearn.metrics import r2_score

from mpl_toolkits.mplot3d import Axes3D

%matplotlib notebook



col_rebound=['player_height','player_weight','reb']

df_rebound= ana_df[col_rebound]



X_reb=['player_height','player_weight']

y_reb=['reb']



X_reb_train,X_reb_test,y_reb_train,y_reb_test= train_test_split(df_rebound[X_reb],df_rebound[y_reb])



#Simple Linear Regression model

linear_model_reb=LinearRegression().fit(X_reb_train,y_reb_train)

print('training_set score:',linear_model_reb.score(X_reb_train,y_reb_train),'\ntest_set score:',linear_model_reb.score(X_reb_test,y_reb_test),

      '\ntraining_set r2_score:',r2_score(y_reb_train,linear_model_reb.predict(X_reb_train)),'\ntest_set r2_score:',r2_score(y_reb_test,linear_model_reb.predict(X_reb_test)))

plt.subplot(311)

plt.scatter(df_rebound['player_height'],df_rebound['player_weight'],alpha=0.5)

plt.subplot(312)

plt.scatter(df_rebound['player_height'],df_rebound['reb'],alpha=0.5)

plt.subplot(313)

plt.scatter(df_rebound['player_weight'],df_rebound['reb'],alpha=0.5)

#Ridge Regression model

from sklearn.linear_model import Ridge

ridge_model_reb=Ridge(alpha=10).fit(X_reb_train,y_reb_train)

print('training_set score:',ridge_model_reb.score(X_reb_train,y_reb_train),'\ntest_set score:',ridge_model_reb.score(X_reb_test,y_reb_test),

      '\ntraining_set r2_score:',r2_score(y_reb_train,ridge_model_reb.predict(X_reb_train)),'\ntest_set r2_score:',r2_score(y_reb_test,ridge_model_reb.predict(X_reb_test)))