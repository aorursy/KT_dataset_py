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

        train=pd.read_csv(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV



import os



# hide warnings

import warnings

warnings.filterwarnings('ignore')





train
from sklearn_pandas import CategoricalImputer

imputer = CategoricalImputer()

train['society']=imputer.fit_transform(train['society'])

train['location']=imputer.fit_transform(train['location'])

train['size']=imputer.fit_transform(train['size'])

train['balcony']=imputer.fit_transform(train['balcony'])

train['bath']=imputer.fit_transform(train['bath'])





text='Sq. Meter'

ind=list(train.loc[train.total_sqft.str.contains(text)].index)

for i in ind:

    train.at[i,'units']=train.at[i,'total_sqft'][-len(text):]

    train.at[i,'total_sqft']=train.at[i,'total_sqft'].replace('Sq. Meter','')

    
text='Sq. Yards'

ind=list(train.loc[train.total_sqft.str.contains(text)].index)

for i in ind:

    train.at[i,'units']=train.at[i,'total_sqft'][-len(text):]

    train.at[i,'total_sqft']=train.at[i,'total_sqft'].replace('Sq. Yards','')



text='Acres'

ind=list(train.loc[train.total_sqft.str.contains(text)].index)

for i in ind:

    train.at[i,'units']=train.at[i,'total_sqft'][-len(text):]

    train.at[i,'total_sqft']=train.at[i,'total_sqft'].replace(text,'')



text='Grounds'

ind=list(train.loc[train.total_sqft.str.contains(text)].index)

for i in ind:

    train.at[i,'units']=train.at[i,'total_sqft'][-len(text):]

    train.at[i,'total_sqft']=train.at[i,'total_sqft'].replace(text,'')



    
text='Perch'

ind=list(train.loc[train.total_sqft.str.contains(text)].index)

for i in ind:

    train.at[i,'units']=train.at[i,'total_sqft'][-len(text):]

    train.at[i,'total_sqft']=train.at[i,'total_sqft'].replace(text,'')



    
text='Cents'

ind=list(train.loc[train.total_sqft.str.contains(text)].index)

for i in ind:

    train.at[i,'units']=train.at[i,'total_sqft'][-len(text):]

    train.at[i,'total_sqft']=train.at[i,'total_sqft'].replace(text,'')



    
text='Guntha'

ind=list(train.loc[train.total_sqft.str.contains(text)].index)

for i in ind:

    train.at[i,'units']=train.at[i,'total_sqft'][-len(text):]

    train.at[i,'total_sqft']=train.at[i,'total_sqft'].replace(text,'')
train.groupby('area_type')['price'].mean().plot(kind='bar')
train=train[~train['total_sqft'].str.contains('-')]

train.loc[train['availability']!='Ready To Move','availability']='Under Construction'
train.units.fillna('Sq. Foot',inplace=True)

Unit={'Sq. Foot':1,'Sq. Meter':10.7639,'Sq. Yards':9,'Acres':43560,'Cents':435.6,'Guntha':1089,'Grounds':2400,'Perch':272.25,}

train.units=train.units.replace(Unit)
def conv(x):

    

        return str(x).replace('Bedroom','BHK')



train['size']=train['size'].apply(conv)
train['total_sqft']=train.total_sqft.astype(float)

train.total_sqft=train.total_sqft*train.units
df_list_train = list(train.dtypes[train.dtypes !='object'].index)

def remove_outliers(df,df_):

    list = []

    for col in df_:

        Q1 = df[col].quantile(.25)

        Q3 = df[col].quantile(.99)

        IQR = Q3-Q1

        df =  df[(df[col] >= (Q1-(1.5*IQR))) & (df[col] <= (Q3+(1.5*IQR)))] 

    return df   



train_df = remove_outliers(train,df_list_train)
import matplotlib.pyplot as plt

plt.figure(figsize=(20,20))

plt.scatter(train_df['total_sqft'],train_df['price'])

plt.show()
train_df
import seaborn as sns

plt.figure(figsize=(20,20))

sns.boxplot(train_df['bath'],train_df['price'])

#train_df.groupby(['area_type','size'])['price'].mean().plot(kind='bar')

plt.show()
train_df
train_df.groupby('availability')['price'].mean().plot(kind='bar')


train_df.drop(['units','society'],inplace=True,axis=1)


cols_train=train_df.select_dtypes(include=['object'])





dummy_train=pd.get_dummies(train_df[cols_train.columns],drop_first=True)

train_df=pd.concat([train_df,dummy_train],axis=1)

train_df.drop(cols_train,axis=1,inplace=True)



train_df
kde_col=train_df.select_dtypes(['int','float64'])

for co in kde_col:

    plt.figure(figsize=(5,5))

    sns.distplot(train_df[co])

    plt.show()
train_df['total_sqft']=np.log(train_df['total_sqft'])

train_df['price']=np.log(train_df['price'])
kde_col=train_df.select_dtypes(['int','float64'])

for co in kde_col:

    plt.figure(figsize=(5,5))

    sns.distplot(train_df[co])

    plt.show()
train_df
#train_df['bath']=train_df['bath'].astype('object')

#train_df['balcony']=train_df['balcony'].astype('object')
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

train_df['total_sqft']=scaler.fit_transform(pd.DataFrame(train_df['total_sqft']))

from sklearn.linear_model import LinearRegression

lr=LinearRegression()



y_train=train_df['price']

X_train=train_df.drop('price',axis=1)

lm=lr.fit(X_train,y_train)
import statsmodels.api as sm 

X_train_lm = sm.add_constant(X_train)

lm = sm.OLS(X_train_lm,y_train).fit() 
from sklearn.decomposition import PCA

pca1=PCA(n_components=600,random_state=41)

pca1.fit(X_train)
def get_pca_components(pca1,X_train):



    n_pcs= pca1.n_components





    most_important = [np.abs(pca1.components_[i]).argmax() for i in range(n_pcs)]



    initial_feature_names = X_train.columns



    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]





    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}





    df_pca1 = pd.DataFrame(dic.items())

    return list(df_pca1[1])
df1=get_pca_components(pca1,X_train_lm)
var_cum=np.cumsum(pca1.explained_variance_ratio_)

plt.figure()

plt.plot(var_cum)

plt.show()
# list of alphas to tune

params = {'alpha': [0.000001, 0.000005, 0.000009, 0.00005, 0.00009, 

 0.0001, 0.0003, 0.0004 ,.0005,.0009,.001,.005,.009,.01,1,2,5,10]}





ridge = Ridge()



# cross validation

folds = 5

model_cv = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'r2', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

model_cv.fit(X_train_lm[df1], y_train) 

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=200]

cv_results.head()

# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')





# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

#plt.ylim(0,1)

plt.show()
alpha = 1

ridge = Ridge(alpha=alpha)



ridge.fit(X_train_lm, y_train)

ridge.coef_
params = {'alpha': [0.000001, 0.000005, 0.000009, 0.00005, 0.00009, 

 0.0001, 0.0003, 0.0004 ,.0005,.0009,.001,.005,.009,.01,1,2]}















lasso = Lasso()



# cross validation

model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'r2', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

folds = 3

model_cv.fit(X_train_lm[df1], y_train) 

cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results = cv_results

cv_results.head()



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('R2')



plt.title("R2 and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
alpha =.0001



lasso = Lasso(alpha=alpha)

        

lasso.fit(X_train_lm, y_train) 

print(lasso.score(X_train_lm,y_train))
lasso_train_score = lasso.score(X_train_lm,y_train)





#print(lasso_train_score,lasso_test_score)

rdge_train=ridge.score(X_train_lm,y_train)

#rdge_test=ridge.score(X_test,y_test)

print(rdge_train,lasso_train_score)
y_predict = lasso.predict(X_train_lm)
from sklearn.metrics import r2_score

r2_score(y_train,y_predict)
import xgboost as xgb

from xgboost import XGBRegressor

from xgboost import plot_importance

from sklearn.metrics import r2_score
xgb1 = XGBRegressor()

xgb1.fit(X_train_lm, y_train)

y_train_pred=xgb1.predict(X_train_lm)

#y_test_pred_xgb=xgb1.predict(X_test)

roc = r2_score(y_train, y_train_pred)

print("R2: %.2f" % (roc ))







# creating a KFold object 

folds = 3



# specify range of hyperparameters

param_grid = {'learning_rate': [.1,.5,.7,1], 

             'subsample': [0.3, 0.6, 0.9],          

             'max_depth':[1,2,3,4]}



# specify model

xgb_model = XGBRegressor(max_depth=2)



# set up GridSearchCV()

model_cv = GridSearchCV(estimator = xgb_model, 

                        param_grid = param_grid, 

                        scoring= 'r2', 

                        cv = folds, 

                        verbose = 1,

                        return_train_score=True)      





model_cv.fit(X_train, y_train) 

cv_results = pd.DataFrame(model_cv.cv_results_)

print(cv_results)

plt.figure(figsize=(16,6))



param_grid = {'learning_rate': [.1,.5,.7,1], 

             'subsample': [0.3, 0.6, 0.9],

              'max_depth':[1,2,3,4]}



for n, subsample in enumerate(param_grid['subsample']):

    



    # subplot 1/n

    plt.subplot(1,len(param_grid['subsample']), n+1)

    df = cv_results[cv_results['param_subsample']==subsample]



    plt.plot(df["param_learning_rate"], df["mean_test_score"])

    plt.plot(df["param_learning_rate"], df["mean_train_score"])

    plt.xlabel('learning_rate')

    plt.ylabel('R2')

    plt.title("subsample={0}".format(subsample))

    

    plt.legend(['test score', 'train score'], loc='upper left')

    







importance = dict(zip(X_train.columns, xgb1.feature_importances_))



fig, ax = plt.subplots(figsize=(10, 10))

plot_importance(xgb1, ax=ax)
from sklearn.ensemble import RandomForestRegressor

rfc=RandomForestRegressor()

rfc.fit(X_train,y_train)

rfc_pred=rfc.predict(X_train)

rf_metrics=r2_score(y_train,rfc_pred)

rf_metrics
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 3



# parameters to build the model on

parameters = {'ccp_alpha': [.00001,.00002,0.00005,.00009,.0003,.0006,.0009,.001,1,2]}



# instantiate the model

rf = RandomForestRegressor()





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="r2")

rf.fit(X_train, y_train)

cv_results = pd.DataFrame(model_cv.cv_results_)

#print(cv_results)

scores = rf.cv_results_

pd.DataFrame(scores).head()

plt.figure()

plt.plot(scores["param_ccp_alpha"], 

         scores['mean_score_time'], 

         label="training accuracy")

plt.plot(scores["param_ccp_alpha"], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("alpha")

plt.ylabel("R2")

plt.legend()

plt.show()









from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV





# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

parameters = {'n_estimators': range(1, 20, 1)}



# instantiate the model (note we are specifying a max_depth)

rf = RandomForestRegressor(max_depth=15)





# fit tree on training data

rf = GridSearchCV(rf, parameters, 

                    cv=n_folds, 

                   scoring="r2")

rf.fit(X_train, y_train)



scores = rf.cv_results_

pd.DataFrame(scores).head()









plt.figure()

plt.plot(scores["param_n_estimators"], 

         scores['mean_score_time'], 

         label="training accuracy")

plt.plot(scores['param_n_estimators'], 

         scores["mean_test_score"], 

         label="test accuracy")

plt.xlabel("n_estimators")

plt.ylabel("R2")

plt.legend()

plt.show()







from sklearn.linear_model import SGDRegressor

from sklearn.ensemble import StackingRegressor





estimators=[

     ('lr', Lasso(alpha=0.0001)),

    #('sg',SGDRegressor()),

    

    

     ('rf',RandomForestRegressor(n_estimators=10,max_depth=10))

     

]

stc=StackingRegressor(estimators)

stc.fit(X_train,y_train)

stc_pred=stc.predict(X_train)

stc=r2_score(y_train,rfc_pred)

stc