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
import matplotlib.pyplot as plt #To plot the relation between dependent and independent variables.
import seaborn as sns
df=pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/car data.csv')
df.head()
df.describe()
df.info()
cat_features=[features for features in df.columns if len(df[features].unique())<=16 ]
cat_features
num_features=[features for features in df.columns if len(df[features].unique())>16]
num_features
for features in num_features:
    if features not in ['Car_Name']:
        sns.boxplot(df[features], orient='v')
        plt.show()
for features in cat_features:
    df[features].value_counts(normalize=True).sort_values(ascending=False).plot.bar()
    plt.xlabel(features)
    plt.ylabel('Numbers in each category')
    plt.show()
for features in cat_features:
    df.groupby(features)['Selling_Price'].median().sort_values(ascending=False).plot.bar()
    plt.ylabel('Selling Price')
    plt.xlabel(features)
    plt.show()
corrmatrix = df.corr()
top_corr_features = corrmatrix.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
for features in num_features:
    if features != 'Car_Name':
        sns.distplot(df[features])
        plt.show()
for features in cat_features:
    for features2 in cat_features:
        if features != 'Year' and features2!='Year':
            sns.barplot(x=df[features], y=df['Selling_Price'], data=df, hue=df[features2])
            plt.title('Relation between '+ features + '(at x-axis) and Selling Price at y-axis with hue= '+ features2)
            plt.legend(loc='upper right')
            plt.show()
for features in num_features:
    if features not in ['Car_Name','Selling_Price']:
        plt.scatter(df[features],df['Selling_Price'])
        plt.title('Relation between ' + features + ' and Selling_Price')
        plt.ylabel('Selling_Price')
        plt.xlabel(features)
        plt.show()
dfc=df.copy()
dfc['current_year']=2020
dfc.loc[:,'Year']=dfc['current_year']-dfc.loc[:,'Year']
dfc
dfc.drop(['Car_Name','Year','current_year'], axis=1, inplace=True)
dfc
dfc=pd.get_dummies(dfc, drop_first=True )
dfc
X=dfc.iloc[:,1:]
Y=dfc.iloc[:,0]
print('X shape:{} , Y shape:{}'.format(X.shape,Y.shape))
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_log_error
X_Train, X_Test, Y_Train, Y_Test= train_test_split(X, Y, test_size=0.33)
data_list=[[X_Train, Y_Train, X_Test, Y_Test]]

print('Shape of X_train: ' + str(X_Train.shape))
print('Shape of Y_train: ' + str(Y_Train.shape))
print('Shape of X_test:' + str(X_Test.shape))
print('Shape of Y_test:' + str(Y_Test.shape))
LR=LinearRegression()
RFR=RandomForestRegressor()
DTR=DecisionTreeRegressor()
SVRE=SVR()
MLPR=MLPRegressor()
LO=Lasso()
GBR=GradientBoostingRegressor()
model_list=[LR,RFR,DTR,SVRE,MLPR,LO,GBR]
i=-1
pred=[]
score_train=[]
score_test=[]
cv_score_train=[]
cv_score_test=[]
MSE=[]
model_name=['LR','RFR','DTR','SVRE','MLPR','Lasso','GBR']
score_name=['noise','train score','test score','cv train score','cv test score','MSE']
print('WITHOUT HYPERPARAMETER TUNING')
for model in model_list:
    i+=1
    for data in data_list:
        model.fit(data[0],data[1])
        pred.append((model.predict(data[2])))
        score_train.append(model.score(data[0],data[1]))
        score_test.append(model.score(data[2],data[3]))
        cv_score_train.append(np.mean(cross_val_score(model,data[0],data[1], cv=5)))
        cv_score_test.append(np.mean(cross_val_score(model,data[2],data[3], cv=5)))
        MSE.append(mean_squared_error(pred[i], data[3], squared=False))
        
#         print(model_name[i] +  ' has train score :' + str(np.round(score_train[i],3)) + ' , test score : ' + str(np.round(score_test[i],3))+ ',cv train score: '+ str(np.round(cv_score_train[i],3))+', cv test score: '+ str(np.round(cv_score_test[i],3)) + ', MSE:'+str(np.round(MSE[i],3)))

pd.DataFrame(data=[pred,score_train,score_test,cv_score_train,cv_score_test,MSE], columns=model_name , index=score_name).drop('noise')
Rfr=RandomForestRegressor()
param={'n_estimators':[3,5,50,200,400,800],'max_depth':[2,4,8,16,32,64,None]}
cv=GridSearchCV(Rfr, param, cv=5, n_jobs=-1)
cv.fit(X_Train,Y_Train)
print('Best params are {}'.format(cv.best_params_))
meanz=cv.cv_results_['mean_test_score']
stdz=cv.cv_results_['std_test_score']
paz=cv.cv_results_['params']
for mean,std,par in zip(meanz,stdz,paz):
    print( str(mean)+ '        '+ str(std)+ '        ' +str(par) +'\n' )
model_rfr=cv.best_estimator_

print('Train Score:{} and Test score:{}'.format(model_rfr.score(X_Train,Y_Train),model_rfr.score(X_Test,Y_Test)))
pred_rfr=model_rfr.predict(X_Test)
plt.scatter(Y_Test,pred_rfr)
sns.distplot(Y_Test-pred_rfr)
parms={'n_estimators':[50,100,200,500,1000],'max_depth':[5,10,15],'learning_rate':[0.05,0.1,0.5]}
cv=GridSearchCV(GBR,parms,cv=5)
cv.fit(X_Train, Y_Train)
print('Best params are {}'.format(cv.best_params_))
meanz=cv.cv_results_['mean_test_score']
stdz=cv.cv_results_['std_test_score']
paz=cv.cv_results_['params']
for mean,std,par in zip(meanz,stdz,paz):
    print( str(mean)+ '        '+ str(std)+ '        ' +str(par) +'\n' )
model_gbr=cv.best_estimator_
print('Train Score:{} and Test score:{}'.format(model_gbr.score(X_Train,Y_Train),model_gbr.score(X_Test,Y_Test)))
pred_gbr=model_rfr.predict(X_Test)
plt.scatter(Y_Test,pred_gbr)
sns.distplot(Y_Test-pred_gbr)



