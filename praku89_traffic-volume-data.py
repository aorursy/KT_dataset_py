# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Generic library

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
train_df=pd.read_csv("../input/Train.csv")

test_df=pd.read_csv("../input/Test.csv")

submission_df=pd.read_csv("../input/sample_submission.csv")
print("Train size : rows",train_df.shape[0]," and columns",train_df.shape[1])

print("Test size : rows",test_df.shape[0]," and columns",test_df.shape[1])

print("Submission size : rows",submission_df.shape[0]," and columns",submission_df.shape[1])
train_df.columns
train_df.columns.difference(test_df.columns)
train_df["source"] = "train"

test_df["source"] = "test"

df = pd.concat([train_df,test_df])
df.dtypes
df.shape
df.describe().transpose()
df.duplicated().sum()
df.isna().sum()
df.head(5)
# Identify discrete and continous columns

col_disc=[]

col_medium=[]

col_cont=[]

print("Attributes with their distinct count and their classification")

for i in df.columns:

    if df[i].nunique() <=10:

        print(i,"==",df[i].nunique(),"== disc")

        col_disc.append(i)

    elif (df[i].nunique() >10 and df[i].nunique() <100):

        col_medium.append(i)    

        print(i,"==",df[i].nunique(),"== medium")

    else:

        col_cont.append(i)

        print(i,"==",df[i].nunique(),"== cont")
for i in col_disc:

    print(i ,"with distinct values \n",df[i].unique())

for i in col_medium:

    print(i ,"with distinct values \n",df[i].unique())
df.corr()
df.cov()
df.corr()['traffic_volume'].sort_values()
col_cont_key_df=df[col_cont]

num_cont_list=col_cont_key_df.columns.drop("date_time")

sns.pairplot(data=df,vars=num_cont_list,hue='weather_type')
col_cont_key_df=df[col_cont]

num_cont_list=col_cont_key_df.columns.drop("date_time")

sns.pairplot(data=df,vars=num_cont_list,hue='is_holiday')



plt.figure(figsize=(20, 10))

sns.boxplot(data=df,orient="h")
agg_func=dict(Count='count',Avg='mean',Median='median',Deviation='std',Min='min',Max='max')

df.groupby("weather_type").agg({

        'traffic_volume': agg_func,

    }).sort_values(('traffic_volume', 'Count'))
plt.figure(figsize=(30, 20))

sns.boxplot(data=df,x=df["weather_type"],y=df["traffic_volume"])
agg_func=dict(Count='count',Avg='mean',Median='median',Deviation='std',Min='min',Max='max')

df.groupby("is_holiday").agg({

        'traffic_volume': agg_func,

    }).sort_values(('traffic_volume', 'Count'))
plt.figure(figsize=(30, 20))

sns.boxplot(data=df,x=df["is_holiday"],y=df["traffic_volume"])
agg_func=dict(Count='count',Avg='mean',Median='median',Deviation='std',Min='min',Max='max')

df.groupby("weather_description").agg({

        'traffic_volume': agg_func,

    }).sort_values(('traffic_volume', 'Count'))
pd.crosstab(df["weather_type"],df["is_holiday"], values=df.traffic_volume, aggfunc='median',dropna=False,margins=True,margins_name="Total Median")
agg_func=dict(Count='count',Avg='mean',Median='median',Deviation='std',Min='min',Max='max')

df.groupby(["is_holiday","weather_type"]).agg({

        'traffic_volume': agg_func,

    }).sort_values(('traffic_volume', 'Count'))
agg_func=dict(Count='count',Avg='mean',Median='median',Deviation='std',Min='min',Max='max')

df.groupby(["weather_type","weather_description"]).agg({

        'traffic_volume': agg_func,

    }).sort_values(('traffic_volume', 'Count'))
# one-hot encoding

cat_col=['weather_type','is_holiday','weather_description']

one_hot=pd.get_dummies(df[cat_col])

traffic_vol_procsd_df=pd.concat([df,one_hot],axis=1)
one_hot['traffic_volume']=df['traffic_volume']

one_hot.corr()['traffic_volume'].sort_values()
traffic_dt_df=traffic_vol_procsd_df.drop(columns=['weather_type','is_holiday','weather_description'])

traffic_dt_df.set_index('date_time',inplace=True)

traffic_dt_df.head(10)
train_final = traffic_dt_df[traffic_dt_df.source=="train"]

test_final = traffic_dt_df[traffic_dt_df.source=="test"]
train_final.drop(columns="source",inplace=True)

test_final.drop(columns="source",inplace=True)

#split train and data

train_X = train_final.drop(columns=["traffic_volume"])

train_Y = train_final["traffic_volume"]

test_X = test_final.drop(columns=["traffic_volume"])

test_Y = test_final["traffic_volume"]
train_final[train_final["traffic_volume"]==0]
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import cross_val_score



model_metrics={}

def calc_metrics_and_predict(model,model_label,train_X,test_X,train_Y,test_Y,test_final, scaler,cv):

    if scaler is not None:

        # Scaling

        print("Scaling applied")

        train_X = scaler.fit_transform(train_X)

        test_X = scaler.transform(test_X)

    print("fit data")    

    model.fit(train_X, train_Y)

    print("predict data")

    yhat_train = model.predict(train_X)

    rmse=np.sqrt(mean_squared_error(train_Y, yhat_train))

    mae=mean_absolute_error(train_Y,yhat_train)

    #mape=np.mean(np.abs((train_Y - yhat_train) / train_Y)) * 100

    mape=mae * 100

    if cv==True:

        print("Cross-Validation score")

        scores = cross_val_score(model, train_X, train_Y, cv = 10,scoring='neg_mean_squared_error') 

        avg_cross_val_score = np.mean(np.sqrt(np.abs(scores)))

    else:

        avg_cross_val_score=None

    #avg_cross_val_score=0

    model_metrics[model_label]=[rmse,mae,mape,avg_cross_val_score]

    # Predict test data and add to test_fnal dataframe

    test_final[model_label] =  model.predict(test_X)

    return model_metrics
# Linear regression

from sklearn.linear_model import LinearRegression



linear_model = LinearRegression()

        

model_metrics=calc_metrics_and_predict(linear_model,'linear_reg',train_X,test_X,train_Y,test_Y,test_final,None,True)

model_metrics
# linear model with cross-validation

from sklearn.model_selection import cross_val_score

a = cross_val_score(linear_model, train_X, train_Y, cv=10, scoring='neg_mean_squared_error')

np.mean(np.sqrt(np.abs(a)))
# Regularization technique

from sklearn.linear_model import Ridge, Lasso 



# List to maintain the different cross-validation scores 

cross_val_scores_ridge = [] 

  

# List to maintain the different values of alpha 

alpha = [] 

  

# Loop to compute the different values of cross-validation scores 

for i in range(1, 9): 

    ridgeModel = Ridge(alpha = i * 0.25) 

    ridgeModel.fit(train_X, train_Y) 

    scores = cross_val_score(ridgeModel, train_X, train_Y, cv = 10,scoring='neg_mean_squared_error') 

    avg_cross_val_score = np.mean(np.sqrt(np.abs(scores)))

    cross_val_scores_ridge.append(avg_cross_val_score) 

    alpha.append(i * 0.25) 

  

# Loop to print the different values of cross-validation scores 

for i in range(0, len(alpha)): 

    print(str(alpha[i])+' : '+str(cross_val_scores_ridge[i])) 
#Building ridge model

ridgeModel = Ridge(alpha = 1 * 0.25) 

model_metrics=calc_metrics_and_predict(ridgeModel,'regularization',train_X,test_X,train_Y,test_Y,test_final,None,True)

model_metrics
# KNN implementation

#from sklearn.neighbors import KNeighborsRegressor

#from sklearn.preprocessing import StandardScaler

#from sklearn.preprocessing import PolynomialFeatures

#from math import sqrt





# Scaling

#train_scaled_X = StandardScaler().fit_transform(train_X)

      



#df_len=round(sqrt(len(traffic_dt_df)))

#Train Model and Predict  

#for k in range(3,7):

#    neigh = KNeighborsRegressor(n_neighbors = k).fit(train_scaled_X,train_Y)

#    yhat_train = neigh.predict(train_scaled_X)

#    train_rmse=sqrt(mean_squared_error(train_Y,yhat_train))

#    print("RMSE for train : ",train_rmse," with k =",k)

    
#Predict on testing data:

from sklearn.neighbors import KNeighborsRegressor



neigh = KNeighborsRegressor(n_neighbors = 3)

model_metrics=calc_metrics_and_predict(neigh,'KNN',train_X,test_X,train_Y,test_Y,test_final,StandardScaler(),True)

model_metrics
# Gaussian NB

from sklearn.naive_bayes import GaussianNB



NB=GaussianNB()



model_metrics=calc_metrics_and_predict(NB,'NB_Gaussian',train_X,test_X,train_Y,test_Y,test_final,None,True)

model_metrics





# MultinomialNB

from sklearn.naive_bayes import MultinomialNB



NB=MultinomialNB()



model_metrics=calc_metrics_and_predict(NB,'NB_MultinomialNB',train_X,test_X,train_Y,test_Y,test_final,None,True)

model_metrics





# Bernoulli

from sklearn.naive_bayes import BernoulliNB



BL=BernoulliNB()



model_metrics=calc_metrics_and_predict(BL,'NB_BernoulliNB',train_X,test_X,train_Y,test_Y,test_final,None,True)

model_metrics

from sklearn.ensemble import RandomForestRegressor



rfm=RandomForestRegressor(random_state = 0, criterion='mse',n_jobs = -1, 

        n_estimators = 100, max_depth = None,min_samples_leaf=1,min_samples_split=2)



model_metrics=calc_metrics_and_predict(rfm,'Random_forest',train_X,test_X,train_Y,test_Y,test_final,None,True)

model_metrics
# Decision Tree regressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import make_scorer

from sklearn.model_selection import GridSearchCV



# Choose the type of classifier. 

clf = DecisionTreeRegressor()





# Choose some parameter combinations to try

parameters = {'criterion' : ['mse','mae','friedman_mse'],

              'max_features': ['log2', 'sqrt','auto'],

              'max_depth': range(2,16,2), 

              'min_samples_split': range(2,16,2),

              'min_samples_leaf': range(2,16,2)             



             }









#print("Grid search started")

#start_time = time.time()

# Run the grid search

#grid_obj = GridSearchCV(clf, parameters, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

#grid_obj = grid_obj.fit(train_X, train_Y)

#elapsed_time = time.time() - start_time

#print(elapsed_time)

# Set the clf to the best combination of parameters

#clf = grid_obj.best_estimator_

#clf
from sklearn.tree import DecisionTreeRegressor

clf=DecisionTreeRegressor(criterion='mse', max_depth=100, max_features='log2', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=5, min_samples_split=8, min_weight_fraction_leaf=0.0, presort=False, random_state=None, splitter='best')





model_metrics=calc_metrics_and_predict(clf,'Decision_Tree',train_X,test_X,train_Y,test_Y,test_final,None,True)

model_metrics
# Ada boost classifier

from sklearn.ensemble import AdaBoostClassifier,AdaBoostRegressor

#dtree = DecisionTreeClassifier(criterion='',max_depth=1)



adabst_fit = AdaBoostRegressor(base_estimator= clf,n_estimators=5000,learning_rate=0.05,random_state=42)



model_metrics=calc_metrics_and_predict(clf,'Adaboost',train_X,test_X,train_Y,test_Y,test_final,None,True)

model_metrics
# Gradientboost Classifier

from sklearn.ensemble import GradientBoostingRegressor



gbc_fit = GradientBoostingRegressor(loss='quantile',learning_rate=0.05,n_estimators=200,min_samples_split=8,min_samples_leaf=5,max_depth=100,random_state=42,max_features='log2')



#gbc_fit = GradientBoostingRegressor(base_estimator= clf,n_estimators=5000,learning_rate=0.05,random_state=42)



    

model_metrics=calc_metrics_and_predict(gbc_fit,'Gradient_boosting',train_X,test_X,train_Y,test_Y,test_final,None,False)

model_metrics
# Xgboost Classifier

import xgboost as xgb



xgb_fit = xgb.XGBRegressor(learning_rate=0.05,n_estimators=100,min_samples_split=8,min_samples_leaf=5,max_depth=100,random_state=42,max_features='log2' )







model_metrics=calc_metrics_and_predict(xgb_fit,'XGBoosting',train_X,test_X,train_Y,test_Y,test_final,None,False)

model_metrics
from vecstack import stacking



# Get your data



# Initialize 1st level models



# Get your stacking features in a single line

#S_train, S_test = stacking(models, X_train, y_train, X_test, regression = True, verbose = 2)



# Use 2nd level model with stacking features

#Complete examples

#Regression

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from vecstack import stacking





# Caution! All models and parameter values are just 

# demonstrational and shouldn't be considered as recommended.

# Initialize 1st level models.

models = [

    linear_model,

    neigh,

    rfm,

#    xgb_fit

    ]

    

# Compute stacking features

S_train, S_test = stacking(models, train_X, train_Y, test_X, 

    regression = True, metric = mean_absolute_error, n_folds = 4, 

    shuffle = True, random_state = 0, verbose = 2)



# Initialize 2nd level model

#model = XGBRegressor(seed = 0, n_jobs = -1, learning_rate = 0.1,     n_estimators = 100)

model_stacking= xgb_fit



# Fit 2nd level model

model_stacking = model_stacking.fit(S_train, train_Y)



# Predict

y_pred = model_stacking.predict(S_train)



# Final prediction score

print('Final prediction score: [%.8f]' % mean_absolute_error(train_Y, y_pred))
model_metrics=calc_metrics_and_predict(model_stacking,'Stacking',train_X,test_X,train_Y,test_Y,test_final,None,False)

model_metrics
model_comparison_df=pd.DataFrame.from_dict(model_metrics)

model_comparison_df
test_final.head(10)
test_final.columns
models_lst=['linear_reg', 'regularization',

       'KNN', 'NB_Gaussian', 'NB_MultinomialNB', 'NB_BernoulliNB',

       'Random_forest', 'Decision_Tree', 'Adaboost', 'Gradient_boosting', 'XGBoosting',

       'Stacking']

for i in models_lst:

    Linear_submission = test_final[[i]]

    Linear_submission.head(2)

    #Linear_submission.columns = submission_df.columns

    Linear_submission.to_csv(i+".csv")

 