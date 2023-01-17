# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/woman-hackathon/train_yhhx1Xs/train.csv')
test=pd.read_csv('/kaggle/input/woman-hackathon/test_QkPvNLx.csv')
sub=pd.read_csv('/kaggle/input/woman-hackathon/sample_submission_pn2DrMq.csv')
train.head()
train['Sales'].unique()
test.head()
sub.head()
sub.columns
train.columns
train.dtypes
train.isnull().mean()
train['Long_Promotion'].unique()
train['Course_Domain'].unique()
train['Course_Type'].unique()
train['Short_Promotion'].unique()
train['Public_Holiday'].unique()
train['User_Traffic'].unique()
train['Competition_Metric'].unique()
def impute_na_numeric(train,test,var):
    mean = train[var].mean()
    median = train[var].median()
    
    train[var+"_mean"] = train[var].fillna(mean)
    train[var+"_median"] = train[var].fillna(median)
    
    var_original = train[var].std()**2
    var_mean = train[var+"_mean"].std()**2
    var_median = train[var+"_median"].std()**2
    
    print("Original Variance: ",var_original)
    print("Mean Variance: ",var_mean)
    print("Median Variance: ",var_median)
    
    if((var_mean < var_original) | (var_median < var_original)):
        if(var_mean < var_median):
            train[var] = train[var+"_mean"]
            test[var] = test[var].fillna(mean)
        else:
            train[var] = train[var+"_median"]
            test[var] = test[var].fillna(median)
    else:
        test[var] = test[var].fillna(median)
    train.drop([var+"_mean",var+"_median"], axis=1, inplace=True)
impute_na_numeric(train,test,'Competition_Metric')
train.isnull().mean()
import seaborn as sns
import matplotlib.pyplot as plt
sns.catplot(x="Course_Type", y="Sales",jitter=False, data=train)
sns.catplot(x="Course_Domain", y="Sales",jitter=False, data=train)
sns.catplot(x="Public_Holiday", y="Sales",  kind="bar", data=train);
sns.catplot(x="Long_Promotion", y="Sales",  kind="bar", data=train);
# Explore Fare distribution 
g = sns.distplot(train["Competition_Metric"], color="m", label="Skewness : %.2f"%(train["Competition_Metric"].skew()))
g = g.legend(loc="best")
import warnings
warnings.filterwarnings('ignore')
# Apply log to Fare to reduce skewness distribution
train["Competition_Metric"] = train["Competition_Metric"].map(lambda i: np.log(i) if i > 0 else 0)
# Explore Fare distribution 
g = sns.distplot(train["User_Traffic"], color="m", label="Skewness : %.2f"%(train["User_Traffic"].skew()))
g = g.legend(loc="best")
# Apply log to Fare to reduce skewness distribution
train["User_Traffic"] = train["User_Traffic"].map(lambda i: np.log(i) if i > 0 else 0)
# Explore Fare distribution 
g = sns.distplot(train["User_Traffic"], color="m", label="Skewness : %.2f"%(train["User_Traffic"].skew()))
g = g.legend(loc="best")
# Explore Fare distribution 
g = sns.distplot(train["Competition_Metric"], color="m", label="Skewness : %.2f"%(train["Competition_Metric"].skew()))
g = g.legend(loc="best")
from sklearn.preprocessing import LabelEncoder
onehoten=LabelEncoder()
train['Course_Domain']=onehoten.fit_transform(train['Course_Domain'])
train['Course_Type']=onehoten.fit_transform(train['Course_Type'])

train['Course_Type'].unique()
train.columns
drop_cols = ['ID', 'Day_No', 'Course_ID']
train.drop(drop_cols,axis=1).drop(["Sales"],axis=1).values
train.drop(drop_cols,axis=1).drop(["Sales"],axis=1).columns
X = train.drop(drop_cols,axis=1).drop(["Sales"],axis=1).values
y = train["Sales"].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=101)
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
#ss.fit(X_train)
X_train_ss=ss.fit_transform(X_train)
X_test_ss = ss.transform(X_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
from sklearn.linear_model import LinearRegression,ElasticNet,Lasso,Ridge,SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,VotingRegressor

regression_models = ['SGDRegressor',
                    'DecisionTreeRegressor','RandomForestRegressor','AdaBoostRegressor']
mse = []
rmse = []
mae = []
models = []
estimators = []
for reg_model in regression_models:
    
    model = eval(reg_model)()
    
    model.fit(X_train_ss,y_train)
    y_pred = model.predict(X_test_ss)
    
    models.append(type(model).__name__)
    estimators.append((type(model).__name__,model))
    
    mse.append(mean_squared_error(y_test,y_pred))
    rmse.append(mean_squared_error(y_test,y_pred)**0.5)
    mae.append(mean_absolute_error(y_test,y_pred))
    
model_dict = {"Models":models,
             "MSE":mse,
             "RMSE":rmse,
             "MAE":mae}
model_df = pd.DataFrame(model_dict)
model_df
model_df["Inverse_Weights"] = model_df['RMSE'].map(lambda x: np.log(1.0/x))
model_df
vr = VotingRegressor(estimators=estimators,weights=model_df.Inverse_Weights.values)
vr.fit(X_train_ss,y_train)
y_pred = vr.predict(X_test_ss)
models.append("Voting_Regressor")
mse.append(mean_squared_error(y_test,y_pred))
rmse.append(mean_squared_error(y_test,y_pred)**0.5)
mae.append(mean_absolute_error(y_test,y_pred))
sub_file = pd.DataFrame(y_pred,columns=["Sale"])
sub_file.head()