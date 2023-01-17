import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.metrics import mean_squared_error,accuracy_score,mean_squared_log_error,r2_score
import sklearn

sklearn.__version__
import os

os.getcwd()
train = pd.read_csv("../input/creditconsumptiondataset/Train.csv")

test = pd.read_csv("../input/creditconsumptiondataset/Test.csv")
#basic statistical description of the train data

train.describe()
# see how many null values are there in percentage

train.isnull().sum()/train.shape[0] *100 
# Adaptive binning on 'Age' using quantiles

quantile_list = [0, .25, .5, .75, 1.]

quantiles = train['age'].quantile(quantile_list)

quantile_labels = ['22-30', '31-34', '35-39', '> 40']

train['age_group'] = pd.qcut(train['age'], q=quantile_list, labels=quantile_labels)

train.head()
sns.distplot(train.age)
def detect_outliers(dataframe):

    cols = list(dataframe)

    outliers = pd.DataFrame(columns=['Feature','Number of Outliers'])

    

    for column in cols:

        if column in dataframe.select_dtypes(include=np.number).columns:

            q1 = dataframe[column].quantile(0.25) 

            q3 = dataframe[column].quantile(0.75)

            iqr = q3 - q1

            fence_low = q1 - (1.5*iqr)

            fence_high = q3 + (1.5*iqr)

            outliers = outliers.append({'Feature':column,'Number of Outliers':dataframe.loc[(dataframe[column] < fence_low) | (dataframe[column] > fence_high)].shape[0]},ignore_index=True)

    return outliers



detect_outliers(train)
X = train.drop(columns=['ID','cc_cons','age','region_code'], axis=1)  

#removed age because we made age bin groups in the above code

#considering that the region code does not affect modelling

y = train['cc_cons']

y.skew()

sns.distplot(y)
y = np.log1p(y)    #transfomed target variable because it was right skewed

X_cols = X.columns

sns.distplot(y)
# Split into categorical and numerical columns

num_cols = X.select_dtypes(exclude=['object','category']).columns

cat_cols = [i for i in X_cols if i not in X[num_cols].columns]

for i in cat_cols:

    X[i] = X[i].astype('category')
 # Function to treat outliers 

def treat_outliers(dataframe):

    cols = list(dataframe)

    for col in cols:

        if col in dataframe.select_dtypes(include=np.number).columns:

            dataframe[col] = winsorize(dataframe[col], limits=[0.1, 0.1],inclusive=(True, True))

    

    return dataframe    





X[num_cols] = treat_outliers(X[num_cols])



# Checking for outliers after applying winsorization

detect_outliers(X)
#Label Encoding to be able to use categorical variables like age group in the regression eq

cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

le = LabelEncoder()

for i in cols:

    X[i] = le.fit_transform(X[i])





# Standardization

scaler = StandardScaler()

X = scaler.fit_transform(X)

X = pd.DataFrame(X, columns=X_cols)



# Spliting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=72)
lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print('Train RMSE:',np.sqrt(mean_squared_error(y_train, lr.predict(X_train))))

print('Test RMSE:',np.sqrt(mean_squared_error(y_test, lr.predict(X_test))));
def rmsle(actual_column, predicted_column):

    sum=0.0

    for x,y in zip(actual_column, predicted_column):

        if x<0 or y<0: #check for negative values. 

            continue

        p = np.log(y+1)

        r = np.log(x+1)

        sum = sum + (p - r)**2

    return (sum/len(predicted_column))**0.5
rmsle(y_test,y_pred)
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor ,RandomForestRegressor

from xgboost import XGBRegressor 

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import Ridge,Lasso
#Decision Trees



dt_reg = DecisionTreeRegressor(random_state=42)

param_grid = dict(max_depth=range(5,20), min_samples_split=range(50,200,10), min_samples_leaf=range(25,100,10), max_leaf_nodes=range(8,32,2), min_impurity_decrease=(0.3,1.0,0.1))

grid = RandomizedSearchCV(dt_reg, param_grid, scoring='neg_mean_squared_error', n_jobs=4, cv=5, random_state=33)

grid.fit(X_train,y_train);





print('Train RMSE:',np.sqrt(mean_squared_error(y_train, grid.best_estimator_.predict(X_train))))

print('Test RMSE:',np.sqrt(mean_squared_error(y_test, grid.best_estimator_.predict(X_test))));
y_pred_dt = grid.best_estimator_.predict(X_test)

rmsle(y_test,y_pred_dt) 
#random forest

rf_model = RandomForestRegressor(random_state=33)

param_grid = dict(n_estimators=range(10,100,10),max_depth=range(3,20),min_samples_split=range(50,500,20),min_samples_leaf=range(25,75,10),max_leaf_nodes=range(8,32,2))

grid = RandomizedSearchCV(rf_model,param_grid,scoring='neg_mean_squared_error',n_jobs=-1,cv=5,random_state=33)

grid.fit(X_train,y_train);





print('Train RMSLE:',np.sqrt(mean_squared_error(y_train, grid.best_estimator_.predict(X_train))))

print('Test RMSLE:',np.sqrt(mean_squared_error(y_test, grid.best_estimator_.predict(X_test))));
y_pred_rf = grid.best_estimator_.predict(X_test)

rmsle(y_test,y_pred_rf)
#feature selection

from sklearn.feature_selection import RFE
def feature_selection(predictors,target,number_of_features,model):



    models = model()

    rfe = RFE(models,number_of_features)

    rfe = rfe.fit(X,y)

    feature_ranking = pd.Series(rfe.ranking_, index=X.columns)

    plt.show()

    print('Features  to be selected for {} are:'.format(str(i[0])))

    print(feature_ranking[feature_ranking.values==1].index.tolist())

    print('===='*30)
# Choosing the models. If you want to specify additional models, kindly specify them as a key-value pair as shown below.

models = {'Random Forest':RandomForestRegressor}

# Selecting 8 number of features

for i in models.items():

    feature_selection(X,y,15,i[1])
X_train = X_train[['cc_cons_apr', 'dc_cons_apr', 'cc_cons_may', 'dc_cons_may', 'cc_cons_jun', 'dc_cons_jun', 'debit_amount_apr', 'credit_amount_apr', 'max_credit_amount_apr', 'debit_amount_may', 'max_credit_amount_may', 'debit_amount_jun', 'credit_amount_jun', 'max_credit_amount_jun', 'emi_active']]

X_test = X_test[['cc_cons_apr', 'dc_cons_apr', 'cc_cons_may', 'dc_cons_may', 'cc_cons_jun', 'dc_cons_jun', 'debit_amount_apr', 'credit_amount_apr', 'max_credit_amount_apr', 'debit_amount_may', 'max_credit_amount_may', 'debit_amount_jun', 'credit_amount_jun', 'max_credit_amount_jun', 'emi_active']]



rf_model=RandomForestRegressor(random_state=72)

param_grid=dict(n_estimators=range(10,100,10),max_depth=range(3,20),min_samples_split=range(50,500,20),min_samples_leaf=range(25,75,10),max_leaf_nodes=range(8,32,2))

grid=RandomizedSearchCV(rf_model,param_grid,scoring='neg_mean_squared_error',n_jobs=-1,cv=5,random_state=33)

grid.fit(X_train,y_train);



print('Train RMSE:',np.sqrt(mean_squared_error(y_train, grid.best_estimator_.predict(X_train))))

print('Test RMSE:',np.sqrt(mean_squared_error(y_test, grid.best_estimator_.predict(X_test))));
import pandas as pd

test_df = pd.read_csv('./Test.csv')

id_col = test_df['ID']





quantiles = test_df['age'].quantile(quantile_list)

test_df['age_group'] = pd.qcut(test_df['age'], q=quantile_list, labels=quantile_labels)
test_df.head()
test_df.isnull().sum()/test_df.shape[0] *100  
test_df.drop(columns=['ID','age','region_code'], axis=1, inplace=True)
cols = test_df.select_dtypes(include=['object', 'category']).columns.tolist()

le = LabelEncoder()

for i in cols:

    test_df[i] = le.fit_transform(test_df[i])





#Scaling

test_df_cols = test_df.columns

test_df = scaler.transform(test_df)

test_df = pd.DataFrame(test_df, columns=test_df_cols)
cols
#Predicting on Test using random forest best features and best parameters

test_df = test_df[['cc_cons_apr', 'dc_cons_apr', 'cc_cons_may', 'dc_cons_may', 'cc_cons_jun', 'dc_cons_jun', 'debit_amount_apr', 'credit_amount_apr', 'max_credit_amount_apr', 'debit_amount_may', 'max_credit_amount_may', 'debit_amount_jun', 'credit_amount_jun', 'max_credit_amount_jun', 'emi_active']]

test_df['cc_cons'] = grid.predict(test_df) 

test_df['cc_cons'] = np.exp(test_df['cc_cons'])-1
#Creating Final Submission file

submissions_5 = pd.concat([id_col, test_df['cc_cons']], axis=1)

submissions_5.to_csv('submission.csv', index=False)

submissions_5
test_df.head()