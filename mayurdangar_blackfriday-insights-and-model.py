import numpy as np # linear algebra
import pandas as pd # data processing, 

# data visualization
import seaborn as sns
import matplotlib.pyplot as plt

# data preprocessing
import sklearn
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Machine Learning Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb
from xgboost.sklearn import XGBRegressor

# utils
import os
import warnings
import pickle
from math import sqrt


# To ignore warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# standard scaler object
stdscaler = StandardScaler()

# check the data files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# create Training Dataframe
train_df = pd.read_csv("/kaggle/input/black-friday/train.csv")
train_df.head()
train_df.info()
train_df.describe(include='all')
print("All columns -> {}".format(list(train_df.columns)))
print()
print("==============================================")
print("Total Transactions -> {}".format(train_df.shape[0]))

unique_users = len(train_df.User_ID.unique())
print("Total unique users -> {}".format(unique_users))

unique_products = len(train_df.Product_ID.unique())
print("Total unique products -> {}".format(unique_products))
# Creating Count plots for Important categorical fields 
fig,axis = plt.subplots(nrows=2,ncols=3,figsize=(17,10))

sns.countplot(train_df["Age"],ax=axis[0,0])
sns.countplot(train_df["Gender"],ax=axis[0,1])
sns.countplot(train_df["Occupation"],ax=axis[0,2])
sns.countplot(train_df["City_Category"],ax=axis[1,1])
sns.countplot(train_df["Stay_In_Current_City_Years"],ax=axis[1,0])
sns.countplot(train_df["Marital_Status"],ax=axis[1,2])
# Creating Count plots for Important categorical fields 

fig,axis = plt.subplots(nrows=2,ncols=3,figsize=(17,10))

train_df.groupby(["Age"])["Purchase"].sum().plot(kind='pie',ax=axis[0,0])
train_df.groupby(["Gender"])["Purchase"].sum().plot(kind='bar',ax=axis[0,1])
train_df.groupby(["Occupation"])["Purchase"].sum().plot(kind='pie',ax=axis[0,2])
train_df.groupby(["City_Category"])["Purchase"].sum().plot(kind='bar',ax=axis[1,1])
train_df.groupby(["Stay_In_Current_City_Years"])["Purchase"].sum().plot(kind='pie',ax=axis[1,0])
train_df.groupby(["Marital_Status"])["Purchase"].sum().plot(kind='bar',ax=axis[1,2])
fig,axis = plt.subplots(nrows=2,ncols=3,figsize=(17,10))

train_df.groupby(["Age"])["Purchase"].mean().plot(kind='bar',ax=axis[0,0])
train_df.groupby(["Gender"])["Purchase"].mean().plot(kind='bar',ax=axis[0,1])
train_df.groupby(["Occupation"])["Purchase"].mean().plot(kind='bar',ax=axis[0,2])
train_df.groupby(["City_Category"])["Purchase"].mean().plot(kind='bar',ax=axis[1,1])
train_df.groupby(["Stay_In_Current_City_Years"])["Purchase"].mean().plot(kind='bar',ax=axis[1,0])
train_df.groupby(["Marital_Status"])["Purchase"].mean().plot(kind='bar',ax=axis[1,2])
fig,axis = plt.subplots(nrows=2,ncols=2,figsize=(17,10))

train_df.groupby(["Age","Gender"])[["Purchase"]].mean().unstack().plot(kind='bar',rot=0, ax = axis[0,0])
train_df.groupby(["Occupation","Gender"])[["Purchase"]].mean().unstack().plot(kind='bar',rot=0, ax = axis[0,1])
train_df.groupby(["Marital_Status","Gender"])[["Purchase"]].mean().unstack().plot(kind='bar',rot=0, ax = axis[1,0])
train_df.groupby(["Stay_In_Current_City_Years","Gender"])[["Purchase"]].mean().unstack().plot(kind='bar',rot=0, ax = axis[1,1])
sns.pairplot(train_df,diag_kind="kde",corner=True,
             markers="+",
             plot_kws=dict(s=1, edgecolor="b", linewidth=1),
             diag_kws=dict(shade=True) )
(train_df.isna().sum()*100/train_df.shape[0]).sort_values(ascending=False).to_frame().rename(columns={0:"Percentage of missing values"})
train_df.loc[train_df.Product_ID=="P00265242",["User_ID","Product_ID","Product_Category_1" ,"Product_Category_2","Product_Category_3"]]
fig,axis = plt.subplots(nrows=1,ncols=3,figsize=(20,8))

sns.countplot(train_df["Product_Category_1"],ax=axis[0])
sns.countplot(train_df["Product_Category_2"],ax=axis[1])
sns.countplot(train_df["Product_Category_3"],ax=axis[2])
train_df[["Product_Category_2","Product_Category_3"]] = train_df[["Product_Category_2","Product_Category_3"]].fillna(0)
(train_df.isna().sum()*100/train_df.shape[0]).sort_values(ascending=False).to_frame().rename(columns={0:"Percentage of missing values"})
# Label encoder object
le = LabelEncoder()

train_df["Age"] = le.fit_transform(train_df["Age"])
train_df["Stay_In_Current_City_Years"] = le.fit_transform(train_df["Stay_In_Current_City_Years"])
train_df["City_Category"] = le.fit_transform(train_df["City_Category"])

# dropped unnecessary fields
train_dropped_df = train_df.drop(['User_ID', 'Product_ID'],axis=1)
print("Dropped the user and product id field")

train_dropped_df = pd.get_dummies(train_dropped_df)

X = train_dropped_df.drop(columns= ["Purchase"])
# separate dataframes one is for independant fields and another for dependant field (Target Field)
y = train_dropped_df['Purchase'].values
colormap = plt.cm.RdBu
plt.figure(figsize=(14,9))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_dropped_df.corr(),
            vmin=-1,
            vmax=1,
            cmap='RdBu',
            annot=True)
print("Input shape -> {}".format(X.shape))
print("Output shape -> {}".format(y.shape))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Input train shape -> {}".format(X_train.shape))
print("Input test shape -> {}".format(X_test.shape))
def train_and_evaluate(model,X_train,y_train,X_test,y_test):
    '''
    This function is to fit the machine learning model and evaluate the R2 score for train and test data
    
    INPUT:
    model - Machine Learning model
    X_train - Training data 
    y_train - Training output values
    X_test - Testing data
    y_test - Testing output values
    
    OUTPUT:
    model - Trained Machine Learning  model
    '''
    model.fit(X_train,y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Score method gives the R2 score actually, so we can directly check the R2 score
    print("Train R-2 Score -> {}".format(r2_score(y_train, y_pred_train)))
    print("Test R-2 Score -> {}".format(r2_score(y_test,y_pred_test)))
    print()
    print("=============================================")
    print()
    print("Train RMSE  -> {}".format(sqrt(mean_squared_error(y_train, y_pred_train))))
    print("Test RMSE  -> {}".format(sqrt(mean_squared_error(y_test,y_pred_test))))
    return model
lr = LinearRegression(n_jobs=-1)
train_and_evaluate(lr,X_train,y_train,X_test,y_test)
dtr = DecisionTreeRegressor(max_depth=8,
                            min_samples_split=5,
                           max_leaf_nodes=10,
                            min_samples_leaf=2,
                            random_state=142)
train_and_evaluate(dtr,X_train,y_train,X_test,y_test)
rf = RandomForestRegressor(max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=142)
train_and_evaluate(rf,X_train,y_train,X_test,y_test)
knnreg = KNeighborsRegressor(n_neighbors = 6)
train_and_evaluate(knnreg,X_train,y_train,X_test,y_test)
# Various hyper-parameters to tune
xgbr = XGBRegressor()
parameters = {
              'objective':['reg:squarederror'],
              'learning_rate': [.03, 0.05, .07], #so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

xgb_grid = GridSearchCV(xgbr,
                        parameters,
                        cv = 5,
                        n_jobs = -1,
                        verbose=True)

fitted_xgb = train_and_evaluate(xgb_grid,X_train,y_train,X_test,y_test)
print(fitted_xgb.best_score_)
print(fitted_xgb.best_params_)
!mkdir /kaggle/model
# Save the model to file in the current working directory

Pkl_Filename = "/kaggle/model/BlackFriday_XGB_Model_V3.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(fitted_xgb, file)