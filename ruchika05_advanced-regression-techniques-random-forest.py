import numpy as np # Data manipulation
import pandas as pd # Dataframe manipulation

from sklearn.feature_extraction.text import TfidfVectorizer
# For conversion of categorical to dummy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # For Dummy variable conversion
#  For splitting arrays into train/test
from sklearn.model_selection import train_test_split,GridSearchCV # For Splitting data
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import Ridge #  ridge regression

from sklearn.preprocessing import StandardScaler # For scaling dataset
from sklearn.decomposition import PCA
import xgboost as xgb #XGboot
import lightgbm as lgb # lightgbm
import matplotlib.pyplot as plt #Visualization 
import seaborn as sns # For graphics

import time
from scipy import sparse
from sklearn import metrics
# To plot pretty figures
%matplotlib inline
test_data = pd.read_csv("../input/test.csv");
train_data = pd.read_csv("../input/train.csv");

train_data.shape
test_data.shape
data_merge = [train_data, test_data]
def ExamineData(x):
    """Prints various data charteristics, given x
    """
    print("Data shape:", x.shape)
    print("\nColumn:", x.columns)
    print("\nData types", x.dtypes)
    print("\nDescribe data", x.describe())
    print("\nData ", x.head(2))
    print ("\nSize of data", sys.getsizeof(x)/1000000, "MB" )    # Get size of dataframes
    print("\nAre there any NULLS", np.sum(x.isnull()))
# Get a list of columns having NAs
def ColWithNAs(x):            
    z = x.isnull()
    df = np.sum(z, axis = 0)       # Sum vertically, across rows
    col = df[df > 0].index.values 
    return (col)

def numericalCol(x):            
     return x.select_dtypes(include=[np.number]).columns.values

# Fill the missing values
def MissingValuesAsOther(t , filler = "other"):
    return(t.fillna(value = filler))

def MissingValuesAsMean(t ):
    return(t.fillna(t.mean(), inplace = True))

def MissingValuesAsZero(t ):
     return(t.fillna(value=0, inplace = True))

# Delete the columns
def ColDelete(x,drop_column):
    x.drop(drop_column, axis=1, inplace = True)
# Convert categorical features to dummy
def DoDummy(x):
   #data_dummy =pd.get_dummies(x)
   # Try: le.fit_transform(list('abc'))
    le = LabelEncoder()
    # Apply across all columns of x
    y = x.apply(le.fit_transform)
    # Try:  enc.fit_transform([[1,2],[2,1]]).toarray()
    enc = OneHotEncoder(categorical_features = "all")  # ‘all’: All features are treated as categorical.
    enc.fit(y)
    trans = enc.transform(y)
    return(trans)   

def RegressionEvaluationMetrics(regr,X_test_sparse,y_test_sparse,title):
    predictions=regr.predict(X_test_sparse)
    plt.figure(figsize=(8,6))
    plt.scatter(predictions,y_test_sparse,cmap='plasma')
    plt.title(title)
    plt.show()
    print('MAE:', metrics.mean_absolute_error(y_test_sparse, predictions))
    print('MSE:', metrics.mean_squared_error(y_test_sparse, predictions))
    print('RMSE:', np.sqrt(metrics.mean_squared_error(np.log1p(y_test_sparse), np.log1p(predictions))))

def Regression(regr,X_test_sparse,y_test_sparse):
    start = time.time()
    regr.fit(X_train_sparse,y_train_sparse)
    end = time.time()
    rf_model_time=(end-start)/60.0
    print("Time taken to model: ", rf_model_time , " minutes" ) 
    
def SaveResult(SalePrice,test_ids,file):
    OutputRF = pd.DataFrame(data=SalePrice,columns = ['SalePrice'])
    OutputRF['Id'] = test_ids
    OutputRF = OutputRF[['Id','SalePrice']]
    OutputRF.to_csv(file,index=False)
# Assign 'Saleprice' column to a variable and drop it from tr
#     Also drop train_id/test_id columns
y = train_data['SalePrice']             # This is also the target variable
test_ids = test_data['Id']

train_data.drop( ['SalePrice', 'Id'], inplace = True, axis = 'columns')
test_data.drop( ['Id'], inplace = True, axis = 'columns')    

print('Train columns with null values:\n', train_data.isnull().sum().sort_values())
print("-"*20)

print('Test/Validation columns with null values:\n', train_data.isnull().sum().sort_values())
print("-"*20)
#delete the columns Alley/PoolQC/MiscFeature in train dataset, has all NA values
drop_column = ['MiscFeature','PoolQC', 'Alley','Fence','FireplaceQu']
    
#complete or delete missing values in train and test/validation dataset
for dataset in data_merge:   
    ColDelete(dataset,drop_column)
    columns_replace_with_mean_value = numericalCol(dataset)
    dataset[columns_replace_with_mean_value] = MissingValuesAsMean(dataset[columns_replace_with_mean_value])
    columns_with_na = ColWithNAs(dataset)
    print("Columns with NA :",columns_with_na,"\n")
    dataset[columns_with_na] = MissingValuesAsOther(dataset[columns_with_na])
    print('columns with null values:\n', dataset.isnull().sum().sort_values())
    print("-"*20)

preProcessData = pd.concat(data_merge, axis = 'index')
preProcessData.shape
numerical_columns_name = numericalCol(preProcessData)

categorical_columns_name = preProcessData.columns.difference(numerical_columns_name)
preProcessData[categorical_columns_name].head(10)
#PCA
scaler = StandardScaler()
scaler.fit(preProcessData[numerical_columns_name])
scaled_data = scaler.transform(preProcessData[numerical_columns_name])
scaled_data.shape
type(scaled_data)


model_pca = PCA()
pca = PCA(n_components=4)
pca.fit(scaled_data)
num_pca = pca.transform(scaled_data)

num_pca.shape
type(num_pca)
sparse_num_pca=sparse.csr_matrix(num_pca)
print(pca.components_)

df_comp = pd.DataFrame(pca.components_,columns=numerical_columns_name)

plt.figure(figsize=(15,10))
sns.heatmap(df_comp,cmap='plasma',)
#Categorical values
Nunique = preProcessData[categorical_columns_name].nunique()
Nunique= Nunique.sort_values()
Nunique
categorical_data = DoDummy(preProcessData[categorical_columns_name])
categorical_data.shape
# Concatenate Categorical + Numerical Data
df_sp = sparse.hstack([categorical_data,sparse_num_pca], format = 'csr')

df_sp.shape
type(df_sp)
#Split training and test data
##  Unstack train and test, sparse matrices
df_train = df_sp[ : train_data.shape[0] , : ]
df_test = df_sp[train_data.shape[0] :, : ]
df_train.shape
df_test.shape

#  PArtition datasets into train + validation
#y_train = np.log1p(y)    # Criterion is rmsle
y_train = y
X_train_sparse, X_test_sparse, y_train_sparse, y_test_sparse = train_test_split(
                                     df_train, y_train,
                                     test_size=0.30,
                                     random_state=42
                                     )

y_train_sparse
y_test_sparse
# Instantiate a RandomRegressor object
MAXDEPTH = 60
regr = RandomForestRegressor(n_estimators=1020,       # No of trees in forest
                             criterion = "mse",       # Can also be mae
                             max_features = "sqrt",  # no of features to consider for the best split
                             max_depth= MAXDEPTH,    #  maximum depth of the tree
                             min_samples_split= 2,   # minimum number of samples required to split an internal node
                             min_impurity_decrease=0, # Split node if impurity decreases greater than this value.
                             oob_score = True,       # whether to use out-of-bag samples to estimate error on unseen data.
                             n_jobs = -1,            #  No of jobs to run in parallel
                             random_state=0,
                             verbose = 12            # Controls verbosity of process
                             )


#  Do regression
Regression(regr,X_test_sparse,y_test_sparse)
 
#  Prediction and performance Of RandomForestRegressor
RegressionEvaluationMetrics(regr,X_test_sparse,y_test_sparse,"Random Forest Model")

# Prediction on test Data
SalePrice = regr.predict(df_test)

## Save Result
#SaveResult(SalePrice,test_ids,"Random_Forest_Regression_output.csv")
# Linear Models
# Ridge Regression
model = Ridge(alpha = 63.0,            # Regularization strength. 
              solver = "lsqr",        # auto,svd,cholesky,lsqr,sparse_cg,sag,saga
              fit_intercept=False     # Data is already normalized and centered
              )

#  Do regression
Regression(model,X_test_sparse,y_test_sparse)
 
#  Prediction and performance Of RandomForestRegressor
RegressionEvaluationMetrics(model,X_test_sparse,y_test_sparse,"Ridge Regression Model")

# Prediction on test Data
SalePrice = model.predict(df_test)

## Submission into kaggle
sub = pd.DataFrame()
sub['Id'] = test_ids
sub['SalePrice'] = SalePrice
sub.head()
sub.to_csv('grad_boost_pred_final.csv', index=False)
sub.to_csv('grad_boost_pred_final.csv', index=False)
sub.head()
xgregression = xgb.XGBRegressor(max_depth=3,
                min_child_weight=15,
                colsample_bytree=0.5,
                objective='reg:linear',
                n_estimators=6060,
                learning_rate=0.01)
Regression(xgregression, X_test_sparse, y_test_sparse)

XGregression = RegressionEvaluationMetrics(xgregression, X_test_sparse, y_test_sparse, "XGBoost")
XGregression