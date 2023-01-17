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
import pandas as pd
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
                    
submission = pd.read_csv('../input/sample_submission.csv')
submission.head()
X = train.drop(["Id","SalePrice"],axis = 1)
y = train["SalePrice"]
X.head(5)
X.columns
X.shape
def check_missing_data(X):
    total = X.isnull().sum().sort_values(ascending = False)
    percent = ((X.isnull().sum()/X.isnull().count())*100).sort_values(ascending = False)
    return pd.concat([total,percent], axis = 1, keys=['Total', 'Percent'])
check_missing_data(X).head()

check_missing_data(test).head()
''''X = X.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)
y = y.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)'''
df_tmp=pd.DataFrame(X.nunique().sort_values(),columns=['num_unique_values']).reset_index().rename(columns={'index':'Column_name'})
df_tmp.head()
def col_name_with_n_unique_value(X,n):
    df1=pd.DataFrame(X.nunique().sort_values(),columns=['num_unique_values']).reset_index()
    col_name=list(df1[df1.num_unique_values==1]['index'])
    print('number of columns with only',n,'unique values are: ',len(col_name))
    return col_name
col_to_drop=col_name_with_n_unique_value(X,1)
#correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns
corrmat = X.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, annot=True);
# most correlated features
corrmat = train.corr()
top_corr_features = corrmat.index[abs(corrmat['SalePrice'])>0.5]
plt.figure(figsize=(10,10))
g = sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="RdYlGn")
sns.barplot(train.OverallQual,train.SalePrice)
sns.barplot(train.YearBuilt,train.SalePrice)
print("Find most important features relative to target")
corr = train.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)
# Differentiate numerical features (minus the target) and categorical features
categorical_features = X.select_dtypes(include=['object']).columns
categorical_features
numerical_features = X.select_dtypes(exclude = ["object"]).columns
numerical_features
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
test = test.select_dtypes(exclude=[np.object])
test.info()
test = test.fillna(test.mean(), inplace=True)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


# pull data into target (y) and predictors (X)
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

# Create training predictors data
train_X = X[predictor_cols]
my_model = RandomForestRegressor()
my_model.fit(train_X, y)
my_model.score(train_X, y)
test = pd.read_csv("../input/test.csv")
test_X = test[predictor_cols]
#  model to make predictions
predicted_prices = my_model.predict(test_X)
#  at the predicted prices to ensure something sensible.
print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('anwesha_submission.csv', index=False)
from sklearn.linear_model import LinearRegression as lr

model = lr()
model.fit(train_X, y)
print(model.score(train_X,y))
pridected_prices = model.predict(test_X)
model.score(train_X,y)
print(model.coef_)
print(model.intercept_)
X_df = train_X

score,model = compute_scores(train_X, y)
score.sort_values("Error")
model1 = LGBMRegressor()
model1.fit(X_df, y)
model1.score(X_df, y)
model1.predict(test_X)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('anwesha_LGBM_submission.csv', index=False)
model2 = GradientBoostingRegressor()
model2.fit(X_df, y)
model2.score(X_df, y)
model2.predict(test_X)
model2.score(X_df, y)

model5 = ElasticNet()
model5.fit(X_df, y)
model5.score(X_df, y)
model5.predict(test_X)
model5.score(X_df, y)
from sklearn.model_selection import cross_val_score
ridgeReg1 = Ridge(alpha=0.000001, normalize=True)
scores = cross_val_score(ridgeReg1, X_df, y, cv=10)
print(scores)
print(scores.mean())
from sklearn.linear_model import Ridge

ridgeReg = Ridge(alpha=0.000001, normalize=True)

ridgeReg.fit(X_df,y)

print(ridgeReg.score(X_df,y))
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

my_submission.to_csv('anwesha_Gradient_submission.csv', index=False)
my_submission.head(5)
my_submission.describe()
my_submission.info
