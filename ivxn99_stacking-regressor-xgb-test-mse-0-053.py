import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
train_data_path = '/kaggle/input/house-price-prediction-challenge/train.csv'
train_data = pd.read_csv(train_data_path)
train_data = train_data.drop(['ADDRESS'], axis=1) #We have no use of this column

print(train_data.isna().sum()) #Check if there are null values
sns.countplot(x=train_data['POSTED_BY'], data=train_data)
fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.boxplot(x='POSTED_BY',y='TARGET(PRICE_IN_LACS)',data=train_data)
fig=sns.stripplot(x='POSTED_BY',y='TARGET(PRICE_IN_LACS)',data=train_data,jitter=True,edgecolor='gray')
#Strip plot
fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.stripplot(x='UNDER_CONSTRUCTION',y='TARGET(PRICE_IN_LACS)',data=train_data,jitter=True,edgecolor='gray',size=8,palette='winter',orient='v')
train_data.corr(method='pearson').sort_values('TARGET(PRICE_IN_LACS)', ascending=False)
categorical_val = []
continous_val = []
for column in train_data.columns:
	if len(train_data[column].unique()) < 10:
		categorical_val.append(column)
	else:
		continous_val.append(column)
        
train_data[categorical_val].apply(lambda x: x.nunique())
le = LabelEncoder()

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

train_data = MultiColumnLabelEncoder(columns = categorical_val).fit_transform(train_data)

#TODO: STANDARDIZE THE CONTINUOS_VALUES

scaler = StandardScaler()
train_data[continous_val] = scaler.fit_transform(train_data[continous_val])
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

models = []
models.append(("Lasso", linear_model.Lasso(alpha=0.1)))
models.append(("RandomForest", RandomForestRegressor()))
models.append(("XGB", XGBRegressor()))
models.append(("GradientBoosting", GradientBoostingRegressor()))
models.append(("LGBM", LGBMRegressor()))

estimators = [
              ('rfr', RandomForestRegressor()),
              ('gb', GradientBoostingRegressor()),
              ('lgbm', LGBMRegressor())
              #('cb', CatBoostRegressor())
]

models.append(("StackingRegressor", StackingRegressor(estimators=estimators, final_estimator=XGBRegressor())))

results = []
names = []
for name,model in models:
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  result = mean_squared_error(y_test, y_pred)
  names.append(name)
  results.append(result)

for i in range(len(names)):
    print(names[i],results[i].mean())