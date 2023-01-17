import numpy as np  

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/ecommerce-data/Train.csv')

test = pd.read_csv('/kaggle/input/ecommerce-data/Test.csv')

sub = pd.read_excel('/kaggle/input/ecommerce-data/Sample_Submission.xlsx')
train.shape, test.shape, sub.shape
train.head(5)
train.isnull().sum()
train.nunique()
sns.distplot(train['Selling_Price'])
train = train[train['Selling_Price'] > 50]

train = train[train['Selling_Price'] < 100000]
df = train.append(test,ignore_index=True)

df.shape
df['Date'] = pd.to_datetime(df['Date'])

df['Day'] = df['Date'].dt.day

df['Month'] = df['Date'].dt.month

df['Year'] = df['Date'].dt.year

#df['Dayofweek'] = pd.to_datetime(df['Date']).dt.dayofweek

#df['DayOfyear'] = pd.to_datetime(df['Date']).dt.dayofyear

#df['WeekOfyear'] = pd.to_datetime(df['Date']).dt.weekofyear

df['Is_month_start'] =  df['Date'].dt.is_month_start 

df['Is_month_end'] = df['Date'].dt.is_month_end 

df['Is_quarter_start'] = df['Date'].dt.is_quarter_start

df['Is_quarter_end'] = df['Date'].dt.is_quarter_end 

df['Is_year_start'] = df['Date'].dt.is_year_start 

df['Is_year_end'] = df['Date'].dt.is_year_end
df.head(3)
calc = df.groupby(['Product_Brand'], axis=0).agg({'Product_Brand':[('op1', 'count')]}).reset_index() 

calc.columns = ['Product_Brand','Product_Brand Count']

df = df.merge(calc, on=['Product_Brand'], how='left')



calc = df.groupby(['Item_Category'], axis=0).agg({'Item_Category':[('op1', 'count')]}).reset_index() 

calc.columns = ['Item_Category','Item_Category Count']

df = df.merge(calc, on=['Item_Category'], how='left')



calc = df.groupby(['Subcategory_1'], axis=0).agg({'Subcategory_1':[('op1', 'count')]}).reset_index() 

calc.columns = ['Subcategory_1','Subcategory_1 Count']

df = df.merge(calc, on=['Subcategory_1'], how='left')



calc = df.groupby(['Subcategory_2'], axis=0).agg({'Subcategory_2':[('op1', 'count')]}).reset_index() 

calc.columns = ['Subcategory_2','Subcategory_2 Count']

df = df.merge(calc, on=['Subcategory_2'], how='left')
agg_func = {

    'Item_Rating': ['mean','min','max','sum']    

}

agg_func = df.groupby('Product_Brand').agg(agg_func)

agg_func.columns = [ 'Product_Brand_' + ('_'.join(col).strip()) for col in agg_func.columns.values]

agg_func.reset_index(inplace=True)

df = df.merge(agg_func, on=['Product_Brand'], how='left')



agg_func = {

    'Item_Rating': ['mean','min','max','sum']    

}

agg_func = df.groupby('Item_Category').agg(agg_func)

agg_func.columns = [ 'Item_Category_' + ('_'.join(col).strip()) for col in agg_func.columns.values]

agg_func.reset_index(inplace=True)

df = df.merge(agg_func, on=['Item_Category'], how='left')



agg_func = {

    'Item_Rating': ['mean','min','max','sum']    

}

agg_func = df.groupby('Subcategory_1').agg(agg_func)

agg_func.columns = [ 'Subcategory_1_' + ('_'.join(col).strip()) for col in agg_func.columns.values]

agg_func.reset_index(inplace=True)

df = df.merge(agg_func, on=['Subcategory_1'], how='left')



agg_func = {

    'Item_Rating': ['mean','min','max','sum']    

}

agg_func = df.groupby('Subcategory_2').agg(agg_func)

agg_func.columns = [ 'Subcategory_2_' + ('_'.join(col).strip()) for col in agg_func.columns.values]

agg_func.reset_index(inplace=True)

df = df.merge(agg_func, on=['Subcategory_2'], how='left')
for c in ['Product_Brand', 'Item_Category', 'Subcategory_1', 'Subcategory_2']:

    df[c] = df[c].astype('category')



#df = pd.get_dummies(df, columns=['Product_Brand', 'Item_Category', 'Subcategory_1', 'Subcategory_2'], drop_first=True)
#df['Diff'] = (pd.to_datetime("today") - df['Date']) / np.timedelta64(1,'D')
agg_func = {

    'Item_Rating': ['mean','min','max','sum']    

}

agg_func = df.groupby(['Product_Brand', 'Item_Category']).agg(agg_func)

agg_func.columns = [ 'Product_Brand_Item_Category_' + ('_'.join(col).strip()) for col in agg_func.columns.values]

agg_func.reset_index(inplace=True)

df = df.merge(agg_func, on=['Product_Brand', 'Item_Category'], how='left')



agg_func = {

    'Item_Rating': ['mean','min','max','sum']    

}

agg_func = df.groupby(['Item_Category', 'Subcategory_2']).agg(agg_func)

agg_func.columns = [ 'Item_Category_Subcategory_2_' + ('_'.join(col).strip()) for col in agg_func.columns.values]

agg_func.reset_index(inplace=True)

df = df.merge(agg_func, on=['Item_Category', 'Subcategory_2'], how='left')
df.drop(['Product','Date'], axis=1, inplace=True)
train_df = df[df['Selling_Price'].isnull()!=True]

test_df = df[df['Selling_Price'].isnull()==True]

test_df.drop(['Selling_Price'], axis=1, inplace=True)
train_df['Selling_Price'] = np.log1p(train_df['Selling_Price'])
X = train_df.drop(labels=['Selling_Price'], axis=1)

y = train_df['Selling_Price'].values



X.shape, y.shape
from math import sqrt 

import lightgbm as lgb

from sklearn.metrics import mean_squared_error, mean_squared_log_error
Xtest = test_df
from catboost import CatBoostRegressor

from sklearn.model_selection import KFold



errcat = []

y_pred_totcat = []

categorical_features_indices = np.where(X.dtypes == 'category')[0]



fold = KFold(n_splits=10, shuffle=True, random_state=42)



for train_index, test_index in fold.split(X):

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    cat = CatBoostRegressor(loss_function='RMSE', 

                         eval_metric='RMSE', 

                         depth=7,

                         random_seed=42, 

                         iterations=1000, 

                         learning_rate=0.1,

                         leaf_estimation_iterations=1,

                         l2_leaf_reg=1,

                         bootstrap_type='Bayesian', 

                         bagging_temperature=1, 

                         random_strength=1,

                         od_type='Iter', 

                         od_wait=200)

    

    cat.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0, early_stopping_rounds=200, cat_features=categorical_features_indices)



    y_pred_cat = cat.predict(X_test)

    print("RMSLE: ", sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(y_pred_cat))))



    errcat.append(sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_cat))))

    p = cat.predict(Xtest)

    y_pred_totcat.append(p)
np.mean(errcat,0)  
final = np.exp(np.mean(y_pred_totcat,0))
feature_imp = pd.DataFrame(sorted(zip(cat.feature_importances_, X.columns), reverse=True)[:], columns=['Value','Feature'])

#feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importances_, X.columns), reverse=True)[:], columns=['Value','Feature'])

plt.figure(figsize=(14,8))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('Features')

plt.tight_layout()

plt.show()
pd.DataFrame(sorted(zip(cat.feature_importances_, X.columns), reverse=True)[:], columns=['Value','Feature'])
sub['Selling_Price'] = final
sub.head()
sub.to_excel('Output.xlsx', index=False)