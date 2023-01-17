import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import StratifiedKFold



import string

import warnings

warnings.filterwarnings('ignore')
df_train=pd.read_csv('../input/train.csv')#, index_col='Id')

df_test=pd.read_csv('../input/test.csv')#, index_col='Id')
print('Number of Training Examples = {}'.format(df_train.shape[0]))

print('Number of Test Examples = {}\n'.format(df_test.shape[0]))

print('Training X Shape = {}'.format(df_train.shape))

print('Training y Shape = {}\n'.format(df_train['SalePrice'].shape[0]))

print('Test X Shape = {}'.format(df_test.shape))

print('Test y Shape = {}\n'.format(df_test.shape[0]))

print(df_train.columns)

print(df_test.columns)
print(df_train.info())

df_train.sample(3)

print(df_test.info())

df_test.sample(3)
plt.figure(figsize=(10, 10))

sns.heatmap(df_train.corr())

plt.plot()
#df_train.corr().sort().abs()

"""

df_train_corr = df_train.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

df_train_corr.drop(df_train_corr.iloc[1::2].index, inplace=True)

df_train_corr_nd = df_train_corr.drop(df_train_corr[df_train_corr['Correlation Coefficient'] == 1.0].index)

# Training set high correlations

corr = df_train_corr_nd['Correlation Coefficient'] > 0.1

df_train_corr_nd[corr]

"""



fig, axs = plt.subplots(nrows=2, figsize=(20, 20))



sns.heatmap(df_train.corr(), ax=axs[0], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})

sns.heatmap(df_test.corr(), ax=axs[1], annot=True, square=True, cmap='coolwarm', annot_kws={'size': 14})



for i in range(2):    

    axs[i].tick_params(axis='x', labelsize=14)

    axs[i].tick_params(axis='y', labelsize=14)

    

axs[0].set_title('Training Set Correlations', size=15)

axs[1].set_title('Test Set Correlations', size=15)



plt.show()
df_corr=df_train.corr().sort_values(kind="quicksort", ascending=False, by='SalePrice').abs()

df_corr.drop(axis=1, columns=df_corr.columns.drop('SalePrice'), inplace=True)

df_corr
# df_train['Alley'].isnull().sum() = show number of NaN rows in 'Alley' column
def concat_df(train_data, test_data):

    # Returns a concatenated df of training and test set on axis 0

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)



df_all = concat_df(df_train, df_test)



df_train.name = 'Training Set'

df_test.name = 'Test Set'

df_all.name = 'All Set' 



dfs = [df_train, df_test]
def display_missing(df):

    for col in df.columns:

        print(col, df[col].isnull().sum())

    print('\n')

    

for df in dfs:

    print(format(df.name))

    display_missing(df)
"""### Dealing with missing values

to improve in future - may be not median of overall but 

Missing values in 'LotFrontage' feature are filled with the median LotFrontage, but using the median age of the whole data set is not a good choice. Median age of a group is much better because the new values would be more informative. Median age of Pclass groups is the best choice because of its high correlation with Age (0.408106) and Survived (0.338481) features

"""
df_train['OverallQual'].describe()
#df_train_copy=df_train['LotFrontage'].copy()

    #df_train_copy

#df_train_copy.fillna(value=df_train['LotFrontage'].mean())
#df_train['LotFrontage'].mean()
#Dele

df_train_no_obj=df_train.select_dtypes(exclude=['object'])

df_test_no_obj=df_test.select_dtypes(exclude=['object'])
def display_missing(df):

    for col in df.columns:

        print(col, df[col].isnull().sum())

    print('\n')

    

display_missing(df_test_no_obj)
#drop NA values in train set - for time save

df_train_no_obj.dropna(inplace=True)

#df_test_no_obj.dropna(inplace=True)
# change NA values in test set - to mean

def nan_to_mean(df):

    for col in df.columns:

        df[col].fillna(value=df[col].mean(), inplace=True)

        print(col, df[col].mean())

        

nan_to_mean(df_test_no_obj)
display_missing(df_test_no_obj)
y_train = df_train_no_obj['SalePrice'].values

drop_columns= ['SalePrice','Id']

X_train = df_train_no_obj.drop(columns=drop_columns)

X_val= df_test_no_obj.drop(columns='Id')
y_train.shape
y_train=y_train.reshape(-1,1)

#X_train_no_obj.dropna(inplace=True)

X_train=X_train.astype(np.float32)

X_val=X_val.astype(np.float32)
X_val.shape
X_val
forest_model = RandomForestRegressor()

#forest_model = RandomForestClassifier

forest_model.fit(X_train, y_train)

y_pred = forest_model.predict(X_val)

#print(mean_squared_error(y_val, survival_preds))

#print(mean_absolute_error(y_val, survival_preds))
submission_df = pd.DataFrame(columns=['Id', 'SalePrice'])

submission_df['Id'] = df_test_no_obj['Id']

submission_df['SalePrice'] = y_pred

submission_df.to_csv('submissions.csv', header=True, index=False)

submission_df.head(10)