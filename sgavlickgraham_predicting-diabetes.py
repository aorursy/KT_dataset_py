import pandas as pd

import numpy as np

import matplotlib.pylab as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn import datasets

import xgboost as xgb

import seaborn as sns
def nulls_by_col(df):

# Calculate the number and percent of null values in each column.

    num_missing = df.isnull().sum()

    rows = df.shape[0]

    pct_missing = num_missing/rows

    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'pct_rows_missing': pct_missing})

    return cols_missing
def nulls_by_row(df):

# Calculate the number of percent of null values in each row.

    num_cols_missing = df.isnull().sum(axis=1)

    pct_cols_missing = df.isnull().sum(axis=1)/df.shape[1]*100

    rows_missing = pd.DataFrame({'num_cols_missing': num_cols_missing, 'pct_cols_missing': pct_cols_missing}).reset_index().groupby(['num_cols_missing','pct_cols_missing']).count().rename(index=str, columns={'index': 'num_rows'}).reset_index()

    return rows_missing
def df_summary(df):

# Print information about the data including its shape, datatypes, number of values, 

# number of null values in each row and column, the number of unique rows, etc.

    print('--- Shape: {}'.format(df.shape))

    print('\n--- Info')

    display(df.info())

    print('\n--- Descriptions')

    display(df.describe(include='all'))

    print('\n--- Nulls By Column')

    display(nulls_by_col(df))

    print('\n--- Nulls By Row')

    display(nulls_by_row(df))

    print('\n---Unique Rows')

    display(df.apply(lambda x: x.nunique()))
def get_scaled_df(df):

# Return a dataframe that contains only numeric data so that we can scale it for XGBoost.

# This is not necessary for this data as it is already scaled, but it is part of a 

# pre-existing function that I wrote so I am leaving it here.

    numerics = ['int64', 'float64', 'float']

    scaled_df = df.select_dtypes(include=numerics)

    col = scaled_df.columns

    scaled_df = preprocessing.scale(scaled_df)

    scaled_df = pd.DataFrame(scaled_df, columns=col)

    return scaled_df
def xgb_rank(df,target_variable,feature_percent=80,mode='gain'):

    '''

    This function receives a dataframe and the target variable, and then returns 

    a sorted feature list, a sorted scaled feature list, and a dataframe. 

    

    For the input parameters:

        - feature_percent is the optional cut-off (default is 80 percent) for features 

        - mode is optional. The default value is 'gain' which shows the importance. 

          Another possible value for mode is 'weight.'

     

     For the returned:

        - feature_list, scaled_features: lists of features, both including those that 

          satisfy the cumulative percentage limit. 

        - scaled_df: dataframe that has all features in decending order 

        - importance_df: dataframe showing all cumulative percent rankings 

    '''    



    scaled_df = get_scaled_df(df) 

    xgb_params = {'max_depth': 8,'seed' : 123}

    dtrain = xgb.DMatrix(scaled_df, target_variable, feature_names=scaled_df.columns.values)

    model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)

    importance_dict = model.get_score(importance_type=mode)

    sorted_importance_dict = sorted(importance_dict.items(), key=lambda kv: kv[1])

    importance_df = pd.DataFrame.from_dict(sorted_importance_dict)

    importance_df.columns = ['feature',mode] 

    importance_df.sort_values(mode, inplace = True) 

    importance_df['rank'] = importance_df[mode].rank(ascending = False)

    importance_df.sort_values('rank', inplace = True) 

    importance_df.set_index('rank', inplace = True)

    importance_df.reset_index(inplace=True) 

    importance_df[mode] = importance_df[mode].apply(lambda x: round(x, 2))

    importance_df['cum_sum'] = round(importance_df[mode].cumsum(),2)

    importance_df['cum_perc'] = round(100*importance_df.cum_sum/importance_df[mode].sum(),2)

    feature_list = []

    scaled_features = [] 



    for i in range((importance_df.shape[0])): 



        feature_name = importance_df.iloc[i,1].replace('scaled_','')

        scaled_name = 'scaled_' + feature_name

        importance_df.iloc[i,1] = feature_name

        cum_percent = importance_df.iloc[i,4]



        if cum_percent > feature_percent:

            break

        else:

            feature_list.append(feature_name)

            scaled_features.append(scaled_name)

    return feature_list, scaled_features, scaled_df, importance_df
diabetes = datasets.load_diabetes() # load data

data = np.c_[diabetes.data, diabetes.target]

columns = np.append(diabetes.feature_names, 'target')

df = pd.DataFrame(data, columns=columns)

print(type(df))

df.head()
df_summary(df)
X = df.drop(columns=['target'])

y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
feature_list, scaled_features, scaled_df, importance_df = xgb_rank(X_train, y_train)
print('feature_list: ', feature_list, '\n')

print('scaled_features: ', scaled_features, '\n')

print('\nscaled_df:')

display(scaled_df.head())

print('\ny_train:')

display(y_train.head())

print('\nimportance_df:')

display(importance_df)
full_scaled_df = scaled_df.copy()

full_scaled_df['target'] = preprocessing.scale(y_train)

display(full_scaled_df.head())
g = sns.PairGrid(full_scaled_df)

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter);
# Set the background to white so it won't show after adding the mask.

sns.set(style="white")



# Compute the correlation matrix from train_df.

corr = full_scaled_df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(20, 11))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.xticks(rotation=60)
X_train = X_train.drop(columns=['s2', 's4', 'age', 's1', 'sex'])

print('X_train.head(2):')

display(X_train.head(2))

X_test = X_test.drop(columns=['s2', 's4', 'age', 's1', 'sex'])

print('X_test.head(2):')

display(X_test.head(2)) # Just to confirm we have the right features there 

                        # for when we do our test.

# And while we're at it, let's drop those columns from X.

X = df.drop(columns=['s2', 's4', 'age', 's1', 'sex'])

print('X.head(2):')

display(X.head(2)) # Again, just to confirm.
plt.figure(figsize=(15,10))

plt.tight_layout()

sns.distplot(y_train)
# There are three steps to model something with sklearn

# 1. Set up the model

model = LinearRegression()

# 2. Use fit

model.fit(X_train, y_train)

# 3. Check the score

print(model.score(X_test, y_test))

#4. Check the regression metrics

y_pred = model.predict(X_test)
#To retrieve the intercept:

print('The intercept is ', model.intercept_)

#For retrieving the slope:

print('The slope is ', model.coef_)
coeff_df = pd.DataFrame(model.coef_, X_train.columns, columns=['Coefficient'])  

coeff_df.sort_values(by='Coefficient', ascending=False)
eval_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

eval_df_top25 = eval_df.head(25)

display(eval_df_top25)
eval_df_top25.plot(kind='bar',figsize=(10,8))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
print('Mean value of the target variable is:', y_test.mean())

print('R-squared:', metrics.r2_score(y_test, y_pred))

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('mean/RMSE = ', np.sqrt(metrics.mean_squared_error(y_test, y_pred))/y_test.mean())
sns.residplot(y_test, y_pred, lowess=True, color="g")
# There are three steps to model something with sklearn

# 1. Set up the model

model = LogisticRegression(solver='newton-cg', multi_class='ovr')

# 2. Use fit

model.fit(X_train, y_train)

# 3. Check the score

print('score = ', model.score(X_test, y_test))

# 4. Check the regression metrics

y_pred = model.predict(X_test)