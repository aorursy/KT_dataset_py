import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import mean_absolute_error
def less_values(frame, percentile):

    """

    Checks which columns have more than certain number of missing values and drops them.

    """

    df = frame

    for col in df.columns:

        count = df[col].count()

        if count < 1460*float(percentile):

            df.drop(col, axis=1, inplace=True)

    

    return df
pd.set_option('display.max_columns', None)
path = '../input/house-prices-advanced-regression-techniques/train.csv'

df = pd.read_csv(path, index_col = 0)
# Blank page
import category_encoders as ce

from sklearn.impute import SimpleImputer
# Some columns have an integer data type, while they are categorical.

# For future convenience, I'll change the data type to object



for col in ['MSSubClass','OverallQual', 'OverallCond']:

    df[col] = df[col].astype('object')
# As there are some columns with insignificant amount of values

df = less_values(frame=df, percentile=0.8)
# I want to focus more on continuous and categorcical data, so I'll just drop inforamtion about date

date = np.array(['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold'])

df = df.drop(date, axis = 1)
# Define numerical data columns

num_cols = np.array(df.select_dtypes(include=['int64', 'float64']).columns)

num_cols = np.delete(num_cols, len(num_cols) - 1) # Removing SalePrice i.e target



# Define categorical data columns

cat_cols = np.array(df.select_dtypes(include=['object']).columns)
# Since I want to use imputers and encoders, I need train data to fit them

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:, -1], test_size=.3, random_state=42)
num_transformer = SimpleImputer(strategy='median', missing_values=np.nan)

X_train[num_cols] = num_transformer.fit_transform(X_train[num_cols])

print('Missing values: {}'.format(X_train[num_cols].isnull().any().any()))
X_test[num_cols] = num_transformer.transform(X_test[num_cols])

print('Missing values: {}'.format(X_test[num_cols].isnull().any().any()))
cat_imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)

X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])

print('Missing values: {}'.format(X_train[cat_cols].isnull().any().any()))
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

print('Missing values: {}'.format(X_test[cat_cols].isnull().any().any()))
t_encoder = ce.TargetEncoder(cols=cat_cols)
X_train_target = X_train.copy()

X_train_target[cat_cols] = t_encoder.fit_transform(X_train[cat_cols], y_train)

X_train_target[cat_cols[0:5]].head(1) # Checking the results
X_test_target = X_test.copy()

X_test_target[cat_cols] = t_encoder.transform(X_test[cat_cols])

X_test_target[cat_cols[0:5]].head(1)
glmm = ce.glmm.GLMMEncoder()
X_train_glmm = X_train.copy()

X_train_glmm[cat_cols] = glmm.fit_transform(X_train[cat_cols], y_train)

X_train_glmm[cat_cols[5:10]].head(1)
X_test_glmm = X_test.copy()

X_test_glmm[cat_cols] = glmm.transform(X_test[cat_cols])

X_test_glmm[cat_cols[5:10]].head(1)
# Blank page
# Blank page
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=45) # I have found out best k simply by trial and error
X_train_new = selector.fit_transform(X_train_target, y_train)

X_train_new = pd.DataFrame(selector.inverse_transform(X_train_new),

                          index=X_train.index,

                          columns=X_train.columns)

selected_columns_target = X_train_new.columns[X_train_new.var() != 0]
print('First 5 features from target encoded data: {}'.format(list(selected_columns_target[:5])))
X_train_new = selector.fit_transform(X_train_glmm, y_train)

X_train_new = pd.DataFrame(selector.inverse_transform(X_train_new),

                          index=X_train.index,

                          columns=X_train.columns)

selected_columns_glmm = X_train_new.columns[X_train_new.var() != 0]
print('First 5 features from glmm encoded data: {}'.format(list(selected_columns_glmm[:5])))
# Blank page
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score
reg = LinearRegression()

forest = RandomForestRegressor(random_state=42)

xgb = XGBRegressor(random_state=42)
models = {'Linear Regression': reg,

         'Random Forest': forest,

         'XGB Regression': xgb}
# For future cross validation I'll use all data I have

all_target = pd.concat((X_train_target, X_test_target))

all_glmm = pd.concat((X_train_glmm, X_test_glmm))

all_y = pd.concat((y_train, y_test))
results = {}

target_means = []

glmm_means = []



for i in models:

    # Calculating cross validation scores and mean for target encoded data

    models[i].fit(X_train_target[selected_columns_target], y_train)

    t_score = cross_val_score(models[i], all_target[selected_columns_target], all_y, scoring='neg_mean_absolute_error', cv=3)

    t_score = -1 * t_score

    t_score = np.round(t_score, 1)

    t_mean = np.round(t_score.mean(), 1)

    t_score = ('targer score', t_score, t_mean)

    target_means.append(t_mean)

    

    # Calculating cross validation scores and mean for glmm encoded data

    models[i].fit(X_train_glmm[selected_columns_glmm], y_train)

    glmm_score = cross_val_score(models[i], all_glmm[selected_columns_glmm], all_y, scoring='neg_mean_absolute_error', cv=3)

    glmm_score = -1 * glmm_score

    glmm_score = np.round(glmm_score, 1)

    glmm_mean = np.round(glmm_score.mean(), 1)

    glmm_score = ('glmm score', glmm_score, glmm_mean)

    glmm_means.append(glmm_mean)

    

    # score

    score = [t_score, glmm_score]

    results[i] = score
sns.set_style('whitegrid')
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13, 5))

plt.subplots_adjust(top=0.80)

fig.suptitle('Cross validation results from train data', fontsize=16)



x_axis = list(models.keys())

ax1 = sns.barplot(x=x_axis, y=target_means, palette='Set2', ax=ax1)

ax1.set_title('Target encoded data', fontsize=14)

ax1.set_ylim((0, 22000))

ax1.set_ylabel('Mean absolute error')



ax2 = sns.barplot(x=x_axis, y=glmm_means, palette='Set2', ax=ax2)

ax2.set_title('Glmm encoded data', fontsize=14)

ax2.set_ylim((0, 22000))

ax2.set_ylabel('Mean absolute error')





plt.show()
# More detailed

for key in results:

    print(key + ':')

    print(results[key])

    

    if key == 'XGB Regression':

        print('_'*125)

    else:

        print('\n')
def train_model(model, X_train, X_test, y_train=y_train, y_test=y_test):

    """

    Fitting model, and scoring with mean absolute error

    """

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    score = mean_absolute_error(y_test, pred)

    return score
test_target = []

test_glmm = []



for i in models:

    # Calculating mean absolute error for target encoded data

    score = train_model(models[i], X_train_target[selected_columns_target], X_test_target[selected_columns_target])

    score = np.round(score, 1)

    test_target.append(score)

    

    # Calculating mean absolute error for glmm encoded data

    score = train_model(models[i], X_train_glmm[selected_columns_glmm], X_test_glmm[selected_columns_glmm])

    score = np.round(score, 1)

    test_glmm.append(score)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(13, 5))

plt.subplots_adjust(top=0.8)

fig.suptitle('Evaluation results from test data', fontsize=16)



x_axis = list(models.keys())

ax1 = sns.barplot(x=x_axis, y=test_target, palette='Set2', ax=ax1)

ax1.set_title('Target encoded data', fontsize=14)

ax1.set_ylim((0, 22000))

ax1.set_ylabel('Mean absolute error')



ax2 = sns.barplot(x=x_axis, y=test_glmm, palette='Set2', ax=ax2)

ax2.set_title('Glmm encoded data', fontsize=14)

ax2.set_ylim((0, 22000))

ax2.set_ylabel('Mean absolute error')



plt.show()
# More detailed

for i in range(3):

    print(x_axis[i] + ':')

    print('  Target encoded data: ' + str(test_target[i]))

    print('  Glmm encoded data: ' + str(test_glmm[i]))

    

    if i == 2:

        print('_'*125)

    else:

        print('\n')
# Blank page
X_train_new = X_train_glmm[selected_columns_glmm]

X_test_new = X_test_glmm[selected_columns_glmm]
forest = RandomForestRegressor(n_estimators=1000,

                              min_samples_leaf=2,

                              max_features='sqrt',

                              oob_score=True,

                              n_jobs=4,

                              random_state=42)
forest.fit(X_train_new, y_train)
pred = forest.predict(X_test_new)
from sklearn.metrics import r2_score
error = mean_absolute_error(y_test, pred)

error = np.round(error, 0)



r2 = r2_score(y_test, pred)

r2 = np.round(r2, 2)



oob = forest.oob_score_

oob = np.round(oob, 2)
# More detailed metrics

pd.DataFrame(index=['Mean absolute error', 'R^2', 'Out-of-bag'], columns=['Random Forest'], data=[error, r2, oob])
# Blank page
x = np.arange(0, 800001, 100000)
plt.figure(figsize=(10,6))



sns.lineplot(x=x, y=x, color='orange')

sns.scatterplot(y=y_test, x=pred, color='deepskyblue', s=50)



plt.xlim(0,800000)

plt.ylim(0,800000)

plt.title('Relation between true and predicted values\n', fontsize=16)

plt.xlabel('Predicted values', fontsize=14)

plt.ylabel('True values', fontsize=14)



plt.text(100000, 70000, '  2  ', ha="center", va="center", rotation=0, size=20,

    bbox=dict(boxstyle="circle,pad=0.3", lw=2, color='darkslategray', fill=False))

plt.text(480000, 620000, '  1  ', ha="center", va="center", rotation=0, size=60,

    bbox=dict(boxstyle="circle,pad=0.3", lw=2, color='darkslategray', fill=False))



plt.show()
pred_table = pd.DataFrame({'True values': y_test,

             'Predicted values': pred}, 

            index=y_test.index)
pred_table['Absolute'] = pred_table['True values'] - pred_table['Predicted values']

pred_table['Absolute'] = pred_table['Absolute'].apply('abs')

pred_table['Percentage'] = pred_table['Absolute'] / pred_table['True values']
def highlight_columns(df, rows=20, color='lightgreen', columns_to_highlight=[], columns_to_show=[]):

    """

    Highlights selected columns of dataframe

    """

    highlight = lambda slice_of_df: 'background-color: %s' % color

    sample_df = df.head(rows)

    if len(columns_to_show) != 0:

        sample_df = sample_df[columns_to_show]

    highlighted_df = sample_df.style.applymap(highlight, subset=pd.IndexSlice[:, columns_to_highlight])

    return highlighted_df
abs_error = pred_table.sort_values(by='Absolute', ascending=False)

highlight_columns(abs_error[:10], columns_to_highlight=['Absolute'])
plt.figure(figsize=(8,5))



sns.distplot(abs_error['True values'], color='deepskyblue')

plt.axvline(x=400000, color='orange', linewidth = 2)

plt.title('Distribution of True values', fontsize=16)



plt.show()
# Blank page
prc_error = pred_table.sort_values(by='Percentage', ascending=False)

highlight_columns(prc_error[:10], columns_to_highlight=['Percentage'])
plt.figure(figsize=(8,5))



sns.distplot(abs_error['True values'], color='deepskyblue')

plt.axvline(x=70000, color='orange', linewidth = 2)

plt.title('Distribution of True values', fontsize=16)



plt.show()
# Blank page