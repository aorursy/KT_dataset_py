#Importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

from scipy.stats import chi2_contingency

from scipy.stats import spearmanr

%matplotlib inline

import itertools

import os

import calendar

from datetime import datetime

from scipy import stats

from scipy.special import inv_boxcox

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split as split

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

def get_numerical_and_categorical_col(df):

    numerical_col = []

    cat_col = []

    for col in df.columns:

        if str(df[col].dtype).startswith(('int','float')):

            numerical_col.append(col)

        elif str(df[col].dtype) == 'category':

            cat_col.append(col)

    return numerical_col, cat_col
def change_to_categorical(data, max_cat=10):

  df = data.copy()

  for col in df.columns:

        if df[col].dtype == object or str(df[col].dtype).startswith(('int','float')):

            count = len(df[col].unique())

            if count <= max_cat:

                df[col] = df[col].astype('category')

  return df
# Helper method for printing percentage on count plot

def print_percent_count_plot(value_counts, ax):

    total = sum(value_counts)

    for idx, count in value_counts.iteritems():            

      percent_val = (count*100)/total

      add_to_idx = 0

      if min(value_counts.index) > 0:

        add_to_idx = 1

      plt.text(idx - add_to_idx-.1,count/2,str(round(percent_val))+'%')
def get_count_plot(x, df, ax, y=None, value_counts = None, print_percent = False):

  if value_counts is None:

    counts = df[x].value_counts().sort_index()

  else:

    counts = df[value_counts]

  #counts.plot.bar()

  #sns bars are just more colorful :P

  if y is None:

    sns.countplot(x, data=df, ax=ax)

  else:

    sns.barplot(x, y, data=df, ax=ax)

  if print_percent:

    print_percent_count_plot(counts, ax)
def get_count_plot_for_categorical(df, n_cols = 2, y='cnt', list_cat=None, value_counts=None, print_percent=False):

  if list_cat is None:

    num_col, cat_col = get_numerical_and_categorical_col(df)

  else:

    cat_col = list_cat

  f, axs, n_rows = get_fig_and_axis_for_subplots(len(cat_col), n_cols)

  for i, col in enumerate(cat_col):

    ax = plt.subplot(n_rows, n_cols, i+1)

    get_count_plot(col, df, ax, y, value_counts, print_percent)
def get_target_dist_with_categorical(df, n_cols = 2, y='cnt', list_cat=None, plot_type = 'box'):

  if list_cat is None:

    num_col, cat_col = get_numerical_and_categorical_col(df)

  else:

    cat_col = list_cat

  f, axs, n_rows = get_fig_and_axis_for_subplots(len(cat_col), n_cols)

  for i, col in enumerate(cat_col):

    ax = plt.subplot(n_rows, n_cols, i+1)

    if plot_type == 'box':

      sns.boxplot(x=col, data=df,y=y,orient="v",ax=ax)

    else:

      sns.violinplot(col, data=df,y=y,orient="v",ax=ax)
def get_fig_and_axis_for_subplots(total, n_cols = 2):

  rows = total/ n_cols

  n_rows = int(np.ceil(rows))

  f, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))

  plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=.25, wspace=0.4)

  if rows* n_cols < n_rows * n_cols:

    diff = n_rows - rows

    for i in range(n_cols-int(np.ceil(diff))):

      if n_rows == 1:

        f.delaxes(axs[n_cols-1-i])

      else:

        f.delaxes(axs[n_rows-1, n_cols-1-i])     

  return f, axs, n_rows
def get_plot_for_numerical(df, n_cols = 2, plot_type='probability',list_col=None, hist=True, kde=True):

  if list_col is None:

      num_col, cat_col = get_numerical_and_categorical_col(df)

  else:

      num_col = list_col

  f, axs, n_rows = get_fig_and_axis_for_subplots(len(num_col), n_cols)

  for i, col in enumerate(num_col):

    ax = plt.subplot(n_rows, n_cols, i+1)

    if plot_type == 'probability':

      sns.distplot(df[col], hist=hist, kde=hist, 

             color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

    elif plot_type == 'box':

      sns.boxplot(data=df,y=col,orient="v",ax=ax)

    else:

      sns.violinplot(data=df,y=col,orient="v",ax=ax)
# visualize correlation matrix

def visualize_corr_matrix(data):

    numerical_col, cat_col = get_numerical_and_categorical_col(data)

    df = data[numerical_col]

    corr = df.corr()# plot the heatmap

    #generating masks for upper triangle so that values are not repeated

    mask_ut=np.triu(np.ones(corr.shape)).astype(np.bool)

    sns.heatmap(corr, mask=mask_ut, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
def remove_outliers_for_variable_by_quantiles(data, col, q1=.25, q2=.75):

  df = data.copy()

  median = df[col].median()

  q25, q75 = df[col].quantile([q1,q2])

  iqr = q75-q25

  upper_wh = q75 +1.5*iqr

  lower_wh = q25 - 1.5*iqr

  whiskers = int(np.floor(lower_wh)), int(np.ceil(upper_wh))

  df.drop(df[~df[col].between(whiskers[0], whiskers[1]) & (~np.isnan(df[col]))].index, inplace=True)

  return df
def remove_outliers_for_variable_by_std(data, col):

  df = data.copy()

  df = df[np.abs(df[col]-df[col].mean())<=(3*df[col].std())] 

  return df
#loop for chi square values

def calculate_chi_square_values(df, alpha=.05):

    chi2_dict = {}

    numerical_col, cat_col  = get_numerical_and_categorical_col(df)

    for i in cat_col:

        for j in cat_col:

            if i!=j and (j+' '+i) not in chi2_dict.keys():

                chi2, p, dof, ex = chi2_contingency(pd.crosstab(df[i], df[j]))

                chi2_dict[i+' '+j] = 'Independent? '+ str(p>alpha)

    return chi2_dict
def rmsle(y, y_,convertExp=True):

    

    if convertExp:

        y = inv_boxcox(y, fitted_lambda),

        y_ = inv_boxcox(y_, fitted_lambda)

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
def plot_prediction(test, test_pred, train, train_pred, convert_to_original_form = False):

    if convert_to_original_form:

        test = inv_boxcox(test, fitted_lambda),

        test_pred = inv_boxcox(test_pred, fitted_lambda)

        train = inv_boxcox(train, fitted_lambda),

        train_pred = inv_boxcox(train_pred, fitted_lambda)

    f, ax = plt.subplots(1,2, figsize=(10, 5))

    ax1 = plt.subplot(1,2,1)

    sns.distplot(test, hist=True, kde=True, 

             color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4}, ax = ax1)

    sns.distplot(test_pred, hist=True, kde=True, 

             color = 'red', 

             hist_kws={'edgecolor':'red'},

             kde_kws={'linewidth': 4}, ax = ax1)

    ax1.set_title("Actual vs Predicted (Test)")

    

    ax2 = plt.subplot(1,2,2)

    sns.distplot(train, hist=True, kde=True, 

             color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4}, ax = ax2)

    sns.distplot(train_pred, hist=True, kde=True, 

             color = 'red', 

             hist_kws={'edgecolor':'red'},

             kde_kws={'linewidth': 4}, ax = ax2)

    ax2.set_title("Actual vs Predicted (Train)")
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_hour = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv")

df_test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv")
df_hour.head()
df_hour["dteday"] = df_hour.datetime.apply(lambda x : x.split()[0])

df_hour["yr"] = df_hour.datetime.apply(lambda x : x.split()[0][:4])

df_hour['yr'] = df_hour.yr.map({'2011': 0, '2012':1})

df_hour["hr"] = df_hour.datetime.apply(lambda x : x.split()[1].split(":")[0])

df_hour["weekday"] = df_hour.dteday.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])

df_hour.weekday = df_hour.weekday.map({'Saturday':6, 'Sunday':0, 'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5})

df_hour["mnth"] = df_hour.dteday.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])

df_hour.mnth = df_hour.mnth.map({'January':0, 'February':1, 'March':2, 'April':3, 'May':4, 'June':5, 'July':6,

       'August':7, 'September':8, 'October':9, 'November':10, 'December':11})

df_hour["weathersit"] = df_hour.weather

df_hour['dteday'] = pd.to_datetime(df_hour['dteday'])

del df_hour['weather']

del df_hour['datetime']
#performing same on test

#performing same on test

df_test["dteday"] = df_test.datetime.apply(lambda x : x.split()[0])

df_test["yr"] = df_test.datetime.apply(lambda x : x.split()[0][:4])

df_test['yr'] = df_test.yr.map({'2011': 0, '2012':1})

df_test["hr"] = df_test.datetime.apply(lambda x : x.split()[1].split(":")[0])

df_test["weekday"] = df_test.dteday.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])

df_test.weekday = df_test.weekday.map({'Saturday':6, 'Sunday':0, 'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5})

df_test["mnth"] = df_test.dteday.apply(lambda dateString : calendar.month_name[datetime.strptime(dateString,"%Y-%m-%d").month])

df_test.mnth = df_test.mnth.map({'January':0, 'February':1, 'March':2, 'April':3, 'May':4, 'June':5, 'July':6,

       'August':7, 'September':8, 'October':9, 'November':10, 'December':11})

df_test["weathersit"] = df_test.weather

df_test['dteday'] = pd.to_datetime(df_test['dteday'])

del df_test['weather']
df_hour['cnt'] = df_hour['count']

del df_hour['count']

df_hour.shape
df_hour.info()
df_hour.columns
#Creating a copy of df to preserve original df

df = df_hour.copy()
#Analysing which of these are categorical variables

for col in df.columns:

  print('Count of unique values for ', col, ': ', len(df[col].unique()))
'''We know maximum categories for any categorical col is 24 (for month)

Hence we can use this to def function to convert variable to categorical type'''

df = change_to_categorical(df, max_cat=24)
df.dtypes
get_numerical_and_categorical_col(df)
dataTypeDf = (df.dtypes.astype(str).value_counts()).reset_index().rename(columns={"index":"variableType",0:"count"})

fig,ax = plt.subplots()

fig.set_size_inches(12,5)

get_count_plot('variableType',dataTypeDf, ax, 'count', value_counts='count', print_percent=True)
df[get_numerical_and_categorical_col(df)[0]].describe()
df[get_numerical_and_categorical_col(df)[1]].describe(include='all')
get_plot_for_numerical(df,3)
plt.figure(figsize=(7,5))

visualize_corr_matrix(df)
'''Let's start creating a list which contains all the variables to be deleted.

We can delete them once we're done with our exploratory analysis'''

cols_to_remove = ['registered','casual','windspeed']

#besides atemp should be deleted immediately for obvious reasons!

del df['atemp']

del df_test['atemp']
get_plot_for_numerical(df, 3, plot_type='box')
get_target_dist_with_categorical(df,n_cols=3)
get_target_dist_with_categorical(df, n_cols=2, plot_type='violin')
#Let's perform categorical test chi2 to decide which categorical columns to delete

chi2_dict = calculate_chi_square_values(df)

chi2_dict
cols_to_remove.append('season')

cols_to_remove.append('holiday')

cols_to_remove.append('weekday')

cols_to_remove.append('weathersit')
sns.pointplot(x='hr',y='cnt',data=df, hue='season', markers = 'x')
sns.pointplot(x='hr',y='cnt',data=df, hue='weekday', markers = 'x')
#to visualize similar plot for type of user, we would need to use melt

#what melt would do, take each hour and generate rows for value variables. Next we'll use this to find mean for each hour and for each type of users

hr_users_type = pd.melt(df[["hr","casual","registered"]], id_vars=['hr'], value_vars=['casual', 'registered']).sort_values(by='hr')

hr_users_type.head()
hr_users_type_mean = pd.DataFrame(hr_users_type.groupby(["hr","variable"],sort=True)["value"].mean()).reset_index()

hr_users_type_mean.head()
sns.pointplot(x=hr_users_type_mean["hr"], y=hr_users_type_mean["value"],hue=hr_users_type_mean["variable"],hue_order=["casual","registered"], data=hr_users_type_mean, join=True)
#We done with visually exploring data, let's just see how many variables we decided to drop

#Also we should be dropping dteday too

cols_to_remove.append('dteday')

cols_to_remove
df.columns
#deleting outliers from all numerical variables

for col in get_numerical_and_categorical_col(df)[0]:

 df = remove_outliers_for_variable_by_std(df, col)
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

# drop target columns

df_original = df.copy()

drop_cols=['cnt', 'dteday','registered','casual']

X = df.drop(drop_cols, axis = 1) # X = independent columns (potential predictors)

y = df['cnt'] # y = target column (what we want to predict)

# instantiate RandomForestClassifier

rf_model = RandomForestRegressor()

rf_model.fit(X,y)

feat_importances = pd.Series(rf_model.feature_importances_, index=X.columns)

# determine 20 most important features

df_imp_feat = feat_importances.nlargest(20)

df_imp_feat.plot(kind='bar')

plt.show()

print(df_imp_feat)

print('Comparing with our columns')

print(cols_to_remove)
df_cleaned = df.copy()

df_cleaned.drop(cols_to_remove, axis=1, inplace=True)

df_test.drop(cols_to_remove, axis=1, inplace=True, errors='ignore')

df_cleaned.head()
df_cleaned.describe()
# transform training data & save lambda value 

fitted_data, fitted_lambda = stats.boxcox(df_cleaned.cnt)

df_cleaned['cnt_box_cox'] = fitted_data

df_cleaned['cnt_log'] = np.log(df_cleaned.cnt)

#to be used for last step

df_cleaned['box_cox_reverse'] = inv_boxcox(fitted_data, fitted_lambda)



get_plot_for_numerical(df_cleaned, 3, list_col=['cnt','cnt_box_cox','cnt_log','box_cox_reverse'])
df_cleaned.drop(['cnt','cnt_log','box_cox_reverse'], axis=1, inplace=True)

df_cleaned.rename(columns={'cnt_box_cox':'count_transformed'}, inplace=True)
sc = StandardScaler()
target = 'count_transformed'

X = df_cleaned.drop(target, axis=1)

y = df_cleaned[target]

seed=23

X_train, X_test, y_train, y_test = split(X, y, test_size=.3, random_state=seed)
X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)
from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import mean_squared_error as mse

from sklearn import metrics



# Initialize logistic regression model

lr = LinearRegression()



# Train the model

lr.fit(X_train,y = y_train)



# Make predictions

y_pred = lr.predict(X_test)



y_pred_train = lr.predict(X_train)



print('RMSLE for test: ',rmsle(y_test, y_pred, True))

print('RMSLE for train: ',rmsle(y_train, y_pred_train, True))
coeff_df = pd.DataFrame(lr.coef_, X.columns, columns=['Coefficient'])  

coeff_df
plot_prediction(y_test, y_pred, y_train, y_pred_train, True)
ridge = Ridge()

ridge_param = {'max_iter':[3000], 'alpha':[.1,.03,.3,1,3,10, 30, 100,300]}

rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)

grid_ridge = GridSearchCV(ridge,

                           ridge_param,

                           scoring = rmsle_scorer,

                           cv = 10)

grid_ridge.fit(X_train, y_train)

y_pred_ridge = grid_ridge.predict(X_test)

y_pred_ridge_train = grid_ridge.predict(X_train)

print('Grid Ridge Best Params: ', grid_ridge.best_params_)

print('RMSLE for test: ',rmsle(y_test, y_pred_ridge, True))

print('RMSLE for train: ',rmsle(y_train, y_pred_ridge_train, True))
fig,ax= plt.subplots()

fig.set_size_inches(12,5)

df = pd.DataFrame(grid_ridge.cv_results_)

df["alpha"] = df["params"].apply(lambda x:x["alpha"])

df["rmsle"] = df["mean_test_score"].apply(lambda x:-x)

sns.pointplot(data=df,x="alpha",y="rmsle",ax=ax)
plot_prediction(y_test, y_pred_ridge, y_train, y_pred_ridge_train, True)
lasso = Lasso()

alpha = 1/np.array([.1,.03,.3,1,3,10, 30, 100,300,1000])

lasso_param = {'max_iter':[3000], 'alpha':alpha}

rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)

random_lasso = RandomizedSearchCV(lasso,

                           lasso_param,

                           scoring = rmsle_scorer,

                           cv = 10)

random_lasso.fit(X_train, y_train)

y_pred_lasso = random_lasso.predict(X_test)

y_pred_lasso_train = random_lasso.predict(X_train)

print('Random Lasso Best Params: ', random_lasso.best_params_)

print('RMSLE for test: ',rmsle(y_test, y_pred_lasso, True))

print('RMSLE for train: ',rmsle(y_train, y_pred_lasso_train, True))
fig,ax= plt.subplots()

fig.set_size_inches(12,5)

df = pd.DataFrame(random_lasso.cv_results_)

df["alpha"] = df["params"].apply(lambda x:x["alpha"])

df["rmsle"] = df["mean_test_score"].apply(lambda x:-x)

sns.pointplot(data=df,x="alpha",y="rmsle",ax=ax)
plot_prediction(y_test, y_pred_lasso, y_train, y_pred_lasso_train, True)
from sklearn.tree import DecisionTreeRegressor as dt

dt_m = dt(random_state=0)

dt_m.fit(X_train,y_train)

y_pred_dt=dt_m.predict(X_test)

y_pred_dt_train=dt_m.predict(X_train)

print('RMSLE for test: ',rmsle(y_test, y_pred_dt, True))

print('RMSLE for train: ',rmsle(y_train, y_pred_dt_train, True))
plot_prediction(y_test, y_pred_dt, y_train, y_pred_dt_train, True)
from sklearn.ensemble import RandomForestRegressor as rfr

rf = rfr(n_estimators=100)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

y_pred_rf_train = rf.predict(X_train)

print('RMSLE for test: ',rmsle(y_test, y_pred_rf, True))

print('RMSLE for train: ',rmsle(y_train, y_pred_rf_train, True))
plot_prediction(y_test, y_pred_rf, y_train, y_pred_rf_train, True)
from sklearn.ensemble import GradientBoostingRegressor

gbm = GradientBoostingRegressor(n_estimators=3000,alpha=.03)

gbm.fit(X_train,y_train)

y_pred_gbm = gbm.predict(X_test)

y_pred_gbm_train = gbm.predict(X_train)

print('RMSLE for test: ',rmsle(y_test, y_pred_gbm, True))

print('RMSLE for train: ',rmsle(y_train, y_pred_gbm_train, True))
plot_prediction(y_test, y_pred_gbm, y_train, y_pred_gbm_train, True)
df_test = df_test.sort_values(by='datetime')

datetime_series = df_test.datetime

df_test_for_model = df_test.copy()

df_test_for_model.drop(['datetime'], inplace=True, axis=1)

X_test_ndarry = df_test_for_model.to_numpy()

final_X_test = sc.fit_transform(X_test_ndarry)

final_y_pred = inv_boxcox(gbm.predict(final_X_test), fitted_lambda)
final_y_pred_rf = inv_boxcox(rf.predict(final_X_test), fitted_lambda)
final_y_pred.shape, datetime_series.shape
submission = pd.DataFrame({'datetime':datetime_series, 'count':np.round(final_y_pred)})
submission_rf = pd.DataFrame({'datetime':datetime_series, 'count':final_y_pred_rf})
submission.head()
sns.distplot(final_y_pred, hist=True, kde=True, 

             color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})
#Just to avoid any missing values

submission[submission['count'].isna()]
#let's see values around it to fill these

#intuition knn

submission.iloc[721],submission.iloc[720],submission.iloc[719]
#By this and also by our initial analysis, afternoon is not a preferable time to ride bike

submission.fillna(0, inplace=True)
submission.iloc[725],submission.iloc[726],submission.iloc[727]
submission.to_csv('bike_predictions_rounded.csv', index=False)

submission.to_csv('bike_predictions_random_forest.csv', index=False)