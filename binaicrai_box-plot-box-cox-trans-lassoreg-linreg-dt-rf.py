import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.simplefilter('ignore')

warnings.filterwarnings('ignore')

import copy

import math



import sklearn

from sklearn.model_selection import train_test_split



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error, r2_score

from scipy import stats



from IPython.display import Image  

from sklearn.tree import export_graphviz

import graphviz
invertebrate_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Invertebrate/Invertebrate_dataset.csv')

invertebrate_data.head()
test_new = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Invertebrate/Invertebrate_new_test_data.csv')

test_new.head()
print(invertebrate_data.shape)

print(test_new.shape)
invertebrate_data.dtypes
test_new.dtypes
invertebrate_data.isna().sum()
test_new.isna().sum()
for i in invertebrate_data.columns:

    if invertebrate_data[i].nunique() == 1:

        print('With only 1 unique value: ', i)

    if invertebrate_data[i].nunique() == invertebrate_data.shape[0]:

        print('With all unique value: ', i)
for i in test_new.columns:

    if test_new[i].nunique() == 1:

        print('With only 1 unique value: ', i)

    if test_new[i].nunique() == test_new.shape[0]:

        print('With all unique value: ', i)
plt.figure(figsize = (12,10))

sns.heatmap(invertebrate_data.corr(), vmin=invertebrate_data.values.min(), vmax=1, 

            annot=True, annot_kws={"size":14}, square = False)

plt.show()
# Create correlation matrix

corr_matrix = invertebrate_data.corr().abs()

corr_matrix
# Select upper triangle of correlation matrix as lower does not remove the diagonal 1s

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.85

# Just to illustrate taking 0.85

to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]

to_drop
# Then df with dropped feature/s

# new_df = invertebrate_data.drop(invertebrate_data[to_drop], axis=1)

# new_df.columns
X = invertebrate_data.copy().drop('SWI', axis = 1)

y = invertebrate_data['SWI']
fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=1, ncols=5, figsize=(18,8))



label1 = ['SWF']

label2 = ['temperature']

label3 = ['size']

label4 = ['management']

label5 = ['duration']



# Box plot for SWF

bplot1 = ax1.boxplot(X['SWF'],

                     vert=True,  # vertical box alignment

                     patch_artist=True,  # fill with color

                     labels = label1)  # will be used to label x-ticks

ax1.set_title('Box plot for SWF')



# Box plot for temperature

bplot2 = ax2.boxplot(X['temperature'],

                     vert=True,  # vertical box alignment

                     patch_artist=True,  # fill with color

                     labels = label2)  # will be used to label x-ticks

ax2.set_title('Box plot for temperature')



# Box plot for size

bplot3 = ax3.boxplot(X['size'],

                     vert=True,  # vertical box alignment

                     patch_artist=True,  # fill with color

                     labels = label3)  # will be used to label x-ticks

ax3.set_title('Box plot for size')



# Box plot for management

bplot4 = ax4.boxplot(X['management'],

                     vert=True,  # vertical box alignment

                     patch_artist=True,  # fill with color

                     labels = label4)  # will be used to label x-ticks

ax4.set_title('Box plot for management')



# Box plot for duration

bplot5 = ax5.boxplot(X['duration'],

                     vert=True,  # vertical box alignment

                     patch_artist=True,  # fill with color

                     labels = label5)  # will be used to label x-ticks

ax5.set_title('Box plot for duration')



# Fill with colors

colors = ['orange']

for bplot in (bplot1, bplot2, bplot3, bplot4, bplot5):

    for patch, color in zip(bplot['boxes'], colors):

        patch.set_facecolor(color)



# Adding horizontal grid lines

for ax in [ax1, ax2, ax3, ax4, ax5]:

    ax.yaxis.grid(True)

    ax.set_xlabel('Variables')

    ax.set_ylabel('Observed values')



plt.show()
plt.figure(figsize = (15,8))

sns.distplot(X['SWF'], hist=True, kde=True, 

             color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})
plt.figure(figsize = (15,8))

sns.distplot(X['temperature'], hist=True, kde=True, 

             color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})
# Helpful video on Box-Cox. I had to watch it several times.



from IPython.display import YouTubeVideo



def display_yotube_video(url, **kwargs):

    """

    Displays a Youtube video in a Jupyter notebook.

    

    Args:

        url (string): a link to a Youtube video.

        **kwargs: further arguments for IPython.display.YouTubeVideo

    

    Returns:

        YouTubeVideo: a video that is displayed in your notebook.

    """

    id_ = url.split("=")[-1]

    return YouTubeVideo(id_, **kwargs)



display_yotube_video("https://www.youtube.com/watch?v=2gVA3TudAXI", width=600, height=400)
X_norm = X.copy()



# Transform training data & save lambda value

X_norm['temperature'], fitted_lambda = stats.boxcox(X_norm['temperature'])

print('Skewness before: ', X['temperature'].skew())

print('Skewness after BCT: ', X_norm['temperature'].skew())
fig, ax=plt.subplots(1,2, figsize = (15,8))

sns.distplot(X['temperature'], hist=True, kde=True, color = 'darkblue', 

             hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 4}, ax=ax[0])

sns.distplot(X_norm['temperature'], hist=True, kde=True, color = 'darkblue', 

             hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 4}, ax=ax[1])
test_norm = test_new.copy()

# Use lambda value to transform test data

test_norm['temperature'] = stats.boxcox(test_new['temperature'], fitted_lambda)

print('Skewness before: ', test_new['temperature'].skew())

print('Skewness after BCT: ', test_norm['temperature'].skew())
fig, ax=plt.subplots(1,2, figsize = (15,8))

sns.distplot(test_new['temperature'], hist=True, kde=True, color = 'darkblue', 

             hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 4}, ax=ax[0])

sns.distplot(test_norm['temperature'], hist=True, kde=True, color = 'darkblue', 

             hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 4}, ax=ax[1])
sns.catplot('management', data= invertebrate_data, kind='count', alpha=0.7, height=4, aspect= 3)



# Get current axis on current figure

ax = plt.gca()



# Max value to be set

y_max = invertebrate_data['management'].value_counts().max() 



# Iterate through the list of axes' patches

for p in ax.patches:

    ax.text(p.get_x() + p.get_width()/5., p.get_height(),'%d' % int(p.get_height()),

            fontsize=13, color='blue', ha='center', va='bottom')

plt.title('Frequency plot of management - train', fontsize = 14, color = 'black')

plt.show()
print('For Train')

d1 = X_norm.nunique()

print(sorted(d1))

print("==============================")

print('For Test')

d2 = test_norm.nunique()

print(sorted(d2))
col_train = X_norm.columns

col_test = test_norm.columns
l1 = []

for i in col_train:

    if X_norm[i].nunique() <= 22:

        l1.append(i)
l2 = []

for i in col_test:

    if test_norm[i].nunique() <= 22:

        l2.append(i)
# Checking the columns in train and test are same or not

df = pd.DataFrame(l1, columns = ['train'])

df['test'] = pd.DataFrame(l2)

df
# For now directly changing management to categorical without creating subsets

X_norm[l1] = X_norm[l1].apply(lambda x: x.astype('category'), axis=0)

test_norm[l2] = test_norm[l2].apply(lambda x: x.astype('category'), axis=0)

print('train dtypes:')

print(X_norm[l1].dtypes)

print('======================================')

print('test dtypes:')

print(test_norm[l1].dtypes)
l1
# Function to create dummies

def dummy(train, test, cols):

    X_num = len(train)

    combined_dataset = pd.concat(objs=[train, test], axis=0)

    combined_dataset = pd.get_dummies(combined_dataset, columns=cols, drop_first=True)

    train = copy.copy(combined_dataset[:X_num])

    test = copy.copy(combined_dataset[X_num:])
dummy(X, test_new, l1)

X_num = len(X)

combined_dataset = pd.concat(objs=[X, test_new], axis=0)

combined_dataset = pd.get_dummies(combined_dataset, columns=l1, drop_first=True)

X = copy.copy(combined_dataset[:X_num])

test = copy.copy(combined_dataset[X_num:])
print(X.shape)

print(y.shape)

print(test.shape)
dummy(X_norm, test_norm, l1)

X_norm_num = len(X_norm)

combined_dataset_norm = pd.concat(objs=[X_norm, test_norm], axis=0)

combined_dataset_norm = pd.get_dummies(combined_dataset_norm, columns=l1, drop_first=True)

X_norm = copy.copy(combined_dataset_norm[:X_norm_num])

test_norm = copy.copy(combined_dataset_norm[X_norm_num:])
print(X_norm.shape)

print(y.shape)

print(test_norm.shape)
# Splitting into train and validation sets for non BCT (Box-Cox Transformed) data

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state = 50)
# Splitting into train and validation sets for BCT data

X_train_bct, X_val_bct, y_train_bct, y_val_bct = train_test_split(X_norm, y, test_size=0.2, random_state = 50)
print('On non BCT data')

print('---------------')

lasso_reg = LassoCV(cv=5, random_state=1)

lasso_reg.fit(X_train, y_train)

print("Best alpha using built-in LassoCV: %f" % lasso_reg.alpha_)

print("Best score using built-in LassoCV: %f" % lasso_reg.score(X_train,y_train))

coef = pd.Series(lasso_reg.coef_, index = X_train.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
# Important features

imp_coef = coef.sort_values()

ax1 = imp_coef.plot(kind = "barh" )

plt.rcParams['figure.figsize'] = (15.0, 10.0)
print('On BCT data')

print('---------------')

lasso_reg_norm = LassoCV(cv=5, random_state=1)

lasso_reg_norm.fit(X_train_bct, y_train_bct)

print("Best alpha using built-in LassoCV: %f" % lasso_reg_norm.alpha_)

print("Best score using built-in LassoCV: %f" % lasso_reg_norm.score(X_train_bct,y_train_bct))

coef_norm = pd.Series(lasso_reg_norm.coef_, index = X_train_bct.columns)
print("Lasso picked " + str(sum(coef_norm != 0)) + " variables and eliminated the other " +  

      str(sum(coef_norm == 0)) + " variables")
# Important features

imp_coef_norm = coef_norm.sort_values()

ax1 = imp_coef_norm.plot(kind = "barh" )

plt.rcParams['figure.figsize'] = (15.0, 10.0)
# Predict (train)

y_train_pred = lasso_reg.predict(X_train)



# Model evaluation

mse = mean_squared_error(y_train, y_train_pred)

r2 = r2_score(y_train, y_train_pred)

rmse = math.sqrt(mse)

print('On non BCT data')

print('----------------')

print('R-squared: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred = lasso_reg.predict(X_val)



# Model evaluation (val)

mse = mean_squared_error(y_val, y_val_pred)

r2 = r2_score(y_val, y_val_pred)

rmse = math.sqrt(mse)

print('On non BCT data')

print('----------------')

print('R-squared: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (train)

y_train_pred_bct = lasso_reg_norm.predict(X_train_bct)



# Model evaluation

mse = mean_squared_error(y_train_bct, y_train_pred_bct)

r2 = r2_score(y_train_bct, y_train_pred_bct)

rmse = math.sqrt(mse)

print('On BCT data')

print('----------------')

print('R-squared: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred_bct = lasso_reg_norm.predict(X_val_bct)



# Model evaluation (val)

mse = mean_squared_error(y_val_bct, y_val_pred_bct)

r2 = r2_score(y_val_bct, y_val_pred_bct)

rmse = math.sqrt(mse)

print('On non BCT data')

print('----------------')

print('R-squared: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(X_train, y_train)
# Predict (train)

y_train_pred = linreg.predict(X_train)



# Model evaluation (train)

mse = mean_squared_error(y_train, y_train_pred)

r2 = r2_score(y_train, y_train_pred)

rmse = math.sqrt(mse)

print('On non BCT data')

print('----------------')

print('R-squared: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred = linreg.predict(X_val)



# Model evaluation (val)

mse = mean_squared_error(y_val, y_val_pred)

r2 = r2_score(y_val, y_val_pred)

rmse = math.sqrt(mse)

print('On non BCT data')

print('----------------')

print('R-squared: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
linreg_bct = LinearRegression()

linreg_bct.fit(X_train_bct, y_train_bct)
# Predict (train)

y_train_pred_bct = linreg_bct.predict(X_train_bct)



# Model evaluation

mse = mean_squared_error(y_train_bct, y_train_pred_bct)

r2 = r2_score(y_train_bct, y_train_pred_bct)

rmse = math.sqrt(mse)

print('On BCT data')

print('----------------')

print('R-squared: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred_bct = linreg_bct.predict(X_val_bct)



# Model evaluation (val)

mse = mean_squared_error(y_val_bct, y_val_pred_bct)

r2 = r2_score(y_val_bct, y_val_pred_bct)

rmse = math.sqrt(mse)

print('On non BCT data')

print('----------------')

print('R-squared: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# You can try by removing the max_depth and see the output for Predict (train)
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=50, max_depth=4)

dt.fit(X_train, y_train)
# Predict (train)

y_train_pred = dt.predict(X_train)



# Model evaluation (train)

mse = mean_squared_error(y_train, y_train_pred)

r2 = r2_score(y_train, y_train_pred)

rmse = math.sqrt(mse)

print('On non BCT data')

print('----------------')

print('R-squared: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred = dt.predict(X_val)



# Model evaluation (val)

mse = mean_squared_error(y_val, y_val_pred)

r2 = r2_score(y_val, y_val_pred)

rmse = math.sqrt(mse)

print('On non BCT data')

print('----------------')

print('R-squared: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
from sklearn.ensemble import RandomForestRegressor

rf1 = RandomForestRegressor()

rf1.fit(X = X_train,y = y_train)
# Predict (train)

y_train_pred_rf = rf1.predict(X_train)



# Model evaluation (train)

mse = mean_squared_error(y_train, y_train_pred_rf)

r2 = r2_score(y_train, y_train_pred_rf)

rmse = math.sqrt(mse)

print('On non BCT data')

print('----------------')

print('R-squared: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred_rf = rf1.predict(X_val)



# Model evaluation (val)

mse = mean_squared_error(y_val, y_val_pred_rf)

r2 = r2_score(y_val, y_val_pred_rf)

rmse = math.sqrt(mse)

print('On non BCT data')

print('----------------')

print('R-squared: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
rfgs = RandomForestRegressor(random_state=50)
param_grid = { 

    'n_estimators': [2,3,4,5],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [2,3,4,5],

    'criterion' :['mse']

}
cv_rfc = GridSearchCV(estimator=rfgs, param_grid=param_grid, cv= 5)

cv_rfc.fit(X_train, y_train)
cv_rfc.best_params_
rfgs1 = RandomForestRegressor(random_state=45, **cv_rfc.best_params_)
rfgs1.fit(X_train, y_train)
# Predict (train)

y_train_pred_rf = rfgs1.predict(X_train)



# Model evaluation (train)

mse = mean_squared_error(y_train, y_train_pred_rf)

r2 = r2_score(y_train, y_train_pred_rf)

rmse = math.sqrt(mse)

print('On non BCT data')

print('----------------')

print('R-squared: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)
# Predict (val)

y_val_pred_rf = rfgs1.predict(X_val)



# Model evaluation (val)

mse = mean_squared_error(y_val, y_val_pred_rf)

r2 = r2_score(y_val, y_val_pred_rf)

rmse = math.sqrt(mse)

print('On non BCT data')

print('----------------')

print('R-squared: ', r2)

print('MSE: ', mse)

print('RMSE: ', rmse)