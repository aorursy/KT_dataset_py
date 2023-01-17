# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Matplotlib and seaborn for visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Scipy for statistics

from scipy import stats



# os to manipulate files

import os



from sklearn.metrics import mean_absolute_error,r2_score

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures



colors = [ "#3498db", "#e74c3c", "#2ecc71","#9b59b6", "#34495e", "#95a5a6"]
df = pd.read_csv('../input/hackathon/task_2-owid_covid_data-21_June_2020.csv')

df.head()
# Lets first handle numerical features with nan value

numerical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes!='O']

numerical_nan
## Replacing the numerical Missing Values



for feature in numerical_nan:

    ## We will replace by using median since there are outliers

    median_value=df[feature].median()

    

    df[feature].fillna(median_value,inplace=True)

    

df[numerical_nan].isnull().sum()
# categorical features with missing values

categorical_nan = [feature for feature in df.columns if df[feature].isna().sum()>1 and df[feature].dtypes=='O']

print(categorical_nan)
# replacing missing values in categorical features

for feature in categorical_nan:

    df[feature] = df[feature].fillna('None')
from sklearn.preprocessing import LabelEncoder, StandardScaler

#fill in mean for floats

for c in df.columns:

    if df[c].dtype=='float16' or  df[c].dtype=='float32' or  df[c].dtype=='float64':

        df[c].fillna(df[c].mean())



#fill in -999 for categoricals

df = df.fillna(-999)

# Label Encoding

for f in df.columns:

    if df[f].dtype=='object': 

        lbl = LabelEncoder()

        lbl.fit(list(df[f].values))

        df[f] = lbl.transform(list(df[f].values))

        

print('Labelling done.')
def convert_to_number(col,convert_type=int,changes = ['-']):

    

    # string will be considered as object

    if col.dtype.name == 'object':

        col_temp = col.copy()

        

        # Change any occurence in changes to ''

        for change in changes:

                col_temp = col_temp.str.replace(change,'')

                

        # Changes empty string elements for NaN

        col_temp.loc[(col_temp == '')] = np.nan

        

        # Convert to number the not nan elements

        col_temp[col_temp.notna()] = col_temp[col_temp.notna()].astype(convert_type)

        

        # Fill nan elements with the mean

        col_temp = col_temp.fillna(int(col_temp.mean()))

        

        return col_temp

    else:

        return col
def plot_predictions(X_new,y_new,descr = ''):

    y_col = 'total_cases'



    #cols = ['area', 'hoa','rent amount','property tax','fire insurance']

    cols = ['total_deaths', 'new_cases','total_cases_per_million','population', 'total_tests', 'total_deaths_per_million', 'aged_65_older']

    k = 0

    for x_col in cols:

        plt.close()

        plt.figure(figsize=(8, 5))

        plt.scatter(X_trn[x_col],y_trn,c='lightgray',label = 'Training Dataset',marker='o',zorder=1)

        plt.scatter(X_new[x_col],y_new, label = 'Predictions on Test Dataset',marker='.', c=colors[k], lw = 0.5,zorder=2,alpha = 0.8)

        #plt.scatter(X_tst[x_col],y_pr_tst, label = 'Predictions',marker='.', c='tab:blue', lw = 0.5,zorder=2)





        plt.xlabel(x_col, size = 18)

        plt.ylabel(y_col, size = 18); 

        plt.legend(prop={'size': 12});

        plt.title(descr+y_col+' vs '+x_col, size = 20);

        plt.show()

        k += 1
# Import Dataset

df1 = pd.read_csv('../input/hackathon/task_2-owid_covid_data-21_June_2020.csv').drop('new_tests_smoothed',axis=1)

#df2 = pd.read_csv(os.path.join(dirname,'houses_to_rent_v2.csv'))

#df2.columns = df1.columns



# elements to remove from the dataset

remove = ['R','$',',','-','Sem info','Incluso']



# columns of numerical data

cols = ['total_cases', 'total_deaths', 'new_cases','total_cases_per_million','population', 'total_tests', 'total_deaths_per_million', 'aged_65_older']



# Making the substitutions

for col in cols:

    df1[col]  = convert_to_number(df1[col],changes=remove)

    

# converting floor to int 

#df1['total_cases'] = df1['total_cases'].astype('int')



# Getting dummies

#df1[['continent','date', 'iso_code', 'location', 'tests_units']] = pd.get_dummies(df1[['continent','date', 'iso_code', 'location', 'tests_units']], prefix_sep='_', drop_first=True)



# dealing with outliers

cols = ['total_cases', 'total_deaths', 'new_cases','total_cases_per_million','population', 'total_tests', 'total_deaths_per_million', 'aged_65_older']

for col in cols:

    df1 = df1[np.abs(stats.zscore(df1[col])) < 6]
correlations = df.corr()['total_tests'].abs().sort_values(ascending=False).drop('total_tests',axis=0).to_frame()

correlations.plot(kind='bar');
#totalprice correlation matrix

k = 10 #number of variables for heatmap

plt.figure(figsize=(16,8))

corr = df.corr()



hm = sns.heatmap(corr, 

                 cbar=True, 

                 annot=True, 

                 square=True, fmt='.2f', 

                 annot_kws={'size': 10}, 

                 yticklabels=corr.columns.values,

                 xticklabels=corr.columns.values,

                 cmap="YlGnBu")

plt.show()
# Selecting features and target

x_col = 'total_tests'

y_col = 'total_cases'



X = df[[x_col]]

y = df[y_col]



# splitting

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.333)



# Create regression object

MLR = linear_model.LinearRegression()



poly = PolynomialFeatures(degree=1)

X_trn_pl = poly.fit_transform(X_trn)

X_tst_pl = poly.fit_transform(X_tst)





# Train the model using the training sets

MLR.fit(X_trn_pl,y_trn)



y_pr_tst = MLR.predict(X_tst_pl)

y_pr_trn = MLR.predict(X_trn_pl)



mae = mean_absolute_error(y_tst,y_pr_tst)

r2 = r2_score(y_tst,y_pr_tst)



print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))

print('R2:{:6.3f}'.format(r2))



# Plotting

plt.figure(figsize=(8, 5))

plt.scatter(X_tst,y_tst,c='lightgray',label = 'observations',alpha = 0.6,marker='.',zorder=1)

plt.plot(X_tst,y_pr_tst, label = 'Predictions', c='tab:blue', lw = 3,zorder=2)

plt.xlabel(x_col, size = 18)

plt.ylabel(y_col, size = 18); 

plt.legend(prop={'size': 16});

plt.title(y_col+' vs '+x_col, size = 20);
# Selecting features and target

X = df[['total_tests','population']]

y = df['total_cases']



# splitting

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.333)



# Create regression object

MLR = linear_model.LinearRegression()



poly = PolynomialFeatures(degree=1)

X_trn_pl = poly.fit_transform(X_trn)

X_tst_pl = poly.fit_transform(X_tst)



# Train the model using the training sets

MLR.fit(X_trn_pl,y_trn)



y_pr_tst = MLR.predict(X_tst_pl)

y_pr_trn = MLR.predict(X_trn_pl)



mae = mean_absolute_error(y_tst,y_pr_tst)

r2 = r2_score(y_tst,y_pr_tst)





print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))

print('R2:{:6.3f}'.format(r2))
# Selecting features and target

X = df[['total_tests','total_cases_per_million']]

y = df['total_cases']



# splitting

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.333)



# Create regression object

MLR = linear_model.LinearRegression()



poly = PolynomialFeatures(degree=1)

X_trn_pl = poly.fit_transform(X_trn)

X_tst_pl = poly.fit_transform(X_tst)



# Train the model using the training sets

MLR.fit(X_trn_pl,y_trn)



y_pr_tst = MLR.predict(X_tst_pl)

y_pr_trn = MLR.predict(X_trn_pl)



mae = mean_absolute_error(y_tst,y_pr_tst)

r2 = r2_score(y_tst,y_pr_tst)





print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))

print('R2:{:6.3f}'.format(r2))
## Selecting features and target

X = df[['aged_65_older','stringency_index']]

y = df['total_cases']



# splitting

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.333)



# Create regression object

MLR = linear_model.LinearRegression()



poly = PolynomialFeatures(degree=1)

X_trn_pl = poly.fit_transform(X_trn)

X_tst_pl = poly.fit_transform(X_tst)



# Train the model using the training sets

MLR.fit(X_trn_pl,y_trn)



y_pr_tst = MLR.predict(X_tst_pl)

y_pr_trn = MLR.predict(X_trn_pl)



mae = mean_absolute_error(y_tst,y_pr_tst)

r2 = r2_score(y_tst,y_pr_tst)





print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))

print('R2:{:6.3f}'.format(r2))
# Selecting features and target

X = df[['diabetes_prevalence','new_cases']]

y = df['total_cases']



# splitting

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.333)



# Create regression object

MLR = linear_model.LinearRegression()



poly = PolynomialFeatures(degree=1)

X_trn_pl = poly.fit_transform(X_trn)

X_tst_pl = poly.fit_transform(X_tst)



# Train the model using the training sets

MLR.fit(X_trn_pl,y_trn)



y_pr_tst = MLR.predict(X_tst_pl)

y_pr_trn = MLR.predict(X_trn_pl)



mae = mean_absolute_error(y_tst,y_pr_tst)

r2 = r2_score(y_tst,y_pr_tst)





print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))

print('R2:{:6.3f}'.format(r2))
# Selecting features and target

X = df.drop(['total_cases','new_tests_smoothed_per_thousand'],axis=1).copy()

y = df['total_cases'].copy()



# splitting

X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.333)
MLR = linear_model.LinearRegression()



poly = PolynomialFeatures(degree=2)

X_trn_pl = poly.fit_transform(X_trn)

X_tst_pl = poly.fit_transform(X_tst)

MLR.fit(X_trn_pl,y_trn)



y_pr_tst = MLR.predict(X_tst_pl)

mae = mean_absolute_error(y_tst,y_pr_tst)

r2 = r2_score(y_tst,y_pr_tst)



print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))

print('R2:{:6.3f}'.format(r2))
# Getting a reduced sample to test

size = len(X_tst)

indexes = np.random.choice(len(X_tst), size, replace=False)

X_new = X_tst.iloc[indexes]

y_new = MLR.predict(poly.fit_transform(X_new))



plot_predictions(X_new,y_new,descr = 'Linear Regression: ')
from sklearn.tree import DecisionTreeRegressor



d_tree = DecisionTreeRegressor()

d_tree.fit(X_trn,y_trn)



y_pr_tst = d_tree.predict(X_tst)



mae = mean_absolute_error(y_tst,y_pr_tst)

r2 = r2_score(y_tst,y_pr_tst)



print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))

print('R2:{:6.3f}'.format(r2))
# Getting a reduced sample to test

size = len(X_tst)

indexes = np.random.choice(len(X_tst), size, replace=False)

X_new = X_tst.iloc[indexes]

y_new = d_tree.predict(X_new)



plot_predictions(X_new,y_new,descr = 'Decision Tree: ')
from sklearn.ensemble import RandomForestRegressor



rnd_frst = RandomForestRegressor()

rnd_frst.fit(X_trn,y_trn)



y_pr_tst = rnd_frst.predict(X_tst)



mae = mean_absolute_error(y_tst,y_pr_tst)

r2 = r2_score(y_tst,y_pr_tst)



print('MAE:{:7.2f},{:7.2f}% of mean'.format(mae,100*mae/y_pr_tst.mean()))

print('R2:{:6.3f}'.format(r2))
# Getting a reduced sample to test

size = len(X_tst)

indexes = np.random.choice(len(X_tst), size, replace=False)

X_new = X_tst.iloc[indexes]

y_new = rnd_frst.predict(X_new)



plot_predictions(X_new,y_new,descr = 'Random Forest: ')