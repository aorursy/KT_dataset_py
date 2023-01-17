import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualization

import seaborn as sns

import matplotlib.pyplot as plt



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC



from sklearn.decomposition import PCA

from sklearn import datasets



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
path_data = '../input/HR_comma_sep.csv'

data_df = pd.read_csv(path_data)
data_df.info()
data_df.describe()
data_df.describe(include=['O'])

# data_df.isnull().any()
g = sns.FacetGrid(data_df, col='left')

g.map(plt.hist, 'last_evaluation', bins=10)

# Plot avergage satisfaction line for orientation

plt.axvline(data_df['last_evaluation'].mean(), color='b', linestyle='dashed', linewidth=2)

plt.show()
data_df.loc[ data_df['salary'] == 'low' ,'salary'] = 0

data_df.loc[ data_df['salary'] == 'medium' ,'salary'] = 1

data_df.loc[ data_df['salary'] == 'high' ,'salary'] = 2



# Convert datatype

data_df['salary'] = data_df['salary'].astype(object).astype(int)
# Compute the correlation matrix

corr = data_df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.yticks(rotation=0) 

plt.xticks(rotation=90) 

plt.show()
# Are some job types more likely to be left or to be satisfied?

data_df[['sales', 'left','satisfaction_level']].groupby(['sales'], as_index= False).mean().sort_values(by='satisfaction_level', ascending=False)
data_df[['sales','average_montly_hours']].groupby(['sales'], as_index= False).mean().sort_values(by='average_montly_hours', ascending=False)
data_df[['sales','salary']].groupby(['sales'], as_index= False).mean().sort_values(by='salary', ascending=False)    
data_df[['promotion_last_5years', 'left']].groupby(['left'], as_index= False).mean()
data_df[['time_spend_company', 'left']].groupby(['time_spend_company'], as_index= False).mean().sort_values(by='left', ascending=False)
# How long do employes stay with the company

data_df[['sales', 'time_spend_company']].groupby(['sales'], as_index= False).mean()
# Are people who were promoted recently staying longer with the company?

data_df[['promotion_last_5years', 'time_spend_company']].groupby(['promotion_last_5years'], as_index= False).mean()
# Those who stay very long 

data_df[['promotion_last_5years', 'time_spend_company']].groupby(['promotion_last_5years'], as_index= False).quantile(q=0.9)
# create new categorial feature for working hours

data_df['hours_cat'] = pd.cut(data_df['average_montly_hours'],3)

data_df[['hours_cat', 'left']].groupby(['hours_cat'], as_index=False).mean().sort_values(by='hours_cat', ascending=True) # check boundary



data_df.loc[ data_df['average_montly_hours'] <= 167 ,'monthly_hours'] = 0

data_df.loc[ (data_df['average_montly_hours'] > 167) &  (data_df['average_montly_hours'] <= 238), 'monthly_hours'] = 1

data_df.loc[ data_df['average_montly_hours'] > 238 ,'monthly_hours'] = 2



data_df[['monthly_hours', 'left']].groupby(['monthly_hours'], as_index= False).mean().sort_values(by='left', ascending=False)



# Drop obsolete hours features

data_df = data_df.drop(['average_montly_hours'], axis=1)

data_df = data_df.drop(['hours_cat'], axis=1)



data_df.head()
sales_mapping = {"hr":0,"accouting":1,"technical":2,"support":3,"sales":4,"marketing":5,"IT":6,"product_mng":7,"RandD":8,"management":9}

data_df['sales'] = data_df['sales'].map(sales_mapping)



# Are there any NaN values? How many?

data_df['sales'].isnull().sum()
# Fill NaN rows with extra job category

data_df['sales'] = data_df['sales'].fillna(10)
row_count = data_df.shape[0]

idx = np.arange(0, row_count, dtype=np.int)

chosen_idx = np.random.choice(row_count, replace=False, size=int(row_count*0.7))

unchosen_idx = np.delete(idx, chosen_idx)



data_train = data_df.iloc[chosen_idx]

data_test = data_df.iloc[unchosen_idx]

train_x = data_train.drop('left',axis=1)

train_y = data_train['left']

test_x = data_test.drop('left',axis=1)
logreg = LogisticRegression()

logreg.fit(train_x, train_y)

pred = logreg.predict(test_x)

acc_log = round(logreg.score(train_x, train_y) * 100, 2)

acc_log
svc = SVC()

svc.fit(train_x, train_y)

Y_pred = svc.predict(test_x)

acc_svc = round(svc.score(train_x, train_y) * 100, 2)

acc_svc