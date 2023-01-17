import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings 

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/train.csv')

df.head()
num_rows = df.shape[0]

num_cols = df.shape[1]

(num_rows, num_cols)
df.info()
df.describe()         

# numerical variables
df.describe(include='object')

# categorical variables
df['rating'].value_counts()
df.isnull().any()
missing_values = df.isnull().sum()

missing_values[missing_values > 0]
# replacing missing values using mean of numerical features

df.fillna(value = df.mean(), inplace=True)
# no missing value present anymore

df.isnull().any().any()
sns.boxplot(x='rating', y='feature3', data = df)

# a lot of overlap => might not be a good feature
sns.boxplot(x='rating', y='feature5', data = df)
sns.boxplot(x='rating', y='feature6', data = df)
sns.boxplot(x='rating', y='feature7', data = df)
# Compute the correlation matrix

corr = df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
sns.boxplot(x='rating', y='feature8', data = df)
sns.boxplot(x='rating', y='feature11', data = df)
sns.distplot(df['feature6'],kde = False)
#df['feature6'] = np.log(df['feature6'])

#sns.distplot(df['feature6'],kde = False)
sns.boxplot(x='type', y='rating', data = df)
df.corr()
cols = ['feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9',

        'feature10','feature11','type']

X = df[cols]

y = df['rating']
X = pd.get_dummies(data=X, columns=['type'])
real_test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
real_test.isnull().sum()
real_test.fillna(real_test.mean(), inplace=True)
X2 = real_test[cols]
X2 = pd.get_dummies(data=X2, columns=['type'])
from sklearn.ensemble import ExtraTreesRegressor



clf = ExtraTreesRegressor(n_estimators=4000,random_state=42, bootstrap=True).fit(X,y)

y_pred1 = clf.predict(X2)

y_pred1 = np.rint(y_pred1)
final = pd.DataFrame(real_test['id'])

final['rating'] = y_pred1
final.head()
final.to_csv('pred4.csv', index=False)
real_test = pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')
real_test.isnull().sum()
real_test.fillna(real_test.mean(), inplace=True)
X2 = real_test[cols]
X2 = pd.get_dummies(data=X2, columns=['type'])

from sklearn.ensemble import ExtraTreesRegressor



clf = ExtraTreesRegressor(n_estimators=2000,random_state=4, bootstrap=True).fit(X,y)

y_pred1 = clf.predict(X2)

y_pred1 = np.rint(y_pred1)
final = pd.DataFrame(real_test['id'])

final['rating'] = y_pred1
final.head()
final.to_csv('pred4.csv', index=False)