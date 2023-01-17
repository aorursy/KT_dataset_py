import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline
plt.rcParams['figure.figsize'] = 12,6
import warnings
warnings.filterwarnings('ignore')
seq_col_brew = sns.color_palette("YlGnBu_r", 4)
sns.set_palette(seq_col_brew)
dataset = pd.read_csv('../input/train.tsv', sep='\t', usecols=['item_condition_id', 'shipping', 'brand_name', 'price', 'category_name'])
dataset.head()
x = dataset[['item_condition_id', 'shipping', 'brand_name', 'category_name']]
y = dataset['price']
dataset.info(memory_usage='deep')
x.isnull().sum()
x.shape
x['brand_name'].fillna('Other', inplace=True)
x.isnull().sum()
dataset[dataset['category_name'].isnull()]
x.isnull().sum()
x.shape
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.80, random_state = 0)
x_train.info()
x_train.shape
x_train.isnull().sum()

x_train.head()
#To check Gaussian Distribution.
sns.distplot(y)
#Log distribution
sns.distplot(np.log(y + 1))
sns.distplot(y[y < 100])
y.mode()
y.mean()
y.median()
#How do you know what a high standard deviation is? 
#https://www.researchgate.net/post/What_do_you_consider_a_good_standard_deviation
y.std()
#Viewing the an independent variable and it's different distribution 
x_train.item_condition_id.hist()
#1 if shipping fee is paid by seller and 0 by buyer
x_train.shipping.hist()
#The seller takes a hit by paying for the shipping him/herself
sns.barplot(x='shipping', y='price', data=dataset)
dataset.describe(include=['O']) 
dataset.describe()
sns.kdeplot(data=y, shade=True, bw=.85)
sns.barplot(x='item_condition_id', y='price', hue='shipping', data=dataset)
sns.barplot(x='item_condition_id', y='price', data=dataset)

x_train["brand_name"].value_counts()

x_train.isnull().sum()
#Splitting the categories into sub categories
category_columns = ['Top_Level_Category'] + ['Second_Level_Category'] + ['Third_Level_Category']
category_columns

x_train.head()
new_categories = x_train['category_name'].str.extract('(\w*)\/(\w*)\/(\w*)', expand=True)
new_categories.columns
new_categories.columns = category_columns
x_train = pd.concat([x_train, new_categories], axis=1)
x_train.head()
columns = ['Top_Level_Category', 'Second_Level_Category', 'Third_Level_Category']
x_train.isnull().sum()
new_categories.isnull().sum()

for col in columns:
   x_train[col].fillna('Other', inplace=True)
x_train.head()
x_train = x_train.drop('category_name', axis=1)

x_train['brand_name'] = x_train['brand_name'].astype('category').cat.codes
x_train['Top_Level_Category'] = x_train['Top_Level_Category'].astype('category').cat.codes
x_train['Second_Level_Category'] = x_train['Second_Level_Category'].astype('category').cat.codes
x_train['Third_Level_Category'] = x_train['Third_Level_Category'].astype('category').cat.codes
x_train.head()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
'''linreg = LinearRegression(n_jobs=-1)
cv_scores = (linreg, x_train, y_train, cv=10, scoring='neg_mean_squared_error')
print(cv_scores.mean(), cv_scores.std())'''
submission = pd.read_csv('../input/sample_submission.csv')
pred = np.ones((submission.shape[0]))
pred
pred * y_train.mean()
pred = pred * y_train.mean()
submission.shape

submission['price'] = pred
submission.to_csv('sample_submission.csv', index=False)