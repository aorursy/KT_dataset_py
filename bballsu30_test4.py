#importing various packages that I will use
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import re
from sklearn.model_selection import KFold

#Loading the Datasets I will be doing analysis on
train = pd.read_table('../input/train.tsv')
test = pd.read_table('../input/test_stg2.tsv')
print('Done With Section 1')
train['is_train'] = 1
test['is_train'] = 0
merged = pd.concat([train.drop(['train_id','price'], axis=1), test.drop('test_id', axis=1)], axis=0)
print('Done With Section 2')
#getting information on the data and dropping outlier values after looking at the percentage of data I am losing
print(train.dtypes)
#print(train[train['price'] == 0.0].shape[0]/train.shape[0])
train = train[train['price']>0.0]
train.shape
print('Done With Section 3')
#Transformed the data to a log transformation to make the target variable normalized
train['LogPrice'] = np.log2(train['price'])
plt.hist(train['LogPrice'], bins=20, edgecolor='black')
plt.title('Log Price Distribution of Mercari Items')
plt.show()
print('Done With Section 4')
# Breaking up the Category Section from 'X/X/X' format to a three categories and handling the missing information on category
train['category_name']=train['category_name'].fillna('Missing/Missing/Missing')
categories = train['category_name']
targets = categories.str.split('/',2)
main = []
sub1 = []
sub2 = []
for i in targets:
    a,b,c = i
    main.append(a)
    sub1.append(b)
    sub2.append(c)
train['Main'], train['Sub1'], train['Sub2'] = main, sub1, sub2
print('Done With Section 5')
# dropped category_name due to redundant data
train = train.drop('category_name', axis=1)
print('Done With Section 6')
train['MainSub1'] = (train['Main'] + '|' + train['Sub1']).values
train['MainSub1Sub2'] = (train['Main'] + '|' + train['Sub1'] +'|'+ train['Sub2']).values
print('Done With Section 7')
#Looking at the median prices of each Main category and creating a dictionary based on it
tbl1 = train.groupby(['Main'])['LogPrice'].median().reset_index()
tbl1dict = dict(zip(tbl1['Main'], tbl1['LogPrice']))
print('Done With Section 8')
#Looking at the median prices of each Sub1 category and creating a dictionary based on it

tbl2 = train.groupby(['Main','Sub1'])['LogPrice'].median().reset_index()
comb = []
for i, j in tbl2.iterrows():
    both = j['Main'] + '|' + j['Sub1']
    comb.append(both)
tbl2['comb'] = comb
tbl2dict = dict(zip(tbl2['comb'], tbl2['LogPrice']))
tbl2dict
print('Done With Section 9')
#Looking at the median prices of each Sub 2 category and creating a dictionary based on it

tbl3 = train.groupby(['Main','Sub1','Sub2'])['LogPrice'].median().reset_index()
comb = []
for i, j in tbl3.iterrows():
    both = j['Main'] + '|' + j['Sub1'] + '|' + j['Sub2']
    comb.append(both)
tbl3['comb'] = comb
tbl3dict = dict(zip(tbl3['comb'], tbl3['LogPrice']))
tbl3dict
print('Done With Section 10')
#resets the index of the train set as some of the rows have been removed so I can iterate through the dataframe
train = train.reset_index().drop('index',axis=1)
print('Done With Section 11')
train['MainMed']= train['Main'].map(tbl1dict)
train['Sub1Med']= train['MainSub1'].map(tbl2dict)
train['Sub2Med']= train['MainSub1Sub2'].map(tbl3dict)
print('Done With Section 12')
#adding the median encoded variables and category names for each item in the Test dataset

test['category_name']=test['category_name'].fillna('Missing/Missing/Missing')
categories = test['category_name']
targets = categories.str.split('/',2)
main = []
sub1 = []
sub2 = []
for i in targets:
    a,b,c = i
    main.append(a)
    sub1.append(b)
    sub2.append(c)
test['Main'], test['Sub1'], test['Sub2'] = main, sub1, sub2
print('Done With Section 13')
test['MainSub1'] = (test['Main'] + '|' + test['Sub1']).values
test['MainSub1Sub2'] = (test['Main'] + '|' + test['Sub1']+ '|' + test['Sub2']).values
print('Done With Section 14')
test['MainMed']= test['Main'].map(tbl1dict)
test['Sub1Med']= test['MainSub1'].map(tbl2dict)
test['Sub2Med']= test['MainSub1Sub2'].map(tbl3dict)
test['Sub2Med'] = test['Sub2Med'].fillna(test['Sub1Med'])
print('Done With Section 15')
train['brand_name'] = train['brand_name'].fillna('missing')
test['brand_name'] = test['brand_name'].fillna('missing')
train.groupby('brand_name')['LogPrice'].median().reset_index()
print('Done With Section 16')
#train['brand_name']=train['brand_name'].fillna('missing')
brandprice = train.groupby('brand_name')['LogPrice'].median().reset_index()
#brandcounts = train.groupby('brand_name')['train_id'].count().reset_index()
brandprice['LogPrice'] = brandprice['LogPrice'].fillna(0.0)#brandprice['LogPrice']
plt.hist(brandprice['LogPrice'], bins=30, edgecolor='black')
#plt.boxplot(brandcounts['train_id'])
plt.title('Log Price Distribution of Mercari Items')
plt.show()
print('Done With Section 17')
brandprice['Category'] = pd.cut(brandprice['LogPrice'], bins=5, labels=[1,2,3,4,5])
brandprice
print('Done With Section 18')
branddict = dict(zip(brandprice['brand_name'], brandprice['Category']))
train['brand_category'] = train['brand_name'].map(branddict)
test['brand_category'] = test['brand_name'].map(branddict)
test['brand_category'] = test['brand_category'].fillna(3)
print('Done With Section 19')
merged['brand_name'] = merged['brand_name'].fillna('missing')
brandcounts = merged.groupby('brand_name')['name'].count().reset_index()
brandcounts.describe
bins = [0, 2, 7, 41, 2000000000000]
freqbin = pd.cut(brandcounts['name'], bins, right=False, labels=[1,2,3,4])
brandcounts['brand_frequency'] = freqbin
brandfreqdict = dict(zip(brandcounts['brand_name'], brandcounts['brand_frequency']))
brandfreqdict
print('Done With Section 20')
train['brand_freq'] = train['brand_name'].map(brandfreqdict)
test['brand_freq'] = test['brand_name'].map(brandfreqdict)
test['brand_freq'] = test['brand_freq'].fillna(1)
print('Done With Section 21')
a = train['item_description'].fillna('').apply(lambda x: str(re.sub('[^ a-zA-Z0-9]','',x)))
b = test['item_description'].fillna('').apply(lambda x: str(re.sub('[^ a-zA-Z0-9]','',x)))
print('Done With Section 22')
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
f = tfidf.fit_transform(a.values)
print('Done With Section 23')
ftest = tfidf.transform(b.values)
print('Done With Section 24')
ftest2 = ftest.toarray()
print('Done With Section 25')
#tfidf.get_feature_names()
f2 = f.toarray()
print('Done With Section 26')
scores = np.array([sum(x) for x in f2])
print('Done With Section 27')
train['idf'] = scores
print('Done With Section 28')
test['idf'] = np.array([sum(x) for x in ftest2])
print('Done With Section 29')
rfcols = train[['item_condition_id','LogPrice','shipping', 'MainMed', 'Sub1Med', 'Sub2Med', 'brand_category', 'brand_freq', 'idf']]
y2=rfcols['LogPrice']
X2 = rfcols.drop('LogPrice', axis=1)
XTest = test[['item_condition_id','shipping', 'MainMed', 'Sub1Med', 'Sub2Med', 'brand_category', 'brand_freq', 'idf']]
rf2 = RandomForestRegressor()
rf2.fit(X2,y2)
pred = rf2.predict(XTest)
print('Done With Section 30')
test['price'] = 2**pred
test_sub = test[['test_id', 'price']]
test_sub.to_csv('./testsub.csv', index=False)
print('Done')