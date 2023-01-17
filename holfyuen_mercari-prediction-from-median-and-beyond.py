# import libraries
import numpy as np
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Import data files, check dimensions
train = pd.read_csv('../input/train.tsv', sep = '\t')
# test = pd.read_csv('../input/test.tsv', sep = '\t')
test2 = pd.read_csv('../input/test_stg2.tsv', sep = '\t')
# submit = pd.read_csv('../input/sample_submission.csv')
submit2 = pd.read_csv('../input/sample_submission_stg2.csv')
print (train.shape, test2.shape, submit2.shape)
# Clean data
train['item_condition_id'] = train['item_condition_id'].astype('category')
train['category_name'].fillna('No data/No data/No data', inplace=True)
test2['category_name'].fillna('No data/No data/No data', inplace=True)
train['brand_name'].fillna('None', inplace=True)
test2['brand_name'].fillna('None', inplace=True)
train['item_description'].fillna('No description yet', inplace=True)
train.isnull().sum()
# Randomly generate a few products to inspect
train.sample(5)
train['price'].describe().apply(lambda x: format(x, 'f'))
ref = train.groupby(['category_name', 'brand_name']).median()['price'].reset_index()
ref2 = train.groupby(['category_name']).median()['price'].reset_index()
test2 = pd.merge(test2, ref, how='left', on=['category_name','brand_name'])
test2.isnull().sum()
submit_a = test2.loc[~test2.price.isnull(),['test_id','price']]
# submit_a.head()
test_b = test2.loc[test2.price.isnull(),:].drop('price',axis=1)
test_b = pd.merge(test_b, ref2, how='left', on=['category_name'])
test_b.price.fillna(train.price.median(), inplace=True)
test_b.price.describe()
submit_b = test_b.loc[:,['test_id','price']]
submit_q = pd.concat([submit_a, submit_b])
submit_q.shape
submit_q.to_csv('submit_q_late.csv', index=False)
# Merge back the median prices into training set to form an input variable
train = pd.merge(train, ref, how='left', on=['category_name','brand_name'])
train.rename(columns = {'price_x':'price', 'price_y':'med_price'}, inplace = True)
train.head()
# Create training and validation dataframes
X = pd.concat([train.loc[:,['item_condition_id','shipping']], np.log(train.med_price+1)], axis=1)
y = np.log(train.price+1)
'''Temporarily disabled
Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=167)
model = RandomForestRegressor(random_state=100)
model.fit(Xtr,ytr)
yv_pred = model.predict(Xv)
'''
'''plt.figure(figsize=(10,7))
plt.scatter(yv_pred, yv, s=10)
plt.xlabel('Log Predicted Price')
plt.ylabel('Log Actual Price')
plt.title('Actual vs Predicted Price in Validation Set')
plt.show()
'''
def rmsle(y_true,y_pred):
   assert len(y_true) == len(y_pred)
   return np.square(y_pred - y_true).mean() ** 0.5
# print ("RMSL error of the model is {:.4f}".format(rmsle(yv, yv_pred)))
# print ("RMSL error of the simplest median is {:.4f}".format(rmsle(yv, Xv.med_price)))
# Tfidf Vectorizer on names
'''
tfidf_obj = TfidfVectorizer(ngram_range = (1,1))
tfidf_train = tfidf_obj.fit_transform(train['name'].values.tolist())
print (tfidf_train.shape)
'''
# TruncatedSVD to reduce the name data into 10 dimensions
'''
n_comp = 10
svd_obj = TruncatedSVD(n_components=n_comp, algorithm = 'arpack')
train_svd = pd.DataFrame(svd_obj.fit_transform(tfidf_train))    
train_svd.columns = ['svd_name_'+str(i) for i in range(n_comp)]

train = pd.concat([train, train_svd], axis=1)
train.head()
'''
#X = pd.concat([X, train.iloc[:,-10:]], axis=1)
# Train a model
'''Temporarily disabled
Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=369)
model2 = RandomForestRegressor()
model2.fit(Xtr, ytr)
yv_pred2 = model2.predict(Xv)

print ("RMSL error of the model is {:.4f}".format(rmsle(yv, yv_pred2)))
'''