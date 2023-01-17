import pandas as pd

import numpy as np

import scipy
from sklearn.linear_model import Ridge,LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelBinarizer
#loading data

train = pd.read_csv('../input/train.tsv', sep = '\t')

test = pd.read_csv('../input/test.tsv', sep = '\t')

NUM_BRANDS = 2500

NAME_MIN_DF = 10

MAX_FEAT_DESCP = 50000
#train.head(3)

train.brand_name.value_counts()
print(test.head(3))

print (test.shape)
#concatenating test and train into a single dataset

dt = pd.concat([train,test],0)

dt.head(3)

dt.brand_name.value_counts()
#this gives the number of rows in the training set

nrow_train = train.shape[0]

#print (nrow_train)

#converting the target variable "price" into it's log values for better fit

target = np.log1p(train.price)

target
#filling names and missing values in category_name

dt.category_name = dt.category_name.fillna('Other').astype('category')

dt.brand_name = dt.brand_name.fillna('unknown')
dt.isnull().sum()
pop_brands = dt.brand_name.value_counts().index[:NUM_BRANDS]
pop_brands
dt.loc[~dt.brand_name.isin(pop_brands), 'brand_name'] = 'Other'
dt.head(3)
dt.item_description = dt.item_description.fillna('None')

dt.item_condition_id = dt.item_condition_id.astype('category')

dt.brand_name = dt.brand_name.astype('category')

#encoding name

count = CountVectorizer(min_df = NAME_MIN_DF)

x_name = count.fit_transform(dt.name)
x_name
from scipy.sparse import find
find(x_name)
#encoding category variables

#spliting data info in category_name via split('/)

unique_categories = pd.Series('/'.join(dt.category_name.unique().astype('str')).split('/')).unique()
unique_categories.take(10)
count_category = CountVectorizer()
x_category = count_category.fit_transform(dt.category_name)
#item_description

count_desc = TfidfVectorizer(max_features = MAX_FEAT_DESCP,

                            ngram_range = (1,3),

                            stop_words = 'english')

x_descp = count_desc.fit_transform(dt.item_description)
#brand_name encoder

vect_brand = LabelBinarizer(sparse_output = True)

x_brand = vect_brand.fit_transform(dt.brand_name)
#dummy encoders

x_dummies = scipy.sparse.csr_matrix(pd.get_dummies(dt[['item_condition_id','shipping']],sparse = True).values)
X = scipy.sparse.hstack((x_dummies,

                        x_descp,

                        x_brand,

                        x_category,

                        x_name)).tocsr()
print (x_dummies.shape, x_category.shape, x_name.shape,x_descp.shape,x_brand.shape)
x_train = X[:nrow_train]
clf = Ridge(solver = 'lsqr',fit_intercept = False)
y_train = target
#fitting classifier

clf.fit(x_train,y_train)
x_test = X[nrow_train:]
preds = clf.predict(x_test)
test["price"] = np.expm1(preds)

test[['test_id','price']].to_csv('ridge_clf.csv',index = False)