import numpy as np

import pandas as pd



#visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import operator

from sklearn.model_selection import KFold
train = pd.read_csv('../input/train.tsv', sep='\t')

test = pd.read_csv('../input/test.tsv', sep='\t')
train.describe(include="all")
#get a list of the features within the dataset

print(train.columns)
#see a sample of the dataset to get an idea of the variables

train.sample(5)
#check for any other unusable values

print(pd.isnull(train).sum())
from wordcloud import WordCloud
wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(train['name']))

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
#draw a bar plot of item_condition_id by price

sns.barplot(x="item_condition_id", y="price", data=train)
#draw a bar plot of shipping by price

sns.barplot(x="shipping", y="price", data=train)
#Add new feature with length of the name

train['name_len_words'] = train['name'].apply(lambda x: len(x.split()))

test['name_len_words'] = test['name'].apply(lambda x : len(x.split()))



train['name_len_chars'] = train['name'].apply(lambda x: len(x))

test['name_len_chars'] = test['name'].apply(lambda x : len(x))
train = train.drop('name', axis = 1)

test = test.drop('name', axis = 1)
#print category name and its frequency

category_dict = train['category_name'].value_counts().to_dict()
sorted_category_dict = sorted(category_dict.items(), key=operator.itemgetter(1))
sorted_category_dict[::-1]

sorted_category_list = [v[0] for v in sorted_category_dict]
train['category_name'] = train['category_name'].fillna(pd.Series(np.random.choice(sorted_category_list[0:100], size=len(train.index))))

test['category_name'] = test['category_name'].fillna(pd.Series(np.random.choice(sorted_category_list[0:100], size=len(test.index))))
#check for any other unusable values

print(pd.isnull(train).sum())
#Add column with number of elements in category_name

train['category_name_elements'] = train['category_name'].apply(lambda x: len(x))

test['category_name_elements'] = test['category_name'].apply(lambda x: len(x))
train.sample(5)
train = train.drop('category_name', axis = 1)

test = test.drop('category_name', axis = 1)
# Drop brand name column as more than half of values are 'NA'

train = train.drop('brand_name', axis=1)

test = test.drop('brand_name', axis=1)
#Only 4 elements are None, so fill them with some random string

train['item_description'] = train['item_description'].fillna('Description Not available')

test['item_description'] = test['item_description'].fillna('Description Not available')
#Length of description and number of words as new parameters

train['item_desc_words'] = train['item_description'].apply(lambda x: len(x.split()))

test['item_desc_words'] = test['item_description'].apply(lambda x: len(x.split()))



train['item_desc_chars'] = train['item_description'].apply(lambda x: len(x))

test['item_desc_chars'] = test['item_description'].apply(lambda x: len(x))
train.sample(5)
train = train.drop('item_description', axis=1)

test = test.drop('item_description', axis=1)
train.sample(5)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop(['train_id', 'price'], axis=1), train['price'], test_size=0.10, random_state=42)
X_train.shape
def calculate_RMSE(y, y_pred):

    k = len(y)

    s = 0

    for i in range(k):

        s += np.square((np.log10(y_pred[i] + 1) - np.log10(y[i] + 1)))

    s = s/k

    return np.sqrt(s)
from sklearn import linear_model
clf = linear_model.BayesianRidge()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print (calculate_RMSE(y_test.tolist(), preds))
target = test['test_id']

predictions = clf.predict(test.drop(['test_id'], axis=1))

predictions = predictions/5.0
submission = pd.DataFrame()

submission['test_id'] = target

submission['price'] = predictions
submission.to_csv('submission.csv', index = False)
import xgboost as xgb
xgb_params= {  

            'eta': 0.7,

            'max_depth': 12,

            'objective':'reg:linear',

            'eval_metric':'rmse',

            'silent': 1

}
kf = KFold(n_splits = 5, random_state = 1, shuffle = True)
train_matrix = xgb.DMatrix(X_train, y_train)
validation_matrix = xgb.DMatrix(X_test, y_test)
evallist  = [(validation_matrix,'validation')]
model = xgb.train(xgb_params, train_matrix, 10, evallist, verbose_eval=2)
preds = model.predict(xgb.DMatrix(test.drop(['test_id'], axis=1)), ntree_limit=model.best_ntree_limit);
print (preds)
submission = pd.DataFrame()

submission['test_id'] = test['test_id']

submission['price'] = preds
submission.to_csv('submission.csv', index = False)
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(train.drop(['train_id', 'price'], axis=1), train['price'])
preds = regr.predict(test.drop(['test_id'], axis=1))
print (calculate_RMSE(y_test.tolist(), preds))
submission = pd.DataFrame()

submission['test_id'] = test['test_id']

submission['price'] = preds
submission.to_csv('submission.csv', index = False)