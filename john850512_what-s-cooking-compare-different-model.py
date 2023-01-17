!ls ../input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display # use to print dataframe with beautiful style(in table form)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
default_path = '../input/'
train_df = pd.read_json(default_path+'train.json')
test_df = pd.read_json(default_path+'test.json')
# seeeeeeee
print(train_df.info())
print('if any missing value:', train_df.isnull().any().any())
print('-'*10)

print(test_df.info())
print('if any missing value:', test_df.isnull().any().any())
# find that `id` doesn't continuous
# ex: id = 5 disappear
# but `id` will not influence model predict..
train_df.sort_values(by=['id']).head(10)
ingredients_set = set()
for _, ingredients_list in train_df['ingredients'].iteritems():
    #print(_, ingredients_list)
    for i in ingredients_list:
        ingredients_set.add(i)
print('number of different ingredients:', len(ingredients_set))
# see ingredients count
ingredients_count = dict()
for _, ingredients_list in train_df['ingredients'].iteritems():
    for i in ingredients_list:
        ingredients_count[i] = ingredients_count.get(i, 0) + 1
# change dict to dataframe
ingredients_count_df = pd.DataFrame.from_dict(data=ingredients_count, orient='index')
ingredients_count_df = ingredients_count_df.reset_index()
ingredients_count_df.rename(columns={'index': 'ingredients', 0: 'count'}, inplace=True)
# sort
ingredients_count_df.sort_values(by='count', ascending=False, inplace=True)
ingredients_count_df.head(10)
g = sns.factorplot(x='count', y='ingredients', data=ingredients_count_df.head(50), 
                   kind='bar', palette='hls')
g.fig.set_size_inches(18, 12)

train_ingredients_lists = [' '.join(i) for i in train_df['ingredients']]
test_ingredients_list = [' '.join(i) for i in test_df['ingredients']]
# Bags of words used CountVectorizer
#vectorizer = CountVectorizer(max_features=1000)
#train_ingredients_vector = vectorizer.fit_transform(train_ingredients_lists).toarray()
#test_ingredients_vector = vectorizer.transform(test_ingredients_list).toarray()

# Bags of words used TfidfTransformer
vectorizer = TfidfVectorizer()
train_ingredients_vector = vectorizer.fit_transform(train_ingredients_lists).toarray()
test_ingredients_vector = vectorizer.transform(test_ingredients_list).toarray()
print(train_ingredients_vector.shape, test_ingredients_vector.shape)
train_ingredients_df = pd.DataFrame(train_ingredients_vector, columns=vectorizer.get_feature_names())
test_ingredients_df = pd.DataFrame(test_ingredients_vector, columns=vectorizer.get_feature_names())
display('train_ingredients_df:',train_ingredients_df.head(),
        'test_ingredients_df:',test_ingredients_df.head())
train_df_new = pd.concat([train_df, train_ingredients_df], axis=1)
train_df_new = train_df_new.drop(['ingredients'], axis=1)

test_df_new = pd.concat([test_df, test_ingredients_df], axis=1)
test_df_new = test_df_new.drop(['ingredients'], axis=1)

display('train_df_new:', train_df_new.head(),
        'test_df_new:', test_df_new.head())
X = train_df_new.drop(['id', 'cuisine'], axis=1)
LB = LabelEncoder()
y = LB.fit_transform(train_df['cuisine'])
# notice that can not use train_df_new['cuisine'] as parameter y of train_test_split()
# because in TfidfVectorizer() transform, there have ingredient which name include 'cuisine' 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)
X_test = test_df_new.drop(['id', 'cuisine'], axis=1)
print(X_train.shape, y_train.shape)
print(X_test.shape)
#clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=4)
#clf.fit(X_train, y_train)
#clf.oob_score_
#clf.score(X_val, y_val)
#lgbm = LGBMClassifier(n_estimators=500).fit(X_train, y_train)
#lgbm.score(X_val, y_val)
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(C=15).fit(X_train, y_train)
clf1.score(X_val, y_val)
# different with SVC, it run more fasterrrrrrrrrrrrrrrrrr
from sklearn.svm import LinearSVC
clf3 = LinearSVC(C=0.75).fit(X_train, y_train)
clf3.score(X_val, y_val)
prediction = clf3.predict(X_test)
prediction = LB.inverse_transform(prediction)
submission = pd.DataFrame({'id':test_df['id'], 'cuisine':prediction})
submission = submission[['id', 'cuisine']]
submission.to_csv('submission.csv', index=False)
submission.head()
