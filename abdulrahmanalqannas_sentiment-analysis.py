import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

#sklearn

from sklearn.model_selection import KFold, train_test_split, cross_val_score

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn import metrics, svm

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier







# To ignore unwanted warnings

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/ssd-reviews/ssd_reviews.csv', index_col=0)
df.head()
df.info()
fig, ax = plt.subplots(figsize = (14, 10))

sns.heatmap(df.isnull(), yticklabels=False, ax = ax, cbar=False, cmap='viridis')

ax.set_title('Customers Reviews')

plt.show()
df['date'] = pd.to_datetime(df['date'])
df['rating_stars'].value_counts()
df_cons = df[['cons']].dropna()

df_cons['positive'] = 0
# a lot of values droped around 700 row in cons

df_cons.drop(df_cons[df_cons['cons'].isin(['none', 'none so far', 'non'])].index, inplace=True)
df_cons.rename(columns={'cons':'pros_and_cons'}, inplace=True)
#To make our data balanced betwwen'pros' and 'cons', Here in 'pros' we took only 1562 row because a lot of rows droped in 'cons'

df_pros = df[['pros']][:1562].dropna()
df_pros.rename(columns={'pros':'pros_and_cons'}, inplace=True)
df_pros['positive'] = 1
merged_df = pd.merge(left=df_pros, right=df_cons, left_on=['pros_and_cons', 'positive'],

                     right_on=['pros_and_cons', 'positive'], how='outer')
merged_df
merged_df['positive'].value_counts()
X = merged_df.drop('positive', axis=1)

y = merged_df['positive']
bow_f = CountVectorizer(stop_words='english').fit(X['pros_and_cons'])
print("After eliminating stop words: ", len(bow_f.get_feature_names()))
bow_transform = bow_f.transform(X['pros_and_cons'])
count_vect_df = pd.DataFrame(bow_transform.todense(), columns=bow_f.get_feature_names())

np.sum(count_vect_df).sort_values(ascending=False)[0:20]
X_train, X_test, y_train, y_test = train_test_split(bow_transform, y, test_size=0.3, random_state=42)
#8

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

print('train score' , logreg.score(X_train, y_train))

print('test score' , logreg.score(X_test, y_test))

y_pred = logreg.predict(X_test)
confusion_matrix(y_test, y_pred)
tfidf_vectoriser = TfidfVectorizer(stop_words='english')

tfidf_f = tfidf_vectoriser.fit(X['pros_and_cons'])

tfidf_transform = tfidf_f.transform(X['pros_and_cons'])
tf_X_train, tf_X_test, y_train, y_test = train_test_split(tfidf_transform, y, test_size=0.3)
tf_logreg = LogisticRegression(C=3.2)

tf_logreg.fit(tf_X_train, y_train)

print('train score' , tf_logreg.score(tf_X_train, y_train))

print('test score' , tf_logreg.score(tf_X_test, y_test))



y_pred = tf_logreg.predict(tf_X_test)
cv=KFold(n_splits=5, shuffle=True, random_state=1)

cross_val_score(tf_logreg, tfidf_transform, y, cv=cv).mean()
confusion_matrix(y_test, y_pred)
knn_classifier = KNeighborsClassifier()  

knn_classifier.fit(tf_X_train, y_train)

print(knn_classifier.score(tf_X_train, y_train))

print (knn_classifier.score(tf_X_test, y_test))

y_pred = knn_classifier.predict(tf_X_test)
confusion_matrix(y_test, y_pred)
tree= DecisionTreeClassifier()

tree.fit(tf_X_train, y_train)

print('test score' , tree.score(tf_X_train, y_train))

print('test score' , tree.score(tf_X_test, y_test))

y_pred = tree.predict(tf_X_test)
confusion_matrix(y_test, y_pred)
svm_linear = svm.SVC(kernel='linear')

svm_linear.fit(tf_X_train, y_train)

print('Train : ', svm_linear.score(tf_X_train, y_train))

print('Test: ', svm_linear.score(tf_X_test, y_test))

y_pred = svm_linear.predict(tf_X_test)
confusion_matrix(y_test, y_pred)
cv=KFold(n_splits=5, shuffle=True, random_state=1)

cross_val_score(svm_linear, tfidf_transform, y, cv=cv).mean()
svm_rbf = svm.SVC(kernel='rbf',C=1, probability=True)

svm_rbf.fit(tf_X_train, y_train)

print('Train : ', svm_rbf.score(tf_X_train, y_train))

print('Test: ', svm_rbf.score(tf_X_test, y_test))

y_pred = svm_rbf.predict(tf_X_test)
cv=KFold(n_splits=5, shuffle=True, random_state=1)

cross_val_score(svm_rbf, tfidf_transform, y, cv=cv).mean()
confusion_matrix(y_test, y_pred)
svm_poly = svm.SVC(kernel='poly', C=.7)

svm_poly.fit(tf_X_train, y_train)

print('Train : ', svm_poly.score(tf_X_train, y_train))

print('Test: ', svm_poly.score(tf_X_test, y_test))

y_pred = svm_poly.predict(tf_X_test)
cv=KFold(n_splits=5, shuffle=True, random_state=1)

cross_val_score(svm_poly, tfidf_transform, y, cv=cv).mean()
confusion_matrix(y_test, y_pred)
randomF = RandomForestClassifier()

randomF.fit(tf_X_train, y_train)

print('Train score :',randomF.score(tf_X_train, y_train))

print('Ttest score :',randomF.score(tf_X_test, y_test))

y_pred = randomF.predict(tf_X_test)
cv=KFold(n_splits=5, shuffle=True, random_state=1)

cross_val_score(randomF, tfidf_transform, y, cv=cv).mean()
confusion_matrix(y_test, y_pred)
gnb = GaussianNB(var_smoothing=0.11) 

gnb.fit(tf_X_train.toarray(), y_train) 

print('Train score :',gnb.score(tf_X_train.toarray(), y_train))

print('Ttest score :',gnb.score(tf_X_test.toarray(), y_test))

y_pred = gnb.predict(tf_X_test.toarray())
confusion_matrix(y_test, y_pred)
cv=KFold(n_splits=5, shuffle=True, random_state=1)

cross_val_score(gnb, tfidf_transform.toarray(), y, cv=cv).mean()
mnb = MultinomialNB(alpha=0.22) 

mnb.fit(tf_X_train.toarray(), y_train) 

print('Train score :',mnb.score(tf_X_train.toarray(), y_train))

print('Ttest score :',mnb.score(tf_X_test.toarray(), y_test))

y_pred = mnb.predict(tf_X_test.toarray())
confusion_matrix(y_test, y_pred)
tfidf_comments = tfidf_f.transform(['expensive'])

svm_rbf.predict(tfidf_comments)

round(svm_rbf.predict_proba(tfidf_comments)[0][1], 5)