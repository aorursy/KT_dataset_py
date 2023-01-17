import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# import dataset

df = pd.read_csv('../input/Womens Clothing E-Commerce Reviews.csv')
df.head()
col = ['Review Text', 'Department Name']
df = df[col]

df.isnull().sum()
df = df.dropna()

df.isnull().sum()
df.columns = ['review', 'department']
df['department_id'] = df['department'].factorize()[0]

df.head()
encoded_data, mapping_index = df['department'].factorize()
print(encoded_data)
print(mapping_index)
fig = plt.figure(figsize=(12,6))
df.groupby('department').review.count().plot.bar()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    sublinear_tf=True, 
    min_df=5, 
    norm='l2',
    encoding='latin-1',
    ngram_range=(1,2),
    stop_words='english'
)

features = tfidf.fit_transform(df.review).toarray()
labels = df.department_id

features.shape
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(
    df['review'], 
    df['department_id'],
    random_state=0
)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)
sample1 = df.sample(1)
print(sample1.department)
print(df.review[sample1.index[0]])
pred = clf.predict(count_vect.transform([df.review[sample1.index[0]]]))
print(mapping_index[pred][0])
sample2 = df.sample(1)
print(sample2.department)
print(df.review[sample2.index[0]])
pred = clf.predict(count_vect.transform([df.review[sample2.index[0]]]))
print(mapping_index[pred][0])
pred = clf.predict(count_vect.transform([df.review[14422]]))

print(mapping_index[pred][0])
# find the best model

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score

models = [
    LogisticRegression(random_state=0),
    RandomForestClassifier(n_estimators=200,max_depth=3,random_state=0),
    LinearSVC(),
    MultinomialNB()
]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))

entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))

cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

import seaborn as sns

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
cv_df.groupby('model_name').accuracy.mean()
# df['department_id'] = df['department'].factorize()[0]
department_id_df = df[['department', 'department_id']].drop_duplicates().sort_values('department_id')
department_to_id = dict(department_id_df.values)
id_to_department = dict(department_id_df[['department_id', 'department']].values)
df.head()
from sklearn.svm import LinearSVC
import seaborn as sns

model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=department_id_df.department.values, yticklabels=department_id_df.department.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
from sklearn import metrics

print(metrics.classification_report(y_test, y_pred, target_names=df['department'].unique()))
