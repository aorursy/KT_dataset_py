import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('../input/ISIS Religious Texts v1.csv', delimiter=',', encoding='latin-1')
data.head(8)
pd.unique(data.Magazine)
data['Magazine'].value_counts()
data.Magazine.isna().sum()
data[data.Magazine.isna()]
data.isna().sum()
data.shape
data = data.drop(['Purpose'], axis=1)
data.shape
data.isna().sum()
data = data.dropna()
data.sample(8)
pd.unique(data.Source)
len(pd.unique(data.Source))
len(pd.unique(data.Type))
data['Type'].value_counts()
used_col = ['Quote','Magazine']

data = data[used_col]
data.sample(8)
data['Magazine_id'] = data['Magazine'].factorize()[0]
data.sample(8)
encoding_data, mapping_index = data['Magazine'].factorize()
print(encoding_data)
print(mapping_index)

for i in range(len(mapping_index)):
    print(i,mapping_index[i])
data.groupby(['Magazine'])['Quote'].count().plot.bar()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    norm='l2',
    encoding='latin-1',
    min_df=5,
    sublinear_tf=True
)

features = tfidf.fit_transform(data['Quote']).toarray()
labels = data['Magazine_id']

features.shape
# find the best model

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

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
cv_df.groupby(['model_name'])['accuracy'].mean().sort_values(ascending=False)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

X_train, X_test, y_train, y_test = train_test_split(
    data['Quote'], 
    data['Magazine_id'], 
    test_size=0.3, 
    random_state=0
)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tfidf_transfomer = TfidfTransformer()
X_train_tfidf = tfidf_transfomer.fit_transform(X_train_counts)

clf = LogisticRegression().fit(X_train_tfidf, y_train)
sample1 = data.sample(1)
print(sample1.Magazine)
print(data.Quote[sample1.index[0]])
pred = clf.predict(count_vect.transform([data.Quote[sample1.index[0]]]))
print(mapping_index[pred][0])
sample2 = data.sample(1)
print(sample2.Magazine)
print(data.Quote[sample2.index[0]])
pred = clf.predict(count_vect.transform([data.Quote[sample2.index[0]]]))
print(mapping_index[pred][0])
magazine_id_df = data[['Magazine', 'Magazine_id']].drop_duplicates().sort_values('Magazine_id')
magazine_to_id = dict(magazine_id_df.values)
id_to_magazine = dict(magazine_id_df[['Magazine_id', 'Magazine']].values)
data.head()
model = LogisticRegression()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    features, labels, data.index, test_size=0.33, random_state=0
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
import seaborn as sns

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5,5))

sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=magazine_id_df['Magazine'].values, 
            yticklabels=magazine_id_df['Magazine'].values)

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

from sklearn.metrics import accuracy_score

print('accuracy: %s' % (accuracy_score(y_test, y_pred)))