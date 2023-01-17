import pandas as pd

import nltk

raw_df = pd.read_csv("../input/bbc-text-classification/bbc-text.csv")
raw_df.head()
#checking the distribution of categories

raw_df.groupby('category').category.count()
#converting classes into number

raw_df['class_id']=raw_df['category'].factorize()[0]
#Cleaning text (remove stops words

from nltk.corpus import stopwords

stop = stopwords.words('english')

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

raw_df['text_without_stopwords'] = raw_df['text'].apply(lambda x:' '.join([word for word in x.split() if word not in (stop)]))
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

matrix = vectorizer.fit_transform(raw_df['text_without_stopwords']).toarray()

tfidf_df = pd.DataFrame(matrix, columns=vectorizer.get_feature_names())
tfidf_df.head()
#feature addition

tfidf_df['text']=raw_df['text_without_stopwords'].values

tfidf_df['totalwords'] = tfidf_df['text'].str.split().str.len()

columns_drop = ['text']

X_features = tfidf_df.drop(columns_drop,axis=1)
#Splitting the data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_features, raw_df['class_id'], test_size=0.2)
#Cross validation for Randomforestclassifier default

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, cross_val_score

rf = RandomForestClassifier(n_jobs=-1)

k_fold = KFold(n_splits=5)

cross_val_score(rf, X_features, raw_df['class_id'], cv=k_fold, scoring='accuracy', n_jobs=-1)
#Using Randomforest

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1)

rf_model = rf.fit(X_train, y_train)
#Print feature importance

sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True)[0:50]
y_pred = rf_model.predict(X_test)



from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(y_test, y_pred,)

print('Precision: {} / Recall: {} / Accuracy: {}'.format(precision,

                                                        recall,

                                                        round((y_pred==y_test).sum() / len(y_pred),3)))
#GridsearchCV to find best parameters

from sklearn.metrics import precision_recall_fscore_support as score

def train_RF(n_est, depth):

    rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, n_jobs=-1)

    rf_model = rf.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    precision, recall, fscore, support = score(y_test, y_pred)

    print('Est: {} / Depth: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(

        n_est, depth, precision, recall,

        round((y_pred==y_test).sum() / len(y_pred), 3)))

    

for n_est in [10, 50, 100]:

    for depth in [10, 20, 30, None]:

        train_RF(n_est, depth) 