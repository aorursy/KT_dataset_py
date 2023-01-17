import pandas as pd

df_submission = pd.read_csv("../input/semaine-ia-reddit-challenge/test.csv")

df = pd.read_csv("../input/semaine-ia-reddit-challenge/train.csv")

df.head()
print(sum(df["subreddit"] == "gameofthrones"))

print(sum(df["subreddit"] == "gaming"))

print(sum(df["subreddit"] == "funny"))

print(sum(df["subreddit"] == "news"))

print(sum(df["subreddit"] == "politics"))

print(sum(df["subreddit"] == "science"))
labels = pd.get_dummies(df["subreddit"],prefix='subreddit')

labels.head()
y_data = labels.values

comments = df["comment"].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(comments, y_data, test_size=0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

X_train
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score



knn_classifier = KNeighborsClassifier(n_neighbors=10)

knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)



f1_score(y_test, y_pred, average='macro')
comments_submission = df_submission["comment"].values

submission_tfidf = vectorizer.transform(comments_submission)



submission_tfidf
y_submission = knn_classifier.predict(submission_tfidf)

df_sub_preds = pd.DataFrame(y_submission)
def one_hot_to_subreddit(a):

    if a[0]:

        return "funny"

    elif a[1]:

        return "gameofthrones"

    elif a[2]:

        return "gaming"

    elif a[3]:

        return "news"

    elif a[4]:

        return "politics"

    elif a[5]:

        return "science"

    else:

        return "politics"
df_sub_preds["subreddit"] = df_sub_preds.apply(lambda a: one_hot_to_subreddit(a), axis=1)

submission = pd.DataFrame()

submission["id"] = df_submission["id"]

submission["subreddit"] = df_sub_preds["subreddit"]



submission
submission.to_csv("knn_submission.csv", index=False)