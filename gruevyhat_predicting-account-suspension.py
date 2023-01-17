import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('../input/tweets.csv', encoding="utf8")

junk = re.compile("al|RT|\n|&.*?;|http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
df["cleantweets"] = [junk.sub(" ", t) for t in df.tweets]

X = df.groupby('username', sort=False).cleantweets.apply(lambda x: ' '.join(x))

# The labels below indicate whether or not the username is currently (5/21/16) suspended
# on Twitter.
y = [1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1,  
     1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1,
     0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1,
     1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0,
     1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

tfv = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
svd = TruncatedSVD(n_components=300)
std = StandardScaler()
svm = SVC(kernel="rbf")
clf = Pipeline([('tfv', tfv), ('svd', svd), ('std', std), ('svm', svm)])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
