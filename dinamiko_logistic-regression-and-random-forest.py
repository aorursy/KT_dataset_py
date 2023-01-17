import pandas as pd
df = pd.read_csv('../input/fuga_train.csv')
df.head()
# convert state strings to numbers
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
df['State'] = encoder.fit_transform(df['State'])

# convert yes/no to 1/0
df['Int\'l Plan'].replace(('yes', 'no'), (1, 0), inplace=True)
df['VMail Plan'].replace(('yes', 'no'), (1, 0), inplace=True)

# remove phone column
df = df.drop(columns=['Phone'])

# grab index and remove it from dataset
index = df['index']
df = df.drop(columns=['index'])

df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['fuga']), df['fuga'])
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))