import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import r2_score
df = pd.read_csv('../input/asia_conflicts.csv', parse_dates=['event_date'])

# focus on a single country
df = df[df['iso3'] == 'PAK']
# get X and y such that X is the notes and y is the next days fatality count
# e.g. if provacative words are used today, how many will die tomorrow?
# remove the [:-1] and [1:] to use the days notes to predict the days fatality count
X = df.groupby(['event_date'])['notes'].apply(lambda x: '| '.join(x))[:-1]
y = df.groupby(['event_date'])['fatalities'].sum()[1:]
# convert words to numbers
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X.values)
# split up the data into training and testing datasets for validating accuracy
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.25, random_state=42)
clf = svm.SVR(kernel='linear')
clf.fit(X_train, y_train)
y_predictions = clf.predict(X_test)
result = r2_score(y_test, y_predictions)
# due to the very very low R2 value I'm not taking this any further
print(result)
