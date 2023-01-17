import pandas as pd
df=pd.read_csv("C:\\Users\\sathish\\Desktop\\Data Science Algorithms\\spam.csv",encoding = 'latin-1')
df.head()
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis='columns',inplace=True)
df.head()
df.groupby('Category').describe()
df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df.head()
X=df['Message']
y=df['spam']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=112,test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_transformed = v.fit_transform(X_train.values)
X_test_transformed = v.transform(X_test)
X_train_transformed.toarray()[:2]
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_transformed,y_train)
model.score(X_test_transformed,y_test)
emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_transformed = v.transform(emails)
model.predict(emails_transformed)
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)
clf.predict(X_test)[:4]
clf.score(X_test,y_test)
clf.predict(emails)

