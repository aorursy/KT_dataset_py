#The baseline modules

import numpy as np

import pandas as pd



#For text cleaning

import spacy



#For plotting

import missingno as msno

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px



#Model packages

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression



#Pipeline, Vectorizers and accuracy metrics

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv('../input/reddit-india-flair-detection/datafinal.csv', index_col='Unnamed: 0')

df.head()
df.isnull().sum().any()
df.isnull().sum()
msno.matrix(df)

plt.show()
df.columns
df.drop(['score','url','comms_num','author','timestamp'], axis=1, inplace=True)

df.head()
df['title'][0]
df['body'][0]
df['comments'][0]
df['combined_features'][0]
df.drop(['combined_features'], axis=1, inplace=True)

df.head()
df.info()
df.describe()
df['flair'].unique()
df.groupby('flair')['title'].describe()
fla_df = pd.DataFrame({"Flair":df['flair'].unique(), "Number":df.groupby('flair')['title'].describe()['freq']})



fig = px.bar(fla_df, x='Flair', y='Number', title='Flair Counts by Title in r/india')

fig.show()
fla_df_1 = pd.DataFrame({"Flair":df['flair'].unique(), "Number":df.groupby('flair')['body'].describe()['freq']})



fig = px.bar(fla_df_1, x='Flair', y='Number', title='Flair Counts by body in r/india')

fig.show()
fla_df_2 = pd.DataFrame({"Flair":df['flair'].unique(), "Number":df.groupby('flair')['comments'].describe()['freq']})



fig = px.bar(fla_df_2, x='Flair', y='Number', title='Flair Counts by comments in r/india')

fig.show()
df[df['flair'] == np.nan].describe()
df.dropna(subset=['flair'], inplace=True)
df.dtypes
df['text'] = df['title'].astype(str) + df['body'].astype(str) + df['comments'].astype(str)

df.drop(['title', 'body', 'comments'], axis=1, inplace=True)

df.head()
nlp = spacy.load('en')



def normalize(msg):

    

    doc = nlp(msg)

    res=[]

    

    for token in doc:

        if(token.is_stop or token.is_punct or not(token.is_oov)): #Removing stopwords punctuations and words out of vocab

            pass

        else:

            res.append(token.lemma_.lower())

    

    return " ".join(res)
df['text'] = df['text'].apply(normalize)

df.head()
c = TfidfVectorizer() # Convert our strings to numerical values

mat=pd.DataFrame(c.fit_transform(df["text"]).toarray(),columns=c.get_feature_names(),index=None)

mat
X = mat

y = df["flair"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
pipeline = Pipeline([

    ('classifier',DecisionTreeClassifier()),

    ])



pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy: {:.2f} %".format(accuracy_score(y_test, y_pred)*100))
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['flair'], test_size = 0.2, random_state = 0)
ids = [df.iloc[int(i)]['id'] for i in X_test.index]

final_df = pd.DataFrame({"ID":ids, "Text":X_test, "Flair":y_pred}).reset_index()



final_df.head()
final_df.to_csv('./test.csv')
'''classifiers = {

    'mnb': MultinomialNB(),

    'gnb': GaussianNB(),

    'svm1': SVC(kernel='linear'),

    'svm2': SVC(kernel='rbf'),

    'svm3': SVC(kernel='sigmoid'),

    'mlp1': MLPClassifier(),

    'mlp2': MLPClassifier(hidden_layer_sizes=[100,100]),

    'ada': AdaBoostClassifier(),

    'dtc': DecisionTreeClassifier(),

    'rfc': RandomForestClassifier(),

    'gbc': GradientBoostingClassifier(),

    'lr': LogisticRegression()

}'''
'''acc_scores = dict()

for classifier in classifiers:

    pipeline = Pipeline([

    ('classifier',classifiers[classifier]),

    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc_scores[classifier] = accuracy_score(y_test, y_pred)

    print(classifier, acc_scores[classifier])'''