# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff
df=pd.read_csv("../input/amazon_alexa.tsv",sep='\t')
df.isnull().any().any()
df.head()
df.tail()
df.columns
df.info()
df.describe()
df['verified_reviews']
df
df['length'] = df['verified_reviews'].apply(len)

df.groupby('length').describe().sample(10)
df
pos=df[df['feedback']==1]
neg=df[df['feedback']==0]
pos
neg
ratings = df['rating'].value_counts()

label_rating = ratings.index
size_rating = ratings.values

colors = ['pink', 'lightblue', 'aqua', 'gold', 'crimson']

rating_piechart = go.Pie(labels = label_rating,
                         values = size_rating,
                         marker = dict(colors = colors),
                         name = 'Alexa', hole = 0.3)

da = [rating_piechart]

layout = go.Layout(
           title = 'Distribution of Ratings for Alexa')

fig = go.Figure(data = da,
                 layout = layout)

py.iplot(fig)
feedbacks = df['feedback'].value_counts()

label_feedback = feedbacks.index
size_feedback = feedbacks.values

colors = ['yellow', 'lightgreen']

feedback_piechart = go.Pie(labels = label_feedback,
                         values = size_feedback,
                         marker = dict(colors = colors),
                         name = 'Alexa', hole = 0.3)

df2 = [feedback_piechart]

layout = go.Layout(
           title = 'Distribution of Feedbacks for Alexa')

fig = go.Figure(data = df2,
                 layout = layout)

py.iplot(fig)
sns.countplot(x='feedback',data=df)
sns.countplot(x='rating',data=df)
plt.figure(figsize=[25,10])
sns.barplot(x='variation',y='rating',data=df)


plt.figure(figsize=[20,10])
sns.countplot(y='variation',data=df)
df['length'].value_counts().plot.hist(color = 'skyblue', figsize = (15, 5), bins = 50)
plt.title('Distribution of Length in Reviews')
plt.xlabel('lengths')
plt.ylabel('count')
plt.show()
plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('fivethirtyeight')

sns.boxenplot(df['variation'], df['rating'], palette = 'spring')
plt.title("Variation vs Ratings")
plt.xticks(rotation = 90)
plt.show()
plt.rcParams['figure.figsize'] = (15, 9)
plt.style.use('fivethirtyeight')

sns.swarmplot(df['variation'], df['length'], palette = 'deep')
plt.title("Variation vs Length of Ratings")
plt.xticks(rotation = 90)
plt.show()
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 7)
plt.style.use('fivethirtyeight')

sns.boxplot(df['rating'], df['length'], palette = 'Blues')
plt.title("Length vs Ratings")
plt.show()
from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer(stop_words = 'english')
words = cv.fit_transform(df.verified_reviews)
sum_words = words.sum(axis=0)


words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

plt.style.use('fivethirtyeight')
color = plt.cm.ocean(np.linspace(0, 1, 20))
frequency.head(20).plot(x='word', y='freq', kind='bar', figsize=(15, 6), color = color)
plt.title("Most Frequently Occuring Words - Top 20")
plt.show()
trace = go.Scatter3d(
    x = df['length'],
    y = df['rating'],
    z = df['variation'],
    name = 'Amazon Alexa',
    mode='markers',
    marker=dict(
        size=10,
        color = df['rating'],
        colorscale = 'Viridis',
    )
)
df3 = [trace]

layout = go.Layout(
    title = 'Length vs Variation vs Ratings',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data = df3, layout = 
                layout)
iplot(fig)
df
variation_dummies = pd.get_dummies(df['variation'], drop_first = True)
variation_dummies
df.drop(['variation'],axis=1,inplace=True)
df = df.drop(['date', 'rating'],axis=1)
df
df= pd.concat([df, variation_dummies], axis=1)
df
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
alexa_countvectorizer = vectorizer.fit_transform(df['verified_reviews'])

alexa_countvectorizer.shape
print(vectorizer.get_feature_names())
print(alexa_countvectorizer.toarray()) 
df.drop(['verified_reviews'], axis=1, inplace=True)
reviews = pd.DataFrame(alexa_countvectorizer.toarray())
df= pd.concat([df, reviews], axis=1)
df
X = df.drop(['feedback'],axis=1)
X
y=df['feedback']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=5)
X_train
X_test.shape
y_train.shape
y_test.shape
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
randomforest_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')
randomforest_classifier.fit(X_train, y_train)

y_predict = randomforest_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot=True)
print(classification_report(y_test, y_predict))
print("Training Accuracy :",randomforest_classifier.score(X_train, y_train))
print("Testing Accuracy :",randomforest_classifier.score(X_test, y_test))
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = randomforest_classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy :", accuracies.mean())
print("Standard Variance :", accuracies.std())
X_test






