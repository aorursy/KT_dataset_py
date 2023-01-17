# Data Cleaning Libraries
import numpy as np
import pandas as pd
# Data Visulation Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()
%matplotlib inline
# Data Prediction Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
# Data set (Amazon Musical Instruents Review)
AMIR = pd.read_csv('/kaggle/input/amazon-music-reviews/Musical_instruments_reviews.csv')
# AMIR dataset info.
AMIR.info()     # Eagle Eye View
# Dropping null rows
AMIR = AMIR.dropna()
AMIR.info()
# AMIR Dataset
AMIR.head()
# Length of words in each message of review text column
AMIR['Length of Words'] = AMIR['reviewText'].apply(lambda x : len(x.split()))
AMIR.rename(columns={'overall':'rating'},inplace=True)  # renaming 'overall' column with 'rating' 
AMIR.head(4)
# Total Number of Users who rated the product as per rating category
AMIR.groupby(by='rating').helpful.count()
# Overall Rating with respect to Length of words in reviewtext messages
g = sns.FacetGrid(AMIR,col='rating',sharex=True)
g.map(sns.kdeplot,'Length of Words',color='red')
# Total number of people that rated the products as per rating category
go.Figure(data=[go.Pie(values=AMIR.groupby(by='rating').helpful.count(),labels=[1,2,3,4,5],
                       title='Volume received by each rating category.')])
# Predicting whether the reviewText message is positive or negative
# Considering Rating '1,2,3' as 'Negative Review' 
# Considering Rating '4,5' as 'Positive Review'

review = {1:'Negative',2:'Negative',3:'Negative',4:'Positive',5:'Positive'}
AMIR['review'] = AMIR['rating'].map(review)
AMIR[['reviewText','rating','review']].head()
# Selecting Features & Labels
X = AMIR['reviewText']        # features
y = AMIR['review']            # labels
# Splitting data into Training Data & Test Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Pipeline 
pipeline = Pipeline([
    ('Count Vectorizer',CountVectorizer()),
    ('Model',MultinomialNB())
])
# Training Data
pipeline.fit(X_train,y_train)
# Model Prediction
y_pred = pipeline.predict(X_test)
# Model Evaluation
print(confusion_matrix(y_test,y_pred))
print('\n')
print(classification_report(y_test,y_pred))