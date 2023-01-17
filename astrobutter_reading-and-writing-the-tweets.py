#Import some libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Read in the labelled data
dftrain=pd.read_csv("../input/innovation-challenge-2020/data_training.csv",header=0)

#And the target tweet ids for the solution (and the dummy classifcations)
dfsolution=pd.read_csv("../input/innovation-challenge-2020/data_example_random.csv",header=0)

#Hydrate following the steps in the Data page.
#Read in the hydrated data
dfhydrate=pd.read_csv("../input/hydrateddata/hydrated_training.csv",header=0)

#Read in the hydrated data for the target tweet ids
dftarget=pd.read_csv("../input/hydrateddata/hydrated_solution.csv",header=0)
#Merge the two datasets (tweet ids from data_training and the tweets from hydrated_training). 
#Note: the number of tweetids availble to hydrate may have changed since the labelled dataset was made.
df=pd.merge(left=dfhydrate,right=dftrain, left_on='id', right_on='tweetid')
df.head()
#What are all the column names?
df.columns
#Import some sci-kit learn libraries to make a simple classifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
#Build our processing pipeline
text_clf = Pipeline([
    #Turn the sentences into something we can classify
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    #And choose the classifier
    ('clf', MultinomialNB()),
    ])
predicted = text_clf.predict(df.text)

print("F1 Score:", f1_score(df.tweet_label_int.values,predicted))
#Nevertheless, lets create the final output data to upload up to kaggle for submission
#Test our classifier on the hydrated target tweets
predicted_targets = text_clf.predict(dftarget.text)
#Set the solution to our predicted targets
dfsolution.tweet_label_int=predicted_targets
dfsolution
#And save it out to upload to competition!
dfsolution.to_csv("my_kaggle_solution.csv",index=False)
