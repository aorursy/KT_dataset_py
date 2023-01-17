!pip install vaderSentiment
import numpy as np 
import pandas as pd 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import os
for dirname, _, filenames in os.walk('/kaggle/input/tweet-sentiment-extraction/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
data.shape
data.head()
data.tail()
data.info()
data.isnull().sum()
data.dropna(inplace=True)
data.info()
analyzer = SentimentIntensityAnalyzer()
def calculate_sentiment_scores(sentence):
    sntmnt = analyzer.polarity_scores(sentence)['compound']
    return(sntmnt)
start = time.time()

eng_snt_score =  []

for comment in data.text.to_list():
    snts_score = calculate_sentiment_scores(comment)
    eng_snt_score.append(snts_score)
    
end = time.time()

# total time taken
print(f"Runtime of the program is {(end - start)/60} minutes or {(end - start)} seconds")
data['sentiment_score'] = np.array(eng_snt_score)
data.head()
i = 0

vader_sentiment = [ ]

while(i<len(data)):
    if ((data.iloc[i]['sentiment_score'] >= 0.05)):
        vader_sentiment.append('positive')
        i = i+1
    elif ((data.iloc[i]['sentiment_score'] > -0.05) & (data.iloc[i]['sentiment_score'] < 0.05)):
        vader_sentiment.append('neutral')
        i = i+1
    elif ((data.iloc[i]['sentiment_score'] <= -0.05)):
        vader_sentiment.append('negative')
        i = i+1
data['vader_sentiment_labels'] = vader_sentiment
data.head(15)
data['actual_label'] = data['sentiment'].map({'positive': 1, 'neutral': 0, 'negative':-1})
data['predicted_label'] = data['vader_sentiment_labels'].map({'positive': 1, 'neutral': 0, 'negative':-1})

data.head()
from sklearn.metrics import accuracy_score
y_act = data['actual_label'].values
y_pred = data['predicted_label'].values
accuracy_score(y_act, y_pred)