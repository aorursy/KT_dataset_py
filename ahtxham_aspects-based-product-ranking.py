from textblob import TextBlob

import pandas as pd

import re
def clean_text(text): 

        ''' 

        Utility function to clean text by removing links,

        special characters using simple regex statements. 

        '''

        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split()) 



def get_text_sentiment(text):

    ''' 

    Utility function to classify sentiment of passed

    text using textblob's sentiment method 

    '''

    # create TextBlob object of passed text 

    analysis = TextBlob(clean_text(text)) 

    # set sentiment 

    if analysis.sentiment.polarity > 0: 

        return 'positive'

    elif analysis.sentiment.polarity == 0: 

        return 'neutral'

    else: 

        return 'negative'    

def feature_extraction(text): 

        blob = TextBlob(text)

        return blob.noun_phrases
dataset = pd.read_csv("../input/Amazon_Unlocked_Mobile.csv")

dataset.head()
#subset_sample['Reviews']

#sentiment = []

#for reviews in dataset['Reviews']:

 #   sentiment.append(get_text_sentiment(str(reviews)))

    

#dataset["Sentiment"] = sentiment

#dataset.head()
#subset_sample['Reviews']

#features = []

#for reviews in dataset['Reviews']:

 #   features.append(feature_extraction(reviews))

    

#dataset["features"] = features

#dataset.head()
#Selecting a subset from dataset

subset_sample=dataset.loc[dataset['Product Name'] == "Samsung Convoy U640 Phone for Verizon Wireless Network with No Contract (Gray) Rugged"]

subset_sample.head()
#subset_sample['Reviews']

#Extracting Features from subset

features = []

for reviews in subset_sample['Reviews']:

    features.append(feature_extraction(reviews))

    

subset_sample["features"] = features

subset_sample.head()
#subset_sample['Reviews']

sentiment = []

for reviews in subset_sample['Reviews']:

    sentiment.append(get_text_sentiment(reviews))

    

subset_sample["Sentiment"] = sentiment

subset_sample.head()
#Features With their counts

counts = {}

def feature_count(feature):

        if  feature in counts:

            counts[feature] += 1

        else:

            counts[feature] = 1

for feature_list in subset_sample['features']:

    for feature in feature_list:

        feature_count(feature)

        
print(counts)