import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
dataset = '../input/amazon_alexa.tsv'
reviews_df = pd.read_csv(dataset, sep='\t', index_col=0, header=0)
vader_score = []
vader_class = []

for review in reviews_df.verified_reviews:
    ss = sid.polarity_scores(review)
    compound_score = ss.get('compound')
    vader_score.append(compound_score)
    if (compound_score >= 0):
        vader_class.append(1)
    else:
        vader_class.append(0)
        
reviews_df['vader_score'] = vader_score
reviews_df['vader_class'] = vader_class
reviews_df.loc[~(reviews_df['feedback'] == reviews_df['vader_class'])]
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def remove_stopwords(text):
    #tokenization
    tokens = word_tokenize(text) 
    
    #stopwords removal
    temp = [word for word in tokens if word not in stopwords.words('english')]
    
    #detokenization
    return "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in temp]).strip()
vader_score = []
vader_class = []

for review in reviews_df.verified_reviews:
    ss = sid.polarity_scores(remove_stopwords(review))
    compound_score = ss.get('compound')
    vader_score.append(compound_score)
    if (compound_score >= 0):
        vader_class.append(1)
    else:
        vader_class.append(0)
        
reviews_df['vader_score'] = vader_score
reviews_df['vader_class'] = vader_class
reviews_df.loc[~(reviews_df['feedback'] == reviews_df['vader_class'])]
example = "It's got great sound and bass but it doesn't work all of the time. Its still hot or miss when it recognizes things"
ss = sid.polarity_scores(example)
ss.get('compound')
ss = sid.polarity_scores(remove_stopwords(example))
ss.get('compound')
