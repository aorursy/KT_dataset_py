
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid_obj = SentimentIntensityAnalyzer() 
sentence="how are you?"
sentiment_dict = sid_obj.polarity_scores(sentence) 
print("Overall sentiment dictionary is : ", sentiment_dict) 
print("Sentence Overall Rated As", end = " ") 
  
# decide sentiment as positive, negative and neutral 
if sentiment_dict['compound'] >= 0.05 : 
    print("Positive") 
  
elif sentiment_dict['compound'] <= - 0.05 : 
    print("Negative") 
  
else : 
    print("Neutral") 