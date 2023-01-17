#turicreate ml library for built-in machine learning models

import turicreate as tc
#Here, we are having different-different test and train dataset. 

#So, we do not need to split them 



#train dataset

tweets_train_data = tc.SFrame.read_csv('../input/train.csv')

#test dataset

tweets_test_data = tc.SFrame.read_csv('../input/test.csv')
tweets_train_data.head()
tweets_test_data.head()
tweets_train_data.show()
len(tweets_train_data)
tweets_train_data['Sentiment'].show()
tweets_train_data['word_count'] = tc.text_analytics.count_words(tweets_train_data['SentimentText'])

tweets_test_data['word_count'] = tc.text_analytics.count_words(tweets_test_data['SentimentText'])
tweets_train_data.head()
twitter_sentiment_model = tc.logistic_classifier.create(tweets_train_data, 

                                                       target='Sentiment',

                                                       features=['word_count'],

                                                       validation_set=None)
tweets_test_data['Sentiment'] = twitter_sentiment_model.predict(tweets_test_data)
tweets_test_data
text = tc.SFrame({'ItemID':[1,2,3],'SentimentText':['Be Happy', ' Have a simle on your face :)','it is a good guy']})
text['word_count'] = tc.text_analytics.count_words(text['SentimentText'])
text
text['Sentiment'] = twitter_sentiment_model.predict(text)

if(text['Sentiment'][2]):

    print("Positive Sentiment")

else:

    print("Negative Sentiment")
text[2]