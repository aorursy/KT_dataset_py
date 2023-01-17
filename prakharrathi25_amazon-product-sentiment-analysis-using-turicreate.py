! pip install turicreate

import turicreate
# Reading the data and creating an SFrame of the data

products = turicreate.SFrame.read_csv('../input/reviews-of-amazon-baby-products/amazon_baby.csv')

products
# Looking at our dataset format

products
# Grouping Data by names and number of reviews 

products.groupby('name',operations={'count':turicreate.aggregate.COUNT()}).sort('count',ascending=False)
giraffe_reviews = products[products['name']=='Vulli Sophie the Giraffe Teether']
giraffe_reviews
# Number of Vulli Sophie Reviews

len(giraffe_reviews)
# Let's look at it in a more categorical format to look at individual ratings 

giraffe_reviews['rating'].show()
# With Turicreate, tokenization and vectorization happens with just one function rather than multiple processes

products['word_count'] = turicreate.text_analytics.count_words(products['review'])
# Let's look at the dataset to look at the word_count filed which adds the wordcount vector

products
products['rating'].show()
# Let's ignore 3 star products because they seem to be neutral in opinion 

products = products[products['rating']!= 3]
# Define positive sentiment = 4-star or 5-star reviews

products['sentiment'] = products['rating'] >= 4
products.head(20)
# Let's look at the distribution of the sentiments across the dataframe

products['sentiment'].show()
# Start by splitting the data into training and testing data

train_data,test_data = products.random_split(.8,seed=0)
# Building the sentiment model already there in the turiCreate Library

sentiment_model = turicreate.logistic_classifier.create(train_data,

                                                        target='sentiment', 

                                                        features=['word_count'], 

                                                        validation_set=test_data)
# Using AUC-ROC curve for evaluation of the model

sentiment_model.evaluate(test_data, metric='roc_curve')
giraffe_reviews['predicted_sentiment'] = sentiment_model.predict(giraffe_reviews, output_type = 'probability')
giraffe_reviews
giraffe_reviews = products[products['name']== 'Vulli Sophie the Giraffe Teether']
giraffe_reviews
giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)
giraffe_reviews
giraffe_reviews.tail()
giraffe_reviews[0]['review']
giraffe_reviews[1]['review']
giraffe_reviews[-1]['review']
giraffe_reviews[-2]['review']