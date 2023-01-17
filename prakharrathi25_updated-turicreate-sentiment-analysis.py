! pip install turicreate

import turicreate
# Reading the data and creating an SFrame of the data

products = turicreate.SFrame.read_csv('../input/reviews-of-amazon-baby-products/amazon_baby.csv')

products
products
products.groupby('name',operations={'count':turicreate.aggregate.COUNT()}).sort('count',ascending=False)


selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

products['word_count'] = turicreate.text_analytics.count_words(products['review'] )
# Loop through word counts to create a classifier for only a few words 

# Created an individual column for each item 

for word in selected_words:

    products[word] = products['word_count'].apply(lambda counts: counts.get(word, 0))



products
for word in selected_words:

    print("\nThe number of times {} appears: {}".format(word, products[word].sum()))
train_data,test_data = products.random_split(.8, seed=0)
# Features to be trained on 

features = selected_words
products['rating'].show()
#ignore all 3*  reviews

products = products[products['rating']!= 3]
#positive sentiment = 4-star or 5-star reviews

products['sentiment'] = products['rating'] >= 4
products
products['sentiment'].show()
train_data,test_data = products.random_split(.8,seed=0)
# Original Analysis Model 

sentiment_model = turicreate.logistic_classifier.create(train_data,target='sentiment', 

                                                        features=['word_count'], 

                                                        validation_set=test_data)
# Creating the model with selected words 

selected_words_model = turicreate.logistic_classifier.create(train_data,target='sentiment', 

                                                        features=features, 

                                                        validation_set=test_data)
# Calling and descreibing our coefficients and weights allotted to each word

selected_words_model.coefficients.sort(key_column_names='value', ascending=True)
# Evaluate the orginal analysis model first 

sentiment_model.evaluate(test_data)
# Evaluate the limited words model 

selected_words_model.evaluate(test_data)
# Extract only the relevant data

diaper_champ_reviews = products[products['name']== 'Baby Trend Diaper Champ']

diaper_champ_reviews
selected_words_model.predict(diaper_champ_reviews[0:1], output_type='probability')
diaper_champ_reviews['predicted_sentiment'] = selected_words_model.predict(diaper_champ_reviews, output_type = 'probability')
products
diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment', ascending=False)
diaper_champ_reviews
diaper_champ_reviews.tail()
diaper_champ_reviews[0]['review']
diaper_champ_reviews[1]['review']
diaper_champ_reviews[-1]['review']
diaper_champ_reviews[-2]['review']