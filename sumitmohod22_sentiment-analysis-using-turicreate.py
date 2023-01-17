! pip install turicreate

import turicreate



from bokeh.io import output_notebook, show

from bokeh.plotting import figure

from bokeh.models import HoverTool

output_notebook()
# Reading the data and creating an SFrame of the data

products = turicreate.SFrame.read_csv('../input/amazon-baby-sentiment-analysis/amazon_baby.csv')



# Exploring dataset

products
products.groupby('name',operations={'count':turicreate.aggregate.COUNT()}).sort('count', ascending= False).head(5)
giraffe_reviews = products[products['name']=='Vulli Sophie the Giraffe Teether']

giraffe_reviews['rating'].show()
products['word_count'] = turicreate.text_analytics.count_words(products['review'])

products.head(5)
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

# Loop through word counts to create a classifier for only a few words 

# Created an individual column for each item 

for word in selected_words:

    products[word] = products['word_count'].apply(lambda counts: counts.get(word, 0))



products.head(5)
products['rating'].show()
#ignore all 3*  reviews

products = products[products['rating']!= 3]



#positive sentiment = 4-star or 5-star reviews

products['sentiment'] = products['rating'] >= 4



products.head(5)
products['sentiment'].show()
train_data,test_data = products.random_split(.8,seed=0)                  # using 80% data for trainning and the rest for Testing
# Classification Model using all words

sentiment_model = turicreate.logistic_classifier.create(train_data,target='sentiment', features=['word_count'], validation_set=test_data)
predictions = sentiment_model.classify(test_data)

print (predictions)
roc = sentiment_model.evaluate(test_data, metric= 'roc_curve')

roc
p = figure(title= 'ROC Curve for all words Sentiment Model', plot_width=600, plot_height=400)



p.line(x= roc['roc_curve']['fpr'], y= roc['roc_curve']['tpr'], line_width=2 , legend_label="ROC Curve Class")

p.line([0, 1], [0, 1], line_dash="dotted", line_color="indigo", line_width=2)

p.add_tools(HoverTool(tooltips=[("False Positive Rate", "@x"), ("True Positive Rate", "@y")])) 

p.xaxis.axis_label = 'False Positive Rate'

p.yaxis.axis_label = 'True Positive Rate'

p.legend.location = 'bottom_right'

show(p)
result = sentiment_model.evaluate(test_data)

print ("Accuracy             : {}".format(result['accuracy']))

print ("Area under ROC Curve : {}".format(result['auc']))

print ("Confusion Matrix     : \n{}".format(result['confusion_matrix']))

print ("F1_score             : {}".format(result['f1_score']))

print ("Precision            : {}".format(result['precision']))

print ("Recall               : {}".format(result['recall']))

print ("Log_loss             : {}".format(result['log_loss']))
productsdata = products.copy()

productsdata['predicted_sentiment'] = sentiment_model.predict(productsdata, output_type = 'probability')

# As above identified the most popular Amazon Baby Product is 'Vulli Sophie the Giraffe Teether'

giraffe_reviews = productsdata[productsdata['name']== 'Vulli Sophie the Giraffe Teether']

giraffe_reviews = giraffe_reviews.sort('predicted_sentiment', ascending=False)



# Most positive review for most popular Amazon Baby Product

print('Most Positive review for Vulli Sophie the Giraffe Teether:\n\n ', giraffe_reviews[0]['review'])

print('\n\n')

# Most negative review for most popular Amazon Baby Product

print('Most Negative review for Vulli Sophie the Giraffe Teether:\n\n ', giraffe_reviews[-1]['review'])
# Features to be trained on selected words Model

selected_words_feat = selected_words
# Classification Model using selected words

selected_words_model = turicreate.logistic_classifier.create(train_data,target='sentiment', features= selected_words_feat, validation_set=test_data)
predictions = selected_words_model.classify(test_data)

print (predictions)
roc_swm = selected_words_model.evaluate(test_data, metric= 'roc_curve')

roc_swm
p = figure(title= 'ROC Curve for selected words Sentiment Model', plot_width=600, plot_height=400)



p.line(x= roc_swm['roc_curve']['fpr'], y= roc_swm['roc_curve']['tpr'], line_width=2 , legend_label="ROC Curve Class")

p.line([0, 1], [0, 1], line_dash="dotted", line_color="indigo", line_width=2)

p.add_tools(HoverTool(tooltips=[("False Positive Rate", "@x"), ("True Positive Rate", "@y")])) 

p.xaxis.axis_label = 'False Positive Rate'

p.yaxis.axis_label = 'True Positive Rate'

p.legend.location = 'bottom_right'

show(p)
result_swm = selected_words_model.evaluate(test_data)

print ("Accuracy             : {}".format(result_swm['accuracy']))

print ("Area under ROC Curve : {}".format(result_swm['auc']))

print ("Confusion Matrix     : \n{}".format(result_swm['confusion_matrix']))

print ("F1_score             : {}".format(result_swm['f1_score']))

print ("Precision            : {}".format(result_swm['precision']))

print ("Recall               : {}".format(result_swm['recall']))

print ("Log_loss             : {}".format(result_swm['log_loss']))
for word in selected_words:

    print("\nThe number of times {} appears: {}".format(word, products[word].sum()))
swm_weights= selected_words_model.coefficients.sort(key_column_names='value', ascending=False)

swm_weights.head(5)
print('Out of the 11 words in selected_words, Most Positive: ', 

      swm_weights[swm_weights['value'] == swm_weights['value'].max()]['name'][0])

print('\n')

print('Out of the 11 words in selected_words, Most Negative: ', 

      swm_weights[swm_weights['value'] == swm_weights['value'].min()]['name'][0])
# For sentiment_model

diaper_champ_reviews = products[products['name'] == 'Baby Trend Diaper Champ']            # extracts data only product named 'diaper_champ_reviews'

diaper_champ_reviews['predicted_sentiment'] = sentiment_model.predict(diaper_champ_reviews, output_type = 'probability')

diaper_champ_reviews = diaper_champ_reviews.sort('predicted_sentiment', ascending=False)

diaper_champ_reviews.head(5)
# Predicted Sentiment for the most positive review 

print('Predicted Sentiment for Most Positive review:  ', diaper_champ_reviews[0]['predicted_sentiment'])

# Most positive review for ‘Baby Trend Diaper Champ’

print('Most positive review for ‘Baby Trend Diaper Champ’:\n\n ', diaper_champ_reviews[0]['review'])

print('\n\n')



# Predicted Sentiment for the most negative review

print('Predicted Sentiment for Most Negative review:  ', diaper_champ_reviews[-1]['predicted_sentiment'])

# Most negative review for ‘Baby Trend Diaper Champ’

print('Most negative review for ‘Baby Trend Diaper Champ’:\n\n ', diaper_champ_reviews[-1]['review'])
# For selected_words_model

dcr_swm = products[products['name'] == 'Baby Trend Diaper Champ']            # extracts data only product named 'diaper_champ_reviews'

dcr_swm['predicted_sentiment'] = selected_words_model.predict(dcr_swm, output_type = 'probability')

dcr_swm = dcr_swm.sort('predicted_sentiment', ascending=False)

dcr_swm.head(5)
# Predicted Sentiment for the most positive review 

print('Predicted Sentiment for Most Positive review:  ', dcr_swm[0]['predicted_sentiment'])

# Most positive review for ‘Baby Trend Diaper Champ’

print('Most positive review for ‘Baby Trend Diaper Champ’:\n\n ', dcr_swm[0]['review'])

print('\n\n')



# Predicted Sentiment for the most negative review

print('Predicted Sentiment for Most Negative review:  ', dcr_swm[-1]['predicted_sentiment'])

# Most negative review for ‘Baby Trend Diaper Champ’

print('Most negative review for ‘Baby Trend Diaper Champ’:\n\n ', dcr_swm[-1]['review'])
dcr_swm[dcr_swm['word_count'] == diaper_champ_reviews['word_count'][0]]
def Calculate_y_hat(scores):

    y_hat = []

    for score in scores:

        if score>0:

            y_hat.append(1)

        else:y_hat.append(-1)

    return y_hat



def get_classification_accuracy(model, data, true_labels):

    # First get the predictions

    scores = model.predict(data, output_type='margin')

    

    # Compute the number of correctly classified examples

    count_correct_classified_samples = 0

    y_hat =  Calculate_y_hat(scores)

    

    for i in range(len(scores)):

        if y_hat[i] == true_labels[i]:

            count_correct_classified_samples+=1



    # Then compute accuracy by dividing num_correct by total number of examples

    accuracy = count_correct_classified_samples/(len(scores))

    

    return accuracy
get_classification_accuracy(sentiment_model, test_data, test_data['sentiment'])
get_classification_accuracy(selected_words_model, test_data, test_data['sentiment'])