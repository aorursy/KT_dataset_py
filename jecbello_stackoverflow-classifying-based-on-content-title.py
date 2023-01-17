import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')
import bq_helper

from bq_helper import BigQueryHelper

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

stackOverflow = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="stackoverflow")
stack_db = BigQueryHelper("bigquery-public-data", "stackoverflow")

stack_db.list_tables()
stack_db.head("posts_questions", num_rows=5)
stack_db.table_schema("posts_questions")
query1 = """

         SELECT

             title,

             body as question,

             tags as labels, 

             view_count as views

         FROM

             `bigquery-public-data.stackoverflow.posts_questions`

         WHERE 

             (tags LIKE '%python%' OR

             tags LIKE '%java%' OR 

             tags LIKE '%sql%' OR 

             tags LIKE '%|r|%' OR

             tags LIKE 'r|%') AND

             LENGTH(body) < 1000

         LIMIT

             7500;

         """



questions_df = stackOverflow.query_to_pandas(query1)
#import necessary packages for analysis 

import nltk

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('whitegrid')
questions_df['labels'] = questions_df['labels'].str.split('|')
def return_tags(labels):

    langauges = [lang for lang in labels if lang in ['python','java','sql','r','javascript']] 

    return langauges
questions_df['labels'] = questions_df['labels'].apply(return_tags)
# Find rows that contain only a single tag

processed_df = questions_df[(questions_df['labels'].apply(len) > 0) & (questions_df['labels'].apply(len) < 2)]  
#verify that only questions with a single language tag are included

processed_df.head()
processed_df.info()
# In order to properly work with these labels, we must convert them from lists into strings.

def lst_to_str(lst):

    unpacked = ''.join(lst)

    return unpacked



processed_df.loc[:,'labels'] = processed_df.loc[:,'labels'].apply(lst_to_str)



#processed_df.describe()

#processed_df.groupby('labels').describe()
grid = sns.FacetGrid(processed_df[processed_df['views'] < 4000], col = 'labels', height = 5, aspect = 0.6)

grid.map(plt.hist, 'views', bins = 75)

axes = grid.axes

axes[0,1].set_xlim([0,4000])



plt.tight_layout()
processed_df.loc[:,'length_of_question'] = processed_df.loc[:,'question'].apply(len)
# verify the procedure worked.

processed_df.head(2)
# Plot the length of each question based on what language they were written in

grid = sns.FacetGrid(processed_df, hue = 'labels', height = 5, aspect = 2)

grid.map(plt.hist, 'length_of_question', bins = 50, alpha = 0.5)

axes = grid.axes



axes[0,0].set_title('Distribution of question lengths for each programming language')

axes[0,0].set_xlim([0,1000])

axes[0,0].legend()
# An example of a question posted on stack overflow. Notice the html syntax and code delimiters.

processed_df.head()
import lxml.html



def find_code(html_str):

    final_list = []



    dom = lxml.html.fromstring(html_str)

    codes = dom.xpath('//code')



    for code in codes:

        if code.text is None: 

            final_list.append('')

        else:

            final_list.append(code.text)

        

        

    final_list = ' '.join(final_list)

    return final_list 
processed_df.loc[:,'code'] = processed_df.loc[:,'question'].apply(find_code)
def count_colons(txt):

    return txt.count(':')



def count_semicolons(txt):

    return txt.count(';')



def count_slashes(txt):

    return txt.count('/')

                                      

def count_cbrackets(txt):

    return txt.count('{') + txt.count('}')



def count_sbrackets(txt):

    return txt.count('[') + txt.count(']')



def count_quotes(txt):

    return txt.count('"') + txt.count("'")



def count_arithmetic(txt):

    return txt.count('<') + txt.count('>') + txt.count('-') + txt.count('+') 



def count_period(txt):

    return txt.count('.')
processed_df.loc[:,'colon count']     = processed_df.loc[:,'code'].apply(count_colons)

processed_df.loc[:,'semicolon count'] = processed_df.loc[:,'code'].apply(count_semicolons)

processed_df.loc[:,'slash count']     = processed_df.loc[:,'code'].apply(count_slashes)

processed_df.loc[:,'cbracket count']  = processed_df.loc[:,'code'].apply(count_cbrackets)

processed_df.loc[:,'sbracket count']  = processed_df.loc[:,'code'].apply(count_sbrackets)

processed_df.loc[:,'quote count']     = processed_df.loc[:,'code'].apply(count_quotes)

processed_df.loc[:,'operator count']  = processed_df.loc[:,'code'].apply(count_arithmetic)

processed_df.loc[:,'period count']    = processed_df.loc[:,'code'].apply(count_period)
# Verify the functions worked.

processed_df.head()
fig, axis = plt.subplots(figsize=(25,30), nrows = 5)



python_features = processed_df[processed_df['labels'] == 'python'].loc[:,'colon count':]

js_features = processed_df[processed_df['labels'] == 'javascript'].loc[:,'colon count':]

java_features = processed_df[processed_df['labels'] == 'java'].loc[:,'colon count':]

sql_features = processed_df[processed_df['labels'] == 'sql'].loc[:,'colon count':]

r_features = processed_df[processed_df['labels'] == 'r'].loc[:,'colon count':]



sql_features.head()



sns.heatmap(python_features, cmap = 'viridis', ax = axis[0])

axis[0].set_title('Python features heatmap')



sns.heatmap(js_features, cmap = 'inferno', ax = axis[1])

axis[1].set_title('Javascript features heatmap')



sns.heatmap(java_features, cmap = 'viridis', ax = axis[2])

axis[2].set_title('Java features heatmap')



sns.heatmap(sql_features, cmap = 'inferno', ax = axis[3])

axis[3].set_title('SQL features heatmap')



sns.heatmap(r_features, cmap = 'viridis', ax = axis[4])

axis[4].set_title('R features heatmap')

total_syntax_features = processed_df.groupby('labels').sum(axis=1).loc[:,'colon count':]



fig, aggregate_axis = plt.subplots(figsize=(15,6))

sns.heatmap(total_syntax_features, cmap = 'plasma', ax = aggregate_axis)

aggregate_axis.set_title('Syntactic Features of Programming Languages')
from sklearn.preprocessing import MinMaxScaler



min_max_scaler = MinMaxScaler()

total_syntax_features_scaled = pd.DataFrame(min_max_scaler.fit_transform(total_syntax_features.T), columns = total_syntax_features.index, index = total_syntax_features.columns)

total_syntax_features_scaled.head(10)



fig, axis_scaled = plt.subplots(figsize=(15,6))

sns.heatmap(total_syntax_features_scaled.T, cmap = 'plasma', ax = axis_scaled)

axis_scaled.set_title('Normalized Code Features')
from bs4 import BeautifulSoup



def find_text(html_str):

    full_text = ''



    parsedContent = BeautifulSoup(html_str, 'html.parser')



    text = parsedContent.findAll('p')

    

    for paragraph in text:

        full_text = full_text + paragraph.getText()

        

    return full_text    
import re 

import string

from nltk.corpus import stopwords 



stop_words = stopwords.words()

translation_table = dict.fromkeys(map(ord, string.punctuation), None)



def remove_punc_and_stopwords(full_text):

    cleaned_text = full_text.translate(translation_table)

    word_lst = re.findall('[a-zA-Z]+', cleaned_text)

    return " ".join(word_lst)
def clean_html_text(text):

    final_text = find_text(text)

    bag_of_words = remove_punc_and_stopwords(final_text)

    return bag_of_words
processed_df.head(2)
# Remove digits and special characters from title column. Note that it does not require an html parser.

processed_df.loc[:,'title'] = processed_df.loc[:,'title'].apply(remove_punc_and_stopwords)
# Extract the text of the question column, and remove digits and special characters.

processed_df.loc[:,'question'] = processed_df.loc[:,'question'].apply(clean_html_text)
# Observe that there are no longer any punctuation or digits in the questions.

processed_df.head(10)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



cv_title = CountVectorizer().fit(processed_df['title'])

vectorized_title = cv_title.transform(processed_df['title'])

vectorized_title_df = pd.DataFrame(vectorized_title.toarray(), columns = cv_title.get_feature_names())



# implement tfidf

tfidf_title = TfidfTransformer().fit(vectorized_title)

vectorized_tfidf_title = tfidf_title.transform(vectorized_title)

vectorized_tfidf_title_df = pd.DataFrame(vectorized_tfidf_title.toarray(), columns = cv_title.get_feature_names())
cv_question = CountVectorizer().fit(processed_df['question'])

vectorized_question = cv_question.transform(processed_df['question'])

vectorized_question_df = pd.DataFrame(vectorized_question.toarray(), columns = cv_question.get_feature_names())



# implement tfidf

tfidf_question = TfidfTransformer().fit(vectorized_question)

vectorized_tfidf_question = tfidf_question.transform(vectorized_question)

vectorized_tfidf_question_df = pd.DataFrame(vectorized_tfidf_question.toarray(), columns = cv_question.get_feature_names())
# combine dataframes 

cv_bow = pd.concat([vectorized_title_df, vectorized_question_df], axis = 1)

cv_tfidf_bow = pd.concat([vectorized_tfidf_title_df, vectorized_tfidf_question_df], axis = 1)
# import multinomial NB classifier and fit to dataset

from sklearn.naive_bayes import MultinomialNB



nb_cv_words_only = MultinomialNB().fit(cv_bow, processed_df['labels'])

nb_cv_tfidf_words_only = MultinomialNB().fit(cv_tfidf_bow, processed_df['labels'])
# make predictions

cv_bow_predictions = nb_cv_words_only.predict(cv_bow)

cv_tfidf_bow_predictions = nb_cv_tfidf_words_only.predict(cv_tfidf_bow)
# display classification report + confusion matrix

from sklearn.metrics import classification_report, confusion_matrix

print('Multinomial Naive-Bayes Classification Report: ')

print(classification_report(processed_df['labels'], cv_bow_predictions))

print('\n')

print('Multinomial Naive-Bayes with TFIDF Classification Report: ')

print(classification_report(processed_df['labels'], cv_tfidf_bow_predictions))



fig, axes = plt.subplots(nrows=2,figsize=(10,10))



sns.heatmap(confusion_matrix(processed_df['labels'], cv_bow_predictions), ax = axes[0], annot=True, cmap='magma', fmt = 'g',

            xticklabels=['java', 'javascript','python','r','sql'], yticklabels=['java', 'javascript','python','r','sql'])

axes[0].set_title('Confusion Matrix of Naive-Bayes Classification')

axes[0].set_xlabel('Predicted Label')

axes[0].set_ylabel('True Label')



sns.heatmap(confusion_matrix(processed_df['labels'], cv_tfidf_bow_predictions), ax = axes[1], annot=True, cmap='magma', fmt = 'g',

            xticklabels=['java', 'javascript','python','r','sql'], yticklabels=['java', 'javascript','python','r','sql'])

axes[1].set_title('Confusion Matrix of Naive-Bayes Classification with TFIDF')

axes[1].set_xlabel('Predicted Label')

axes[1].set_ylabel('True Label')



plt.tight_layout()
# Repeat process for bag of words including code features

code_features = processed_df.loc[:,'colon count':]



cv_words_and_code = pd.concat([cv_bow.reset_index(drop = True), code_features.reset_index(drop = True)], axis = 1)

cv_words_and_code_tfidf = TfidfTransformer().fit_transform(cv_words_and_code)



nb_cv_words_and_code = MultinomialNB().fit(cv_words_and_code, processed_df['labels'])

nb_cv_words_and_code_tfidf = MultinomialNB().fit(cv_words_and_code_tfidf, processed_df['labels'])



nb_complete_predictions = nb_cv_words_and_code.predict(cv_words_and_code)

nb_tfidf_complete_predictions = nb_cv_words_and_code.predict(cv_words_and_code_tfidf)
print('Multinomial Naive-Bayes with code features: ')

print(classification_report(processed_df['labels'], nb_complete_predictions))



print('\n')



print('Multinomial Naive-Bayes with TFIDF and code features: ')

print(classification_report(processed_df['labels'], nb_tfidf_complete_predictions))



fig, axes = plt.subplots(nrows=2,figsize=(10,10))

sns.heatmap(confusion_matrix(processed_df['labels'], nb_complete_predictions), ax = axes[0], annot=True, cmap='magma', fmt = 'g',

            xticklabels=['java', 'javascript','python','r','sql'], yticklabels=['java', 'javascript','python','r','sql'])

axes[0].set_title('Confusion Matrix of Naive-Bayes Classification with code features')

axes[0].set_xlabel('Predicted Label')

axes[0].set_ylabel('True Label')

sns.heatmap(confusion_matrix(processed_df['labels'], nb_tfidf_complete_predictions), ax = axes[1], annot=True, cmap='magma', fmt = 'g',

            xticklabels=['java', 'javascript','python','r','sql'], yticklabels=['java', 'javascript','python','r','sql'])

axes[1].set_title('Confusion Matrix of Naive-Bayes Classification with TFIDF and code features')

axes[1].set_xlabel('Predicted Label')

axes[1].set_ylabel('True Label')
from sklearn.preprocessing import StandardScaler
views_and_length = processed_df.loc[:,['views','length_of_question']]

code_features = processed_df.loc[:,'colon count':]



# Scale the views column

ss_view_length = StandardScaler()

scaled_views_and_length = ss_view_length.fit_transform(views_and_length.astype(float)) 

scaled_views_and_length_df = pd.DataFrame(scaled_views_and_length, columns = views_and_length.columns)



# Scale the code features

ss_code = StandardScaler()

scaled_code_features = ss_code.fit_transform(code_features.astype(float))

scaled_code_features_df = pd.DataFrame(scaled_code_features, columns = code_features.columns)



# Concatenate the results into complete preprocessed dataframe.

final_scaled_vectorized_df = pd.concat([vectorized_title_df, vectorized_question_df, scaled_views_and_length_df, scaled_code_features_df], axis = 1)

final_scaled_vectorized_df.head(2)
final_scaled_vectorized_df.info()
from keras.models import Sequential

from keras.layers import Dense, Dropout



classifier = Sequential()

classifier.add(Dense(units = 10000,

                     input_shape = (30040,),

                     kernel_initializer = 'glorot_uniform',

                     activation = 'relu'

                    )

              )  

#classifier.add(Dropout(0.35))

classifier.add(Dense(units = 1150,

                     kernel_initializer = 'glorot_uniform',

                     activation = 'relu'

                    )

              )

#classifier.add(Dropout(0.25))

classifier.add(Dense(units = 130,

                     kernel_initializer = 'glorot_uniform',

                     activation = 'relu'

                    )

              )

#classifier.add(Dropout(0.25))

classifier.add(Dense(units = 50,

                     kernel_initializer = 'glorot_uniform',

                     activation = 'relu'

                    )

              )

classifier.add(Dense(units = 5,

                     kernel_initializer = 'glorot_uniform',

                     activation = 'softmax'

                    )

              )



classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
from sklearn.model_selection import train_test_split



X = final_scaled_vectorized_df

y = pd.get_dummies(processed_df['labels'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#training the data. A batch size and number of epochs were chosen to remain within the memory constraints.

classifier.fit(X_train, y_train, batch_size = 42, epochs = 75)
# store the predictions. 

predictions = classifier.predict(X_test)

predictions
# Retrieve labels by taking the max value in each row and convert it to a 1, effectively 'labeling' the resutls.

from sklearn.preprocessing import LabelBinarizer



labels = np.argmax(predictions, axis = 1)

lb = LabelBinarizer()

labeled_predictions = lb.fit_transform(labels)
from sklearn.metrics import classification_report



print(classification_report(y_test.values, labeled_predictions))
from sklearn.metrics import confusion_matrix



print('Confusion matrix for Java')

print(confusion_matrix(y_test.loc[:,'sql'], labeled_predictions[:,0]))

print('\n')



print('Confusion matrix for Javascript')

print(confusion_matrix(y_test.loc[:,'javascript'], labeled_predictions[:,1]))

print('\n')



print('Confusion matrix for Python')

print(confusion_matrix(y_test.loc[:,'python'], labeled_predictions[:,2]))

print('\n')



print('Confusion matrix for R')

print(confusion_matrix(y_test.loc[:,'r'], labeled_predictions[:,3]))

print('\n')



print('Confusion matrix for SQL')

print(confusion_matrix(y_test.loc[:,'sql'], labeled_predictions[:,4])) 
