import pandas as pd



import nltk

nltk.download( 'popular' )

nltk.download( 'rslp' )

from nltk.stem import RSLPStemmer

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from string import punctuation

from sklearn.feature_extraction.text import CountVectorizer



from keras.utils import np_utils

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import Adam

from sklearn.metrics import multilabel_confusion_matrix



import numpy as np

import matplotlib.pyplot as plt



from sklearn.naive_bayes import MultinomialNB

from sklearn import svm

from sklearn.metrics import f1_score



from xgboost import XGBClassifier



from scipy.sparse import hstack



import warnings
df_order_reviews = pd.read_csv("../input/brazilian-ecommerce/olist_order_reviews_dataset.csv")

df_order_reviews.head()
important_columns = ['review_comment_message', 'review_comment_title', 'review_score']

df_order_reviews_imptt_cols = df_order_reviews[important_columns]

df_order_reviews_imptt_cols.head()
print( 'Total of lines                         = {0:,.0f}'.format( df_order_reviews_imptt_cols.shape[0] ))

print( 'Total of lines without comment message = {0:,.0f}'.format( df_order_reviews_imptt_cols['review_comment_message'].isna().sum() ))

print( 'Total of lines without comment title   = {0:,.0f}'.format( df_order_reviews_imptt_cols['review_comment_title'].isna().sum() ))
#df_order_reviews_imptt_cols_na = df_order_reviews_imptt_cols.dropna(subset=['review_comment_message']).fillna({'review_comment_title':''}) # only drop messages nans and clean title

df_order_reviews_imptt_cols_na = df_order_reviews_imptt_cols.dropna(subset=['review_comment_message', 'review_comment_title']) # drop messages and titles nans
df_order_reviews_imptt_cols_na['review_score'].value_counts()
# set the amount of records for every label

#amount = 10000

#amount = 2000

amount = df_order_reviews_imptt_cols_na.shape[0]



df = pd.DataFrame( columns=df_order_reviews_imptt_cols_na.columns )

for i in df_order_reviews_imptt_cols_na['review_score'].unique():

    df_i = df_order_reviews_imptt_cols_na[df_order_reviews_imptt_cols_na['review_score'] == i].sample(frac=1)

    df = df.append( df_i.iloc[0:amount, :] )



df['review_score'].value_counts()
warnings.filterwarnings('ignore')



# tokinize

df['tok_message'] = df.apply(lambda row: word_tokenize(row['review_comment_message'], language='portuguese'), axis=1)

df['tok_title'] = df.apply(lambda row: word_tokenize(row['review_comment_title'], language='portuguese'), axis=1)

print( 'Words examples {0}'.format( df.iloc[0, -2] ))
# remove stopwords

stw = stopwords.words('portuguese')

print( 'Stopword examples {0}'.format( stw[:8] ))



ponc = list( punctuation )

print( 'Punctuation examples {0}'.format( ponc[:8] ))



remove_words = set( stw + ponc )

df['tok_message_stw'] = df.apply(lambda row: [word for word in row['tok_message'] if word not in remove_words], axis=1)

df['tok_title_stw'] = df.apply(lambda row: [word for word in row['tok_title'] if word not in remove_words], axis=1)

print( 'Words with out stopwords examples {0}'.format( df.iloc[0, -2] ))
# stemmer words

stemmer = RSLPStemmer()

df['stem_tok_message'] = df.apply(lambda row: [ stemmer.stem( word ) for word in row['tok_message_stw'] ], axis=1)

df['stem_tok_title'] = df.apply(lambda row: [ stemmer.stem( word ) for word in row['tok_title_stw'] ], axis=1)

print( 'Stemmer examples {0}'.format( df.iloc[0, -2] ))
df.head()
# vectorize words (create dictionary)

message_words = []

title_words = []

for i in range( df.shape[0] ):

    for j in range( len(df.iloc[i, -2]) ):

        message_words.append( df.iloc[i, -2][j] )

    for j in range( len(df.iloc[i, -1]) ):

        title_words.append( df.iloc[i, -1][j] )



vectorizer_message = CountVectorizer( analyzer='word' )

vectorizer_title = CountVectorizer( analyzer='word' )



vectorizer_message.fit( message_words )

vectorizer_title.fit( title_words )



print( 'Amount of words in dictionary for messages = {0}'.format( len( vectorizer_message.vocabulary_ ) ))

print( 'Amount of words in dictionary for titles = {0}'.format( len( vectorizer_title.vocabulary_ ) ))

print( 'Dictionary for messages examples {0}'.format( list( vectorizer_message.vocabulary_)[:8] ))

print( 'Dictionary for titles examples {0}'.format( list( vectorizer_title.vocabulary_)[:8] ))





warnings.filterwarnings('default')
messages = df['review_comment_message'].values #X1

titles = df['review_comment_title'].values #X2

scores = df['review_score'].values #Y

scores -= 1 # transform values 1 to 5, to 0 to 4 



print( 'Total number of lines = {0}'.format( messages.shape[0] ))
scores_dummy = np_utils.to_categorical(scores)

size_scores = len(scores_dummy[0])

print( 'Scores examples = {0}'.format( scores[:5] ))

print( 'Dummy scores examples = {0}'.format( scores_dummy[:5] ))
x_train_m, x_test, y_train, y_test = train_test_split( messages, scores_dummy, test_size=0.3, random_state=42 )

x_val_m, x_test_m, y_val, y_test = train_test_split( x_test, y_test, test_size=0.5, random_state=42 )

x_train_t, x_test, y_train, y_test = train_test_split( titles, scores_dummy, test_size=0.3, random_state=42 )

x_val_t, x_test_t, y_val, y_test = train_test_split( x_test, y_test, test_size=0.5, random_state=42 )



print('Total number of lines for training   = {0}'.format( x_train_m.shape ))

print('Total number of lines for validation = {0}'.format( x_val_m.shape ))

print('Total number of lines for testing    = {0}'.format( x_test_m.shape ))
bow_x_train_m = vectorizer_message.transform( x_train_m )

bow_x_train_t = vectorizer_title.transform( x_train_t )

bow_x_train = hstack([bow_x_train_m, bow_x_train_t], format='csr') # join the two bows

#bow_x_train = bow_x_train_m # use only the commentaries



bow_x_val_m = vectorizer_message.transform( x_val_m )

bow_x_val_t = vectorizer_title.transform( x_val_t )

bow_x_val = hstack([bow_x_val_m, bow_x_val_t], format='csr') # join the two bows

#bow_x_val = bow_x_val_m # use only the commentaries



bow_x_test_m = vectorizer_message.transform( x_test_m )

bow_x_test_t = vectorizer_title.transform( x_test_t )

bow_x_test = hstack([bow_x_test_m, bow_x_test_t], format='csr') # join the two bows

#bow_x_test = bow_x_test_m # use only the commentaries



print('Bag of words format for training   = {0}'.format( bow_x_train.shape ))

print('Bag of words format for validation = {0}'.format( bow_x_val.shape ))

print('Bag of words format for testing    = {0}'.format( bow_x_test.shape ))
model = Sequential()



#model.add( Dense( 2048, activation='relu', input_shape=( bow_x_train.shape[1], )))

model.add( Dense( 1024, activation='relu', input_shape=( bow_x_train.shape[1], )))

#model.add( Dense( 256, activation='relu', input_shape=( bow_x_train.shape[1], )))

#model.add( Dense( 64, activation='relu', input_shape=( bow_x_train.shape[1], )))

#model.add( Dense( 32, activation='relu' ))

#model.add( Dense( 16, activation='relu' ))

#model.add( Dense( 8, activation='relu' ))



model.add( Dense( size_scores, activation='softmax' ))



model.compile( Adam(), loss='categorical_crossentropy', metrics=['accuracy'] )



model.summary()
history = model.fit( bow_x_train, y_train, epochs=5, validation_data=( bow_x_val, y_val ))
plt.plot( history.history['loss'], 'b', label='Training' )

plt.plot( history.history['val_loss'], 'bo', label='Validation' )

plt.title( 'Model loss function' )

plt.xlabel( 'Epoch' )

plt.ylabel( 'Loss' )

plt.legend()

plt.show()



plt.plot( history.history['accuracy'], 'b', label='Training' )

plt.plot( history.history['val_accuracy'], 'bo', label='Validation' )

plt.title( 'Model accuracy' )

plt.xlabel( 'Epoch' )

plt.ylabel( 'Accuracy' )

plt.legend()

plt.show()
train_metrics = model.evaluate( bow_x_train, y_train )

print('Loss function in train = {0:.4f}\nAccuracy in train = {1:.4f}'.format( train_metrics[0], train_metrics[1] ))



val_metrics = model.evaluate( bow_x_val, y_val )

print('Loss function in validation = {0:.4f}\nAccuracy in validation = {1:.4f}'.format( val_metrics[0], val_metrics[1] ))



test_metrics = model.evaluate( bow_x_test, y_test )

print('Loss function in test = {0:.4f}\nAccuracy in test = {1:.4f}'.format( test_metrics[0], test_metrics[1] ))
predict_train = np.argmax( model.predict( bow_x_train ), axis=1 )

predict_val = np.argmax( model.predict( bow_x_val ), axis=1 )

predict_test = np.argmax( model.predict( bow_x_test ), axis=1 )



print( 'F1 Score for NN in training   = {0:.4f}'.format( f1_score( np.argmax( y_train, axis=1 ), predict_train, average='weighted' )))

print( 'F1 Score for NN in validation = {0:.4f}'.format( f1_score( np.argmax( y_val, axis=1 ), predict_val, average='weighted' )))

print( 'F1 Score for NN in testing    = {0:.4f}'.format( f1_score( np.argmax( y_test, axis=1 ), predict_test, average='weighted' )))
y_pred = model.predict( bow_x_test )

y_pred = np.round( y_pred )



conf_matrix = multilabel_confusion_matrix( y_test, y_pred )

conf_matrix
test1_m = 'Não gostei e nunca vou recomendar'

test1_t = 'Horrível'



test2_m = 'Não assistiria de novo'

test2_t = 'Bem fraco'



test3_m = 'Não impressiona nem decepciona'

test3_t = 'Mais ou menos'



test4_m = 'Achei bem bacana'

test4_t = 'Bom'



test5_m = 'Pretendo assistir mais vezes de tão bom'

test5_t = 'Maravilhoso'



test_list_m = [ test1_m, test2_m, test3_m, test4_m, test5_m ]

test_list_t = [ test1_t, test2_t, test3_t, test4_t, test5_t ]



test_vec_m = vectorizer_message.transform( test_list_m )

test_vec_t = vectorizer_title.transform( test_list_t )

test_vec = hstack( [test_vec_m, test_vec_t], format='csr' ) # join the two bows



test_pred = model.predict( test_vec )



test_pred_max = np.argmax( test_pred, axis=1 ) + 1



for i in range( len(test_list_m) ):

    print('Test {0} model predicted {1}'.format( i, test_pred_max[i] ))
x_train, x_test, y_train, y_test = train_test_split( messages, scores, test_size=0.3, random_state=42 )

x_val, x_test, y_val, y_test = train_test_split( x_test, y_test, test_size=0.5, random_state=42 )



y_train = y_train.astype('int')

y_val = y_val.astype('int')

y_test = y_test.astype('int')
# Multinomial Naive Bayes

nb_clf = MultinomialNB()

nb_clf.fit( bow_x_train, y_train )
predict_train = nb_clf.predict( bow_x_train )

predict_val = nb_clf.predict( bow_x_val )

predict_test = nb_clf.predict( bow_x_test )



print( 'F1 Score for MNB in training   = {0:.4f}'.format( f1_score( y_train, predict_train, average='weighted' )))

print( 'F1 Score for MNB in validation = {0:.4f}'.format( f1_score( y_val, predict_val, average='weighted' )))

print( 'F1 Score for MNB in testing    = {0:.4f}'.format( f1_score( y_test, predict_test, average='weighted' )))
# Support Vector Machines Multi-class Classification

svm_clf = svm.SVC()

svm_clf.fit( bow_x_train, y_train )
predict_train = svm_clf.predict( bow_x_train )

predict_val = svm_clf.predict( bow_x_val )

predict_test = svm_clf.predict( bow_x_test )



print( 'F1 Score for SVM in training   = {0:.4f}'.format( f1_score( y_train, predict_train, average='weighted' )))

print( 'F1 Score for SVM in validation = {0:.4f}'.format( f1_score( y_val, predict_val, average='weighted' )))

print( 'F1 Score for SVM in testing    = {0:.4f}'.format( f1_score( y_test, predict_test, average='weighted' )))
# XGBoost Classifier

xgb_clf = XGBClassifier(learning_rate = 0.01,

                         max_depth = 10, 

                         n_estimators = 1000,

                         objective = 'binary:logistic',

                         verbosity =0,

                         seed = 42,

                         reg_lambda = 8,

                         reg_alpha = 2,

                         gamma = 5,

                         subsample= 0.8,

                         #tree_method = 'gpu_hist'

                         )

xgb_clf.fit( bow_x_train, y_train )
predict_train = xgb_clf.predict( bow_x_train )

predict_val = xgb_clf.predict( bow_x_val )

predict_test = xgb_clf.predict( bow_x_test )



print( 'F1 Score for XGB in training   = {0:.4f}'.format( f1_score( y_train, predict_train, average='weighted' )))

print( 'F1 Score for XGB in validation = {0:.4f}'.format( f1_score( y_val, predict_val, average='weighted' )))

print( 'F1 Score for XGB in testing    = {0:.4f}'.format( f1_score( y_test, predict_test, average='weighted' )))