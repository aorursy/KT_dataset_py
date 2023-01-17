import numpy as np

import pandas as pd

from IPython.display import display # Allows the use of display() for DataFrames

import matplotlib.pyplot as pl

import matplotlib.patches as mpatches





# Pretty display for notebooks

%matplotlib inline



#the table combined_news= pd.read_csv('Combined_News_DJIA.csv') has the Top25 news headlines as single columns, the date 

# and the label

# label 1: the stock price increased

# label 2: the stock prce decreased





combined_news= pd.read_csv('../input/Combined_News_DJIA.csv')



#combine the 25 headlines to one single long string in one column

combined_news['CombinedTop'] = combined_news.iloc[:,2:].astype(str).apply(' '.join, axis=1)



#add continous data from the DJIA_table.csv table: difference from open and close value

djia_data = pd.read_csv('../input/DJIA_table.csv')

djia_data['diff']=djia_data['Close']-djia_data['Open']



#merge both tables to one

combined_data = pd.merge(left=djia_data[['Date', 'diff']],right=combined_news[['Date',  'Label', 'CombinedTop']], 

                         left_on='Date', right_on='Date')

print (combined_data.shape)

display(combined_data.head(5))

#missing data?



missing = combined_data.loc[combined_data['CombinedTop'] == '']

print (missing)



missing = combined_data.loc[combined_data['CombinedTop'] == 'NaN']

print (missing)



missing_diff = combined_data.loc[combined_data['diff'] == None]

print (missing_diff)

print("Describing statistics about the diff feature.")

print (combined_data['diff'].describe())

pl.xlabel('Diff')

pl.ylabel('Amount')

pl.hist(combined_data['diff'])

pl.show()
from sklearn.feature_extraction.text import CountVectorizer

from collections import Counter



ngram_vectorizer = CountVectorizer(analyzer='word', stop_words = 'english', ngram_range=(1, 1), min_df=1)



# X matrix where the row represents sentences and column is our one-hot vector for each token in our vocabulary

X = ngram_vectorizer.fit_transform(combined_data['CombinedTop'])



# Vocabulary

vocab = list(ngram_vectorizer.get_feature_names())



# Column-wise sum of the X matrix.

counts = X.sum(axis=0).A1



freq_distribution = Counter(dict(zip(vocab, counts)))

most_frequent = freq_distribution.most_common(25)



print (most_frequent)



word_list, amount_list = zip(*most_frequent)



df = pd.DataFrame(list(zip(amount_list, word_list))).set_index(1)



df.plot.barh()

#textlength



pl.xlabel('Length of words in headlines')

length_of_words = combined_data['CombinedTop'].str.split(" ").str.len()

pl.hist(length_of_words)

pl.show()

display(length_of_words.describe())
# Split the data into training and testing sets

# as instructed in the dataset the train / test split is at the 01 Jan 2015



train = combined_data[combined_data['Date'] < '2015-01-01']

test = combined_data[combined_data['Date'] > '2014-12-31']



train_features = train['CombinedTop']

test_features = test['CombinedTop']



labels_train = train['Label']

labels_test = test['Label']



diff_y_train = train['diff']

diff_y_test = test['diff']
#Can accuracy be used as a valid measure? Meaning: how is the proportion of label 0 and 1



print (combined_data['Label'].value_counts())
#takes an vectorizer and returnes the transformed train and test set as well as the names of each feature

def vectorize_words (vectorizer):

   

    features_train_transformed = vectorizer.fit_transform(train_features)

    features_test_transformed  = vectorizer.transform(test_features).toarray()

    feature_names = vectorizer.get_feature_names()



    print("Shape of vectorized words: " + str(features_train_transformed.shape))

    

    return features_train_transformed, features_test_transformed, feature_names
#use tfidf vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=.5,

                             stop_words='english')



standard_train, standard_test, standard_feature_names = vectorize_words(vectorizer)



#use tfidf vectorizer with multiple words

vectorizer_mult = TfidfVectorizer(sublinear_tf=True, min_df = 0.04, max_df=.8, ngram_range=(2, 3),

                             stop_words='english')



mult_train, mult_test, mult_feature_names = vectorize_words(vectorizer_mult)
#takes a percentile, train and test data and the feature names as input

#returns the x percent best features for prediction of the given label



def select_percentile(percentile, train_data, test_data, target_variable, feature_names):

        selector = SelectPercentile(f_classif, percentile=percentile)

        selector.fit(train_data, target_variable)

        final_train = selector.transform(train_data)

        final_test  = selector.transform(test_data)

        index_converter =  np.asarray(feature_names)[selector.get_support()]

        print("Matrix of selected words: ")

        display(final_train)

        

        return final_train, final_test, index_converter
from sklearn.feature_selection import SelectPercentile, f_classif



### feature selection, because text is super high dimensional and 

### can be really computationally chewy as a result

standard_final_train, standard_final_test, standard_index_converter = select_percentile(10, 

                                                                standard_train, standard_test, labels_train, standard_feature_names)



#2-3 word snippets already reduced to 52 words. No further selection needed





# continous data as target variable

cont_final_train, cont_final_test, cont_index_converter = select_percentile(10, 

                                                                standard_train, standard_test, diff_y_train, standard_feature_names)




from sklearn.metrics import accuracy_score



#input variables: the decision tree classifier, train and test data and the wordlist of the features

#output: the accuracy of the model and the sorted list of best predicting features



def use_decision_tree(dtClassifier, train_data, test_data, word_list):

    

    dtClassifier.fit(train_data, labels_train)

   

    index = []

    scores = []

    words = []

    

    #put the feature importance of each feature together with the wordd

    for x  in range (0, len(dtClassifier.feature_importances_)): 

        if dtClassifier.feature_importances_[x] > 0.015:

            index.append('Index: ' + str(x))

            scores.append(dtClassifier.feature_importances_[x])

            words.append(word_list[x])



    decision_tree_selection = pd.DataFrame({'Index' : index, 

                            'Scores' : scores, 'Words': words})



    decision_tree_selection = decision_tree_selection.sort_values(['Scores'], ascending=False)

    return (dtClassifier, decision_tree_selection)

   
from sklearn.tree import DecisionTreeClassifier



dtClassifier = DecisionTreeClassifier(random_state= 0,  min_samples_split= 10)

dtClassifier, standard_dt_words = use_decision_tree(dtClassifier, 

                                        standard_final_train, standard_final_test, standard_index_converter)



# to check for overfitting: how high is the accuarcy of the train data compared to the test data?

dtPredTrain = dtClassifier.predict(standard_final_train)

dtAccTrain = accuracy_score(dtPredTrain, labels_train)    

print ("Accuracy score for the decision tree train data: " + str(dtAccTrain))



# display the accuracy of the test data

dtPred = dtClassifier.predict(standard_final_test)

dtAcc = accuracy_score(dtPred, labels_test)    

print ("Accuracy score for the decision tree test data: " + str(dtAcc))

print

print ("most important words detected by decision tree:")





#plot the most important words

display (standard_dt_words)

index = np.arange(len(standard_dt_words))

bar_width = 0.35

pl.bar(index, standard_dt_words['Scores'], bar_width)

pl.xticks(index + bar_width / 2, (standard_dt_words['Words']))

pl.xlabel('Words')

pl.ylabel('Scores')

pl.title('Feature importance scoring')

pl.tight_layout()

pl.show
# calculate the average accuracy for a certain min_sample_split over 100 different random states

min_samples_split = 10



dtAcc_for_loop = 0

for random_state in range (0, 99):

    dtClassifier_for_loop = DecisionTreeClassifier(random_state= random_state,  min_samples_split= min_samples_split)

    dtClassifier_for_loop, standard_dt_words = use_decision_tree(dtClassifier_for_loop, 

                                        standard_final_train, standard_final_test, standard_index_converter)

    dtPred_for_loop = dtClassifier_for_loop.predict(standard_final_test)

    dtAcc_for_loop += accuracy_score(dtPred_for_loop, labels_test)  

print ("Over 100 random states average accuracy for min sample size of ", min_samples_split , ": ", dtAcc_for_loop/100)
# test the decision tree with the multiple words as features

dtClassifier_mult = DecisionTreeClassifier(random_state= 0, min_samples_split= 10)



dtClassifier_mult, mult_dt_words = use_decision_tree(dtClassifier_mult, mult_train, mult_test, mult_feature_names)

dtPredMult = dtClassifier_mult.predict(mult_test)

dtAccMult = accuracy_score(dtPredMult, labels_test)    

print ("Accuracy score for the decision tree: " + str(dtAccMult))

print

print ("most important words detected by decision tree:")

display(mult_dt_words)
#use logistic regression

from sklearn.linear_model import LogisticRegression



def logRegression (train_data, test_data, words):

    logRegression = LogisticRegression()

    logRegTest = logRegression.fit(train_data, labels_train)    

    basiccoeffs = logRegTest.coef_.tolist()[0]

    coeffdf = pd.DataFrame({'Words' : words, 

                            'Coefficients' : basiccoeffs})

    coeffdf = coeffdf.sort_values(['Coefficients', 'Words'], ascending=[0, 1])



    return logRegTest, coeffdf
logReg, standard_coeffdf = logRegression(standard_final_train, standard_final_test, standard_index_converter)



logRegPred = logReg.predict(standard_final_train)

standard_log_accuracy_train = accuracy_score(labels_train, logRegPred)    

print('Standard - Logistic Regression accuracy Train Data: ', standard_log_accuracy_train)



logRegPred = logReg.predict(standard_final_test)

standard_log_accuracy = accuracy_score(labels_test, logRegPred)    



print('Standard - Logistic Regression accuracy Test Data: ',standard_log_accuracy)

standard_coeffdf.head(10)
logRegMult, mult_coeffdf = logRegression(mult_train, mult_test, mult_feature_names)

logRegPredMult = logRegMult.predict(mult_test)

mult_log_accuracy = accuracy_score(labels_test, logRegPredMult) 



print('Mult - Logistic Regression accuracy: ',mult_log_accuracy)

mult_coeffdf.head(10)
from keras.models import Sequential

import tensorflow as tf

from keras.layers.core import Dense, Dropout, Activation, Lambda

from keras.utils import np_utils

from sklearn.model_selection import KFold



batch_size = 32

nb_classes = 2

input_dim = standard_final_train.shape[1]





#one-hot encode labels

labels_train_encoded = Y_test = np_utils.to_categorical(labels_train, nb_classes)

labels_test_encoded = Y_test = np_utils.to_categorical(labels_test, nb_classes)





#build the sequential model with multiple dense, activation and dropout layers

model = Sequential()

model.add(Dense(256, input_dim=input_dim))

model.add(Activation('relu'))

model.add(Dropout(0.4))

model.add(Dense(128))

model.add(Activation('relu'))

model.add(Dropout(0.4))

model.add(Dense(nb_classes))

model.add(Activation('softmax'))

model.summary()



model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
epochs = 20



model.fit(standard_final_train, labels_train_encoded, epochs= epochs, batch_size=16, validation_split=0.15)

dlPredTrain = model.predict_classes(standard_final_train, verbose=0)

dlAccTrain = accuracy_score(labels_train, dlPredTrain)

print (dlAccTrain)



dlPred = model.predict_classes(standard_final_test, verbose=0)

dlAcc = accuracy_score(labels_test, dlPred)

print (dlAcc)
from sklearn import linear_model

  

def lasso_reg(train_data, test_data):

    reg = linear_model.Lasso(alpha=0.1)

    reg.fit(train_data, diff_y_train)

    return reg



reg = lasso_reg(cont_final_train, cont_final_test)

lassoAccTrain = reg.score(cont_final_train, diff_y_train)

print ('standard - Lasso Regression R2 Train Data: ', lassoAccTrain)

lassoAcc = reg.score(cont_final_test, diff_y_test)

print ('standard - Lasso Regression R2 Test Data: ', lassoAcc)



regMult = lasso_reg(mult_train, mult_test)

lassoAccMult = regMult.score(mult_test, diff_y_test)

print ('mult - Lasso Regression R2: ', lassoAccMult) 
# create and fit the LSTM network

from keras.layers import LSTM





standard_final_train_new= standard_final_train.toarray()



# reshape input to be [samples, time steps, features]

trainX = np.reshape(standard_final_train_new, (standard_final_train_new.shape[0], 1, standard_final_train_new.shape[1]))

testX = np.reshape(standard_final_test, (standard_final_test.shape[0], 1, standard_final_test.shape[1]))



model_lstm = Sequential()

model_lstm.add(LSTM(4, batch_input_shape= (1, trainX.shape[1], trainX.shape[2]), stateful=True))

model_lstm.add(Dense(1))

model_lstm.summary()



model_lstm.compile(loss='mean_squared_error', optimizer='RMSProp')



epochs_lstm = 5



#for i in range(epochs_lstm):

model_lstm.fit(trainX, diff_y_train, epochs=epochs_lstm, batch_size=1, verbose=2, shuffle=False)

#    model_lstm.reset_states()
from math import sqrt

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



# make predictions

trainPredict = model_lstm.predict(trainX, 1)

testPredict = model_lstm.predict(testX, 1)



# report performance

rmse = sqrt(mean_squared_error(diff_y_train, trainPredict))

print('Train RMSE: %.3f' % rmse)



r2_train = r2_score(diff_y_train, trainPredict)

print('Train r2: %.3f' % r2_train)



rmse = sqrt(mean_squared_error(diff_y_test, testPredict))

print('Test RMSE: %.3f' % rmse)

r2_test = r2_score(diff_y_test, testPredict)

print('Test r2: %.3f' % r2_test)