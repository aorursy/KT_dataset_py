# importing required packages



import nltk

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from nltk.stem import WordNetLemmatizer

from nltk.corpus import wordnet

import re, collections

from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.svm import SVR

from sklearn import ensemble

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import cohen_kappa_score
# getting data in a pandas dataframe



dataframe = pd.read_csv("../input/essays_and_scores.csv", encoding = 'latin-1')
%matplotlib inline

dataframe.boxplot(column = 'domain1_score', by = 'essay_set', figsize = (10, 10))

# getting relevant columns



data = dataframe[['essay_set','essay','domain1_score']].copy()



print(data)



# Tokenize a sentence into words



def sentence_to_wordlist(raw_sentence):

    

    clean_sentence = re.sub("[^a-zA-Z0-9]"," ", raw_sentence)

    tokens = nltk.word_tokenize(clean_sentence)

    

    return tokens



# tokenizing an essay into a list of word lists



def tokenize(essay):

    stripped_essay = essay.strip()

    

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    raw_sentences = tokenizer.tokenize(stripped_essay)

    

    tokenized_sentences = []

    for raw_sentence in raw_sentences:

        if len(raw_sentence) > 0:

            tokenized_sentences.append(sentence_to_wordlist(raw_sentence))

    

    return tokenized_sentences



# calculating average word length in an essay



def avg_word_len(essay):

    

    clean_essay = re.sub(r'\W', ' ', essay)

    words = nltk.word_tokenize(clean_essay)

    

    return sum(len(word) for word in words) / len(words)



# calculating number of words in an essay



def word_count(essay):

    

    clean_essay = re.sub(r'\W', ' ', essay)

    words = nltk.word_tokenize(clean_essay)

    

    return len(words)





# calculating number of characters in an essay



def char_count(essay):

    

    clean_essay = re.sub(r'\s', '', str(essay).lower())

    

    return len(clean_essay)



# calculating number of sentences in an essay



def sent_count(essay):

    

    sentences = nltk.sent_tokenize(essay)

    

    return len(sentences)



# calculating number of lemmas per essay



def count_lemmas(essay):

    

    tokenized_sentences = tokenize(essay)      

    

    lemmas = []

    wordnet_lemmatizer = WordNetLemmatizer()

    

    for sentence in tokenized_sentences:

        tagged_tokens = nltk.pos_tag(sentence) 

        

        for token_tuple in tagged_tokens:

        

            pos_tag = token_tuple[1]

        

            if pos_tag.startswith('N'): 

                pos = wordnet.NOUN

                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))

            elif pos_tag.startswith('J'):

                pos = wordnet.ADJ

                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))

            elif pos_tag.startswith('V'):

                pos = wordnet.VERB

                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))

            elif pos_tag.startswith('R'):

                pos = wordnet.ADV

                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))

            else:

                pos = wordnet.NOUN

                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))

    

    lemma_count = len(set(lemmas))

    

    return lemma_count



# checking number of misspelled words



def count_spell_error(essay):

    

    clean_essay = re.sub(r'\W', ' ', str(essay).lower())

    clean_essay = re.sub(r'[0-9]', '', clean_essay)

    

    #big.txt: It is a concatenation of public domain book excerpts from Project Gutenberg 

    #         and lists of most frequent words from Wiktionary and the British National Corpus.

    #         It contains about a million words.

    data = open('../input/big.txt').read()

    

    words_ = re.findall('[a-z]+', data.lower())

    

    word_dict = collections.defaultdict(lambda: 0)

                       

    for word in words_:

        word_dict[word] += 1

                       

    clean_essay = re.sub(r'\W', ' ', str(essay).lower())

    clean_essay = re.sub(r'[0-9]', '', clean_essay)

                        

    mispell_count = 0

    

    words = clean_essay.split()

                        

    for word in words:

        if not word in word_dict:

            mispell_count += 1

    

    return mispell_count



# calculating number of nouns, adjectives, verbs and adverbs in an essay



def count_pos(essay):

    

    tokenized_sentences = tokenize(essay)

    

    noun_count = 0

    adj_count = 0

    verb_count = 0

    adv_count = 0

    

    for sentence in tokenized_sentences:

        tagged_tokens = nltk.pos_tag(sentence)

        

        for token_tuple in tagged_tokens:

            pos_tag = token_tuple[1]

        

            if pos_tag.startswith('N'): 

                noun_count += 1

            elif pos_tag.startswith('J'):

                adj_count += 1

            elif pos_tag.startswith('V'):

                verb_count += 1

            elif pos_tag.startswith('R'):

                adv_count += 1

            

    return noun_count, adj_count, verb_count, adv_count

    

    

# getiing Bag of Words (BOW) counts



def get_count_vectors(essays):

    

    vectorizer = CountVectorizer(max_features = 10000, ngram_range=(1, 3), stop_words='english')

    

    count_vectors = vectorizer.fit_transform(essays)

    

    feature_names = vectorizer.get_feature_names()

    

    return feature_names, count_vectors

# splitting data into train data and test data (70/30)



# feature_names_cv, count_vectors = get_count_vectors(data[data['essay_set'] == 1]['essay'])



X_cv = count_vectors.toarray()



y_cv = data[data['essay_set'] == 1]['domain1_score'].as_matrix()



X_train, X_test, y_train, y_test = train_test_split(X_cv, y_cv, test_size = 0.3)





# # Training a Linear Regression model using only Bag of Words (BOW)



# linear_regressor = LinearRegression()



# linear_regressor.fit(X_train, y_train)



# y_pred = linear_regressor.predict(X_test)



# # The coefficients

# print('Coefficients: \n', linear_regressor.coef_)



# # The mean squared error

# print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))



# # Explained variance score: 1 is perfect prediction

# print('Variance score: %.2f' % linear_regressor.score(X_test, y_test))



# # Cohen’s kappa score: 1 is complete agreement

# print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))

# Training a Lasso Regression model (l1 regularization) using only Bag of Words (BOW)



alphas = np.array([3, 1, 0.3, 0.1, 0.03, 0.01])



lasso_regressor = Lasso()



grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas))

grid.fit(X_train, y_train)



y_pred = grid.predict(X_test)



# summarize the results of the grid search

print(grid.best_score_)

print(grid.best_estimator_.alpha)



# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))



# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % grid.score(X_test, y_test))



# Cohen’s kappa score: 1 is complete agreement

print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))

# extracting essay features



def extract_features(data):

    

    features = data.copy()

    

    features['char_count'] = features['essay'].apply(char_count)

    

    features['word_count'] = features['essay'].apply(word_count)

    

    features['sent_count'] = features['essay'].apply(sent_count)

    

    features['avg_word_len'] = features['essay'].apply(avg_word_len)

    

    features['lemma_count'] = features['essay'].apply(count_lemmas)

    

    features['spell_err_count'] = features['essay'].apply(count_spell_error)

    

    features['noun_count'], features['adj_count'], features['verb_count'], features['adv_count'] = zip(*features['essay'].map(count_pos))

    

    return features

# extracting features from essay set1



features_set1 = extract_features(data[data['essay_set'] == 1])



print(features_set1)



# splitting data (BOW + other features) into train data and test data (70/30)

    

features_set1 = extract_features(data[data['essay_set'] == 1])



X = np.concatenate((features_set1.iloc[:, 3:].as_matrix(), X_cv), axis = 1)



y = features_set1['domain1_score'].as_matrix()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)



# Training a Linear Regression model using all the features (BOW + other features)



linear_regressor = LinearRegression()



linear_regressor.fit(X_train, y_train)



y_pred = linear_regressor.predict(X_test)



# The coefficients

print('Coefficients: \n', linear_regressor.coef_)



# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))



# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % linear_regressor.score(X_test, y_test))



# Cohen’s kappa score: 1 is complete agreement

print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))

# Training a Ridge Regression model (l2 regularization) using all the features (BOW + other features)



alphas = np.array([3, 1, 0.3, 0.1])



ridge_regressor = Ridge()



grid = GridSearchCV(estimator = ridge_regressor, param_grid = dict(alpha=alphas))

grid.fit(X_train, y_train)



y_pred = grid.predict(X_test)



# summarize the results of the grid search

print(grid.best_score_)

print(grid.best_estimator_.alpha)



# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))



# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % grid.score(X_test, y_test))



# Cohen’s kappa score: 1 is complete agreement

print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))
# Training a Lasso Regression model (l1 regularization) using all the features (BOW + other features)



alphas = np.array([3, 1, 0.3, 0.1])



lasso_regressor = Lasso()



grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas))

grid.fit(X_train, y_train)



y_pred = grid.predict(X_test)



# summarize the results of the grid search

print(grid.best_score_)

print(grid.best_estimator_.alpha)



# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))



# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % grid.score(X_test, y_test))



# Cohen’s kappa score: 1 is complete agreement

print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))
# Training a Gradient Boosting Regression model using all the features (BOW + other features)



params = {'n_estimators':[100, 1000], 'max_depth':[2], 'min_samples_split': [2],

          'learning_rate':[3, 1, 0.1, 0.3], 'loss': ['ls']}



gbr = ensemble.GradientBoostingRegressor()



grid = GridSearchCV(gbr, params)

grid.fit(X_train, y_train)



y_pred = grid.predict(X_test)



# summarize the results of the grid search

print(grid.best_score_)

print(grid.best_estimator_)



mse = mean_squared_error(y_test, y_pred)

print("MSE: %.4f" % mse)



# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % grid.score(X_test, y_test))



# Cohen’s kappa score: 1 is complete agreement

print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))



# Plot feature importance - to find the main factors affecting the final grade

feature_importance = grid.best_estimator_.feature_importances_



# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

feature_names = list(features_set1.iloc[:, 3:].columns.values)

feature_names = np.asarray(feature_names + feature_names_cv)

sorted_idx = np.argsort(feature_importance)

# get top 100

sorted_idx = sorted_idx[9910:]

pos = np.arange(sorted_idx.shape[0]) + .5

plt.subplot(1, 2, 2)

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, feature_names[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
# splitting data (only 10 numerical/POS/orthographic features) into train data and test data (70/30)

    

X = features_set1.iloc[:, 3:].as_matrix()



y = features_set1['domain1_score'].as_matrix()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
# Training a Linear Regression model using only 10 numerical/POS/orthographic features



linear_regressor = LinearRegression()



linear_regressor.fit(X_train, y_train)



y_pred = linear_regressor.predict(X_test)



# The coefficients

print('Coefficients: \n', linear_regressor.coef_)



# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))



# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % linear_regressor.score(X_test, y_test))



# Cohen’s kappa score: 1 is complete agreement

print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))
# Training a Ridge Regression model (l2 regularization) using only 10 numerical/POS/orthographic features



alphas = np.array([3, 1, 0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001])



ridge_regressor = Ridge()



grid = GridSearchCV(estimator = ridge_regressor, param_grid = dict(alpha=alphas))

grid.fit(X_train, y_train)



y_pred = grid.predict(X_test)



# summarize the results of the grid search

print(grid.best_score_)

print(grid.best_estimator_.alpha)



# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))



# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % grid.score(X_test, y_test))



# Cohen’s kappa score: 1 is complete agreement

print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))
# Training a Lasso Regression model (l1 regularization) using only 10 numerical/POS/orthographic features



alphas = np.array([3, 1, 0.3, 0.1, 0.3])



lasso_regressor = Lasso()



grid = GridSearchCV(estimator = lasso_regressor, param_grid = dict(alpha=alphas))

grid.fit(X_train, y_train)



y_pred = grid.predict(X_test)



# summarize the results of the grid search

print(grid.best_score_)

print(grid.best_estimator_.alpha)



# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))



# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % grid.score(X_test, y_test))



# Cohen’s kappa score: 1 is complete agreement

print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))
# Training a Gradient Boosting Regression model using only 10 numerical/POS/orthographic features



params = {'n_estimators':[50, 100, 500, 1000], 'max_depth':[2], 'min_samples_split': [2],

          'learning_rate':[1, 0.1, 0.3, 0.01], 'loss': ['ls']}



gbr = ensemble.GradientBoostingRegressor()



grid = GridSearchCV(gbr, params)

grid.fit(X_train, y_train)



y_pred = grid.predict(X_test)



# summarize the results of the grid search

print(grid.best_score_)

print(grid.best_estimator_)



mse = mean_squared_error(y_test, y_pred)

print("MSE: %.4f" % mse)



# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % grid.score(X_test, y_test))

print()



# Cohen’s kappa score: 1 is complete agreement

print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))



# Plot feature importance - to find the main factors affecting the final grade

feature_importance = grid.best_estimator_.feature_importances_



# make importances relative to max importance

feature_importance = 100.0 * (feature_importance / feature_importance.max())

feature_names = list(features_set1.iloc[:, 3:].columns.values)

feature_names = np.asarray(feature_names)

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

plt.subplot(1, 2, 2)

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, feature_names[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
# Training a Support Vector Regression model using only 10 numerical/POS/orthographic features



svr = SVR()



parameters = {'kernel':['linear', 'rbf'], 'C':[1, 100], 'gamma':[0.1, 0.001]}



grid = GridSearchCV(svr, parameters)

grid.fit(X_train, y_train)



y_pred = grid.predict(X_test)



# summarize the results of the grid search

print(grid.best_score_)

print(grid.best_estimator_)



# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % grid.score(X_test, y_test))



# Cohen’s kappa score: 1 is complete agreement

print('Cohen\'s kappa score: %.2f' % cohen_kappa_score(np.rint(y_pred), y_test))