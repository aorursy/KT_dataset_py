import numpy as np

import matplotlib as plt

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



from sklearn.naive_bayes import MultinomialNB

from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import cross_val_score

from sklearn.pipeline import make_pipeline
South_Park_raw = pd.read_csv('../input/All-seasons.csv')

South_Park_raw.describe()
# Head and shape of dataset

print(South_Park_raw.head())

print(South_Park_raw.shape)
print (South_Park_raw.describe())
#Select just speakers with more than 500 lines



top_speakers = South_Park_raw.groupby(['Character']).size().loc[South_Park_raw.groupby(['Character']).size() > 500]

print (top_speakers.sort_values(ascending=False))



#Select rows top speakers   

""" This is the dataset we will be working with"""



main_char_lines = pd.DataFrame(South_Park_raw.loc[South_Park_raw['Character'].isin(top_speakers.index.values)])

del main_char_lines['Season']

del main_char_lines['Episode']



main_char_lines = main_char_lines.reset_index(drop=True)



print (main_char_lines.describe())
# define X and y

X = main_char_lines.Line

y = main_char_lines.Character



#print (y.value_counts(normalize=True))



# split the new DataFrame into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
#pipe = make_pipeline(TfidfVectorizer(), MultinomialNB())

#pipe.steps



#param_grid = {}

#param_grid["tfidfvectorizer__max_features"] = [500, 1000, 15000]

#param_grid["tfidfvectorizer__ngram_range"] = [(1,1), (1,2), (2,2)]

#param_grid["tfidfvectorizer__lowercase"] = [True, False]

#param_grid["tfidfvectorizer__stop_words"] = ["english", None]

#param_grid["tfidfvectorizer__strip_accents"] = ["ascii", "unicode", None]

#param_grid["tfidfvectorizer__analyzer"] = ["word", "char"]

#param_grid["tfidfvectorizer__binary"] = [True, False]

#param_grid["tfidfvectorizer__norm"] = ["l1", "l2", None]

#param_grid["tfidfvectorizer__use_idf"] = [True, False]

#param_grid["tfidfvectorizer__smooth_idf"] = [True, False]

#param_grid["tfidfvectorizer__sublinear_tf"] = [True, False]



#grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')



#Helpful for understanding how to create your param grid.

#grid.get_params().keys()
#grid.fit(X,y)
#print(grid.best_params_)

#print(grid.best_score_)
vect = TfidfVectorizer(analyzer='word', stop_words='english', max_features = 850, ngram_range=(1, 1), 

                       binary=False, lowercase=True, norm=None, smooth_idf=True, strip_accents=None,

                       sublinear_tf=True, use_idf=False)



mcl_transformed = vect.fit_transform(X)



nb_SP_Model = MultinomialNB()

nb_SP_Model.fit(mcl_transformed, y)

print ("Model accuracy within dataset: ", nb_SP_Model.score(mcl_transformed, y))
print ("Model accuracy with cross validation:", cross_val_score(MultinomialNB(), mcl_transformed.toarray(), 

                                                                y, cv=5, scoring="accuracy").mean())
# Predict on new text

new_text = ["Well, I guess we'll have to roshambo for it. I'll kick you in the nuts as hard as I can, then you kick me square in the nuts as hard as you can..."]

new_text_transform = vect.transform(new_text)



print (nb_SP_Model.predict(new_text_transform)," most likely said it.")
SP_prob=pd.DataFrame(nb_SP_Model.predict_proba(new_text_transform))

SP_prob=pd.DataFrame.transpose(SP_prob)

SP_prob.columns = ['Likelihood']



top_speakers_index = top_speakers.reset_index()

top_speakers_index.columns = ['Character', 'Lines']

top_speakers_index = top_speakers_index.drop('Lines', 1)



Result = pd.concat([top_speakers_index, SP_prob], axis=1)



print (Result.sort_values('Likelihood',ascending=False))
cartman = pd.DataFrame(South_Park_raw.loc[South_Park_raw['Character'].isin(top_speakers.index.values)])

del cartman['Season']

del cartman['Episode']



cartman.Character[cartman.Character != 'Cartman'] = 'Not Cartman'

cartman.Character[cartman.Character == 'Cartman'] = 'Cartman'

print (cartman)
cartman.Character.value_counts(normalize=True)
X_cartman = cartman.Line

y_cartman = cartman.Character

vect_cartman =CountVectorizer(stop_words='english')

Xdtm_cartman = vect_cartman.fit_transform(X_cartman)

nb_cartman = MultinomialNB()

nb_cartman.fit(Xdtm_cartman,y_cartman)

nb_cartman.score(Xdtm_cartman,y_cartman)
tokens_cartman = vect_cartman.get_feature_names()

len(tokens_cartman)
print (vect_cartman.get_feature_names()[:50])
nb_cartman.feature_count_
nb_cartman.feature_count_.shape
token_count_cartman= nb_cartman.feature_count_[0,:]

token_count_cartman
token_count_not_cartman = nb_cartman.feature_count_[1, :]

token_count_not_cartman
# create a DataFrame of tokens with their separate Not-Cartman and Cartman counts

cartman_tokens = pd.DataFrame({'token':tokens_cartman, 'Cartman':token_count_cartman, 'Not_Cartman':token_count_not_cartman}).set_index('token')

cartman_tokens.sample(10, random_state=3)
# add 1 to Cartmen and Not Cartman counts to avoid dividing by 0

cartman_tokens['Cartman'] = cartman_tokens.Cartman + 1

cartman_tokens['Not_Cartman'] = cartman_tokens.Not_Cartman + 1

cartman_tokens.sample(10, random_state=3)
# Naive Bayes counts the number of observations in each class

nb_cartman.class_count_
# convert the Cartman and Not Cartman counts into frequencies

cartman_tokens['Cartman'] = cartman_tokens.Cartman / nb_cartman.class_count_[0]

cartman_tokens['Not_Cartman'] = cartman_tokens.Not_Cartman / nb_cartman.class_count_[1]

cartman_tokens.sample(10, random_state=3)
# calculate the ratio of Cartman-to-Not_Cartman for each token

cartman_tokens['spam_ratio'] = cartman_tokens.Cartman / cartman_tokens.Not_Cartman

cartman_tokens.sample(10, random_state=3)
# examine the DataFrame sorted by spam_ratio

cartman_tokens.sort_values('spam_ratio', ascending=False).head(10)
#Try looking up scores of different words

word = "nyah"

cartman_tokens.loc[word, 'spam_ratio']
stan = pd.DataFrame(South_Park_raw.loc[South_Park_raw['Character'].isin(top_speakers.index.values)])

del stan['Season']

del stan['Episode']



stan.Character[stan.Character != 'Stan'] = 'Not Stan'

stan.Character[stan.Character == 'Stan'] = 'Stan'



X_stan = stan.Line

y_stan = stan.Character

vect_stan =CountVectorizer(stop_words='english')

Xdtm_stan = vect_stan.fit_transform(X_stan)

nb_stan = MultinomialNB()

nb_stan.fit(Xdtm_stan,y_stan)

nb_stan.score(Xdtm_stan,y_stan)



tokens_stan = vect_stan.get_feature_names()



token_count_stan= nb_stan.feature_count_[0,:]

token_count_not_stan = nb_stan.feature_count_[1, :]



stan_tokens = pd.DataFrame({'token':tokens_stan, 'Stan':token_count_stan, 'Not_Stan':token_count_not_stan}).set_index('token')



stan_tokens['Stan'] = stan_tokens.Stan + 1

stan_tokens['Not_Stan'] = stan_tokens.Not_Stan + 1



stan_tokens['Stan'] = stan_tokens.Stan / nb_stan.class_count_[0]

stan_tokens['Not_Stan'] = stan_tokens.Not_Stan / nb_stan.class_count_[1]



stan_tokens['spam_ratio'] = stan_tokens.Stan / stan_tokens.Not_Stan



# examine the DataFrame sorted by spam_ratio

stan_tokens.sort_values('spam_ratio', ascending=False).head(10)
kyle = pd.DataFrame(South_Park_raw.loc[South_Park_raw['Character'].isin(top_speakers.index.values)])

del kyle['Season']

del kyle['Episode']



kyle.Character[kyle.Character != 'Kyle'] = 'Not Kyle'

kyle.Character[kyle.Character == 'Kyle'] = 'Kyle'



X_kyle = kyle.Line

y_kyle = kyle.Character

vect_kyle =CountVectorizer(stop_words='english')

Xdtm_kyle = vect_kyle.fit_transform(X_kyle)

nb_kyle = MultinomialNB()

nb_kyle.fit(Xdtm_kyle,y_kyle)

nb_kyle.score(Xdtm_kyle,y_kyle)



tokens_kyle = vect_kyle.get_feature_names()



token_count_kyle= nb_kyle.feature_count_[0,:]

token_count_not_kyle = nb_kyle.feature_count_[1, :]



kyle_tokens = pd.DataFrame({'token':tokens_kyle, 'Kyle':token_count_kyle, 'Not_Kyle':token_count_not_kyle}).set_index('token')



kyle_tokens['Kyle'] = kyle_tokens.Kyle + 1

kyle_tokens['Not_Kyle'] = kyle_tokens.Not_Kyle + 1



kyle_tokens['Kyle'] = kyle_tokens.Kyle / nb_kyle.class_count_[0]

kyle_tokens['Not_Kyle'] = kyle_tokens.Not_Kyle / nb_kyle.class_count_[1]



kyle_tokens['spam_ratio'] = kyle_tokens.Kyle / kyle_tokens.Not_Kyle



# examine the DataFrame sorted by spam_ratio

kyle_tokens.sort_values('spam_ratio', ascending=False).head(10)