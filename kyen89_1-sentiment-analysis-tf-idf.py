
# Import libraries

import pandas as pd
import numpy as np
# Read the data 

X_train = pd.read_csv("../input/labeledTrainData.tsv",quoting = 3, delimiter = "\t", header= 0)
X_test = pd.read_csv("../input/testData.tsv", quoting = 3, delimiter = "\t", header = 0)
# Read only the first 600 sentences of the first review.

X_train['review'][0][:600]
print('Training set dimension:',X_train.shape)
print('Test set dimension:',X_test.shape)
X_train.head()

from bs4 import BeautifulSoup
import re
import nltk
def prep(review):
    
    # Remove HTML tags.
    review = BeautifulSoup(review,'html.parser').get_text()
    
    # Remove non-letters
    review = re.sub("[^a-zA-Z]", " ", review)
    
    # Lower case
    review = review.lower()
    
    # Tokenize to each word.
    token = nltk.word_tokenize(review)
    
    # Stemming
    review = [nltk.stem.SnowballStemmer('english').stem(w) for w in token]
    
    # Join the words back into one string separated by space, and return the result.
    return " ".join(review)
    
# test whether the function successfully preprocessed.
X_train['review'].iloc[:2].apply(prep).iloc[0]

# If there is no problem at the previous cell, let's apply to all the rows.
X_train['clean'] = X_train['review'].apply(prep)
X_test['clean'] = X_test['review'].apply(prep)
X_train['clean'].iloc[3]
print('Training dim:',X_train.shape, 'Test dim:', X_test.shape)

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import words
# analyzer is the parameter that the vectorizer reads the input data in word unit or character unit to create a matrix
# vocabulary is the parameter that the vectorizer creates the matrix by using only input data or some other source 
# Other parameters are self-explanatory and already mentioned in other notebooks.

tv = TfidfVectorizer(
                    ngram_range = (1,3),
                    sublinear_tf = True,
                    max_features = 40000)
# Handle with care especially when you transform the test dataset. (Wrong: fit_transform(X_test))

train_tv = tv.fit_transform(X_train['clean'])
test_tv = tv.transform(X_test['clean'])
# Create the list of vocabulary used for the vectorizer.

vocab = tv.get_feature_names()
print(vocab[:5])
print("Vocabulary length:", len(vocab))
dist = np.sum(train_tv, axis=0)
checking = pd.DataFrame(dist,columns = vocab)
checking
print('Training dim:',train_tv.shape, 'Test dim:', test_tv.shape)

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

%matplotlib inline
def cloud(data,backgroundcolor = 'white', width = 800, height = 600):
    wordcloud = WordCloud(stopwords = STOPWORDS, background_color = backgroundcolor,
                         width = width, height = height).generate(data)
    plt.figure(figsize = (15, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    
cloud(' '.join(X_train['clean']))
cloud(' '.join(X_test['clean']))

# We need to split each words in cleaned review and then count the number of each rows of data frame.

X_train['freq_word'] = X_train['clean'].apply(lambda x: len(str(x).split()))
X_train['unique_freq_word'] = X_train['clean'].apply(lambda x: len(set(str(x).split())))
                                                 
X_test['freq_word'] = X_test['clean'].apply(lambda x: len(str(x).split()))
X_test['unique_freq_word'] = X_test['clean'].apply(lambda x: len(set(str(x).split())))                                                 
fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(10,5)

sns.distplot(X_train['freq_word'], bins = 90, ax=axes[0], fit = stats.norm)
(mu0, sigma0) = stats.norm.fit(X_train['freq_word'])
axes[0].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu0, sigma0)],loc='best')
axes[0].set_title("Distribution Word Frequency")
axes[0].axvline(X_train['freq_word'].median(), linestyle='dashed')
print("median of word frequency: ", X_train['freq_word'].median())


sns.distplot(X_train['unique_freq_word'], bins = 90, ax=axes[1], color = 'r', fit = stats.norm)
(mu1, sigma1) = stats.norm.fit(X_train['unique_freq_word'])
axes[1].set_title("Distribution Unique Word Frequency")
axes[1].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1)],loc='best')
axes[1].axvline(X_train['unique_freq_word'].median(), linestyle='dashed')
print("median of uniuqe word frequency: ", X_train['unique_freq_word'].median())

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
kfold = StratifiedKFold( n_splits = 5, random_state = 2018 )

# LinearSVC

sv = LinearSVC(random_state=2018)

param_grid2 = {
    'loss':['squared_hinge'],
    'class_weight':[{1:4}],
    'C': [0.2]
}


gs_sv = GridSearchCV(sv, param_grid = [param_grid2], verbose = 1, cv = kfold, n_jobs = 1, scoring = 'roc_auc')
gs_sv.fit(train_tv, X_train['sentiment'])
gs_sv_best = gs_sv.best_estimator_
print(gs_sv.best_params_)

# {'C': 0.1, 'class_weight': {1: 3}, 'loss': 'squared_hinge'} - 0.87220
# {'C': 0.1, 'class_weight': {1: 4}, 'loss': 'squared_hinge'} - 0.86060
# {'C': 0.2, 'class_weight': {1: 4}, 'loss': 'squared_hinge'} - 0.87952
submission1 = gs_sv.predict(test_tv)
print(gs_sv.best_score_)

bnb = BernoulliNB()
gs_bnb = GridSearchCV(bnb, param_grid = {'alpha': [0.001],
                                         'binarize': [0.001]}, verbose = 1, cv = kfold, n_jobs = 1, scoring = "roc_auc")
gs_bnb.fit(train_tv, X_train['sentiment'])
gs_bnb_best = gs_bnb.best_estimator_
print(gs_bnb.best_params_)

# {'alpha': 0.001, 'binarize': 0.001} - 0.86960

submission2 = gs_bnb.predict(test_tv)
print(gs_bnb.best_score_)

MLP = MLPClassifier(random_state = 2018)

mlp_param_grid = {
    'hidden_layer_sizes':[(5)],
    'activation':['relu'],
    'solver':['adam'],
    'alpha':[0.3],
    'learning_rate':['constant'],
    'max_iter':[1000]
}


gsMLP = GridSearchCV(MLP, param_grid = mlp_param_grid, cv = kfold, scoring = 'roc_auc', n_jobs= 1, verbose = 1)
gsMLP.fit(train_tv,X_train['sentiment'])
print(gsMLP.best_params_)
mlp_best0 = gsMLP.best_estimator_

# {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': (1,), 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'adam'} - 0.89996
# {'activation': 'relu', 'alpha': 0.1, 'hidden_layer_sizes': (5,), 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'adam'} - 0.89896
# {'activation': 'relu', 'alpha': 0.2, 'hidden_layer_sizes': (1,), 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'adam'} - 0.90284
# {'activation': 'relu', 'alpha': 0.3, 'hidden_layer_sizes': (5,), 'learning_rate': 'constant', 'max_iter': 1000, 'solver': 'adam'} - 0.90356
submission3 = gsMLP.predict(test_tv)
print(gsMLP.best_score_)

lr = LogisticRegression(random_state = 2018)

lr2_param = {
    'penalty':['l2'],
    'dual':[True],
    'C':[6],
    'class_weight':[{1:1}]
    }

lr_CV = GridSearchCV(lr, param_grid = [lr2_param], cv = kfold, scoring = 'roc_auc', n_jobs = 1, verbose = 1)
lr_CV.fit(train_tv, X_train['sentiment'])
print(lr_CV.best_params_)
logi_best = lr_CV.best_estimator_

# {'C': 6, 'class_weight': {1: 1}, 'dual': True, 'penalty': 'l2'} - 90.360
submission6 = lr_CV.predict(test_tv)
print(lr_CV.best_score_)

# Extract the coefficients from the best model Logistic Regression and sort them by index.
coefficients = logi_best.coef_
index = coefficients.argsort()
# Extract the feature names.
feature_names = np.array(tv.get_feature_names())
# From the smallest to largest.
feature_names[index][0][:30]
# From the smallest to largest.
feature_names[index][0][-31::1]
# feature names: Smallest 30 + largest 30.
feature_names_comb = list(feature_names[index][0][:30]) + list(feature_names[index][0][-31::1])
# coefficients magnitude: Smallest 30 + largest 30.
index_comb = list(coefficients[0][index[0][:30]]) + list(coefficients[0][index[0][-31::1]])
# Make sure the x-axis be the number from 0 to the length of the features selected not the feature names.
# Once the bar is plotted, the features are placed as ticks.
plt.figure(figsize=(25,10))
barlist = plt.bar(list(i for i in range(61)), index_comb)
plt.xticks(list(i for i in range(61)),feature_names_comb,rotation=75,size=15)
plt.ylabel('Coefficient magnitude',size=20)
plt.xlabel('Features',size=20)

# color the first smallest 30 bars red
for i in range(30):
    barlist[i].set_color('red')

plt.show()

output = pd.DataFrame( data = {'id': X_test['id'], 'sentiment': submission6 })
output.to_csv('submission26.csv', index = False, quoting = 3)
