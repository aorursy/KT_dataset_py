import pandas as pd
import numpy as np
# Read the labeled training and test data
# Header = 0 indicates that the first line of the file contains column names, 
# delimiter = \t indicates that the fields are seperated by tabs, and 
# quoting = 3 tells python to ignore doubled quotes

train = pd.read_csv("../input/labeledTrainData.tsv", header = 0, delimiter = "\t", quoting = 3)
test = pd.read_csv("../input/testData.tsv", header = 0, delimiter = "\t", quoting = 3)
# Display check the dimensions and the first 2 rows of the file.

print('train dim:', train.shape, 'test dim:', test.shape)
train.iloc[0:2]
# Let's check the first review.

train.iloc[0]["review"][:len(train.iloc[0]["review"])//2]
from bs4 import BeautifulSoup
example1 = BeautifulSoup(train["review"][0], "html.parser")

# Without the second argument "html.parser", it will pop out the warning message.
print(example1.get_text())
import re

letters = re.sub("[^a-zA-Z]", " ", example1.get_text())
letters = letters.lower()
print(letters)
# Import Natural Language Toolkit
import nltk
# Instead of using just split() method, used word_tokenize in nltk library.
word = nltk.word_tokenize(letters)
word
from nltk.corpus import stopwords
print(stopwords.words("english"))
# Exclude the stop words from the original tokens.

word = [w for w in word if not w in set(stopwords.words("english"))]
word
snow = nltk.stem.SnowballStemmer('english')
stems = [snow.stem(w) for w in word]
stems
def cleaning(raw_review):
    import nltk
    
    # 1. Remove HTML.
    html_text = BeautifulSoup(raw_review,"html.parser").get_text()
    
    # 2. Remove non-letters.
    letters = re.sub("[^a-zA-Z]", " ", html_text)
    
    # 3. Convert to lower case.
    letters = letters.lower()
    
    # 4. Tokenize.
    tokens = nltk.word_tokenize(letters)
    
    # 5. Convert the stopwords list to "set" data type.
    stops = set(nltk.corpus.stopwords.words("english"))
    
    # 6. Remove stop words. 
    words = [w for w in tokens if not w in stops]
    
    # 7. Stemming
    words = [nltk.stem.SnowballStemmer('english').stem(w) for w in words]
    
    # 8. Join the words back into one string separated by space, and return the result.
    return " ".join(words)

    
# Add the processed data to the original data. Perhaps using apply function would be more elegant and concise than using for loop
train['clean'] = train['review'].apply(cleaning)
test['clean'] = test['review'].apply(cleaning)
train.head()
test.head()
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
    
cloud(' '.join(train['clean']))
cloud(' '.join(test['clean']))
# We need to split each words in cleaned review and then count the number of each rows of data frame.

train['freq_word'] = train['clean'].apply(lambda x: len(str(x).split()))
train['unique_freq_word'] = train['clean'].apply(lambda x: len(set(str(x).split())))
                                                 
test['freq_word'] = test['clean'].apply(lambda x: len(str(x).split()))
test['unique_freq_word'] = test['clean'].apply(lambda x: len(set(str(x).split())))                                                 
fig, axes = plt.subplots(ncols=2)
fig.set_size_inches(10,5)

sns.distplot(train['freq_word'], bins = 90, ax=axes[0], fit = stats.norm)
(mu0, sigma0) = stats.norm.fit(train['freq_word'])
axes[0].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu0, sigma0)],loc='best')
axes[0].set_title("Distribution Word Frequency")
axes[0].axvline(train['freq_word'].median(), linestyle='dashed')
print("median of word frequency: ", train['freq_word'].median())


sns.distplot(train['unique_freq_word'], bins = 90, ax=axes[1], color = 'r', fit = stats.norm)
(mu1, sigma1) = stats.norm.fit(train['unique_freq_word'])
axes[1].set_title("Distribution Unique Word Frequency")
axes[1].legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu1, sigma1)],loc='best')
axes[1].axvline(train['unique_freq_word'].median(), linestyle='dashed')
print("median of uniuqe word frequency: ", train['unique_freq_word'].median())
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word", 
                             tokenizer = None, 
                             preprocessor = None, 
                             stop_words = None, 
                             max_features = 18000,
                             min_df = 2,
                             ngram_range = (1,3)
                            )
from sklearn.pipeline import Pipeline
pipe = Pipeline( [('vect', vectorizer)] )
# Complete form of bag of word for machine learning input. We will be using this for machine learning algorithms.

train_bw = pipe.fit_transform(train['clean'])

# We only call transform not fit_transform due to the risk of overfitting.

test_bw = pipe.transform(test['clean'])
print('train dim:', train_bw.shape, 'test dim:', test_bw.shape)
# Get the name fo the features

lexi = vectorizer.get_feature_names()
lexi[:5]
# Instead of 1 and 0 representation, create the dataframe to see how many times each word appears (just sum of 1 of each row)

train_sum = pd.DataFrame(np.sum(train_bw, axis=0), columns = lexi)
train_sum.head()
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
    'loss':['hinge'],
    'class_weight':[{1:1}],
    'C': [0.01]
}

gs_sv = GridSearchCV(sv, param_grid = [param_grid2], verbose = 1, cv = kfold, n_jobs = -1, scoring = 'roc_auc' )
gs_sv.fit(train_bw, train['sentiment'])
gs_sv_best = gs_sv.best_estimator_
print(gs_sv.best_params_)

# {'C': 0.01, 'class_weight': {1: 1}, 'loss': 'hinge'} - 0.88104
submission1 = gs_sv.predict(test_bw)
print(gs_sv.best_score_)
bnb = BernoulliNB()
gs_bnb = GridSearchCV(bnb, param_grid = {'alpha': [0.03],
                                         'binarize': [0.001]}, verbose = 1, cv = kfold, n_jobs = -1, scoring = 'roc_auc')
gs_bnb.fit(train_bw, train['sentiment'])
gs_bnb_best = gs_bnb.best_estimator_
print(gs_bnb.best_params_)

# {'alpha': 0.1, 'binarize': 0.001} - 0.85240
# {'alpha': 0.03, 'binarize': 0.001} - 0.85240
submission2 = gs_bnb.predict(test_bw)
print(gs_bnb.best_score_)
lr = LogisticRegression(random_state = 2018)


lr2_param = {
    'penalty':['l2'],
    'dual':[False],
    'C':[0.05],
    'class_weight':['balanced']
    }

lr_CV = GridSearchCV(lr, param_grid = [lr2_param], cv = kfold, scoring = 'roc_auc', n_jobs = -1, verbose = 1)
lr_CV.fit(train_bw, train['sentiment'])
print(lr_CV.best_params_)
logi_best = lr_CV.best_estimator_


# {'C': 0.1, 'class_weight': 'balanced', 'dual': False, 'penalty': 'l2'} - 0.87868
# {'C': 0.05, 'class_weight': 'balanced', 'dual': False, 'penalty': 'l2'} - 0.88028
submission4 = lr_CV.predict(test_bw)
print(lr_CV.best_score_)