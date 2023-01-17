# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python





# The following packages were used:

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import string



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score, make_scorer, confusion_matrix, classification_report

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler



import re

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

import category_encoders as ce



!pip install pyspellchecker

from spellchecker import SpellChecker

from sklearn import decomposition







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train_data.head(50)
# Shuffeling the training data

train_data = train_data.sample(frac=1, random_state=1)

train_data.head(25)
# some basic data explorations showing the amount of samples in the data sets,

# unique keywords and locations and number of missing values in keywords and locations 

print("train data (cols, rows):\n", train_data.shape)

print("test data (cols, rows):\n", test_data.shape)

print("Unique keywords in train data:\n", len(pd.unique(train_data['keyword'])))  

print("Unique locations in train data:\n", len(pd.unique(train_data['location']))) 

print("Missing keywords in train data:\n", train_data['keyword'].isnull().sum())

print("Missing locations in train data:\n", train_data['location'].isnull().sum())
# plotting the distribution between the number of disaster and non disaster tweets.

labels = train_data.target.value_counts()



# Creating a nice graph for visualiziation

sns.barplot(labels.index, labels)

plt.gca().set_title('Number of Tweets per category')

plt.gca().set_ylabel('samples')

plt.gca().set_xticklabels(['0: No disaster', '1: Disaster'])
# The URL's like youtube links do not provide valuable info so are removed

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



# HTML code can also be removed.

def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



# emoji are not written in plain text and are thus removed

def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



# combines the above 3 functions

def clean_text(text):

    text = remove_URL(text)

    text = remove_html(text)

    text = remove_emoji(text)

    return text



# Tweets are written fast with lots of spelling errors.

# the function belows does some basic spell checking.

# Hopefully we can now categorize misspelled and correctly spelled words together. 

# Spellchecking slightly improves model performance

spell = SpellChecker(distance=1) # distance=2 is standard but very slow

def correct_spelling(text):

    corrected_text = []

    misspelled_words = spell.unknown(text.split())

    for word in text.split():

        if word in misspelled_words:

            corrected_text.append(spell.correction(word))

        else:

            corrected_text.append(word)

    return " ".join(corrected_text)





# The following function applies the clean text and correct spelling functions

# furthermore featuers are created like the number of punctuation and word count.

def feature_engineer(dataframe):

    # apply the clean text function

    dataframe['text'] = dataframe.text.apply(lambda x: clean_text(x))

    num_char = dataframe.text.apply(lambda x: len(x))

    num_space = dataframe.text.apply(lambda x: x.count(' '))

    

    # 4 extra features are created that may be valluable for the model

    dataframe['num_punc'] = dataframe.text.apply(lambda x: len([c for c in x if c in string.punctuation]))

    dataframe['num_upper'] = dataframe.text.apply(lambda x: len([letter for letter in x if letter.isupper()]))

    dataframe['mean_word_length'] = dataframe['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

    dataframe['word_count'] = dataframe['text'].apply(lambda x: len(str(x).split()))

    

    # Punctuation is removed and the spelling corrector is used

    dataframe['text'] = dataframe.text.apply(lambda x: "".join([c for c in x if c not in string.punctuation]))

    dataframe['text'] = dataframe.text.apply(lambda x: correct_spelling(x))

    dataframe = dataframe.drop(columns=['id'])

    return dataframe  # return a dataframe with clean text and new features

    

# applying the Feature engineer function above to both the train and test data set.

train = feature_engineer(train_data)

test = feature_engineer(test_data)



# Here we split the train data in an X and Y (target) variable.

y = train['target']

x = train.drop(columns=['target'])



# showing the new cleaned train dataframe.

# Upper case letters are tranformed to lower case in a later stage

train.head(10)
f, axes = plt.subplots(2,2, figsize=(10,10))

sns.distplot(train.num_upper[train.target == 0], label='Not Disaster', color='green', ax=axes[0][0])

sns.distplot(train.num_upper[train.target == 1], label='Disaster', color='red', ax=axes[0][0])

axes[0][0].legend()

axes[0][0].set_title('Distribution of upper case letters')

axes[0][0].set_xlabel('Number of upper case letters')



sns.distplot(train.num_punc[train.target == 0], label='Not Disaster', color='green', ax=axes[0][1])

sns.distplot(train.num_punc[train.target == 1], label='Disaster', color='red', ax=axes[0][1])

axes[0][1].legend()

axes[0][1].set_title('Distribution of punctuation marks')

axes[0][1].set_xlabel('Number of punctuation marks')



sns.distplot(train.word_count[train.target == 0], label='Not Disaster', color='green', ax=axes[1][0])

sns.distplot(train.word_count[train.target == 1], label='Disaster', color='red', ax=axes[1][0])

axes[1][0].legend()

axes[1][0].set_title('Distribution of words per tweet')

axes[1][0].set_xlabel('Number of words per tweet')



sns.distplot(train.mean_word_length[train.target == 0], label='Not Disaster', color='green', ax=axes[1][1])

sns.distplot(train.mean_word_length[train.target == 1], label='Disaster', color='red', ax=axes[1][1])

axes[1][1].legend()

axes[1][1].set_title('Distribution of word lenght per tweet')

axes[1][1].set_xlabel('Average word lenght per tweet')



plt.tight_layout()

plt.show()
# stop wrods are used a lot but contain little information

stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):

    nostop = [word for word in tokens if word not in stop_words]

    return nostop



# stemming and lemmatizer function

wnl = WordNetLemmatizer()

def lemmatizer(tokens):

    lem_tokens = [wnl.lemmatize(word) for word in tokens]

    return lem_tokens



ps = PorterStemmer()

def stemmer(tokens):

    stem_tokens = [ps.stem(word) for word in tokens]

    return stem_tokens





# This analyzer function is used to process the text in the model piplein

def analyzer(text):

    tokens = word_tokenize(text.lower())  # Creates a list of all word in a tweet in lower case

    tokens = remove_stopwords(tokens)     # removes the stop words

    

    ## The lematizer OR the stemmer is used (not both): Stemming provides better predictions

    #     tokens = lemmatizer(tokens)

    tokens = stemmer(tokens)

    text =  ' '.join([w for w in tokens])

    return text

## TRANSFORMERS



# Preprocessing for keyword (and Location) data

# Location data does not improve the modle

keyword_transformer = ce.TargetEncoder()



# PREPROCESSING FOR TEXT DATA

# A vectorizer is used to change the words in numbers (like one hot encoding)

# TFIDF is more sophisticated. However the more simple count vectorizer provides better results

# min_df=2 removes all words that are only used once in all tweets (slight improvement in model performance) 

# Ngrams did not improve model performance

text_transformer = CountVectorizer(preprocessor=analyzer, min_df = 2)

# text_transformer = TfidfVectorizer(analyzer=analyzer, min_df = 2, ngram_range = (1,2))





# Numeric collumns are scaled with standardscalar

# Number of upper case letters and punctuation marks did not improve the model and are thus not used

numeric_cols = ['word_count', 'mean_word_length']

numeric_transformer = StandardScaler()





# Bundle preprocessing for numerical, text and keyword data

preprocessor = ColumnTransformer(

    transformers=[

        ('key', keyword_transformer, 'keyword'),   # Location does not provide an improvement

        ('tex', text_transformer, 'text'),

        ('num', numeric_transformer, numeric_cols)],

        remainder = 'drop')





## MODEL SELECTION

# 6 different types of models were tried (manually). (see the from .... import .... list below)

# A basic LogisticRegression classification model provided the best results and is fast to run

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier



# I manually changed the model type here

# The model below was optimized a bit, in a later stage of this notebook

model = LogisticRegression(C = 0.3, penalty= 'l2', random_state = 1, solver='lbfgs')



## FEATURE SELECTION

# The count vectorizer creates a new feature for each unique word.

# In total this is more than 5000 features.

# I tried to select the K best features with a variety of metrics but this did not improve the model performance

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif, chi2, mutual_info_classif

# selector = SelectKBest(f_classif, k=300)





# The LogisticRegression (LR) pipeline

LR_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

#                             ('selector', selector),  # The feature selection (selectKbest) is not used

                              ('model', model)

                             ])



# 5 fold Cross validation to evaluate model performance

scores = cross_val_score(LR_pipeline, x, y,

                         cv=5,

                         scoring='f1',  # evaluation metric of the competition

                         n_jobs=-1)

# Printing the f1 score results

print(scores)

print(np.mean(scores))

# calculating cross validation predictions

LR_pred = cross_val_predict(LR_pipeline, x, y,

                            cv=5,

                            n_jobs=-1)



# calculating the confusion matrix

conf = confusion_matrix(y, LR_pred, labels=[1, 0])



# Making a better readable and flashy looking confusion matrix

ax= plt.subplot()

sns.heatmap(conf, annot=True, ax = ax)

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')

ax.set_title('Confusion Matrix')

ax.xaxis.set_ticklabels(['Disaster', 'No disaster']); ax.yaxis.set_ticklabels(['Disaster', 'No disaster'])



# showing the classification report

print(classification_report(y, LR_pred))
# Run the data preprocessing of the Pipeline

PCA_fit = LR_pipeline.fit(x,y)

processed_data = PCA_fit.named_steps['preprocessor'].transform(x)



# Calculate the Principal components and select the 2 most important ones

pca2 = decomposition.PCA(n_components=2)

X_pca2 = pca2.fit_transform(processed_data.toarray())

X_pca2 = pd.DataFrame(X_pca2, columns=['pca1', 'pca2']) #create small df for seaborn

X_pca2['Disaster?'] = y #add the target classes for plotting



sns.scatterplot(x=X_pca2['pca1'], y=X_pca2['pca2'], hue=X_pca2['Disaster?'])

plt.title('First two principle components')



print("Percentage of variance explained by first 2 PCA components:\n", pca2.explained_variance_ratio_)

# SimpleImputer hasn no get_feature_names function. These features are removed from this pipeline

# I am more interested in the most importantd words in the text anyway.

importance_preprocessor = ColumnTransformer(transformers=[

                                            ('key', keyword_transformer, 'keyword'),

                                            ('tex', text_transformer, 'text')],

#                                           ('num', numeric_transformer, numeric_cols)],

                                            remainder = 'drop')



# Create a pipeline with the new Columntranformer

importance_pipeline = Pipeline(steps=[('preprocessor', importance_preprocessor),

                                      ('model', model)

                                     ])



# Fit the new pipeline

importance_pipeline.fit(x, y)    



# Calculate a logistic regression coefficient and the corresponding feature names

feature_importance = importance_pipeline.named_steps['model'].coef_[0]

feature_names = importance_pipeline.named_steps['preprocessor'].get_feature_names()



# Place the feature names and coefficients in a dataframe to sort the values in ascending order

# Show the top 25 coefficient in a table

df = pd.DataFrame({'feature_names': feature_names, 'feature_coefficient': feature_importance})

df = df.sort_values(by='feature_coefficient', ascending=False)

df.head(25)
# Create hyperparameter options for regualrization strength C

# hyperparameters = {'model__C': [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]} # Ran this first, result was around 0.5

hyperparameters = {'model__C': [0.2, 0.3, 0.4, 0.5, 0.6]} # More finetuning around 0.5



# Create grid search using 5-fold cross validation

logistic_gs = GridSearchCV(LR_pipeline, hyperparameters, cv=5, scoring = make_scorer(f1_score), verbose=0, n_jobs=-1)



# Fitting the results

LR_best_model = logistic_gs.fit(x, y)



# Printing the accuracy for each fold

# Printing the average model accuracy over all 3 folds

print('Best score and parameter combination = ')

print(LR_best_model.best_score_)    

print(LR_best_model.best_params_) 





###################################################

# Best score and parameter combination = 

# 0.7604063775130128

# {'model__C': 0.6, 'model__penalty': 'l2'}
# LR_pipeline_fit = LR_pipeline.fit(x,y)

test_pred = LR_best_model.predict(test)





sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")

sample_submission["target"] = test_pred

sample_submission.to_csv("submission.csv", index=False)

sample_submission.head(25)