import re

import spacy

import numpy as np

import pandas as pd 

from textblob import TextBlob

#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer





from nltk.corpus import stopwords

from nltk.stem import PorterStemmer



from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer, TfidfTransformer

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import f1_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier



from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import PolynomialFeatures

polynomial_features= PolynomialFeatures(degree=2, interaction_only=True)



from sklearn.feature_selection import chi2, SelectKBest, f_classif

from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor



training_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
training_data.head(2)
def indicate_pattern(col, pattern):

    if re.search(pattern, col):

        return 1

    else:

        return 0

    

    

def make_indicator_col(df, input_col_index, pattern='', suffix='pattern'):

    df['contains_' + suffix] = df[input_col_index].apply(lambda x: indicate_pattern(x, pattern))

    

    

def make_indicators(df):

    column = 'text'

    pattern_list=['@[\w]*', '#[\w]*', r'http.?://[^\s]+[\s]?'] 

    suffix_list=['mention', 'hashtag', 'URL']

  

    # Append indicator columns to original DataFrame

    for pattern, suffix in zip(pattern_list, suffix_list):

        make_indicator_col(df, column, pattern, suffix)

    

    # Make list of new indicator columns

    new_columns = ['contains_' + suffix for suffix in suffix_list]

    

    # Output with length columns

    output = df[new_columns]

    

    # Drop new columns from original DataFrame

    df.drop(new_columns, axis=1, inplace=True)

    

    return df.join(output)



get_indicators = FunctionTransformer(make_indicators, validate = False)
text_column = 'text'

punc = [';'

        , "'"

        , '--'

        , ':'

        , '"'

        , "!"

        , "?"

        , '-'

        , ','

        , '.'

        , "("

        , ")"

        , '$'

        ,'`'

        ,'~'

        ,'/'

        ,'&'

        ,'%'

        ,'#'

        ,'@'

       ]

        

        

        

def features(dataframe):

    dataframe['word_count'] = dataframe[text_column].apply(lambda x : len(x.split()))

    dataframe['char_count'] = dataframe[text_column].apply(lambda x : len(x.replace(" ","")))

    dataframe['word_density'] = dataframe['word_count'] / (dataframe['char_count'] + 1)

    dataframe['punc_count'] = dataframe[text_column].apply(lambda x : len([a for a in x if a in punc]))

    dataframe['total_length'] = dataframe[text_column].apply(len)

    dataframe['capitals'] = dataframe[text_column].apply(lambda x: sum(1 for c in x if c.isupper()))

    dataframe['caps_vs_length'] = dataframe.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)

    dataframe['num_exclamation_marks'] =dataframe[text_column].apply(lambda x: x.count('!'))

    dataframe['num_question_marks'] = dataframe[text_column].apply(lambda x: x.count('?'))

    dataframe['num_punctuation'] = dataframe[text_column].apply(lambda x: sum(x.count(w) for w in '.,;:'))

    dataframe['num_symbols'] = dataframe[text_column].apply(lambda x: sum(x.count(w) for w in '*&$%'))

    dataframe['num_unique_words'] = dataframe[text_column].apply(lambda x: len(set(w for w in x.split())))

    dataframe['words_vs_unique'] = dataframe['num_unique_words'] / dataframe['word_count']

    dataframe["word_unique_percent"] =  dataframe["num_unique_words"]*100/dataframe['word_count']

    

    desc_blob = [TextBlob(desc) for desc in dataframe[text_column]]

    dataframe['tb_Pol'] = [b.sentiment.polarity for b in desc_blob]

    dataframe['tb_Subj'] = [b.sentiment.subjectivity for b in desc_blob]



    return dataframe

#Create Transformer for pipeline

get_word_count = FunctionTransformer(features, validate=False)
###Now a function called get_indicators





# training_data_ind = make_indicators(training_data, 'text', 

#                                     pattern_list=['@[\w]*', '#[\w]*', r'http.?://[^\s]+[\s]?'], 

#                                     suffix_list=['mention', 'hashtag', 'URL'])

# test_data_ind = make_indicators(training_data, 'text', 

#                                 pattern_list=['@[\w]*', '#[\w]*', r'http.?://[^\s]+[\s]?'], 

#                                 suffix_list=['mention', 'hashtag', 'URL'])
# # Handle duplicates in only the training set

# print('Before:', training_data.shape)

# training_data.drop_duplicates(inplace=True)

# print('After:', training_data.shape)
# Remove menntions, special characters, and extra whitespace from tweets



def clean_tweet(tweet):

    tweet = tweet.lower()

    # Remove mentions

    tweet = re.sub(r'@\w+', '', tweet)

    # Remove URLs

    tweet = re.sub(r'http.?://[^\s]+[\s]?', '', tweet)

    # Remove special characters except hash

    tweet = re.sub('[^a-zA-Z\s]', ' ', tweet)

    # Remove extra whitespace

    tweet = re.sub(" +", ' ', tweet)

    tweet = tweet.lstrip()

    tweet = tweet.rstrip()

    return tweet



def fill_missing_text(df, columns, with_text=''):

    df[columns] = df[columns].fillna(with_text)

    return df



def remove_stopwords(tweet):

    combined_stopwords = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))

    tokens = [token for token in tweet.split() if token not in combined_stopwords]

    return ' '.join(tokens)



def clean_df(df):

    df['cleaned_text'] =  df['text'].apply(clean_tweet)

#     df['cleaned_text'] = df['cleaned_text'].apply(remove_stopwords)

    df['original_length'] = df['text'].apply(lambda x: len(x))

    df['cleaned_length'] = df['cleaned_text'].apply(lambda x: len(x))

    fill_missing_text(df, ['keyword', 'location'], '')

    df = df.drop('text', axis=1)

    return df



get_clean_text = FunctionTransformer(clean_df, validate = False)
### sklearn.feature_extraction.text

cvec = CountVectorizer()

tfidf = TfidfVectorizer()

hvec = HashingVectorizer()



### Token Patterns

TOKENS_ALPHANUMERIC_d ='(?u)\\b\\w\\w+\\b' # Default

TOKENS_ALPHANUMERIC_1 = '[A-Za-z0-9]+(?=\\s+)'

TOKENS_ALPHANUMERIC_2 = r'\w{1,}'



### arguments applicable to all text vectorizers

base_args = {

        'encoding' : 'utf-8'

        ,'decode_error' : 'strict'

        ,'strip_accents' : None

        ,'lowercase' : True

        ,'preprocessor': None

        ,'tokenizer' : None

        ,'analyzer' : 'word'

        ,'stop_words': None

        ,'token_pattern' : TOKENS_ALPHANUMERIC_2

        ,'ngram_range' : (1,3)

    }



### specific to Count Vectorizer

cvec_args = {

        'max_df' : 1.0

        ,'min_df' : 1

        ,'max_features' : None

        ,'vocabulary': None

    }



### specific to Tfidf Vectorizer

tfidf_args = {

        'max_df' : 1.0

        ,'min_df' : 1

        ,'max_features' : None

        ,'vocabulary' : None

        ,'use_idf' : True

        ,'smooth_idf' : True

        ,'sublinear_tf' : False

    }



### specific to Hashing Vectorizer

hashing_args = {

        #'n_features' : 1048576

        'norm' : 'l2'

        ,'alternate_sign' : True

    ,'binary' : True

    }



### Merge contents of dict2 and dict1 to dict3

cvec_params = {**base_args , **cvec_args}

tfidf_params = {**base_args , **tfidf_args}

hashing_params = {**base_args , **hashing_args}
#set the parameters for the text vectorizers

cvec.set_params(**cvec_params)

tfidf.set_params(**tfidf_params)

hvec.set_params(**hashing_params)

print('done')
from sklearn.feature_selection import chi2, SelectKBest, f_classif

chi_k = 3000

fs__kbest = SelectKBest(chi2, chi_k)

fs__f_classif = SelectKBest(f_classif, chi_k)
#declare which columns to will be unioned with the numeric features.

text = 'cleaned_text'

get_text_data = FunctionTransformer(lambda x: x[text], validate=False)
def reset_index(dataframe):

    dataframe = dataframe.reset_index(inplace = False)

    return dataframe



#Create Transformer for pipeline

get_reset_index = FunctionTransformer(reset_index, validate=False)
#construct a pipeline 

text_features = Pipeline([

                    ('reset_index', get_reset_index),



                    ('cleanText',get_clean_text),

                    ('selector', get_text_data),

                    #('t_vectorizer', tfidf),

#                     ('c_vectorizer', cvec),

                     ('h_vectorizer', hvec),

                    ('feature_selection', fs__f_classif)

                ])
numeric= ['word_count'

          ,'char_count'

          ,'word_density'

          ,'punc_count'

          ,'total_length'

          ,'capitals'

          ,'caps_vs_length'

          ,'num_exclamation_marks'

          ,'num_question_marks'

          ,'num_punctuation'

          ,'num_symbols'

          ,'num_unique_words'

          ,'words_vs_unique'

          ,'word_unique_percent'

          ,'contains_mention'

          ,'contains_hashtag'

          ,'contains_URL'

          ,'tb_Subj'

          ,'tb_Pol'

         ]



get_numeric_data = FunctionTransformer(lambda x: x[numeric], validate=False)        
numeric_features = Pipeline([

                    ('reset_index', get_reset_index),

                    ('word_count', get_word_count),

                    ('indicators',get_indicators),

                    ('selector', get_numeric_data),

                    #('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

                    ('features', polynomial_features)

                ])
### Combine the text features with the numeric ceatures

union_features = Pipeline(steps =[('union', FeatureUnion(

                transformer_list = [

                    ('numeric_features', numeric_features),

                    ('text_features', text_features)

                ]))  

            ])



classifiers = [

        LogisticRegression(C=1.0),

        #MultinomialNB(),

        KNeighborsClassifier(20),

        #SVC(kernel="rbf", C=0.025, probability=True),

        #NuSVC(probability=True),

        DecisionTreeClassifier(),

        RandomForestClassifier(),

        AdaBoostClassifier(),

        GradientBoostingClassifier()

        ]



features = training_data.drop(['target'], axis=1)



X = features

y = training_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y

                                                   , test_size = .3

                                                   #, stratify=y

                                                   , random_state = 42

                                                   , shuffle=True

                                                   )    





for classifier in classifiers:

    pl = Pipeline(steps=[('preprocessor', union_features),

                      ('classifier', classifier)])

    

    pl.fit(X_train, y_train)   

    

    print(classifier)

    print("model score: %.3f" % pl.score(X_test, y_test))
classifier = GradientBoostingClassifier()

param_grid = {

        'classifier__criterion' : ['friedman_mse'] #friedman_mse, mae, mse

        ,'classifier__n_estimators' : [700]

        ,'classifier__max_depth' : [7]

        ,'classifier__learning_rate': [0.025]

        ,'classifier__random_state' : [42]

        ,'classifier__min_samples_leaf' : [21]

        ,'classifier__min_samples_split' : [27]

        ,'classifier__loss' : ['deviance']  #‘deviance’, ‘exponential’

#         ,'classifier__init' : [None]

        ,'classifier__max_features' : ['sqrt'] #None, 'auto', 'sqrt', 'log2'

#         ,'classifier__max_leaf_nodes': [None]

#         ,'classifier__min_impurity_decrease' : [0.0]

#         ,'classifier__min_impurity_split' : [None]

#         ,'classifier__min_weight_fraction_leaf' : [0.0]

#         ,'classifier__n_iter_no_change' : [None]

#         ,'classifier__subsample' : [1.0]

#         ,'classifier__tol' : [0.0001]

#         ,'classifier__validation_fraction' : [0.1]

#         ,'classifier__verbose' : [0]

#         ,'classifier__warm_start' : [False]

    }
pl = Pipeline(steps=[('preprocessor', union_features),

                  ('classifier', classifier)], verbose = True)
pl.fit(X_train, y_train)

pl.score(X_train, y_train)





pred = pl.predict(X_test)

f1_score(y_test, pred)
training_data_cleaned = clean_df(training_data)
# Cleaning our test data to be consistent with training set

test_data_cleaned = clean_df(test_data)
# Make a custom stemming tokenizer

def tokenizer_stems(document):

    stemmer = PorterStemmer()

    tokens = document.split()

    return [stemmer.stem(token) for token in tokens]
# Make a lemmatization tokenizer

# regezp used in CountVectorizer

regexp = re.compile('(?u)\\b\\w\\w+\\b')



# load spacy language model

en_nlp = spacy.load('en', disable=['parser', 'ner'])

old_tokenizer = en_nlp.tokenizer



# replace the tokenizer with the preceding regexp

en_nlp.tokenizer = lambda string: old_tokenizer.tokens_from_list(regexp.findall(string))



# custom tokenizer

def tokenizer_lemma(document):

    doc_spacy = en_nlp(document)

    return [token.lemma_ for token in doc_spacy]
training_data_cleaned['target_class'] = training_data_cleaned['target']
training_data_cleaned.head()
X = training_data_cleaned.drop(['keyword', 'location', 'target'], axis=1)

y = training_data_cleaned['target_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train.head()
vectorizer = TfidfVectorizer(min_df=2)

vectorizer.fit(X_train['cleaned_text'])

v = vectorizer.transform(X_train['cleaned_text']).todense()



features = vectorizer.get_feature_names()

vect_df = pd.DataFrame(v, columns=features)



train_combined_df = pd.concat([training_data_cleaned, vect_df], axis=1)
v = vectorizer.transform(test_data_cleaned['cleaned_text']).todense()



features = vectorizer.get_feature_names()

vect_df = pd.DataFrame(v, columns=features)



test_combined_df = pd.concat([test_data_cleaned, vect_df], axis=1)
X = train_combined_df.drop(['keyword', 'location', 'target', 'id', 'cleaned_text'], axis=1)

y = train_combined_df['target_class']
classifiers = [

    ('LinearSVC', LinearSVC()),

    ('SGD', SGDClassifier()),

    ('BernNB', BernoulliNB()),

    ('MultNB', MultinomialNB()),

    ('Logistic', LogisticRegression()),

    ('RandomForests', RandomForestClassifier()),

    ('ExtraTrees', ExtraTreesClassifier()),

]



param_grid = [

    {

        'clf': [LinearSVC()],

        'clf__C': np.linspace(0.01, 0.5, 5)

    }, {

        'clf': [SGDClassifier()],

        'clf__penalty': ['l1', 'l2'],

        'clf__alpha': np.linspace(0.0001, 0.1, 5)

    }, {

        'clf': [BernoulliNB()],

        'clf__alpha': np.linspace(0.01, 1, 5)

    }, {

        'clf': [MultinomialNB()],

        'clf__alpha': np.linspace(0.01, 1, 5)

    }, {

        'clf': [LogisticRegression()]

    }, {

        'clf': [RandomForestClassifier()],

        'clf__n_estimators': [300]

    }, {

        'clf': [ExtraTreesClassifier()],

        'clf__n_estimators': [300]

    }

]

        

for (name, classifier), params in zip(classifiers, param_grid):

    clf_pipe = Pipeline([

        ('clf', classifier),        

    ])



    gs_clf = GridSearchCV(clf_pipe, params, cv=3, n_jobs=-1)

    clf = gs_clf.fit(X_train, y_train)

    print("{} test score: {}".format(name, clf.score(X_test, y_test)))

    print("best params: {}\n".format(clf.best_params_))
# Unorganized stuff follows
X_train, X_test, y_train, y_test = train_test_split(X['cleaned_text'], y, test_size=.2, random_state=42)
param_grid = [

    {

        'vect__ngram_range': [(1, 1), (1, 2)],

        'vect__min_df': [1, 3, 5],

        'clf': [LinearSVC()],

        'clf__C': np.linspace(0.001, 2, 5)

    }, {

        'vect__ngram_range': [(1, 1), (1, 2)],

        'vect__min_df': [1, 3, 5],

        'clf': [SGDClassifier()]

    }, {

        'vect__ngram_range': [(1, 1), (1, 2)],

        'vect__min_df': [1, 3, 5],

        'clf': [BernoulliNB()]

    }, {

        'vect__ngram_range': [(1, 1), (1, 2)],

        'vect__min_df': [1, 3, 5],

        'clf': [MultinomialNB()]

    }

]



classifiers = [

    ('LinearSVC', LinearSVC()),

    ('SGD', SGDClassifier()),

    ('BernNB', BernoulliNB()),

    ('MultNB', MultinomialNB())

]



param_grid = [

    {

        'vect__ngram_range': [(1, 1), (1, 2)],

        'vect__min_df': [1, 2, 3],

        'clf': [LinearSVC()],

        'clf__C': np.linspace(0.01, 0.5, 5)

    }, {

        'vect__ngram_range': [(1, 1), (1, 2)],

        'vect__min_df': [1, 2, 3],

        'clf': [SGDClassifier()],

        'clf__penalty': ['l1', 'l2'],

        'clf__alpha': np.linspace(0.0001, 0.1, 5)

    }, {

        'vect__ngram_range': [(1, 1), (1, 2)],

        'vect__min_df': [1, 2, 3],

        'clf': [BernoulliNB()],

        'clf__alpha': np.linspace(0.01, 1, 5)

    }, {

        'vect__ngram_range': [(1, 1), (1, 2)],

        'vect__min_df': [1, 2, 3],

        'clf': [MultinomialNB()],

        'clf__alpha': np.linspace(0.01, 1, 5)

    }

]

        

for (name, classifier), params in zip(classifiers, param_grid):

    clf_pipe = Pipeline([

        ('vect', TfidfVectorizer()),

        ('clf', classifier),        

    ])



    gs_clf = GridSearchCV(clf_pipe, params, cv=3, n_jobs=-1)

    clf = gs_clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    print("{} test score: {}".format(name, score))

    print("best params: {}\n".format(clf.best_params_))
param_grid = {

    'vect__ngram_range': [(1, 1), (1, 2)],

    'vect__min_df': [2],

    'clf__alpha': np.linspace(1.01, 1.15, 10)

}
pipe = Pipeline([

    ('vect', TfidfVectorizer()),

    ('clf', BernoulliNB())

])
grid_search = GridSearchCV(pipe, param_grid, cv=5, return_train_score=True, n_jobs=-1)

grid_search.fit(X_train, y_train)
grid_search.best_params_
pipe = Pipeline([

    ('vect', TfidfVectorizer(min_df=2, ngram_range=(1, 1))),

    ('clf', BernoulliNB(alpha=1.087))

])

    

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)