!python -m spacy download en_core_web_lg
# importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_rows', 200)

import spacy



plt.rcParams["figure.figsize"] = (15,5)
# Loading data

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

df = train.drop('target',axis=1)

traintest = pd.concat([df,test])

print(train.shape, test.shape, traintest.shape)

train.head(10)
train.tail(10)
test.head(10)
test.tail(10)
train.isnull().sum(), test.isnull().sum()
train.nunique(dropna = False), test.nunique(dropna = False)
train['target'].value_counts().plot(kind='bar')
# top 20 keyword frequency in real and not real disaster tweets



# we will fill NaN value with 'NA' string

train['keyword'] = train['keyword'].fillna('NA')

real = train[train['target'] == 1]

noreal = train[train['target'] == 0]



# Data visualization for top 20 keywords for train set

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,10))

sns.countplot(y='keyword',hue = 'target', data = train,order = real['keyword'].value_counts()[:20].index, ax= ax1)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

ax1.set_title('Top 20 keywords for real tweet')

sns.countplot(y='keyword',hue = 'target', data = train,order = noreal['keyword'].value_counts()[:20].index, ax= ax2)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

ax2.set_title('Top 20 keywords for not real tweet')
# Top 20 keyword for test set

test['keyword'] = test['keyword'].fillna('NA')

fig, (ax1) = plt.subplots(1,1, figsize=(15,5))

sns.countplot(data = test, y = 'keyword', order = test['keyword'].value_counts(dropna = False)[:20].index,orient = 'h',ax= ax1)

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

ax1.set_title('Top 20 keywords for test set')

# Let's Analyze duplicate tweets

dup_tweets = train[train.duplicated(['text'], keep = False)].sort_values('text')

print('Total duplicates',dup_tweets.shape[0],'\nTotal unique',dup_tweets['text'].nunique(dropna = False))

dup_tweets.head(10)
disaster = dup_tweets.groupby(['keyword','text'])['target'].sum()

duplicates = dup_tweets.groupby(['keyword','text'])['target'].count()

whole = pd.DataFrame([disaster,duplicates], index=['disaster','Total']).T

whole['target'] = whole['disaster']/whole['Total']

whole.reset_index(inplace=True)

whole.head(10)
whole['target'] = whole['target'].astype(int)



# drop less significant column and duplicated rows based on text from test set 

train.drop(['location','id'], axis = 1 , inplace = True)

train.drop_duplicates(subset = 'text', keep = False, inplace =True)

print(train.shape)



# Adding filtered row in train set 

train = pd.concat([train,whole[['keyword','text','target']]],sort =False, axis =0)



print(train.shape)

train.head()

import re



# removing urls starting with 'http' from text column

train['text'] = train['text'].str.replace(r'https?://\S*','', regex=True)

# Function for emoji sentiment extraction



def emoji_sentiment(df):

    df['emoji'] = 'NA'

    selected = df['text'].str.contains(r'[:;][D\]})]', regex = True)

    df.loc[selected,'emoji'] ='Positive'

    selected = df['text'].str.contains(r'>?[:;][(c\[/]', regex = True)

    df.loc[selected,'emoji'] = 'Negative'

    

    return df



train = emoji_sentiment(train)



print(train['emoji'].value_counts())

sns.countplot(x='emoji',hue='target',data=train,order=train['emoji'].value_counts(dropna = False)[1:].index)
import string



# remove digits occured in text as it not make much sense in modelling

train['text'] = train['text'].str.replace(r"\d","", regex= True)





# remove punctuation Remove : https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove

def remove_punctuations(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



train['text'] = train['text'].apply(remove_punctuations)


from sklearn.feature_extraction.text import TfidfVectorizer



# It performs lemmatization, lowering case, stop word removal in text column and create bag of words data frame

parser = spacy.load('en_core_web_lg')



stop_words = spacy.lang.en.stop_words.STOP_WORDS



def spacy_tokenizer(sentence):

    tokens = parser(sentence)

    

    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]

    

    tokens = [ tok for tok in tokens if tok not in stop_words]

    

    return tokens





tf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer,analyzer = 'word', ngram_range = (1,1))



X = tf_vector.fit_transform(train['text'])

features = tf_vector.get_feature_names()



bow_df = pd.DataFrame(columns = features, data = X.toarray())

print(bow_df.shape)

bow_df.head()
# Creating Final Data Frame



train.reset_index(inplace=True, drop = True)

y = train['target'] 

train.drop(['target','text'],axis = 1, inplace = True)

train = pd.concat([train,bow_df], axis =1)

print(train.shape)

train.head()



# Label Encodeing and compressing sparse matrix for faster execution 



from scipy.sparse import csr_matrix

from sklearn.preprocessing import LabelEncoder



lekw = LabelEncoder()

lekw.fit(train['keyword'])

train['keyword'] = lekw.transform(train['keyword'])



leemo = LabelEncoder()

leemo.fit_transform(train['emoji'])

train['emoji'] = leemo.transform(train['emoji'])



print(train.columns)

X = csr_matrix(train.values)



print(X.shape, y.shape)
# Rescaling input to standard scalar

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(with_mean = False)

scaler.fit(X)

X= scaler.transform(X)


from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import f1_score,roc_auc_score, make_scorer

import xgboost as xgb





f1_scoring = make_scorer(f1_score,greater_is_better = True)



def model_evaluation(all_X,all_y):

    models = [

              { 'name' : 'Random Forest Regressor',

                'estimator': RandomForestClassifier(random_state = 1),

                'hyperparameters' : {'n_estimators': [300,700],

                                     'criterion':['gini','entropy'],

                                     'min_samples_split' : [2,4],

                                     'min_samples_leaf':[1,3],

                                     'n_jobs':[-1]                                     

                                    }

              },

        

        

             { 'name' : 'Support Vector Machine',

                'estimator': SVC(random_state=1),

                'hyperparameters' : {'kernel':['rbf','poly','sigmoid'],

                                    'C':[0.5,1,5],

                                    'class_weight' : [{0:0.7,1:1.0}]

                                    }

             },

             {'name' : 'Adaboost Classifier',

              'estimator': AdaBoostClassifier(random_state=1),

              'hyperparameters' : {'n_estimators':[100,300, 500, 700]}

             }

    ]

    for model in models:

        print(model['name'])

        print('*'*len(model['name']))    



        grid  = GridSearchCV(model['estimator'],param_grid= model['hyperparameters'],scoring = f1_scoring, cv = 3, n_jobs = -1)

        grid.fit(all_X,all_y)



        model["best_params"] = grid.best_params_

        model['best_model'] = grid.best_estimator_



        pred = grid.predict(all_X)

        score = f1_score(all_y,pred)

        print('f1 Score for best model: {:.4f}'.format(score))

        print('AUC score: {:.4f}'.format(roc_auc_score(all_y,pred)))

               

        print('Best Params:{}\n'.format(model['best_params']))

    

    return models







best_models = model_evaluation(X,y)

holdout = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

holdout_id =holdout['id']



holdout.drop(['id','location'],axis = 1,inplace = True)

def prepare_data(df):

    # fill NaN with NA

    df['keyword'] = df['keyword'].fillna('NA')

    print(df['keyword'].value_counts())

    

    # Remove URL

    df['text'] = df['text'].str.replace(r'https?://\S*','', regex=True)

    

    # Create emoji feature

    df = emoji_sentiment(df)

    

    # Remove Digits and punctuation

    df['text'] = df['text'].str.replace(r"\d","", regex= True)

    df['text'] = df['text'].apply(remove_punctuations)

    

    

    #TF IDF creation

    

    X = tf_vector.transform(df['text'])

    features = tf_vector.get_feature_names()

    bow_df = pd.DataFrame(columns = features, data = X.toarray())

    df = df.drop('text',axis = 1)

    df = pd.concat([df,bow_df],axis = 1)

    

    print(df.shape)

    

    return df



final_df = prepare_data(holdout)



final_df['keyword'] = lekw.transform(final_df['keyword'])

final_df['emoji'] = leemo.transform(final_df['emoji'])



print(final_df.columns)

final_df = csr_matrix(final_df.values)

final_df = scaler.transform(final_df)





def save_submission_file(holdout, model, filename="submission.csv"):

    

    predictions = model.predict(holdout)

    print(predictions)



    submission_df = {"id": holdout_id,"target": predictions}

    submission = pd.DataFrame(submission_df)



    submission.to_csv(filename,index=False)



best_model = best_models[0]["best_model"]



save_submission_file(final_df, best_model)