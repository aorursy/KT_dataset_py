import numpy as np

import pandas as pd

from pathlib import Path



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score



from matplotlib import pyplot as plt

%config InlineBackend.figure_format = 'retina'
PATH_TO_DATA = Path('../input/hierarchical-text-classification/')
train_df = pd.read_csv(PATH_TO_DATA / 'train_40k.csv').fillna(' ')

valid_df = pd.read_csv(PATH_TO_DATA / 'val_10k.csv').fillna(' ')
train_df.head()


export_train_df = train_df.loc[:,['Cat1','Cat2','Title','Text',]]

export_valid_df = valid_df.loc[:,['Cat1','Cat2','Title','Text',]]

export_train_df.columns = ['category_1','category_2','title','text']

export_valid_df.columns = ['category_1','category_2','title','text']

#export_train_df.to_excel('hierarchical_multiclass_text_amazon_reviews_train.xlsx', index=False)

#export_valid_df.to_excel('hierarchical_multiclass_text_amazon_reviews_valid.xlsx', index=False)



export_train_df.head()
export_valid_df.head()
from bs4 import BeautifulSoup

import regex

data_columns = ['title','text',]

Y_columns = ['category_1','category_2',]

def preprocess_dataframe(input_df,data_columns,Y_columns):





    df = input_df.loc[:,Y_columns]



    df['text'] = input_df[data_columns].apply(lambda x: ' '.join(x.map(str)), axis=1)

    df['text'] = df['text'].apply( lambda x: BeautifulSoup(str(x),'html.parser').get_text())



    pattern = regex.compile('[\W\d_]+', regex.UNICODE)

    df['text'] = df['text'].apply( lambda x: pattern.sub(' ',str(x)))

    return df
df_train = preprocess_dataframe(export_train_df,data_columns,Y_columns)
print(df_train)
df_valid = preprocess_dataframe(export_valid_df,data_columns,Y_columns)
print(df_valid)
from nltk.corpus import stopwords

language_stop_words = stopwords.words('english')



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=2) #ngram_range=(1,2)



import numpy as np



#https://stackoverflow.com/a/55742601/4634344

vectorizer.fit(df_train['text'].apply(lambda x: np.str_(x)))

X_train = vectorizer.transform(df_train['text'].apply(lambda x: np.str_(x)))



# we need the class labels encoded into integers for functions in the pipeline

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder()

Y_train = oe.fit_transform(df_train[Y_columns].values.reshape(-1, 2))



X_valid = vectorizer.transform(df_valid['text'].apply(lambda x: np.str_(x)))

Y_valid = oe.transform(df_valid[Y_columns].values.reshape(-1, 2))



print('X training shape', X_train.shape, X_train.dtype)

print('Y training shape', Y_train.shape, Y_train.dtype)

print('X validation shape', X_valid.shape, X_valid.dtype)

print('Y validation shape', Y_valid.shape, Y_valid.dtype)
from sklearn.multioutput import ClassifierChain

from sklearn.linear_model import SGDClassifier



clf=ClassifierChain(SGDClassifier(random_state=0, class_weight='balanced', n_jobs=-1))


from sklearn.metrics import jaccard_score, f1_score, make_scorer



def concat_categories(Y):

  return np.apply_along_axis(lambda a: str(a[0]) + '-' + str(a[1]), 1, Y)



# score for predicting category_1

def js_0(y,y_pred, **kwargs):

  return jaccard_score(y[:,0], y_pred[:,0], average='micro')

# score for predicting category_2

def js_1(y,y_pred, **kwargs):

  return jaccard_score(y[:,1], y_pred[:,1], average='micro')

def f1_0(y,y_pred, **kwargs):

  return f1_score(y[:,0], y_pred[:,0], average='micro')

def f1_1(y,y_pred, **kwargs):

  return f1_score(y[:,1], y_pred[:,1], average='micro')

# score for predicting 'category_1-category_2' (concatenated strings)

def js_01(y,y_pred, **kwargs):

  return jaccard_score(concat_categories(y), concat_categories(y_pred), average='micro')

def f1_01(y,y_pred, **kwargs):

  return f1_score(concat_categories(y), concat_categories(y_pred), average='micro')



js_0_scorer = make_scorer(score_func=js_0, greater_is_better=True, needs_proba=False, needs_threshold=False)

js_1_scorer = make_scorer(score_func=js_1, greater_is_better=True, needs_proba=False, needs_threshold=False)

js_01_scorer = make_scorer(score_func=js_01, greater_is_better=True, needs_proba=False, needs_threshold=False)

f1_0_scorer = make_scorer(score_func=f1_0, greater_is_better=True, needs_proba=False, needs_threshold=False)

f1_1_scorer = make_scorer(score_func=f1_1, greater_is_better=True, needs_proba=False, needs_threshold=False)

f1_01_scorer = make_scorer(score_func=f1_01, greater_is_better=True, needs_proba=False, needs_threshold=False)

clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_valid)
print('For both Level 1 and Level 2  concatenated:\n\tF1 micro (=accuracy): {}'.format(f1_01(Y_valid,Y_pred).round(3)))
print('Just the Level 1:\n\tF1 micro (=accuracy): {}'.format(f1_0(Y_valid,Y_pred).round(3)))
print('Just the Level 2:\n\tF1 micro (=accuracy): {}'.format(f1_1(Y_valid,Y_pred).round(3)))