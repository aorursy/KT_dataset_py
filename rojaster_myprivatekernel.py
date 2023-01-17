# Для вас представлена задача, сделанная на открытых данных.

# Вам нужно предсказать цену товара по другим характеристикам.

# Метрика качества RMSLE 

# np.sqrt(np.mean((np.log(y_test + 1) - np.log(y_hat + 1))**2))
# np.sqrt(np.mean(np.power(np.log1p(y_test)-np.log1p(y_hat), 2)))
# np.sqrt(mean_squared_log_error(y_test, predictions))

# can_buy - возможно ли купить этот товар,

# can_promote - возможность продвигать товар,

# category - категория товара,

# contacts_visible - видны ли контакты,

# date_created - таймстемп создания товара,

# delivery_available - возможность доставки,

# description - описание,

# fields - свойства товара,

# id - идентификатор продукта,

# images - id картинок,

# location - местрорасположение,

# mortgage_available - возможность ипотеки,

# name - навзаник,

# payment_available - возможность онлайн оплаты,

# price - цена,

# subcategory - подкатегория,

# subway - метро,
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
import seaborn as sns
import datetime as dt
warnings.filterwarnings('ignore')
%matplotlib inline

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
STOPWORDS = []
STOPWORDS.extend(stopwords.words('english'))
STOPWORDS.extend(stopwords.words('russian'))
# import pymorphy2
# morph  = pymorphy2.MorphAnalyzer()
# will care only about stopwords, neither isalnum or is_digit
# morphy_tokenizer = lambda text: [morph.parse(w)[0].normal_form for w in text.split() if w not in STOPWORDS] 
def nltk_tokenizer(text):
    return [w for w in nltk.word_tokenize(text) if w.isalnum() and w not in STOPWORDS]
# np.sqrt(np.mean((np.log(y + 1) - np.log(y_hat + 1))**2))    
# np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y_hat), 2))) 
# np.sqrt(mean_squared_log_error(y, y_hat)) 
def RMLSE_SCORE(y, y_hat):
    assert not np.any(np.isnan(y_hat))
    assert np.all(np.isfinite(y_hat))
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y_hat), 2)))
    #return np.sqrt(mean_squared_log_error(y, y_hat)) 

# E - model. estimator 
# X - data
# y - target
def RMLSE_SCORER(E, X, y):
    return RMLSE_SCORE(y, E.predict(X))
# Get rid of from all non helpful things from the text: по, (текст какой-то в скобках), slashes and etc
import re
re_parentheses_trash_ripper = re.compile("[!#$%&'*+.,^_`|~:;]|\(.*?\)", flags=re.IGNORECASE)
re_slash_dash_replacer = re.compile("[-/[\\]]")
re_double_quotes_ripper = re.compile("\".*?\"")
re_html_escapers = re.compile("&\w+;")
re_punctuation_ripper = re.compile("[!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]", flags=re.IGNORECASE)

def erase_chars_by_pattern(_text, _re_compiled_pattern, _substitution=''):
    _text = re.sub(_re_compiled_pattern, 
                   _substitution, 
                   str(_text)).strip()
    return _text

def name_descr_preproc(df):
    # I am too far away from doing it more efficiently putting it into pipeline(loop) with line of the parameters
    # so, suck it up=)
    df['name'] = df['name'].map(lambda value: erase_chars_by_pattern(value, re_parentheses_trash_ripper))
    df['name'] = df['name'].map(lambda value: erase_chars_by_pattern(value, re_slash_dash_replacer, ' '))
    df['name'] = df['name'].map(lambda value: erase_chars_by_pattern(value, re_double_quotes_ripper))
    df['name'] = df['name'].map(lambda value: str(value).lower())
    
    df['description'] = df['description'].map(lambda value: erase_chars_by_pattern(value, re_slash_dash_replacer, ' '))
    df['description'] = df['description'].map(lambda value: erase_chars_by_pattern(value, re_punctuation_ripper))
    df['description'] = df['description'].map(lambda value: erase_chars_by_pattern(value, re_html_escapers))
    df['description'] = df['description'].map(lambda value: str(value).lower())
    df['description'] = df['description'].map(lambda value: ' '.join([word for word in value.split() if word.isalnum()]))
    
    return df
def load_pd_from_pickle(pickle_file=None):
    assert pickle_file
    return pickle.load(open(pickle_file, 'rb'))
# The kernel dies at this point
# train_sample_pd = load_pd_from_pickle('../input/hackathon-sf-ml-3/train_hack.pckl')

# train4col - name descr price 
df1_train = pd.read_csv('../input/hsfml12/train_4_col.csv', sep='\t', header=0, index_col=0)

# train4col2 - cat subcat
df2_train = pd.read_csv('../input/hsfml12/train_4_col_2.csv', sep='\t', header=0, index_col=0)

# train4col3 - id fields
# df3 = pd.read_csv('../input/hsfml12/train_4_col_3.csv', sep='\t', header=0, index_col=0)

# test_3_col - name descr
df1_test = pd.read_csv('../input/hsfml12/test_3_col.csv', sep='\t', header=0, index_col=0)

# test_3_col2 - cat subcat
df2_test = pd.read_csv('../input/hsfml12/test_3_col_2.csv', sep='\t', header=0, index_col=0)
df1_train.loc[df1_train['price']<=0]

# train dataset
df1_train.set_index('id',drop=True, inplace=True)
df1_train['description'] = df1_train['description'].map(lambda v: 'Не указано' if str(v) in 'nan' or not str(v).strip() else str(v)).astype('U')
df1_train['name'] = df1_train['name'].astype('U')
df1_train['price'] = df1_train['price'] / 100
df2_train.set_index('id',drop=True, inplace=True)
# df3.set_index('id',drop=True, inplace=True)


# test_dataset
df1_test.set_index('id',drop=True, inplace=True)
df1_test['description'] = df1_test['description'].map(lambda v: 'Не указано' if str(v) in 'nan' or not str(v).strip() else str(v)).astype('U')
df1_test['name'] = df1_test['name'].astype('U')
df2_test.set_index('id',drop=True, inplace=True)

df1_train.loc['ca3f9e05f4f955401837cb75']
df1_train.loc[df1_train['description'] == 'Не указано'].__len__() / df1_train.shape[0]
df1_train.head()
df1_test.loc[df1_test['description'] == 'Не указано'].__len__() / df1_test.shape[0]
df1_test.head()
df1_train.loc[df1_train['price'] <= 0]
df1_combined = df1_train.append(df1_test)
cv_name = CountVectorizer(tokenizer=nltk_tokenizer, ngram_range=(1,1))
cv_desc = CountVectorizer(tokenizer=nltk_tokenizer, ngram_range=(1,1))
cv_name.fit(df1_combined['name'])
cv_desc.fit(df1_combiner['description'])
cv_name.vocabulary_.__len__()
cv_desc.vocabulary_.__len__()
pickle.dump(cv_name, open('cv_name.pckl', 'wb'))
pickle.dump(cv_desc, open('cv_desc.pckl', 'wb'))
from sklearn.decomposition import TruncatedSVD
tsvd = TruncatedSVD(n_components=8)
from sklearn.ensemble import RandomForestRegressor
regressor =  RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=2)
from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, col_name='all'):
        self.col_name = col_name
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.col_name == 'all':
            return X
        return X[self.col_name]
    
class ColumnsDropper(BaseEstimator, TransformerMixin):
    def __init__(self, col_names):
        assert isinstance(col_names,list), 'col_names must be a list of exists columns in dataframe'
        assert all([True if type(c) is str else False for c in col_names]), 'col_name must be a string'
        self.col_names = col_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.drop(self.col_names, axis=1) #.values ???
        return X
# df_train = df1_combined.query('sample == TRAIN')
# df_test = df1_combined.query('sample == TEST')
df_train = df1_train
df_test = df1_test
df_train = pd.concat([df_train, df2_train], axis=1)
df_test = pd.concat([df_test, df2_test], axis=1)
df_train.head()
df_test.head()
df_train = df_train.query('price > 0') # keep items only with positive price
y = df_train['price']
X = df_train.drop(['price'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
X_train.shape, X_test.shape
model = Pipeline([
    ('union', FeatureUnion([
        ('name_pipe', Pipeline([
            ('col_sel', ColumnSelector('name')),
            ('cv_name', cv_name),
            ('best', tsvd)
        ]))
        ,
        ('desc_pipe', Pipeline([
            ('col_sel', ColumnSelector('description')),
            ('cv_desc', cv_desc),
            ('best', tsvd)
        ]))
        ,
        ('cat_pipe', Pipeline([
            ('dropper', ColumnsDropper(['name', 'description']))
        ]))
    ]))
    ,
    ('regressor', regressor)
])
model.fit(X_train, y_train)
# scores = cross_val_score(model, X_train, y_train, cv=3, scoring=RMLSE_SCORER)
np.mean(scores)

y_hat = model.predict(X_test)
RMLSE_SCORE(y_test, y_hat)
preds = pd.DataFrame(data=y_hat, columns = ['predictions'])

#generating a submission file
result = pd.concat([df1_test.index.values, preds], axis=1)
result.set_index('id', inplace = True)
result.head()













