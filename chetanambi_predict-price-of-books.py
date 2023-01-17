import re

import nltk

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')



from nltk.corpus import stopwords

stopwords = stopwords.words('english')
train = pd.read_excel('/kaggle/input/Data_Train.xlsx')

test = pd.read_excel('/kaggle/input/Data_Test.xlsx')
train.shape, test.shape
train.duplicated().sum(), test.duplicated().sum()
train.head()
train.info()
train.nunique()
plt.figure(figsize=(10,6))

sns.distplot(np.log1p(train['Price']))
train = train[train['Price'] < 10000].reset_index(drop=True)
df = train.append(test,ignore_index=True)

df.shape
df.columns = ['Author', 'BookCategory', 'Edition', 'Genre', 'Price', 'Reviews', 'Ratings', 'Synopsis', 'Title']

df.shape
df['Author'] = df['Author'].str.replace('Agrawal P. K.','Agrawal P.K')

df['Author'] = df['Author'].str.replace('Ajay K. Pandey','Ajay K Pandey')

df['Author'] = df['Author'].str.replace('B. A. Paris','B A Paris')
df.head(2)
df['Title_1'] = df['Title'].str.extract(r"\((.*?)\)", expand=False) 

df['Title_1'] = df['Title_1'].fillna('missingTitle') 

df['Genre'] = df['Genre'].str.replace(r"\(.*\)","")

df['Reviews'] = df['Reviews'].str.replace(',','')

df['Reviews'] = df['Reviews'].str.split().str.get(0).astype(float)

df['Ratings'] = df['Ratings'].str.split().str.get(0).astype(float)

df['BookCategory_1'] = df['BookCategory'].str.split(',').str[0]

df['BookCategory_2'] = df['BookCategory'].str.split(',').str[1]

df['BookCategory_2'] = df['BookCategory_2'].fillna('missingBookCategory') 
df['Edition_1'] = df['Edition'].str.split(',').str[0]

df['Edition_2'] = df['Edition'].str.split(',').str[1]

df['Edition_2'] = df['Edition_2'].str.replace('– ','')

df['Edition_year'] = pd.to_datetime(df['Edition_2'], errors='coerce').dt.year

df['Edition_month'] = pd.to_datetime(df['Edition_2'], errors='coerce').dt.month

df['Edition_date'] = pd.to_datetime(df['Edition_2'], errors='coerce').dt.day
df.isnull().sum()
df.head(2)
agg_func = {

    'Reviews': ['mean','median','min','max','sum'],

    'Ratings': ['mean','median','min','max','sum']

}

agg_BookCategory_1 = df.groupby('BookCategory_1').agg(agg_func)

agg_BookCategory_1.columns = [ 'BookCategory_1_' + ('_'.join(col).strip()) for col in agg_BookCategory_1.columns.values]

agg_BookCategory_1.reset_index(inplace=True)

df = df.merge(agg_BookCategory_1, on=['BookCategory_1'], how='left')
agg_func = {

    'Reviews': ['mean','median','min','max','sum'],

    'Ratings': ['mean','median','min','max','sum']

    

}

agg_BookCategory_2 = df.groupby('BookCategory_2').agg(agg_func)

agg_BookCategory_2.columns = [ 'BookCategory_2_' + ('_'.join(col).strip()) for col in agg_BookCategory_2.columns.values]

agg_BookCategory_2.reset_index(inplace=True)

df = df.merge(agg_BookCategory_2, on=['BookCategory_2'], how='left')
agg_func = {

    'Reviews': ['mean','median','min','max','sum'],

    'Ratings': ['mean','median','min','max','sum']

}

agg_Author = df.groupby('Author').agg(agg_func)

agg_Author.columns = [ 'Author_' + ('_'.join(col).strip()) for col in agg_Author.columns.values]

agg_Author.reset_index(inplace=True)

df = df.merge(agg_Author, on=['Author'], how='left')
agg_func = {

    'Reviews': ['mean','median','min','max','sum'],

    'Ratings': ['mean','median','min','max','sum']

}

agg_Genre = df.groupby('Genre').agg(agg_func)

agg_Genre.columns = [ 'Genre_' + ('_'.join(col).strip()) for col in agg_Genre.columns.values]

agg_Genre.reset_index(inplace=True)

df = df.merge(agg_Genre, on=['Genre'], how='left')
agg_func = {

    'Reviews': ['mean','median','min','max','sum'],

    'Ratings': ['mean','median','min','max','sum']

}

agg_Title = df.groupby('Title').agg(agg_func)

agg_Title.columns = [ 'Title_' + ('_'.join(col).strip()) for col in agg_Title.columns.values]

agg_Title.reset_index(inplace=True)

df = df.merge(agg_Title, on=['Title'], how='left')
calc = df.groupby(['Title'], axis=0).agg({'Title':[('op1', 'count')]}).reset_index() 

calc.columns = ['Title','Title Count']

df = df.merge(calc, on=['Title'], how='left')



calc = df.groupby(['BookCategory_1'], axis=0).agg({'BookCategory_1':[('op1', 'count')]}).reset_index() 

calc.columns = ['BookCategory_1','BookCategory_1 Count']

df = df.merge(calc, on=['BookCategory_1'], how='left')



calc = df.groupby(['Edition_year'], axis=0).agg({'Edition_year':[('op1', 'count')]}).reset_index() 

calc.columns = ['Edition_year','Edition_year Count']

df = df.merge(calc, on=['Edition_year'], how='left')



calc = df.groupby(['Edition_month'], axis=0).agg({'Edition_month':[('op1', 'count')]}).reset_index() 

calc.columns = ['Edition_month','Edition_month Count']

df = df.merge(calc, on=['Edition_month'], how='left')



calc = df.groupby(['Edition_date'], axis=0).agg({'Edition_date':[('op1', 'count')]}).reset_index() 

calc.columns = ['Edition_date','Edition_date Count']

df = df.merge(calc, on=['Edition_date'], how='left')
df['Title_Synopsis'] = df['Title'] + ' ' + df['Synopsis'] + ' ' + df['Author']
from nltk.stem.wordnet import WordNetLemmatizer

lemma = WordNetLemmatizer()
# function for data cleaning and lemmatization

def clean_reviews(review_text, logging=False):

    counter = 1

    clean_text = []

    for texts in review_text:

        if counter % 1000 == 0 and logging:

            print("Processed %d records." % (counter))

        counter += 1

        texts = texts.lower()

        texts = re.sub(r'www.[^ ]+', ' ', texts)

        texts = re.sub(r'https?://[^ ]+', ' ', texts)

        texts = re.sub(r'https://[^ ]+', ' ', texts)

        texts = re.sub(r'[^a-z]', ' ', texts)

        tokens = nltk.word_tokenize(texts)

        tokens = [tok for tok in tokens if len(tok) > 2]

        tokens = [lemma.lemmatize(token, pos='n') for token in tokens]

        tokens = ' '.join(tokens)

        clean_text.append(tokens)

    return pd.Series(clean_text)



clean_text = clean_reviews(df['Title_Synopsis'], logging=True)

df['Title_Synopsis'] = clean_text
from sklearn.feature_extraction.text import TfidfVectorizer

tf1 = TfidfVectorizer(ngram_range=(1, 3), min_df=5, token_pattern=r'\w{3,}', max_features=15000)

df_title = tf1.fit_transform(df['Title_Synopsis'])

df_title = pd.DataFrame(data=df_title.toarray(), columns=tf1.get_feature_names())

df_title.shape
df = pd.concat([df, df_title], axis=1) 

df.shape
for col in ['BookCategory_1', 'BookCategory_2', 'Edition_1', 'Edition_2', 'Genre', 'Title_1']:

    df[col] = df[col].astype('category')
df.drop(['Synopsis','Title','BookCategory','Edition','Title_Synopsis','Author'], axis=1, inplace=True)
train_df = df[df['Price'].isnull()!=True]

test_df = df[df['Price'].isnull()==True]

test_df.drop(['Price'], axis=1, inplace=True)
train_df.shape, test_df.shape
train_df['Price'] = np.log1p(train_df['Price'])
train_df.head(2)
X = train_df.drop(labels=['Price'], axis=1)

y = train_df['Price'].values



from sklearn.model_selection import train_test_split

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.25, random_state=1)
X_train.shape, y_train.shape, X_cv.shape, y_cv.shape
from math import sqrt 

from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train)

test_data = lgb.Dataset(X_cv, label=y_cv)



param = {'objective': 'regression',

         'boosting': 'gbdt',  

         'metric': 'l2_root',

         'learning_rate': 0.01, 

         'num_iterations': 3500,

         'num_leaves': 80,

         'max_depth': -1,

         'min_data_in_leaf': 11,

         'bagging_fraction': 0.80,

         'bagging_freq': 1,

         'bagging_seed': 3,

         'feature_fraction': 0.80,

         'feature_fraction_seed': 2,

         'early_stopping_round': 200,

         'max_bin': 250

         }



lgbm = lgb.train(params=param, verbose_eval=100, train_set=train_data, valid_sets=[test_data])



y_pred_lgbm = lgbm.predict(X_cv)

print('RMSLE:', sqrt(mean_squared_log_error(np.expm1(y_cv), np.expm1(y_pred_lgbm))))
feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importance(), X.columns), reverse=True)[:50], 

                           columns=['Value','Feature'])

plt.figure(figsize=(12, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()
Xtest = test_df
from sklearn.model_selection import KFold

from lightgbm import LGBMRegressor



errlgb = []

y_pred_totlgb = []



fold = KFold(n_splits=15, shuffle=True, random_state=42)



for train_index, test_index in fold.split(X):

    X_train, X_test = X.loc[train_index], X.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]



    lgbm = LGBMRegressor(**param)

    lgbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0, early_stopping_rounds=200)



    y_pred_lgbm = lgbm.predict(X_test)

    print("RMSLE LGBM: ", sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_lgbm))))



    errlgb.append(sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_lgbm))))

    p = lgbm.predict(Xtest)

    y_pred_totlgb.append(p)
np.mean(errlgb,0)
lgbm_final = np.expm1(np.mean(y_pred_totlgb,0))

lgbm_final
df_sub = pd.DataFrame(data=lgbm_final, columns=['Price'])

writer = pd.ExcelWriter('Output.xlsx', engine='xlsxwriter')

df_sub.to_excel(writer,sheet_name='Sheet1', index=False)

writer.save()