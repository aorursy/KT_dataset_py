# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

datapath= "../input/chefling/data.csv"



df = pd.read_csv(datapath)
df.head(3)
df.shape
targetdf=df['revenue']
df.isnull()
df.info()
# print(df.columns['']
df.drop(['imdb_id','poster_path'],axis=1,inplace=True)
df['status'].value_counts()
df.drop(['status'],axis=1,inplace=True)
#function for plotting the distribution plot , Violin Plot and Box Plot of the Feature Columns

def univariate(df,col,vartype,hue =None):

    sns.set(style="whitegrid")



    fig, ax=plt.subplots(nrows =1,ncols=3,figsize=(10,4))

    ax[0].set_title("Distribution Plot")

    sns.distplot(df[col],ax=ax[0])

    ax[1].set_title("Violin Plot")

    sns.violinplot(data =df, x=col,ax=ax[1], inner="quartile")

    ax[2].set_title("Box Plot")

    sns.boxplot(data =df, x=col,ax=ax[2],orient='v')

plt.show()
#For Budget

univariate(df=df,col='budget',vartype=0)
#For Revenue

univariate(df=df,col='revenue',vartype=0)
sns.jointplot(x="budget", y="revenue", data=df, size=5)
df['log_budget'] = np.log1p(df['budget'])

df['log_revenue']=np.log1p(df['revenue'])
sns.jointplot(x="log_budget", y="log_revenue", data=df, size=5)
df[['revenue', 'budget', 'runtime']].describe()

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

sns.boxplot(x='original_language', y='revenue', data=df.loc[df['original_language'].isin(df['original_language'].value_counts().head(10).index)]);

plt.title('Mean revenue per language');

plt.subplot(1, 2, 2)

sns.boxplot(x='original_language', y='log_revenue', data=df.loc[df['original_language'].isin(df['original_language'].value_counts().head(10).index)]);

plt.title('Mean log revenue per language');
sns.jointplot(x="popularity", y="log_revenue", data=df, size=5)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(

            sublinear_tf=True,

            analyzer='word',

            token_pattern=r'\w{1,}',

            ngram_range=(1, 2),

            min_df=5)



overview_text = vectorizer.fit_transform(df['overview'].fillna(''))
from sklearn.linear_model import LinearRegression

import eli5

linreg = LinearRegression()

linreg.fit(overview_text, df['log_revenue'])

eli5.show_weights(linreg, vec=vectorizer, top=20, feature_filter=lambda x: x != '<BIAS>')
sns.jointplot(x="runtime", y="log_revenue", data=df, size=5)
import ast

df.genres = df.genres.apply(lambda x: list(map(lambda d: list(d.values())[1], ast.literal_eval(x)) if isinstance(x, str) else []))

df.genres.head()
#  df['genres'].fillna("[]", inplace=True)
from collections import Counter

import itertools

genres = Counter(itertools.chain.from_iterable((df.genres.values)))

print(genres)
temp_train = df[['id', 'genres']]

for g in genres:

    temp_train[g] = temp_train.genres.apply(lambda x: 1 if g in x else 0)

#     temp_test[g] = temp_test.genres.apply(lambda x: 1 if g in x else 0)
temp_train.head(3)
df = pd.concat([df, temp_train.iloc[:,1:]], axis=1) 
print(df.shape)
df.production_countries = df.production_countries.apply(lambda x: list(map(lambda d: list(d.values())[0], ast.literal_eval(x)) if isinstance(x, str) else []))
c = Counter(itertools.chain.from_iterable((df.production_countries.values)))
print(c)
from sklearn.decomposition import PCA

big_countries = [p for p in c if c[p] > 30]

df.production_countries = df.production_countries.apply(lambda l: list(map(lambda x: x if x in big_countries else 'other', l)))



temp_train = df[['id', 'production_countries']]



for p in big_countries + ['other']:

    temp_train[p] = temp_train.production_countries.apply(lambda x: 1 if p in x else 0)

    

print(temp_train.head(2))

X_train = temp_train.drop(['production_countries', 'id'], axis=1).values



output_features=5

reduced_countries = PCA(output_features)

reduced_countries.fit_transform(X_train)
for i in range(5):

    df['reduced_countries{}'.format(i)] = X_train[:, i]
print(df.shape)
df.production_companies = df.production_companies.apply(lambda x: list(map(lambda d: list(d.values())[0], ast.literal_eval(x)) if isinstance(x, str) else []))
companies = Counter(itertools.chain.from_iterable((df.production_companies.values)))

print("Number of different production companies:", len(companies))
filtered_companies = [p for p in companies if companies[p] > 30]

df.production_companies = df.production_companies.apply(lambda l: list(map(lambda x: x if x in filtered_companies else 'other', l)))



temp_train = df[['id', 'production_companies']]



for p in filtered_companies + ['other']:

    temp_train[p] = temp_train.production_companies.apply(lambda x: 1 if p in x else 0)

    

X_train = temp_train.drop(['production_companies', 'id'], axis=1).values
output_features=5

reduced_companies = PCA(output_features)

reduced_companies.fit_transform(X_train)
for i in range(5):

    df['reduced_companies{}'.format(i)] = X_train[:, i]
print(df.shape)
X_final_train = df

X_final_train.info()
X_final_train=X_final_train.drop(['cast','crew','belongs_to_collection','homepage','genres','original_language','original_title','overview'],axis=1)
X_final_train=X_final_train.drop(['production_countries','production_companies'],axis=1)
X_final_train=X_final_train.drop(['spoken_languages','release_date','tagline','title','Keywords'],axis=1)
# X_final_train=X_final_train.drop(['homepage','original_title','overview'],axis=1)
X_final_train.shape
X_final_train.info()
import seaborn as sns

# data['target']=labels

f, ax = plt.subplots(figsize=(20, 16))

corr = X_final_train.corr()

sns.heatmap(corr[(corr >= 0.1) | (corr <= -0.9)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);

train=X_final_train[:2500]

test=X_final_train[:500]
print(test.shape)


X_train=train.drop(['log_revenue','revenue'],axis=1)

Y_train=train['log_revenue']

X_test=test.drop(['log_revenue','revenue'],axis=1)

Y_test=test['log_revenue']
print(X_train.shape)

print(Y_train.shape)

print(X_test.shape)

print(Y_test.shape)
import xgboost as xgb

xgb_data = [(xgb.DMatrix(X_train, Y_train), 'train'), (xgb.DMatrix(X_test, Y_test), 'valid')]
params = {'objective': 'reg:linear', 

          'eta': 0.01, 

          'max_depth': 6, 

          'min_child_weight': 3,

          'subsample': 0.8,

          'colsample_bytree': 0.8,

          'colsample_bylevel': 0.50, 

          'gamma': 1.45, 

          'eval_metric': 'rmse', 

          'seed': 12, 

          'silent': True    

}
xgb_model = xgb.train(params, 

                  xgb.DMatrix(X_train, Y_train),

                  5000,  

                  xgb_data, 

                  verbose_eval=200,

                  early_stopping_rounds=200)

xgb_pred = np.expm1(xgb_model.predict(xgb.DMatrix(X_test), ntree_limit=xgb_model.best_ntree_limit))
print(xgb_pred.shape)
print(xgb_pred[1])