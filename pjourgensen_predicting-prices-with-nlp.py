#Math imports

import pandas as pd

import numpy as np

from scipy.stats import zscore



#Visualization Imports

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')

import plotly.plotly as py

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 



#Modeling Imports

from sklearn import metrics

from sklearn.feature_extraction import FeatureHasher

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler



#Text Processing Imports

import string

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.decomposition import TruncatedSVD, PCA

wine = pd.read_csv('../input/winemag-data-130k-v2.csv')
wine.info()
wine.loc[wine['country']== 'US', 'country'] = 'United States of America'

test = wine.loc[wine['price'].isnull()]

train = wine.loc[wine['price'].isnull()==False]
by_country = wine.groupby('country')
by_country = pd.concat([by_country.count()['description'],

                        by_country.max()[['points','price']]],

                        axis=1)



by_country.columns = ['count', 'Highest Rating','Highest Price']
data = dict(

        type = 'choropleth',

        locations = by_country.index,

        z = by_country['count'],

        colorscale = 'Viridis',

        locationmode = 'country names',

        text = by_country['count'],

        colorbar = {'title' : '# Wines by Country'},

      ) 



layout = dict(

    title = '# Wines by Country',

    geo = dict(

        showframe = False,

        projection = {'type':'mercator'}

    )

)



choromap = go.Figure(data = [data],layout = layout)



iplot(choromap)
plt.figure(figsize=(16,6))

wine['country'].value_counts().head(10).plot.bar()

plt.title('Top 10 Producers by Country')

plt.xlabel('Country')

plt.ylabel('Count')
data = dict(

        type = 'choropleth',

        locations = by_country.index,

        colorscale = 'Viridis',

        z = by_country['Highest Rating'],

        locationmode = 'country names',

        text = by_country['count'],

        colorbar = {'title' : 'Highest Rating'},

      ) 



layout = dict(

    title = 'Highest rating',

    geo = dict(

        showframe = False,

        projection = {'type':'mercator'}

    )

)



choromap = go.Figure(data = [data],layout = layout)



iplot(choromap)
#wine.sort_values(by='points', ascending=False)['points'].head(10)



plt.figure(figsize=(16,6))

wine[wine['points']==100]['country'].value_counts().head(10).plot.bar()

plt.title("Number of 'Perfect' Wines by Country")

plt.xlabel('Country')

plt.ylabel('Count')
data = dict(

        type = 'choropleth',

        locations = by_country.index,

        colorscale = 'Viridis',

        z = by_country['Highest Price'],

        locationmode = 'country names',

        text = by_country['count'],

        colorbar = {'title' : 'Highest Price'},

      ) 



layout = dict(

    title = 'Highest Price',

    geo = dict(

        showframe = False,

        projection = {'type':'mercator'}

    )

)



choromap = go.Figure(data = [data],layout = layout)



iplot(choromap)
plt.figure(figsize=(16,6))

wine[wine['price']>=1000]['country'].value_counts().head(10).plot.bar()

plt.title("Number of Wines over $1000 by Country")

plt.xlabel('Country')

plt.ylabel('Count')
sns.jointplot(wine[wine['price']<=1000]['points'], 

              wine[wine['price']<=1000]['price'], 

              kind="scatter", 

              color="#4CB391",

              size=10,

              s=10)
wine.head(2)
wine.drop('Unnamed: 0',inplace = True,axis = 1)
wine['country'].nunique()
sum(wine['country'].isnull())
wine['country'].fillna('unknown',inplace=True)
bins = range(0,200,5)



plt.figure(figsize = (16,6))



countries = wine['country'].value_counts().head(10).index



for i in countries:

    plt.hist(train.loc[train['country']==i]['price'],

             bins,alpha=0.5,label=i,edgecolor='white')





plt.title('Price Distribution by Country')

plt.xlabel('Price')

plt.ylabel('Count')

plt.legend()

plt.show()
one_hot = pd.get_dummies(wine['country'])
pca = PCA(n_components=5)
pca_encoded = pca.fit_transform(one_hot)
pca_df = pd.DataFrame(data=pca_encoded,columns=['country_1','country_2','country_3','country_4','country_5'])
wine = pd.concat([wine,pca_df],axis=1)
wine.drop('country',axis=1,inplace=True)
wine_desc = wine['description']
wine.drop('description',inplace = True, axis = 1)
wine['designation'].nunique()
sum(wine['designation'].isnull())
bins = range(0,200,5)



plt.figure(figsize = (16,6))



designations = wine['designation'].value_counts().head(10).index



for i in designations:

    plt.hist(train.loc[train['designation']==i]['price'],

             bins,alpha=0.5,label=i,edgecolor='white')





plt.title('Price Distribution by Designation')

plt.xlabel('Price')

plt.ylabel('Count')

plt.legend()

plt.show()
wine.drop('designation', inplace = True, axis = 1)
sum(wine['points'].isnull())
scaler = MinMaxScaler()

wine['points'] = scaler.fit_transform(wine['points'].values.reshape(-1,1))
wine['province'].nunique()
sum(wine['province'].isnull())
wine['province'].fillna('unknown',inplace=True)
bins = range(0,200,5)



plt.figure(figsize = (16,6))



provinces = wine['province'].value_counts().head(5).index



for i in provinces:

    plt.hist(train.loc[train['province']==i]['price'],

             bins,alpha=0.5,label=i,edgecolor='white')





plt.title('Price Distribution by Province')

plt.xlabel('Price')

plt.ylabel('Count')

plt.legend()

plt.show()
one_hot = pd.get_dummies(wine['province'])

pca = PCA(n_components=5)

pca_encoded = pca.fit_transform(one_hot)

pca_df = pd.DataFrame(data=pca_encoded,columns=['prov_1','prov_2','prov_3','prov_4','prov_5'])

wine = pd.concat([wine,pca_df],axis=1)

wine.drop('province',axis=1,inplace=True)
wine['region_1'].nunique()
sum(wine['region_1'].isnull())
wine['region_2'].nunique()
sum(wine['region_2'].isnull())
wine.drop(['region_1','region_2'],axis=1,inplace=True)
wine['taster_name'].nunique()
sum(wine['taster_name'].isnull())
wine['taster_name'].fillna('unknown',inplace=True)
tasters = wine['taster_name'].unique()

dic = dict(zip(tasters, list(range(1,len(tasters)+1))))

wine['taster_name'] = wine['taster_name'].apply(lambda x: str(dic[x]))
plt.figure(figsize=(12,12))

sns.boxplot(x='taster_name',y='price',data=wine,showfliers=False)

plt.tight_layout
one_hot = pd.get_dummies(wine['taster_name'])

pca = PCA(n_components=5)

pca_encoded = pca.fit_transform(one_hot)

pca_df = pd.DataFrame(data=pca_encoded,columns=['taster_1','taster_2','taster_3','taster_4','taster_5'])

wine = pd.concat([wine,pca_df],axis=1)

wine.drop('taster_name',axis=1,inplace=True)
wine.drop(['taster_twitter_handle'],axis=1,inplace=True)
test = wine.loc[0,'title']
def impute_year(S):

    year = [int(x) for x in S.split() if x.isdigit()]

    possible_years = list(range(1950,2018,1))

    if not year:

        return 0

    else:

        for i in range(len(year)):

            if year[i] in possible_years:

                return year[i]
wine['year'] = wine['title'].apply(lambda x: impute_year(x))
wine['year'].fillna(0,inplace=True)
plt.figure(figsize=(12,6))

plt.hist(wine['year'],

         bins=range(1990,2020,1),

         edgecolor='white')

plt.show()
wine.loc[wine['year']==0,'year'] = wine.loc[wine['year']>0,'year'].mean()
plt.figure(figsize=(12,6))

plt.hist(wine['year'],

         bins=range(1990,2020,1),

         edgecolor='white')

plt.show()
wine.drop(['title'],axis=1,inplace=True)
wine['age'] = wine['year'].apply(lambda x: 2018 - x)
wine.drop('year',axis=1,inplace=True)
scaler = MinMaxScaler()

wine['age'] = scaler.fit_transform(wine['age'].values.reshape(-1,1))
wine['variety'].nunique()
sum(wine['variety'].isnull())
wine['variety'].fillna('unknown',inplace=True)
bins = range(0,200,5)



plt.figure(figsize = (16,6))



varieties = wine['variety'].value_counts().head(5).index



for i in varieties:

    plt.hist(train.loc[train['variety']==i]['price'],

             bins,alpha=0.5,label=i,edgecolor='white')





plt.title('Price Distribution by Variety')

plt.xlabel('Price')

plt.ylabel('Count')

plt.legend()

plt.show()
one_hot = pd.get_dummies(wine['variety'])

pca = PCA(n_components=5)

pca_encoded = pca.fit_transform(one_hot)

pca_df = pd.DataFrame(data=pca_encoded,columns=['var_1','var_2','var_3','var_4','var_5'])

wine = pd.concat([wine,pca_df],axis=1)

wine.drop('variety',axis=1,inplace=True)
wine['winery'].nunique()
sum(wine['winery'].isnull())
bins = range(0,200,5)



plt.figure(figsize = (16,6))



wineries = wine['winery'].value_counts().head(5).index



for i in wineries:

    plt.hist(train.loc[train['winery']==i]['price'],

             bins,alpha=0.5,label=i,edgecolor='white')





plt.title('Price Distribution by Winery')

plt.xlabel('Price')

plt.ylabel('Count')

plt.legend()

plt.show()
len_features = 5

column_names=[]



for i in range(len_features):

    column_names.append('w_'+str(i))



fh = FeatureHasher(n_features=len_features, input_type='string')

hashed_features = fh.fit_transform(wine['winery'])

hashed_features = hashed_features.toarray()

hashed_features = pd.DataFrame(hashed_features,columns=column_names)

wine = pd.concat([wine,hashed_features],axis=1)

wine.drop('winery',axis=1,inplace=True)
test = wine.loc[wine['price'].isnull()]

train = wine.loc[wine['price'].isnull()==False]
outliers = train.loc[abs(zscore(train['price']))>2,:].index

train.drop(outliers,inplace=True)
X = train.drop('price',axis=1)

y = train['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
y_mean = y_train.mean()
y_mean_list = [y_mean]*len(y_test)
np.sqrt(metrics.mean_squared_error(y_test, y_mean_list))
test = wine.loc[wine['price'].isnull()]

train = wine.loc[wine['price'].isnull()==False]
outliers = train.loc[abs(zscore(train['price']))>2,:].index

train.drop(outliers,inplace=True)
X = train.drop('price',axis=1)

y = train['price']
#X = PolynomialFeatures(degree=2,include_bias=True).fit_transform(X).astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
lm = LinearRegression()

lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
plt.scatter(list(range(len(y_test))),y_test-predictions)
sns.distplot((y_test-predictions),bins=50);
np.sqrt(metrics.mean_squared_error(y_test, predictions))
wine_desc.head()
wine_desc = pd.concat([wine_desc,wine['price']],axis=1)
def text_process(mess):

    nopunc = [char for char in mess if char not in string.punctuation]

    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
bow_transformer = CountVectorizer(analyzer=text_process).fit(wine_desc['description'])
descriptions_bow = bow_transformer.transform(wine_desc['description'])
descriptions_bow.shape
tfidf_transformer = TfidfTransformer().fit(descriptions_bow)
descriptions_tfidf = tfidf_transformer.transform(descriptions_bow)
print(descriptions_tfidf[0])
train_indices = wine_desc.loc[wine_desc['price'].isnull()==False].index
descriptions_tfidf
y = wine_desc.loc[train_indices,'price']
X = descriptions_tfidf[list(train_indices)]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
lm = LinearRegression()

lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, predictions))
svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
X_svd = svd.fit_transform(X)
X_svd.shape
X_train, X_test, y_train, y_test = train_test_split(X_svd, y, test_size=0.2, random_state=100)
lm = LinearRegression()

lm.fit(X_train,y_train)

predictions = lm.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test, predictions))