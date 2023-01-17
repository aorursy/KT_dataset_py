import numpy as np

import pandas as pd

pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('ggplot')

import datetime

from scipy import stats

from scipy.sparse import hstack, csr_matrix

from wordcloud import WordCloud

from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.preprocessing import StandardScaler

#stop = set(stopwords.words('english'))

import os

#py.init_notebook_mode(connected=True)

import xgboost as xgb

#import lightgbm as lgb

from sklearn import model_selection

from sklearn.metrics import accuracy_score

import json

import ast

#from catboost import CatBoostRegressor

from urllib.request import urlopen



from PIL import Image

from sklearn.preprocessing import LabelEncoder

import time

import datetime

train = pd.read_csv(r'C:\Users\HP\Downloads\train_jup.csv')



test = pd.read_csv('C:/Users/HP/Downloads/test.csv/test.csv')
train1 = pd.read_csv(r'C:\Users\HP\Downloads\train_jup.csv')

pop1=train1['popularity']
train.head()
train.shape, test.shape

# the train has 3000 rows and

# 23 columns 
train.columns

(train.describe(include='all'))
test.describe(include='all')
#to see the missing value percent

missing_data=train.isna()

missing_data_test=test.isna()

missing_data=missing_data.sum()

missing_data_test=missing_data_test.sum()

missing_data_percent=missing_data/len(train)*100

missing_data_test_percent=missing_data_test/len(test)*100

plt.figure(figsize=(16,6))

#plt.yticks(np.arange(1, 11))

missing_data_percent.plot.bar()

plt.title("Missing Data Distribution In Train Data")

plt.ylabel("Missing Data Percent", fontsize=20)

plt.xlabel("Columns", fontsize=18)

plt.show()
plt.figure(figsize=(16,6))

#plt.yticks(np.arange(1, 11))

missing_data_test_percent.plot.bar()

plt.title("Missing Data Distribution In Test Data")

plt.ylabel("Missing Data Percent", fontsize=20)

plt.xlabel("Columns", fontsize=18)

plt.show()
# Both test and train data has like almost same missing features and both are need to be care
plt.figure(figsize=(15,6))

plt.scatter(range(train.shape[0]), np.sort(train.revenue.values))

plt.title("Revenue Distribution", fontsize=22)

plt.xlabel('Index', fontsize=12)

plt.ylabel('Revenue', fontsize=12)

plt.show()
for i,e in enumerate(train['revenue']):

    if(e<5):

        print(i)

        print(e)

train['revenue'].describe()

# From this result we ge there are some outliers in revenue, also the mean is like 6 times greater 

# the median this shows the data is positively skewed

#hence we take the log of revenue to remove the skewness
train['log_revenue']=train['revenue'].apply(lambda x: np.log(x))

plt.figure(figsize=(15,6))

plt.scatter(range(train.shape[0]), np.sort(train.log_revenue.values), color='m')

plt.title("LOGRevenue Distribution", fontsize=22)

plt.xlabel('Index', fontsize=16, color='k')

plt.ylabel('Log Revenue', fontsize=18, color='k')

plt.xticks(color='k')

plt.show()
for val in train['belongs_to_collection'][:5]:

    print(val)
#so we infer two things in important in collection, one is collection name and 

# other things is that whether it may have any collection or not

train.loc[train['belongs_to_collection'].isnull(),['belongs_to_collection']] = train.loc[train['belongs_to_collection'].isnull(),'belongs_to_collection'].apply(lambda x: {})

#train['belongs_to_collection']= train['belongs_to_collection'].apply(lambda x: {} if x )

for val in train['belongs_to_collection'][:5]:

    print(val)

# same action on test data too:

test.loc[test['belongs_to_collection'].isnull(),['belongs_to_collection']] = test.loc[test['belongs_to_collection'].isnull(),'belongs_to_collection'].apply(lambda x: {})
#we need to do similar operation with other features also namely: genres, production_company, 

# production country 

dictionary_features=['genres', 'production_companies', 'production_countries', 

                     'spoken_languages','Keywords','cast', 'crew']

def text_to_dict(df):

    for column in dictionary_features:

        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )

    return df

text_to_dict(train)

text_to_dict(test)
train['has_collection'] = train['belongs_to_collection'].apply(lambda x: 1 if x != {} else 0)

#print(train['has_collection'])

plt.figure(figsize=(10,6))

plt.title("has_collection")

sns.swarmplot(x=train['has_collection'], y=train['revenue'])

# clearly the movie with the collection is earning more than without a collection
test['has_collection'] = test['belongs_to_collection'].apply(lambda x: 1 if x != {} else 0)
plt.figure(figsize=(10,6))

plt.subplot(1,2,1)

plt.title("Budget effect on Revenue", fontsize=18)

sns.regplot(x=train['budget'], y=train['revenue'])

plt.ylabel("Revenue", color='k', fontsize=16)

plt.xlabel("Budget", color='k', fontsize=16)

plt.subplot(1,2,2)

plt.title("Budget effect on Log Revenue", fontsize=18)

sns.regplot(x=train['budget'], y=train['log_revenue'])

plt.ylabel("Revenue", color='k', fontsize=16)

plt.xlabel("Budget", color='k', fontsize=16)

# definetely budget is influency the movie revenue, clearly shown with the positive slope line

# also as we see earlier that many values in budget have 0 values that can't possible

plt.figure(figsize=(16,6))

plt.scatter(range(train.shape[0]), np.sort(train.budget.values))

plt.title("Budget Distribution over the train Data", fontsize=22)

plt.xlabel('index', fontsize=12)

plt.ylabel('budget', fontsize=12)
count=0

sum_budget=0

for i,e in enumerate(train['budget']):

    if e>100:

        sum_budget+=e

        count+=1

mean=sum_budget/count

train['budget_mean']=train['budget'].apply(lambda x: mean if x<10000 else x )

# this budget_mean can works as alternative to budget feature but still, its a weak assumption
train['runtime']=train['runtime'].apply(lambda x: train['runtime'].mean() if x<30 else x)

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)

sns.regplot(x=train['runtime'], y=train['revenue'])

plt.title("Runtime effect on Revenue")

plt.ylabel("Revenue", fontsize=12, color='k')

plt.xlabel("Runtime", fontsize=12, color='k')

plt.subplot(1,2,2)

sns.regplot(x=train['runtime'], y=train['log_revenue'])

plt.title("Runtime effect on LogRevenue")

plt.ylabel("Log Revenue", fontsize=12, color='k')

plt.xlabel("Runtime", fontsize=12, color='k')
plt.figure(figsize=(12,6))

sns.regplot(x=train['popularity'], y=train['revenue'])

plt.title("Popularity Effect on Revenue")

plt.xlabel("Popularity")

plt.ylabel("Reveneue")
train['release_date']

# by printing we get to know that like 1987 is given as 87 in year,so we need to manipualte 

#this attribute

train[['release_month','release_day','release_year']]=train['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)

# now the time to correct that mistake:

# in case of missing value we can't do anything so we replace the missing value by "-1"

train['release_year']=train['release_year'].apply(lambda x: x+2000 if x<19 else x+1900)



# Now pandas has some amazing feature regarding the manipulation of date into the weekday at that 

# particular date

releaseDate = pd.to_datetime(train['release_date'])

train['weekday']=releaseDate.dt.dayofweek

train['release_quarter'] = releaseDate.dt.quarter

dict_day={0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}

dict_month={1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr",5: "May",6: "Jun",7: "Jul",8: "Aug",9: "Sep",10: "Oct",11: "Nov",12: "Dec"}

train['weekday']=train['weekday'].apply(lambda x: dict_day[x])

train['release_month']=train['release_month'].apply(lambda x: dict_month[x])
test[['release_month','release_day','release_year']]=test['release_date'].str.split('/',expand=True).replace(np.nan, -1).astype(int)

test['release_year']=test['release_year'].apply(lambda x: x+2000 if x<19 else x+1900)
plt.figure(figsize=(12,6))

sns.barplot(x=train['weekday'], y=train['revenue'])

plt.xlabel("WeekDay")

plt.ylabel("Revenue")

# the result showing that those movies release on wednesday earn much higher than other weekdays
plt.figure(figsize=(9,6))

plt.title("Weekday",fontsize=20)

sns.countplot(train['weekday'])

# as expected most of the movies do release on Friday
plt.figure(figsize=(16,6))

plt.title("Release month revenue")

sns.barplot(x=train['release_month'], y=train['revenue'])
plt.figure(figsize=(16,6))

plt.title("Release Month Movie Count", fontsize=22)

sns.countplot(train['release_month'])

plt.xlabel("Release Month")

plt.ylabel("Movie Count")
plt.figure(figsize=(15,10))

plt.title("Movies count in a YEAR")

plt.xticks(np.arange(1920,2019,4), fontsize=12,rotation=90)

sns.countplot(train['release_year'])
plt.figure(figsize=(16,6))

plt.title("Mean revenue per year")



mean_revenue_year=train.groupby('release_year')['revenue'].aggregate('mean')

plt.xticks(np.arange(1921, 2018, 5), rotation=90)

plt.xlabel("Release Year")

plt.ylabel("Mean revenue")

sns.lineplot(data=mean_revenue_year)





# probably due to inflation
train['has_tagline']=train['tagline'].isna()

(train['has_tagline'])=train['has_tagline'].apply(lambda x: 1 if x==False else 0)

test['has_tagline']=test['tagline'].isna()

(test['has_tagline'])=test['has_tagline'].apply(lambda x: 1 if x==False else 0)

#print(train['has_tagline'])

plt.figure(figsize=(16,6))

plt.title("Tagline Effect")

sns.barplot(x=train['has_tagline'], y=train['revenue'])

plt.xlabel("Tagline")

plt.ylabel("Revenue")
plt.figure(figsize=(12,12))

text = ' '.join(train['tagline'].fillna('').values)

my_cloud=WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)

plt.imshow(my_cloud)

plt.title('Top words in Tagline')

plt.axis("off")

plt.show()
def str_length(x):

    return len(x)

train['has_homepage']=train['homepage'].fillna('').values

train['has_homepage']=train['has_homepage'].apply(lambda x : 1 if str_length(x)>0 else 0)

test['has_homepage']=test['homepage'].fillna('').values

test['has_homepage']=test['has_homepage'].apply(lambda x : 1 if str_length(x)>0 else 0)

plt.figure(figsize=(10,4))

sns.barplot(x=train['has_homepage'], y=train['revenue'])

plt.xlabel("Has Homepage")

plt.ylabel("Revenue")


plt.figure(figsize=(25,10))

plt.rcParams.update({'font.size': 22})

sns.countplot(np.sort(pd.concat([train, test]).original_language.values))

plt.ylabel("Count")

plt.xlabel("Language")

plt.title("Original Language Count in Full Data",fontsize=30)



plt.show()
newDF = pd.DataFrame(pd.concat([train, test], sort=False).original_language.value_counts())

newDF.head(n=10)
train['original_language'].head()

train["original_language"].describe(include='all')

plt.figure(figsize=(16,12))

plt.xticks(fontsize=14)

sns.barplot(x=train['original_language'], y=train['revenue'])

plt.xlabel("Original Language")

plt.ylabel("Revenue")
plt.figure(figsize=(15,6))

sns.boxplot(x='original_language', y='revenue', data=train.loc[train['original_language'].isin(train['original_language'].value_counts().head(10).index)])

plt.title("Mean Revenue per Language", fontsize=20)

plt.xticks(fontsize=14)

plt.xlabel("Language")

plt.ylabel("Revenue")
plt.figure(figsize=(12,6))

sns.boxplot(x='original_language', y='log_revenue', data=train.loc[train['original_language'].isin(train['original_language'].value_counts().head(10).index)])

plt.title("Mean LogRevenue per Language")

plt.ylabel("Mean Log Revenue")

plt.xlabel("Original Language")
train['has_english']=train['original_language'].apply(lambda x: "english" if x=='en' else "non-english")

test['has_english']=test['original_language'].apply(lambda x: "english" if x=='en' else "non-english")

plt.figure(figsize = (16, 6))

plt.subplot(1,2,1)

#plt.title("english language effect")

sns.boxplot(x=train['has_english'], y=train['revenue'])

plt.xlabel("English")

plt.ylabel("Revenue")

plt.subplot(1,2,2)

#plt.title("english language on median revenue")

sns.barplot(x=train['has_english'], y=train['revenue'])

plt.xlabel("Non-English")

plt.ylabel("Revenue")


train['length_original_title']=train['original_title'].apply(lambda x: len(x) )

plt.figure(figsize=(12,6))

plt.title("Title Length vs LOG Revenue")

sns.regplot(x=train["length_original_title"], y=train['log_revenue'])

plt.xlabel("Title Length")

plt.ylabel("Revenue")



# we can drop this variable since it does'nt affect revenue


train['genres'].apply(lambda x: len(x) if x != {} else 0).value_counts()


list_of_genres = list(train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

plt.figure(figsize=(12,10))

text=" ".join( i for j in list_of_genres for i in j )
plt.figure(figsize=(12,10))

plt.title("Common Genre")

my_cloud=WordCloud(max_font_size=None, background_color='black', width=1200, height=1000).generate(text)

plt.imshow(my_cloud)

plt.title('Top Words in Genre')

plt.axis("off")

plt.show()
text_list=[]

for j in list_of_genres:

    for i in j:

        text_list.append(i)

print(Counter(text_list))
D=Counter(text_list)

plt.figure(figsize=(18,6))

plt.bar(range(len(D)), D.values(), align='center')

plt.xticks(range(len(D)), list(D.keys()), fontsize=10)

plt.title("Genre Count")

plt.ylabel("Count")

plt.xlabel("Genre")

plt.show()


#train[train["Electrical"].isnull()][null_columns]

def count(x):

    count_cast=0

    for cast_inf in x:

        count_cast+=1

    return count_cast

train['no_of_cast']=train['cast'].apply(lambda x : count(x) )

test['no_of_cast']=test['cast'].apply(lambda x: count(x))

plt.figure(figsize=(16,6))

plt.subplot(1,2,1)

plt.title("NUMBER OF CAST VS REVENUE", fontsize=18)

sns.regplot(x=train['no_of_cast'], y=train['revenue'])

plt.xlabel("Number of cast")

plt.ylabel("Revenue")

plt.subplot(1,2,2)

plt.title("NUMBER OF CAST VS BUDGET", fontsize=18)

sns.regplot(x=train['no_of_cast'], y=train['budget'])

plt.xlabel("Number of cast")

plt.ylabel("Budget")
train['cast'][3]

gender={1:'female', 2:'male', 0:'other'}

def count_f(x):

    count=0

    for ch in x:

        if(ch['gender']==1):

            count+=1

    return count

train['female_cast']=train['cast'].apply(lambda x: count_f(x) )

def count_m(x):

    count=0

    for ch in x:

        if(ch['gender']==2):

            count+=1

    return count

train['male_cast']=train['cast'].apply(lambda x: count_m(x) )

plt.figure(figsize=(16,6))

plt.subplot(1,2,1)

plt.title('Female CAST Effect on revenue', fontsize=18)

sns.regplot(x=train['female_cast'], y=train['revenue'])

plt.xlabel("Female Cast")

plt.ylabel("Revenue")

plt.subplot(1,2,2)

plt.title("Male CAST Effect on Revenue", fontsize=18)

sns.regplot(x=train['male_cast'], y=train['revenue'])

plt.xlabel("Male Cast")

plt.ylabel("Revenue")
train['production_companies'].head()

production_company_list=list(train['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

company_list=[]

for i in production_company_list:

    for j in i:

        company_list.append(j)

print(Counter(company_list))

# take 'Warner Bros.': 202, 'Universal Pictures': 188, 'Paramount Pictures': 161, 'Twentieth Century Fox Film Corporation': 138

# , 'Columbia Pictures': 91, 'Metro-Goldwyn-Mayer (MGM)': 84, 'New Line Cinema': 75, 'Touchstone Pictures': 63, 

# 'Walt Disney Pictures': 62,  'Columbia Pictures Corporation': 61  as big production companies

company_text=' '

for i in range(len(company_list)):

    company_text=company_text+' '+company_list[i]





plt.figure(figsize=(12,10))

my_cloud=WordCloud(max_font_size=None, background_color='black', width=1200, height=1000).generate(company_text)

plt.imshow(my_cloud)

plt.title('Top Company')

plt.axis("off")

plt.show()
def big_producer(x):

    big_company_list=['Warner Bros.', 'Universal Pictures', 'Paramount Pictures', 'Twentieth Century Fox Film Corporation', 

        'Columbia Pictures', 'Metro-Goldwyn-Mayer (MGM)', 'New Line Cinema', 'Touchstone Pictures', 'Walt Disney Pictures']

    for i in x:

        if(i['name']==big_company_list[0]):

            return True

        elif(i['name']==big_company_list[1]):

            return True

        elif(i['name']==big_company_list[2]):

            return True

        elif(i['name']==big_company_list[3]):

            return True

        elif(i['name']==big_company_list[4]):

            return True

        elif(i['name']==big_company_list[5]):

            return True

        elif(i['name']==big_company_list[6]):

            return True

        elif(i['name']==big_company_list[7]):

            return True

        elif(i['name']==big_company_list[8]):

            return True

        else:

            return False



train['company_size']=train['production_companies'].apply(lambda x: 1 if big_producer(x) else 0)

plt.figure(figsize=(16,6))

plt.subplot(1,2,1)

sns.boxplot(x=train['company_size'], y=train['revenue'])

plt.xticks(range(2),('Small Producer', 'Big Producer'))

plt.subplot(1,2,2)

sns.barplot(x=train['company_size'], y=train['revenue'])

plt.xticks(range(2),('Small Producer', 'Big Producer'))

def count(x):

    cnt=0

    for ch in x:

        cnt+=1

    return cnt

train['no_of_pr_companies']=train['production_companies'].apply(lambda x: count(x))

train['no_of_production_countries']=train['production_countries'].apply(lambda x: count(x) )

train['no_of_spoken_languages']=train['spoken_languages'].apply(lambda x: count(x))

test['no_of_pr_companies']=test['production_companies'].apply(lambda x: count(x))

test['no_of_production_countries']=test['production_countries'].apply(lambda x: count(x) )

test['no_of_spoken_languages']=test['spoken_languages'].apply(lambda x: count(x))



plt.figure(figsize=(16,6))

plt.subplot(1,3,1)

sns.boxplot(x=train['no_of_pr_companies'], y=train['revenue'])

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.xlabel("Number of Production Companies", fontsize=14)

plt.ylabel("Revenue", fontsize=14)

plt.subplot(1,3,2)

sns.boxplot(x=train['no_of_production_countries'], y=train['revenue'])

plt.xlabel("Number of Production Countries", fontsize=14)

plt.ylabel("Revenue", fontsize=14)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.subplot(1,3,3)

sns.boxplot(x=train['no_of_spoken_languages'], y=train['revenue'])

plt.xlabel("Number of Spoken Languages", fontsize=14)

plt.ylabel("Revenue", fontsize=14)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)
plt.figure(figsize=(16,6))

plt.subplot(1,3,1)

sns.regplot(x=train['no_of_pr_companies'], y=train['revenue'])

plt.xlabel("Number of Production Companies", fontsize=14)

plt.ylabel("Revenue", fontsize=14)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.subplot(1,3,2)

sns.regplot(x=train['no_of_production_countries'], y=train['revenue'])

plt.xlabel("Number of Production Countries", fontsize=14)

plt.ylabel("Revenue", fontsize=14)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.subplot(1,3,3)

sns.regplot(x=train['no_of_spoken_languages'], y=train['revenue'])

plt.xlabel("Number of Spoken Languages", fontsize=14)

plt.ylabel("Revenue", fontsize=14)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)


train['no_of_crew']=train['crew'].apply(lambda x: len(x) )

test['no_of_crew']=test['crew'].apply(lambda x: len(x) )



plt.figure(figsize=(10,6))

plt.title("Total crew vs log Revenue")

sns.regplot(x=train['no_of_crew'], y=train['log_revenue'])

plt.xlabel("Number of Crew")

plt.ylabel("Log Revenue")

#np.where(pd.isnull(train['budget']))


cor_features = train[['revenue', 'budget',  'popularity', 'runtime', 'log_revenue',

            'release_day', 'length_original_title','no_of_crew', 'release_year','no_of_cast','has_tagline','has_homepage','female_cast','male_cast','no_of_pr_companies','no_of_production_countries','no_of_spoken_languages']]

f,ax = plt.subplots(figsize=(20, 16))

sns.heatmap(cor_features.corr(), annot=True, linewidths=.7, fmt= '.2f',ax=ax)

plt.show()

# Release quarter, number of production countries ,release day,  number of spoken languages shows a very little corelation with revenue,

# we can remove these from our attribute list.
null_columns=train.columns[train.isnull().any()]

train[null_columns].isnull().sum()

# this shows us that we need to fill 
null_columns=test.columns[test.isnull().any()]

test[null_columns].isnull().sum()
train['runtime'].fillna(train['runtime'].mean(), inplace=True)

test['runtime'].fillna(test['runtime'].mean(), inplace=True)

train['no_of_cast']=train['no_of_cast'].apply(lambda x: train['no_of_cast'].aggregate('mean'))

test['no_of_cast']=test['no_of_cast'].apply(lambda x: test['no_of_cast'].aggregate('mean'))

train['no_of_crew']=train['no_of_crew'].apply(lambda x: train['no_of_crew'].aggregate('mean'))

test['no_of_crew']=test['no_of_crew'].apply(lambda x: test['no_of_crew'].aggregate('mean'))
train['popularity']=pop1

train['norm_popularity']=train['popularity'].apply(lambda x: (x-train['popularity'].min())/(train['popularity'].max()-train['popularity'].min())*100000000)

#test['popularity']=test['popularity'].apply(lambda x: (x-0)/294.337037*100000000)

test['norm_popularity']=test['popularity'].apply(lambda x: (x-test['popularity'].min())/(test['popularity'].max()-test['popularity'].min())*100000000)

train['budget'].describe()

# we see that the budget column has like 25% data with misiing values, which is technically incorrect, so what we need is 

# to replace 0 values by some imputed value
cor_features.corr().iloc[1]

# so this shows the budget corelation with all the other variables, 

# now we can take popularity, release year, number of cast , number of production companies and the crew size to predict this.
test.columns
cor_features_test=test[['budget',  'popularity', 'runtime','release_day', 'release_year','no_of_cast','has_tagline','has_homepage','no_of_pr_companies','no_of_production_countries','no_of_spoken_languages']]

print(cor_features_test.corr().iloc[0])
features_need=['budget', 'popularity', 'no_of_pr_companies','no_of_cast', 'no_of_crew' ]

budget_data1=train[features_need]

budget_data2=test[features_need]

budget_data=pd.concat([budget_data1, budget_data2])

budget_data=budget_data.loc[budget_data['budget']>1000]









# now time for some training

X=budget_data[['popularity', 'no_of_pr_companies','no_of_cast', 'no_of_crew']]

y=budget_data['budget']

from sklearn.tree import DecisionTreeRegressor



# Define model. Specify a number for random_state to ensure same results each run

from sklearn.linear_model import LinearRegression

budget_model=LinearRegression()

budget_model.fit(X,y)

train_budget_incorrect_data=train.loc[train['budget']<=1000]

test_budget_incorrect_data=test.loc[test['budget']<=1000]

train_budget_correct=train.loc[train['budget']>1000]

test_budget_correct=test.loc[test['budget']>1000]
train['impute_budget']=train['budget']

test['impute_budget']=test['budget']



INDEX=train.loc[train['budget']<=1000].index

budget_imputer=budget_model.predict(train.loc[train['budget']<=1000][['popularity', 'no_of_pr_companies','no_of_cast', 'no_of_crew']])



for i,e in enumerate(INDEX):

    train.loc[e,'impute_budget']=budget_imputer[i]

    
INDEX_test=test.loc[test['budget']<=1000].index

budget_imputer_test=budget_model.predict(test.loc[test['budget']<=1000][['popularity', 'no_of_pr_companies','no_of_cast', 'no_of_crew']])

for i,e in enumerate(INDEX_test):

    test.loc[e,'impute_budget']=budget_imputer_test[i]