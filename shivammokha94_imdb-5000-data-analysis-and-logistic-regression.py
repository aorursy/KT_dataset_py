import pandas as pd                     #Load libraries

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from pandas import Series

from scipy import stats

import requests, re

from scipy.stats import norm

from sklearn import preprocessing

from sklearn.preprocessing import minmax_scale

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.feature_selection import RFE

from sklearn.linear_model import Ridge

import os
#Data cleaning and exploration

print(os.listdir("../input"))
movie_data = pd.DataFrame(pd.read_csv("../input/imdb-5000-movie-dataset/movie_metadata.csv"))  #Load data
movie_data.head()     #Check data
movie_data.shape      #Check number of rows and columns
movie_data.info()  #Information about movie columns
movie_data.isna().sum()        #Null values
movie_data.groupby('color')['color'].count()    #Distribution check
movie_data['color'] = movie_data['color'].fillna('Color')         #Fill null values
movie_data['director_name'] = movie_data['director_name'].fillna('Unknown/Multiple')    #Fill null values
duration = movie_data[~movie_data['duration'].isna()]      #Considering only complete data for duration
plt.boxplot(duration['duration'])   #Boxplot
duration['duration'].describe()     #Stats for duration column
movie_data['duration'] = movie_data['duration'].fillna(duration['duration'].mean())   #Fill null values with mean
movie_data['num_critic_for_reviews'] = movie_data['num_critic_for_reviews'].fillna(0)

movie_data['director_facebook_likes'] = movie_data['director_facebook_likes'].fillna(0)    #Fill null values with 0

movie_data['actor_1_facebook_likes'] = movie_data['actor_1_facebook_likes'].fillna(0)

movie_data['actor_2_facebook_likes'] = movie_data['actor_2_facebook_likes'].fillna(0)

movie_data['actor_3_facebook_likes'] = movie_data['actor_3_facebook_likes'].fillna(0)

movie_data['facenumber_in_poster'] = movie_data['facenumber_in_poster'].fillna(0)

movie_data['num_user_for_reviews'] = movie_data['num_user_for_reviews'].fillna(0)
movie_data['actor_1_name'] = movie_data['actor_1_name'].fillna('Unknown')

movie_data['actor_2_name'] = movie_data['actor_2_name'].fillna('Unknown')     #Fill null values

movie_data['actor_3_name'] = movie_data['actor_3_name'].fillna('Unknown')
movie_data.groupby('language')['language'].count()     #Count of languages
movie_data['language'] = movie_data['language'].fillna('English')   #Fill null values
movie_data.groupby('country')['country'].count()        #Check distribution
movie_data['country'] = movie_data['country'].fillna('USA')  #Fill null values
movie_data['title_year'].hist()       #Check distribution
movie_data = movie_data[~movie_data['title_year'].isna()]  #Most rows with empty title year are documentries or series 

#Most rows with empty title year do not have gross or budget
movie_data.groupby('content_rating')['content_rating'].count()  #Check distribution
movie_data['content_rating'] = movie_data['content_rating'].fillna('Unrated') #Fill null values
del movie_data['aspect_ratio']         #Remove unecessary column
movie_data.isna().sum()      #Null value check
movie_data.shape     #Check rows and columns left
def link_replacer(link):             #Replace movies URL with IMDB URL

    link = link.replace('http://www.movie.com/title/', 'https://www.imdb.com/title/')

    link = link.replace('/?ref_=fn_tt_tt_1', '/')

    return link





def budget_extractor(link):           #Webscrape data from IMDB to get budget value  

    

    print(link)

    r = requests.get(link)

    if re.search('<h4 class="inline">Budget:</h4>(.*?)<span class="attribute">',r.text, re.DOTALL) is None:

        return np.NAN

    content = re.search('<h4 class="inline">Budget:</h4>(.*?)<span class="attribute">',r.text,re.DOTALL).group(1)

    content = re.sub('\W+','', content)

    content = re.search(r'\d+', content).group()

    content = int(content)

    return content





def gross_extractor(link):         #Webscrape data from IMDB to get gross value

    print(link)

    r = requests.get(link)

    if re.search('<h4 class="inline">Gross USA:</h4>(.*?)</div>',r.text, re.DOTALL) is None:

                                                    #Check if Gross USA values exists

        if re.search('<h4 class="inline">Cumulative Worldwide Gross:</h4>(.*?)</div>',r.text, re.DOTALL) is None:

            return np.NAN                       #Check if Cummulative Gross values exists

        else:

            content = re.search('<h4 class="inline">Cumulative Worldwide Gross:</h4>(.*?)</div>',r.text,re.DOTALL).group(1)

            content = re.sub('\W+','', content)

            content = re.search(r'\d+', content).group()

            content = int(content)

            return content

    else:

        content = re.search('<h4 class="inline">Gross USA:</h4>(.*?)</div>',r.text,re.DOTALL).group(1)

        content = re.sub('\W+','', content)

        content = re.search(r'\d+', content).group()

        content = int(content)

        return content

#movie_data['movie_imdb_link'] = movie_data['movie_imdb_link'].apply(lambda x: link_replacer(x))



#gross = movie_data[~movie_data['gross'].isna()]

#nt_gross = movie_data[movie_data['gross'].isna()]

#nt_gross['gross'] = nt_gross['movie_imdb_link'].apply(lambda x: gross_extractor(x))

#mv1 = pd.concat([gross, nt_gross], ignore_index = True)



#budget = mv1[~mv1['budget'].isna()

#nt_budget = mv1[mv1['budget'].isna()]

#nt_budget['budget'] = nt_budget['movie_imdb_link'].apply(lambda x: budget_extractor(x))

#mv2 = pd.concat([budget, nt_budget], ignore_index = True)



#movie_data = mv2



#movie_data.isna().sum()
#movie_data.to_csv('movie_cln_v4.csv')     #Save data
movie_data = pd.DataFrame(pd.read_csv('../input/movie-clean/movie_cln_v5.csv')) #Read clean dataset
movie_data = movie_data.drop(columns=['color', 'Unnamed: 0'])   #Remove unnecessary columns
movie_data.shape
movie_data.isna().sum()
movie_data = movie_data.dropna(subset = ['gross', 'budget'], how = 'all')       #Drop null which exist in both gross AND budget
movie_data.shape     #Check cleaned dataset
movie_data_nn = movie_data.dropna(subset = ['gross'], how = 'all')

movie_data_nn = movie_data_nn.dropna(subset = ['budget'], how = 'all')
movie_data_nn.shape       #Dataframe with no null values
movie_data_nn.isna().sum()
movie_data_nn.to_csv('movie_data_no_null.csv')
movie_data.to_csv('movie_cln_v5.csv')
movie_data_nn['net'] = movie_data_nn['gross'] - movie_data_nn['budget']
movie_data_nn = pd.DataFrame(pd.read_csv('/Users/smokha/Downloads/DSCS/movie_data_no_null.csv'))
movie_data_nn.info()
#Plotting heat map:

plt.figure(figsize=(18,8),dpi=100,)

plt.subplots(figsize=(18,8))

sns.heatmap(data=movie_data.corr(),square=True,vmax=0.8,annot=True)
#Plotting heat map for all non null values:

plt.figure(figsize=(18,8),dpi=100,)

plt.subplots(figsize=(20,10))

sns.heatmap(data=movie_data_nn.corr(),square=True,vmax=0.8,annot=True)
sns.distplot(a=movie_data_nn['movie_facebook_likes'],hist=True,bins=10,fit=norm,color="red")

plt.title("IMDB Movie Review")

plt.ylabel("frequency")         #Distribution of movie Facebook likes

plt.show()
mean, variance=norm.fit(movie_data_nn['movie_facebook_likes'])         
mean
variance
col_n = ['num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross', 'num_voted_users', 'cast_total_facebook_likes', 'facenumber_in_poster', 'num_user_for_reviews', 'title_year', 'actor_2_facebook_likes', 'movie_score', 'movie_facebook_likes', 'net']
plt.boxplot(movie_data_nn['net']) #Distribution of net 
movie_data_nn.info()
sns.jointplot(x=(movie_data_nn['budget']),y=(movie_data_nn['gross']),kind="reg", height=8.27) #Budget V/S Gross
data = movie_data_nn.drop(movie_data_nn[(movie_data_nn['budget']>200000000)].index).reset_index(drop=True)  #Remove outliers
sns.jointplot(x=(data['budget']),y=(data['gross']),kind="reg", height=8.27)   #Higher budget can mean higher gross
data['budget'].corr(data['gross'])
data['net'] = data['gross'] - data['budget']
data['budget'].corr(data['net'])
sns.jointplot(x=(data['budget']),y=(data['net']),kind="reg", height=8.27)   #Budget does not have an effect on net profit or loss
#Plotting heat map for all non null values:

plt.figure(figsize=(18,8),dpi=100,)

plt.subplots(figsize=(20,10))

sns.heatmap(data=data.corr(),square=True,vmax=0.8,annot=True)
sns.jointplot(x=(data['gross']),y=(data['net']),kind="reg", height=8.27)   #Higher budget can mean higher gross
movie_data_sp = data      #New dataframe to split genres
# Split genres and create a new entry for each of the genre a movie falls into

s = movie_data_sp['genres'].str.split('|').apply(Series, 1).stack()

s.index = s.index.droplevel(-1)

s.name = 'genres'

del movie_data_sp['genres']

md_split_genres = movie_data_sp.join(s)
md_split_genres.info()
md_split_genres.groupby('genres')['genres'].count()
md_split_genres['genres_cat'] = md_split_genres['genres'].astype('category').cat.codes
data_3 = data[data['title_year']>1950]
sns.relplot(x="title_year", y="budget", kind="line", ci="sd", data=data_3, height=8.27);
sns.countplot(x="genres_cat", data=md_split_genres, color="c");
data.info()
sns.jointplot(x=(data['cast_total_facebook_likes']),y=(data['gross']),kind="reg", height=8.27)
sns.jointplot(x=(data_2['cast_total_facebook_likes']),y=(data_2['gross']),kind="reg", height=8.27)
sns.jointplot(x=(data_3['net']),y=(data_3['num_voted_users']),kind="reg", height=8.27)
#Plotting heat map for all non null values:

plt.figure(figsize=(18,8),dpi=100,)

plt.subplots(figsize=(20,10))

sns.heatmap(data=data_3.corr(),square=True,vmax=0.8,annot=True)
sns.jointplot(x=(data_3['gross']),y=(data_3['num_voted_users']),kind="reg", height=8.27)
sns.jointplot(x=(data_3['net']),y=(data_3['num_user_for_reviews']),kind="reg", height=8.27)
#Hypothesis       -- We define a block buster movie as follows,

#For a movie with a budget less than $5million, gross over $2million and having a profit greater than 500% of net

#For a movie with a budget more than $5mil having a profit greater than 150% of net
movie_data_nn.isna().sum()   #Check null values
movie_data_nn['net'] = (movie_data_nn['gross'] - movie_data_nn['budget'])/movie_data_nn['budget']  #Create net column

movie_data_nn['blockbuster'] = 0   
movie_data_nn.sort_values(by=['budget'])
#Removing outliers

movie_data_nn = movie_data_nn.drop(movie_data_nn[(movie_data_nn['budget']>200000000)].index).reset_index(drop=True)
movie_data_nn['budget'].hist()
mov_l = movie_data_nn[movie_data_nn['budget']<5000000]   #Split data
mov_m = movie_data_nn[movie_data_nn['budget']>=5000000]        #Split data
mov_l.sort_values(by=['net'])
mov_l_ntg = mov_l[mov_l['gross']<2000000]

mov_l_g = mov_l[mov_l['gross']>=2000000]   #Split data
mov_l_g['blockbuster'] = mov_l_g['net'].apply(lambda x: 1 if x>=5 else 0)   #Setting blockbuster condition
mov_l = pd.concat([mov_l_g, mov_l_ntg])
mov_l.groupby('blockbuster')['blockbuster'].count()
mov_m['blockbuster'] = mov_m['net'].apply(lambda x: 1 if x>=1.5 else 0)  #Setting blockbuster condition
mov_m.groupby('blockbuster')['blockbuster'].count()
movie_data_nn = pd.concat([mov_l, mov_m])
movie_data_nn.groupby('blockbuster')['blockbuster'].count()
movie_data_nn = movie_data_nn.drop(columns=['director_name', 'actor_2_name', 'actor_1_name', 'actor_3_name', 'movie_title'], axis = 1)   #Drop unecessary columns
movie_data_nn.info()
sns.distplot(movie_data['title_year'])
movie_data_nn['year_bin'] = pd.cut(movie_data_nn['title_year'], 20)   #Creating bins for title year
movie_data_nn.groupby('year_bin')['year_bin'].count()
movie_data_nn.info()
movie_data_nn['language'] = movie_data_nn['language'].apply(lambda x: 'English' if x =='English' else 'Others')  #Convert data
def country(x):    #Function to convert country data

    if x != 'UK' and x != 'Canada' and x != 'Australia' and x != 'France' and x != 'Germany' and x != 'USA':

        x = 'Others'  

    return x



movie_data_nn['country'] = movie_data_nn['country'].apply(lambda x: country(x))
def rating(x):    #Function to convert country data

    if x != 'PG-13' and x != 'PG' and x != 'G' and x != 'Not Rated' and x != 'Unrated' and x != 'R':

        x = 'Others' 

        

    if x == 'Not Rated':

        x = 'Unrated'

    return x



movie_data_nn['content_rating'] = movie_data_nn['content_rating'].apply(lambda x: rating(x))
movie_data_nn['duration_bins'] = pd.cut(movie_data_nn['duration'], 12)
# Split genres and create a new entry for each of the genre a movie falls into

s = movie_data_nn['genres'].str.split('|').apply(Series, 1).stack()

s.index = s.index.droplevel(-1)

s.name = 'genres'

del movie_data_nn['genres']

movie_data_nn = movie_data_nn.join(s)
movie_data_nn.info()
cols_ohc = ['content_rating', 'language', 'country', 'genres', 'duration_bins', 'year_bin']
md1 = pd.get_dummies(movie_data_nn, prefix_sep="__", columns=cols_ohc)  #Creating dummy variables
md1 = md1.drop(columns=['duration', 'title_year', 'net'], axis=1)  #Drop unecessary columns
md1['gross'] = md1['gross']/1000000      #Scaling down values

md1['budget'] = md1['budget']/1000000
y = md1.pop('blockbuster')       #Sepereate feature matrix and output

x = md1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.45, random_state = 1)    #Train test split
logreg=LogisticRegression()  
logreg.fit(x_train,y_train)             #Fit logistic regression to train data
predictions = logreg.predict(x_test)       #Logistic regression prediction
score = logreg.score(x_test, y_test)  #Prediction Score

print(score)
cm = metrics.confusion_matrix(y_test, predictions)       #Confusion matrix

print(cm)  
import statsmodel.api as sm   #Tool not present on kernel, but required for OLS regression and F-statistics
X2 = sm.add_constant(x)    #OLS regression to get F-statistics and t-statistics

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())
x_train.shape
##Logistic regression without 'Gross' variable

x.pop('gross')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.45, random_state = 1)
logreg.fit(x_train,y_train)

predictions = logreg.predict(x_test)

score = logreg.score(x_test, y_test)

print(score)
cm = metrics.confusion_matrix(y_test, predictions)  #Confusion matrix

print(cm)
X2 = sm.add_constant(x)   #OLS regression to get F-statistics and t-statistics

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())
from sklearn.decomposition import PCA
x2 = x       #PCA for dimensionality reduction

pca = PCA(n_components=3)

x2 = pca.fit_transform(x)
x4 = sm.add_constant(x2)

est = sm.OLS(y, x4)   #OLS regression to get F-statistics and t-statistics

est2 = est.fit()

print(est2.summary())
ridge = Ridge(alpha=1.0)

ridge.fit(x,y)    #Ridge regression for diamensionality regression
# A helper method for pretty-printing the coefficients

def pretty_print_coefs(coefs, names = None, sort = False):

    if names == None:

        names = ["X%s" % x for x in range(len(coefs))]

    lst = zip(coefs, names)

    if sort:

        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))

    return " + ".join("%s * %s" % (round(coef, 3), name)

                                   for coef, name in lst)



print ("Ridge model:", pretty_print_coefs(ridge.coef_))
from sklearn.feature_selection import SelectKBest, chi2
test = SelectKBest(score_func=chi2, k=4)   #KBest regression for diamensionality regression

fit = test.fit(x,y)
print(fit.scores_)