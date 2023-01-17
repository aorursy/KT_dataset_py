import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt





%matplotlib inline

#Data_Cleaning_App_Store
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
reviews = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv')

df = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
df.head()
reviews.head()
df.info()
df.describe()
df.shape
df.rename(columns={'Content Rating':'Content_Rating', 'Last Updated':'Last_Updated', 'Current Ver':'Current_Ver','Android Ver':'Android_Ver'}, inplace=True)
df.head(10)
df[df.Rating.isnull()].head()
df.Category.unique()
df[df.Category == '1.9']
df.drop(10472,axis=0,inplace=True)
df[df.Category == '1.9']
df.Reviews = df.Reviews.astype(int)
Applications = df[df.Rating.isnull()].App

reviews_applications = reviews.App.unique()

for i in Applications:

        if i in reviews_applications:

            print(i)
df[df.App == 'Blood Pressure']
df.iloc[2513,2]=df[df.App == 'Blood Pressure'].Rating.mean()
df[df.App == 'Blood Pressure']
drp = df[df.Rating.isnull()].index
drp = list(drp)

df.drop(drp, inplace = True)
df.sample(5)
lds = np.array(df.Size.str.find('k'))
ls = np.where(lds>0) #list of indexes where size is in kb
df.Size = df.Size.str.replace('M','');

df.Size = df.Size.str.replace('k','');
df_temp = df.copy()
df_temp.drop(df[df.Size == 'Varies with device'].index,axis=0, inplace=True)
df_temp.Size = df_temp.Size.astype('float') 
df.reset_index(False,inplace=True)
df.drop('index',axis=1, inplace=True)
for x,y in df.iterrows():

    if y.Size == 'Varies with device':

        val = y.Category

        df.iloc[x,4]=df_temp.groupby('Category').Size.median()[val]

df.Size = df.Size.astype('float')
df.sample()
for x in ls:

    df.iloc[x,4]=df.iloc[x,4]*0.001

    

#Size column is ready
df.Installs.unique()
df.Installs=df.Installs.str.replace('+','')

df.Installs=df.Installs.str.replace(',','')
df.Installs=df.Installs.astype('int')
df.sample()
df.Price.unique()
df.Price = df.Price.str.replace('$', '')
df.Price = df.Price.astype('float')
df.Price.unique()
df.sample(5)
#8535, 2017
temp = np.array(df.Genres.str.find(';'))

ls = np.where(temp>0) #Finding out the Indexes where there are multiple genres
df2 = df.iloc[np.r_[ls],:]  #Copying only those indexes to another dataframe to split them
df.shape
df2.shape
df.iloc[np.r_[1,4,9,25],:]
df.Genres = df.Genres.apply(lambda x: x.split(';')[0])
df2.head()
df2.Genres = df2.Genres.apply(lambda x: x.split(';')[1])
df2.head()
df = df.append(df2, ignore_index=True)
df.shape
del df2
df.info()
df.head()
df.Last_Updated = pd.to_datetime(df['Last_Updated']) 
df.info()
df.head()
ind = df[df['Current_Ver']=='Varies with device'].index
df.Current_Ver.mode()
df_t = df.copy()
df_t.drop(ind, axis=0, inplace=True)
df_t.Current_Ver.mode()
df.Current_Ver = df.Current_Ver.replace('Varies with device', '1.0')
df.head()
df.drop('Android_Ver', axis=1, inplace=True)
df[df.duplicated()]
df.drop_duplicates(keep='first', inplace=True)
df.head()
df.reset_index(inplace=True)

df.drop('index',axis=1, inplace=True)

df.head()
df.info()
df.describe()
df.corr()
df.head()
sns.distplot(df.Rating);
sns.boxplot(df.Rating);
df.Rating.skew()
q3, q1 = np.percentile(df.Rating, [75,25])

max = q3 + (1.5*(q3-q1))

min = q1 - (1.5*(q3-q1))
ls3=[]

for i,x in enumerate(df.Rating):

    if x < min:

        ls3.append(i)

        

len(ls3)
df.head(1)
df.Category.shape[0]
fig = plt.figure(figsize=(16,8)) 

plt.bar(df.Category.unique(), df.Category.value_counts());

plt.xticks(rotation=90);



for i,v in enumerate(df.Category.value_counts()):

    plt.text(i-0.40,v+10,round((v/df.Category.shape[0])*100,2))
df.Category.value_counts()
plt.bar(df.Type.unique(), df.Type.value_counts());



for i,v in enumerate(df.Type.value_counts()):

    plt.text(i-0.2,v-500,str(v)+" ("+ str(round((v/df.shape[0])*100,2))+'%)');
df.Content_Rating.unique()
plt.bar(df.Content_Rating.unique(), df.Content_Rating.value_counts())

plt.xticks(rotation=90);



for i,v in enumerate(df.Content_Rating.value_counts()):

    plt.text(i-0.25,v+20,str(str(round((v/df.shape[0])*100,2))+'%'))
df.Genres.value_counts()
fig = plt.figure(figsize=(16,8)) 

plt.bar(df.Genres.unique(), df.Genres.value_counts());

plt.xticks(rotation=90);

df.head()
sns.distplot(df.Reviews);
df.Reviews.skew()
df['Reviews_log'] = np.log(df.Reviews)
df.head()
df.drop('Reviews', axis=1, inplace=True)
df.Reviews_log.skew()
sns.distplot(df.Reviews_log);
sns.boxplot(df.Reviews_log);
sns.distplot(df.Size);
df.Size.skew()
df['Size_log'] = np.log(df.Size)
df.sample()
df.Size_log.skew()
df.drop('Size', axis=1, inplace=True)
sns.distplot(df.Size_log);
sns.boxplot(df.Size_log);
df.Installs.skew()
sns.distplot(df.Installs)
df['Installs_log'] = np.log(df.Installs)
df['Installs_log'].skew()
sns.boxplot(df['Installs_log'])
df.drop('Installs', axis=1, inplace=True)
df.head()
sns.distplot(df[df['Type']=='Paid'].Price);
sns.boxplot(df[df['Type']=='Paid'].Price);
df['Price_log'] = np.log(df[df['Type']=='Paid'].Price)
df.Price_log.fillna(0,inplace=True)
df.head()
sns.distplot(df[df['Type']=='Paid'].Price_log);

df[df['Type']=='Paid'].Price_log.skew()
sns.boxplot(df[df['Type']=='Paid'].Price_log);
df.head()
fig = plt.figure(figsize=(40,8)); 

sns.violinplot(x=df.Category, y=df.Rating);

plt.xticks(rotation=90);
df.groupby(['Category']).Rating.mean()
df.groupby(['Type']).Rating.mean()
plt.bar(df.Type.unique(), df.groupby(['Type']).Rating.mean());



for i,v in enumerate(df.groupby(['Type']).Rating.mean()):

    plt.text(i-0.05,v-0.50,round(v,2))
df.Content_Rating.unique()
plt.bar(df.Content_Rating.unique(), df.groupby(['Content_Rating']).Rating.mean());

plt.xticks(rotation = 30);



for i,v in enumerate(df.groupby(['Content_Rating']).Rating.mean()):

    plt.text(i-0.1,v-0.50,round(v,2))
sns.heatmap(df.corr(),vmax=.8, cmap="BuPu");
df.corr()
pd.plotting.scatter_matrix(df);
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from patsy import dmatrices

from sklearn.feature_selection import RFE

from sklearn import metrics
df.head()
sns.boxplot(df.Rating);
sns.boxplot(df.Size_log);
sns.boxplot(df[df['Type']=='Paid'].Price_log);
df_o = df.copy()
q3, q1 = np.percentile(df.Rating, [75,25])

max = q3 + (1.5*(q3-q1))

min = q1 - (1.5*(q3-q1))



ls3=[]

for i,x in enumerate(df.Rating):

    if x < min:

        ls3.append(i)
df_o.drop(ls3, axis=0, inplace=True);
df.head()
df.sample()
df.Content_Rating.unique()
df.Content_Rating = df.Content_Rating.str.replace('+', '')

df.Content_Rating = df.Content_Rating.str.replace(' ', '_')
df.Content_Rating.unique()
df.Genres.unique()
df.Genres = df.Genres.str.replace(' ', '_')
df.Genres = df.Genres.str.replace('&', '')
df.Genres.unique()
#df.to_csv('temp.csv',index=False)
ls_dummy = ['Category', 'Type', 'Content_Rating', 'Genres']
for i in range(0,len(ls_dummy)):

    df[sorted(df[ls_dummy[i]].unique())[1:]] = pd.get_dummies(df[ls_dummy[i]],drop_first=True)
df.sample()
col = np.array(df.columns)
col
X = ['Price', 'Reviews_log', 'Size_log',

       'Installs_log', 'Price_log', 'AUTO_AND_VEHICLES', 'BEAUTY',

       'BOOKS_AND_REFERENCE', 'BUSINESS', 'COMICS', 'COMMUNICATION',

       'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FAMILY',

       'FINANCE', 'FOOD_AND_DRINK', 'GAME', 'HEALTH_AND_FITNESS',

       'HOUSE_AND_HOME', 'LIBRARIES_AND_DEMO', 'LIFESTYLE',

       'MAPS_AND_NAVIGATION', 'MEDICAL', 'NEWS_AND_MAGAZINES',

       'PARENTING', 'PERSONALIZATION', 'PHOTOGRAPHY', 'PRODUCTIVITY',

       'SHOPPING', 'SOCIAL', 'SPORTS', 'TOOLS', 'TRAVEL_AND_LOCAL',

       'VIDEO_PLAYERS', 'WEATHER', 'Paid', 'Everyone', 'Everyone_10',

       'Mature_17', 'Teen', 'Unrated', 'Action__Adventure', 'Adventure',

       'Arcade', 'Art__Design', 'Auto__Vehicles', 'Beauty', 'Board',

       'Books__Reference', 'Brain_Games', 'Business', 'Card', 'Casino',

       'Casual', 'Comics', 'Communication', 'Creativity', 'Dating',

       'Education', 'Educational', 'Entertainment', 'Events', 'Finance',

       'Food__Drink', 'Health__Fitness', 'House__Home', 'Libraries__Demo',

       'Lifestyle', 'Maps__Navigation', 'Medical', 'Music',

       'Music__Audio', 'Music__Video', 'News__Magazines', 'Parenting',

       'Personalization', 'Photography', 'Pretend_Play', 'Productivity',

       'Puzzle', 'Racing', 'Role_Playing', 'Shopping', 'Simulation',

       'Social', 'Sports', 'Strategy', 'Tools', 'Travel__Local', 'Trivia',

       'Video_Players__Editors', 'Weather', 'Word']
Y = df.Rating
from sklearn.linear_model import LinearRegression
ln = LinearRegression()

rfe = RFE(ln,30)

rfe = rfe.fit(df[X], Y.values.ravel())

print(rfe.support_)

print(rfe.ranking_)
rfe.ranking_
col = []

s = 'Rating ~ '

for i in range(0,len(rfe.ranking_)):

    if rfe.ranking_[i] == 1:

        s = s + str(X[i]) + ' + '

        

s = s[:-3]
s
y, x = dmatrices('Rating ~ Reviews_log + Installs_log + BEAUTY + COMICS + COMMUNICATION + DATING + ENTERTAINMENT + EVENTS + LIFESTYLE + MAPS_AND_NAVIGATION + PERSONALIZATION + Art__Design + Beauty + Books__Reference + Brain_Games + Card + Comics + Communication + Creativity + Dating + Education + Events + Lifestyle + Music__Audio + Music__Video + Parenting + Puzzle + Strategy + Trivia + Word', df, return_type='dataframe')



vif = pd.DataFrame()

vif['VIF Factor'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]

vif['features'] = x.columns
vif
y, x = dmatrices('Rating ~ Installs_log + ENTERTAINMENT + MAPS_AND_NAVIGATION + PERSONALIZATION + Art__Design + Beauty + Books__Reference + Brain_Games + Card + Comics + Communication + Creativity + Dating + Education + Events + Lifestyle + Music__Audio + Music__Video + Parenting + Puzzle + Strategy + Trivia + Word', df, return_type='dataframe')



vif = pd.DataFrame()

vif['VIF Factor'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]

vif['features'] = x.columns
vif
X = vif['features']
X = list(X)
df['intercept'] = 1



#Adding an intercept to the dataframe
df.sample()
df.rename(columns={'intercept':'Intercept'}, inplace=True);
Y = df['Rating']
mlr = sm.OLS(Y, df[X])

result = mlr.fit()

print(result.summary())
rm = ['Beauty','ENTERTAINMENT', 'Card', 'Comics', 'Communication', 'Creativity', 'Lifestyle', 'Music__Audio', 'Music__Video', 'Strategy', 'Trivia']

for i in rm:

    X.remove(i)
mlr = sm.OLS(Y, df[X])

result = mlr.fit()

print(result.summary())
df.info()
X_train, X_test, y_train, y_test = train_test_split(df[X], y, test_size=0.30, random_state=1001)

ln = LinearRegression()

ln.fit(X_train, y_train)
y_pred = ln.predict(X_test)

print('Accuracy of linear regression classifier on test set: {:.2f}'.format(ln.score(X_test, y_test)))
rmse = sqrt(mean_squared_error(y_test, y_pred))

print(rmse)
df.info()
dft = df.copy()
ls3[:5]
dft.drop(ls3, axis=0, inplace=True)
dft.reset_index(inplace=True)
dft.drop('index', axis=1, inplace=True)
mlr2 = sm.OLS(dft['Rating'], dft[X])
result = mlr2.fit()

print(result.summary())
Y2 = dft['Rating']
X_train2, X_test2, y_train2, y_test2 = train_test_split(dft[X], Y2, test_size=0.30, random_state=1001)

ln = LinearRegression()

ln.fit(X_train2, y_train2)
y_pred2 = ln.predict(X_test2)

print('Accuracy of linear regression classifier on test set: {:.2f}'.format(ln.score(X_test2, y_test2)))
rmse = sqrt(mean_squared_error(y_test2, y_pred2))

print(rmse)
df_new = y_test.copy()

df_new['Prediction'] = y_pred

df_new.rename(columns={'grade_MBA':'Actual'},inplace=True);
df_new.head()