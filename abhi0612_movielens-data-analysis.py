#MovieLens dataset
import pandas as pd

import numpy as np

import matplotlib.pyplot as pltd

import seaborn as sns
data_movies=pd.read_csv('../input/movies.dat',sep='::',header=None,names=['MovieId','Title','Genres'])

data_ratings=pd.read_csv('../input/ratings.dat',sep='::',header=None,names=['UserId','MovieId','Rating','Timestamp'])

data_users=pd.read_csv('../input/users.dat',sep='::',header=None,names=['UserId','Gender','Age','Occupation','Zip-code'])
data_movies.head(6)
#Sum=0 means all are false and no duplicate values

data_movies['Title'].duplicated().sum()
data_movies['Title'].describe()
data_ratings.head(5)
#No of times UserId = 1 appared

data_ratings.loc[data_ratings['UserId']==1].info()
data_users.head()
data_users.info()
df1=pd.merge(data_ratings,data_users[['UserId','Gender','Age','Occupation']],on='UserId',how='inner')

df1.head(6)
df2=pd.merge(df1[['UserId','MovieId','Rating','Gender','Age','Occupation']],data_movies[['MovieId','Title']],on='MovieId',

             how='inner')

df2.head(4)
df3=df2.sort_values(by='UserId',ascending=True)

df3.head()
df4=df3
df4['Age'].value_counts()
df4['Age Group']=np.nan
df4.head()
df4.loc[df4['Age'] ==1, 'Age Group'] = 'Under 18'

df4.loc[df4['Age'] ==18, 'Age Group'] = '18-24'

df4.loc[df4['Age'] ==25, 'Age Group'] = '25-34'

df4.loc[df4['Age']==35, 'Age Group'] = '35-44'

df4.loc[df4['Age'] ==45, 'Age Group'] = '45-49'

df4.loc[df4['Age'] ==50, 'Age Group'] = '50-55'

df4.loc[df4['Age']==56, 'Age Group'] = '56+'
df4.head(3)
#UserIDs range between 1 and 6040 

#The MovieIDs range between 1 and 3952

#Ratings are made on a 5-star scale (whole-star ratings only)

#A timestamp is represented in seconds since the epoch is returned by time(2)

#Each user has at least 20 ratings

 

df4.reset_index(drop=True,inplace=True)

df4.head(10)
#No missing value found

df4.isna().sum()
df4.info()
df4.describe(include='all')
#univariate analysis

df4['Rating'].value_counts()
#No of distinct movies

movies_set=set(list(df4['Title']))

movies_list=list(movies_set)

len(movies_list)
#or

df4['Title'].unique().shape
#countplot for rating

sns.countplot(x='Rating',data=df4,hue='Gender',palette='coolwarm')
#Age exploration

unq_userid=df4['UserId'].unique()

unq_userid.shape
#Create a list for unique age from df4

i=1

list_age=[]

while(i<=6040):

     a= list(df4.loc[df4['UserId']==i]['Age'].unique())

     list_age = list_age + a

     i=i+1

#Avg age of the viewer

len(list_age)

pd.Series(list_age).mean()
sns.countplot(pd.Series(list_age),hue='Gender',data=df4,palette='coolwarm',)
pd.Series(list_age).value_counts()
#Create a list for unique Occupation per userid from df4

i=1

list_occ=[]

while(i<=6040):

     a= list(df4.loc[df4['UserId']==i]['Occupation'].unique())

     list_occ = list_occ + a

     i=i+1
len(list_occ)
#median value of occupation

pd.Series(list_occ).median()
#modevalue of occupation

pd.Series(list_occ).value_counts().head()
sns.countplot(pd.Series(list_occ))
#Exploration

#User Age Distribution

sns.distplot(list_age,kde=False,bins=10)
#User rating of the movie “Toy Story”

df3.loc[df4['Title']=='Toy Story (1995)']['Rating'].mean()
#value counts of movie Toy story

df3.loc[df4['Title']=='Toy Story (1995)']['Rating'].value_counts()
#count plot for Toy story based on rating

sns.countplot(df4.loc[df4['Title']=='Toy Story (1995)']['Rating'])
#Find and visualize the viewership of the movie “Toy Story” by age group

df4.loc[df4['Title']=='Toy Story (1995)','Age Group'].value_counts()
#count plot for Toy Story as per Age Group

sns.countplot(df4.loc[df4['Title']=='Toy Story (1995)','Age Group'])
#Top 25 movies by viewership rating

#finding mean rating of all the movies

se1=df4.groupby('Title')['Rating'].agg('mean')
se1.head()
#Top 25 movies

se1.nlargest(25)
#visualization of top 25 movies based on rating

plot1=sns.barplot(x=se1.nlargest(25).index,y=se1.nlargest(25).values)

plot1.set_xticklabels(plot1.get_xticklabels(),rotation=90)
#Find the ratings for all the movies reviewed by for a particular user of user id = 2696

df4.loc[df4['UserId']==2696]
#Visualize the rating data by user of user id = 2696

plot2=sns.barplot(x=df4.loc[df3['UserId']==2696]['Title'],y=df4.loc[df3['UserId']==2696]['Rating'])

plot2.set_xticklabels(plot1.get_xticklabels(),rotation=90)
#Find out all the unique genres 

data_movies_gen_list=data_movies['Genres'].tolist()

data_movies_gen_list[0:5]
type(data_movies_gen_list)
data_movies_gen_list[1].split(sep='|')
len(data_movies_gen_list)

i=0

list1=[]

while(i<3883):

    list1=list1+data_movies_gen_list[i].split(sep='|')

    i=i+1

print(list1)   

type(list1)

set1=set(list1)

genre_list=list(set1)

genre_list
df4.head()
#Determine the features affecting the ratings of any particular movie.

df5=df4[['Rating', 'Gender', 'Age Group', 'Occupation']]
df5.head()
#df6=pd.get_dummies(df5,columns=['Gender','Age Group']).iloc[:,[0,1,2,4,5,6,7,8,9,11]]

#df6.head(3)

df5['Occupation'].value_counts()
#Applying Machine learning on first 500 dataset

dataset=df5.iloc[0:500,]

X=dataset.iloc[:,[1,2,3]]

Y=dataset.iloc[:,0]
X.head()
Y.head()
x1=pd.get_dummies(data=X)
x2=pd.get_dummies(X['Occupation'],prefix='Occupation')
x=pd.concat([x1,x2],axis=1)
x.columns
x.drop(['Occupation','Gender_F','Age Group_56+','Occupation_20'],axis=1,inplace = True)
x.head()
Y.head()
#Determine the features affecting the ratings of any particular movie.

XY=pd.concat([x,Y],axis=1)
XY.head(5)
XY.corr()
sns.heatmap(XY.corr())
sns.pairplot(XY.corr())
#Splitting data into Training and test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.3,random_state=102)
#Building Model Using Linear regression

from sklearn.linear_model import LinearRegression

lm=LinearRegression()
lm.fit(x_train,y_train)
print(lm.intercept_)
print(lm.coef_)
coffecient=pd.DataFrame(data=lm.coef_,index=x_train.columns,columns=['coffecient'])
coffecient
pred_y=lm.predict(X=x_test)
pred_y
x_test.head()
#import sklearn metric package

from sklearn import metrics
metrics.mean_absolute_error(y_test,pred_y)
metrics.mean_squared_error(y_test,pred_y)
np.sqrt(metrics.mean_squared_error(y_test,pred_y))
lm.score(x_test,y_test)
#Create a histogram for movie, age, and occupation

sns.distplot(data_users['Age'])
sns.distplot(data_users['Occupation'])