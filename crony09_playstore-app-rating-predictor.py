import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

plt.rcParams['figure.figsize']=[10,6]
df=pd.read_csv('../input/playstore-analysis/googleplaystore.csv')

df.head(4)
df.drop(['Current Ver','Android Ver'],axis=1,inplace=True)
df.isnull().sum()
#Since Rating is the target feature, we can't replace NaN values with Mean/Median values. 

#Therefore, we drop all all rows that have Null values

df.dropna(inplace=True)

df.reset_index(inplace=True)

df.isnull().sum()
def getnumber(s):

    number=""

    for i in s:

        if i>='0' and i<='9':

            number+=i

        if i=='.':

            number+=i

    if number=="":

        print('No digits')

    else:

        return float(number)

## This function returns all the digits from a string 
def getalphabet(s):

    alphabet=""

    for i in s:

        if i>='a' and i<='z':

            alphabet+=i

        elif i>='A' and i<='Z':

            alphabet+=i

    return alphabet

## This function returns all the alphabets from a string 
size_index = df.columns.get_loc('Size')

installs_index = df.columns.get_loc('Installs')

rows,columns=df.shape

all_sizes=[]

for i in range(0,rows):

    all_sizes.append(getalphabet(df.iloc[i,size_index]))

print(set(all_sizes))
## So we understand that few apps have sizes 'Varieswithdevice'.Hence, we check if such values have any integers associated with it.

vwd_index=[]

for i in range(0,len(all_sizes)):

    if all_sizes[i]=='Varieswithdevice':

        vwd_index.append(i)

for i in vwd_index:

    print(df.iloc[i,size_index])

print('Number of missing/ambiguous size values:',len(vwd_index))
## 1637 of 9366 values don't have size data. Initially, we remove the rows with missing sizes.

## After analysis, if we find that there's not a strong correlation between size and ratings, we remove the size column altogether

data=df.copy()

for i in vwd_index:

    data.drop(index=i,inplace=True)
rows,columns=data.shape

all_sizes=[]

for i in range(0,rows):

    all_sizes.append(getalphabet(data.iloc[i,size_index]))

print(set(all_sizes))
## Now we have all the sizes in either KB or MB. We now convert MB to KB and everything from String to Int.

for i in range(0,rows):

    if getalphabet(data.iloc[i,size_index])=='M':

        data.iloc[i,size_index]=getnumber(data.iloc[i,size_index])*1000

    else:

        data.iloc[i,size_index]=getnumber(data.iloc[i,size_index])   
for i in range(0,rows):

    data.iloc[i,installs_index]=getnumber(data.iloc[i,installs_index])
data.drop('index',axis=1,inplace=True)
price_index = data.columns.get_loc('Price')

for i in range(0,rows):

    data.iloc[i,price_index]=getnumber(data.iloc[i,price_index])
print('Maximum app rating=',data['Rating'].max())

print('Minimum app rating=',data['Rating'].min())
data.head(4)
## Dropping values where number of installs are less than the number of reviews.

data.drop(data[data.Installs<data.Reviews].index,axis=0,inplace=True)
## Checking if all the Free apps have price = 0

data[data.Type=='Free'].Price.sum()  ##The sum of prices of all the Free apps should be 0.
plt.boxplot(data[data.Type!='Free'].Price,vert=False)
## From the BoxPlot we can definitely see that the Prices above 50$ are obvious outliers. Therefore we first remove those and then compute the quartiles of the remaining data.

data.drop(data[data.Price>50].index,axis=0,inplace=True)

plt.boxplot(data[data.Type!='Free'].Price,vert=False)
#we remove records with over 10 million reviews

data.drop(data[data.Reviews>data['Reviews'].quantile(0.95)].index,axis=0,inplace=True)

plt.boxplot(data.Reviews,vert=False)
data['Rating'].plot.hist(bins=8,rwidth=0.99)

plt.xlabel('App rating')

plt.ylabel('Number of Apps')
## Majority of the apps have higher ratings.(most between 4.0 to 4.5)

data['Size'].plot.hist(bins=8,rwidth=0.99)

plt.xlabel('App size (in KB)')

plt.ylabel('Number of Apps')
sns.lmplot(x="Price",y="Rating",hue='Type',data=data,fit_reg=False)
## Apparently,there is no strong correlation between Ratings and Price.Majority of the paid apps have a higher rating.

sns.jointplot(x=data[data.Type=='Paid']['Price'],y=data[data.Type=='Paid']['Rating'],data=data,kind="kde")
data[data.Type=='Free']['Rating'].plot.hist(bins=8) ##Checking out the trend for Free Apps
sns.jointplot(x='Size',y='Rating',data=data,kind='scatter')

#We can conclude that the percentage of large apps having high ratings(>4) is more!
sns.jointplot(x='Reviews',y='Rating',data=data,kind='scatter') #Similar to the Price vs Rating graph. For higher number of reviews, the rating is always greater than 3.5
sns.boxplot(x="Content Rating",y="Rating",data=data)
sns.countplot(x='Content Rating',data=data)
#We reduce some of the categories by clubbing similar ones together

category_index=data.columns.get_loc('Category')

rows,columns=data.shape

for i in range(0,rows):

    if data.iloc[i,category_index]=='ART_AND_DESIGN' or data.iloc[i,category_index]=='BEAUTY':

        data.iloc[i,category_index]='Art'

    if data.iloc[i,category_index]=='BOOKS_AND_REFERENCE' or data.iloc[i,category_index]=='COMICS' or data.iloc[i,category_index]=='LIBRARIES_AND_DEMO' or data.iloc[i,category_index]=='NEWS_AND_MAGAZINES':

        data.iloc[i,category_index]='Books'

    if data.iloc[i,category_index]=='BUSINESS' or data.iloc[i,category_index]=='FINANCE':

        data.iloc[i,category_index]='Money'

    if data.iloc[i,category_index]=='SPORTS' or data.iloc[i,category_index]=='GAME':

        data.iloc[i,category_index]='Sport/Game'

    if data.iloc[i,category_index]=='HEALTH_AND_FITNESS' or data.iloc[i,category_index]=='LIFESTYLE' or data.iloc[i,category_index]=='MEDICAL':

        data.iloc[i,category_index]='Health'

    if data.iloc[i,category_index]=='MAPS_AND_NAVIGATION' or data.iloc[i,category_index]=='TRAVEL_AND_LOCAL' or data.iloc[i,category_index]=='WEATHER':

        data.iloc[i,category_index]='Maps/Travel'

    if data.iloc[i,category_index]=='SOCIAL' or data.iloc[i,category_index]=='DATING':

        data.iloc[i,category_index]='Social'
set(data['Category'])
sns.boxplot(x="Category",y="Rating",data=data)
data.drop(['Genres','Last Updated'],axis=1,inplace=True)

data.head()
data.drop(data[data.Installs>data['Installs'].quantile(0.95)].index,axis=0,inplace=True)

plt.scatter(data['Reviews'],data['Installs'])
data['Installs'].quantile(0.95)
data['Reviews'].max()
category_variables = pd.get_dummies(data['Category'])

category_variables.drop('Art',axis=1,inplace=True)

data = pd.concat([data,category_variables],axis=1)

data.drop('Category',axis=1,inplace=True)

data.head()
conrating_variables = pd.get_dummies(data['Content Rating'])

conrating_variables.drop('Everyone',axis=1,inplace=True)

data = pd.concat([data,conrating_variables],axis=1)

data.drop('Content Rating',axis=1,inplace=True)

data.head()
X_data=data[['Reviews','Size','Installs']]

y=data['Rating']

X_data
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

scaler.fit(X_data)

X=scaler.transform(X_data)

X=pd.DataFrame(data=X,columns=['Rev','Size_new','Inst'],index=X_data.index)

X=pd.concat([data,X],axis=1)

X
X.drop(['Reviews','Size','Installs'],axis=1,inplace=True)

X.head(3)
X=X.drop('Rating',axis=1)

X
X.drop('Type',axis=1,inplace=True)

X.drop('App',axis=1,inplace=True)
from sklearn.neighbors import KNeighborsRegressor

knr=KNeighborsRegressor(20)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score

rf_random = RandomForestRegressor()

y_test
rf_random.fit(X_train,y_train)

predictions=rf_random.predict(X_test)

predictions
#results=pd.DataFrame(data=y_test,columns='Y_test_actual',index=range(len(y_test)))

#results

y_test.reset_index(drop=True,inplace=True)
y_test
results=pd.DataFrame(columns=['Y_test_actual','Predicted'])

results['Y_test_actual']=y_test

results['Predicted']=predictions
results
sns.jointplot(x='Y_test_actual',y='Predicted',data=results,kind="reg")

plt.plot([1,5],[1,5])

plt.show()
plt.scatter(y_test,predictions)

plt.plot([1,5],[1,5])