#Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import datetime as dt
#Reading csv file using pandas

df=pd.read_csv('../input/hack_startup_funding.csv')
#Checking the head files

df.head()
#Droppin irrelevant columns

df.drop(['SNo','Remarks'],axis=1,inplace=True)
#Checking data again

df.head()
#First step is to convert object into date time to do this we need to first structure the data

df['Date']=df['Date'].replace({"12/05.2015":"12/05/2015"})

df['Date']=df['Date'].replace({"13/04.2015":"13/04/2015"})

df['Date']=df['Date'].replace({"22/01//2015":"22/01/2015"})

df['Date']=df['Date'].replace({"15/01.2015":"15/01/2015"})
#Converting date column into type date

df['Date']=pd.to_datetime(df['Date'])
#Extracting month from date column

df['month']=df['Date'].dt.month
#Extracting year from date column



df['year']=df['Date'].dt.year
#Extracting year from date column



df['quarter']=df['Date'].dt.quarter
#Extracting days from date column



df['day']=df['Date'].dt.day
#Combining year and month together

df["yearmonth"] = (pd.to_datetime(df['Date'],format='%d/%m/%Y').dt.year*100)+ (pd.to_datetime(df['Date'],format='%d/%m/%Y').dt.month)
year_month = df['yearmonth'].dropna().value_counts()
#Dropping date column as we have all the information extracted

df.drop('Date',axis=1,inplace=True)
#Calculating percentage of all the date categories for eda

year=df['year'].value_counts()/len(df['year'])*100

month=df['month'].value_counts()/len(df['month'])*100

quarter=df['quarter'].value_counts()/len(df['quarter'])*100

day=df['day'].value_counts()/len(df['day'])*100
#Plotting funding according to its timeline

plt.figure(figsize=(15,8))

sns.barplot(year_month.index, year_month.values, alpha=0.9,color='skyblue')



plt.xticks(rotation='vertical')

plt.xlabel('Year-Month ', fontsize=18)

plt.ylabel('Number of fundings made', fontsize=18)

plt.title("Year-Month - Number of Funding Distribution", fontsize=20)

plt.show()

#There seems to be no trend in the amount of funding that has been made

#but from 201612 to 201712 the funding has been gradually dropped
#Checking if there is trend in investment as per month

plt.figure(figsize=(10,4))

sns.barplot(month.index, month.values, alpha=0.9,color='skyblue')

plt.title('Investment per month',fontsize=25)

plt.xlabel('Month',fontsize=17)

plt.ylabel('Number of investments',fontsize=17)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.show()

#There does not seem to be any trend but January and June seems to have the most number of investments
#PLotting funding frequency according to year

plt.figure(figsize=(7,6))

sns.barplot(year.index, year.values, alpha=0.9,color='skyblue')

plt.title('Investment per year',fontsize=19)

plt.xlabel('Year',fontsize=16)

plt.ylabel('Number of investments',fontsize=16)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()

#It looks like funding has increased from 2015 to 2016. 

#It makes sense if the funding for 2017 will be greater than 2016

#But we have data for 2017 till october so we cannot really say if the investment has increased after 2017. 

#According to this data, investment has dropped from 39.6% to 18.6% from 2016 to 2017

#Plotting funding frequency according to the quarter

plt.figure(figsize=(7,4))

sns.barplot(quarter.index, quarter.values, alpha=0.9,color='skyblue')

plt.title("Investments per quarter",fontsize=18)

plt.xlabel('Quarter',fontsize=16)

plt.ylabel('Number of investment',fontsize=16)

plt.xticks(fontsize=14)

plt.yticks(fontsize=11)

plt.show()

#First 2 quarter seems to have slightly more funding than 3rd and 4th quarter
#Cleaning the target variable

df['AmountInUSD']=(df['AmountInUSD'].str.replace(',','')).astype('float')
#Making a function which gives 95% confidence point and interval, also the range between the lower and upper value.

import scipy.stats as stats



def mean_confidence_interval(data, confidence=0.95):

    a = 1.0 * np.array(data)

    n = len(a)

    m, se = np.mean(a), stats.sem(a)

    h = se * stats.t.ppf((1 + confidence) / 2., n-1)

    print('Confidence point:-',np.around(m,decimals=2))

    print('From:-',(np.around(m-h,decimals=2)))

    print('To:-',np.around(m+h,decimals=2))

    print('Range:-',(np.around((m+h)-(m-h),decimals=2)))

    



#Calculating 95% confidence interval for out target variable

mean_confidence_interval(df['AmountInUSD'].dropna())
#Printing confidence interval as per year

for x in df['year'].value_counts().index:

    print(x)

    print('*'*30)

    mean_confidence_interval((df['AmountInUSD'][df['year']==x]).dropna())

    print('\n')
#Printing confidence interval for top all top3 categories

for cols in df.columns[0:6]:

    print(cols)

    for x in df[cols].value_counts()[0:3].index:

        print('*'*30)        



        print(x)

        

        print('*'*30)

        

        mean_confidence_interval((df['AmountInUSD'][df[cols]==x]).dropna())

        

        print('\n')
#Statistics for Amount of investment

#The amount that is invested seems to be very flexible

print('skewness',df['AmountInUSD'].skew())

print('kurtosis',df['AmountInUSD'].kurtosis())

print('median  ',df['AmountInUSD'].median())

print(df['AmountInUSD'].describe())

#Checking the distribution for the target variable AmountInUSD

import statsmodels.api as sm

sm.qqplot(df['AmountInUSD'].dropna())
#Analyzing the startup that got the max funding

df[df['AmountInUSD']==1400000000.0]
#Analyzing the startup that got the least funding

df[df['AmountInUSD']==16000]
#Before splitting the dataset we will first clean the data
#There are many variables where the data is same but the name is different

#First we will lower all the strings

#Later we will personally rename all the variable names

df['StartupName']=df['StartupName'].str.lower()
df['StartupName']=df['StartupName'].replace("practo","practo")

df['StartupName']=df['StartupName'].replace("couponmachine.in","couponmachine")

df['StartupName']=df['StartupName'].replace("olacabs","ola cabs")

df['StartupName']=df['StartupName'].replace("ola","ola cabs")

df['StartupName']=df['StartupName'].replace("olipkart.com","flipkart")

df['StartupName']=df['StartupName'].replace("paytm marketplace","paytm")

df['StartupName'][df['StartupName']=='flipkart.com']='flipkart'



df['StartupName'][df['StartupName']=='oyo']='oyo rooms'
df['IndustryVertical']=df['IndustryVertical'].str.lower()
df['SubVertical']=df['SubVertical'].str.lower()
df['CityLocation']=df['CityLocation'].str.lower().str[0:2]
df['InvestorsName']=df['InvestorsName'].str.split(expand=True)[0].str.lower()
df['InvestmentType']=df['InvestmentType'].str.lower().str[0]
df.head()
#Now we will create our 2 new dataframes

df_test=df[df['AmountInUSD'].isnull()]



df_test.drop('AmountInUSD',axis=1,inplace=True)



df_test=df_test.dropna()



sns.heatmap(df_test.isnull(),cbar=False,cmap='viridis',yticklabels=False)

#All null values were of year 2015

df_train=(df.dropna(subset=['AmountInUSD']))
#Statistics for categorical variables

df.describe(include='object')
#Top 10 statups according to amount invested

(df[['StartupName','AmountInUSD']].dropna()).sort_values(by='AmountInUSD',ascending = False).head(10)



#Making a dataframe of top 20 start up name according to the amount that was invested in them



top20funding=(df[['StartupName','AmountInUSD']].dropna()).sort_values(by='AmountInUSD',ascending = False).head(20)



top20funding
#Counting the frequency of startups in top 20 startups that were funded

top20fundingcount=(df[['StartupName','AmountInUSD']].dropna()).sort_values(by='AmountInUSD',ascending = False).head(20)['StartupName'].value_counts()

top20fundingcount
##Plotting the frequency of startups in top 20 startups that were funded



plt.figure(figsize=(15,8))

sns.barplot(top20fundingcount.index, top20fundingcount.values, alpha=0.9,color='skyblue')



plt.xticks(rotation='vertical')

plt.xlabel('Startup Name', fontsize=18)

plt.ylabel('Number of fundings made', fontsize=18)

plt.title("Frequency of startups in top 20 funded amount", fontsize=20)

plt.show()



#Creating a new dataframe with Startup name and the amount they got as funding

nameamount=df_train[['StartupName','AmountInUSD']]
#Grouping the dataframe according to startup name and sorting it out accoring to number of funding they recieved

nameamount=nameamount.groupby('StartupName').sum().sort_values(by='AmountInUSD',ascending=False)



#How much of funding is recieved by how much of startups

np.sum((nameamount/np.sum(nameamount)*100).head(53))



len(nameamount)



53/1268*100
#Percentage of null values in dataframe

len(df[df['IndustryVertical'].isnull()==True])/len(df['IndustryVertical'])*100
#Industries that recieved funding more than 3 times

df_train['IndustryVertical'].value_counts()[df_train['IndustryVertical'].value_counts()>3]
#Top 3 industries with most frequent funding recieved

(df_train['IndustryVertical'].value_counts()/len(df_train['IndustryVertical'])*100)[0:3]



x_barindustry=np.array(['consumer internet','technology','ecommerce','other 2369'])     

y_barindustry=np.array([30.229508,12.393443,9.770492,47.6])



plt.figure(figsize=(10,8))

sns.barplot(x_barindustry,y_barindustry,color='skyblue')

plt.xticks(rotation='vertical',fontsize=15)

plt.xlabel('Industry', fontsize=20)

plt.ylabel('Number of funding', fontsize=18)

plt.title("Percentage of fundding according to industry", fontsize=20)



plt.show()



#Comparing investments of top 3 industries vs the rest

x_barindustry_top3=np.array(['top3','other 2369'])     

y_barindustry_top3=np.array([53.4,47.6])





plt.figure(figsize=(8,6))

sns.barplot(x_barindustry_top3,y_barindustry_top3,color='skyblue')

plt.xticks(rotation='vertical',fontsize=15)

plt.xlabel('Industry', fontsize=20)

plt.ylabel('Number of funding', fontsize=18)

plt.title("Percentage of funding according to industry", fontsize=20)



plt.show()
#Creating new dataframe with the industry name and amount invested in respective industries

indamount=df_train[['IndustryVertical','AmountInUSD']]



#Grouping the dataframe according to industries and sorting it according to the amount invested

indamount=indamount.groupby('IndustryVertical').sum().sort_values(by='AmountInUSD',ascending=False)



#How much investment is done in how many of industries?

np.sum((indamount/indamount.sum()*100).head(9))



len(indamount)



9/508*100



#Creating new dataframe which is sorted according to frequency of investments that is allocated as per city

topcityfunded=df_train['CityLocation'].value_counts()[df_train['CityLocation'].value_counts()>10]
plt.figure(figsize=(8,8))

sns.barplot(topcityfunded.index, topcityfunded.values, alpha=0.9,color='skyblue')



plt.xticks(rotation='vertical',fontsize=15)

plt.xlabel('Startup Name', fontsize=20)

plt.ylabel('Number of fundings made', fontsize=18)

plt.title("Frequency of startups according to city", fontsize=20)



plt.show()



#Creating new dataframe which is sorted according to the amount of investment that is allocated

cityamount=df[['CityLocation','AmountInUSD']].dropna().sort_values(by='AmountInUSD')



cityamount=cityamount.groupby('CityLocation').sum().sort_values(by='AmountInUSD',ascending=False)



x_barcity=np.array(['ba', 'ne', 'mu', 'gu', 'ch', 'pu', 'hy', 'no', 'ah', 'ja','other'])

y_barcity=np.array([8.42297411e+09,

       2.82019750e+09,

       2.35493450e+09,

       2.06902150e+09,

       4.37205000e+08,

       3.66653000e+08,

       1.95362000e+08,

       1.70638000e+08,

       9.81860000e+07,

       3.55600000e+07,

            104791000.0

            ])



plt.figure(figsize=(15,8))

sns.barplot(x_barcity,y_barcity, alpha=0.9,color='skyblue')



plt.xticks(rotation='vertical',fontsize=15)

plt.xlabel('City', fontsize=20)

plt.ylabel('Funding', fontsize=18)

plt.title("Highest funding according to city", fontsize=20)



plt.show()



#Creating a pie diagram which shows how much funding is given as per cities

x_piecity=np.array(['ba', 'ne', 'mu', 'gu', 'ch', 'other'])

y_piecity=np.array([8.42297411e+09,

       2.82019750e+09,

       2.35493450e+09,

       2.06902150e+09,

       4.37205000e+08,

      971190000.0

            ])



plt.figure(figsize=(10,10))

plt.pie(y_piecity,labels=(x_piecity),autopct='%1.1f%%',colors=['skyblue','pink','plum','lightgreen','coral','gold'],explode=[0,0.1,0,0,0,0],startangle=45)

plt.rcParams['font.size'] = 16

plt.show()
#How much investment is done in how many of cities?



(np.sum((cityamount/np.sum(cityamount)*100)[0:4]))



len(cityamount)



4/32*100



#Top 10 investors according to frequency of funding

investorname=df['InvestorsName'].value_counts().head(10)

investorname
#How much does the top 10 investors contribute in funding

np.sum((df_train['InvestorsName'].value_counts()/len(df_train['InvestorsName'])*100)[0:10])
#How much does the rest of investors contribute in funding

np.sum((df_train['InvestorsName'].value_counts()/len(df_train['InvestorsName'])*100)[10:])
plt.figure(figsize=(8,8))

sns.barplot(investorname.index, investorname.values, alpha=0.9,color='skyblue')



plt.xticks(rotation='vertical',fontsize=15)

plt.xlabel('Investor Name', fontsize=20)

plt.ylabel('Number of fundings made', fontsize=18)

plt.title("Frequency of funding according to investor", fontsize=20)



plt.show()



#Creating new data frame with only Investor names and the amount they invested

investoramount=df_train[['InvestorsName','AmountInUSD']]
investoramount.head(10)
#Grouping investorname with the amount they have invested

investoramount=investoramount.groupby('InvestorsName').sum().sort_values(by='AmountInUSD',ascending=False)

investoramount.head(10)
#How much investment is done by how many of investors?

(np.sum((investoramount/np.sum(investoramount)*100)[0:36]))



len(investoramount)



36/703*100
#Creating new data frame with only Investor names and the amount they invested

subamount=df_train[['SubVertical','AmountInUSD']]
subamount=(subamount.groupby('SubVertical').sum()).sort_values(by='AmountInUSD',ascending=False)
#How much investment is done in Subvertical?



np.sum((subamount/np.sum(subamount)*100).head(39))



len(subamount)



39/815*100



#Grouping the data frame according to startup names

group=df_train.groupby('StartupName')
#Sorting the dataframe according to amount of funding and capturing the 4.1% of startup that got 70% of funding

topstartup=group.first().sort_values('AmountInUSD',ascending=False).head(53)
topstartup.head()
#Top frequency of all variables in this top topstartup dataframe

for x in topstartup.columns:

    print(x)

    print('*'*30)

    print(topstartup[x].value_counts()[topstartup[x].value_counts()>1])

    print('\n')
#Percentage of top frequency of all variables in top topstartup dataframe

for x in topstartup.columns:

    print(x)

    print('*'*30)

    print((topstartup[x].value_counts()[topstartup[x].value_counts()/len(topstartup[x])*100>2])/len(topstartup[x])*100)

    print('\n')
#Dropping the null values

df_train=df_train.dropna()
#As all our features are categorical we need to covert them into numerical type before fitting it into ml algorithm

!pip install feature-engine

from feature_engine import categorical_encoders as ce
#First we will split our data in training and testing dataset

from sklearn.model_selection import train_test_split
X_df_train=df_train.drop('AmountInUSD',axis=1)

y_df_train=df_train['AmountInUSD']
X_train, X_test, y_train, y_test = train_test_split(X_df_train, y_df_train, test_size=0.3, random_state=42)
#It is important to check if our training and testing data is in same shape

X_train.shape
X_test.shape
#Replaces categories by the mean of the target. 



#For example in the variable colour, if the mean of the target for blue, red

#and grey is 0.5, 0.8 and 0.1 respectively, blue is replaced by 0.5, red by 0.8

#and grey by 0.1.

ohe=ce.MeanCategoricalEncoder()
X_train=ohe.fit_transform(X_train,y_train)
X_test=ohe.fit_transform(X_test,y_test)
#To ease our job we will create a class 

#This class has function based on various machine learning algorithm

#We are using 5 algorithms which are as follows

#1) Logistic Regression

#2) Ada Boost Regressor

#3) Decision Tree Regressor

#4) Random Forest Regressor

#5) K nearest neighbor Regressor

#

#The specific function when called will do the following:-

#1) Fit the training data

#2) Predict the training data

#3) Predict the testing data

#4) Give output as r2 score, mae and mse for training as well as testing data. 



from sklearn.linear_model import LinearRegression

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

# to evaluate the models

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score

from sklearn import metrics

from sklearn.metrics import mean_absolute_error,mean_squared_error









class selmod():    

    def linr(x_tra,y_tra,x_tes,y_tes):

        print('Linear Regression')

        print('\n')

        lr=LinearRegression()

        lr.fit(x_tra,y_tra)

        

        print('Training Validation','\n')

        predtlr=lr.predict(x_tra)

        print('R2: ','\n',r2_score(y_tra,predtlr))

        print('\n')

        print('mae:',mean_absolute_error(y_tra,predtlr))

        print('\n')

        print('mse:',mean_squared_error(y_tra,predtlr))



        print('*'*80)

        

        print('Testing Validation')

        print('\n')

        predlr=lr.predict(x_tes)

        print('R2: ','\n',r2_score(y_tes,predlr))

        print('\n')

        print('mae:',mean_absolute_error(y_tes,predlr))

        print('\n')

        print('mse:',mean_squared_error(y_tes,predlr))

        print('*'*80)

        

    def dectree(x_tra,y_tra,x_tes,y_tes):

        print('Decision Tree Regressor')

        print('\n')

        dt=DecisionTreeRegressor()

        dt.fit(x_tra,y_tra)

        

        print('Training Validation','\n')

        predtdc=dt.predict(x_tra)

        print('R2: ','\n',r2_score(y_tra,predtdc))

        print('\n')

        print('mae:',mean_absolute_error(y_tra,predtdc))

        print('\n')

        print('mse:',mean_squared_error(y_tra,predtdc))

        print('*'*80)

        

        print('Testing Validation')

        print('\n')

        preddc=dt.predict(x_tes)

        print('R2: ','\n',r2_score(y_tes,preddc))

        print('\n')

        print('mae:',mean_absolute_error(y_tes,preddc))

        print('\n')

        print('mse:',mean_squared_error(y_tes,preddc))

        print('*'*80)

    

    def ranfo(x_tra,y_tra,x_tes,y_tes):

        print('Random Forest Regressor')

        print('\n')

        rf=RandomForestRegressor()

        rf.fit(x_tra,y_tra)

        

        print('Training Validation','\n')

        predtrf=rf.predict(x_tra)

        print('R2: ','\n',r2_score(y_tra,predtrf))

        print('\n')

        print('mae:',mean_absolute_error(y_tra,predtrf))

        print('\n')

        print('mse:',mean_squared_error(y_tra,predtrf))

        print('*'*80)

        

        print('Testing Validation')

        print('\n')

        predrf=rf.predict(x_tes)

        print('R2: ','\n',r2_score(y_tes,predrf))

        print('\n')

        print('mae:',mean_absolute_error(y_tes,predrf))

        print('\n')

        print('mse:',mean_squared_error(y_tes,predrf))

        print('*'*80)

    

    def ada(x_tra,y_tra,x_tes,y_tes):

        print('Ada Boost Regressor')

        print('\n')

        ad=AdaBoostRegressor()

        ad.fit(x_tra,y_tra)

        

        print('Training Validation','\n')

        predtad=ad.predict(x_tra)

        print('R2: ','\n',r2_score(y_tra,predtad))

        print('\n')

        print('mae:',mean_absolute_error(y_tra,predtad))

        print('\n')

        print('mse:',mean_squared_error(y_tra,predtad))

        print('*'*80)

        

        print('Testing Validation')

        print('\n')

        predad=ad.predict(x_tes)

        print('R2: ','\n',r2_score(y_tes,predad))

        print('\n')

        print('mae:',mean_absolute_error(y_tes,predad))

        print('\n')

        print('mse:',mean_squared_error(y_tes,predad))

        print('*'*80)

    

    def kneigh(x_tra,y_tra,x_tes,y_tes):

        print('KNN Regressor')

        print('\n')

        knn=KNeighborsRegressor()

        knn.fit(x_tra,y_tra)

        

        print('Training Validation','\n')

        predtknn=knn.predict(x_tra)

        print('R2: ','\n',r2_score(y_tra,predtknn))

        print('\n')

        print('mae:',mean_absolute_error(y_tra,predtknn))

        print('\n')

        print('mse:',mean_squared_error(y_tra,predtknn))

        print('*'*80)

        

        print('Testing Validation')

        print('\n')

        predknn=knn.predict(x_tes)

        print('R2: ','\n',r2_score(y_tes,predknn))

        print('\n')

        print('mae:',mean_absolute_error(y_tes,predknn))

        print('\n')

        print('mse:',mean_squared_error(y_tes,predknn))

        print('*'*80)
#Linear Regression

selmod.linr(X_train,y_train,X_test,y_test)
#Decision Tree

selmod.dectree(X_train,y_train,X_test,y_test)
#Random Forest

selmod.ranfo(X_train,y_train,X_test,y_test)
#KNN 

selmod.kneigh(X_train,y_train,X_test,y_test)
#Ada Boost

selmod.ada(X_train,y_train,X_test,y_test)
#For this dataset Linear Regression is the best model where our testing accuracy is 0.976 and training accuracy is 0.960
