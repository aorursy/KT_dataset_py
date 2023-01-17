#Import libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
%matplotlib inline

import statsmodels.api as sm 

import time
import profile

import random
import math
import scipy 
#Read Files
events = pd.read_csv ("../input/athlete_events.csv")
data=events
display("Data snapshot:",data.head()) 
#Subsetting Data
data2 = data[data.Season == 'Summer'] #focus only on summer olympics (exclude winter Olympics)
data3 = data2[data2.Sport != 'Art Competitions'] #exclude art competitions
data4 = data3[data3.Sport == 'Athletics'] #examines only Athletics sport
#Key metrics for athletics over years
z = data4.groupby('Year')
 
eventsa = z["Event"].nunique() 
athletesa = z["Name"].nunique() 
countriesa = z["Team"].nunique() 

plt.plot(eventsa, marker='o', markerfacecolor='coral', markersize=7, color='lightcoral', linewidth=2, label='Events')
plt.plot(countriesa, marker='o', markerfacecolor='forestgreen', markersize=7, color='lightgreen', linewidth=2, label='Countries')
plt.plot(athletesa, marker='o', markerfacecolor='green', markersize=8, color='blue', linewidth=3, label='Athletes')
#athletes were not ploted because they have such a diferent scale than the other ones

plt.xlabel('Year')
plt.ylabel('Number/Frequency')
plt.title('Key Statistics of the Athletics Sport (1896 - 2016)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()  
#Medal Count 
data_count2=data4.groupby('NOC').count()
df = pd.DataFrame(data_count2, columns=['Medal'])  #count of medals for each country

# Medal Count by Top 10 Countries:
Sorted3 = df.sort_values(['Medal'], ascending=False)
top_count3 =Sorted3.head(10)

# Plot Top 10 Countries by Medal Count: Athletics Only
explode = (0.08, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025)
#figure=top_count.plot(kind='pie', figsize=(7,5),subplots=True, legend=True, explode = explode, autopct='%1.1f%%', labels=top_count[top_count.columns[0]])
figure=top_count3.plot(kind='pie', figsize=(7,5),subplots=True, legend=None, explode = explode, autopct='%1.1f%%')
#plt.legend(top_count.index, loc='best', bbox_to_anchor=(1.15, 0.5))
plt.axis('equal')
plt.tight_layout()
plt.title("Top 10 Countries by Medal Count: Athletics Only")
plt.show()   
#Athlete participation by country
df3 = pd.DataFrame(data_count2, columns=['Name']) 

sorted4= df3.sort_values(['Name'], ascending=False) #Total number of athletes by country

# Medal Count by Top 10 Countries:
top_count4 =sorted4.head(10)
top_count4.sum()

#Plot Top 10 Countries by  Athlete Count: Athletics Only
explode = (0.08, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.025)
figure=top_count4.plot(kind='pie', figsize=(7,5),subplots=True, explode = explode, autopct='%1.1f%%', legend = None)
#plt.legend(top_count2.index, loc='best', bbox_to_anchor=(1.15, 0.5))
plt.axis('equal')
plt.title("Top 10 Countries by  Athlete Count: Athletics Only")
plt.tight_layout()
plt.show()    
#Box Plot 
import seaborn as sns
dbox2=pd.DataFrame()
dbox2=data4.copy()
dbox2['Medal'].fillna('No Medal', inplace=True)
dbox2.Medal.replace(['Gold', 'Silver','Bronze'], ['Medal', 'Medal','Medal'], inplace=True) 
sns.set(style="ticks", palette="pastel")
sns.set(rc={'figure.figsize':(20,10)})
sns.boxplot(x="Year", y="Age",hue="Medal",palette='Set3',data=dbox2[dbox2.Year>=1960]).set_title('Box plot of age for medal and non-medal athletes overtime (Athletics Only)')
sns.despine(offset=10, trim=True)   
#Define functions

#Define a function to assign countries into clusters per event
def GetClusters(CountryMetrics):
    
    rel_cols=['gold','silver','bronze']
    #Clustering countries for each event based on number of medals won. No of clusters created are 4 (from elbow graph)
    kmeans=KMeans(n_clusters=4).fit(CountryMetrics[rel_cols])
    CountryMetrics['Cluster']=kmeans.labels_
    #g = sns.pairplot(CountryMetrics,hue='Cluster')
    
    return CountryMetrics

#Define a function to get logistic regression 
def GetLogitModel(trainingData):
    
    #Select independent variables
    reg_cols = ['Age' , 'Height', 'Weight' ,'Cluster_1', 'Cluster_2', 'Cluster_3']
    #Run model. Response variable is Medal Won or the likelihood that the participant wins atleast one medal
    logit = sm.Logit(trainingData['MedalWon'], trainingData[reg_cols])
    #fit the model
    result = logit.fit(disp=0)
    
    return result

#Define a function to get Top 6 candidates most likely to win
def GetWinner(testingData,result):
    
    #Data Cleaning
    #print("Initial Rows:",testingData.shape)
    testingData=testingData.dropna(subset=['Height','Weight','Age'])
    #print("Data after Cleaning:",testingData.shape)
    
    #Prediction for testing dataset
    reg_cols = ['Age' , 'Height', 'Weight' ,'Cluster_1', 'Cluster_2', 'Cluster_3']
    prediction=pd.DataFrame(result.predict(testingData[reg_cols]))
    prediction['prediction']=result.predict(testingData[reg_cols]) #Store the prediction values in column prediction

    #Join predictions to testing data dataset
    mergedData=pd.merge(testingData,prediction[['prediction']],left_index=True, right_index=True)
    mergedData['rank']=mergedData.prediction.rank(method='dense', axis=0, ascending=False)

    top6=mergedData.sort_values(by='prediction',ascending=False).head(6)
    top6['Win Probability']=top6.prediction*100
    top6disp=top6[['ID','Name','Sex','Age','Height','Weight','Team','NOC','Year','Medal','MedalWon','Win Probability', 'rank']]
    
    return top6disp
    
#Basic Data Checks
print("Rows X Columns: ",events.shape)
print("Total years of data present:",events.Year.nunique())
print("Number of sports:", events.Sport.nunique())
print("Total unique Events", events.Event.nunique())
#Subset Data for Athletics
DataSport=events[(events.Season=='Summer') & (events.Sport=="Athletics")]
#Check how many years of data is present
DataSport.Year.unique()
#Clustering countries for regression based on number of gold, silver, bronze medals won

#Aggregating medals by country in the entire sport (Athletics)
Gold=DataSport[DataSport.Medal=='Gold']
Gold=pd.DataFrame(Gold["NOC"].value_counts())

Silver=DataSport[DataSport.Medal=='Silver']
Silver=pd.DataFrame(Silver["NOC"].value_counts())

Bronze=DataSport[DataSport.Medal=='Bronze']
Bronze=pd.DataFrame(Bronze["NOC"].value_counts())

j= pd.merge(Gold,Silver,how='outer',left_index=True,right_index=True)
CountryMetrics=pd.merge(j,Bronze,how='outer',left_index=True,right_index=True)

CountryMetrics.columns = ['gold','silver','bronze']
CountryMetrics= CountryMetrics.fillna(0)

#Determining number of clusters
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
%matplotlib inline

#Varying number of clusters
nClusters=range(2,11)
rel_cols=['gold','silver','bronze']
sumDistances=[]
for n in nClusters:
    kmeans=KMeans(n_clusters=n).fit(CountryMetrics[rel_cols])
    sumDistances.append(kmeans.inertia_) #Proxy for SSE

plt.plot(nClusters,sumDistances,'-')
plt.xlabel('nClusters')
plt.ylabel('Sum Of Distances')
plt.show()
#Selecting the years you want to check results for. 
#We will use test results for 3 years 2008, 2012 & 2016.
testyr=DataSport[DataSport.Year>=2008] 
testyr=list(testyr.Year.unique())

YearAccuracy={}

start = time.time()
for yr in testyr:
    
    #Finding out events which have individual participation i.e eliminating team events 
    y=list(set(DataSport.Event))
    z=[]
    for i in y:
        data=DataSport[(DataSport.Event==i) & (DataSport.Year==yr) & (DataSport.Medal=='Gold') ]
        if data.Name.nunique()==1:
            z.append(i)
    #print(len(z))
    
    startyear=time.time()
    print("Predicting results for year: ",yr)
    MedalWinners=[]             #Storing how many medal winners the model is able to predict per event
    
    for EventName in z:
        startevent=time.time()
        #-----------------Clustering countries based on medals won in the event in previous years---------------#
        DataEvents=DataSport[(DataSport.Event==EventName)]
        
        #Get medal count for each country for selected event
        Gold=DataEvents[DataEvents.Medal=='Gold']
        Gold=pd.DataFrame(Gold["NOC"].value_counts())

        Silver=DataEvents[DataEvents.Medal=='Silver']
        Silver=pd.DataFrame(Silver["NOC"].value_counts())

        Bronze=DataEvents[DataEvents.Medal=='Bronze']
        Bronze=pd.DataFrame(Bronze["NOC"].value_counts())

        Participants= pd.DataFrame(DataEvents["NOC"].value_counts())

        j= pd.merge(Gold,Silver,how='outer',left_index=True,right_index=True)
        j1=pd.merge(j,Bronze,how='outer',left_index=True,right_index=True)
        CountryMetrics=pd.merge(j1,Participants,how='outer',left_index=True,right_index=True)
        CountryMetrics.columns = ['gold','silver','bronze','participants']
        CountryMetrics= CountryMetrics.fillna(0)
        
        #Get Clusters
        CountryMetrics=GetClusters(CountryMetrics)

        #--------------------------------------PREPARING TRAINING DATA-------------------------------------#
        #Join Cluster Number to subsetted data
        EventData=pd.merge(DataEvents,CountryMetrics,left_on='NOC',right_index=True)

        #Assign 1 if participant has won a medal assign 0 if participant has not won
        EventData['MedalWon']=EventData.Medal.notna()
        EventData.MedalWon=EventData.MedalWon.astype(int)

        #Create dummy variable for cluster
        dummy_Cluster = pd.get_dummies(EventData['Cluster'], prefix='Cluster')

        EventData = EventData.join(dummy_Cluster.loc[:, 'Cluster_1':])

        #Logistic Regression on training dataset. Training dataset is roughly 80% of data
        trainingData=EventData[EventData.Year<2008]
        #print("Data subsetted for no of years: ",trainingData.Year.nunique())

        #Data Cleaning
        #print("Intial rows:", trainingData.shape)
        trainingData=trainingData.dropna(subset=['Height','Weight','Age'])
        #print("Data after cleaning:",trainingData.shape)
        
        #-------------------------------------LOGISTIC REGRESSION--------------------------------------------#

        if len(trainingData)>2:
            #Get Logistic regression result
            result=GetLogitModel(trainingData)
            #print(result.summary2())        

            #-----------------------------------Testing the model--------------------------------------------#

            #Testing Dataset
            testingData=EventData[EventData.Year==yr]
            print("Event: ",EventName)

            #Get top 6 predicted winners for the year
            top6=GetWinner(testingData,result)
            print("\nTop 6 likely to win:")
            display(top6)

            #Calculating accuracy of prediction out of top 6
            ActualWinners=sum(top6.MedalWon)       #Participants who actually won a medal out of top 6

            Accuracy=(ActualWinners/3)*100         #How many of the top 6 predictions contained all medal winners

            print("The model predicts",ActualWinners,"of the 3 medal winners in Top 6")
            #print("The accuracy of model is: ", Accuracy)

            MedalWinners.append([EventName,ActualWinners])
            endEvent=time.time()

    mw=pd.DataFrame(MedalWinners,columns=['EventName', 'ActualWinners'])
    
    YearAccuracy[yr]=mw
    
    endyear=time.time()
endall=time.time()
for yr in testyr:
    
    print("\nPredictions for: ", yr)
    mw=YearAccuracy[yr]
    print("No of events predicted for: ",len(mw))
    acc=mw['ActualWinners'].sum()/(len(mw)*3)
    print("Accuracy: ", acc*100)
    
    print("Predicts all winners in top 6 for: ",len(mw[mw['ActualWinners']==3])," events")
    print("Predicts 2 of 3 winners in top 6 for: ",len(mw[mw['ActualWinners']==2])," events")
    print("Predicts 1 of 3 winners in top 6 for: ",len(mw[mw['ActualWinners']==1])," events")
    print("Predicts 0 of 3 winners in top 6 for: ",len(mw[mw['ActualWinners']==0])," events")
    
