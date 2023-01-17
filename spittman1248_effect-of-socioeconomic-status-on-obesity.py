
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as np
import sys
import matplotlib 
import seaborn as sns
import numpy as np
from subprocess import check_output
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

%matplotlib inline


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Location of file
Location = '../input/Nutrition__Physical_Activity__and_Obesity_-_Behavioral_Risk_Factor_Surveillance_System.csv'

df = pd.read_csv(Location)

df.info()
df.head(5)
#Getting Rid of All Extraneous Info

df.drop(['Low_Confidence_Limit','High_Confidence_Limit ','YearEnd','Topic','Class','Datasource','Data_Value_Unit','QuestionID','ClassID','TopicID','DataValueTypeID','Data_Value_Type','Data_Value_Footnote_Symbol','Data_Value_Footnote','StratificationCategoryId1','StratificationID1'],1);

#Create separate Dataform from df by gender df2, by education level dfedu, and by income dfedu

df2=df[(df['Stratification1']=='Male')|(df['Stratification1']=='Female')]
dfedu=df[df['StratificationCategory1']=='Education']
dfinc=df[df['StratificationCategory1']=='Income']

#reset index for each of the new dataforms

df2 = df2.reset_index(drop = True)
dfedu = dfedu.reset_index(drop = True)
dfinc = dfinc.reset_index(drop = True)
#Each category has the same survey questions

df2['Question'].unique()
#here we are interested in the survey question directly about obesity and overweight percent

X=['Percent of adults aged 18 years and older who have obesity','Percent of adults aged 18 years and older who have an overweight classification']


df2=df2[df2['Question']==X[0]]

#In case we wanted both. df3=df2[df2['Question'].apply(lambda x: x in X)]
#survey data covers 2011 - 2014 (all states) or 2016 most states
#choose 2014 since it has the most data 

df2=df2[df2['YearStart']==2014]

#separate out national so that we can calculate the national obesity rate for 2014
df2n=df2[(df2['LocationDesc']=='National')]

#Cut out terriotories that our not included within 50 states + DC data
df2=df2[~(df2['LocationDesc']=='National')]
df2=df2[~(df2['LocationDesc']=='Guam')]
df2=df2[~(df2['LocationDesc']=='Puerto Rico')]
df2['LocationDesc'].unique()
#group data by state and take the mean of men and women rates for each state

sorted_df = df2.sort_values(['LocationDesc'], ascending = [True])
sorted_df=sorted_df[['LocationAbbr','LocationDesc','Data_Value','Gender']]
sorted_df = sorted_df.groupby('LocationDesc', as_index=False).mean()

#calculate the average (over men and women) obesity rate for the country

natmeanobesity2014=sum(df2n['Data_Value'])/len(df2n)
print(natmeanobesity2014)
#Let's plot a bar graph of the most and least obese states in the US
#Mark the national average in red

#For those that have LaTex
#plt.rc('text', usetex=True)

plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 14})

sorted_df = sorted_df.sort_values(['Data_Value'], ascending = [True])

plt.figure(figsize = (10,16))

plt.subplot(2,1,1)
ax=sns.barplot(y=sorted_df.tail(10).LocationDesc,x=sorted_df.tail(10).Data_Value,palette="Blues_d")
ax.set_ylabel('US State')
ax.set_xlabel('Obesity Rate (%)')
ax.set_title('10 Most Obese States in 2014')

plt.plot([natmeanobesity2014,natmeanobesity2014],[-1,10], '--',color = 'r')

plt.subplot(2,1,2)
ax=sns.barplot(y=sorted_df.head(10).LocationDesc,x=sorted_df.head(10).Data_Value,palette="Blues_d")
ax.set_ylabel('US State')
ax.set_xlabel('Obesity Rate (%)')
ax.set_title('10 Least Obese States in 2014')

plt.plot([natmeanobesity2014,natmeanobesity2014],[-1,10], '--',color = 'r')




!pip install plotly

sorted_df = df2.sort_values(['LocationDesc'], ascending = [True])
sorted_df=sorted_df[['LocationAbbr','LocationDesc','Data_Value','Gender']]
sorted_df2 = sorted_df.groupby('LocationAbbr', as_index=False).mean()
#Let's make a map to see the geographic locations of the obesity rates

import plotly.plotly as py


scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]


data = [ dict(
        type='choropleth',
        colorscale = 'YlOrRd',
        autocolorscale = False,
        reversescale = True,
        locations = sorted_df2['LocationAbbr'],
        z = sorted_df2['Data_Value'].astype(float),
        locationmode = 'USA-states',
        text = sorted_df2['LocationAbbr'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "% Obesity")
        ) ]

layout = dict(
        #title = '2011 US Agriculture Exports by State<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )
py.iplot( fig, filename ='somename' )

#Now let's explore if obesity is somehow correlated with education level

dfedu.head()
#Just like for the gender one, we need to isolate just the obesity question and the year


X=['Percent of adults aged 18 years and older who have obesity','Percent of adults aged 18 years and older who have an overweight classification']

#df3=df2[df2['Question'].apply(lambda x: x in X)]
dfedu=dfedu[dfedu['Question']==X[0]]
dfedu=dfedu[dfedu['YearStart']==2014]

#Cut out all locations that aren't within the 50 states + DC
dfedu=dfedu[~(dfedu['LocationDesc']=='National')]
dfedu=dfedu[~(dfedu['LocationDesc']=='Guam')]
dfedu=dfedu[~(dfedu['LocationDesc']=='Puerto Rico')]
dfedu['LocationDesc'].unique()
#select the four relevant columns to analyze the obesity rate versus educational level

dfedu = dfedu.reset_index(drop = True)
dfedu=dfedu[['YearStart','LocationDesc','Data_Value','Education']]
dfedu.head(10)
#Create a list of all 4 educational levels
ledu=dfedu.Education.unique()


#want to treat education levels as dummy variables, so this assigns 1 or 0 depending on the group
for i in ledu:
    dfedu[i]=dfedu['Education'].apply(lambda x: int(x==i))

#select the four relevant columns to analyze the obesity rate versus educational level

dfedu = dfedu.reset_index(drop = True)
dfedu=dfedu[['YearStart','LocationDesc','Data_Value','Education']]
dfedu.head(10)
#Create a list of all 4 educational levels
ledu=dfedu.Education.unique()


#want to treat education levels as dummy variables, so this assigns 1 or 0 depending on the group
for i in ledu:
    dfedu[i]=dfedu['Education'].apply(lambda x: int(x==i))

#Let's just take a look at how the obesity varies amongst states for those without highschool
#From the bar graph below, it is clear that Wyomming is an outlier. 

#plt.rc('text', usetex=True)
plt.rcParams.update(plt.rcParamsDefault)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 14})



dfeduLHS=dfedu[dfedu[ledu[0]]==1]
dfeduLHS = dfeduLHS.reset_index(drop = True)

plt.figure(figsize = (10,16))
ax=sns.barplot(y=dfeduLHS.LocationDesc,x=dfeduLHS.Data_Value,palette="Blues_d")
ax.set_ylabel('US State')
ax.set_xlabel('Obesity Rate for People Without Highschool education (%)')

plt.show()

#Let's see how obesity compares for those with a HS education

dfeduHS=dfedu[dfedu[ledu[1]]==1]
dfeduHS = dfeduHS.reset_index(drop = True)


plt.figure(figsize = (10,16))
ax=sns.barplot(y=dfeduHS.LocationDesc,x=dfeduHS.Data_Value,palette="Blues_d")
ax.set_ylabel('US State')
ax.set_xlabel('Obesity Rate for People With Highschool Education (%)')


#plt.plot(df.non_weighted_all_weekly, df.M_weekly,'o')
#plt.plot(df.non_weighted_all_weekly, df.F_weekly,'o')
#plt.legend(['Males','Females'])
#plt.xlabel('Field Median Salary')
#plt.ylabel('Salary')
plt.show()
#Wyoming is an outlier AND has a small population and is very rural (i.e. different than other states) 
#I am going to take Wyoming it out of the Linear Regression Data Set without much harm

#Note that since I have 4 dummy variables, need only three coefficients 
dfedu=dfedu[~(dfedu['LocationDesc']=='Wyoming')]
model = LinearRegression()
columns = dfedu.columns[5:8]
X = dfedu[columns]

X_std = StandardScaler().fit_transform(X)
y = dfedu['Data_Value']

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.18, random_state=42)

model.fit(X_train,y_train)

plt.barh([0,1,2],model.coef_)
plt.yticks(range(3),dfedu.columns[4:7], fontsize = 10)
plt.title('Regression Coefficients')

plt.show()

#Regression R^2 value shows that lack of education has a "moderate" effect on obesity rate
print('R^2 on training...',model.score(X_train,y_train))
print('R^2 on test...',model.score(X_test,y_test))

print('Model Coefficients',model.coef_)
print('Model Coefficients',model.intercept_)
dfinc.columns
#Now let's look at the effect income has on obesity

X=['Percent of adults aged 18 years and older who have obesity','Percent of adults aged 18 years and older who have an overweight classification']

#df3=df2[df2['Question'].apply(lambda x: x in X)]
dfinc=dfinc[dfinc['Question']==X[0]]
dfinc=dfinc[dfinc['Question']==X[0]]
dfinc=dfinc[dfinc['YearStart']==2014]
dfinc=dfinc[~(dfinc['LocationDesc']=='National')]
dfinc=dfinc[~(dfinc['LocationDesc']=='Guam')]
dfinc=dfinc[~(dfinc['LocationDesc']=='Puerto Rico')]

dfinc['LocationDesc'].unique()
dfinc.Income.unique()
dfinc = dfinc.reset_index(drop = True)
dfinc=dfinc[['YearStart','LocationDesc','Data_Value','Income']]

linc=dfinc.Income.unique()

#Create Dummy Variables from the income
for i in linc:
    dfinc[i]=dfinc['Income'].apply(lambda x: int(x==i))


dfinc=dfinc[~(dfinc.Income=='Data not reported')]

dfinc.head(10)
#Let's test out whether income has an effect on obesity



model = LinearRegression()
columns = dfinc.columns[4:9]
X = dfinc[columns]

#X_std = StandardScaler().fit_transform(X)
y = dfinc['Data_Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)

model.fit(X_train,y_train)

plt.figure(figsize = (10,10))
plt.barh([0,1,2,3,4],model.coef_)
plt.yticks(range(5),dfinc.columns[4:9], fontsize = 10)
plt.title('Regression Coefficients')

plt.show()

#Yikes!  Such low regression coefficients illustrate that there is only a weak effect if any
print('R^2 on training...',model.score(X_train,y_train))
print('R^2 on test...',model.score(X_test,y_test))

print('Model Coefficients',model.coef_)
print('Model Coefficients',model.intercept_)  
dfavginc = dfinc.groupby('Income', as_index=False).mean()
#Let's look out the mean obesity rate for each income category to investigate

dfavginc['IncomeOrder']=[1,2,3,4,5,0]


sorted_df = dfavginc.sort_values(['IncomeOrder'], ascending = [True])
plt.figure(figsize = (10,10))



#National average marked in red
#From this graph, we see that obesity doesn't monotonically decrease with increasing income.  
##In a sense, these income categories really depend on the cost of living in each state.  
#For example, $75K will buy you a lot in MS and places you in high income bracket, but not necessarily in MA   

sorted_df.Data_Value.plot(kind='barh')

plt.yticks(range(6),sorted_df.Income, fontsize = 12,family='serif')
plt.plot([natmeanobesity2014,natmeanobesity2014],[-0.5,5.5], '--',color = 'r')
plt.title('Mean Obesity By Income Group in 2014',family='serif')
plt.xlim([0,38])
