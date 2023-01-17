# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import datetime as dt

from datetime import timedelta



from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score, silhouette_samples

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error,mean_squared_log_error, r2_score, make_scorer

from sklearn.preprocessing import PolynomialFeatures



import scipy.cluster.hierarchy as sch



import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf





import plotly.express as px

import plotly.offline as py

import plotly.graph_objs as go

from plotly.subplots import make_subplots

py.init_notebook_mode(connected= True)



from fbprophet import Prophet





import warnings

warnings.filterwarnings('ignore')
df1= pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df1.head(5)
print('Shape of the dataset: \n', df1.shape,'\n\n')

print('Check for NULL values: \n', df1.isnull().sum(), '\n\n')

print('Datatype of each independent variable: ', df1.dtypes)
# Dropping the SNo column



df1.drop(["SNo"], 1, inplace= True)

df1.head(2)
# Converting the given date-time format into pandas provided datetime format



df1['ObservationDate']= pd.to_datetime(df1['ObservationDate'])

df1['Last Update']= pd.to_datetime(df1['Last Update'])

df1.head(2)
# Check if the data is updated

print("Dataset Description")

print("Earliest Entry: ",df1['ObservationDate'].min())

print("Last Entry:    ",df1['ObservationDate'].max())

print("Total Days:    ",(df1['ObservationDate'].max() - df1['ObservationDate'].min()))
# Grouping the dataframe by 'ObservationDate' and 'Country/Region' and aggregating with 'Confirmed', 'Recovered' and 'Death' cases.



cou_grp= df1.groupby(['Country/Region','ObservationDate']).agg({

                                                "Confirmed": 'sum',

                                                "Recovered": 'sum',

                                                "Deaths": 'sum'

})



cou_grp.head(2)
# Adding a column of 'Active Cases' in the cou_grp dataframe

cou_grp['Active Cases']= cou_grp['Confirmed']- cou_grp['Recovered']- cou_grp['Deaths']



# applying and adding coumns for log transformation to 'Confirmed' and 'Active' columns in the cou_grp dataframe to initiate the removal of skewness in them if any.

cou_grp['log_confirmed']= np.log(cou_grp['Confirmed'])

cou_grp['log_active']= np.log(cou_grp['Active Cases'])



cou_grp.head(2)
# Creating a temporary dataframe by grouping different types of cases present in the new dataframe as per date and aggregating with 'Confirmed', 'Recovered' and 'Deaths' cases



date_wise= df1.groupby(['ObservationDate']).agg({

                            "Confirmed": 'sum',

                            "Recovered": 'sum',

                            "Deaths": 'sum'})



#Adding a column 'Days Since' in 'date_wise' dataframe to keep the count of the days from initial date



date_wise['Days Since']= date_wise.index- date_wise.index.min()



date_wise.head(2)
print("No. of Countries suffering: ",len(df1['Country/Region'].unique()))

print("Total Confirmed Cases worldwide:  ",date_wise['Confirmed'].iloc[-1])

print("Total Recovered Cases worldwide:  ",date_wise['Recovered'].iloc[-1])

print("Total Death Cases worldwide:  ",date_wise['Deaths'].iloc[-1])

#print("Total Active Cases worldwide:  ",(date_wise['Confirmed']- date_wise['Recovered']- date_wise['Deaths'])

#print("Total Closed Cases worldwide:  ",date_wise['Recovered'] + date_wise['Deaths'])







print("Confirmed Cases/Day worldwide: ",np.round(date_wise['Confirmed'].iloc[-1]/date_wise.shape[0]))

print("Recovered Cases/Day worldwide: ",np.round(date_wise['Recovered'].iloc[-1]/date_wise.shape[0]))

print("Deaths Cases/Day worldwide: ",np.round(date_wise['Deaths'].iloc[-1]/date_wise.shape[0]))





print("Confirmed Cases/Hour worldwide: ",np.round(date_wise['Confirmed'].iloc[-1]/date_wise.shape[0]*24))

print("Recovered Cases/Hour worldwide: ",np.round(date_wise['Recovered'].iloc[-1]/date_wise.shape[0]*24))

print("Death Cases/Hour worldwide: ",np.round(date_wise['Deaths'].iloc[-1]/date_wise.shape[0]*24))





print("Confirmed Cases in last 24 hrs: ",date_wise['Confirmed'].iloc[-1]-date_wise['Confirmed'].iloc[-2])

print("Recovered Cases in last 24 hrs: ",date_wise['Recovered'].iloc[-1]-date_wise['Recovered'].iloc[-2])

print("Death Cases in last 24 hrs: ",date_wise['Deaths'].iloc[-1]-date_wise['Deaths'].iloc[-2])
# Distribution of Number of Active Cases



fig= px.bar(x=date_wise.index, y=date_wise["Confirmed"]- date_wise["Recovered"]- date_wise["Deaths"])

fig.update_layout(title="Distribution of number of Active cases", xaxis_title="Date", yaxis_title="Number of Cases")

fig.show()
# Distribution of number of Closed cases



fig= px.bar(x= date_wise.index, y= date_wise['Recovered']+ date_wise['Deaths'])

fig.update_layout(title=" Distribution of number of Closed Cases", xaxis_title="Date", yaxis_title="Number of Cases")

fig.show()
# Weekly Growth of Different types of cases(Recovered, Confirmed, Deaths)





date_wise["WeekOfYear"]= date_wise.index.weekofyear



week_num=[]

week_wise_confirmed= []

week_wise_recovered= []

week_wise_deaths= []



w=1



for i in list(date_wise["WeekOfYear"].unique()):

                        week_wise_confirmed.append(date_wise[date_wise['WeekOfYear']==i]["Confirmed"].iloc[-1])

                        week_wise_recovered.append(date_wise[date_wise['WeekOfYear']==i]["Recovered"].iloc[-1])

                        week_wise_deaths.append(date_wise[date_wise['WeekOfYear']==i]["Deaths"].iloc[-1])

                        week_num.append(w)

                        w+=1



fig= go.Figure()

fig.add_trace(go.Scatter(x= week_num, y= week_wise_confirmed, mode='lines+markers', name= 'Weekly Growth of Confirmed cases'))

fig.add_trace(go.Scatter(x= week_num, y= week_wise_recovered, mode='lines+markers', name= 'Weekly Growth of Recovered cases'))

fig.add_trace(go.Scatter(x= week_num, y= week_wise_deaths, mode='lines+markers', name= 'Weekly Growth of Death cases'))



fig.update_layout(title="Weekly Growth of Different types of Cases worldwide ", xaxis_title='Week Number', yaxis_title='Number of Cases', legend= dict(x=0, y=1, traceorder='normal'))

fig.show()                                                   

                 
# Weekly variation of confirmed and Death cases



fig, (ax1, ax2)= plt.subplots(1,2, figsize=(15,5))



sns.barplot(x= week_num, y= pd.Series(week_wise_confirmed).diff().fillna(0), ax= ax1)

sns.barplot(x= week_num, y= pd.Series(week_wise_deaths).diff().fillna(0), ax= ax2)\



ax1.set_xlabel("Week Number")

ax1.set_ylabel("Number of Confirmed Cases")

ax1.set_title("Weekly variation in number of Confirmed cases")



ax2.set_xlabel("Week Number")

ax2.set_ylabel("Number of Death Cases")

ax2.set_title("Weekly variation in number of Death cases")

# Growth of different types of the cases( Confirmed, Recovered, Deaths)



fig= go.Figure()



fig.add_trace(go.Scatter(x=date_wise.index, y=date_wise['Confirmed'], mode='lines+markers', name='Confirmed Cases'))

fig.add_trace(go.Scatter(x=date_wise.index, y=date_wise['Recovered'], mode='lines+markers', name='Recovered Cases'))

fig.add_trace(go.Scatter(x=date_wise.index, y=date_wise['Deaths'], mode='lines+markers', name='Death Cases'))



fig.update_layout(title=" Growth of Different types of cases", xaxis_title="Date", yaxis_title="Number of Cases", legend= dict(x=0, y=1, traceorder="normal"))

fig.show()
# Calculation of Mortality Rate and Recovery Rate



date_wise['Mortality Rate']= (date_wise['Deaths']/date_wise['Confirmed'])*100

date_wise['Recovery Rate']= (date_wise['Recovered']/date_wise['Confirmed'])*100

date_wise['Active Cases']= date_wise['Confirmed']- date_wise['Recovered']- date_wise['Deaths']

date_wise['Closed Cases']= date_wise['Recovered']+ date_wise['Deaths']



print("Average Mortality Rate :", date_wise['Mortality Rate'].mean())

print("Median Mortality Rate :", date_wise['Mortality Rate'].median())

print("Average Recovery Rate :", date_wise['Recovery Rate'].mean())

print("Median Recovery Rate :", date_wise['Recovery Rate'].median())



# Plotting Mortality and Recoevry Rate



fig= make_subplots(rows=2, cols=1, subplot_titles=("Recovery Rate", "Mortality Rate"))



fig.add_trace(

    go.Scatter(x=date_wise.index, y=(date_wise["Recovered"]/date_wise["Confirmed"])*100,name="Recovery Rate"),

    row=1, col=1

)

fig.add_trace(

    go.Scatter(x=date_wise.index, y=(date_wise["Deaths"]/date_wise["Confirmed"])*100,name="Mortality Rate"),

    row=2, col=1

)



fig.update_layout(height=1000, legend=dict(x=0, y=1, traceorder='normal'))

fig.update_xaxes(title_text="Date", row=1, col=1)

fig.update_yaxes(title_text="Recovery Rate", row=1, col=1)

fig.update_xaxes(title_text="Date", row=1, col=2)

fig.update_yaxes(title_text="Mortality Rate", row=1, col=2)

fig.show()
# Daily variation of different types (Recovered, Confirmed, Deaths) of cases



print("Average Daily increase in number of Confirmed Cases: ", np.round(date_wise['Confirmed'].diff().fillna(0).mean()))

print("Average Daily increase in number of Recovered Cases: ", np.round(date_wise['Recovered'].diff().fillna(0).mean()))

print("Average Daily increase in number of Death Cases: ", np.round(date_wise['Deaths'].diff().fillna(0).mean()))



fig= go.Figure()

fig.add_trace(go.Scatter(x=date_wise.index, y= date_wise['Confirmed'].diff().fillna(0), mode='lines+markers', name='Confirmed cases'))

fig.add_trace(go.Scatter(x=date_wise.index, y= date_wise['Recovered'].diff().fillna(0), mode='lines+markers', name='Recovered cases'))

fig.add_trace(go.Scatter(x=date_wise.index, y= date_wise['Deaths'].diff().fillna(0), mode='lines+markers', name='Death cases'))



fig.update_layout(title='Daily variattion of different types (Recovered, Confirmed, Deaths) of cases', xaxis_title='Date', yaxis_title='Number of Cases', legend= dict(x=0,y=1, traceorder= 'normal'))

fig.show()




fig= go.Figure()

fig.add_trace(go.Scatter(x=date_wise.index, y=date_wise['Confirmed'].diff().rolling(window=7).mean(), mode='lines+markers', name='Confirmed cases'))

fig.add_trace(go.Scatter(x=date_wise.index, y=date_wise['Recovered'].diff().rolling(window=7).mean(), mode='lines+markers', name='Recovered cases'))

fig.add_trace(go.Scatter(x=date_wise.index, y=date_wise['Deaths'].diff().rolling(window=7).mean(), mode='lines+markers', name='Death cases'))



fig.update_layout(title='7 days rolling Mean of daily Increase of Confirmed, Recovered and Death Cases', xaxis_title='Date', yaxis_title='Number of Cases', legend=dict(x=0,y=1, traceorder='normal'))

fig.show()
# Datewise Growth Factor of Active and Closed Cases



fig= go.Figure()

fig.add_trace(go.Scatter(x=date_wise.index, y=(date_wise['Confirmed']-date_wise['Recovered']- date_wise['Deaths'])/(date_wise['Confirmed']-date_wise['Recovered']- date_wise['Deaths']).shift(), mode='lines', name=' Growth factor of Active cases'))

fig.add_trace(go.Scatter(x=date_wise.index, y=(date_wise['Recovered']+ date_wise['Deaths'])/(date_wise['Recovered']+date_wise['Deaths']).shift(), mode='lines', name=' Growth factor of Closed cases'))



fig.update_layout(title='Datewise Growth Factor of Active and Closed Cases', xaxis_title='Date', yaxis_title='Growth Factor', legend=dict(x=0,y=1, traceorder='normal'))

fig.show()
# Calculation of Countrywise mortality and recovery rate



country_wise= df1[df1['ObservationDate']== df1['ObservationDate'].max()].groupby(['Country/Region']).agg({

    "Confirmed": 'sum',

    "Recovered": 'sum',

    "Deaths": 'sum'

}).sort_values(['Confirmed'], ascending= False)



# adding two more columns to the country_wise dataframe named 'Mortality' and 'Recovery' which will be keeping the values of Recovery and Mortality rate.



country_wise['Mortality']= (country_wise['Deaths']/country_wise['Confirmed'])*100

country_wise['Recovery']= (country_wise['Recovered']/country_wise['Confirmed'])*100
country_wise_last_24_confirmed=[]

country_wise_last_24_recovered=[]

country_wise_last_24_deaths=[]



for country in country_wise.index:

    country_wise_last_24_confirmed.append((cou_grp.loc[country].iloc[-1]- cou_grp.loc[country].iloc[-2])['Confirmed'])

    country_wise_last_24_recovered.append((cou_grp.loc[country].iloc[-1]- cou_grp.loc[country].iloc[-2])['Recovered'])

    country_wise_last_24_deaths.append((cou_grp.loc[country].iloc[-1]- cou_grp.loc[country].iloc[-2])['Deaths'])
last_day_country_wise= pd.DataFrame(list(zip(country_wise.index, country_wise_last_24_confirmed, country_wise_last_24_recovered, country_wise_last_24_deaths)),

                                    columns=["Country Name", "Last 24 Hours Confirmed", "Last 24 Hours Recovered", "Last 24 Hours Deaths"])
top_15_confirmed_last24= last_day_country_wise.sort_values(['Last 24 Hours Confirmed'], ascending= False).head(15)

top_15_recovered_last24= last_day_country_wise.sort_values(['Last 24 Hours Recovered'], ascending= False).head(15)

top_15_deaths_last24= last_day_country_wise.sort_values(['Last 24 Hours Deaths'], ascending= False).head(15)





fig, (ax1, ax2, ax3)= plt.subplots(3,1, figsize=(10,20))



sns.barplot(x= top_15_confirmed_last24['Last 24 Hours Confirmed'], y=top_15_confirmed_last24['Country Name'], ax= ax1)

ax1.set_title("Top 15 Countries with highest number of confirmed cases in last 24 hours")



sns.barplot(x= top_15_recovered_last24['Last 24 Hours Recovered'], y=top_15_recovered_last24['Country Name'], ax= ax2)

ax2.set_title("Top 15 Countries with highest number of Recovered cases in last 24 hours")



sns.barplot(x= top_15_deaths_last24['Last 24 Hours Deaths'], y=top_15_deaths_last24['Country Name'], ax= ax3)

ax3.set_title("Top 15 Countries with highest number of Death cases in last 24 hours")
# Proportion of countries in confirmed, recovered and death cases



last_day_country_wise['Proportion got Confirmed']= (last_day_country_wise['Last 24 Hours Confirmed']/(date_wise['Confirmed'].iloc[-1]- date_wise['Confirmed'].iloc[-2]))*100

last_day_country_wise['Proportion got Recovered']= (last_day_country_wise['Last 24 Hours Recovered']/(date_wise['Recovered'].iloc[-1]- date_wise['Recovered'].iloc[-2]))*100

last_day_country_wise['Proportion got dead']= (last_day_country_wise['Last 24 Hours Deaths']/(date_wise['Deaths'].iloc[-1]- date_wise['Deaths'].iloc[-2]))*100





last_day_country_wise[['Country Name', 'Proportion got Confirmed', 'Proportion got Recovered', 'Proportion got dead']].sort_values(['Proportion got Confirmed', 'Proportion got Recovered', 'Proportion got dead'], ascending= False).style.background_gradient(cmap="Reds")
# Top 15 Countries as per number of confirmed and death cases



fig, (ax1, ax2)= plt.subplots(2,1, figsize=(10,12))



top_15_countries_confirmed= country_wise.sort_values(['Confirmed'], ascending= False).head(15)

top_15_countries_deaths= country_wise.sort_values(['Deaths'], ascending= False).head(15)



sns.barplot(x= top_15_countries_confirmed['Confirmed'], y= top_15_countries_confirmed.index, ax=ax1)

sns.barplot(x= top_15_countries_deaths['Deaths'], y= top_15_countries_deaths.index, ax=ax2)



ax1.set_title("Top 15 countries as per no. of Confirmed Cases")

ax2.set_title("Top 15 countries as per no. of Death Cases")
# Top 10 Countries as per Mortality and Recovery Rate with more than 1000 Confirmed cases



fig, (ax1, ax2)= plt.subplots(2,1, figsize=(10,15))



mortality_country_wise= country_wise[country_wise['Confirmed']>1000].sort_values(['Mortality'], ascending= False).head(10)

recovery_country_wise= country_wise[country_wise['Confirmed']>1000].sort_values(['Recovery'], ascending= False).head(10)



sns.barplot(x= mortality_country_wise['Mortality'], y=mortality_country_wise.index, ax= ax1)

sns.barplot(x= recovery_country_wise['Recovery'], y=recovery_country_wise.index, ax= ax2)



ax1.set_title("Top 10 Countries according to high Mortality Rate (Confirmed Cases > 1000)")

ax2.set_title("Top 10 Countries according to high Recovery Rate (Confirmed Cases > 1000)")



ax1.set_xlabel("Mortality (%)")

ax2.set_xlabel("Recovery (%)")

# Top 10 Countries as per Survival Probability



fig, (ax1)= plt.subplots(1,1, figsize=(10,15))



country_wise['Survival Probability']=(1-(country_wise['Deaths']/country_wise['Confirmed']))*100



top_10_countries_sur_prob= country_wise[country_wise['Confirmed']> 1000].sort_values(['Survival Probability'], ascending=False).head(50)



sns.barplot(x=top_10_countries_sur_prob['Survival Probability'], y=top_10_countries_sur_prob.index, ax=ax1)

ax1.set_title("Top 10 Countries with Maximum Survival Probability in % (Confirmed cases >1000)")



print("Mean of Survival Probability: ", country_wise['Survival Probability'].mean())

print("Median of Survival Probability: ", country_wise['Survival Probability'].median())

print("Mean of Death Probability: ", 100-country_wise['Survival Probability'].mean())

print("Median of Death Probability: ", 100-country_wise['Survival Probability'].median())

fig= go.Figure()



for country in country_wise.head(10).index:

    fig.add_trace(go.Scatter(x=cou_grp.loc[country]['log_confirmed'], y= cou_grp.loc[country]['log_active'], mode='lines', name= country))

    fig.update_layout(height=600, title='Transition of some worstly affected countries with Active vs Confirmed cases',

                     xaxis_title='Confirmed Cases (Log Scale)', yaxis_title='Active Cases (log Scale)', legend= dict(x=0, y=1, traceorder='normal'))



fig.show()
X= country_wise[['Mortality', 'Recovery']]



# Preprocessing the data with standardisation and normalisation as KMeans Clustering works well with normalised data

std= StandardScaler()

X= std.fit_transform(X)





ss=[]   # Initiating array for Sum of Squares

sil=[]  # Initiating array for silhoutte scores



for i in range(2,11):

    clf= KMeans(n_clusters=i, init='k-means++', random_state=42)

    clf.fit(X)

    labels= clf.labels_

    centroids= clf.cluster_centers_

    sil.append(silhouette_score(X, labels, metric="euclidean"))

    ss.append(clf.inertia_)

    

x= np.arange(2,11)

plt.figure(figsize=(10,5))

plt.plot(x, ss, marker='o')

plt.xlabel('Number of Clusters')

plt.ylabel("Within Cluster Sum of Squares")

plt.title("Elbow Curve")

plt.figure(figsize=(20,15))

dendogram= sch.dendrogram(sch.linkage(X, method="ward"))
# Taking k=3 for KMeans Clustering and observing the summary



clf_fin= KMeans(n_clusters=3, init="k-means++", random_state=6)

clf_fin.fit(X)



country_wise['Clusters']= clf_fin.predict(X)





# Summary of Clustering





summary= pd.concat([country_wise[country_wise['Clusters']==1].head(10), country_wise[country_wise['Clusters']==2].head(10), country_wise[country_wise['Clusters']==0].head(10)])

summary.style.background_gradient(cmap='Reds').format("{:.2f}")

# Statistical findings from Clusters



print("Average Mortality rate in Cluster 0: ", country_wise[country_wise['Clusters']==0]['Mortality'].mean())

print("Average Recovery rate in Cluster 0: ", country_wise[country_wise['Clusters']==0]['Recovery'].mean())



print("Average Mortality rate in Cluster 1: ", country_wise[country_wise['Clusters']==1]['Mortality'].mean())

print("Average Recovery rate in Cluster 1: ", country_wise[country_wise['Clusters']==1]['Recovery'].mean())



print("Average Mortality rate in Cluster 2: ", country_wise[country_wise['Clusters']==2]['Mortality'].mean())

print("Average Recovery rate in Cluster 2: ", country_wise[country_wise['Clusters']==2]['Recovery'].mean())
# Observing it in  plot



plt.figure(figsize=(10,5))

sns.scatterplot(x= country_wise['Recovery'], y=country_wise['Mortality'], hue=country_wise['Clusters'], s=100)



plt.axvline(((date_wise['Recovered']/date_wise['Confirmed'])*100).mean(), color='green', linestyle="--", label='Mean Recovery rate worldwide')

plt.axhline(((date_wise['Deaths']/date_wise['Confirmed'])*100).mean(), color='blue', linestyle="--", label='Mean Mortality rate worldwide')

plt.legend()
# Countries belonging to different clusters



print("Countries belong to Cluster (0): ", list(country_wise[country_wise['Clusters']==0].index))

print("Countries belong to Cluster (1): ", list(country_wise[country_wise['Clusters']==1].index))

print("Countries belong to Cluster (2): ", list(country_wise[country_wise['Clusters']==2].index))
data_india= df1[df1['Country/Region']== 'India']

date_wise_india= data_india.groupby(['ObservationDate']).agg({"Confirmed": 'sum', "Recovered":'sum', "Deaths": 'sum'})



print(date_wise_india.iloc[-1])



print("Total Active Cases: ", date_wise_india['Confirmed'].iloc[-1]-date_wise_india['Recovered'].iloc[-1]- date_wise_india['Deaths'].iloc[-1])

print("Total Closed Cases: ", date_wise_india['Recovered'].iloc[-1]+ date_wise_india['Deaths'].iloc[-1])
# Growth of different types of cases in India



fig= go.Figure()



fig.add_trace(go.Scatter(x= date_wise_india.index, y=date_wise_india['Confirmed'], mode='lines+markers', name='Confirmed Cases'))

fig.add_trace(go.Scatter(x= date_wise_india.index, y=date_wise_india['Recovered'], mode='lines+markers', name='Recovered Cases'))

fig.add_trace(go.Scatter(x= date_wise_india.index, y=date_wise_india['Deaths'], mode='lines+markers', name='Death Cases'))



fig.update_layout(title='Growth of different types of cases in India', xaxis_title='Date', yaxis_title='Number of Cases', legend=dict(x=0, y=1, traceorder='normal'))

fig.show()
# Distribution of Active Cases in India



fig= px.bar(x= date_wise_india.index, y= date_wise_india['Confirmed']- date_wise_india['Recovered']- date_wise_india['Deaths'])

fig.update_layout(title='Distribution of Active Cases in India', xaxis_title='Date', yaxis_title='Number of Cases')

fig.show()
# Datewise Growth factors of different cases in India



ind_confirm_inc=[]

ind_recover_inc=[]

ind_death_inc=[]



for i in range(date_wise_india.shape[0]-1):

    ind_confirm_inc.append(((date_wise_india['Confirmed'].iloc[i+1])/date_wise_india['Confirmed'].iloc[i]))

    ind_recover_inc.append(((date_wise_india['Recovered'].iloc[i+1])/date_wise_india['Recovered'].iloc[i]))

    ind_death_inc.append(((date_wise_india['Deaths'].iloc[i+1])/date_wise_india['Deaths'].iloc[i]))

    

    

ind_confirm_inc.insert(0,1)

ind_recover_inc.insert(0,1)

ind_death_inc.insert(0,1)



fig= go.Figure()



fig.add_trace(go.Scatter(x=date_wise_india.index, y= ind_confirm_inc, mode='lines', name='Growth of Confirmed Cases'))

fig.add_trace(go.Scatter(x=date_wise_india.index, y= ind_recover_inc, mode='lines', name='Growth of Recovered Cases'))

fig.add_trace(go.Scatter(x=date_wise_india.index, y= ind_death_inc, mode='lines', name='Growth of Death Cases'))





fig.update_layout(title='Datewise Growth Factor of different types cases in India', xaxis_title='Date', yaxis_title='Growth Factor', legend= dict(x=0, y=1, traceorder='normal'))

fig.show()
# Daily increase in Different types of cases in India



fig= go.Figure()



fig.add_trace(go.Scatter(x= date_wise_india.index, y= date_wise_india['Confirmed'].diff().fillna(0), mode= 'lines+markers', name='Confirmed Cases'))

fig.add_trace(go.Scatter(x= date_wise_india.index, y= date_wise_india['Recovered'].diff().fillna(0), mode= 'lines+markers', name='Recovered Cases'))

fig.add_trace(go.Scatter(x= date_wise_india.index, y= date_wise_india['Deaths'].diff().fillna(0), mode= 'lines+markers', name='Death Cases'))



fig.update_layout(title='Daily Variation of Different types of cases in India', xaxis_title='Dates', yaxis_title='Number of Cases', legend= dict(x=0, y=1, traceorder='normal'))

fig.show()
# India's week wisie variatioin in confirmed and death cases



date_wise_india['WeekOfYear']= date_wise_india.index.weekofyear





week_num_india=[]



ind_confirm_weekwise=[]

ind_recover_weekwise=[]

ind_death_weekwise=[]

w=1



for i in list(date_wise_india['WeekOfYear'].unique()):

    ind_confirm_weekwise.append(date_wise_india[date_wise_india['WeekOfYear']==i]['Confirmed'].iloc[-1])

    ind_recover_weekwise.append(date_wise_india[date_wise_india['WeekOfYear']==i]['Recovered'].iloc[-1])

    ind_death_weekwise.append(date_wise_india[date_wise_india['WeekOfYear']==i]['Deaths'].iloc[-1])

    week_num_india.append(w)

    w+=1

    



fig, (ax1, ax2)= plt.subplots(1,2, figsize=(15,5))



sns.barplot(x=week_num_india, y= pd.Series(ind_confirm_weekwise).diff().fillna(0), ax= ax1)

sns.barplot(x=week_num_india, y= pd.Series(ind_death_weekwise).diff().fillna(0), ax= ax2)



ax1.set_xlabel("Week Number")

ax2.set_xlabel("Week Number")



ax1.set_ylabel("Number of Confirmed Cases")

ax2.set_ylabel("Number of Death Casses")



ax1.set_title("India's week wise variatioin in Confirmed cases")

ax2.set_title("India's week wise variatioin in Death cases")
# Data Preprocessing



date_wise['Days Since']= date_wise.index- date_wise.index[0]

date_wise['Days Since']= date_wise['Days Since'].dt.days



model_scores=[]



train_set= date_wise.iloc[:int(date_wise.shape[0]*0.95)]

test_set= date_wise.iloc[int(date_wise.shape[0]*0.95):]



# Standardisation of training dataset



X_train= np.array(train_set['Days Since']).reshape(-1,1)

y_train= np.array(train_set['Confirmed']).reshape(-1,1)

X_test= np.array(test_set['Days Since']).reshape(-1,1)

y_test= np.array(test_set['Confirmed'])





# Instantiating Linear Regression model



lin_reg= LinearRegression(normalize=True)



# fitting the model to the training set



lin_reg.fit(X_train, y_train)



# Predicting the model with test dataset



y_pred= lin_reg.predict(X_test)



# Model accuracy metrics



model_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))



print("RMSE for Linear Regression: ",np.sqrt(mean_squared_error(y_test, y_pred)))
# Prediction plot for Confirmed Cases by Linear Regression Model



lin_reg_output=[]



y_pred_date_wise= lin_reg.predict(np.array(date_wise['Days Since']).reshape(-1,1))

for i in range(y_pred_date_wise.shape[0]):

    lin_reg_output.append(y_pred_date_wise[i][0])

    

plt.figure(figsize=(10,5))



fig= go.Figure()

fig.add_trace(go.Scatter(x= date_wise.index, y=date_wise['Confirmed'], mode='lines', name='Actual Confirmed Cases'))

fig.add_trace(go.Scatter(x= date_wise.index, y=lin_reg_output, mode='lines+markers', name='Predicted Confirmed Cases'))



fig.update_layout(title='Prediction plot for Confirmed Cases by Linear Regression Model', xaxis_title='Date', yaxis_title='No. of Confirmed Cases', legend= dict(x=0, y=1, traceorder='normal'))



fig.show()
# Adding polynomial features to the model



poly= PolynomialFeatures(degree=7)



X_train_poly= poly.fit_transform(X_train)

y_train_poly= train_set['Confirmed']

X_test_poly= poly.fit_transform(X_test)

y_test_poly= test_set['Confirmed']



# Instantiating the Polynomial Regression Model



poly_lin_reg= LinearRegression(normalize= True)



# Fitting the model to the training data



poly_lin_reg.fit(X_train_poly, y_train_poly)



# Predicting the model on the test data



y_pred_poly= poly_lin_reg.predict(X_test_poly)



# Model Accuracy through RSME



rmse_poly= np.sqrt(mean_squared_error(y_test_poly, y_pred_poly))



model_scores.append(rmse_poly)



print("RMSE for Polynomial Regression: ", rmse_poly)

# Prediction Plot for by Polynomial Regression Model



X_test_poly_datewise= poly.fit_transform(np.array(date_wise['Days Since']).reshape(-1,1))

y_pred_poly_datewise= poly_lin_reg.predict(X_test_poly_datewise)



lin_reg_output=[]



y_pred_date_wise= lin_reg.predict(np.array(date_wise['Days Since']).reshape(-1,1))

for i in range(y_pred_date_wise.shape[0]):

    lin_reg_output.append(y_pred_date_wise[i][0])

plt.figure(figsize=(10,5))

fig= go.Figure()



fig.add_trace(go.Scatter(x= date_wise.index, y= date_wise['Confirmed'], mode='lines', name="Actual Confirmed Cases"))

fig.add_trace(go.Scatter(x= date_wise.index, y= y_pred_poly_datewise, mode='lines+markers', name="Predicted Confirmed Cases"))



fig.update_layout(title='Prediction Plot for Confirmed Cases by Polynomial Regression Model', xaxis_title='Date', yaxis_title='No. of Confirmed Cases', legend= dict(x=0, y=1, traceorder='normal'))

fig.show()
# Instantiating the model



svm= SVR(C=1, degree=5, kernel='poly', epsilon=0.01)



# Fitting the model to the training data



svm.fit(X_train, y_train)



# Predicting the model with the test data



y_pred_svm= svm.predict(X_test)



# Model Accuracy through RSME metric



rmse_svm= np.sqrt(mean_squared_error(y_test, y_pred_svm))



model_scores.append(rmse_svm)



print("RSME for Support Vector Machine Model: ",rmse_svm)
# Prediction Plot for Confirmed Cases by Support Vector Machine Model



y_pred_svm_datewise= svm.predict(np.array(date_wise['Days Since']).reshape(-1,1))



plt.figure(figsize=(10,5))

fig= go.Figure()



fig.add_trace(go.Scatter(x= date_wise.index, y=date_wise['Confirmed'], mode= 'lines', name='Actual Confirmed Cases'))

fig.add_trace(go.Scatter(x= date_wise.index, y=y_pred_svm_datewise, mode='lines+markers', name='Predicted Confirmed Cases'))

fig.update_layout(title='Prediction Plot for Confirmed Cases by Support Vector Machine Model', xaxis_title='Date',yaxis_title='No. of Confirmed Cases', legend= dict(x=0, y=1, traceorder='normal'))

fig.show()
# Instantiating the model



prophet_c= Prophet(interval_width=0.95, weekly_seasonality=True)

prophet_confirmed= pd.DataFrame(zip(list(date_wise.index), list(date_wise['Confirmed'])), columns=['ds','y'])



# Fitting the model to the data



prophet_c.fit(prophet_confirmed)





forecast_c= prophet_c.make_future_dataframe(periods=17)

forecast_confirmed= forecast_c.copy()



# Predicting with the model



confirmed_forecast= prophet_c.predict(forecast_c)



# Model accuracy check through RMSE



rmse_prophet= np.sqrt(mean_squared_error(date_wise['Confirmed'], confirmed_forecast['yhat'].head(date_wise.shape[0])))



print("RMSE for Facebook Prophets Model: ", rmse_prophet)



model_scores.append(rmse_prophet)
# Plot for forecast of Confirmed cases by Facebook Prophet Model.



print(prophet_c.plot(confirmed_forecast))
# Plot of components of Prophets model



print(prophet_c.plot_components(confirmed_forecast))
new_date=[]



new_pred_lr=[]

new_pred_poly=[]

new_pred_svm=[]



for i in range(1,25):

    new_date.append(date_wise.index[-1]+ timedelta(days=i))

    new_pred_lr.append(lin_reg.predict(np.array(date_wise['Days Since'].max()+i).reshape(-1,1))[0][0])

    new_pred_svm.append(svm.predict(np.array(date_wise['Days Since'].max()+i).reshape(-1,1))[0])
new_pred_prophet= list(confirmed_forecast['yhat'].tail(25))
pd.set_option('display.float_format', lambda x: '%.2f' %x)



model_preds= pd.DataFrame(zip(new_date, new_pred_lr, new_pred_svm, new_pred_prophet), columns=['Dates', 'Linear Reg. Pred.', 'SVM Pred.', 'Facebook Prophet Pred.'])

model_preds
mod_list=['Linear Regression','Polynomial Linear Regression', 'Support Vector Machine Regressor', 'Facebook Prophet Model']

mod_summary= pd.DataFrame(zip(mod_list, model_scores), columns=["Model Name", "RMSE"]).sort_values(["RMSE"])

mod_summary
# Reading the dataset



df2= pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])

df2.head(5)
# Data Preprocessing and Cleaning



cases= ['Confirmed', 'Recovered', 'Active', 'Deaths']



df2['Active']= df2['Confirmed']- df2['Recovered']- df2['Deaths']



# Filling missing values



df2[['Province/State']]= df2[['Province/State']].fillna('')

df2[cases]= df2[cases].fillna(0)



# Latest trends



df2_latest= df2[df2['Date']== max(df2['Date'])].reset_index()

df2_grouped= df2_latest.groupby('Country/Region')['Confirmed', 'Recovered', 'Active', 'Deaths'].sum().reset_index()
df2_temp= df2_grouped.sort_values(by=['Confirmed', 'Recovered', 'Deaths', 'Active'], ascending=False)

df2_temp= df2_temp.reset_index(drop= True)

df2_temp.style.background_gradient(cmap='Reds')
# Time Series plot to observe the spread



fig= go.Figure()



fig.add_trace(go.Scatter(x=df2.Date,

                         y=df2['Confirmed'],

                         name='Confirmed',

                         line_color='orange',

                         opacity=0.8))



fig.add_trace(go.Scatter(x=df2.Date,

                         y=df2['Recovered'],

                         name='Recovered',

                         line_color='blue',

                         opacity=0.8))



fig.add_trace(go.Scatter(x=df2.Date,

                         y=df2['Deaths'],

                         name='Deaths',

                         line_color='green',

                         opacity=0.8))



fig.add_trace(go.Scatter(x=df2.Date,

                         y=df2['Active'],

                         name='Active',

                         line_color='pink',

                         opacity=0.8))



fig.update_layout(title= 'Time Series for Confirmed, Recovered, Death and Active Cases', xaxis_rangeslider_visible= True)

fig.show()
# Progression of spread across the globe



df2_formatted= df2.groupby(['Date', 'Country/Region'])['Confirmed', 'Recovered', 'Deaths'].max().reset_index()

df2_formatted['Date']= pd.to_datetime(df2_formatted['Date']).dt.strftime('%m-%d-%Y')

df2_formatted['size']= df2_formatted['Confirmed'].pow(0.3)



fig= px.scatter_geo(df2_formatted, locations='Country/Region',

                    locationmode='country names',

                    color='Confirmed', size='size',

                    hover_name='Country/Region',

                    projection='natural earth',

                    range_color= [0, max(df2_formatted['Confirmed'])+2],

                    animation_frame='Date',

                    title='Progression of spread of COVID-19')



fig.update(layout_coloraxis_showscale= False)

fig.show()
# defining small functions to convert the datatype of some of the features



def p2f(x):

    

    try:

        return float(x.strip('%'))/100

    except:

        return np.nan



def age2int(x):

    

    try:

        return int(x)

    except:

        return np.nan



def fert2float(x):

   

    try:

        return float(x)

    except:

        return np.nan



# Reading the dataset for features like population, density, fertility rate, median age, urban population for modeling. 



df_country = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv", converters={'Urban Pop %':p2f,

                                                                                                             'Fert. Rate':fert2float,

                                                                                                             'Med. Age':age2int})

df_country.rename(columns={'Country (or dependency)': 'Country/Region',

                             'Population (2020)' : 'Population',

                             'Density (P/Km²)' : 'Density',

                             'Fert. Rate' : 'Fertility',

                             'Med. Age' : "Age",

                             'Urban Pop %' : 'Urban Percentage'}, inplace=True)







df_country['Country/Region'] = df_country['Country/Region'].replace('United States', 'US')

df_country = df_country[["Country/Region", "Population", "Density", "Fertility", "Age", "Urban Percentage"]]



df_country.head()

# Merging the above sub dataframe to our main 2nd dataframe



df2= pd.merge(df2, df_country, on='Country/Region')
# Reading the ICU-beds-per-Country data 



df_icu= pd.read_csv('../input/icu-beds-per-1000-individuals/API_SH.MED.BEDS.ZS_DS2_en_csv_v2_887506.csv')

df_icu.head(5)
# Checking for null values



df_icu.isnull().sum()
# Data Preprocessing and Cleaning



df_icu['Country Name']= df_icu['Country Name'].replace({

                                                        'United States': 'US',

                                                        'Russian Federation': 'Russia',

                                                        'Iran, Islamic Rep.': 'Iran',

                                                        'Egypt, Arab Rep.': 'Egypt',

                                                        'Venezuela, RB': 'Venezuela',

                                                        'Czechia': 'Czech Republic'})
df_icu_temp= pd.DataFrame()

df_icu_temp['Country/Region']= df_icu['Country Name']

df_icu_temp['ICU']= np.nan



for year in range(1960, 2020):

    df_year= df_icu[str(year)].dropna()

    df_icu_temp['ICU'].loc[df_year.index]= df_year.values

    

df_icu_temp.head(5)
# Merging the ICU beds dataset to main 2nd dataframe.



df2= pd.merge(df2, df_icu_temp, on='Country/Region')

df2.head(2)
df2['Province/State']= df2['Province/State'].fillna('')



temp= df2[[col for col in df2.columns if col!='Province/State']]



df_temp= temp[temp['Date']== max(temp['Date'])].reset_index()

df_temp_grp= df_temp.groupby('Country/Region')['ICU'].mean().reset_index()



fig= px.choropleth(df_temp_grp, locations="Country/Region",

                  locationmode='country names', color='ICU',

                  hover_name='Country/Region', range_color=[1,15],

                  color_continuous_scale='algae',

                  title='Ratio of ICU beds per 1000 peoples')

fig.show()
# Reading temperature dataset containing some other features like humidity, sunHour and wind speed.



df_temperature= pd.read_csv('../input/temperature-dataframe/temperature_dataframe.csv')

df_temperature.head(2)
# Data preprocessing and Cleaning



df_temperature['country']= df_temperature['country'].replace({'USA': 'US', 'UK': 'United Kingdom'})

df_temperature= df_temperature[['country','province','date', 'humidity','sunHour', 'tempC', 'windspeedKmph']].reset_index()

df_temperature.rename(columns={'country': 'Country/Region',

                               'province': 'Province/State',

                               'date': 'Date',

                               'humidity': 'Humidity',

                               'tempC': 'Temp. in Celcious',

                               'windspeedKmph': 'Wind Speed in KMPH'}, inplace= True)

df_temperature['Date'] =pd.to_datetime(df_temperature['Date'])

df_temperature['Province/State']= df_temperature['Province/State'].fillna('')

df_temperature.drop(['index'], axis=1, inplace= True)

df_temperature.head()
# Merging teh rest of the features to the main 2nd dataframe



df2= df2.merge(df_temperature, on=['Country/Region', 'Date', 'Province/State'], how='inner')

df2.head(2)
# Check if the data is updated

print("Dataset Description")

print("Earliest Entry: ",df2['Date'].min())

print("Last Entry:     ",df2['Date'].max())

print("Total Days:     ",(df2['Date'].max() - df2['Date'].min()))
# Preparing the training set



train= df2

train.info()
# Checking for null values



train.isnull().sum()
# Dealing with null values



train.fillna(0, inplace= True)

train.info()
# Filtering the training dataset above a threshold barrier of 0 i.e only using the values which are having Infected rate >0



barrier=0

train['Infection Rate']= round(train['Confirmed']/train['Population']*100,6)

train= train[train['Infection Rate']>= barrier]

train.info()
# Further data- preprocessing



train= train.drop(['Country/Region', 'Province/State', 'Date', 'Lat', 'Long', 'Active', 'Recovered', 'Infection Rate', 'WHO Region', 'Fertility'], axis=1).dropna()

train.info()
# Preparing data for train test split



y=train[['Confirmed','Deaths']]

X=train.drop(['Confirmed','Deaths'], axis= 1)
y.shape
X.shape
# Checking the correlation between all the features



plt.figure(figsize=(20,10))

sns.heatmap(train.corr(), annot= True)
# Defining a function to calculate Root Mean Square Log Error



def rmsle(y_test, y_pred):

    

    return np.sqrt(mean_squared_log_error(y_test, y_pred))



# Making the rsmle as scoring metric



rmsle_scorer= make_scorer(rmsle)



# Performing train-test split



X_train,X_test,y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42, shuffle= True)



# Instantiating the Decision Tree Regressor Model



dt_reg= DecisionTreeRegressor(random_state=42, criterion="mae")



# Performing the cross validation on trained data for confirmed cases

scores_for_confirmed_cases= cross_val_score(dt_reg, X_train, y_train['Confirmed'], cv=5, scoring=rmsle_scorer)



# fitting the model to training data



dt_reg.fit(X_train, y_train['Confirmed'])



# Checking model accuracy by RMSLE as metric

rmsle_dt_for_confirmed_cases= rmsle(y_test['Confirmed'], dt_reg.predict(X_test))



print("Validation of Confirmed Cases by RMSLE: ", rmsle_dt_for_confirmed_cases)



# Performing the cross validation on trained data for death cases

scores_for_death_cases= cross_val_score(dt_reg, X_train, y_train['Deaths'], cv=5, scoring=rmsle_scorer)



# fitting the model to training data



dt_reg.fit(X_train, y_train['Deaths'])



# Checking model accuracy by RMSLE as metric



rmsle_dt_for_death_cases= rmsle(y_test['Deaths'], dt_reg.predict(X_test))

print("Validation of Death Cases by RMSLE: ", rmsle_dt_for_death_cases)
dt_reg_confirmed= dt_reg.fit(X, y['Confirmed'])

dt_reg_deaths= dt_reg.fit(X, y['Deaths'])



# Defining a function to get the feature importance of various features 



def feature_importance(dt):

    importance= dt.feature_importances_

    indices= np.argsort(importance)[::-1]

    

    plt.figure(figsize=(20,10))

    plt.bar(range(X.shape[1]), importance[indices])

    plt.xticks(range(X.shape[1]), X.columns[indices], rotation='vertical')

    plt.show()
# Feature importantce for Confirmed Cases



feature_importance(dt_reg_confirmed)
# Feature importance for Death cases



feature_importance(dt_reg_deaths)
# Dropping 'Wind Speed in KMPH' column from df2 dataframe.



train= train.drop(['Wind Speed in KMPH'], axis=1).dropna()

train.info()
y=train[['Confirmed','Deaths']]

X=train.drop(['Confirmed','Deaths'], axis= 1)
y.shape
X.shape
# Checking the correlation between all the features



plt.figure(figsize=(20,10))

sns.heatmap(train.corr(), annot= True)
# Defining a function to calculate Root Mean Square Log Error



def rmsle(y_test, y_pred):

    

    return np.sqrt(mean_squared_log_error(y_test, y_pred))



# Making the rsmle as scoring metric



rmsle_scorer= make_scorer(rmsle)



# Performing train-test split



X_train,X_test,y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42, shuffle= True)



# Instantiating the Decision Tree Regressor Model



dt_reg= DecisionTreeRegressor(random_state=42, criterion="mae")



# Performing the cross validation on trained data for confirmed cases

scores_for_confirmed_cases= cross_val_score(dt_reg, X_train, y_train['Confirmed'], cv=5, scoring=rmsle_scorer)



# fitting the model to training data



dt_reg.fit(X_train, y_train['Confirmed'])



# Checking model accuracy by RMSLE as metric

rmsle_dt_for_confirmed_cases= rmsle(y_test['Confirmed'], dt_reg.predict(X_test))



print("Validation of Confirmed Cases by RMSLE: ", rmsle_dt_for_confirmed_cases)



# Performing the cross validation on trained data for death cases

scores_for_death_cases= cross_val_score(dt_reg, X_train, y_train['Deaths'], cv=5, scoring=rmsle_scorer)



# fitting the model to training data



dt_reg.fit(X_train, y_train['Deaths'])



# Checking model accuracy by RMSLE as metric



rmsle_dt_for_death_cases= rmsle(y_test['Deaths'], dt_reg.predict(X_test))

print("Validation of Death Cases by RMSLE: ", rmsle_dt_for_death_cases)
dt_reg_confirmed= dt_reg.fit(X, y['Confirmed'])

dt_reg_deaths= dt_reg.fit(X, y['Deaths'])



# Defining a function to get the feature importance of various features 



def feature_importance(dt):

    importance= dt.feature_importances_

    indices= np.argsort(importance)[::-1]

    

    plt.figure(figsize=(20,10))

    plt.bar(range(X.shape[1]), importance[indices])

    plt.xticks(range(X.shape[1]), X.columns[indices], rotation='vertical')

    plt.show()
# Feature importantce for Confirmed Cases



feature_importance(dt_reg_confirmed)
# Feature importantce for Confirmed Cases



feature_importance(dt_reg_deaths)