# import libraries

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt
#Reading Data

data=pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv' ,parse_dates=['Last Update'])

data.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)



#Earliest Cases

data.head()
data.shape
#Missing Values

data.isnull().sum().to_frame('nulls')
df = data.groupby(["Date", "Country"])[['Date', 'Country', 'Confirmed', 'Deaths', 'Recovered']].sum().reset_index()

sorted_By_Confirmed=df.sort_values('Confirmed',ascending=False)

sorted_By_Confirmed=sorted_By_Confirmed.drop_duplicates('Country')



world_Confirmed_Total=sorted_By_Confirmed['Confirmed'].sum()

world_Deaths_Total=sorted_By_Confirmed['Deaths'].sum()

world_Recovered_Total=sorted_By_Confirmed['Recovered'].sum()



Active=world_Confirmed_Total-world_Deaths_Total-world_Recovered_Total



world_Deaths_rate=(world_Deaths_Total*100)/world_Confirmed_Total

world_Recovered_rate=(world_Recovered_Total*100)/world_Confirmed_Total



China=sorted_By_Confirmed[sorted_By_Confirmed['Country']=='Mainland China']

China_Recovered_rate=(int(China['Recovered'].values)*100)/int(China['Confirmed'].values)





veri={'Total Confirmed cases  in the world':world_Confirmed_Total,'Total Deaths cases in the world':world_Deaths_Total,'Total Recovered cases in the world':world_Recovered_Total,'Total Active Cases':Active,'Rate of Recovered Cases %':world_Recovered_rate,'Rate of Deaths Cases %':world_Deaths_rate,'Rate of Recovered China cases %':China_Recovered_rate}

veri=pd.DataFrame.from_dict(veri, orient='index' ,columns=['Total'])

print("04/08/2020") 

veri.style.background_gradient(cmap='Greens')
Recovered_rate=(sorted_By_Confirmed['Recovered']*100)/sorted_By_Confirmed['Confirmed']

Deaths_rate=(sorted_By_Confirmed['Deaths']*100)/sorted_By_Confirmed['Confirmed']

cases_rate=(sorted_By_Confirmed.Confirmed*100)/world_Confirmed_Total



sorted_By_Confirmed['Active']=sorted_By_Confirmed['Confirmed']-sorted_By_Confirmed['Deaths']-sorted_By_Confirmed['Recovered']

sorted_By_Confirmed['Recovered Cases Rate %']=pd.DataFrame(Recovered_rate)

sorted_By_Confirmed['Deaths Cases Rate %']=pd.DataFrame(Deaths_rate)

sorted_By_Confirmed['Total Cases Rate %']=pd.DataFrame(cases_rate)





print("Sorted By Confirmed Cases")

sorted_By_Confirmed.style.background_gradient(cmap='Reds')

sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(15,15 ))



sns.barplot(x="Confirmed", y="Country", data=sorted_By_Confirmed.head(20),

            label="Confirmed", color="b")



# Plot the crashes where alcohol was involved

sns.set_color_codes("muted")

sns.barplot(x="Recovered", y="Country", data=sorted_By_Confirmed.head(20),

            label="Recovered", color="g")



sns.set_color_codes("muted")

sns.barplot(x="Deaths", y="Country", data=sorted_By_Confirmed.head(20),

            label="Deaths", color="r")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

df_Difference = data.groupby(["Date", "Country"])[['Date', 'Country', 'Confirmed','Deaths']].sum().reset_index()

sorted_By_Confirmed_Difference=df_Difference.sort_values('Country',ascending=False)



x1=sorted_By_Confirmed_Difference[sorted_By_Confirmed_Difference.Date=='04/08/2020'].reset_index().drop('index',axis=1)

x2=sorted_By_Confirmed_Difference[sorted_By_Confirmed_Difference.Date=='04/07/2020'].reset_index().drop('index',axis=1)



h=pd.merge(x2,x1,on='Country')

h['New Confirmed Cases']=h['Confirmed_y']-h['Confirmed_x']

h['New Deaths ']=h['Deaths_y']-h['Deaths_x']



h1=h.sort_values('New Confirmed Cases',ascending=False).head(50)

h1=h1.drop(['Confirmed_x','Deaths_x','Date_x','Confirmed_y','Deaths_y'],axis=1).style.background_gradient(cmap='Greens')

print("The New Cases in 08/04")

h1
sorted_By_Confirmed1=sorted_By_Confirmed.head(10)

x=sorted_By_Confirmed1.Country

y=sorted_By_Confirmed1.Confirmed

plt.rcParams['figure.figsize'] = (12, 10)

sns.barplot(x,y,order=x ,palette="rocket").set_title('Total Cases / Deaths / Recovered')  #graf çizdir (Most popular)
Top7=sorted_By_Confirmed.iloc[0:9,-1].values

others=sorted_By_Confirmed.iloc[9:,-1].sum()

x=np.array(Top7)

x2=np.array(others)

rates=np.concatenate((x, x2), axis=None)



rate_perCountry=pd.DataFrame(data=rates,index=[sorted_By_Confirmed['Country'].head(10)] ,columns=['rate'])

rate_perCountry.rename(index={'Belgium': "other Countries"},inplace=True)





labels=rate_perCountry.index

sizes=rate_perCountry['rate'].values



explode = None  # explode 1st slice

plt.subplots(figsize=(8,8))

plt.pie(sizes, explode=explode, labels=labels,autopct='%1.1f%%', shadow=False, startangle=0)

plt.axis('equal')

print("cases rate per country of total cases in the world ")

plt.show()
cases_per_Day = data.groupby(["Date"])['Confirmed','Deaths', 'Recovered'].sum().reset_index()

sorted_By_Confirmed1=cases_per_Day.sort_values('Date',ascending=False)



x=cases_per_Day.index



y=cases_per_Day.Confirmed

y1=cases_per_Day.Deaths

y2=cases_per_Day.Recovered



sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(12,10 ))



plt.scatter(x,y,color='blue' , label='Confirmed Cases')

plt.scatter(x,y1,color='red' ,label="Deaths Cases")

plt.scatter(x,y2,color='green',label="Recovered Cases")

plt.title("Increasing infections cases in the world per day .")

ax.legend(ncol=2, loc='upper left', frameon=True)

plt.show()
sorted_By_Confirmed1.style.background_gradient(cmap='Reds')
#Train & Test Data 

x_data=pd.DataFrame(cases_per_Day.index)

y_data=pd.DataFrame(cases_per_Day.Confirmed)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.1,random_state=0)
#Polynomal Regression (degree=5)

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures



poly_reg=PolynomialFeatures(degree=5)

x_poly=poly_reg.fit_transform(x_train)

lin_reg2=LinearRegression()

lin_reg2.fit(x_poly,y_train)
cases_per_Day = data.groupby(["Date"])['Confirmed','Deaths', 'Recovered'].sum().reset_index()

sorted_By_Confirmed1=cases_per_Day.sort_values('Date',ascending=False)



x=cases_per_Day.index



y=cases_per_Day.Confirmed



plt.scatter(x,y,color='red')

plt.plot(x_test,lin_reg2.predict(poly_reg.fit_transform(x_test)),color='blue')

plt.title("Polynomial Regression Model ")

plt.show()
y_pred=lin_reg2.predict(poly_reg.fit_transform(x_test))



result=pd.DataFrame(y_pred)

result['Real Value']=y_test.iloc[:,:].values

result['Predicted Value']=pd.DataFrame(y_pred)

result=result[['Real Value','Predicted Value']]

result


from sklearn.metrics import r2_score



print('Polynomial Regession  R2 Score   : ',r2_score(y_test, y_pred))
#today is 03/29/2020

print("After {0} day will be {1} case in the world".format((88-len(cases_per_Day)),lin_reg2.predict(poly_reg.fit_transform([[88]]))))

print("After {0} day will be {1} case in the world".format((98-len(cases_per_Day)),lin_reg2.predict(poly_reg.fit_transform([[98]]))))

print("After {0} day will be {1} case in the world".format((108-len(cases_per_Day)),lin_reg2.predict(poly_reg.fit_transform([[108]]))))