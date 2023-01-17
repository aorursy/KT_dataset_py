# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_rows', None)



import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go



from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score,silhouette_samples

from sklearn.metrics import mean_squared_error,r2_score



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data.drop("SNo", axis=1, inplace= True)

dtt = data.iloc[84:84+41,:]

dtt["Confirmed"].sum()

print(data.isna().sum())

print(data.shape)
data["ObservationDate"] = pd.to_datetime(data["ObservationDate"]) #.dt.strftime('%d %B, %Y')

data["Last Update"] = pd.to_datetime(data["Last Update"])

data.head(5)
#Grouping different types of cases as per the date

datewise=data.groupby(["ObservationDate"]).agg({"Confirmed":'sum',"Recovered":'sum',"Deaths":'sum'})

datewise["Days Since"]=datewise.index-datewise.index.min()



datewise.head()
date_seprate = data.groupby(["ObservationDate"])[["Confirmed","Deaths","Recovered"]].count()

date_seprate.head()
print("Basic Information")

print("Totol number of countries with Disease Spread: ",len(data["Country/Region"].unique()))

print("Total number of Confirmed Cases around the World: ",datewise["Confirmed"].iloc[-1])

print("Total number of Recovered Cases around the World: ",datewise["Recovered"].iloc[-1])

print("Total number of Deaths Cases around the World: ",datewise["Deaths"].iloc[-1])

print("Total number of Active Cases around the World: ",(datewise["Confirmed"].iloc[-1]-datewise["Recovered"].iloc[-1]-datewise["Deaths"].iloc[-1]))

print("Total number of Closed Cases around the World: ",datewise["Recovered"].iloc[-1]+datewise["Deaths"].iloc[-1])

print("Approximate number of Confirmed Cases per Day around the World: ",np.round(datewise["Confirmed"].iloc[-1]/datewise.shape[0]))

print("Approximate number of Recovered Cases per Day around the World: ",np.round(datewise["Recovered"].iloc[-1]/datewise.shape[0]))

print("Approximate number of Death Cases per Day around the World: ",np.round(datewise["Deaths"].iloc[-1]/datewise.shape[0]))
# fig = plt.figure(figsize=(10,10))



# ax = fig.gca()



fig = px.bar(x=datewise.index, y = datewise["Confirmed"] )

fig.update_layout(title="Distribution of Number of Active Cases",

                  xaxis_title="Date",yaxis_title="Number of Cases",)

fig.show()
fig = plt.figure(figsize=(20,8))

x= datewise.index



ax = fig.gca()

ax.grid()

# ax.set_xticklabels(datewise.index,rotation= 45, fontsize= 12)

sns.lineplot(x=datewise.index, y= datewise["Deaths"], ax= ax )
datewise["WeekOfYear"]=datewise.index.weekofyear



week_num = []

datewise_confirmed = []

datewise_death = []

datewise_recovered = []



weekOfYear = list(datewise["WeekOfYear"].unique())

w = 1

for i in weekOfYear:

        confirmed = datewise.loc[(datewise["WeekOfYear"] == i), "Confirmed"][-1]

        death = datewise.loc[(datewise["WeekOfYear"] == i), "Deaths"][-1]

        recovered = datewise.loc[(datewise["WeekOfYear"] == i), "Recovered"][-1]

        datewise_confirmed.append(confirmed)

        datewise_death.append(death)

        datewise_recovered.append(recovered)

        week_num.append(w)

        w+=1
fig=go.Figure()



fig.add_trace(go.Scatter(x= week_num, y = datewise_confirmed, name= "Weekly New Cases"))

fig.add_trace(go.Scatter(x= week_num, y = datewise_death, name= "New Deaths Cases"))

fig.add_trace(go.Scatter(x= week_num, y = datewise_recovered, name= "Weekly Recovered Cases"))



fig.show()
datewise
datewise["Death Rate"] = np.round((datewise["Deaths"]/ datewise["Confirmed"]) *100,2)

datewise["Recovery Rate"] = np.round((datewise["Recovered"]/ datewise["Confirmed"]) *100,2)

datewise["Active Cases"] = np.round((datewise["Confirmed"])- (datewise["Deaths"]+ datewise["Recovered"]),2)



datewise
fig = go.Figure()



fig.add_trace(go.Scatter(x= datewise.index, y = datewise["Death Rate"],mode='lines+markers', name= "Daily Death Rate"))

fig.show()

fig = go.Figure()

fig.add_trace(go.Scatter(x= datewise.index, y = datewise["Recovery Rate"],mode='lines+markers', name= "Daily Recovery Rate"))

fig.show()



fig = go.Figure()

fig.add_trace(go.Scatter(x= datewise.index, y= datewise["Active Cases"],mode='lines+markers', name= "Daily Active Cases"))

fig.show()
print("Average Mortality rate is %1.2f" %datewise["Death Rate"].mean())

print("Average Recovery rate is %1.2f" %datewise["Recovery Rate"].mean())
fig = go.Figure()



fig.add_trace(go.Scatter(x= datewise.index, y = datewise["Confirmed"].diff().fillna(0), name= "Daily New Cases",mode='lines+markers' ))

fig.add_trace(go.Scatter(x= datewise.index, y = datewise["Deaths"].diff().fillna(0), name= "Daily Death Cases",mode='lines+markers' ))

fig.add_trace(go.Scatter(x= datewise.index, y = datewise["Recovered"].diff().fillna(0), name= "Daily Recovered Cases",mode='lines+markers' ))

fig.update_layout(title= "Daily New Cases", xaxis_title="Date", yaxis_title="Total Numbers",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
countrywise = data.groupby(["Country/Region", "ObservationDate"], as_index= False).sum()

countrywise["Death Rate"] = np.round((countrywise["Deaths"]/ countrywise["Confirmed"])*100,2)

countrywise["Recovery Rate"] = np.round((countrywise["Recovered"]/ countrywise["Confirmed"])*100,2)

countrywise.head()
def get_countrywise(country_name):

    countrywise_country = countrywise[countrywise["Country/Region"] == country_name]

    countrywise_country["Active Cases"] = countrywise_country["Confirmed"]- countrywise_country["Deaths"]-countrywise_country["Recovered"]

    countrywise_country["Daily Death"] = np.abs(countrywise_country["Deaths"].diff().fillna(0))

    countrywise_country["Daily Recovered"] = np.abs(countrywise_country["Recovered"].diff().fillna(0))

    countrywise_country["Daily New Cases"] = np.abs(countrywise_country["Confirmed"].diff().fillna(0))

    return countrywise_country

countrywise_egypt = get_countrywise("Egypt")

countrywise_egypt
def draw_countrywise_stat(countrywise_country):

    selected_country_name = countrywise_country["Country/Region"].iloc[1]



    fig = go.Figure()



    fig.add_trace(go.Scatter(x= countrywise_country["ObservationDate"],

                             y= countrywise_country["Daily New Cases"]

                            ,name= "Daily New Cases", mode = "lines+markers"))



    fig.add_trace(go.Scatter(x= countrywise_country["ObservationDate"],

                             y= countrywise_country["Daily Recovered"]

                            ,name= "Daily Recovered", mode = "lines+markers"))



    fig.add_trace(go.Scatter(x= countrywise_country["ObservationDate"],

                             y= countrywise_country["Daily Death"]

                            ,name= "Daily Death", mode = "lines+markers"))



    fig.update_layout(title=selected_country_name+" Daily statstics", xaxis_title= "Date",

                      yaxis_title="Total Number", legend = dict(x=0,y=1))

    fig.show()

draw_countrywise_stat(countrywise_egypt)
countrywise_UAE = get_countrywise("United Arab Emirates")

countrywise_UAE
draw_countrywise_stat(countrywise_UAE)
draw_countrywise_stat(get_countrywise("Azerbaijan"))
def get_activeCases(countrywise_list):

    fig = go.Figure()

    for country in countrywise_list:

        country_name = country["Country/Region"].iloc[1] 

        fig.add_trace(go.Scatter(x= country["ObservationDate"]

                                ,y=country["Active Cases"],

                                mode = "lines+markers", name= country_name+" Active Cases"))



    fig.show()



get_activeCases([countrywise_egypt, countrywise_UAE, get_countrywise("Israel")

                ,get_countrywise("Azerbaijan")])
countrywise_rates = countrywise[countrywise["Confirmed"]> 500]

countrywise_confiremd_cases = countrywise_rates[["Country/Region","Confirmed"]]

countrywise_confiremd_cases = countrywise_confiremd_cases.groupby("Country/Region").max()

countrywise_rates = countrywise_rates.groupby("Country/Region",

                                        as_index=False)["Death Rate","Recovery Rate"].mean()

countrywise_rates.sort_values(by ="Death Rate",ascending= True)



countrywise_confiremd_cases
print("WorldWide Recover Rate Average is %1.2f" %countrywise_rates["Recovery Rate"].mean()+"%")

print("WorldWide Death Rate Average is %1.2f" %countrywise_rates["Death Rate"].mean()+"%")

print("WorldWide Death Rate Average is %1.2f" %countrywise_confiremd_cases["Confirmed"].mean()+"%")
countrywise_rates
countrywise_rates = countrywise_confiremd_cases.merge(countrywise_rates.set_index("Country/Region"),

                                                      left_index=True, right_index=True)

countrywise_rates.reset_index(inplace= True)
std = StandardScaler()

X = countrywise_rates.drop("Country/Region", axis=1)

X = std.fit_transform(X)

k = np.arange(2,10,1)

inertia = []

for i in k:

    clf = KMeans(n_clusters=i,init='k-means++',random_state=42)

    clf.fit(X)

    inertia.append(clf.inertia_)
fig = go.Figure()

fig.add_trace(go.Scatter(x=k, y=inertia,  mode="lines+markers"))

fig.show()

clf = KMeans(n_clusters=4,init='k-means++',random_state=42)

clf.fit(X)

countrywise_rates["Cluster"] = clf.predict(X)

countrywise_rates
plt.rc('xtick', labelsize=14) 

plt.rc('ytick', labelsize=14)

plt.rc('font', size=14)



fig, ax = plt.subplots(figsize=(10,10))



sns.scatterplot(x="Recovery Rate", y="Death Rate",data= countrywise_rates,

                hue= "Cluster", ax=ax,s = 100, palette="Set2", style="Cluster")



fig.show()
countrywise_rates
def get_cluster_info(cluster):

    countrywise_cluster = countrywise_rates[countrywise_rates["Cluster"] == cluster]

    confirmed_avrg = countrywise_cluster["Confirmed"].mean()

    print("*"*80)

    print("Cluster number %1d Confirmed Cases Average is %1.2f" %(cluster, confirmed_avrg))

    death_avrg = countrywise_cluster["Death Rate"].mean()

    print("Cluster number %1d Death Cases Average is %1.2f" %(cluster, death_avrg))

    rec_avrg = countrywise_cluster["Recovery Rate"].mean()

    print("Cluster number %1d Recovery Cases Average is %1.2f" %(cluster, rec_avrg))

get_cluster_info(0)

get_cluster_info(1)

get_cluster_info(2)

get_cluster_info(3)
countrywise_rates[countrywise_rates["Cluster"]==1]