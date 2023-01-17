import datetime as dt
dt_string = dt.datetime.now().strftime("%d/%m/%Y")
print(f"Kernel last updated: {dt_string}")
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime as dt
import folium
from folium.plugins import HeatMap, HeatMapWithTime
%matplotlib inline
print(os.listdir('/kaggle/input'))
DATA_FOLDER = "/kaggle/input/coronavirus-2019ncov"
print(os.listdir(DATA_FOLDER))

data_df = pd.read_csv(os.path.join(DATA_FOLDER, "covid-19-all.csv"))
world_economy = pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQjU32w7PB4A3j2pWeH6TFoqJ1eOFe1apxHmJFz-P5KqrTh0SHwLZJmVNGoRP9sIQnAqP6nHyCAvDMs/pub?output=csv')
print(world_economy.shape)
world_eco=world_economy.iloc[:,0:3]
world_eco.head()
world_eco['date'] = pd.to_datetime(world_eco['date'])
print(f"Date - unique values: {world_eco['date'].nunique()} ({min(world_eco['date'])} - {max(world_eco['date'])})")
world_eco=world_eco[ (world_eco['date'] >= '2020-01-22 00:00:00')]
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))
missing_data(world_eco)
print(f"Countries:{world_eco['country'].nunique()}")

data_df.tail()
for column in data_df.columns:
    print(f"{column}:{data_df[column].dtype}")
print(f"Date - unique values: {data_df['Date'].nunique()} ({min(data_df['Date'])} - {max(data_df['Date'])})")
data_df['Date'] = pd.to_datetime(data_df['Date'])
for column in data_df.columns:
    print(f"{column}:{data_df[column].dtype}")
print(f"Date - unique values: {data_df['Date'].nunique()} ({min(data_df['Date'])} - {max(data_df['Date'])})")
missing_data(data_df)
print(f"Countries/Regions:{data_df['Country/Region'].nunique()}")
print(f"Province/State:{data_df['Province/State'].nunique()}")
covid_data=data_df
economy=world_eco
print(world_eco.shape)
print(covid_data.shape)
economy=economy.rename(columns={'country':'Country/Region','date':'Date'})
economy.head()
dataset=pd.merge(covid_data,economy,on=['Country/Region','Date'])
dataset
missing_data(dataset)
def plot_count(feature, value, title, df, size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    df = df.sort_values([value], ascending=False).reset_index(drop=True)
    g = sns.barplot(df[feature][0:30], df[value][0:30], palette='Set3')
    g.set_title("Number of {} - first 30 by number".format(title))
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.show()    
def plot_time_variation(df, y='Confirmed', hue='Province/State', size=1, is_log=False):
    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))
    
    g = sns.lineplot(x="Date", y=y, hue=hue, data=df)
    plt.xticks(rotation=90)
    plt.title(f'{y} cases grouped by {hue}')
    if(is_log):
        ax.set(yscale="log")
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()  
def plot_time_variation_all(df, title='Mainland China', size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))
    g = sns.lineplot(x="Date", y='Confirmed', data=df, color='blue', label='Confirmed')
    g = sns.lineplot(x="Date", y='Recovered', data=df, color='green', label='Recovered')
    g = sns.lineplot(x="Date", y='Deaths', data=df, color = 'red', label = 'Deaths')
    plt.xlabel('Date',fontsize=0.2)
    plt.ylabel(f'Total {title} cases')
    plt.xticks(rotation=90)
    plt.title(f'Total {title} cases')
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()  

def plot_time_variation_mortality(df, title='Mainland China', size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))
    g = sns.lineplot(x="Date", y='Mortality (D/C)', data=df, color='blue', label='Mortality (Deaths / Confirmed)')
    g = sns.lineplot(x="Date", y='Mortality (D/R)', data=df, color='green', label='Mortality (Death / Recovered)')
    plt.xlabel('Date')
    ax.set_yscale('log')
    plt.ylabel(f'Mortality {title} [%]')
    plt.xticks(rotation=90)
    plt.title(f'Mortality percent {title}\nCalculated as Deaths/Confirmed cases and as Death / Recovered cases')
    ax.grid(color='black', linestyle='dashed', linewidth=1)
    plt.show()  
data_ct = data_wd.sort_values(by = ['Country','Date'], ascending=False)
filtered_data_ct_last = data_wd.drop_duplicates(subset = ['Country'], keep='first')
data_ct_agg = data_ct.groupby(['Date']).sum().reset_index()
filtered_data_ct_last.head()
plot_count('Country', 'Confirmed', 'Confirmed cases - all World', filtered_data_ct_last, size=4)
data_ct.where(data_ct['Date']=='2020-06-05',data_ct).tail(2422)
plot_count('Country', 'Deaths', 'Deaths - all World', filtered_data_ct_last, size=4)
world_data_with_eco = dataset
world_data_with_eco .info()
world_data_with_eco = dataset

world_data_with_eco = pd.DataFrame(world_data_with_eco.groupby(['Country/Region', 'Date','currency rate'])['Confirmed', 'Recovered', 'Deaths','currency rate'].sum()).reset_index()

world_data_with_eco.columns = ['Country', 'Date', 'Confirmed', 'Recovered', 'Deaths','currency rate']
world_data_with_eco  = world_data_with_eco.sort_values(by = ['Country','Date'], ascending=False)

world_data_ct = world_data_with_eco.sort_values(by = ['Country','Date'], ascending=False)

world_filtered_data_ct_last = world_data_with_eco.drop_duplicates(subset = ['Country'], keep='first')

world_data_ct_agg = world_data_ct.groupby(['Date']).sum().reset_index()
#print(world_data_ct_agg)
world_filtered_data_ct_last
print(f"Countries:{world_filtered_data_ct_last['Country'].nunique()}")
missing_data(world_filtered_data_ct_last)
world_data_ct
def plot_time_variation_all(df, title='Mainland China', size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,2*size))
    g = sns.lineplot(x="Date", y='currency rate', data=df, color='blue', label='Currency rates')
    g = sns.lineplot(x="Date", y='Recovered', data=df, color='green', label='Recovered')
    g = sns.lineplot(x="Date", y='Deaths', data=df, color = 'red', label = 'Deaths')
    plt.xlabel('Date',fontsize=0.2)
    plt.ylabel(f'Total {title} cases')
    plt.xticks(rotation=90)
    plt.title(f'Total {title} cases')
    
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()  

def plot_time_variation_countries_world(df, countries, case_type='Confirmed', size=3, is_log=False):
    f, ax = plt.subplots(1,1, figsize=(4*size,4*size))
    for country in countries:
        df_ = df[(df['Country']==country) & (df['Date'] > '2020-02-01')] 
        g = sns.lineplot(x="Date", y=case_type, data=df_,  label=country)  
        #ax.text(max(df_['Date']), (df_.loc[df_['Date']==max(df_['Date']), case_type]), str(country))
    plt.xlabel('Date')
    plt.ylabel(f'Total  {case_type}')
    plt.title(f'Total {case_type}')
    plt.xticks(rotation=90)
    if(is_log):
        ax.set(yscale="log")
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()  
world_data_select_agg = world_data_ct.groupby(['Country', 'Date']).sum().reset_index()
world_data_select_agg['Active'] =  world_data_select_agg['Deaths'] - world_data_select_agg['Recovered']
countries = ['China', 'Italy', 'Iran', 'Spain', 'Germany', 'Switzerland', 'US', 'South Korea', 'United Kingdom', 'France', 'Netherlands', 'Japan', 'Romania']
plot_time_variation_countries_world(world_data_select_agg, countries,case_type='Deaths', size=4)
countries = ['China', 'Italy', 'Iran', 'Spain', 'Germany', 'Switzerland', 'US', 'South Korea', 'United Kingdom', 'France', 'Netherlands', 'Japan', 'Romania']
plot_time_variation_countries_world(world_data_select_agg, countries,case_type='currency rate', size=4)
countries = ['China', 'Italy', 'Iran', 'Spain', 'Germany', 'Switzerland', 'US', 'South Korea', 'United Kingdom', 'France',\
             'Netherlands', 'Austria', 'Japan', 'Romania']
plot_time_variation_countries(data_select_agg, countries, case_type = 'Active', size=4)
data_ps = data_df.sort_values(by = ['Province/State','Date'], ascending=False)
filtered_data_ps = data_ps.drop_duplicates(subset = ['Province/State'],keep='first').reset_index()

data_cr = data_df.sort_values(by = ['Country/Region','Date'], ascending=False)
filtered_data_cr = data_cr.drop_duplicates(subset = ['Country/Region'],keep='first').reset_index()

filtered_data_cr = filtered_data_cr.loc[~filtered_data_cr.Latitude.isna()]
filtered_data_cr = filtered_data_cr.loc[~filtered_data_cr.Longitude.isna()]
filtered_data = pd.concat([filtered_data_cr, filtered_data_ps], axis=0).reset_index()
m = folium.Map(location=[0,0], zoom_start=2)
max_val = max(filtered_data['Confirmed'])
HeatMap(data=filtered_data[['Latitude', 'Longitude', 'Confirmed']],\
        radius=15, max_zoom=12).add_to(m)
m
def plot_time_variation_mortality_countries_world(df, countries):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(4,4,figsize=(18, 16))

    for country in countries:
        plt.subplot(4,4,i + 1)
        df_ = df.loc[(df['Country']==country) & (df['Date'] > '2020-02-01')] 
        df_['date'] = df_['Date'].apply(lambda x: x.timetuple().tm_yday)
        #df_['Mortality (D/C)'] = df_['Deaths'] / df_['Confirmed'] * 100
        df_['Mortality (D/R)'] = df_['Deaths'] / df_['Recovered'] * 100
        #g = sns.lineplot(x="date", y='Mortality (D/C)', data=df_,  label='M (D/C)')
        g = sns.lineplot(x="date", y='currency rate', data=df_,  label='CR')
        g = sns.lineplot(x="date", y='Mortality (D/R)', data=df_,  label='M (D/R)')
        plt.title(f'{country}') 
        ax[i//4, i%4].set_yscale('log')
        plt.xlabel('')
        i = i + 1
    fig.suptitle('Mortality Deaths/Recovered (D/R) &their currency rates (CR)')
    plt.show()  
countries = ['Austria', 'Azerbaijan', 'China', 'Croatia', 
             'Denmark', 'Germany', 'Iceland', 'Iran', 
             'Malaysia', 'New Zealand',  'South Korea', 'Switzerland', 
             'Japan', 'Italy', 'US', 'Romania']
plot_time_variation_mortality_countries_world(world_data_select_agg, countries)
def plot_time_variation_mortality_countries_world(df, countries, title, size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,4*size))
    colors = ['red', 'blue', 'green', 'magenta', 'lightgreen', 'black']
    for i, country in enumerate(countries):
        dc_df = df.loc[df.Country == country].copy()
        dc_df_agg = dc_df.groupby(['Date']).sum().reset_index()
        #dc_df_agg['Mortality (D/C)'] = dc_df_agg['Deaths'] / dc_df_agg['Confirmed'] * 100
        dc_df_agg['Mortality (D/R)'] = dc_df_agg['Deaths'] / dc_df_agg['Recovered'] * 100
        g = sns.lineplot(x="Date", y='currency rate', data=dc_df_agg, linestyle='-', color=colors[i], label=f'Curency rate of - {country}')
        #ax.text(max(dc_df_agg['Date']), (dc_df_agg['Mortality (D/C)'].tail(1)), str(country))
        g = sns.scatterplot(x="Date", y='Mortality (D/R)', data=dc_df_agg, linestyle='-.', color=colors[i],label=f'Mortality (Deaths/Recovered) - {country}')
        #ax.text(max(dc_df_agg['Date']), (dc_df_agg['Mortality (D/R)'].tail(1)), str(country))
    plt.xlabel('Date')
    plt.xlim('2020-01-15','2020-06-12')
    ax.set_yscale('log')
    plt.ylabel(f'Mortality {title} [%]')
    plt.xticks(rotation=90)
    plt.title(f'Mortality percent {title}\nCalculated as currency rate (US dollar) and as Death / Recovered cases')
    ax.grid(color='black', linestyle='dashed', linewidth=1)
    plt.show()  
countries = ['US', 'Italy', 'China', 'South Korea', 'Japan', 'Romania']
plot_time_variation_mortality_countries_world(world_data_select_agg, countries, '- selection of World Countries',5)
world_data_select_agg.to_csv("world_data_select_agg.csv")
data =data_df
currdata=world_economy
data
currdata
print(currdata.shape)
currdata = currdata.iloc[:,0:3]
currdata.shape
currdata.head()

currdata = currdata[(currdata['date']>='2020-01-22 00:00:00')]
currdata.head()
print(f"Countries:{currdata['country'].nunique()}")
print(f"Rows:{data.shape[0]}, Columns: {data.shape[1]}")
data.head()
for column in data.columns:
    print(f"{column}:{data[column].dtype}")
print(f"Date - unique values: {data['Date'].nunique()} ({min(data['Date'])} - {max(data['Date'])})")
data['Date'] = pd.to_datetime(data['Date'])
currdata['date'] = pd.to_datetime(currdata['date'])
for column in data.columns:
    print(f"{column}:{data[column].dtype}")
for column in currdata.columns:
    print(f"{column}:{currdata[column].dtype}")
print(f"Date - unique values: {data['Date'].nunique()} ({min(data['Date'])} - {max(data['Date'])})")
missing_data(data)
missing_data(currdata)
print(f"Countries/Regions:{data['Country/Region'].nunique()}")
print(f"Province/State:{data['Province/State'].nunique()}")
covid_data=data
economy=currdata
print(currdata.shape)
print(covid_data.shape)
economy=economy.rename(columns={'country':'Country/Region','date':'Date'})
dataset=pd.merge(covid_data,economy,on=['Country/Region','Date'])
dataset
world = world_data_select_agg
world.tail(10)
import random as rd
import matplotlib.pyplot as plt
X = world[['Active','currency rate']]
plt.scatter(X["currency rate"],X["Active"],c='black')
plt.xlabel('Currency Rate')
plt.ylabel('Active Cases')
plt.show()
X1 = world[['Deaths','currency rate']]
X1 = X1.values
from sklearn.cluster import KMeans
Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(X1)
    kmeans.fit(X1)
    Error.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()
kmeans = KMeans(n_clusters=3)
y_means = kmeans.fit_predict(X1)
print(y_means)
kmeans.cluster_centers_
plt.scatter(X1[:,0],X1[:,1],c=y_means,cmap ='rainbow')
world.plot(x='Deaths', y='currency rate', style='o')  
plt.title('Deaths vs Currency Rate')  
plt.xlabel('Deaths')  
plt.ylabel('Currency Rate')  
plt.show()
A = world[['Country','Deaths']]
A.count()
A.Deaths.sum()
A.groupby('Country').sum()
T=A.groupby('Country', as_index=False)['Deaths'].sum()
T.columns
T.head()
T.dtypes
T['Country']=T['Country'].astype('category')
T.dtypes
T['Country']=T["Country"].cat.codes
T.head()
x = T.values
from sklearn.cluster import KMeans
Error =[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i).fit(x)
    kmeans.fit(x)
    Error.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 11), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()
kmeans3 = KMeans(n_clusters=3)
y_means3 = kmeans3.fit_predict(x)
print(y_means3)
kmeans3.cluster_centers_
plt.scatter(x[:,0],x[:,1],c=y_means3,cmap ='rainbow')
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
world.describe
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(world['currency rate'])
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(world['Deaths'])
xw = world['Deaths'].values.reshape(-1,1)
yw = world['currency rate'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(xw, yw, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
