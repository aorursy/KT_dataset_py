import pandas as pd

import numpy as np

import plotly.express as px

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore') # jupyter notebookの警告回避
# 国とアメリカ各州の緯度経度データ

coordinates = pd.read_csv("/kaggle/input/latitude-and-longitude-for-every-country-and-state/world_country_and_usa_states_latitude_and_longitude_values.csv")



# 地域ごとに感染者、死者数、回復者数をまとめた時系列データ

df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
coordinates.head(100)
df.head()
# 緯度経度データの整形

country_coordinates = coordinates[['country_code','latitude','longitude','country']]

state_coordinates = coordinates[['usa_state_code','usa_state_latitude','usa_state_longitude','usa_state']]



# 時系列データの整形

df["Country/Region"].replace(["Mainland China"], "China", inplace=True) # セルの値の変換

df["Country/Region"].replace(["US"], "United States", inplace=True)

df["Country"] = df["Country/Region"]

df_jap = df[df["Country"] == "Japan"]

def convert_date(s):

    m, d, y = map(int, s.split("/"))

    return pd.Timestamp(y, m, d)



df_jap = df_jap[["ObservationDate", "Confirmed"]]

df_jap['ObservationDate'] = df_jap['ObservationDate'].map(convert_date)

TODAYS_DATE = np.max(df.ObservationDate) # 最新の観測値のみ使用

df = df[df.ObservationDate==np.max(df.ObservationDate)]



# 時系列データの整形 (国レベル)

df_deaths = pd.DataFrame(df.groupby("Country")["Deaths"].sum())

df_confirmed = pd.DataFrame(df.groupby("Country")["Confirmed"].sum())

df_recovered = pd.DataFrame(df.groupby("Country")["Recovered"].sum())

df_confirmed['Deaths'] = df_deaths['Deaths']

df_confirmed['Recovered'] = df_recovered["Recovered"]



df_global = df_confirmed # 感染者の死亡率, 感染者数, 死亡者数, 国の緯度経度

df_global["Mortality Rate"] = np.round((df_global.Deaths.values / df_global.Confirmed.values) * 100, 2)

df_global = df_global.reset_index()

df_global = df_global.merge(country_coordinates, left_on='Country', right_on='country') # coordinateとのマージ

df_global = df_global[['Country','Confirmed','Deaths','Mortality Rate',"Recovered",'latitude','longitude','country_code']]

df_global.columns = ['Country','Confirmed','Deaths','Mortality Rate', "Recovered",'Latitude','Longitude','Country_Code'] # 列名の改名

df_global.to_csv('/kaggle/working/global_covid19_mortality_rates.csv')



# 時系列データの整形 (アメリカ各州)

df_usa = df[df['Country/Region']=='United States']

df_usa = df_usa[df_usa.ObservationDate==np.max(df_usa.ObservationDate)]

df_usa['State'] = df_usa['Province/State']

df_usa['Mortality Rate'] = np.round((df_usa.Deaths.values/df_usa.Confirmed.values)*100,2)

df_usa.sort_values('Mortality Rate', ascending= False).head(10)

df_usa = df_usa.merge(state_coordinates, left_on='State', right_on='usa_state')

df_usa['Latitude'] = df_usa['usa_state_latitude']

df_usa['Longitude'] = df_usa['usa_state_longitude']

df_usa = df_usa[['State','Confirmed','Deaths','Recovered','Mortality Rate',"Recovered",'Latitude','Longitude','usa_state_code']]

df_usa.columns = ['State','Confirmed','Deaths','Recovered','Mortality Rate',"Recovered",'Latitude','Longitude','USA_State_Code']

df_usa.to_csv('/kaggle/working/usa_covid19_mortality_rates.csv')
df_global.head()
df_usa.head()
df_jap.head()
fig = px.choropleth(df_global,

                   locations="Country",

                   color="Confirmed",

                   locationmode="country names",

                   hover_name="Country",

                    range_color=[0, 300000],

                   title="Global COVID-19 Infection as of {}".format(TODAYS_DATE))

fig.show()



fig = px.choropleth(df_global, 

                    locations="Country", 

                    color="Deaths", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,25000],

                    title='Global COVID-19 Deaths as of {}'.format(TODAYS_DATE))

fig.show()



fig = px.choropleth(df_global, 

                    locations="Country", 

                    color="Mortality Rate", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,10],

                    title='Global COVID-19 Mortality Rates as of {}'.format(TODAYS_DATE))

fig.show()



fig = px.choropleth(df_global, 

                    locations="Country", 

                    color="Recovered", 

                    locationmode = 'country names', 

                    hover_name="Country",

                    range_color=[0,10000],

                    title='Global COVID-19 recovered as of {}'.format(TODAYS_DATE))

fig.show()
fig = px.bar(df_global.sort_values('Confirmed',ascending=False)[0:20], 

             x="Country", 

             y="Confirmed",

             title='Global COVID-19 Infections as of '+TODAYS_DATE)

fig.show()



fig = px.bar(df_global.sort_values('Deaths',ascending=False)[0:20], 

             x="Country", 

             y="Deaths",

             title='Global COVID-19 Deaths as of '+TODAYS_DATE)

fig.show()



fig = px.bar(df_global.sort_values('Deaths',ascending=False)[0:20], 

             x="Country", 

             y="Mortality Rate",

             title='Global COVID-19 Mortality Rates as of '+TODAYS_DATE+' for Countries with Top 20 Most Deaths')

fig.show()



fig = px.bar(df_global.sort_values('Mortality Rate',ascending=False)[0:20], 

             x="Country", 

             y="Mortality Rate",

             title='Global COVID-19 Mortality Rates as of '+TODAYS_DATE)

fig.show()



fig = px.bar(df_global.sort_values('Recovered',ascending=False)[0:20], 

             x="Country", 

             y="Recovered",

             title='Global COVID-19 Recovered as of '+TODAYS_DATE)

fig.show()




def makeCalcFrame(days, start_date):

    y, m, d = start_date

    t_1 = pd.Timestamp(y, m, d) # 計算開始日

    td = pd.Timedelta('1 days')

    #

    npd = [[t_1 + td * i, 0, 0, 0 ] for i in range(0,days)]

    df1 = pd.DataFrame(npd)

    df1.columns = ['date', 'Ppre','Pat', 'R0']

    #

    return df1



def mergeCalcFrame(df1, df2):

    return pd.merge(df1, df2, left_on='ObservationDate', right_on="date").fillna(0)



def calcR0(df, keys):

    lp = keys['lp']

    ip = keys['ip']

    nrow = len(df)

    getP = lambda s: df.loc[s, 'Confirmed'] if s < nrow else np.NaN

    for t in range(nrow):

        df.loc[t, 'Ppre'] = sum([ getP(s) for s in range(t+1, t + ip + 1)])

        df.loc[t, 'Pat' ] = getP(t + lp + ip)

        if df.loc[t, 'Ppre'] > 0:

            df.loc[t, 'R0'  ] = ip * df.loc[t, 'Pat'] / df.loc[t, 'Ppre']

        else:

            df.loc[t, 'R0'  ] = np.NaN

    return df
keys = {'lp':5, 'ip':8 }

cal_df = makeCalcFrame(80, (2020, 2, 1))
df_jap= mergeCalcFrame(df_jap, cal_df)
result_df = calcR0(df_jap, keys)
def showResult(df, title):

    # R0=1 : 収束のためのターゲット

    ptgt = pd.DataFrame([[df.iloc[0,0],1],[df.iloc[len(df)-1,0],1]])

    ptgt.columns = ['ObservationDate','R0']

    # show R0

    plt.rcParams["font.size"] = 12

    ax = df.plot(title=title,x='ObservationDate',y='R0', figsize=(10,7))

    ptgt.plot(x='ObservationDate',y='R0',style='r--',ax=ax)

    ax.grid(True)

    ax.set_ylim(0,)

    plt.show()





showResult(result_df, "COVID-19 Japan basic reprodection number")