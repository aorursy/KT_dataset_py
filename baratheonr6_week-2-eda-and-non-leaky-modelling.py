# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the "../input/" directory.

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input/'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # Any results you write to the current directory are saved as output.
import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

for dirname, _, filenames in os.walk('/kaggle/input/covid19-global-forecasting-week-2'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sns

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

sns.set()

pd.options.display.max_rows=1000

train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

train['Date_str']=train['Date'].copy()

train['Date']=pd.to_datetime(train['Date'])

test['Date_str']=test['Date'].copy()

test['Date']=pd.to_datetime(test['Date'])
total_country_date=train.groupby(['Country_Region','Date'])['ConfirmedCases','Fatalities'].sum().reset_index()

total_country_max=total_country_date.groupby(['Country_Region'])['ConfirmedCases','Fatalities'].max().sort_values(by=['ConfirmedCases'],ascending=False).reset_index()

total_country_max['Fatality Rate']=np.round((100*total_country_max['Fatalities'])/total_country_max['ConfirmedCases'],2)

total_country_max.head(10)
import plotly.express as px

toplot=total_country_max[total_country_max['ConfirmedCases']>=10]

def plot_cases_fatal(total_country_max):

    fig = go.Figure()

    fig = px.scatter(total_country_max, x="ConfirmedCases", y="Fatalities",hover_name=list(total_country_max.Country_Region.values),

                     hover_data=["Fatality Rate",'ConfirmedCases','Fatalities'],color="ConfirmedCases")

    fig.update_layout(height=500, width=900,title_text="Fatality vs  ConfirmedCases ")

    xd=np.arange(1/0.1,total_country_max['ConfirmedCases'].max(),100)

    xd_5=np.arange(1/0.05,total_country_max['ConfirmedCases'].max(),100)

    xd_2=np.arange(1/0.02,total_country_max['ConfirmedCases'].max(),100)

    xd_1=np.arange(1/0.01,total_country_max['ConfirmedCases'].max(),100)

    yd=0.1*xd

    yd_5=0.05*xd_5

    yd_2=0.02*xd_2

    yd_1=0.01*xd_1



    fig.add_trace(go.Scatter(x=xd, y=yd, name='10% fatality rate',opacity=0.7,

                             line=dict(color='grey', width=3,

                                  dash='dash')))

    fig.add_trace(go.Scatter(x=xd_5, y=yd_5, name='5% fatality rate',opacity=0.5,

                             line=dict(color='grey', width=3,

                                  dash='dash')))

    fig.add_trace(go.Scatter(x=xd_2, y=yd_2, name='2% fatality rate',opacity=0.3,

                             line=dict(color='grey', width=3,

                                  dash='dash')))

    fig.add_trace(go.Scatter(x=xd_1, y=yd_1, name='1% fatality rate',opacity=0.3,

                             line=dict(color='grey', width=3,

                                  dash='dash')))



    fig.update_layout(coloraxis_showscale=False)

    fig.update_layout( yaxis_type="log", xaxis_type="log")

    fig.update_layout(

            xaxis_title="ConfirmedCases",

            yaxis_title="Fatalities",

            )

    fig.show()

plot_cases_fatal(toplot)
def plot_dist(total_country_date,total_country_max,combinations_to_plot=[[10,100],[100,1000],[1000,10000]]):

    val=[]

    

    my_fig = make_subplots(rows=1, cols=2)

    my_fig.update_layout(height=500, width=1200,title_text=" Distribution of Days")

    for combinations in combinations_to_plot:

        comb_val={}

        lower_limit,upper_limit=combinations[0],combinations[1]

        for country in total_country_date['Country_Region'].unique():

            if total_country_max[total_country_max['Country_Region']==country].ConfirmedCases.values[0]>=upper_limit:

                country_data=total_country_date[(total_country_date['Country_Region']==country)]

                if (country_data[country_data.ConfirmedCases<=lower_limit].tail(1).ConfirmedCases.values.size==0):

                    continue

                elif (country_data[country_data.ConfirmedCases<=lower_limit].tail(1).ConfirmedCases.values[0]==0):

                    continue 

                else:

                    comb_val[country]=pd.to_timedelta((country_data[country_data.ConfirmedCases>=upper_limit].head(1).Date.values[0]-country_data[country_data.ConfirmedCases<=lower_limit].tail(1).Date.values[0]),'D').days

        val.append(comb_val)

    fig = ff.create_distplot([list(xx.values()) for xx in val],[ '{} to {}'.format(xx[0],xx[1]) for xx in combinations_to_plot])

    

    distplot=fig['data']

    my_fig = make_subplots(rows=1, cols=2)

    my_fig.update_layout(height=500, width=1200)

    

    for i in range(2*len(combinations_to_plot)):

        my_fig.append_trace(distplot[i], 1, 1)



    

    for xx,comb in zip(val,combinations_to_plot):

        my_fig.add_trace(go.Box(y=list(xx.values()),name='{} to {}'.format(comb[0],comb[1] )),row=1,col=2)

    my_fig.update_layout(height=500, width=1200,title_text=" Distribution of Days")

    my_fig.show()

            

plot_dist(total_country_date,total_country_max)      
world_date_data=train.groupby(['Date','Date_str'])['ConfirmedCases','Fatalities'].sum().reset_index()

world_date_data['new_cases']=world_date_data.ConfirmedCases-world_date_data.ConfirmedCases.shift()

world_date_data['new_deaths']=world_date_data.Fatalities-world_date_data.Fatalities.shift()

fig = make_subplots(rows=1, cols=1)

fig.update_layout(height=500, width=800,title_text='Cumilative visualization for the world')

fig.add_trace(

                  go.Scatter(

                            x=world_date_data['Date_str'].values,

                            y=world_date_data['ConfirmedCases'].values

                            ,mode='lines+markers',

                            text=list(world_date_data.Fatalities.values),

                             name='world cumulative cases',fill='tozeroy'

                            ) 

                  )

fig.add_trace(

                  go.Scatter(

                            x=world_date_data['Date_str'].values,

                            y=world_date_data['new_cases'].values

                            ,mode='lines+markers',

                            text=list(world_date_data.Fatalities.values),fill='tozeroy',

                             name='daily new confirmed cases',

                            ) 

                  )

fig.add_trace(

                  go.Scatter(

                            x=world_date_data['Date_str'].values,

                            y=world_date_data['Fatalities'].values

                            ,mode='lines+markers',

                            text=list(world_date_data.Fatalities.values),fill='tozeroy',

                             name='world cumulative Fatalities',

                            ) 

                  )

fig.update_layout( yaxis_type="log")
pop_dict={'Afghanistan': 38928346,'Albania': 2877797,'Algeria': 43851044,'Andorra': 77265,'Argentina': 45195774,'Australia': 25499884,'Austria': 9006398,'Azerbaijan': 10139177,'Bahrain': 1701575,'Bangladesh': 164689383,'Belgium': 11589623,

          'Bosnia and Herzegovina': 3280819,'Brazil': 212559417,'Bulgaria': 6948445,'Burkina Faso': 20903273,'Canada': 37742154,'Chile': 19116201,'China': 1439323776,'Colombia': 50882891,'Costa Rica': 5094118,'Croatia': 4105267,'Cuba': 11326616,

          'Cyprus': 1207359,'Denmark': 5792202,'Dominican Republic': 10847910,'Ecuador': 17643054,'Egypt': 102334404,'Finland': 5540720,'France': 65273511,'Gabon': 2225734,'Germany': 83783942,'Ghana': 31072940,

          'Greece': 10423054,'Guatemala': 17915568,'Guyana': 786552,'Hungary': 9660351,'Iceland': 341243,'India': 1380004385,'Indonesia': 273523615,'Iran': 83992949,'Iraq': 40222493,'Ireland': 4937786,'Israel': 8655535,'Italy': 60461826,

          'Jamaica': 2961167,'Japan': 126476461,'Kazakhstan': 18776707,'Korea, South': 51269185,'Lebanon': 6825445,'Lithuania': 2722289,'Luxembourg': 625978,'Malaysia': 32365999,'Martinique': 375265,'Mauritius': 1271768,'Mexico': 128932753,

          'Moldova': 4033963,'Montenegro': 628066,'Morocco': 36910560,'Netherlands': 17134872,'Nigeria': 206139589,'North Macedonia': 2083374,'Norway': 5421241,'Pakistan': 220892340,'Panama': 4314767,'Paraguay': 7132538,'Peru': 32971854,

          'Philippines': 109581078,'Poland': 37846611,'Portugal': 10196709,'Romania': 19237691,'Russia': 145934462,'San Marino': 33931,'Saudi Arabia': 34813871,'Serbia': 8737371,'Seychelles': 98347,'Singapore': 5850342,

          'Slovakia': 5459642,'Slovenia': 2078938,'Somalia': 15893222,'South Africa': 59308690,'Spain': 46754778,'Sri Lanka': 21413249,'Sudan': 43849260,'Suriname': 586632,'Sweden': 10099265,'Switzerland': 8654622,'Thailand': 69799978,

          'Tunisia': 11818619,'Turkey': 84339067,'US': 331002651,'Ukraine': 43733762,'United Arab Emirates': 9890402,'United Kingdom': 67886011,'Uruguay': 3473730,'Uzbekistan': 33469203,'Venezuela': 28435940,'Vietnam': 97338579}

class plot_per_million:

    def __init__(self,list_of_count_to_plot,data,column_to_plot):

        self.list_of_count_to_plot=list_of_count_to_plot

        self.train=data

        self.column_to_plot=column_to_plot

    

    def initiate_plotting(self,):

        fig = make_subplots(rows=1, cols=1)

        fig.update_layout(height=500, width=800, title_text="Number of {} per million population vs no of days since 10 {} per million population ".format(self.column_to_plot,self.column_to_plot))

        for countt in self.list_of_count_to_plot:

            self.plotter(self.return_count_data(countt,self.train),countt,fig)

        fig.update_layout(

            xaxis_title="no of days since 10 cases",

            yaxis_title="Number of {} per million population".format(self.column_to_plot),

                            )

        fig.update_layout( yaxis_type="log")

        fig.show()



    def return_count_data(self,countt,train):

        iran_db=train[train['Country_Region']==countt].groupby(['Date','Date_str']).agg({'ConfirmedCases':['sum'],'Fatalities':['sum']}).reset_index()

        iran_db.columns=[	'Date',	'Date_str',	'ConfirmedCases',	'Fatalities']

        iran_db_tp=iran_db[(iran_db[self.column_to_plot]/(pop_dict[countt]/1000000))>=10].copy()

        iran_db_tp['to_sub']=[iran_db_tp.head(1).Date_str.values[0]]*iran_db_tp.shape[0]

        iran_db_tp['to_sub']=pd.to_datetime(iran_db_tp['to_sub'])

        iran_db_tp['diff']=(iran_db_tp['Date']-iran_db_tp['to_sub']).dt.days

        return iran_db_tp



    def plotter(self,db_tp,countt,fig):

        fig.add_trace(go.Scatter(

                            x=db_tp['diff'].values,

                            y=db_tp[self.column_to_plot].values/(pop_dict[countt]/1000000)

                             ,mode='lines+markers',

                              text=list(db_tp.Date_str.values),name=countt

                            ) )

list_of_countries_to_plot=list(total_country_max.head(10).Country_Region.values)

Feature_to_plot_on='ConfirmedCases'

plotter_obj=plot_per_million(list_of_countries_to_plot,train,Feature_to_plot_on)

plotter_obj.initiate_plotting()
train=train.fillna('nan')

test=test.fillna('nan')

total_highest_ccp=train.groupby(['Country_Region','Province_State']).ConfirmedCases.max().reset_index().sort_values(by=['ConfirmedCases'],ascending=False).head(10)[['Country_Region','Province_State','ConfirmedCases']]

to_map_dict={}

for c in list(total_highest_ccp.Country_Region.values):

    to_map_dict[c]=list(total_highest_ccp[total_highest_ccp['Country_Region']==c].Province_State.values)

to_map_dict 
train=train.fillna('nan')

test=test.fillna('nan')

from scipy.optimize import curve_fit

def sigmoid(x, m, alpha, beta):

    return m / ( 1 + np.exp(-beta * (x - alpha)))

test['ConfirmedCases']=[0]*test.shape[0]

test['Fatalities']=[0]*test.shape[0]

model_param_dict_cc={}

model_param_dict_f={}

total_world=pd.DataFrame()

fig, ax = plt.subplots(int(len(to_map_dict.values())/2)+1, 2,figsize=(20,10*int(len(to_map_dict.keys())/2)))

ax=ax.ravel()

counter=0

for country in test['Country_Region'].unique():

    for province in test[test['Country_Region']==country].Province_State.unique():

        

        # Removing Dates that are present in train

        not_to_include=list(set(train[(train['Country_Region']==country)&(train.Province_State==province)].Date_str.values).intersection(\

            set(test[(test['Country_Region']==country)&(test.Province_State==province)].Date_str.values)))

        to_train=train[(train['Country_Region']==country)&(train.Province_State==province)&(~(train.Date_str.isin(not_to_include)))].copy()

        to_train['days']=(pd.to_datetime(to_train['Date'])-pd.to_datetime(\

                         to_train[to_train.Date==to_train.Date.min()].Date_str.unique()[0])).dt.days

        

        to_test=test[(test['Country_Region']==country)&(test.Province_State==province)].copy()

        to_train['istest']=[0]*to_train.shape[0]

        to_test['istest']=[1]*to_test.shape[0]



        total=pd.concat([to_train,to_test],sort=False)

        total['pred_cc']=[0]*total.shape[0]

        total['pred_f']=[0]*total.shape[0]

        total['days']=(pd.to_datetime(total['Date'])-pd.to_datetime(\

                         total[total.Date==total.Date.min()].Date_str.unique()[0])).dt.days

        if to_train.ConfirmedCases.max()<10:

            total['pred_cc']=[to_train.ConfirmedCases.max()]*total.shape[0]

        else:

            popt_cc, pcov_cc = curve_fit(sigmoid, to_train['days'].values,to_train.ConfirmedCases.values,bounds=([0,0,0],np.inf),maxfev=5000)

            total['pred_cc']=sigmoid(total.days.values,popt_cc[0],popt_cc[1],popt_cc[2])

            model_param_dict_cc['{}|{}'.format(country,province)]=popt_cc

            

        if to_train.Fatalities.max()<10:

            total['pred_f']=[to_train.Fatalities.max()]*total.shape[0]

        else:

            popt_f, pcov_f = curve_fit(sigmoid, to_train['days'].values,to_train.Fatalities.values,bounds=([0,0,0],np.inf),maxfev=5000)

            total['pred_f']=sigmoid(total.days.values,popt_f[0],popt_f[1],popt_f[2])

            model_param_dict_f['{}|{}'.format(country,province)]=popt_f

        total_world=total_world.append(total)

        

        

        if (country in list(to_map_dict.keys())):

            if province in list(np.array(to_map_dict[country]).ravel()):



                data=total_world[(total_world['Country_Region']==country)&(total_world['Province_State']==province)&

                                (total_world['istest']==0)]

                data_pred=total_world[(total_world['Country_Region']==country)&(total_world['Province_State']==province)&

                                (total_world['istest']==1)]

                ax[counter].scatter(data['days'].values,data['ConfirmedCases'].values,color='blue',label='true')

                ax[counter].scatter(data_pred['days'].values,data_pred['pred_cc'].values,color='red',label='predicted')

                ax[counter].title.set_text('{} {}'.format(country,province))

                ax[counter].legend()

                counter+=1

        

        

    

plt.show()
zz=total_world[total_world.istest==1][['ForecastId','pred_cc','pred_f']].astype(int)

zz.columns=['ForecastId','ConfirmedCases','Fatalities']

zz.to_csv('submission.csv',index=False)