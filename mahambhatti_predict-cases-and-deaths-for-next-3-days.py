!pip install opencage
import pandas as pd

import numpy as np

import datetime



import folium 

from folium import plugins



import matplotlib.pyplot as plt

import seaborn as sns



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.graph_objs import *



from opencage.geocoder import OpenCageGeocode



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error, mean_absolute_error



import warnings



%config InlineBackend.figure_format = 'retina'



warnings.filterwarnings('ignore')



%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
confirmed = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_confirmed_global.csv')

recovered = pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_recovered_global.csv')

deaths= pd.read_csv('/kaggle/input/ece657aw20asg4coronavirus/time_series_covid19_deaths_global.csv')
test_data= pd.read_csv('/kaggle/input/covid19-challenges/test_data_canada.csv')

test_data.head()
recovered_df = pd.read_csv('/kaggle/input/covid19-challenges/canada_recovered.csv')
recovered_df = pd.read_csv('/kaggle/input/covid19-challenges/canada_recovered.csv')

study = recovered_df.loc[recovered_df['date']=='2020-04-03',['province','cumulative_recovered']]

study.index = study['province']

study.drop('province',axis=1,inplace=True)

study
death_df = pd.read_csv('/kaggle/input/covid19-challenges/canada_mortality.csv')
test_intl=pd.read_csv('/kaggle/input/covid19-challenges/test_data_intl.csv')

test_intl.head()
intl_death= pd.read_csv('/kaggle/input/covid19-challenges/international_mortality.csv')



f_column = intl_death["deaths"]

intl_death.tail()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



case_vs_recovered = pd.concat([test_data['province'].value_counts(),study,death_df['province'].value_counts()],axis=1,sort=False)

case_vs_recovered.index.name='province'

case_vs_recovered.columns = ['Confirmed','Recovered','Death']





case_vs_recovered.fillna(0,inplace=True)

case_vs_recovered = case_vs_recovered.astype(int)





display(case_vs_recovered)



recover_rate = pd.DataFrame([elem + "%" if elem!="nan" else "0%" for elem in map(str,round(case_vs_recovered['Recovered'] / case_vs_recovered['Confirmed'] * 100,2))],index=case_vs_recovered.index,columns=['Recover Rate(%)'])

death_rate = pd.DataFrame([elem + "%" if elem!="nan" else "0%" for elem in map(str,round(case_vs_recovered['Death'] / case_vs_recovered['Confirmed'] * 100,2))],index=case_vs_recovered.index,columns=['Death Rate(%)'])

total_rate = pd.DataFrame([round(case_vs_recovered['Recovered'].sum() / case_vs_recovered['Confirmed'].sum() * 100, 2),round(case_vs_recovered['Death'].sum() / case_vs_recovered['Confirmed'].sum() * 100 , 2)],index=['Total Recover Rate','Total Death Rate'],columns=['Percentage(%)'])

display(recover_rate,death_rate,total_rate)
ax = case_vs_recovered.plot.bar(rot=0,figsize=(35,10),width=0.8)

plt.xlabel('Province'),plt.ylabel('Cases'),plt.autoscale()



for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

ax
last_update = '4/17/20'

current_cases = confirmed

current_cases = current_cases[['Country/Region',last_update]]



current_cases = current_cases.groupby('Country/Region').sum().sort_values(by=last_update, ascending=False)



current_cases['recovered'] = recovered[['Country/Region',last_update]].groupby('Country/Region').sum().sort_values(by=last_update,ascending=False)



current_cases['deaths'] = deaths[['Country/Region',last_update]].groupby('Country/Region').sum().sort_values(by=last_update,ascending=False)



current_cases['active'] = current_cases[last_update]-current_cases['recovered']-current_cases['deaths']



current_cases = current_cases.rename(columns={last_update:'confirmed'

                                              ,'recovered':'recovered'

                                              ,'deaths':'deaths'

                                              ,'active':'active'})



current_cases.style.background_gradient(cmap='Blues')
top_10_confirmed = confirmed[(confirmed['Country/Region']=='Brazil') |

                             (confirmed['Country/Region']=='US') |

                             (confirmed['Country/Region']=='China') |

                             (confirmed['Country/Region']=='Italy') |

                             (confirmed['Country/Region']=='Spain') |

                             (confirmed['Country/Region']=='Germany') |

                             (confirmed['Country/Region']=='France') |

                             (confirmed['Country/Region']=='Iran') |

                             (confirmed['Country/Region']=='United Kingdom') |

                             (confirmed['Country/Region']=='Russia') |

                             (confirmed['Country/Region']=='Turkey')]



top_10_confirmed = top_10_confirmed.groupby(top_10_confirmed['Country/Region']).sum()



top_10_confirmed = top_10_confirmed.drop(['Lat','Long'], axis = 1)

top_10_confirmed = top_10_confirmed.transpose()
top_10_countries = top_10_confirmed.drop('Brazil', axis = 1)



layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)'

    , plot_bgcolor='rgba(0,0,0,0)'

    , title="Cases over time in top 10 countries in confirmed cases numbers"

)



index = top_10_countries.index

data = top_10_countries



fig = go.Figure(data=[

    

    go.Line(name='US', x = index, y=data['US'])

    , go.Line(name='China', x = index, y=data['China'])

    , go.Line(name='Italy', x = index, y=data['Italy'])

    , go.Line(name='Spain', x = index, y=data['Spain'])

    , go.Line(name='Germany', x=index, y=data['Germany'])

    , go.Line(name='France', x=index , y=data['France'])

    , go.Line(name='Iran', x = index, y=data['Iran'])

    , go.Line(name='United Kingdom', x = index, y=data['United Kingdom'])

    , go.Line(name='Russia', x = index, y=data['Russia'])

    , go.Line(name='Turkey', x = index, y=data['Turkey'])

    

])



fig['layout'].update(layout)



fig.show()
# Taking confirmed cases since first case appear in 2/26/2020

cases = test_intl['cases'].groupby(test_intl['date']).sum().sort_values(ascending=True)

cases = cases[cases>0].reset_index().drop('date',axis=1)



deaths = intl_death['deaths'].groupby(intl_death['date']).sum().sort_values(ascending=True)

deaths = deaths[deaths>0].reset_index().drop('date',axis=1)



# add new 3 days here

cases = cases[0:55]

deaths = deaths[0:55]
# Converting our data into a array

days_since_first_case = np.array([i for i in range(len(cases.index))]).reshape(-1, 1)

intl_cases = np.array(cases).reshape(-1, 1)



days_since_first_death = np.array([i for i in range(len(deaths.index))]).reshape(-1, 1)

intl_deaths = np.array(deaths).reshape(-1, 1)
#Preparing indexes to predict next 15 days

days_in_future = 3

future_forcast = np.array([i for i in range(len(cases.index)+days_in_future)]).reshape(-1, 1)

adjusted_dates = future_forcast[:-3]



future_forcast_deaths = np.array([i for i in range(len(deaths.index)+days_in_future)]).reshape(-1, 1)

adjusted_dates_deaths = future_forcast_deaths[:-3]
#Splitting data into train and test to evaluate our model

X_train, X_test, y_train, y_test = train_test_split(days_since_first_case

                                                    , intl_cases

                                                    , test_size= 0.2

                                                    , shuffle=False

                                                    , random_state = 42) 



X_train_death, X_test_death, y_train_death, y_test_death = train_test_split(days_since_first_death

                                                    , intl_deaths

                                                    , test_size= 0.2

                                                    , shuffle=False

                                                    , random_state = 42) 
mae = 10000

degree = 0

for i in range(101):

    # Transform our cases data for polynomial regression

    poly = PolynomialFeatures(degree=i)

    poly_X_train = poly.fit_transform(X_train)

    poly_X_test = poly.fit_transform(X_test)

    poly_future_forcast = poly.fit_transform(future_forcast)



    # polynomial regression cases

    linear_model = LinearRegression(normalize=True, fit_intercept=False)

    linear_model.fit(poly_X_train, y_train)

    test_linear_pred = linear_model.predict(poly_X_test)

    linear_pred = linear_model.predict(poly_future_forcast)



    # evaluating with MAE and MSE

    m = mean_absolute_error(test_linear_pred, y_test)

    if(m<mae):

        mae = m

        degree = i

    if(i==100):

        print('the best mean absolute error is:',round(mae,2))

mae = 10000

degree = 0

for i in range(101):

    # Transform our death data for polynomial regression

    poly_death = PolynomialFeatures(degree=i)

    poly_X_train_death = poly_death.fit_transform(X_train_death)

    poly_X_test_death = poly_death.fit_transform(X_test_death)

    poly_future_forcast_death = poly_death.fit_transform(future_forcast_deaths)



    # polynomial regression deaths

    linear_model_death = LinearRegression(normalize=True, fit_intercept=False)

    linear_model_death.fit(poly_X_train_death, y_train_death)

    test_linear_pred_death = linear_model_death.predict(poly_X_test_death)

    linear_pred_death = linear_model_death.predict(poly_future_forcast_death)



    # evaluating with MAE and MSE

    m = mean_absolute_error(test_linear_pred_death, y_test_death)

    if(m<mae):

        mae = m

        degree = i

    if(i==100):

        print('the best mean absolute error is:',round(mae,2))
# Transform our cases data for polynomial regression

poly = PolynomialFeatures(degree=3)

poly_X_train = poly.fit_transform(X_train)

poly_X_test = poly.fit_transform(X_test)

poly_future_forcast = poly.fit_transform(future_forcast)



# Transform our death data for polynomial regression

poly_death = PolynomialFeatures(degree=3)

poly_X_train_death = poly_death.fit_transform(X_train_death)

poly_X_test_death = poly_death.fit_transform(X_test_death)

poly_future_forcast_death = poly_death.fit_transform(future_forcast_deaths)
# polynomial regression cases

linear_model = LinearRegression(normalize=True, fit_intercept=False)

linear_model.fit(poly_X_train, y_train)

test_linear_pred = linear_model.predict(poly_X_test)

linear_pred = linear_model.predict(poly_future_forcast)



# evaluating with MAE and MSE

print('Mean Absolute Error:', mean_absolute_error(test_linear_pred, y_test))
plt.figure(figsize=(12,7))



plt.plot(y_test, label = "Real cases")

plt.plot(test_linear_pred, label = "Predicted")

plt.title("Predicted vs Real cases", size = 20)

plt.xlabel('Days', size = 15)

plt.ylabel('Cases', size = 15)

plt.xticks(size=12)

plt.yticks(size=12)



# defyning legend config

plt.legend(loc = "upper left"

           , frameon = True

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1

           , prop={'size': 15});
plt.figure(figsize=(16, 9))



plt.plot(adjusted_dates

         , intl_cases

         , label = "Real cases")



plt.plot(future_forcast

         , linear_pred

         , label = "Polynomial Regression Predictions"

         , linestyle='dashed'

         , color='orange')



plt.title('Global cases over the time: Predicting Next 3 days', size=30)

plt.xlabel('Days Since 2/26/20', size=30)

plt.ylabel('Cases', size=30)

plt.xticks(size=20)

plt.yticks(size=20)



plt.axvline(len(X_train), color='black'

            , linestyle="--"

            , linewidth=1)

plt.text(18, 5000

         , "model training"

         , size = 15

         , color = "black")



plt.text((len(X_train)+0.2), 15000

         , "prediction"

         , size = 15

         , color = "black")



# defyning legend config

plt.legend(loc = "upper left"

           , frameon = True

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1

           , prop={'size': 15})



plt.show();
last_predictions = pd.DataFrame([13108,14898,16905

                                 ,17720,19683,21806

                                 ,23105,23135,27166

                                ,29127,31544,34091

                                ,36399,38975,41622]

                                , columns = ['Predicted']

                                , index = ['04/06/20','04/07/20','04/08/20'

                                           ,'04/09/20','04/10/20','04/11/20'

                                          ,'04/12/20','04/13/20','04/14/20'

                                          ,'04/15/20','04/16/20','04/17/20'

                                          ,'04/18/20','04/19/20','04/20/20'])

last_predictions['Real cases'] = [12056,13717,15927

                                  ,17857,19638,20727

                                  ,22169,23430,25262

                                 ,28320,30425,33682

                                 ,36599,38654,40581]



layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)'

    , plot_bgcolor='rgba(0,0,0,0)'

    , title="Last predictions vs Real cases"

)



fig = go.Figure(data=[

    

    go.Line(name='Predicted', x=last_predictions.index , y=last_predictions['Predicted'])

    , go.Line(name='Real cases', x=last_predictions.index , y=last_predictions['Real cases'])

    

])



fig.update_layout(barmode='stack')

fig['layout'].update(layout)



fig.show()
pd.DataFrame(linear_pred[len(cases):].astype('Int64'), columns = ['Predicted'], index = ['04-21-20','04-22-20','04-23-20']).style.background_gradient(cmap='Blues')
# polynomial regression deaths

linear_model_death = LinearRegression(normalize=True, fit_intercept=False)

linear_model_death.fit(poly_X_train_death, y_train_death)

test_linear_pred_death = linear_model_death.predict(poly_X_test_death)

linear_pred_death = linear_model_death.predict(poly_future_forcast_death)



# evaluating with MAE and MSE

print('MAE:', mean_absolute_error(test_linear_pred_death, y_test_death))
plt.figure(figsize=(12,7))



plt.plot(test_linear_pred_death, label = "Predicted")

plt.plot(y_test_death, label = "Real deaths")

plt.title("Predicted vs Real deaths", size = 20)

plt.xlabel('Days', size = 15)

plt.ylabel('Deaths', size = 15)

plt.xticks(size=12)

plt.yticks(size=12)



# defyning legend config

plt.legend(loc = "upper left"

           , frameon = True

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1

           , prop={'size': 15});
plt.figure(figsize=(16, 9))



plt.plot(adjusted_dates_deaths

         , intl_deaths

         , label = "Real deaths")



plt.plot(future_forcast_deaths

         , linear_pred_death

         , label = "Polynomial Regression Predictions"

         , linestyle='dashed'

         , color='red')



plt.title('Global deaths over the time: Predicting Next 3 days', size=30)

plt.xlabel('Days Since 03/17/20', size=30)

plt.ylabel('Deaths', size=30)

plt.xticks(size=20)

plt.yticks(size=20)



plt.axvline(len(X_train_death), color='black'

            , linestyle="--"

            , linewidth=1)



plt.text(10, 200

         , "model training"

         , size = 15

         , color = "black")

plt.text((len(X_train_death)+0.2), 600

         , "prediction"

         , size = 15

         , color = "black")



# defyning legend config

plt.legend(loc = "upper left"

           , frameon = True

           , ncol = 2 

           , fancybox = True

           , framealpha = 0.95

           , shadow = True

           , borderpad = 1

           , prop={'size': 15})



plt.show();
last_predictions = pd.DataFrame([550,634,727

                                 ,919,1060,1218

                                ,1321,1490,1673

                                ,1584,1722,1864

                                ,2249,2456,2676]

                                , columns = ['Predicted']

                                , index = ['04/06/20','04/07/20','04/08/20'

                                           ,'04/09/20','04/10/20','04/11/20'

                                          ,'04/12/20','04/13/20','04/14/20'

                                          ,'04/15/20','04/16/20','04/17/20'

                                          ,'04/18/20','04/19/20','04/20/20'])



last_predictions['Real cases'] = [553,667,800

                                  ,941,1056,1124

                                 ,1223,1328,1532

                                 ,1736,1924,2141

                                 ,2347,2462,2575]



layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)'

    , plot_bgcolor='rgba(0,0,0,0)'

    , title="Last predictions vs Real deaths"

)

fig = go.Figure(data=[

    

    go.Line(name='Predicted', x=last_predictions.index , y=last_predictions['Predicted'])

    , go.Line(name='Real deaths', x=last_predictions.index , y=last_predictions['Real cases'])

    

])



fig.update_layout(barmode='stack')

fig['layout'].update(layout)



fig.show()
pd.DataFrame(linear_pred_death[len(deaths):].astype('Int64'), columns = ['Predicted'], index = ['04-21-20','04-22-20','04-23-20']).style.background_gradient(cmap='Reds')