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
import matplotlib.pyplot as plt

import itertools

import statsmodels.api as sm

from scipy import stats

import numpy as np

import pandas as pd

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from matplotlib import rcParams

rcParams['figure.figsize'] = 20,10
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")

submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/submission.csv")
country_dict= dict()

for itr in range(len(train)):

    if train.loc[itr]['Country_Region'] not in country_dict.keys():

        country_dict[train.loc[itr]['Country_Region']]= []

    else:

        if len(country_dict[train.loc[itr]['Country_Region']])>=75:

            continue

    country_dict[train.loc[itr]['Country_Region']].append([[train.loc[itr]['Date']],[train.loc[itr]['ConfirmedCases']],[train.loc[itr]['Fatalities']]])    

    

time_series_dict = dict()

for country in country_dict.keys():

    for case in ['ConfirmedCases','Fatalities']:

        tsz=train.loc[(train['Country_Region']==country)]

        tsz=tsz[['Date',case]]

        x = []

        for itr in tsz.index:

            x.append([pd.to_datetime(tsz.loc[itr]['Date']),tsz.loc[itr][case]])

        tsz = pd.DataFrame(x,columns = ['Date',case])

        tsz=tsz.set_index('Date')

        tsz

        if country not in time_series_dict.keys():

            time_series_dict[country] = dict()

        time_series_dict[country][case] = tsz
rank_country = dict()

for country in country_dict.keys():

    rank_country[country]=[max(time_series_dict[country]['ConfirmedCases']['ConfirmedCases']),max(time_series_dict[country]['Fatalities']['Fatalities'])]

rank_country = sorted(rank_country.items(), key = lambda kv:(kv[1][0],kv[1][1], kv[0]),reverse = True)[:20]



labels = [y[0] for y in rank_country]

ConfirmedCases = [y[1][0] for y in rank_country]

Fatalities = [y[1][1] for y in rank_country]



x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, ConfirmedCases, width, label='ConfirmedCases',color = '#FFBF00')

rects2 = ax.bar(x + width/2, Fatalities, width, label='Fatalities',color = 'blue')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Number of Cases',fontsize=30, fontweight=20)

ax.set_title('COVID-19',fontsize=30, fontweight=20)

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()





def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')



autolabel(rects1)

autolabel(rects2)

fig.tight_layout()

plt.show()
color_pallete = ['#FFBF00','red']

for country in ['India','China','US','Italy','Spain']:

    case_number = 1

    for case in ['ConfirmedCases','Fatalities']:

        plt.subplot(1,2,case_number)

        plt.title(case, loc='center', fontsize=20, fontweight=10)

        if case_number==2:

            plt.ylim(bottom,top)

        plt.plot(time_series_dict[country][case][:74], color=color_pallete[case_number-1], linewidth=3, alpha=1)

        plt.xlabel('Date', fontsize=20)

        plt.ylabel('Number of Cases', fontsize=20)

        if case_number==1:

            bottom,top = plt.ylim()

        case_number = case_number + 1

    plt.suptitle(country, fontsize=30, fontweight=20)

    plt.show()
import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.metrics import mean_squared_error, mean_absolute_error
train.Date = pd.to_datetime(train.Date)

test.Date = pd.to_datetime(test.Date)

train['Date'] = train['Date'].dt.strftime("%d%m").astype(int)

test['Date'] = test['Date'].dt.strftime("%d%m").astype(int)
country_dict= dict()

for itr in range(len(train)):

    if train.loc[itr]['Country_Region'] not in country_dict.keys():

        country_dict[train.loc[itr]['Country_Region']]= dict()

    if str(train.iloc[itr]['Province_State']) != 'nan':

        if train.loc[itr]['Province_State'] not in country_dict[train.loc[itr]['Country_Region']].keys():

            country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Province_State']] = dict()

            country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Province_State']]['ConfirmedCases'] = []

            country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Province_State']]['Fatalities'] = []

        country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Province_State']]['ConfirmedCases'].append([train.loc[itr]['Date'],train.loc[itr]['ConfirmedCases']])

        country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Province_State']]['Fatalities'].append([train.loc[itr]['Date'],train.loc[itr]['Fatalities']])

        if str(train.loc[itr]['Province_State']) == 'nan':

            if train.loc[itr]['Country_Region'] not in country_dict[train.loc[itr]['Country_Region']].keys():

                country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Country_Region']] = dict()

                country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Country_Region']]['ConfirmedCases'] = []

                country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Country_Region']]['Fatalities'] = []

                country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Country_Region']]['ConfirmedCases'].append([train.loc[itr]['Date'],train.loc[itr]['ConfirmedCases']])

                country_dict[train.loc[itr]['Country_Region']][train.loc[itr]['Country_Region']]['Fatalities'].append([train.loc[itr]['Date'],train.loc[itr]['Fatalities']])
test_dates = []

for itr in range(len(test)):

    if test.iloc[itr]['Country_Region'] == 'Afghanistan':

        test_dates.append(test.iloc[itr]['Date'])

test_dates = np.array(test_dates)
pred_dict = dict()

for country in country_dict.keys():

    pred_dict[country] = dict()

    for province in country_dict[country].keys():

        train_x_ConfirmedCases = train_x_Fatalities = pd.DataFrame(country_dict[country][province]['ConfirmedCases'])[0].values.reshape(-1,1)

        train_y_ConfirmedCases = pd.DataFrame(country_dict[country][province]['ConfirmedCases'])[1].values

        train_y_Fatalities = pd.DataFrame(country_dict[country][province]['Fatalities'])[1].values

        test_x_ConfirmedCases = test_x_Fatalities = test_dates.reshape(-1,1)

        pred_y_ConfirmedCases = xgb.XGBRegressor(n_estimators=500).fit(train_x_ConfirmedCases, train_y_ConfirmedCases).predict(test_x_ConfirmedCases)

        pred_y_Fatalities = xgb.XGBRegressor(n_estimators=500).fit(train_x_Fatalities, train_y_Fatalities).predict(test_x_Fatalities)

        pred_dict[country][province] = dict()

        pred_dict[country][province]['ConfirmedCases'] = pred_y_ConfirmedCases

        pred_dict[country][province]['Fatalities'] = pred_y_Fatalities
ForecastId = 1

submission_out = []

for country in country_dict.keys():

    for province in country_dict[country].keys():

        for i in range(len(pred_dict[country][province]['ConfirmedCases'])):

            submission_out.append([ForecastId,pred_dict[country][province]['ConfirmedCases'][i],pred_dict[country][province]['Fatalities'][i]])

            ForecastId = ForecastId + 1

submission_file = pd.DataFrame(submission_out,columns=['ForecastId','ConfirmedCases','Fatalities'])

submission_file.to_csv('submission.csv',index = False)