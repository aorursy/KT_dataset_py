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
PATH_WEEK4 = '/kaggle/input/covid19-global-forecasting-week-4'



df_Train = pd.read_csv(f'{PATH_WEEK4}/train.csv', parse_dates=["Date"], engine='python')

df_Test = pd.read_csv(f'{PATH_WEEK4}/test.csv', parse_dates=["Date"], engine='python')

#df_Sub = pd.read_csv(f'{PATH_WEEK4}/submission.csv')
def fillState(state, country):

    if state == EMPTY_VAL: return country

    return state
df_Train.rename(columns={'Country_Region':'Country'}, inplace=True)

df_Test.rename(columns={'Country_Region':'Country'}, inplace=True)



EMPTY_VAL = "EMPTY_VAL"



df_Train.rename(columns={'Province_State':'State'}, inplace=True)

df_Train['State'].fillna(EMPTY_VAL, inplace=True)

df_Train['State'] = df_Train.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)



df_Test.rename(columns={'Province_State':'State'}, inplace=True)

df_Test['State'].fillna(EMPTY_VAL, inplace=True)

df_Test['State'] = df_Test.loc[:, ['State', 'Country']].apply(lambda x : fillState(x['State'], x['Country']), axis=1)
df_groupByCountry = df_Train.loc[:, ['Country', 'State', 'ConfirmedCases', 'Fatalities']].groupby(['Country', 'State']).max().reset_index().groupby('Country').sum().sort_values(by='ConfirmedCases', ascending=False).reset_index()

df_groupByCountry[:15].style.background_gradient(cmap='viridis_r')
import plotly.express as px



countries = df_groupByCountry.Country.unique().tolist()

df_plot = df_Train.loc[(df_Train.Country.isin(countries[:10])) & (df_Train.Date >= '2020-03-11'), ['Date', 'Country', 'State', 'ConfirmedCases', 'Fatalities']].groupby(['Date', 'Country', 'State']).max().reset_index().groupby(['Date', 'Country']).sum().sort_values(by='ConfirmedCases', ascending=False).reset_index()



fig = px.bar(df_plot, x="Date", y="ConfirmedCases", color="Country", barmode="stack")

fig.update_layout(title='Rise of Confirmed Cases around top 10 countries', annotations=[dict(x='2020-03-21', y=150, xref="x", yref="y", text="Coronas Rise exponentially from here", showarrow=True, arrowhead=1, ax=-150, ay=-150)])

fig.show()
df_Train.loc[: , ['Country', 'State', 'ConfirmedCases', 'Fatalities']].groupby(['Country', 'State']).max().reset_index().nlargest(15, "ConfirmedCases").style.background_gradient(cmap='nipy_spectral')
import plotly.express as px



df_plot = df_Train.loc[: , ['Date', 'Country', 'ConfirmedCases', 'Fatalities']].groupby(['Date', 'Country']).max().reset_index()



df_plot.loc[:, 'Date'] = df_plot.Date.dt.strftime("%Y-%m-%d")

df_plot.loc[:, 'Size'] = np.power(df_plot["ConfirmedCases"]+1,0.3)-1 #np.where(df_plot['Country'].isin(['China', 'Italy']), df_plot['ConfirmedCases'], df_plot['ConfirmedCases']*300)



fig = px.scatter_geo(df_plot,

                     locations="Country",

                     locationmode = "country names",

                     hover_name="Country",

                     color="ConfirmedCases",

                     animation_frame="Date", 

                     size='Size',

                     #projection="natural earth",

                     title="Rise of Coronavirus Confirmed Cases")

fig.show()
import plotly.express as px



countries = df_groupByCountry.Country.unique().tolist()

df_plot = df_Train.loc[df_Train.Country.isin(countries[:10]), ['Date', 'Country', 'ConfirmedCases']].groupby(['Date', 'Country']).max().reset_index()



fig = px.line(df_plot, x="Date", y="ConfirmedCases", color='Country')

fig.update_layout(title='No.of Confirmed Cases per Day for Top 10 Countries',

                   xaxis_title='Date',

                   yaxis_title='No.of Confirmed Cases')

fig.show()
MIN_TEST_DATE = df_Test.Date.min()



df_train = df_Train.loc[df_Train.Date < MIN_TEST_DATE, :]

y1_Train = df_train.iloc[:, -2]

y2_Train = df_train.iloc[:, -1]
def extractDate(df, colName = 'Date'):

    """

    This function does extract the date feature in to multiple features

    - week, day, month, year, dayofweek

    """

    assert colName in df.columns

    df = df.assign(

        week = df.loc[:, colName].dt.week,

        month = df.loc[:, colName].dt.month,

        #year = df.loc[:, colName].dt.year,

        day = df.loc[:, colName].dt.day,

        dayofweek = df.loc[:, colName].dt.dayofweek,

        dayofyear = df.loc[:, colName].dt.dayofyear,

                  )

    return df
def createNewDataset(df):

    """

    This function does create a new dataset for modelling.

    """

    df_New = df.copy()

    

    #df_New = extractDate(df_New)

    df_New.loc[:, 'Date_Int'] = (df_New.loc[:, 'Date'].dt.strftime("%m%d")).astype('int16')

    df_New.drop(columns=['Date'], axis=1, inplace=True)

    

    #df_New.loc[:, 'Country_State'] = df_New.loc[:, 'Country'] + '_' + df_New.loc[:, 'State']

    #df_New.loc[:, 'Country_State'] = df_New[["State", "Country"]].apply(lambda row: str(row[0]) + "_" + str(row[1]),axis=1)

    #df_New.drop(columns=['Country', 'State'], axis=1, inplace=True)

    

    return df_New
X_Train = createNewDataset(df_train)

X_Test = createNewDataset(df_Test)
X_Train[X_Train.Country == 'China'].tail()
df_Train[df_Train.Country == 'India'].tail()
days = range(1, 11)

def getDaysShift(df):

    newDCols = []

    # CD{day} - Change in last 1 day or 2 days, etc

    for day in days:

        newDCol = f'D_{day}'

        df['C'+newDCol] = df.groupby(['Country', 'State'])['LConfirmedCases'].shift(day)

        df['F'+newDCol] = df.groupby(['Country', 'State'])['LFatalities'].shift(day)

        newDCols.append(newDCol)

    

    return df
days_change = [1, 2, 3, 5, 7, 10]

def getChangeGrowth(df):

    newCCols = []

    newGCols = []

    # CC{day} - Where CC refers to Confirmedcases Change per given {day(s)}

    for day in days_change:

        newCCol = f'C_{day}'

        df['C'+newCCol] = df['LConfirmedCases'] - df[f'CD_{day}']

        df['F'+newCCol] = df['LFatalities'] - df[f'FD_{day}']

        newCCols.append(newCCol)

        newGCol = f'G_{day}'

        df['C'+newGCol] = df['C'+newCCol] / df[f'CD_{day}']

        df['F'+newGCol] = df['F'+newCCol] / df[f'FD_{day}']

        newGCols.append(newGCol)



    df.fillna(0, inplace=True)

    return df
windows = [1, 2, 3, 5, 7]

def getMA(df):

    newCMACols = []

    newGMACols = []

    for window in windows:

        for day in days_change:

            newCMACol = f'CMA_{day}_{window}'

            df['C'+newCMACol] = df[f'CC_{day}'].rolling(window).mean()

            df['F'+newCMACol] = df[f'FC_{day}'].rolling(window).mean()

            newCMACols.append(newCMACol)

            newGMACol = f'GMA_{day}_{window}'

            df['C'+newGMACol] = df[f'CG_{day}'].rolling(window).mean()

            df['F'+newGMACol] = df[f'FG_{day}'].rolling(window).mean()

            newGMACols.append(newGMACol)



    df.fillna(0, inplace=True)

    return df
cases = [1, 50, 100, 500, 1000, 5000, 35000, 75000, 100000]

def getCDSC(df):

    newCDSCCols = []

    for case in cases:

        newDSCCol = f'{case}_CDSC'

        df.loc[df.CD_1 == 0, newDSCCol] = 0

        df.loc[df.CD_1 >= case, newDSCCol] = df[df.CD_1 >= case].groupby(['Country', 'State']).cumcount()

        newCDSCCols.append(newDSCCol)

        

    df.fillna(0, inplace=True)

    return df
deaths = [1, 50, 100, 500, 1000, 5000, 35000]

def getFDSC(df):

    newFDSCCols = []

    for death in deaths:

        newDSCCol = f'{death}_FDSC'

        df.loc[df.FD_1 == 0, newDSCCol] = 0

        df.loc[df.FD_1 >= death, newDSCCol] = df[df.FD_1 >= death].groupby(['Country', 'State']).cumcount()

        newFDSCCols.append(newDSCCol)

    

    df.fillna(0, inplace=True)

    return df
df = pd.concat([df_Train, df_Test[df_Test.Date > df_Train.Date.max()]], axis=0, sort=False, ignore_index=True)
df['LConfirmedCases'] = np.log1p(df['ConfirmedCases'])

df['LFatalities'] = np.log1p(df['Fatalities'])
df.loc[(df.Date >= df_Test.Date.min()) & (df.Date <= df_Train.Date.max()), 'ForecastId'] = df_Test.loc[(df_Test.Date >= df_Test.Date.min()) & (df_Test.Date <= df_Train.Date.max()), 'ForecastId'].values
df = getDaysShift(df)

df = getChangeGrowth(df)

df = getMA(df)

df = getCDSC(df)

df = getFDSC(df)
df['CSId'] = df.groupby(['Country', 'State']).cumcount()
df.loc[df.ForecastId > 0, 'ForecastId'].nunique()
df[df.Country =='Italy'][['ConfirmedCases', 'CD_1', 'CC_1']][70:95]
from warnings import filterwarnings

filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder

cLEncoder = LabelEncoder()

sLEncoder = LabelEncoder()



df.loc[:, 'Country'] = cLEncoder.fit_transform(df.loc[:, 'Country'])

df.loc[:, 'State'] = sLEncoder.fit_transform(df.loc[:, 'State'])
X_Train = df[df.Date <= df_Train.Date.max()]

X_Train.loc[:, 'Date_Int'] = (X_Train.loc[:, 'Date'].dt.strftime("%m%d")).astype('int16')



yC_Train = X_Train.CC_1

yF_Train = X_Train.FC_1

X_Train = X_Train.drop(columns=['Id', 'ForecastId', 'Date', 'ConfirmedCases', 'Fatalities', 'CC_1', 'FC_1'])

print(X_Train.shape, yC_Train.shape, yF_Train.shape)
X_Train.tail()
from lightgbm import LGBMRegressor



cLGBMR = LGBMRegressor(num_leaves=85,learning_rate=10**-1.89,n_estimators=100,min_sum_hessian_in_leaf=(10**-4.1),min_child_samples=2,subsample=0.97,subsample_freq=10,

                   colsample_bytree=0.68,reg_lambda=10**1.4,random_state=7,n_jobs=-1)

fLGBMR = LGBMRegressor(num_leaves=26,learning_rate=10**-1.63,n_estimators=100,min_sum_hessian_in_leaf=(10**-4.04),min_child_samples=14,subsample=0.66,subsample_freq=5,

                   colsample_bytree=0.8,reg_lambda=10**1.92,random_state=7,n_jobs=-1)
cLGBMR.fit(X_Train, yC_Train, categorical_feature=['Country','State'])

fLGBMR.fit(X_Train, yF_Train, categorical_feature=['Country','State'])
X_Test = df[df.Date >= df_Test.Date.min()]

X_Test.loc[:, 'Date_Int'] = (X_Test.loc[:, 'Date'].dt.strftime("%m%d")).astype('int16')



X_Test.drop(columns=['Id', 'ForecastId', 'Date', 'ConfirmedCases', 'Fatalities', 'CC_1', 'FC_1'], axis=1, inplace=True)
nIds = range(X_Train.CSId.max()+1, X_Test.CSId.max()+1)



yC_Pred = []

yF_Pred = []

for nId in nIds:

    df = getDaysShift(df)

    df = getChangeGrowth(df)

    df = getMA(df)

    df = getCDSC(df)

    df = getFDSC(df)

    df.loc[df.CSId == nId, 'CC_1'] = cLGBMR.predict(X_Test[X_Test.CSId == nId])

    df.loc[df.CSId == nId,'LConfirmedCases'] = df.loc[df.CSId == nId, 'CC_1'] + df.loc[df.CSId == nId, 'CD_1']

    df.loc[df.CSId == nId,'ConfirmedCases'] = np.exp(df.loc[df.CSId == nId,'LConfirmedCases']) - 1

    df.loc[df.CSId == nId, 'FC_1'] = fLGBMR.predict(X_Test[X_Test.CSId == nId])

    df.loc[df.CSId == nId,'LFatalities'] = df.loc[df.CSId == nId, 'FC_1'] + df.loc[df.CSId == nId, 'FD_1']

    df.loc[df.CSId == nId,'Fatalities'] = np.exp(df.loc[df.CSId == nId,'LFatalities']) - 1
df.loc[:, 'Country'] = cLEncoder.inverse_transform(df.loc[:, 'Country'])

df.loc[:, 'State'] = sLEncoder.inverse_transform(df.loc[:, 'State'])
df[df.Country =='Italy'][['ConfirmedCases', 'CC_1', 'Fatalities', 'FC_1', 'CSId']][75:99]
df_out = df.loc[df.ForecastId > 0, ['ForecastId', 'ConfirmedCases', 'Fatalities']]
df_out.info()
df_out.ForecastId = df_out.ForecastId.astype('int')
df_out.iloc[np.r_[1, 42, 45, 97, 143, 175, 267, 327, 350, 420, 450, 540, 590, 680, 730, 2880, 2900, 2960, 3000, 3050, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, -1], :]
df_out.to_csv('submission.csv', index=False)
import plotly.express as px



country = 'India'

df_plot = df.loc[(df.Country == country) & (df.Date > '2020-04-01'), ['Date', 'Country', 'State', 'ConfirmedCases', 'Fatalities']].groupby(['Date', 'Country', 'State']).max().reset_index().groupby(['Date', 'Country']).sum().sort_values(by='ConfirmedCases', ascending=False).reset_index()

df_plot.ConfirmedCases = round(df_plot.ConfirmedCases)

fig = px.bar(df_plot, x="Date", y="ConfirmedCases", color="ConfirmedCases")

fig.update_layout(title='Rise of Confirmed Cases in India', annotations=[dict(x=pd.to_datetime('today'), y=150, xref="x", yref="y", text="Today's Stats is here", showarrow=True, arrowhead=1, ax=-150, ay=-150)])

fig.show()