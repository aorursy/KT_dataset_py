import numpy as np

import pandas as pd
#data

df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

df_sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
def poly_fit(df, x_col, y_col):

    '''

    returns dict of fitted polynomials

    df - train data

    x_col - must be days (int)

    y_col - target

    '''

    

    x = df[x_col].values

    y = df[y_col].values

    

    #ugly workaround for no data

    if len(x) < 2:

        x = np.concatenate((x, x + 1))

        y = np.concatenate((y, y + 1))

        print(x)

        print(y)    

        

    p = {} #polynomials



    for i in range(3, 6):

        p['p' + str(i)] = np.poly1d(np.polyfit(x, y, i))

        

    return p
def predict_next_values(p, estimations):

    '''

    returns list of next values in the series

    p - dict of polynomials

    estimations - collection of days (int)

    '''

    

    means = []



    for i in estimations:

        mean = 0



        for k, v in p.items():

            pred = int(v(i))

            mean += pred



        mean = int(mean / len(p))

        means.append(mean)



    return means
df_submission = pd.DataFrame(index=df_test.index, columns=['ForecastId', 'ConfirmedCases', 'Fatalities'])

df_submission['ForecastId'] = df_test['ForecastId']



#for each country

for i, g in df_train.groupby('Country_Region'):

    #country specific

    df = g[g['ConfirmedCases'] > 0]

    df_tmp = df_test[df_test['Country_Region'] == i] #tmp test

    

    #if no cases so far

    if df.shape[0] == 0:

        print('No cases so far:', i)

        

        for i, r in df_tmp.iterrows():

            df_submission.loc[i, 'ConfirmedCases'] = 0

            df_submission.loc[i, 'Fatalities'] = 0

            

        continue

    else:

        print(i)



    #fit and predict

    df['Days'] = pd.to_datetime(df['Date']).map(lambda x: (x - pd.to_datetime(df['Date'].values[0])).days)

    df_tmp['Days'] = pd.to_datetime(df_tmp['Date']).map(lambda x: (x - pd.to_datetime(df['Date'].values[0])).days)



    #poly fit

    p_confirmed = poly_fit(df, 'Days', 'ConfirmedCases')

    p_fatal = poly_fit(df, 'Days', 'Fatalities')



    #assemble submission

    for i, r in df_tmp.iterrows():

        #exists in train

        d = r['Date']



        if d in df_train['Date'].values:

            df_submission.loc[i, 'ConfirmedCases'] = df_train.loc[df_train['Date'] == d, 'ConfirmedCases'].values[0]

            df_submission.loc[i, 'Fatalities'] = df_train.loc[df_train['Date'] == d, 'Fatalities'].values[0]



        #future

        df_submission.loc[i, 'ConfirmedCases'] = predict_next_values(p_confirmed, [r['Days']])[0]

        df_submission.loc[i, 'Fatalities'] = predict_next_values(p_fatal, [r['Days']])[0]    



#export

df_submission.to_csv('submission.csv', index=False)

df_submission.tail()    
df_test[df_test['Country_Region'] == 'Israel'].join(df_submission.add_prefix('Sub '))[['Country_Region', 'Date', 'Sub ConfirmedCases', 'Sub Fatalities']]