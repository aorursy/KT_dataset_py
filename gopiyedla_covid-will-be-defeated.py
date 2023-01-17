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



from fbprophet import Prophet

# Any results you write to the current directory are saved as output.
path = '/kaggle/input/covid19-global-forecasting-week-2/'

pd.set_option('display.max_rows', None)

train_df = pd.read_csv(path+'train.csv')

test_df = pd.read_csv(path+'test.csv')

submission_df = pd.read_csv(path+'submission.csv')
train_df.tail()
test_df.tail()
submission_df.head()
test_df['Date']= pd.to_datetime(test_df['Date']) 

train_df['Date']= pd.to_datetime(train_df['Date']) 
null_vals = pd.DataFrame(round ( 100 * (train_df.isnull().sum() / len(train_df.index)), 3))

null_vals.columns = ['% null values']

null_vals
null_vals = pd.DataFrame(round ( 100 * (test_df.isnull().sum() / len(test_df.index)), 3))

null_vals.columns = ['% null values']

null_vals
train_df['Province_State'].fillna('NO_STATE', inplace=True)

test_df['Province_State'].fillna('NO_STATE', inplace=True)
train_grouped_df = train_df.groupby(['Country_Region', 'Province_State'])
cluster0 = ['Botswana', 'Fiji', 'Gabon', 'Libya', 'Seychelles', 'Suriname', 'Mongolia', 'El Salvador', 'Barbados', 'Jamaica', 'Paraguay', 'Trinidad and Tobago','Montenegro', 'Georgia', 'Mauritius', 'Venezuela', 'Belarus', 'Malta', 'Oman', 'Albania', 'Cyprus', 'Kuwait', 'Jordan', 'Azerbaijan','Kazakhstan', 'Uruguay', 'Costa Rica', 'Bulgaria', 'Tunisia', 'Bosnia and Herzegovina', 'Latvia', 'Lebanon', 'Hungary', 'Armenia','Lithuania', 'Bahrain', 'Ukraine', 'Algeria', 'United Arab Emirates', 'Qatar', 'Estonia', 'Slovenia', 'Serbia', 'Colombia', 'Argentina','Dominican Republic', 'Peru', 'Mexico', 'Panama', 'Greece', 'South Africa', 'Indonesia', 'Saudi Arabia', 'Thailand', 'Ecuador', 'Poland','Romania', 'Malaysia', 'Brazil', 'Israel', 'Portugal', 'Turkey']

cluster0_confirmed_cap = 1500

cluster0_fatality_cap = 750



cluster1 = ['Italy', 'Spain']

cluster1_confirmed_cap = 50

cluster1_fatality_cap = 25



cluster2 = ['Iceland', 'Luxembourg', 'Switzerland']

cluster2_confirmed_cap = 500

cluster2_fatality_cap = 250



cluster3 = ['Angola', 'Bangladesh', 'Belize', 'Benin', 'Bhutan', 'Burkina Faso', 'Cambodia', 'Cameroon', 'Chad', 'Egypt', 'Ethiopia', 'Ghana', 'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Honduras', 'Iraq', 'Kenya', 'Liberia', 'Madagascar', 'Mali', 'Mauritania', 'Morocco', 'Mozambique', 'Namibia', 'Nepal','Nicaragua', 'Niger', 'Nigeria', 'Pakistan', 'Papua New Guinea', 'Philippines', 'Rwanda', 'Senegal', 'Sri Lanka', 'Sudan', 'Togo', 'Uganda','Zambia', 'Zimbabwe']

cluster3_confirmed_cap = 300

cluster3_fatality_cap = 150



cluster4 = ['China']

cluster4_confirmed_cap = 1.5

cluster4_fatality_cap = 1.35



cluster5 = ['Hong Kong', 'Singapore']

cluster5_confirmed_cap = 40

cluster5_fatality_cap = 28



cluster6 = ['United States']

cluster6_confirmed_cap = 200

cluster6_fatality_cap = 140



cluster6 = ['United States']

cluster6_confirmed_cap = 500

cluster6_fatality_cap = 350



cluster7 = ['Australia', 'Austria', 'Belgium', 'Canada', 'Chile', 'Denmark', 'Finland', 'Ireland', 'Japan', 'Netherlands', 'New Zealand', 'Norway', 'South Korea','Sweden', 'United Kingdom']

cluster7_confirmed_cap = 100

cluster7_fatality_cap = 70



cluster8 = ['France', 'Germany', 'Iran']

cluster8_confirmed_cap = 50

cluster8_fatality_cap = 35



cluster9 = ['India']

cluster9_confirmed_cap = 30

cluster9_fatality_cap = 21





predictions_dataset =  pd.DataFrame([])



for key, state_df in train_grouped_df:

    print('Key:', key)

    #create the dataset and do the prediction

    train_start_date = state_df['Date'].iloc[0]

    state_df['days_elapsed'] = (state_df['Date'] - train_start_date).dt.days

    train_state_df = state_df.drop(['Id' , 'Country_Region', 'Province_State'], axis = 1)

    #print(train_state_df.shape)



    y_train_case_count =  train_state_df['ConfirmedCases']

    y_train_fatalities =  train_state_df['Fatalities']



    x_train_state_case_df = train_state_df.drop(['Fatalities'], axis = 1)

    x_train_state_fatalities_df = train_state_df.drop('ConfirmedCases', axis = 1)



    test_state_df = pd.DataFrame([])

    test_state_df = test_df[ (test_df['Country_Region'] == key[0]) & (test_df['Province_State'] == key[1])]

    #test_state_df_copy = test_state_df.copy()

    test_state_df['days_elapsed'] = (test_state_df['Date'] - train_start_date).dt.days

    #print('test_state_df----', test_state_df)

    x_pred_state_df = test_state_df.drop(['ForecastId' ,'Province_State', 'Country_Region'], axis = 1)

    

    

    prophet_conf = Prophet(growth='logistic')

    prophet_fatalities = Prophet(growth='logistic')

    train_dataset =  pd.DataFrame()

    test_dataset =  pd.DataFrame()

    train_dataset['y'] = x_train_state_case_df['ConfirmedCases']

    train_dataset['ds'] = x_train_state_case_df['Date'] 

    #train_dataset['cap'] = 2.0

    #test_dataset['cap'] = 2.0

    test_dataset['ds'] = test_state_df['Date']

    

    train_current_fatalities = train_state_df['Fatalities'].max()

    train_current_cases = train_state_df['ConfirmedCases'].max()

    train_min_fatalities = train_state_df['Fatalities'].min()

    train_min_cases = train_state_df['ConfirmedCases'].min()



    train_dataset['cap'] =  train_current_cases * 1000

    test_dataset['cap'] =  train_current_cases * 1000



    

    train_dataset['floor'] =  0

    test_dataset['floor'] =  0

    

    

    #print(test_dataset)

    #test_dataset['']

    

    train_cases_cap = train_current_cases

    if train_current_fatalities == 0 :

        train_current_fatalities = 1

    if train_current_cases == 0 :

        train_current_cases = 1



    test_dataset['cap'] =  train_current_cases * 1000

    

    if train_current_cases > 50000 :

        train_dataset['cap'] =  train_current_cases * 50

        test_dataset['cap'] =  train_current_cases * 50



    if (train_current_cases > 20000 and train_current_cases <= 50000):

        train_dataset['cap'] =  train_current_cases * 100

        test_dataset['cap'] =  train_current_cases * 100



    if (train_current_cases > 10000 and train_current_cases <= 20000):

        train_dataset['cap'] =  train_current_cases * 200

        test_dataset['cap'] =  train_current_cases * 200



    if (train_current_cases > 5000 and train_current_cases <= 10000):

        train_dataset['cap'] =  train_current_cases * 400

        test_dataset['cap'] =  train_current_cases * 400



    if (train_current_cases > 1000 and train_current_cases <= 5000):

        train_dataset['cap'] =  train_current_cases * 600

        test_dataset['cap'] =  train_current_cases * 600



    if (train_current_cases > 100 and train_current_cases <= 1000):

        train_dataset['cap'] =  train_current_cases * 800

        test_dataset['cap'] =  train_current_cases * 800



    if (train_current_cases > 0 and train_current_cases <= 100):

        train_dataset['cap'] =  train_current_cases * 1000

        test_dataset['cap'] =  train_current_cases * 1000

        



    if key[0] in cluster0 :

        train_dataset['cap'] =  train_current_cases * cluster0_fatality_cap

        test_dataset['cap'] =  train_current_cases * cluster0_fatality_cap



    if key[0] in cluster1 :

        train_dataset['cap'] =  train_current_cases * cluster1_fatality_cap

        test_dataset['cap'] =  train_current_cases * cluster1_fatality_cap



    if key[0] in cluster2 :

        train_dataset['cap'] =  train_current_cases * cluster2_fatality_cap

        test_dataset['cap'] =  train_current_cases * cluster2_fatality_cap

        

    if key[0] in cluster3 :

        train_dataset['cap'] =  train_current_cases * cluster3_fatality_cap

        test_dataset['cap'] =  train_current_cases * cluster3_fatality_cap



    if key[0] in cluster4 :

        train_dataset['cap'] =  train_current_cases * cluster4_fatality_cap

        test_dataset['cap'] =  train_current_cases * cluster4_fatality_cap



    if key[0] in cluster5 :

        train_dataset['cap'] =  train_current_cases * cluster5_fatality_cap

        test_dataset['cap'] =  train_current_cases * cluster5_fatality_cap



#    if key[0] in cluster6 :

#        train_dataset['cap'] =  train_current_fatalities * cluster6_fatality_cap

#        test_dataset['cap'] =  train_current_fatalities * cluster6_fatality_cap



    if key[0] in cluster7 :

        train_dataset['cap'] =  train_current_cases * cluster7_fatality_cap

        test_dataset['cap'] =  train_current_cases * cluster7_fatality_cap



    if key[0] in cluster8 :

        train_dataset['cap'] =  train_current_cases * cluster8_fatality_cap

        test_dataset['cap'] =  train_current_cases * cluster8_fatality_cap



    if key[0] in cluster9 :

        train_dataset['cap'] =  train_current_cases * cluster9_fatality_cap

        test_dataset['cap'] =  train_current_cases * cluster9_fatality_cap

        

        

    prophet_conf.fit(train_dataset)

        

        



    y_case_predict = prophet_conf.predict(test_dataset)

    #merged_test_state_df = pd.concat([test_state_df, y_case_predict], axis = 1)

    #test_state_df['ConfirmedCases'] = y_case_predict

    y_case_predict.rename(columns = {'trend':'ConfirmedCases'}, inplace = True) 

    test_state_df = pd.merge(test_state_df,  y_case_predict[['ConfirmedCases', 'ds']], left_on='Date', right_on = 'ds', how='left'  )



    #test_state_df ['ConfirmedCases'] =   y_case_predict['yhat_upper']

    

    train_dataset['y'] = train_state_df['Fatalities']

    test_dataset['cap'] =  train_current_fatalities * 1000



    test_dataset['floor'] =  0

    

    train_dataset['cap'] =  train_current_fatalities * 1000

    train_dataset['floor'] =  0



    if train_current_cases > 50000 :

        train_dataset['cap'] =  train_current_fatalities * 50

        test_dataset['cap'] =  train_current_fatalities * 50



    if (train_current_cases > 20000 and train_current_cases <= 50000):

        train_dataset['cap'] =  train_current_fatalities * 70

        test_dataset['cap'] =  train_current_fatalities * 70



    if (train_current_cases > 10000 and train_current_cases <= 20000):

        train_dataset['cap'] =  train_current_fatalities * 140

        test_dataset['cap'] =  train_current_fatalities * 140



    if (train_current_cases > 5000 and train_current_cases <= 10000):

        train_dataset['cap'] =  train_current_fatalities * 280

        test_dataset['cap'] =  train_current_fatalities * 280



    if (train_current_cases > 1000 and train_current_cases <= 5000):

        train_dataset['cap'] =  train_current_fatalities * 420

        test_dataset['cap'] =  train_current_fatalities * 420



    if (train_current_cases > 100 and train_current_cases <= 1000):

        train_dataset['cap'] =  train_current_fatalities * 560

        test_dataset['cap'] =  train_current_fatalities * 560



        

    if (train_current_cases > 0 and train_current_cases <= 100):

        train_dataset['cap'] =  train_current_fatalities * 700

        test_dataset['cap'] =  train_current_fatalities * 700

        

    if key[0] in cluster0 :

        train_dataset['cap'] =  train_current_fatalities * cluster0_fatality_cap

        test_dataset['cap'] =  train_current_fatalities * cluster0_fatality_cap

        #print('0------', cluster0_fatality_cap)



    if key[0] in cluster1 :

        train_dataset['cap'] =  train_current_fatalities * cluster1_fatality_cap

        test_dataset['cap'] =  train_current_fatalities * cluster1_fatality_cap

        #print('1------', cluster1_fatality_cap)



    if key[0] in cluster2 :

        train_dataset['cap'] =  train_current_fatalities * cluster2_fatality_cap

        test_dataset['cap'] =  train_current_fatalities * cluster2_fatality_cap

        #print('2------', cluster2_fatality_cap)

        

    if key[0] in cluster3 :

        train_dataset['cap'] =  train_current_fatalities * cluster3_fatality_cap

        test_dataset['cap'] =  train_current_fatalities * cluster3_fatality_cap

        #print('3------', cluster3_fatality_cap)



    if key[0] in cluster4 :

        train_dataset['cap'] =  train_current_fatalities * cluster4_fatality_cap

        test_dataset['cap'] =  train_current_fatalities * cluster4_fatality_cap

        #print('4------', cluster4_fatality_cap)



    if key[0] in cluster5 :

        train_dataset['cap'] =  train_current_fatalities * cluster5_fatality_cap

        test_dataset['cap'] =  train_current_fatalities * cluster5_fatality_cap



#    if key[0] in cluster6 :

#        train_dataset['cap'] =  train_current_fatalities * cluster6_fatality_cap

#        test_dataset['cap'] =  train_current_fatalities * cluster6_fatality_cap



    if key[0] in cluster7 :

        train_dataset['cap'] =  train_current_fatalities * cluster7_fatality_cap

        test_dataset['cap'] =  train_current_fatalities * cluster7_fatality_cap



    if key[0] in cluster8 :

        train_dataset['cap'] =  train_current_fatalities * cluster8_fatality_cap

        test_dataset['cap'] =  train_current_fatalities * cluster8_fatality_cap



    if key[0] in cluster9 :

        train_dataset['cap'] =  train_current_fatalities * cluster9_fatality_cap

        test_dataset['cap'] =  train_current_fatalities * cluster9_fatality_cap



    

    prophet_fatalities.fit(train_dataset)

    y_fatalities_predict = prophet_fatalities.predict(test_dataset)

    #y_fatalities_predict.info()

#    test_state_df ['Fatalities'] =  y_fatalities_predict['yhat_upper']

    y_fatalities_predict.rename(columns = {'trend':'Fatalities'}, inplace = True) 

    test_state_df = pd.merge(test_state_df,  y_fatalities_predict[['Fatalities', 'ds']], left_on='Date', right_on = 'ds', how='left'  )

    

    predictions_dataset = predictions_dataset.append(test_state_df, ignore_index=True)

    

    #y_predict['Date'] = y_case_predict['ds']

    #y_predict['Country/Region'] = key[0]  

    #y_predict['Province/State'] = key[1]

    #test_df = pd.merge(test_df, y_predict, on=['Country/Region', 'Province/State', 'Date'])

    

    #print('------------------------------------------------------', key)

    #print( test_state_df)

    #print('------------------------------------------------------')

predictions_dataset.head()
submissions = predictions_dataset.drop(['Country_Region', 'Province_State', 'Date', 'days_elapsed', 'ds_x', 'ds_y'], axis = 1)



submissions['ConfirmedCases']  = submissions['ConfirmedCases'].astype(int)

submissions['Fatalities']  = submissions['Fatalities'].astype(int)
submissions.to_csv ('submission.csv', index=False)

print('--Done--')