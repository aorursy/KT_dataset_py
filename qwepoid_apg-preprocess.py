import pandas as pd

import regex as re

import numpy as np
train = pd.read_csv('../input/foundation/foundation.csv')

test = pd.read_csv('../input/testfinaldata/testdata.csv')

train.head()
test.head()
train['year'] = -1

for i in range(train.shape[0]):

    train['year'][i] = '20' + train['Start Date'][i][7:]

train.head()
train['winbool'] = 0

for i in range(train.shape[0]):

    if train['Winner'][i] == train['Team'][i]:

        train['winbool'][i] = 1

train.head()
train['TeamA Home'] = 0

train['TeamB Home'] = 0

train['Match Country'] = 'qwerty'

train.head()
Australia = {'Melbourne', 'Hobart', 'Sydney', 'Adelaide', 'Brisbane','Perth','Canberra','Townsville'}

Bangladesh= {'Dhaka','Chattogram','Khulna','Fatullah','Sylhet'}

Cannada = {'King City (NW)','Toronto'}

England = {'The Oval','Bristol','Birmingham', 'Leeds', 'Lord\'s','Cardiff', 'Nottingham', 'Manchester','Chester-le-Street','Taunton','Southampton'}

HongKong = {'Mong Kok'}

India = {'Dehradun','Thiruvananthapuram','Mumbai (BS)','Guwahati','Greater Noida','Chennai','Kanpur','Jaipur','Pune','Dharamsala','Kochi','Ranchi','Rajkot','Indore', 'Ahmedabad','Cuttack','Visakhapatnam', 'Nagpur', 'Delhi', 'Bengaluru','Mohali','Mumbai','Kolkata','Hyderabad (Deccan)'}

Ireland = {'Belfast','Dublin','Dublin (Malahide)'}

Kenya = {'Mombasa'}

Malaysia= {'Kuala Lumpur'}

NewZealand = {'Wellington','Mount Maunganui','Lincoln','Nelson','Whangarei', 'Queenstown', 'Christchurch', 'Napier', 'Hamilton','Auckland','Dunedin'}

Netherlands = {'The Hague','Amstelveen'}

Pakistan = {'Lahore'}

PNG = {'Port Moresby'}

Scotland = {'Aberdeen','Edinburgh','Ayr'}

SouthAfrica= {'Cape Town','Benoni','St George\'s','Potchefstroom','Kimberley','Bloemfontein', 'Durban', 'Johannesburg', 'Port Elizabeth', 'Centurion','Paarl','East London'}

SriLanka ={'Galle','Dambulla','Colombo (SSC)', 'Hambantota', 'Colombo (RPS)', 'Pallekele'}

UAE ={'Dubai (DSC)', 'Sharjah', 'Abu Dhabi', 'ICCA Dubai'}

WestIndies ={'Gros Islet','Basseterre','Bridgetown','Providence','Port of Spain','North Sound','Kingston','Kingstown'}

Zimbabwe ={'Harare','Bulawayo'}
for i in range(train['Ground'].shape[0]):

    if train['Ground'][i] in Australia:

        train['Match Country'][i] = 'Australia'

    elif train['Ground'][i] in Bangladesh:

        train['Match Country'][i] = 'Bangladesh'

    elif train['Ground'][i] in Cannada:

        train['Match Country'][i] = 'Cannada'

    elif train['Ground'][i] in England:

        train['Match Country'][i] = 'England'

    elif train['Ground'][i] in HongKong:

        train['Match Country'][i] = 'Hong Kong'

    elif train['Ground'][i] in India:

        train['Match Country'][i] = 'India'

    elif train['Ground'][i] in Ireland:

        train['Match Country'][i] = 'Ireland'

    elif train['Ground'][i] in Kenya:

        train['Match Country'][i] = 'Kenya'

    elif train['Ground'][i] in Malaysia:

        train['Match Country'][i] = 'Malaysia'

    elif train['Ground'][i] in NewZealand:

        train['Match Country'][i] = 'New Zealand'

    elif train['Ground'][i] in Netherlands:

        train['Match Country'][i] = 'Netherlands'

    elif train['Ground'][i] in Pakistan:

        train['Match Country'][i] = 'Pakistan'

    elif train['Ground'][i] in PNG:

        train['Match Country'][i] = 'P.N.G.'

    elif train['Ground'][i] in Scotland:

        train['Match Country'][i] = 'Scotland'

    elif train['Ground'][i] in SouthAfrica:

        train['Match Country'][i] = 'South Africa'

    elif train['Ground'][i] in SriLanka:

        train['Match Country'][i] = 'Sri Lanka'

    elif train['Ground'][i] in UAE:

        train['Match Country'][i] = 'U.A.E.'

    elif train['Ground'][i] in WestIndies:

        train['Match Country'][i] = 'West Indies'

    elif train['Ground'][i] in Zimbabwe:

        train['Match Country'][i] = 'Zimbabwe'

train.head()
for i in range(train.shape[0]):

    if train['Match Country'][i] == train['Team'][i]:

        train['TeamA Home'][i] = 1

    elif train['Match Country'][i] == train['Opposition'][i]:

        train['TeamB Home'][i] = 1
train.head()
train.to_csv('preprocessedData.csv', index=False)