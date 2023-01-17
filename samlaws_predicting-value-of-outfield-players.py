import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
sns.set()
data = pd.read_csv('../input/fifacsv/fifa.csv', index_col='ID')
data = data.drop(['Unnamed: 0', 'Photo', 'Flag', 'Club Logo', 'Contract Valid Until', 'Joined', 'Real Face',
                  'Release Clause', 'GKDiving', 'GKHandling', 'GKKicking', 'GKReflexes', 'GKPositioning'], axis=1)
data.head()
data['Loaned From'] = data['Loaned From'].replace(np.nan, 'N/A')
data['Club'] = data['Club'].replace(np.nan, 'N/A')
data.dropna(axis=0, inplace=True)
data['Weight'] = pd.to_numeric(data['Weight'].str.split('lbs', expand=True)[0])/2.2
def feet_to_cm(col, df):
    '''
    A function that converts height from feet and inches in the form X'Y to cm
    '''
    df[col] = pd.to_numeric(df[col].str.split("'", expand=True)[0])*30.48 + pd.to_numeric(df[col].str.split("'", expand=True)[1])*2.54
feet_to_cm('Height', data)
def attr_adder(col_list, df):
    '''
    A function that adds the attributes in each position as when loaded in the data was in the form X+Y
    '''
    for i in col_list:
        df[i] = pd.to_numeric(df[i].str.split('+', expand=True)[0]) + pd.to_numeric(df[i].str.split('+', expand=True)[1])
cols = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB',
        'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
attr_adder(col_list=cols, df=data)
value_suffix = [data.Value.iloc[i][-1] for i in range(len(data.Value))]
wage_suffix = [data.Wage.iloc[i][-1] for i in range(len(data.Value))]
data['Value Factor'] = value_suffix
data['Wage Factor'] = wage_suffix
factors_dict = {'M':1000000,
               'K':1000,
               0:1}
data.replace({'Value Factor': factors_dict}, inplace=True)
data.replace({'Wage Factor': factors_dict}, inplace=True)
data['Value Factor'] = pd.to_numeric(data['Value Factor'])
data['Wage Factor'] = pd.to_numeric(data['Wage Factor'])
data['Value'] = data['Value'].str.translate(str.maketrans('', '', '€MK'))
data['Wage'] = data['Wage'].str.translate(str.maketrans('', '', '€MK'))
data['Value'] = pd.to_numeric(data['Value'])
data['Wage'] = pd.to_numeric(data['Wage'])
data['Value'] = data['Value'] * data['Value Factor']
data['Wage'] = data['Wage'] * data['Wage Factor']
data.drop(['Wage Factor', 'Value Factor'], axis=1, inplace=True)
temp = data.pop('Value')
data['Value'] = temp
categorical_cols = [x for x in data.columns if data[x].dtype == object]
data = shuffle(data)
X_train = data.iloc[:12000, 1:-1]
y_train = data.iloc[:12000, -1]

X_test = data.iloc[12000:, 1:-1]
y_test = data.iloc[12000:, -1]
oh = OneHotEncoder(handle_unknown='ignore')
X_train = oh.fit_transform(X_train)
X_test = oh.transform(X_test)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=101)
rf = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1)
rf.fit(X_train, y_train)
rf.score(X_val, y_val)
y_pred = rf.predict(X_test)
rf.score(X_test, y_test)
