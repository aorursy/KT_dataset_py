# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import metrics

%matplotlib inline

from sklearn.model_selection import train_test_split 

from sklearn import linear_model

from sklearn.ensemble import RandomForestRegressor

from sklearn.datasets import make_regression

from sklearn.neural_network import MLPClassifier, MLPRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/epl-results-19932018/EPL_Set.csv')

data
data['Total_goals'] = data['FTHG'] + data['FTAG']



# Creating the columns for the data that is needed

data2 = pd.DataFrame([], columns = ['MatchPlayed','HomeTeam','AwayTeam', 'PFTHG','PFTAG','PFTG','PFTR','PHTHG','PHTAG','PHTG','PHTR'])



# Populating the HomeTeam

data2['HomeTeam'] = data['HomeTeam']





# Populating the AwayTeam

data2['AwayTeam'] = data['AwayTeam']

data2
for index in range(0, len(data)):

    home_team = data.iloc[index]['HomeTeam']

    away_team = data.iloc[index]['AwayTeam']



    # Getting all games with index where bother HomeTeam and AwayTeam has played together.

    i = data['HomeTeam'].isin([home_team, away_team]) & data['AwayTeam'].isin([home_team, away_team])

    

    # The index of the current team from the list above.

    current_filtered_index = int(data[i].index.get_loc(index))

    

    # The index of the previous team from the list above.

    previous_filtered_index = current_filtered_index - 1

    

    # Getting the previous team.

    if current_filtered_index != 0:

        filtered_data = data[i].iloc[previous_filtered_index]  

    else:

        filtered_data = data[i].iloc[current_filtered_index]

        filtered_data[:] = np.NaN



    # Populating the Previous Full Time Home Goals

    data2['PFTHG'].loc[index] = filtered_data['FTHG']

    

    # Populating the Previous Full Time Away Goals

    data2['PFTAG'].loc[index] = filtered_data['FTAG']

    

    # Populating the Previous Full Time Results

    data2['PFTR'].loc[index] = filtered_data['FTR']

    

    # Populating the Previous Half Time Home Goals

    data2['PHTHG'].loc[index] = filtered_data['HTHG']

    

    # Populating the Previous Half Time Away Goals

    data2['PHTAG'].loc[index] = filtered_data['HTAG']

    

    # Populating the Previous Half Time Results

    data2['PHTR'].loc[index] = filtered_data['HTR']



data2
# Populating the Previous Half Time Goal

data2['PHTG'] = [ (data2['PHTAG'][index] + data2['PHTHG'][index]) for index in range(0,len(data2)) ]





# Populating the Previous Full Time Goal

data2['PFTG'] = [ (data2['PFTAG'][index] + data2['PFTHG'][index]) for index in range(0,len(data2)) ]

data2
match_played = pd.Series([]) 



for index in range(0, len(data)):

    # Split season into list of the years in the season.

    formatted_season = data['Season'][index].split('-')

    

    # Keep the last 2 digits of the year

    formatted_season[0] = formatted_season[0][-2:] if len(formatted_season[0]) == 4 else formatted_season[0]

    

    # Converted from a list of strings to a list of integers.

    formatted_season = list(map(int, formatted_season))

    

    # Keep the last 2 digits of the year.

    formatted_date = int(data['Date'][index][-2:])

    

    # The smaller value of a list is the first match and the larger value of the list is the second match.

    if(formatted_date == min(formatted_season)):

        match_played[index] = 1

    elif(formatted_date == max(formatted_season)):

        match_played[index] = 2

    else:

        match_played[index] = np.NaN



# Number of Matches Played For A Season.

data2['MatchPlayed'] = match_played

data2
data2['HTHG'] = data['HTHG']

data2['HTAG'] = data['HTAG']

data2
data2 = data2.dropna()

data2
def create_label_encoder_dict(df):

    from sklearn.preprocessing import LabelEncoder

    

    label_encoder_dict = {}

    for column in df.columns:

        # Only create encoder for categorical data types

        if not np.issubdtype(df[column].dtype, np.number) and column != 'Age':

            label_encoder_dict[column]= LabelEncoder().fit(df[column])

    return label_encoder_dict
label_encoders = create_label_encoder_dict(data2)

print("Encoded Values for each Label")

print("="*32)

for column in label_encoders:

    print("="*32)

    print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))

    print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns=label_encoders[column].classes_, index=['Encoded Values']  ).T)
# Apply each encoder to the data set to obtain transformed values

epl = data2.copy() # create copy of initial data set

for column in epl.columns:

    if column in label_encoders:

        epl[column] = label_encoders[column].transform(epl[column])



print("Transformed data set")

print("="*32)

epl
epl.dtypes
epl.PFTG = epl.PFTG.astype(int)

epl.PHTG = epl.PHTG.astype(int)

epl.HTHG = epl.HTHG.astype(int)

epl.HTAG = epl.HTAG.astype(int)
epl.dtypes
epl['Total_goals']=data['Total_goals']

epl
epl.plot(kind='scatter',x='Total_goals',y='HomeTeam')
epl.plot(kind='scatter',x='Total_goals',y='AwayTeam')
epl.plot(kind='scatter',x='Total_goals',y='PFTHG')
epl.plot(kind='scatter',x='Total_goals',y='PFTAG')
epl.plot(kind='scatter',x='Total_goals',y='PFTG')
epl.plot(kind='scatter',x='Total_goals',y='PFTR')
epl.plot(kind='scatter',x='Total_goals',y='PHTHG')
epl.plot(kind='scatter',x='Total_goals',y='PHTAG')
epl.plot(kind='scatter',x='Total_goals',y='PHTG')
epl.plot(kind='scatter',x='Total_goals',y='PHTR')
epl.plot(kind='scatter',x='Total_goals',y='HTHG')
epl.plot(kind='scatter',x='Total_goals',y='HTAG')
X_data = epl[['HomeTeam','AwayTeam','HTHG','HTAG']]

Y_data = epl['Total_goals']
reg = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.001)

reg.fit(X_train,y_train)

print("Regression Coefficients")

pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])
predicted = reg.predict(X_test)

predicted = predicted.astype(int)

da = pd.DataFrame({'Actual': y_test, 'Predicted': predicted})

da
da.plot(kind='bar')
regn = MLPRegressor(hidden_layer_sizes=(100,100,100,100))

regn.fit(X_train,y_train)

test_predicted = reg.predict(X_test)

test_predicted = test_predicted.astype(int)

n = pd.DataFrame({'Actual': y_test, 'Predicted': test_predicted})

n
n.plot(kind='bar')
a=32 #home team

b=11 #Away team

c=3 #HTHG

d=2 #HTAG

q = {'home team':[a],'away team':[b],'HTHG':[c],'HTAG':[d]}

new=pd.DataFrame(q)

reg.predict(new)
a=32 #home team

b=11 #Away team

c=3 #HTHG

d=2 #HTAG

q = {'home team':[a],'away team':[b],'HTHG':[c],'HTAG':[d]}

new=pd.DataFrame(q)

regn.predict(new)