# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
forecasts = pd.read_csv('../input/forecasts.csv')

scores = pd.read_csv('../input/scores.csv')
print(len(forecasts), len(scores))

print("*"*20)

print(forecasts[:5])

print("*"*20)

print(scores[:5])
[item for item in forecasts.groupby('date')['country'].apply(list)]
2624/32.0
forecasts.groupby('date').count()#.sort_values(by='date')
forecasts[forecasts['date']=='16 June']
scores
print(forecasts[forecasts['date']=="12 June"][forecasts['country'].isin(['Brazil','Croatia'])])



print(scores[scores['match_date']=="12 June"])
scores.ix[[57,62]]
# games took place 12 June – 13 July 2014

unique_dates = list(map(lambda x:str(x)+" June",range(12,31)))+list(map(lambda x:str(x)+" July",range(1,14)))

for num in range(len(scores)): # loops through each match one by one

    match_data = scores.ix[num]

    date = match_data['match_date']

    time = match_data['match_time']

    team1 = match_data['home_code']

    team2 = match_data['away_code']

    

    # finding forecast data relevant to the 2 teams and which was as close as possible before the relevant game

    forecast_data1 = forecasts[forecasts['country_code'].isin([team1,team2])][forecasts['date']==date][forecasts['time']<time]

    forecast_data2 = forecast_data1[forecast_data1['time']==forecast_data1['time'].max()]

    date2 = date

    while len(forecast_data2)!=2:

        for num2 in range(len(unique_dates)):

            if unique_dates[num2]==date2:

                date2 = unique_dates[num2-1]

        forecast_data1 = forecasts[forecasts['country_code'].isin([team1,team2])][forecasts['date']==date2][forecasts['time']<time]

        forecast_data2 = forecast_data1[forecast_data1['time']==forecast_data1['time'].max()]

    print(forecast_data2)
forecasts[forecasts['date']=='9 July']
## To Be Continued ##