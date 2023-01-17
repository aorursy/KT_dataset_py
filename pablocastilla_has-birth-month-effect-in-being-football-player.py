import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib inline

footballers_dataset =  pd.read_csv('../input/complete-fifa-2017-player-dataset-global/FullData.csv', header=0)
general_dataset =  pd.read_csv('../input/births-in-us-1994-to-2003/births.csv', header=0)
world_cup_players_dataset =  pd.read_csv('../input/fifa-world-cup-2018-players/wc2018-players.csv', header=0)
footballers_birth_dates_months = footballers_dataset.apply(lambda x: datetime.strptime(x.Birth_Date, "%m/%d/%Y").month, axis=1) 
world_cup_footballers_birth_dates_months = world_cup_players_dataset.apply(lambda x: datetime.strptime(x['Birth Date'], "%d.%m.%Y").month, axis=1) 
general_by_month=general_dataset.groupby(['month'])['births'].sum()
general_by_month.plot.bar();
plt.title('FIFA Footballers birth month histogram', fontsize=18)
_ = plt.hist(footballers_birth_dates_months, 10, alpha=0.5, label='Month')
plt.title('World Cup Footballers birth month histogram', fontsize=18)
_ = plt.hist(world_cup_footballers_birth_dates_months, 10, alpha=0.5, label='Month')
spanish_footballers_birth_dates_months = footballers_dataset[footballers_dataset['Nationality']=='Spain'].apply(lambda x: datetime.strptime(x.Birth_Date, "%m/%d/%Y").month, axis=1) 
plt.title('FIFA Spanish Footballers birth month histogram', fontsize=18)
_ = plt.hist(spanish_footballers_birth_dates_months, 10, alpha=0.5, label='Month')
spanish_world_cup_footballers_birth_dates_months = world_cup_players_dataset[world_cup_players_dataset['Team']=='Spain'].apply(lambda x: datetime.strptime(x['Birth Date'], "%d.%m.%Y").month, axis=1) 
plt.title('World Cup Spanish Footballers birth month histogram', fontsize=18)
_ = plt.hist(spanish_world_cup_footballers_birth_dates_months, 10, alpha=0.5, label='Month')
