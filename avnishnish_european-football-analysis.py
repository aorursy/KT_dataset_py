# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from sqlalchemy import create_engine
# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt
import seaborn as sns

import datetime as dt
from dateutil.relativedelta import relativedelta
engine = create_engine('sqlite:///../input/database.sqlite')
con = engine.connect()
rs = con.execute("SELECT * FROM Player_attributes")
player_att = pd.DataFrame(rs.fetchall())
player_att.columns = rs.keys()

rs2 = con.execute("SELECT * FROM Player")
player1 = pd.DataFrame(rs2.fetchall())
player1.columns = rs2.keys()

player = player1.merge(player_att, how="inner", on='player_api_id', copy=False) \
                .set_index('player_api_id') \
                .drop(['player_fifa_api_id_x','player_fifa_api_id_y','id_y'], axis=1)
player.head()
player['birthday'] = pd.to_datetime(player['birthday'])
today = dt.datetime.today()
def age(date):
        res = relativedelta(today, date).years
        return res
player['age'] = player['birthday'].apply(age)    
player[['player_name','age']].tail()    
player2 = player.drop_duplicates(subset=['player_name'])
corr_df = player2.drop(['id_x','player_name','birthday','date'], axis=1).corr()
plt.figure(figsize=(30,15))
sns.heatmap(corr_df, annot=True)
plt.show()
ovr = player2['overall_rating']
corr_ovr = player2.drop(['id_x','player_name','birthday','date','preferred_foot','attacking_work_rate','defensive_work_rate'], axis=1).corrwith(ovr)
plt.figure(figsize=(15,5))
corr_ovr.plot(kind='bar')
plt.xlabel("Attributes")
plt.ylabel('Correlation')
plt.title('Correlation of Overall Rating with other attributes')
plt.show()
ovr_sort = player2.sort_values('overall_rating', ascending=False).head(15)
ovr_sort[['player_name','overall_rating']]
rs3=con.execute("SELECT * FROM Team")
team = pd.DataFrame(rs3.fetchall())
team.columns = rs3.keys()
team.set_index('team_api_id', inplace=True)

rs4=con.execute("SELECT * FROM Team_Attributes")
team_attr = pd.DataFrame(rs4.fetchall())
team_attr.columns = rs4.keys()
team_attr.set_index('team_api_id', inplace=True)
team.head()
team_attr.head()

























