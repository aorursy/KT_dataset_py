# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_nba=pd.read_csv('../input/nba2k20-player-dataset/nba2k20-full.csv')
df_nba.info()
def from_date_to_age(date):
    born=datetime.datetime.strptime(date, '%m/%d/%y')
    today = datetime.date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
def int_weight(weight):
    return weight.split("/")[1].split(" ")[1]
def int_height(height):
    return height.split("/")[1].split(" ")[1]


###### Convertion of type ########

# convert height and weight to float values and rename 
df_nba["height"]=df_nba["height"].apply(lambda x: int_height(x)).astype("float")
df_nba["weight"]=df_nba["weight"].apply(lambda x: int_weight(x)).astype("float")
df_nba.rename({"height":"height_in_m","weight":"weight_in_kg"},axis='columns',inplace=True)

# convert salary to int values
df_nba["salary"] = df_nba["salary"].str[1:].astype("int64")

# convert draft round,peak to int and handle missing value
df_nba["draft_round"] = df_nba["draft_round"].replace({"Undrafted": 0}).astype("int8")
df_nba["draft_peak"] = df_nba["draft_peak"].replace({"Undrafted": 0}).astype("int8")

# 
df_nba["jersey"] = df_nba["jersey"].str[1:].astype('int8')


####### New columns ########


# Compute the body mass for all players
df_nba["body_mass_index"] = np.round(df_nba["weight_in_kg"] / ((df_nba["height_in_m"])**2),1)

# Associate a label to a certain amount of body mass
df_nba.loc[(df_nba["body_mass_index"]>=18.5) & (df_nba["body_mass_index"]<=24.9),"bmi_class"] = "Normal"
df_nba.loc[(df_nba["body_mass_index"]>=25) & (df_nba["body_mass_index"]<=29.9),"bmi_class"] = "Overweight"
df_nba.loc[df_nba["body_mass_index"]>=30,"bmi_class"] = "Obese"




# Indicate if a mba players attended college 
df_nba['college'].isna().astype('int').value_counts() 
df_nba["attended_college"] = df_nba['college'].isna().astype('int')

# Indicate their current age
df_nba["current_age"]=df_nba["b_day"].apply(lambda x: from_date_to_age(x))

# indicate the numbers of years played since they started nba
df_nba["year_played"] = df_nba["current_age"] - (df_nba["draft_year"] - pd.to_datetime(df_nba["b_day"]).dt.year)
df_nba.drop(columns=['b_day'])

df_nba["bmi_class"] = df_nba["bmi_class"].astype('category')
df_nba["bmi_class"] = df_nba["bmi_class"].cat.codes
df_nba["team"] = df_nba["team"].astype('category')
df_nba["team"] = df_nba["team"].cat.codes
from sklearn import preprocessing
from scipy import stats

data_dummy = pd.get_dummies(df_nba, columns=['team', 'position','draft_round', 'country', 'bmi_class'], drop_first= True)
data_dummy = data_dummy.drop(['full_name', 'draft_peak', 'b_day', 'college'], axis = 1)
X, y = data_dummy.drop(['salary'], axis = 1), data_dummy['salary']

print(data_dummy)
#normalizing input features
normalizer = preprocessing.Normalizer().fit(X)
X = normalizer.transform(X)