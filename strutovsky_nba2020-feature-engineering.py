import numpy as np 

import pandas as pd

from sklearn.preprocessing import LabelEncoder
raw_data = pd.read_csv('/kaggle/input/nba2k20-player-dataset/nba2k20-full.csv')

raw_data
raw_data.isna().sum()
cleaned_data = raw_data.copy()

cleaned_data['jersey'] = cleaned_data['jersey'].apply(lambda x: int(x[1:])) # delete '#' symbol

cleaned_data['team'] = cleaned_data['team'].fillna('no team')   # fill all n/a with 'no team' string

cleaned_data['height'] = cleaned_data['height'].apply(lambda x: float(x[2+x.find('/'):])) # convert to meters

cleaned_data['weight'] = cleaned_data['weight'].apply(lambda x: float(x[2+x.find('/'):-4])) # convert to kg

cleaned_data['salary'] = cleaned_data['salary'].apply(lambda x: int(x[1:])) # delete '#' symbol

cleaned_data['draft_round'] = cleaned_data['draft_round'].apply(lambda x: int(x) if x.isdigit() else 0)

cleaned_data['draft_peak'] = cleaned_data['draft_peak'].apply(lambda x: int(x) if x.isdigit() else 0)

cleaned_data['college'] = cleaned_data['college'].fillna('no college')

cleaned_data['experience_years'] = 2020 - cleaned_data['draft_year']

cleaned_data = cleaned_data.drop(['draft_year'], axis=1)



# change bday on age

cleaned_data['b_day'] = cleaned_data['b_day'].apply(lambda x: x[-2:])

cleaned_data['b_day'] = cleaned_data['b_day'].apply(lambda x: int('20'+x) if x[0] == '0' else int('19'+x))

cleaned_data['age'] = 2020 - cleaned_data['b_day']

cleaned_data = cleaned_data.drop(['b_day'], axis=1)
labelencoder = LabelEncoder()

cleaned_data['position_cat'] = labelencoder.fit_transform(cleaned_data['position'])



labelencoder = LabelEncoder()

cleaned_data['team_cat'] = labelencoder.fit_transform(cleaned_data['team'])



labelencoder = LabelEncoder()

cleaned_data['country_cat'] = labelencoder.fit_transform(cleaned_data['country'])



labelencoder = LabelEncoder()

cleaned_data['college_cat'] = labelencoder.fit_transform(cleaned_data['college'])
cleaned_data = cleaned_data.drop(['college', 'country', 'team', 'position', 'full_name'], axis=1)
cleaned_data
cleaned_data.info()