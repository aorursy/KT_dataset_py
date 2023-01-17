import pandas as pd

import numpy as np



import warnings; warnings.filterwarnings('ignore')
df = pd.read_csv('../input/online_sex_work.csv', index_col=0)

df = df.iloc[: 28831, :]



df.head()
df.index = df.index.astype(int)

df['Number_of_Comments_in_public_forum'] = df['Number_of_Comments_in_public_forum'].str.replace(' ', '').astype(int)

df['Number_of_advertisments_posted'] = df['Number_of_advertisments_posted'].astype(int)

df['Number_of_offline_meetings_attended'] = df['Number_of_offline_meetings_attended'].astype(int)

df['Profile_pictures'] = df['Profile_pictures'].astype(int)

df['Friends_ID_list'] = df['Friends_ID_list'].astype(str)

df['Risk'] = df['Risk'].astype(str)



df.head()
def fill_gender_na(row):

    if row['Sexual_orientation'] == 'Homosexual':

        if row['Looking_for'] == 'Men':

            return 'male'

        elif row['Looking_for'] == 'Women':

            return 'female'

    elif row['Sexual_orientation'] == 'Heterosexual':

        if row['Looking_for'] == 'Men':

            return 'female'

        elif row['Looking_for'] == 'Women':

            return 'male'

    

    return np.nan



fill_values = df.apply(fill_gender_na, axis=1)

df['Gender'].fillna(fill_values, inplace=True)

df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)



df.insert(0, 'Female', df['Gender'] == 'female')

del df['Gender']



df.head()
df['Age'] = df['Age'].apply(lambda x: x.replace(',', '.'))

df['Age'] = df['Age'].replace('???', np.nan)

df['Age'] = df['Age'].astype(float)

df['Age'].fillna(df['Age'].mean(), inplace=True)

df['Age'].head()
df['Location'].fillna(df['Location'].mode()[0], inplace=True)
df['Verification'] = df['Verification'] != 'Non_Verified'

df['Verification'].head()
df = pd.concat([df.iloc[:, :4], pd.get_dummies(df['Sexual_orientation']), df.iloc[:, 5:]], axis=1)

df = pd.concat([df.iloc[:, :8], pd.get_dummies(df['Sexual_polarity']), df.iloc[:, 9:]], axis=1)

df = pd.concat([df.iloc[:, :11], pd.get_dummies(df['Looking_for']), df.iloc[:, 12:]], axis=1)

df.head()
df['Points_Rank'].value_counts().head(10)
df['Points_Rank'] = df['Points_Rank'].str.replace(' ', '')

df['Points_Rank'].replace(to_replace='a', value='0', inplace=True)

df['Points_Rank'] = df['Points_Rank'].astype(int)



df['Points_Rank'].head()
df['Last_login'] = df['Last_login'].apply(lambda x: x.split('_')[1]).astype(int)

df['Last_login'].head()
df['Member_since'].replace(to_replace='0,278159722', value=df['Member_since'].mode()[0], inplace=True)

df['Member_since'].replace(to_replace='dnes', value=np.nan, inplace=True)

df['Member_since'] = pd.to_datetime(df['Member_since'], format='%d.%m.%Y')



from datetime import timedelta



df['Member_since'] = df['Member_since'].fillna(df['Member_since'].max() + timedelta(days=1))

df['Member_since'].head()
membersince_df = pd.DataFrame()

membersince_df['Member_since_year'] = df['Member_since'].map(lambda x: x.year)

membersince_df['Member_since_month'] = df['Member_since'].map(lambda x: x.month)

membersince_df['Member_since_day'] = df['Member_since'].map(lambda x: x.day)



df = pd.concat([df.iloc[:, :18], membersince_df, df.iloc[:, 19:]], axis=1)

df.iloc[:5, 15:]
def get_n_minutes(row):

    time_components = row.split(':')

    if len(time_components) == 2:

        return int(time_components[0]) * 60 + int(time_components[1])

    elif len(time_components) == 3:

        return int(time_components[0]) * 1440 + int(time_components[1]) * 60 + int(time_components[2])



df['Time_spent_chating_H:M'] = df['Time_spent_chating_H:M'].str.replace(' ', '')

df['Time_spent_chating_H:M'] = df['Time_spent_chating_H:M'].apply(get_n_minutes)



df['Time_spent_chating_H:M'].head()
def get_n_friends(row):

    friend_ids = row.split(',')

    return len(friend_ids)



df.insert(25, 'Number of Friends', df['Friends_ID_list'].apply(get_n_friends))

df.iloc[:5, 20:]
def get_risk(row):

    if row == 'No_risk':

        return 0

    elif row == 'High_risk':

        return 1

    

    return np.nan



df['Risk'] = df['Risk'].apply(get_risk)

df['Risk'].head()