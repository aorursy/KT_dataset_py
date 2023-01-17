import numpy as np

import pandas as pd

import seaborn as sns
data = pd.read_csv('/kaggle/input/online-sex-work/online_sex_work.csv',low_memory=False, header = 0)
data = data.iloc[:28831,:]
data.shape
data.describe(include = 'all')
data.dtypes
data.sample(4)
data.isna().sum()
data.User_ID = data.User_ID.astype(int)
data.Gender.value_counts(dropna = False)
data[data['Gender'].isna()].index
data.drop(axis = 0,labels = [13065, 14236, 17589, 22927], inplace = True)
data.Age = data.Age.astype('str')
data.Age = data.Age.apply(lambda x: x.replace(',','.'))

data.Age = data.Age.replace('???',np.nan)
data.Age = data.Age.astype('float')
sns.distplot(a = data.Age.dropna())
data.Age.fillna(data.Age.mean(), inplace=True)

sns.distplot(a = data.Age)
data.Location.value_counts(dropna=False)
data.drop(columns = 'Location', inplace = True)
data.Verification.unique()
data[data['Verification'] == 'Administrator']
data.drop(index = 89, inplace = True)
Verified = data.Verification.apply(lambda x: 0 if x == 'Non_Verified' else 1)
data.insert(4, 'Verified',Verified)
data.sample(4)
data.drop('Verification', axis = 1, inplace = True)
data['Looking_for'].value_counts(dropna=False)
data[data['Sexual_orientation'] == 'bicurious']['Looking_for'].value_counts(dropna=False)
data[['Gender','Sexual_orientation','Looking_for']][:5]
def reseting_looking_for(row):

    if row['Sexual_orientation'] == 'Homosexual':

        if row['Gender'] == 'female':

            row['Looking_for'] = 'Women'

        if row['Gender'] == 'male':

            row['Looking_for'] = 'Men'

    if row['Sexual_orientation'] == 'Heterosexual':

        if row['Gender'] == 'female':

            row['Looking_for'] = 'Men'

        if row['Gender'] == 'male':

            row['Looking_for'] = 'Women'

    if row['Sexual_orientation'] == 'bicurious' or row['Sexual_orientation'] == 'bisexual':

        row['Looking_for'] = 'Men_and_Women'

    return row
data = data.apply(reseting_looking_for, axis = 1)
data['Sexual_orientation'].value_counts(dropna=False)
data['Looking_for'].value_counts(dropna=False)
data['Sexual_polarity'].value_counts(dropna = False)
data.head(3)
pd.get_dummies(data['Sexual_orientation']).head(3)
data = pd.concat([data.iloc[:,:4],pd.get_dummies(data['Sexual_orientation']),pd.get_dummies(data['Sexual_polarity']),pd.get_dummies(data['Looking_for']),data.iloc[:,7:]], axis = 1)
data.Points_Rank.replace(' ','',inplace = True, regex =True)
m = data.Points_Rank.mode()
data.Points_Rank.replace(to_replace='a', value='0', inplace=True)
data.Points_Rank = data.Points_Rank.astype(int)
import re
before = [re.findall("\d+", x)[0] for x in data.Last_login]   # split '_' can also b used
list(enumerate(data.columns))[10:]
data.insert(15,'Last_login_before_days',before)
data.drop(columns = 'Last_login', inplace = True)
max(data.Member_since)
data.Member_since = data.Member_since.replace('dnes',np.nan)

data.Member_since = data.Member_since.replace('0,278159722',np.nan)
data.Member_since = pd.to_datetime(data.Member_since, format = '%d.%m.%Y')
data.Member_since.fillna(max(data.Member_since), inplace = True)
list(enumerate(data.columns))[10:]
data.insert(16,'Member_since_year',data.Member_since.dt.year)

data.insert(17,'Member_since_month',data.Member_since.dt.month)

data.insert(18,'Member_since_day',data.Member_since.dt.day)
data.drop(columns = 'Member_since', inplace = True)
data.Number_of_Comments_in_public_forum.unique()
data.Number_of_Comments_in_public_forum.replace(' ','',inplace = True, regex =True)
data.Number_of_Comments_in_public_forum = data.Number_of_Comments_in_public_forum.astype(int)

data.Number_of_advertisments_posted = data.Number_of_advertisments_posted.astype(int)

data.Number_of_offline_meetings_attended = data.Number_of_offline_meetings_attended.astype(int)

data.Profile_pictures = data.Profile_pictures.astype(int)
data['Time_spent_chating_H:M'].replace(' ','', inplace = True, regex = True)
def to_minutes(cell):

    hrs_min = cell.split(':')

    hrs = int(hrs_min[0])

    mins = int(hrs_min[1])

    return hrs*60 + mins
data['Time_spent_chating_H:M'].head(4)
list(enumerate(data.columns))[-7:]
data.insert(20,'Minutes_spent_chatting',data['Time_spent_chating_H:M'].map(to_minutes))
data.drop(columns = 'Time_spent_chating_H:M',inplace = True)
data.Friends_ID_list.head(5)
data.Friends_ID_list.dtype
data.Friends_ID_list = data.Friends_ID_list.astype('str')
def no_of_friends(f):

    if f =='nan':

        return 0

    lst = f.split(',')

    return len(lst)
list(enumerate(data.columns))[-4:]
data.insert(24,'No_of_friends',data.Friends_ID_list.map(no_of_friends))
data.drop(columns = 'Friends_ID_list', inplace = True)
data.Risk.value_counts(dropna = False)
data.Risk.replace('unknown_risk',np.nan, inplace = True)