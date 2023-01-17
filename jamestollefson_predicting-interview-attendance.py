# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Interview.csv')

data = data.drop(['Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27'], axis=1)

data.info()
from collections import defaultdict

column_name_dict = defaultdict(str)

for column in data.columns:

    column_name_dict[column] = ''

new_names = ['Date', 'Client', 'Industry', 'Location', 'Position', 'Skillset', 'Interview_Type', 'ID', 'Gender', 'Candidate_Loc', 'Job_Location', 'Venue', 'Native_Loc', 'Permission', 'Hope', '3_hour_call', 'Alt_Number', 'Resume_Printout', 'Clarify_Venue', 'Share_Letter', 'Expected', 'Attendance', 'Marital_Status']

for idx,key in enumerate(column_name_dict):

    column_name_dict[key] = new_names[idx]



data = data.rename(columns=column_name_dict)

data.info()
data.tail(1)

data = data.drop(data.index[1233])

data.tail()
sns.countplot(data.Hope)

plt.show()

sns.countplot(data.Permission)

plt.show()

sns.countplot(data['3_hour_call'])

plt.show()

sns.countplot(data.Alt_Number)

plt.show()

sns.countplot(data.Resume_Printout)

plt.show()

sns.countplot(data.Clarify_Venue)

plt.show()

sns.countplot(data.Share_Letter)

plt.show()

sns.countplot(data.Expected)

plt.show()
data.Hope = data.Hope.fillna('unsure') #

for i,v in enumerate(data.Hope):

    value = v.lower()

    if value == 'na':

        data.Hope.iloc[i] = 'unsure'

    elif value == 'not sure':

        data.Hope.iloc[i] = 'unsure'

    elif value == 'cant say':

        data.Hope.iloc[i] = 'unsure'

    else:

        data.Hope.iloc[i] = value

        

data.Permission = data.Permission.fillna('no')

for i,v in enumerate(data.Permission):

    value = v.lower()

    if value == 'not yet':

        data.Permission.iloc[i] = 'no'

    elif value == 'yet to confirm':

        data.Permission.iloc[i] = 'no'

    elif value == 'na':

        data.Permission.iloc[i] = 'no'

    else:

        data.Permission.iloc[i] = value



data['3_hour_call'] = data['3_hour_call'].fillna('na')

for i,v in enumerate(data['3_hour_call']):

    value = v.lower()

    if value == 'no dont':

        data['3_hour_call'].iloc[i] = 'no'

    else:

        data['3_hour_call'].iloc[i] = value

        

data.Alt_Number = data.Alt_Number.fillna('no')

for i,v in enumerate(data.Alt_Number):

    value = v.lower()

    if value == 'na':

        data.Alt_Number.iloc[i] = 'no'

    elif value == 'no i have only thi number':

        data.Alt_Number.iloc[i] = 'no'

    else:

        data.Alt_Number.iloc[i] = value

        

data.Resume_Printout = data.Resume_Printout.fillna('na')

for i,v in enumerate(data.Resume_Printout):

    value = v.lower()

    if value == 'no- will take it soon':

        data.Resume_Printout.iloc[i] = 'yes'

    elif value == 'not yet':

        data.Resume_Printout.iloc[i] = 'no'

    else:

        data.Resume_Printout.iloc[i] = value

        

data.Clarify_Venue = data.Clarify_Venue.fillna('na')

for i,v in enumerate(data.Clarify_Venue):

    value = v.lower()

    if value == 'no- i need to check':

        data.Clarify_Venue.iloc[i] = 'no'

    else:

        data.Clarify_Venue.iloc[i] = value



data.Share_Letter = data.Share_Letter.fillna('na')

for i,v in enumerate(data.Share_Letter):

    value = v.lower()

    if value == 'havent checked':

        data.Share_Letter.iloc[i] = 'no'

    elif value == 'need to check':

        data.Share_Letter.iloc[i] = 'no'

    elif value == 'not sure':

        data.Share_Letter.iloc[i] = 'no'

    elif value == 'yet to check':

        data.Share_Letter.iloc[i] = 'no'

    elif value == 'not yet':

        data.Share_Letter.iloc[i] = 'no'

    else:

        data.Share_Letter.iloc[i] = value

                

data.Expected = data.Expected.fillna('uncertain')

for i,v in enumerate(data.Expected):

    value = v.lower()

    if value == '11:00 am':

        data.Expected.iloc[i] = 'yes'

    elif value == '10.30 am':

        data.Expected.iloc[i] = 'yes'

    else:

        data.Expected.iloc[i] = value

    

for column in data.columns:

    print(data[column].unique())
import Levenshtein as lv

clients = data.Client.unique()

of_interest = []

for i,v in enumerate(data.Client):

    for client in clients:

        distance = lv.distance(client, v)

        if distance in range(3,9,1):

            dyad = (client, v)

            if dyad not in of_interest:

                of_interest.append(dyad)

                

for t in of_interest:

    alt = (t[1], t[0])

    if alt in of_interest:

        of_interest.remove(alt)

                

print(of_interest)
for i,v in enumerate(data.Client):

    if v == 'Standard Chartered Bank Chennai':

        data.Client.iloc[i] = 'Standard Chartered Bank'

    elif v == 'Aon Hewitt':

        data.Client.iloc[i] = 'Hewitt'

    elif v == 'Aon hewitt Gurgaon':

        data.Client.iloc[i] = 'Hewitt'

        

for i,v in enumerate(data.Industry):

    if 'IT' in v:

        data.Industry.iloc[i] = 'IT'



for i,v in enumerate(data.Location):

    value = v.lower()

    if 'chennai' in value:

        data.Location.iloc[i] = 'Chennai'

    elif 'gurgaon' in value:

        data.Location.iloc[i] = 'Gurgaon'

 

for i,v in enumerate(data.Interview_Type):

    value = v.lower()

    if value == 'scheduled walkin':

        data.Interview_Type.iloc[i] = 'scheduled walk in'

    elif value == 'walkin ':

        data.Interview_Type.iloc[i] = 'walkin'

    elif value == 'sceduled walkin':

        data.Interview_Type.iloc[i] = 'scheduled walk in'

    else:

        data.Interview_Type.iloc[i] = value



for i,v in enumerate(data.Candidate_Loc):

    value = v.lower()

    if 'chennai' in value:

        data.Candidate_Loc.iloc[i] = value

    else:

        data.Candidate_Loc.iloc[i] = value



for i,v in enumerate(data.Native_Loc):

    value = v.lower()

    if 'cochin' in value:

        data.Native_Loc.iloc[i] = 'cochin'

    elif 'delhi' in value:

        data.Native_Loc.iloc[i] = 'delhi'

    else:

        data.Native_Loc.iloc[i] = value
for i,v in enumerate(data.Skillset):

    data.Skillset.iloc[i] = v.lower()





skills = data.Skillset.unique()

of_interest = []

for i,v in enumerate(data.Skillset):

    for skill in skills:

        distance = lv.distance(skill, v)

        if distance in range(3,4,1):

            dyad = (skill, v)

            if dyad not in of_interest:

                of_interest.append(dyad)

                

for t in of_interest:

    alt = (t[1], t[0])

    if alt in of_interest:

        of_interest.remove(alt)

                

print(of_interest)