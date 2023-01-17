import pandas as pd
people = pd.read_csv("../input/people.csv")

people = people.rename(columns = {'date': 'start_date'})

people['start_date'] = pd.to_datetime(people['start_date'], format = '%Y-%m-%d')



activity = pd.read_csv("../input/act_train.csv")

activity = activity.rename(columns = {'date': 'activity_date'})

activity['activity_date'] = pd.to_datetime(activity['activity_date'], format = '%Y-%m-%d')
people = people[['people_id','start_date','group_1','char_38']]

activity = activity[['people_id','activity_date','activity_category']]
data = pd.merge(left = people,

                right = activity,

                on = 'people_id',

                how = 'inner')
#Max activity date - start date

#(Max activity date - min activity date)/(count - 1)

#Max activity date - min activity date

#Number of activity records

act_records = data.groupby(['people_id']).agg(['count'])

#Number distinct activity categories

#Columns for each activity category and their counts

#Number distinct group_1

#Average char_38



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