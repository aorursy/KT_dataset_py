import pandas as pd

import numpy as np
url="https://raw.githubusercontent.com/beoutbreakprepared/nCoV2019/master/latest_data/latestdata.csv"
df = pd.read_csv(url, index_col=0)
df.head()
df.info()
# checking unwanted columns

missing=(df.isnull().sum()/len(df))*100

missing=missing.sort_values(ascending=False)

print(missing)
to_drop=['chronic_disease',

'notes_for_discussion',

'date_death_or_discharge',

'chronic_disease_binary',

'outcome',

'sequence_available',

'reported_market_exposure',

'symptoms',

'date_admission_hospital',

'travel_history_dates',

'date_onset_symptoms',

'lives_in_Wuhan',

'travel_history_location',

'admin3',

'additional_information',

'location',

'age',

'sex',

'travel_history_binary',

'data_moderator_initials']
##  Dropping data that has more than 80% missing data



df.drop(to_drop, inplace=True, axis=1)

((df.isnull().sum()/len(df))*100).sort_values()
for x in df.columns:

    print("Column: "+ x)

    print("Number of each Values: ",df[x].nunique())

    print("Values: ",df[x].unique(),)

    print("Values count: ",df[x].value_counts(),"\n\n")

    