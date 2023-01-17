# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
crime = pd.read_csv('../input/crime.csv',encoding='latin-1')

offence_codes = pd.read_csv('../input/offense_codes.csv',encoding='latin-1')

crime.head()
offence_codes.head()
crime.info()
crime.columns
crime.describe()
crime['YEAR'].hist(color='blue',bins=40,figsize=(8,4))
sns.set_style="whitegrid"
crime.MONTH.value_counts().plot(kind='bar')
crime.HOUR.value_counts().plot(kind='bar')
sns.distplot(crime['MONTH'].dropna(),kde=False,color='darkred',bins=30)


pal = sns.dark_palette("palegreen", as_cmap=True)

sns.jointplot(x='MONTH',y='YEAR',kind='hex',data=crime)
sns.heatmap(crime.isnull(),yticklabels=False,cbar=False,cmap='viridis')
crime.drop('Lat',axis=1,inplace=True)
crime.drop('Long',axis=1,inplace=True)
def impute_district(cols):

    DISTRICT=cols[0]

    REPORTING_AREA=cols[1]

    if pd.isnull(DISTRICT):

        if REPORTING_AREA==1:

            return 22

        elif REPORTING_AREA==2:

            return 23

        else:

            return 24

    else:

        return DISTRICT

    
crime['DISTRICT']=crime[['DISTRICT','REPORTING_AREA']].apply(impute_district,axis=1)
sns.heatmap(crime.isnull(),yticklabels=False,cbar=False,cmap='viridis')
def impute_shooting(cols):

    SHOOTING=cols[0]

    REPORTING_AREA=cols[1]

    if pd.isnull(SHOOTING):

        if REPORTING_AREA==1:

            return 31

        elif REPORTING_AREA==2:

            return 32

        else:

            return 33

    else:

        return SHOOTING

    
crime['SHOOTING']=crime[['SHOOTING','REPORTING_AREA']].apply(impute_shooting,axis=1)
sns.heatmap(crime.isnull(),yticklabels=False,cbar=False,cmap='viridis')
def impute_street(cols):

    STREET=cols[0]

    REPORTING_AREA=cols[1]

    if pd.isnull(STREET):

        if REPORTING_AREA==1:

            return 42

        elif REPORTING_AREA==2:

            return 43

        else:

            return 44

    else:

        return STREET

    
crime['STREET']=crime[['STREET','REPORTING_AREA']].apply(impute_street,axis=1)
sns.heatmap(crime.isnull(),yticklabels=False,cbar=False,cmap='viridis')
crime.info()
crime.head()
sns.countplot("HOUR", data =crime)
sns.countplot("DAY_OF_WEEK",data=crime)
sns.catplot(x="DISTRICT",hue="MONTH",col="YEAR",data=crime,kind="count")
sns.catplot(x="DISTRICT",hue="DAY_OF_WEEK",col="YEAR",data=crime,kind="count")
crime= pd.DataFrame({'Count':crime.groupby(["YEAR","OFFENSE_CODE_GROUP"]).size()}).reset_index().sort_values('Count',ascending = False).head(12)

crime

sns.barplot(x ="OFFENSE_CODE_GROUP",y= "Count",hue="YEAR", data=crime)