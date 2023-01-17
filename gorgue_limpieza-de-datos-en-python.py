# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import pandas as pd

pd.options.display.max_columns = 100

'''

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

'''





training = pd.read_csv('/kaggle/input/curso-ciencia-datos-ugr-6/traffic_fatality_tra.csv')

training.head()



# Any results you write to the current directory are saved as output.
sns.pairplot(training, hue="Fatality")
numeric_cols, cat_cols = [],[]

for col in training.columns:

    if training[col].dtype == "object":

        cat_cols.append(col)

    else:

        numeric_cols.append(col)

        

print(numeric_cols,cat_cols)
training[cat_cols].describe()
training[numeric_cols].describe()
training[numeric_cols].hist()

training['Gender'].value_counts(dropna=False)
training.Gender  = training.Gender.replace('Not Reported',"Unknown")

training.Gender  = training.Gender.replace('\\N',"Unknown")

training.Gender.value_counts(dropna=False)
sns.catplot(x="Gender",hue="Fatality",kind="count",data=training)
training.Alcohol_Results  = training.Alcohol_Results.replace('\\N',np.NaN)

training.Alcohol_Results = pd.to_numeric(training.Alcohol_Results)  #Convertir de categorico a numerico

sns.pairplot(training[["Alcohol_Results", "Fatality"]], hue="Fatality", height=7)

training.Alcohol_Results.isnull().sum()
sns.boxplot(x="Alcohol_Results",hue="Fatality",data=training)

training.loc[training["Alcohol_Results"] > 0.4 ].count()
training.Drug_Involvement  = training.Drug_Involvement.replace('Not Reported',"Unknown")

training.Drug_Involvement  = training.Drug_Involvement.replace('\\N',"Unknown")

training['Drug_Involvement'].value_counts(dropna=False)

sns.catplot(x="Drug_Involvement",hue="Fatality",kind="count",data=training,height=7)
training.Atmospheric_Condition.value_counts()



training.Atmospheric_Condition  = training.Atmospheric_Condition.replace('Not Reported',"Unknown")

training.Atmospheric_Condition  = training.Atmospheric_Condition.replace('\\N',"Unknown")



training.Atmospheric_Condition = training.Atmospheric_Condition.fillna("Unknown")

training.Atmospheric_Condition.value_counts(dropna=False)



at = sns.catplot(x="Atmospheric_Condition",hue="Fatality",kind="count",data=training,height=20);

at.set_xticklabels(rotation=30)
#training.Roadway.value_counts(dropna=False)

training.Roadway  = training.Roadway.replace('\\N',"Unknown")

training.Roadway = training.Roadway.fillna("Unknown")

training.Roadway.value_counts(dropna=False)

at = sns.catplot(x="Roadway",hue="Fatality",kind="count",data=training,height=10);

at.set_xticklabels(rotation=45)
training.weekday.value_counts(dropna=False)

#= training.Alcohol_Results.replace('\\N',np.NaN)

at = sns.catplot(x="weekday",hue="Fatality",kind="count",data=training,height=7);

training.Age.isnull().sum()

training.Age = training.Age.fillna(0)

training.Age.describe()





sns.pairplot(training[["Age", "Fatality"]], hue="Fatality", height=7)
training.loc[training.Age == 0].count()