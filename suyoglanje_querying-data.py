import pandas as pd
import numpy as np
state_data = pd.read_csv("../input/covid19-in-india/StatewiseTestingDetails.csv")
data = pd.read_csv("../input/covid19-in-india/covid_19_india.csv")
state_data.head()
data.head()
print(state_data.shape)

print(data.shape)
state_data.columns
data.columns
state_data.info()
data.info()
state_data.describe()
state_data.describe(include="object")
data.describe()
data.describe(include="object")
state_data['State'].value_counts()
state_data['State'].value_counts(normalize=True)
data["State/UnionTerritory"].value_counts()
data["State/UnionTerritory"].value_counts(normalize=True)
state_data.loc[0:10,'State':'Positive']
state_data.iloc[0:10,2:5]
state_data[state_data['TotalSamples'] == state_data['TotalSamples'].max()]['State']
data.sort_values(by='Date').head()
data.sort_values(by='Date').tail()
data.sort_values(by=['Date','Confirmed'],ascending=[True,True]).head()
data.groupby(by='State/UnionTerritory')['Confirmed'].describe()
data.groupby(by = 'State/UnionTerritory')['Confirmed'].agg(np.mean)