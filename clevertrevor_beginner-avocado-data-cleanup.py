#imports
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

data = pd.read_csv('../input/avocado.csv')
data.head()
# rename columns: 4046, 4225, 4770
data = data.rename(index=str, columns={"4046" : "Small Hass", "4225" : "Large Hass", "4770" : "XLarge Hass"})
data.head()
data.isna().sum()
import datetime
def validate(date_text):
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Incorrect data format: ", date_text, ", should be YYYY-MM-DD")

for index, row in data.iterrows():
    validate(row.Date)
    
print ("No errors!")
data.AveragePrice.describe()
data.AveragePrice.hist()
data['Total Volume'].describe()
data['Total Volume'].hist()
data.nlargest(10, 'Total Volume')
data.region.unique()
# remove all rows where 'region' = 'TotalUS'
regionsToRemove = ['California', 'GreatLakes', 'Midsouth', 'NewYork', 'Northeast', 'SouthCarolina', 
                   'Plains', 'SouthCentral', 'Southeast', 'TotalUS', 'West']
size = data['Total Volume'].size
data = data[~data.region.isin(regionsToRemove)]
newsize = size - data['Total Volume'].size
print("old size: ", size, ", removed", newsize, "rows")
data['Total Volume'].hist()
data.nlargest(10, 'Total Volume')
sns.distplot(data['Small Hass'], color="yellow", label='Small Hass', kde=False)
sns.distplot(data['Large Hass'], color="orange", label='Large Hass', kde=False)
sns.distplot(data['XLarge Hass'], color="red", label='XLarge Hass', kde=False)
plt.legend()
data['Small Hass'].describe()
data.nlargest(10, 'Small Hass')
data['Large Hass'].describe()
data.nlargest(10, 'Large Hass')
data['XLarge Hass'].describe()
data.nlargest(10, 'Large Hass')
data['Total Bags'].describe()
data['Small Bags'].describe()
data['Large Bags'].describe()
data['XLarge Bags'].describe()
data.type.unique()
data.year.unique()