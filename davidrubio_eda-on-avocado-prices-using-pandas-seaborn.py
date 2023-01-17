import pandas as pd
import seaborn as sns
import matplotlib.style as style
style.use('ggplot')
import matplotlib.pyplot as plt
%matplotlib inline 

df = pd.read_csv('../input/avocado.csv')

df.sample(10)
df['Date'] = pd.to_datetime(df['Date'])
df.shape
df.describe()
df.isnull().sum()
#Distribution for Average Price for all Avocado types

df['AveragePrice'].plot(kind='hist', bins=30, color='green', figsize=(10,10), grid=True, title='Frequency Distribution for Avocado Prices')
plt.legend()
plt.xlabel('Average Price')
plt.ylabel('Frequency Count')
#Conventional type Average price distribution 

Conventional = df[df['type'] == 'conventional']

Conventional['AveragePrice'].plot(kind='hist', bins=30, color='green',figsize=(10,10), grid=True, title='Frquency Distribution for Conventional Avocado Prices')
plt.legend()
plt.xlabel('Average Price Conventional')
plt.ylabel('Frequency Count')
Organic = df[df['type']== 'organic']

Organic['AveragePrice'].plot(kind='hist', bins=30, color='green',figsize=(10,10), grid=True, title='Frquency Distribution for Organic Avocado')
plt.legend()
plt.xlabel('Average Price Organic')
plt.ylabel('Frequency Count')
Diffprice= Organic['AveragePrice'].mean() - Conventional['AveragePrice'].mean()

Diffprice

#Price broken down by year and region/state (Organic)
mask = df['type']=='organic'

g = sns.factorplot('AveragePrice','region', data= df[mask],hue='year',size=13,aspect=1,join=False,)
#Price broken down by year and region/state (Conventional)
mask = df['type']=='conventional'
g = sns.factorplot('AveragePrice','region', data= df[mask], hue='year', size=13, aspect=1, join=False,)