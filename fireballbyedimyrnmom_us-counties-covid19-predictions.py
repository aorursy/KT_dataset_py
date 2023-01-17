import pandas as pd



url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"

df = pd.read_csv(url)



print(df.head())
df.describe()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df.corr().style.background_gradient(cmap='coolwarm')
df.corr().plot.bar()
df.plot(kind='density', subplots=True, layout=(2,2), sharex=False)

plt.show()
sns.regplot(x='cases', y='deaths', data=df, logistic=False, color='green')
plt.figure(figsize=(16,9)) # Figure size

sns.lineplot(x='date', y='cases', data=df, marker='o', color='red') 

plt.title('Cases per day in the US') # Title

plt.xticks(df.date.unique(), rotation=90) # All values in x-axis; rotate 90 degrees

plt.show()