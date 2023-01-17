import pandas as pd
df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')

df
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df.corr().plot.bar()
plt.figure(figsize=(16,9)) # Figure size

sns.lineplot(x='date', y='cases', data=df, marker='o', color='purple') 

plt.title('Cases per day in the US') # Title

plt.xticks(df.date.unique(), rotation=90) # All values in the x axis rotate 90 degrees

plt.show()