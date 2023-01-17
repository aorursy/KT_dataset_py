import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
%time data = pd.read_csv('/kaggle/input/cristiano-rolando-stats/cr7.csv')

print(data.shape)
data.head()
# The find data function allows data to be searched within the dataframe through column and value.
def find_data(column, value):
    return data[data[column] == value]

find_data('Season', '17/18')
goals_by_season = data.groupby('Season').sum().reset_index()

plt.figure(figsize = (20, 10))
gbs = sns.barplot(data = goals_by_season, x = 'Season', y = 'Goals', palette='Reds')
gbs.set_title(label = 'Goals by Season', fontsize = 16)
gbs.set_xlabel(xlabel = 'Season', fontsize = 13)
gbs.set_ylabel(ylabel = 'Goals', fontsize = 13)
plt.show()
sns.catplot(x="Goals",y="Assists", hue="Season", data=goals_by_season, height=10)