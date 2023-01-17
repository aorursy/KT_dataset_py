import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
pd.set_option('display.max_columns', 200)
#df = pd.read_csv('accidents-rio-niteroi-bridge.csv')

df = pd.read_csv('/kaggle/input/accidents-rio-niteroi-bridge/accidents-rio-niteroi-bridge.csv')

df.head()
df['data'] = df['data_inversa'] + ' ' + df['horario']
# unnecessary

df = df.drop(['data_inversa', 'horario', 'dia_semana', 'ano'], axis=1)
df['data'] = pd.to_datetime(df['data'], format='%Y-%m-%d %H')

df = df.reindex(df['data'].sort_values().index)

df.reset_index(drop=True, inplace=True)

df.head()
df['data'][9366] - df['data'][0]
df['data'].dt.day_name().value_counts()
per_day = df['data'].dt.day_name().value_counts().sort_values()

sns.barplot(per_day.index, per_day.values, palette=sns.cubehelix_palette(len(per_day.values)))

plt.xticks(rotation=60, ha='center')

plt.title('Days with more accidents on the rio-niterói bridge \n 2007-2020')



plt.ylabel('Days')

plt.show()
per_year = df['data'].dt.year.value_counts().sort_index()
per_year
plt.plot(per_year.index, per_year.values, color='green', linewidth=5)

plt.title('Accidents per year on the rio-niterói bridge')

plt.xticks()

plt.show()
print('Accident reduction on the rio-niterói bridge:')

print()

for v in range(2008, 2021):

    print(f'{v}: {round(100 - (per_year[v] * 100/per_year[v-1]))}%')
plt.figure(figsize=(10, 5))

by_m = df['data'].dt.month_name().value_counts()

new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

by_m = by_m.reindex(new_order, axis=0,)

by_m.plot(c='red', style='-o')



plt.xticks(rotation=60, ha='center')

plt.title('Number of accidents per month on the rio-niterói bridge \n 2007-2020')

plt.ylabel('Number of accidents')



plt.grid()

plt.show()
per_hour = df['data'].dt.hour.value_counts().sort_index().plot.bar()



plt.title('Most frequent hours of accidents on the rio-niterói bridge \n 2007-2020')

plt.ylabel('Number of accidents')

plt.xlabel('Hours')



plt.show()