import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
filepath = '/kaggle/input/0014823-190918142434337.csv'



ebird_data = pd.read_csv(filepath, delimiter='\t')
data = ebird_data.loc[:, ['recordedBy', 'decimalLatitude', 'decimalLongitude', 'speciesKey', 'eventDate']]
registries_per_author = data.groupby('recordedBy').speciesKey.count()

registries_per_author.hist(log=True, bins=100)

plt.title("Distribuição dos registros por autor no eBird (Brasil)")

plt.ylabel("Número de autores")

plt.xlabel("Número de registros")
data.groupby('recordedBy').speciesKey.nunique().hist(log=True, bins=100)

plt.title("Distribuição dos número de espéscies por autor no eBird (Brasil)")

plt.ylabel("Número de autores")

plt.xlabel("Número de espécies")
sns.set_style('darkgrid')

x = data.decimalLongitude

y = data.decimalLatitude



plt.figure(figsize=(15, 15))

sns.jointplot(x, y, kind='hex', cmap='inferno', marginal_kws={'hist_kws': {'log': True}}, bins=200)

plt.show()
x = data.decimalLongitude

y = data.decimalLatitude



plt.figure(figsize=(15, 15))

plt.plot(x, y, '.', alpha=0.1)

plt.show()
data['eventDate']  = pd.to_datetime(data.eventDate)
filtered_data = data[data.eventDate > '2001-01-01']

plt.figure(figsize=(15, 5))

filtered_data.resample("3m", on='eventDate').recordedBy.count().plot(marker='x')
ebird_data.individualCount.hist(log=True, range=(0, 10000), bins=100)

plt.xlabel("Contagem de pássaros por registro do eBird")