import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
df = pd.read_csv('/kaggle/input/pesquisa-data-hackers-2019/datahackers-survey-2019-anonymous-responses.csv')
df.head(3)
df.columns = [eval(col)[1] for col in df.columns]
df.columns
df.head(3)
df.living_state = [living_state.split('(')[1][:-1] if isinstance(living_state, str) else np.nan for living_state in df.living_state.tolist()]
df_location = df.living_state.value_counts().to_frame().reset_index().rename(columns={'index': 'state', 'living_state': 'count'})

df_location
coordinates = {

    'SP': (250, 90),

    'MG': (300, 120),

    'RJ': (310, 85),

    'PR': (225, 75),

    'SC': (245, 50),

    'RS': (210, 35),

    'ES': (330, 110),

}
df_location['x_coordinate'] = df_location['state'].map(lambda state: coordinates[state][0])

df_location['y_coordinate'] = df_location['state'].map(lambda state: coordinates[state][1])
df_location
def plot_state_dist(df: pd.DataFrame):

    plt.rcParams["figure.figsize"] = (15,15)



    img = plt.imread('/kaggle/input/images/brazil.jpg')



    fig, ax = plt.subplots()



    x = df['x_coordinate'].values

    y = df['y_coordinate'].values

    s = df['count'].values * 8



    ax.imshow(img, extent=[0, 400, 0, 300])

    ax.scatter(x, y, s=s, alpha=0.78, color='c')

    ax.set_title('Distribuição do Número de Respostas por Estado', size=18)

    

    plt.show()
plot_state_dist(df_location)
ax = df['age'].plot.hist(figsize=(12, 7))

ax.set_title('Distribuição por Idade', size=18)

ax.set_xlabel('Age', size=14)

ax.set_ylabel('', size=14)
ax = df['gender'].value_counts().plot.barh(figsize=(12, 4))

ax.set_title('Distribuição por Gênero', size=18)

ax.set_ylabel('Gênero', size=14)

ax.set_xlabel('', size=14)
df_country = df['living_in_brasil'].value_counts()

df_country.index = ['Vivendo no Brasil', 'Vivendo no Exterior']



ax = df_country.plot.barh(figsize=(12, 4))

ax.set_title('Distribuição por Localização de Trabalho Atual', size=18)

ax.set_ylabel('', size=14)

ax.set_xlabel('', size=14)