import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

print("Setup Complete")
df = pd.read_csv('../input/suicide-rates-overview-1985-to-2016/master.csv')

df_ajust = df.groupby(['country', 'year'])[['suicides_no','population']].sum().reset_index(drop=False)



df_ajust['rate'] = (df_ajust.suicides_no / df_ajust.population) * 100000 #a cada 100 mil habitantes



df_ajust
plt.style.use('ggplot')



fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))



# gráfico no pyplot



ax[0].set_facecolor('#f5f5f5')



ax[0].plot(df_ajust.year[df_ajust.country == 'Brazil'],

           df_ajust.rate[df_ajust.country == 'Brazil'], label='Brasil')

ax[0].plot(df_ajust.year[df_ajust.country == 'Uruguay'],

           df_ajust.rate[df_ajust.country == 'Uruguay'], label='Uruguai')

ax[0].plot(df_ajust.year[df_ajust.country == 'United States'],

           df_ajust.rate[df_ajust.country == 'United States'], label='Estados Unidos')

ax[0].plot(df_ajust.year[df_ajust.country == 'Japan'],

           df_ajust.rate[df_ajust.country == 'Japan'], label='Japão')

ax[0].plot(df_ajust.year[df_ajust.country == 'Republic of Korea'],

           df_ajust.rate[df_ajust.country == 'Republic of Korea'], label='Coréia do Sul')



ax[0].set_title('Números de suicídios anuais por 100 mil habitantes', fontsize=15, pad=10, color='grey')



ax[0].set_xlabel('Ano', color='grey', fontsize=10)

ax[0].set_ylabel('Mortes (por 100 mil habitantes)', color='grey', fontsize=10)



ax[0].legend(facecolor='white', edgecolor='white')



# gráfico no seaborn



import seaborn as sns



ax[1].set_facecolor('#f5f5f5')



ax[1] = sns.lineplot(data=df_ajust[df_ajust.country.isin(['Brazil','Uruguay','United States',

                                                          'Japan','Republic of Korea'])],

                     x='year', y='rate', hue='country')



ax[1].set_title('Números de suicídios anuais por 100 mil habitantes', fontsize=15, pad=10, color='grey')



ax[1].set_xlabel('Ano', color='grey', fontsize=10)

ax[1].set_ylabel('Mortes (por 100 mil habitantes)', color='grey', fontsize=10)



ax[1].legend(facecolor='white', edgecolor='white')



plt.subplots_adjust(hspace=0.4) #espaço entre os gráficos



plt.show()