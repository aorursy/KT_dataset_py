import numpy as np 
import pandas as pd
import seaborn as sns
pd.options.display.max_columns = None
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
data = pd.read_csv('/kaggle/input/pesquisa-data-hackers-2019/datahackers-survey-2019-anonymous-responses.csv')
data.head(5)
profile = ProfileReport(data, correlations={"cramers": {"calculate": False}})

profile
data["('P16',_'salary_range')"].isna().sum(), (data["('P10',_'job_situation')"] != 'Empregado (CTL)').sum(), (data["('P19',_'is_data_science_professional')"] == 0).sum(), (data["('P3',_'living_in_brasil')"] == 0).sum()
data = data[data["('P16',_'salary_range')"].notna()]
data.shape
data = data[data["('P10',_'job_situation')"] == 'Empregado (CTL)']
data.shape
data = data[data["('P19',_'is_data_science_professional')"] == 1]
data.shape
data = data[data["('P3',_'living_in_brasil')"] == 1]
data.shape
data['salary'] = data["('P16',_'salary_range')"].map({'Menos de R$ 1.000/mês': 'Menos de R$ 1.000/mês', 
                                                        'de R$ 1.001/mês a R$ 2.000/mês': 'de R$ 1.001 à 2.000/mês', 
                                                        'de R$ 2.001/mês a R$ 3000/mês': 'de R$ 2.001 à 3.000/mês', 
                                                        'de R$ 3.001/mês a R$ 4.000/mês': 'de R$ 3.001 à 4.000/mês', 
                                                        'de R$ 4.001/mês a R$ 6.000/mês': 'de R$ 4.001 à 6.000/mês', 
                                                        'de R$ 6.001/mês a R$ 8.000/mês': 'de R$ 6.001 à 8.000/mês',
                                                        'de R$ 8.001/mês a R$ 12.000/mês': 'de R$ 8.001 à 12.000/mês',
                                                        'de R$ 12.001/mês a R$ 16.000/mês': 'de R$ 12.001 à 16.000/mês',
                                                        'de R$ 16.001/mês a R$ 20.000/mês': 'de R$ 16.001 à 20.000/mês',
                                                        'de R$ 20.001/mês a R$ 25.000/mês': 'de R$ 20.001 à 25.000/mês',
                                                        'Acima de R$ 25.001/mês': 'Acima de R$ 25.001/mês'})
import matplotlib.ticker as ticker

salary_order = ['Menos de R$ 1.000/mês', 'de R$ 1.001 à 2.000/mês', 'de R$ 2.001 à 3.000/mês', 'de R$ 3.001 à 4.000/mês',
                'de R$ 4.001 à 6.000/mês', 'de R$ 6.001 à 8.000/mês', 'de R$ 8.001 à 12.000/mês',
                'de R$ 12.001 à 16.000/mês', 'de R$ 16.001 à 20.000/mês', 'de R$ 20.001 à 25.000/mês',
                'Acima de R$ 25.001/mês']

ncount = len(data)

plt.figure(figsize=(15,10))
ax = sns.countplot(x="salary", data=data, order=salary_order)
plt.title('Salário dos Cientistas de Dados', size=14)
plt.xlabel('Variação dos Salários', size=12)

ax2 = ax.twinx()

ax2.yaxis.tick_left()
ax.yaxis.tick_right()

ax.yaxis.set_label_position('right')
ax2.yaxis.set_label_position('left')

ax.set_ylabel("Número de Entrevistados", size=12, labelpad=12)
ax2.set_ylabel('Frequência [%]', size=12)

ax.tick_params(axis='x', labelsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

for p in ax.patches:
    x = p.get_bbox().get_points()[:,0]
    y = p.get_bbox().get_points()[1,1]
    ax.annotate('{:.1f}%'.format(100. * y / ncount), (x.mean(), y), ha='center', va='bottom')

ax.yaxis.set_major_locator(ticker.LinearLocator(11))
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

ax2.set_ylim(0, 100)
ax.set_ylim(0, 200)

ax2.grid(None)