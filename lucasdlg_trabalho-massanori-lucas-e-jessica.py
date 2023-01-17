# Importando as ferramentas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#print('Python: {}'.format(sys.version))
#print('Numpy: {}'.format(numpy.__version__))
#print('Pandas: {}'.format(pandas.__version__))
#print('Matplotlib: {}'.format(matplotlib.__version__))
#print('Seaborn: {}'.format(seaborn.__version__))
#print('Scipy: {}'.format(scipy.__version__))
# Carregando informações em CSV utilizando o Pandas
data = pd.read_csv('../input/aac_shelter_outcomes.csv')
data.groupby('animal_type').size()
#Data set separado por animais
Dogs =(data[data['animal_type'] == 'Dog'])
Cats = (data[data['animal_type'] == 'Cat'])
Birds = (data[data['animal_type'] == 'Bird'])
Livestocks = (data[data['animal_type'] == 'Livestock'])
Others =(data[data['animal_type'] == 'Other'] )
plt.figure(figsize=(12,4))
sns.countplot(y=data['animal_type'], 
              order=data['animal_type'].value_counts().index)
plt.show()
plt.figure(figsize=(12,4))
sns.countplot(y=data['outcome_type'], 
              order=data['outcome_type'].value_counts().index)
plt.show()
plt.figure(figsize=(12,4))
sns.countplot(y=data['sex_upon_outcome'], 
              order=data['sex_upon_outcome'].value_counts().index)
plt.show()
plt.figure(figsize=(12,6))
sns.countplot(data=data,
              x='outcome_type',
              hue='animal_type')
plt.legend(loc='upper right')
plt.show()
plt.figure(figsize=(12,6))
sns.countplot(data=data,
              x='outcome_type',
              hue='sex_upon_outcome')
plt.legend(loc='upper right')
plt.show()
plt.figure(figsize=(12,6))
sns.countplot(data=data,
              x='sex_upon_outcome',
              hue='animal_type')
plt.legend(loc='upper right')
plt.show()
data['date_of_birth'] = pd.to_datetime(data['date_of_birth'], format='%Y-%m-%d')
data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d')

data['age'] = ((data['datetime'] - data['date_of_birth']).dt.days)
g = sns.FacetGrid(data, hue="animal_type", height=12)
g.map(sns.kdeplot, "age") 
g.add_legend()
g.set(xlim=(0,5000), xticks=range(0,5000,365))
plt.show(g)
Others.groupby('breed').size()