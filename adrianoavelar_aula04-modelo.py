import os

print(os.listdir('../input'))
import pandas as pd



base = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
#exiber toda a base

base.head()
base= base.iloc[:,:32]


base.columns

#total de colunas

len (base.columns)
#organizar em listas pelo hífen

a = list(base.columns)

for i in a:

    print('-',i)
#exibir coluna radius mean

base['radius_mean']
#média da coluna Radius mean

base['radius_mean'].mean()
#descrever a base

base.describe()
#descrever a linha O de diagnosis

base.describe ( include = ['O'] )
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#gráfico de texture_mean



sns.distplot(base ['texture_mean'])
#gráfico de diagnosis e radius mean dos dados malignos e benignos

sns.boxplot ( x = 'radius_mean', y = 'diagnosis', data = base)
#organizar em listas no boxplot de diagnosis

for i in  list (base.columns):

    if (i != 'diagnosis'):

        sns.boxplot( x =i, y='diagnosis', data=base )

        plt.show()
#exibir em barplot os valores maximos e mininos de radius mean

sns.barplot (y= base['radius_mean'], x= base ['diagnosis'])
#gráfico de disperção

sns.scatterplot (x= base ['area_mean'], y=base ['perimeter_mean'])
#criar uma nova base a parti dos dodos 

cor_base = base [ ['diagnosis', 'radius_mean', 'texture_mean', 'area_mean'] ]



cor_base
#

cor_base = base [ ['diagnosis', 'radius_mean', 'texture_mean', 'area_mean'] ]



cor_base.head(5)
#gráficos pareados

sns.pairplot (cor_base, hue= 'diagnosis')
base.isnull().sum()
base['diagnosis'].value_counts()
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

base['diagnosis'] = lb.fit_transform(base['diagnosis'])

base['diagnosis'].head()
base['diagnosis'].value_counts()
#desvio padrão

base.diagnosis.std()
#valor médio

base.diagnosis.mean()