# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#criação do Dataframe 
df_fifa = pd.read_csv('/kaggle/input/fifa19/data.csv')
columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]
df_fifa.drop(columns_to_drop, axis=1, inplace=True)
#removendo valores nulos
df_fifa = df_fifa.dropna()
#separando overall das features
x = df_fifa.drop('Overall',axis=1).values
y = df_fifa.Overall.values
#importando StandardScaler
from sklearn.preprocessing import StandardScaler
#realizando a padronização dos dados
x = StandardScaler().fit_transform(x)
mx_cov = np.cov(x,rowvar=False)
#criando a matriz transposta
mx_exemplo = np.array([[1.76,75],
            [1.80,97.3]])
mx_exemplo.T
#tentativa de visualização da covariância (não da pra tirar muitas conclusões disto dai não...rsrs)
plt.figure(figsize=(30,30))
sns.heatmap(mx_cov,annot=True)
#entendendo a matriz de covariância
np.shape(mx_cov)
#Calculo dos autovalores e autovetores
autovalores,autovetores = np.linalg.eig(mx_cov)
print("Autovalores",autovalores)
print("Autovetores",autovetores)
#criando uma tupla de auto valores e autovetores;
tp_componentes = tuple(zip(autovalores,autovetores))
tp_componentes
#ordenando pelo autovalor
sorted(tp_componentes,reverse=True)
#calculo da variância acumulada
total = sum(autovalores)
var_acum = [autovalor/total for autovalor in sorted(autovalores,reverse=True)]
var_acum
#contando o número de componentes para obter 95% de informação
componentes = np.argmax(np.cumsum(var_acum)>=0.95)
ft_ds = tp_componentes[:componentes]
ft_vetor = list()
for val,vet in ft_ds:
    ft_vetor.append(vet)
df = pd.DataFrame(np.array(ft_vetor).T[:15])
np.shape(ft_vetor)
#Voltando ao Dataframe inicial com as features selecionadas.
featured_ds = np.dot(x[:,:15],np.array(ft_vetor).T[:15]) + x.mean()
df = pd.DataFrame(featured_ds)
df
