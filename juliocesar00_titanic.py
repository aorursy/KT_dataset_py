# remove warnings
import warnings

warnings.filterwarnings('ignore')
# ---

%matplotlib inline
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np

pd.options.display.max_rows = 100
data = pd.read_csv('../input/train.csv')
data['Age'].fillna(data['Age'].median(), inplace=True)
data.head()
data.describe()
# A variável contagem mostra que 177 valores estão faltando na coluna Age.
#Uma solução é substituir os valores nulos pela idade mediana, que é mais robusta para valores aberrantes do que a média.
#Uma solução é substituir os valores nulos pela idade mediana, que é mais robusta para valores aberrantes do que a média.


survived_sex = data[data['Survived']==1]['Sex'].value_counts()
dead_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))
#Visualizar a sobrevivência com base no gênero.
#A variável Sexo parece ser uma característica decisiva. As mulheres são mais propensas a sobreviver.
#Vamos agora correlacionar a sobrevivência com a variável idade.
figure = plt.figure(figsize=(15,8))
plt.hist([data[data['Survived']==1]['Age'], data[data['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()

#Se você seguir o compartimento do gráfico por lixo, você notará que os passageiros com menos de 10 anos são mais propensos a sobreviver do que os 
#mais velhos que são mais de 12 e menos de 50. Os passageiros mais velhos parecem ser resgatados também.

#Esses dois primeiros gráficos confirmam que um antigo código de conduta que os marinheiros e os capitães seguem em caso 
#de situações ameaçadoras: "Mulheres e crianças primeiro!" .
figure = plt.figure(figsize=(15,8))
plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
#Os passageiros com tarifas mais baratas são mais propensos a morrer. De modo diferente, os passageiros com 
#bilhetes mais caros e, portanto, um status social mais importante, parecem serem resgatados primeiro.

plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(data[data['Survived']==1]['Age'],data[data['Survived']==1]['Fare'],c='green',s=40)
ax.scatter(data[data['Survived']==0]['Age'],data[data['Survived']==0]['Fare'],c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)
#Vamos agora combinar a idade, a tarifa e a sobrevivência em um único gráfico.
#Um grupo distinto de passageiros mortos (o vermelho) aparece no gráfico. 
#Essas pessoas são adultos (idade entre 15 e 50) de classe baixa (tarifas mais baixas).
ax = plt.subplot()
ax.set_ylabel('Average fare')
data.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(15,8), ax = ax)
#Um grupo distinto de passageiros mortos (o vermelho) aparece no gráfico. 
#Essas pessoas são adultos (idade entre 15 e 50) de classe baixa (tarifas mais baixas).
survived_embark = data[data['Survived']==1]['Embarked'].value_counts()
dead_embark = data[data['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark,dead_embark])
df.index = ['Survived','Dead']
df.plot(kind='bar', stacked=True, figsize=(15,8))
#Vamos agora ver como o site de embarque afeta a sobrevivência.
#Parece não haver uma correlação distinta aqui.
