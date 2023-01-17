# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.naive_bayes import MultinomialNB







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/spam-or-not-spam-dataset/spam_or_not_spam.csv')
df.head()
df.loc[df.label == 1]
df['label'].value_counts()

def dicionario_todas_palavras(df:pd.DataFrame,coluna:str) -> dict:

 

  palavras_splitadas = df.dropna(subset=[coluna])[coluna].str.lower().str.split()

  colecao_palavras = set()



  for lista in palavras_splitadas:

    try:

      colecao_palavras.update(lista)

    except:

      print(lista)

    

  total_palavras = len(colecao_palavras)





  dicionario = dict(zip(colecao_palavras, range(total_palavras)))



  return dicionario



 

        
def vetorizar_palavras(texto:str, dicionario:dict) -> list:

    vetor = [0] * len(dicionario)

    for palavra in texto:

        if palavra in dicionario:

            posicao = dicionario[palavra]

            vetor[posicao] += 1

    return vetor

def gerador_array_palavras(dicionario:dict,df:pd.DataFrame,coluna):

  palavras_splitadas = df.dropna(subset=[coluna])[coluna].str.lower().str.split()

  

  vetores_texto = [vetorizar_palavras(texto, dicionario) for texto in palavras_splitadas]

  return vetores_texto
dicionario = dicionario_todas_palavras(df,'email')
df = df.sample(frac=1,random_state=1)



email_ok = df.loc[df.label == 0][:500]

email_spam = df.loc[df.label == 1]
emails = pd.concat([email_ok,email_spam])









## Vetor de palavras gerado, e transformado em um numpy array

X = np.array(gerador_array_palavras(dicionario,emails,'email'))



## dropando os valores nulos

Y = emails.dropna().values[:,1]



## transformando o Y em inteiro 

Y = Y.astype('int')





x_treino,x_teste,y_treino,y_teste = train_test_split(X,Y,random_state= 1,test_size = 0.3)




mult  = MultinomialNB(alpha=0.5)



mult.fit(x_treino,y_treino)

predicts = mult.predict(x_teste)

accuracy_score(y_teste,predicts)



sns.heatmap(confusion_matrix(y_teste,predicts), annot=True, fmt="g", cmap=plt.cm.copper);
