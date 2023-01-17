# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
print(df.head())

print(df.info())

print(df.isnull().sum())
# Não há valores nulos...no dataset...Vamos tentar clusterizar inspirados em CLASS 



from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

from sklearn.metrics import v_measure_score

from sklearn.metrics.cluster import homogeneity_score

from sklearn.metrics.cluster import completeness_score

import matplotlib.pyplot as plt





# Minhas labels serão a ordem de grandeza do número de suicídios

# Há as seguintes ordens de grandeza: 0,1,2,3,4 

X = np.array(df.drop(columns=['Class','Time']))

y = np.array(df['Class'])



print(set(y))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.35)



homogeneity_score_list = []

completeness_score_list = []

v_measure_score_list = []

highest_clusters = 10

lowest_clusters= 1

step = 1

values_range = range(lowest_clusters,highest_clusters+1,step)

for n_cluster in values_range:



  

    model = KMeans(n_clusters = n_cluster, max_iter=300)

    model.fit(X_train)

    print('n_clusters treinando...',n_cluster)



    # score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

    #print('v_measure_score, 1 é perfeito, 0 é ruim')

    v_measure_score_list.append(v_measure_score(y_test,model.predict(X_test)))



    #score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling

   #print('homogeneity_score, 1 é perfeito, 0 é ruim')

    homogeneity_score_list.append(homogeneity_score(y_test,model.predict(X_test)))

    #score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

    #print('compleness_score, 1 é perfeito, 0 é ruim')

    completeness_score_list.append(completeness_score(y_test,model.predict(X_test)))



plt.plot(values_range,v_measure_score_list,label='v_measure score')

plt.plot(values_range,homogeneity_score_list,label='homogeneity score')

plt.plot(values_range,completeness_score_list,label='completenes score')

plt.legend()



plt.show()







from sklearn.metrics import roc_curve,auc

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

import warnings

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    



    model = LogisticRegression()



    model.fit(X_train,y_train)

    y_score = model.decision_function(X_test)

    y_pred = model.predict(X_test)



    print('roc_auc_score: ',roc_auc_score(y_test,y_score))

    fpr, tpr, thresholds = roc_curve(y_test, y_score)



    label = 'área sob esta curva = '+str(roc_auc_score(y_test,y_score))

    plt.plot(fpr,tpr,label=label)

    plt.plot(np.arange(len(fpr))/len(fpr),np.arange(len(fpr))/len(fpr))

    plt.xlabel('taxa de falso positivo')

    plt.ylabel('taxa de positivo verdadeiro')

    plt.legend()

    plt.show()



    print('Agora o classificador logístico:')

    print('Acurácia da regressão logística para classificação :',model.score(X_test,y_test))



    print('no treino: ', (y_train==model.predict(X_train)).sum()/y_train.shape[0])



    print('Parece um fitting adequado.')

    #########
