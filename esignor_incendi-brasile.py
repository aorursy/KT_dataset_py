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
import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cmath as math



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_fire = pd.read_csv('/kaggle/input/amazon.csv', nrows=1500, usecols=[3, 4], encoding='latin-1') # con la lettura creo un dataframe di amazon con le colonne number e date

df_fire.head(15)

i = 0

X = []

df_fire['date'] = df_fire['date'].astype(str) # cast di date in string

df_fire['new_date'] = df_fire['date'].str.replace('\D', '').astype(int) # cast di date in int in new_date

while i < len(df_fire):

    p = 0

    lista = []

    while p < 3:

        if(p==0):

            value = 1

        else:

            value = (pow(df_fire['new_date'][i],p))

        lista.insert(p,value)

        p += 1

    X.insert(i, lista)

    i += 1

        

y = df_fire['number'] # memorizzo il contenuto della colonna number del dataframe in y

#print(X)

#print(y)


from sklearn import linear_model

# calcolo dei punti di alpha

n_alphas = 25

alphas = np.logspace(-10, -2, n_alphas) # intervalli per alpha



# calcolo del modello di regressione lineare

coord = []

for a in alphas:

    ridge = linear_model.Ridge(alpha=a, fit_intercept=False) # riduco al minimo la funzione obiettivo

    ridge.fit(X, y) # modello di regressione Fit Ridge

    coord.append(ridge.coef_) # appendo il vettore di peso

    
import matplotlib.pyplot as plt

# disegno del plot



ax = plt.gca()



ax.plot(alphas, coord)

ax.set_xscale('log')

ax.set_xlim(ax.get_xlim()[::-1])

plt.xlabel('alpha')

plt.ylabel('weights')

plt.title('Ridge coordinate as a function of the regularization')

plt.axis('tight')

plt.show()