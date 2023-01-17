# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore') 

# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import pandas as pd

#from math import pi

from collections import Counter

import numpy as np

import seaborn as sns

from sklearn.decomposition import PCA

import pylab
pylab.rcParams['figure.figsize'] = (9.0, 9.0)








train = pd.read_csv('../input/train.csv')#extract values

train_x = train.drop('label',axis = 1)

test = pd.read_csv('../input/test.csv',index_col = None)



#train_x = train.iloc[:,1:].values

#test_x = test.iloc[:,:].values



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)

test_x = scaler.fit_transform(test)







pca = PCA(n_components=2)

principalComponents = pca.fit_transform(train_x)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principalcomponent1',

                                                                  'principalcomponent2'])

label = pd.DataFrame(data = train['label'])

principalDf = pd.concat([principalDf,label],axis = 1,ignore_index=True)



principalDf.columns = ["principalcomponent1", "principalcomponent2", "label"] 

# give a list to the marker argument

sns.regplot(x=principalDf["principalcomponent1"], y=principalDf["principalcomponent2"],

            fit_reg=False, scatter_kws={"color":"darkred","alpha":0.3,"s":200} )



sns.lmplot( x='principalcomponent1', y='principalcomponent2', data=principalDf, fit_reg=False, 

           hue='label', legend=False, palette="Blues")

plt.figure(figsize=(13,10))





flatui = ["#9b59b6", "#3498db", "orange"]

sns.set_palette(flatui)

sns.lmplot( x="principalcomponent1", y="principalcomponent2", data=principalDf, fit_reg=False,

           hue='label', legend=False)



plt.figure(figsize=(13,10))


from sklearn.decomposition import PCA

# Make an instance of the Model

pca2 = PCA(.95)



pca2.fit(train_x)



train_2 = pca.transform(train_x)

test_2 = pca.transform(test_x)



from sklearn.linear_model import LogisticRegression

logisticRegr = LogisticRegression(solver = 'lbfgs')



logisticRegr.fit(train_2, train['label'])

test_pred=logisticRegr.predict(test_2)









submission = pd.DataFrame({

        "ImageId": np.arange(1,28001),

        "Label": test_pred

    })

submission.to_csv('submission.csv', index=False)



#a=np.arange(1,28001)