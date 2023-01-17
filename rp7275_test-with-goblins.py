# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
ghost = pd.read_csv('../input/train.csv')

ghost.head()
import matplotlib.pyplot as plt

%matplotlib inline
ghost.color.unique()
type_c = ['Ghoul', 'Ghost', 'Goblin']

gC = ['b','r','c','m','k','g']



for i in range(len(type_c)):

    plt.scatter(ghost[ghost.type == type_c[i]]['has_soul'], ghost[ghost.type == type_c[i]]['hair_length'], c = gC[i])

#plt.scatter(ghost[ghost.color == color[0]]['bone_length'], ghost[ghost.color == color[0]]['hair_length'], c = gC[0])

#plt.scatter(ghost[ghost.color == color[1]]['bone_length'], ghost[ghost.color == color[1]]['hair_length'], c = gC[1])
ghost.head()
from sklearn import svm

clf = svm.SVC(decision_function_shape='ovo')

X = ghost[['bone_length', 'rotting_flesh','hair_length','has_soul']]

Y = ghost.type

clf.fit(X, Y)
test = pd.read_csv('../input/test.csv')



test['type'] = clf.predict(test[['bone_length', 'rotting_flesh','hair_length','has_soul']])



test[['id', 'type']].to_csv('predction.csv', header = True, index_label='id')