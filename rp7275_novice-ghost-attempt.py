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
test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")



train.head()
def tran_type(s):

    if s == 'Ghoul':

        return 0

    elif s == 'Goblin':

        return 1

    else:

        return 2

    

def tran_color(s):

    color = ['clear', 'green', 'black', 'white', 'blue', 'blood'];

       

    return color.index(s)

    

train['type_1'] = train.type.apply(tran_type)

train['color_1'] = train.color.apply(tran_color)
train.head()
train['color'].unique()
from sklearn.ensemble import RandomForestClassifier

from numpy import genfromtxt, savetxt

rf = RandomForestClassifier(n_estimators=100)
rf.fit(train[["bone_length", "rotting_flesh", "hair_length", "has_soul", "color_1"]], train['type_1'])
test['color_1'] = test.color.apply(tran_color)

test['type'] = rf.predict(test[["bone_length", "rotting_flesh", "hair_length", "has_soul", "color_1"]])



def r_tran_type(s):

    st = ['Ghoul', 'Goblin', 'Ghost']

    return st[s]



test['type'] = test.type.apply(r_tran_type)
test.head()
sample = pd.read_csv('../input/sample_submission.csv')

sample.head()



#test[['id', 'type']].to_csv('output.csv', sep = ',')

test[['id', 'type']].head()
#savetxt('Data/submission2.csv', test[['id', 'type']], delimiter=',', fmt='%f')

test[['id', 'type']].to_csv('result.csv', index = False)