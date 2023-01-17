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

path="../input/train.csv"

raw_data = pd.read_csv(path)

raw_data[:5]
clc_data = raw_data
clc_data[:5]
clc_data=clc_data.drop('Ticket',axis=1)
len(clc_data)
clc_data=clc_data.drop('Embarked',axis=1)
clc_data[:5]
#knowing each variables

parch_data=clc_data['Parch']

parch_data.value_counts()
any_sibling = clc_data['SibSp']

any_sibling.value_counts()
#cabin_data=clc_data['Cabin']

clean_tr = clc_data['Cabin'].fillna('NaN')

clean_tr[clean_tr=='NaN'] ='Unknown'
clean_tr.value_counts()
valueAddingData=(687/891)*100

valueAddingData
clc_data=clc_data.drop('Cabin',axis=1)
clc_data.head()
clc_data=clc_data.drop('Name',axis=1)
clc_data.head()
outcomes = clc_data['Survived']

clc_data =clc_data.drop('Survived',axis = 1 )
def acurracy(outcome,predictor):

    if len(outcomes) == len(predictor):

        return 'acurracy score is {0}'.format((outcome == predictor).mean()*100)

    else :

        return "outcomes do not match with predictor"
#step one cosider if all passenger died

def prediction_0(data):

    predictor =[]

    for _,passenger in data.iterrows():

            predictor.append(0)

            

    return pd.Series(predictor)        
act_1 = prediction_0(clc_data)
acurracy(outcomes,act_1)