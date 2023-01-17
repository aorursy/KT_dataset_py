import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Load data

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
def detect_outliers(df,features):

    outlier_indices = []

    for c in features:

        # 1st quartile

        Q1 = np.percentile(df[c],25)

        # 3rd quartile

        Q3 =np.percentile(df[c],75)

        # IQR

        IQR = Q3 - Q1

        #Outlier Step

        outlier_step = IQR * 1.5

        # Detect outlier and their indices

        outlier_list_col = df[(df[c] <Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        #store indices

        outlier_indices.extend(outlier_list_col)

        

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i,v in outlier_indices.items() if v > 2) 

    

    return multiple_outliers

        

    

    
train_data.loc[detect_outliers(train_data,["Age","SibSp","Parch","Fare"])]
#drop outliers

train_data = train_data.drop(detect_outliers(train_data,["Age","SibSp","Parch","Fare"]),axis = 0).reset_index(drop = True)