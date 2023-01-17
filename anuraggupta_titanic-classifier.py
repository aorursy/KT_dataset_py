import pandas as pd

import seaborn as sns

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

import numpy as np



df_train = pd.read_csv("../input/train.csv")

df_train.info()





def is_female(row):

    if(row=='male'):

        return 0.0

    else:

        return 1.0



    

def is_pclass_high(row):

    if(row==1):

        return 1.0

    else:

        return 0.0





def is_embarked_c(row):

    if(row=='C'):

        return 1.0

    else:

        return 0.0
df_train['is_female'] = df_train['Sex'].apply(is_female)

sns.factorplot("is_female", "Survived", data=df_train)



df_train['is_pclass_high'] = df_train['Pclass'].apply(is_pclass_high)

sns.factorplot("is_pclass_high", "Survived", data=df_train)



df_train['is_embarked_c'] = df_train['Embarked'].apply(is_embarked_c)

sns.factorplot("is_embarked_c", "Survived", data=df_train)



features_train = np.array(df_train[['is_female', 'is_pclass_high', 'is_embarked_c']])

labels_train = np.array(df_train['Survived'])



clf = SVC(kernel='rbf', C=5)



score = cross_val_score(clf, features_train, labels_train, cv=5).mean()

print(score)



    