import numpy as np  

import pandas as pd  

import csv

from sklearn.ensemble import RandomForestClassifier



def munge_data(df):

    '''fill in missing values and convert characters to numerical'''

    

    #transform 'Sex' column to numerical values

    df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



    #fill in nans with median ages for each class

    median_ages = np.zeros((2,3))

    for i in range(0, 2):

        for j in range(0, 3):

            median_ages[i,j] = df[(df['Sex'] == i) & \

                                (df['Pclass'] == j+1)]['Age'].dropna().median()



    for i in range(0, 2):

        for j in range(0, 3):

            df.loc[ (df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1), \

                    'Age'] = median_ages[i,j]



    #transform 'Embarked' column to numerical

    df['Embarked'] = df['Embarked'].dropna().map( {'C': 1, 'S': 2, 'Q': 3} ).astype(int)

    

    #find mode of 'Embarked' column

    mode = df['Embarked'].dropna().mode().astype(int)

     

    #fill NaNs with mode 

    df['Embarked'] = df['Embarked'].fillna(mode)



    #drop columns with strings that we can't use. 

    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)



    return df.fillna(0)



#import dataframe

train_df = pd.read_csv('../input/train.csv', header=0)

test_df  = pd.read_csv('../input/test.csv', header=0)



#save ids from test dataset

ids = test_df['PassengerId'].values



#munge data

train_df = munge_data(train_df)

test_df  = munge_data(test_df)



train_data = train_df.values

test_data = test_df.values



print('Training...')

rf = RandomForestClassifier(n_estimators=100)

rf = rf.fit( train_data[0::,1::], train_data[0::,0] )



print("Accuracy = ", (rf.predict(train_data[0::,1::])==train_data[0::,0]).mean())



#print('Predicting...')

#output = forest.predict(test_data).astype(int)



#predictions_file = open("myfirstforest.csv", "w")

#open_file_object = csv.writer(predictions_file)

#open_file_object.writerow(["PassengerId","Survived"])

#open_file_object.writerows(zip(ids, output))

#predictions_file.close()
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(10,4))



bar_placements = range(len(rf.feature_importances_))

ax.bar(bar_placements, rf.feature_importances_)

ax.set_title("Feature Importances")

ax.set_xticks([tick + .5 for tick in bar_placements])

ax.set_xticklabels(train_df.columns[1::])



f.show()
from sklearn import cross_validation



#default is k=3

scores = cross_validation.cross_val_score(rf, train_data[0::,1::], train_data[0::,0]) 

print(scores.mean())