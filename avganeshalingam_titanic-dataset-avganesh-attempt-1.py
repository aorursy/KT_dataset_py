import pandas as pd

pd.options.display.max_columns = 100

pd.options.display.max_rows = 100

from matplotlib import pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

import numpy as np
def process_names():

    

    global combined

    # we clean the Name variable

    combined.drop('Name',axis=1,inplace=True)

    

    # encoding in dummy variable

    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')

    combined = pd.concat([combined,titles_dummies],axis=1)

    

    # removing the title variable

    combined.drop('Title',axis=1,inplace=True)

    

    status('names')
def process_fares():

    

    global combined

    # there's one missing fare value - replacing it with the mean.

    combined.Fare.fillna(combined.Fare.mean(),inplace=True)

    

    status('fare')
def process_embarked():

    

    global combined

    # two missing embarked values - filling them with the most frequent one (S)

    combined.Embarked.fillna('S',inplace=True)

    

    # dummy encoding 

    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')

    combined = pd.concat([combined,embarked_dummies],axis=1)

    combined.drop('Embarked',axis=1,inplace=True)

    

    status('embarked')
pipeline = grid_search
output = pipeline.predict(test_new).astype(int)

df_output = pd.DataFrame()

df_output['PassengerId'] = test['PassengerId']

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)