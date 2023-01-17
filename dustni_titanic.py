# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
path='../input/'

in_file = path+'test.csv'

full_data = pd.read_csv(in_file)

print('file loaded')
def predictions(data):

    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """

    

    predictions = []

    for _, passenger in data.iterrows():

    

        if passenger['Sex']=='female':

            predictions.append(1)

        elif passenger['Age']<10:

            predictions.append(1)

        else:

            predictions.append(0)

            

            

        

    

    # Return our predictions

    return pd.Series(predictions)



# Make the predictions

predictions = predictions(full_data)

print(predictions)


predictions.to_csv('result.csv')

print('Already output')

            