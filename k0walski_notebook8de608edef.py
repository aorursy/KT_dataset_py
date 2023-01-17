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
titanic_data = pd.read_csv('../input/train.csv', sep=',')



# Set some variables

number_passengers = titanic_data.shape[0]

number_survived = np.sum(titanic_data['Survived'])

proportion_survivors = number_survived / number_passengers 



women_only_stats = titanic_data['Sex'] == "female" 	# This finds where all the women are

men_only_stats = titanic_data['Sex'] != "female" 	# This finds where all the men are (note != means 'not equal')



women_onboard = titanic_data[titanic_data['Sex'] == 'female']

men_onboard = titanic_data[titanic_data['Sex'] != 'female']



proportion_women_survived = np.sum(women_onboard['Survived']) / women_onboard.shape[0]

proportion_men_survived = np.sum(men_onboard['Survived']) / men_onboard.shape[0]



print('Proportion of women who survived is %s' % proportion_women_survived)

print('Proportion of men who survived is %s' % proportion_men_survived)



# First, read in test.csv

test_data = pd.read_csv('../input/test.csv', sep=',')



predictions = test_data[['PassengerId', 'Sex']]

predictions = predictions.apply(lambda row: [row['PassengerId'], 1] if row['Sex'] == 'female' else [row['PassengerId'], 0], axis=1)

predictions.to_csv('./kowalski_gender.csv')