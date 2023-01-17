# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

data = train_data.append(test_data, ignore_index = True)
train_data.groupby('Sex').Survived.value_counts()
#That model achieves 76.5% 

test_data['Survived'] = np.where((test_data.Sex == 'female'), 1, 0)

output = pd.DataFrame(({'PassengerId': test_data.PassengerId, 'Survived': test_data.Survived}))

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")

train_data.groupby(['Sex', 'Pclass']).Survived.value_counts()
data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand = False)





data.Title = data.Title.replace('Ms', 'Miss')

data.Title = data.Title.replace('Mlle', 'Miss')

data.Title = data.Title.replace('Mme', 'Mrs')

data.Title = np.where((data.Title == 'Dr') & (data.Sex == 'female'), 'Mrs', data.Title)

data.Title = np.where((data.Title == 'Dona')| 

                      (data.Title == 'Lady')|

                      (data.Title == 'Countess'), 'Mrs', data.Title)

data.Title = np.where((data.Title == 'Rev')|

                      (data.Title == 'Dr') & (data.Sex == 'male')|

                      (data.Title == 'Col')| 

                      (data.Title == 'Major')|

                      (data.Title == 'Sir')|

                      (data.Title == 'Don')|

                      (data.Title == 'Capt')|

                      (data.Title == 'Jonkheer'), 'Mr', data.Title)



data.Title = data.Title.replace('Mrs', 'Miss')
data['Surname'] = data.Name.str.extract('([A-Za-z]+)\,', expand = False)



data['Surname'] = np.where((data.Title == 'Mr'), 'NoGroup', data.Surname)



train_data = data[:(len(train_data))]

test_data = data[(len(train_data)):]



train_data['SurnameFreq'] = train_data.groupby('Surname')['Surname'].transform('count')



a = train_data.groupby('Surname')['Survived'].sum().reset_index()

a = a.rename(columns ={'Surname': 'Surname_1', 'Survived': 'SSurv'})



train_data = pd.merge(train_data, a, how ='left', left_on=('Surname'), right_on=('Surname_1'))

train_data['SurnameSurv'] = train_data.SSurv/train_data.SurnameFreq



b = pd.DataFrame({'Surname': train_data.Surname, 'SurnameSurv': train_data.SurnameSurv})



test_data = pd.merge(test_data, b, on = ('Surname'), how = 'left')

test_data = test_data.drop_duplicates(subset=['Name'], keep='first')



# NaN values in SurnameSurv I replace with 2 for they are not mixed with other results

test_data.SurnameSurv = test_data.SurnameSurv.fillna(2)

test_data


test_data['Survived'] = 0



test_data['Survived'] = np.where((test_data.Title == 'Miss'), 1, 0)



test_data['Survived'] = np.where((test_data.Title == 'Master') & (test_data.SurnameSurv == 1),

                                  1, test_data.Survived)



test_data['Survived'] = np.where((test_data.Title == 'Miss') & (test_data.SurnameSurv == 0), 

                                  0, test_data.Survived)



output = pd.DataFrame(({'PassengerId': test_data.PassengerId, 'Survived': test_data.Survived}))

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")


