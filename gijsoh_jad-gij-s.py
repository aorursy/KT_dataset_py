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
titanic_df = pd.read_csv('../input/train.csv')

titanic_df.head()
Number_of_survived = np.sum(titanic_df['Survived'] == 1)

print(Number_of_survived)
Number_of_survived_male = np.sum((titanic_df['Survived'] == 1) & (titanic_df['Sex'] == 'male'))

print("Amount of male survivors:", Number_of_survived_male)
titanicdata = pd.read_csv("../input/train.csv")
Number_of_survived_male = np.sum((titanicdata['Survived'] == 1) & (titanicdata['Sex'] == 'male'))

print("Amount of male survivors:", Number_of_survived_male)
Number_of_survived_male = np.sum((titanicdata['Survived'] == 1) & (titanicdata['Sex'] == 'male'))

print("Amount of male survivors:", Number_of_survived_male)

Number_of_survived_female = np.sum((titanicdata['Survived'] == 1) & (titanicdata['Sex'] == 'female'))

print("Amount of female survivors:", Number_of_survived_female)
Number_of_survived_male = np.sum((titanicdata['Survived'] == 1) & (titanicdata['Sex'] == 'male'))

print("Amount of male survivors:", Number_of_survived_male, "of the", np.sum(titanicdata['Sex'] == 'male'), "male visitors")

Number_of_survived_female = np.sum((titanicdata['Survived'] == 1) & (titanicdata['Sex'] == 'female'))

print("Amount of female survivors:", Number_of_survived_female, "of the", np.sum(titanicdata['Sex'] == 'female'), "female visitors")
average_age = np.sum(titanicdata['Age'])

print(average_age)
titanicdata["Age"] = titanicdata["Age"].fillna(titanicdata["Age"].median())

print(titanicdata["Age"])
Average_age = np.average(titanicdata["Age"])

print(Average_age)

print(np.median(titanicdata["Age"]))