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



train_file = pd.read_csv('../input/train.csv')

test_file = pd.read_csv('../input/test.csv')



#train_file.groupby(['Pclass']).sum()

#train_file.plot(x='Fare', y='Survived', kind='scatter')



number_passengers = len(train_file)

number_survived = train_file['Survived'].sum()

proportion_survivors = number_survived / number_passengers 



print("Number of Passengers: " + str(number_passengers))

print("Survived: " + str(number_survived))

print("% Survived: " + str(proportion_survivors))



train_file.describe()

train_file.describe(include=['O'])








