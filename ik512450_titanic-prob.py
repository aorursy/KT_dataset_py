# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

training_sets =pd.read_csv("../input/titanic/train.csv")

training_sets.head()



# spliting male and female on data sets

male = training_sets[training_sets.Sex == "male"]

female = training_sets[training_sets.Sex == "female"]



# compute the servival rate 

women_survival_rate = float(sum(female.Survived))/len(female)

men_survival_rate = float(sum(male.Survived))/len(male)



# compute survival rate 

print("Women surviver rate :-")

print(women_survival_rate)

print("men surviver rate :-")

print(men_survival_rate)