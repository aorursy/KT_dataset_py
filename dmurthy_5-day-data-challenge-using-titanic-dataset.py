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
#separate out dependent (Y) & independent values (X)

read_data = pd.read_csv("../input/train.csv")

#check for null data

read_data.isnull().any()
#summary of the data

print(read_data.describe())
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)

imputer = imputer.fit(read_data['Age'].reshape(-1,1))

read_data['Age'] = imputer.transform(read_data['Age'].reshape(-1,1)).reshape(-1)
#feature selection

X_data = read_data.iloc[:,[2,4,5,6,7,9]].values

Y_data = read_data.iloc[:,1].values
import seaborn as sns
sns.distplot(read_data['Fare'],kde=False).set_title("Fare Distribution")
from scipy.stats import ttest_ind

ttest_ind(read_data['Age'],read_data['Fare'],equal_var=False)
sns.distplot(read_data['Age'],kde = False).set_title("Age Distribution")
sns.distplot(read_data['Fare'],kde = False).set_title("Fare Distribution")
sns.countplot(read_data['Sex']).set_title('Sex Count')
sns.countplot(read_data['Pclass']).set_title('Passenger Class Count')
sns.countplot(x=read_data['Cabin']).set_title('Cabin Class Count')
from scipy.stats import chisquare

chi_sex = chisquare(read_data['Fare'],read_data['Age'])

print(chi_sex)