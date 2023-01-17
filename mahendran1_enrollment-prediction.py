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
file = pd.read_csv("/kaggle/input/education-in-india/Statement_SES_2011-12-Enrlment.csv")

file
data = file.iloc[8:,0:]

data
for col in data.columns:

    print(col)
res = data.drop(columns=['All Categories - Class I-V - Boys',

                        'All Categories - Class I-V - Girls',

                        'All Categories - Class VI-VIII - Boys',

                        'All Categories - Class VI-VIII - Girls',

                        'All Categories - Class IX-X - Boys',

                        'All Categories - Class IX-X - Girls',

                        'All Categories - Class XI-XII - Boys',

                        'All Categories - Class XI-XII - Girls',

                        'Scheduled Caste Category - Class I-V - Boys',

                        'Scheduled Caste Category - Class I-V - Girls',

                        'Scheduled Caste Category - Class VI-VIII - Boys',

                        'Scheduled Caste Category - Class VI-VIII - Girls',

                        'Scheduled Caste Category - Class IX-X - Boys',

                        'Scheduled Caste Category - Class IX-X - Girls',

                        'Scheduled Caste Category - Class XI-XII - Boys',

                        'Scheduled Caste Category - Class XI-XII - Girls',

                        'Scheduled Tribe Category - Class I-V - Boys',

                        'Scheduled Tribe Category - Class I-V - Girls',

                        'Scheduled Tribe Category - Class VI-VIII - Boys',

                        'Scheduled Tribe Category - Class VI-VIII - Girls',

                        'Scheduled Tribe Category - Class IX-X - Boys',

                        'Scheduled Tribe Category - Class IX-X - Girls',

                        'Scheduled Tribe Category - Class XI-XII - Boys',

                        'Scheduled Tribe Category - Class XI-XII - Girls',

                        ])
for col in res.columns:

    print(col)
temp=res
temp.iloc[:,1] += temp.iloc[:,5]+temp.iloc[:,9]

temp.iloc[:,2] += temp.iloc[:,6]+temp.iloc[:,10]

temp.iloc[:,3] += temp.iloc[:,7]+temp.iloc[:,11]

temp.iloc[:,4] += temp.iloc[:,8]+temp.iloc[:,12]

temp
temp = temp.drop(columns=['Scheduled Caste Category - Class I-V - Total',

                          'Scheduled Caste Category - Class VI-VIII - Total',

                          'Scheduled Caste Category - Class IX-X - Total',

                          'Scheduled Caste Category - Class XI-XII - Total',

                          'Scheduled Tribe Category - Class I-V - Total',

                          'Scheduled Tribe Category - Class VI-VIII - Total',

                          'Scheduled Tribe Category - Class IX-X - Total',

                          'Scheduled Tribe Category - Class XI-XII - Total',])

temp
temp = temp.rename(columns={"All Categories - Class I-V - Total":"1-5",

                    "All Categories - Class VI-VIII - Total":"5-8",

                    "All Categories - Class IX-X - Total":"9-10",

                    "All Categories - Class XI-XII - Total":"11-12"})

temp
year=[]

for i in temp.iloc[:,0]:

    i = i[:4]

    year.append(int(i))



temp = pd.DataFrame(temp)

temp.insert(0,"year",year)

temp = temp.drop(columns=['Year'])

temp
data = temp.to_numpy()

data
data=data.T

data
import matplotlib.pyplot as plt

plt.plot(data[0],data[1],'.')

plt.plot(data[0],data[2],'.')

plt.plot(data[0],data[3],'.')

plt.show()
from sklearn.linear_model import LinearRegression

reg1to5 = LinearRegression()

reg6to8 = LinearRegression()

reg9to10 = LinearRegression()
reg1to5.fit(data[0].reshape(-1,1),data[1].reshape(-1,1))

print("Students who are learning from 1 to 5 in 2019 : ",round(reg1to5.predict([[2019]])[0][0],2),"millions")

print("Students who are learning from 1 to 5 in 2020 : ",round(reg1to5.predict([[2020]])[0][0],2),"millions")

reg6to8.fit(data[0].reshape(-1,1),data[2].reshape(-1,1))

print("\nStudents who are learning from 6 to 8 in 2019 : ",round(reg6to8.predict([[2019]])[0][0],2),"millions")

print("Students who are learning from 6 to 8 in 2020 : ",round(reg6to8.predict([[2020]])[0][0],2),"millions")

reg9to10.fit(data[0].reshape(-1,1),data[3].reshape(-1,1))

print("\nStudents who are learning from 9 to 10 in 2019 : ",round(reg9to10.predict([[2019]])[0][0],2),"millions")

print("Students who are learning from 9 to 10 in 2020 : ",round(reg9to10.predict([[2020]])[0][0],2),"millions")