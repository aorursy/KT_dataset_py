# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Math_class = pd.read_csv("../input/student-alcohol-consumption/student-mat.csv")

Por_class = pd.read_csv("../input/student-alcohol-consumption/student-por.csv")
Math_class.columns

Math_class['Famsize'] = Math_class.famsize.map({'GT3':0,'LE3':1})
Por_class['Famsize'] = Math_class.famsize.map({'GT3':0,'LE3':1})
Math_class['Guardian'] = Math_class.guardian.map({'father':0,'mother':1,'other':-1})

Por_class['Guardian'] = Por_class.guardian.map({'father':0,'mother':1,'other':-1})
def add(val):

    total = 0 

    for i in val:

        total += i 

    return total



Math_class['Family_point'] = Math_class.loc[:,['famrel','Guardian','Famsize']].apply(add,axis=1)

Math_class.loc[:,['Family_point']].head()
Por_class['Family_point'] = Por_class.loc[:,['famrel','Guardian','Famsize']].apply(add,axis=1)

Por_class.loc[:,['Family_point']].head()
Math_class['total_C'] = Math_class.loc[:,['Walc','Dalc']].apply(add,axis=1)

Math_class.loc[:,['total_C']].head()
Por_class['total_C'] = Por_class.loc[:,['Walc','Dalc']].apply(add,axis=1)

Por_class.loc[:,['total_C']].head()
M_col_1 = Math_class['Family_point']

M_col_2 = Math_class['total_C']



P_col_1 = Math_class['Family_point']

P_col_2 = Math_class['total_C']



fig, ax1 = plt.subplots()

fig, ax2 = plt.subplots()



ax1.scatter(M_col_1,M_col_2,alpha=0.09,color='black')

ax2.scatter(P_col_1,P_col_2,alpha=0.09,color='black')



ax1.set_xlabel('Family point')

ax1.set_ylabel('Alcohol cons')

ax1.set_title('Family\Alcohol(Math)')





ax2.set_xlabel('Family point')

ax2.set_ylabel('Alcohol cons')

ax2.set_title('Family\Alcohol(Port)')

plt.tight_layout()

plt.show()
vote_M = Math_class.groupby(['Walc','romantic'])['sex'].count()

vote_P = Por_class.groupby(['Walc','romantic'])['sex'].count()
M_yes=[]

M_no=[]

P_yes=[]

P_no=[]





for i in range(0,10,2):

    M_no.append(list(vote_M)[i])

    M_yes.append(list(vote_M)[i+1])

    

for i in range(0,10,2):

    P_no.append(list(vote_P)[i])

    P_yes.append(list(vote_P)[i+1])

cons = [1,2,3,4,5]
x_index = np.arange(len(cons))



plt.bar(x_index,M_yes , width=0.20 , label='yes')

plt.bar(x_index-0.20 , M_no , width=0.20 , label='no')



plt.xticks(ticks=x_index  , labels=cons)



plt.legend(loc='upper right')

plt.title('Consume\Romantic(Math class)')

plt.xlabel('consume')

plt.ylabel('student')

plt.show()



plt.bar(x_index,P_yes , width=0.20 , label='yes')

plt.bar(x_index-0.20 , P_no , width=0.20 , label='no')



plt.xticks(ticks=x_index  , labels=cons)



plt.legend(loc='upper right')

plt.title('Consume\Romantic(Por class)')

plt.xlabel('consume')

plt.ylabel('student')

plt.show()