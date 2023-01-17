# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Import packages

import numpy as np

import pandas as pd

import matplotlib. pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder



plt.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv("../input/ufcdata/data.csv")
#Count number of rows

count_row = df.shape[0] 

print(count_row)
#Drop NAs

df.dropna(how='any', inplace=True)
#Drop Stances which are missing (as mean cannot be obtained from string value)

df_clean = df.copy()

df_clean = df_clean[df_clean['B_Stance'].notna()]

df_clean = df_clean[df_clean['R_Stance'].notna()]

count_row = df_clean.shape[0] 

print(count_row)
"""

Encoding gender

Males are 1, females are 0



Encoding weight classes

"Women's Strawweight":0

"Women's Flyweight":1

"Women's Bantamweight":2

"Women's Featherweight":3

"Flyweight":4

"Bantamweight":5

"Featherweight":6

"Lightweight":7

"Welterweight":8

"Middleweight":9

"Light Heavyweight":10

"Heavyweight":11

"Catch Weight":12

"Open Weight":13

"""



weight_class = df_clean.loc[:,'weight_class']

weight_class_list = weight_class.tolist()

genderList = []

genderCount = [0,0]

weight_class_numbers = []

weight_class_dict = {"Women's Strawweight":0,"Women's Flyweight":1,"Women's Bantamweight":2,"Women's Featherweight":3,"Flyweight":4,"Bantamweight":5,"Featherweight":6,"Lightweight":7,"Welterweight":8,"Middleweight":9,"Light Heavyweight":10,"Heavyweight":11,"Catch Weight":12,"Open Weight":13}

#print(weight_class_gender)

for weights in weight_class_list:

    if "Women" in weights:

        genderList.append("f")

        genderCount[0]+=1

    else: 

        genderList.append("m")

        genderCount[1]+=1

    weight_class_numbers.append(weight_class_dict[weights])



genderValues = np.array(genderList)

# integer encode

label_encoder = LabelEncoder()

gender_encoded = label_encoder.fit_transform(genderValues)
#Adding gender and weight class into DF as numbers

df_clean.insert(0, 'gender', gender_encoded)

df_clean.drop(['weight_class'], axis=1, inplace = True)

df_clean.insert(8, 'weight_class', weight_class_numbers)

df_clean.head(5)
"""

Encoding stances

"Open Stance":0

"Orthodox":1

"Southpaw":2

"Switch":3

"Sideways":4

"""

stances_dict ={"Open Stance":0,"Orthodox":1,"Southpaw":2,"Switch":3, "Sideways":4}

b_stance = df_clean.loc[:,'B_Stance']

r_stance= df_clean.loc[:,'R_Stance']

b_stance_list = b_stance.tolist()

r_stance_list = r_stance.tolist()

b_stance_int_list = []

r_stance_int_list = []

counter = range(len(df_clean.index))

for rows in counter:

    b_stance_int_list.append(stances_dict[b_stance_list[rows]])

    r_stance_int_list.append(stances_dict[r_stance_list[rows]])



b_stanceValues = np.array(b_stance_int_list)

r_stanceValues = np.array(r_stance_int_list)

#Dropping previous 'Stance' columns

df_clean.drop(['R_Stance','B_Stance'], axis=1, inplace = True)

#Adding int stances into DF

df_clean.insert(3, 'B_Stance', b_stanceValues)

df_clean.insert(4, 'R_Stance', r_stanceValues)

df_clean.head(10)
"""

Encoding match results

Red win = 1, Red lose = 0 & draw = 2

Did not use label encoder, because it labels in Alphabetical order

"""

match_results = df_clean.loc[:,'Winner']

match_results_list = match_results.tolist()

matchList = []

#print(match_results_list)

for results in match_results_list:

    if "Blue" in results:

        matchList.append("0")

    elif "Red" in results: 

        matchList.append("1")

    else:

        matchList.append("2")

        

resultValues = np.array(matchList)

#Dropping previous 'Winner' column

df_clean.drop(['Winner'], axis=1, inplace = True)

#Adding results into DF

df_clean.insert(6, 'Winner', resultValues)



#Dropping Draw winners

df_clean.drop(df_clean[df_clean['Winner'] == '2' ].index , inplace=True)

df_clean.head(10)

"""

Encoding Title Bout

True = 1, False = 0

Using label encoder

"""

title = df_clean.loc[:,'title_bout']

title_list = title.tolist()

titleValues = np.array(title_list)

label_encoder_title = LabelEncoder()

title_encoded = label_encoder_title.fit_transform(titleValues)

#Dropping previous 'Winner' column

df_clean.drop(['title_bout'], axis=1, inplace = True)

#Adding results into df_clean

df_clean.insert(7, 'title_bout', title_encoded)

df_clean.head(10)

df_clean.drop(['R_fighter', 'B_fighter','date','location','Referee'], axis=1, inplace = True)
df_clean.head()
df_clean.to_csv("cleaned_data.csv", index=False)