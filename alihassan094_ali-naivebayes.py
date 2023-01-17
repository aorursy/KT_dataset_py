# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import statistics as st

import math

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
url1 = '/kaggle/input/Females_training_data.csv'

url2 = '/kaggle/input/Females_testing_data.csv'

url3 = '/kaggle/input/Males_training_data.csv'

url4 = '/kaggle/input/Males_testing_data.csv'

femalestrain_data=pd.read_csv(url1)

femalestest_data=pd.read_csv(url2)

malestrain_data=pd.read_csv(url3)

malestest_data=pd.read_csv(url4)

features = ['Gender', 'Height', 'Weight', 'Index']

# Training Data

malestraing = malestrain_data.loc['0':, features[0]].values

malestrainh = malestrain_data.loc['0':, features[1]].values 

malestrainw = malestrain_data.loc['0':, features[2]].values 

femalestraing = femalestrain_data.loc['0':, features[0]].values

femalestrainh = femalestrain_data.loc['0':, features[1]].values 

femalestrainw = femalestrain_data.loc['0':, features[2]].values 



# Testing Data

testg = malestest_data.loc['0':, features[0]].values

testh = malestest_data.loc['0':, features[1]].values 

testw = malestest_data.loc['0':, features[2]].values

femalestestg = femalestest_data.loc['0':, features[0]].values

femalestesth = femalestest_data.loc['0':, features[1]].values 

femalestestw = femalestest_data.loc['0':, features[2]].values
test_cell=11

# Finding Means

males_hm=st.mean(malestrainh)

males_wm=st.mean(malestrainw)

females_hm=st.mean(femalestrainh)

females_wm=st.mean(femalestrainw)

# Finding Variance

males_hv=st.variance(malestrainh)

males_wv=st.variance(malestrainw)

females_hv=st.variance(femalestrainh)

females_wv=st.variance(femalestrainw)

# Calculating Exponents

mheight_ex=-((testh[test_cell]-males_hm)**2)/(2*males_hv)

mweight_ex=-((testw[test_cell]-males_wm)**2)/(2*males_wv)

fheight_ex=-((testh[test_cell]-females_hm)**2)/(2*females_hv)

fweight_ex=-((testw[test_cell]-females_wm)**2)/(2*females_wv)
# Finding Probabilities

p_males=0.5

p_females=0.5

pmales_height=1/((2*3.14*males_hv)**.5)*math.exp(mheight_ex)

pmales_weight=1/((2*3.14*males_wv)**.5)*math.exp(mweight_ex)

pfemales_height=1/((2*3.14*females_hv)**.5)*math.exp(fheight_ex)

pfemales_weight=1/((2*3.14*males_wv)**.5)*math.exp(fweight_ex)
# Predicting

Males_probability = p_males * pmales_height * pmales_weight

Females_probability = p_females * pfemales_height * pfemales_weight

print('Males probability = ', Males_probability)

print('Females probability = ', Females_probability)

if Females_probability>Males_probability:

    print("Prediction = Female")

elif Males_probability>Females_probability:

    print("Prediction = Male")

else:

    print("Ambigious")
