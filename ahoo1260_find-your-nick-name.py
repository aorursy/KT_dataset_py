# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

import operator

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
national_df= pd.read_csv("../input/us-baby-names/NationalNames.csv")
target_gender='F'
target_year=1989
target_name='Parva'
year_gender_matched_df=national_df[(national_df['Gender']==target_gender) & (national_df['Year']==target_year)]

year_gender_matched_df.head(20)['Name']

labels=year_gender_matched_df.head(20)['Name']

sizes=year_gender_matched_df.head(20)['Count']

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
new_columns= ['Name','Count', 'Similarity']



new_df=pd.DataFrame(columns=new_columns)



for index, row in year_gender_matched_df.iterrows():

    dist=nltk.edit_distance(target_name, row['Name'])

    temp_df=pd.DataFrame([[ row['Name'], row['Count'],1/dist]], columns=new_columns)

    new_df=pd.concat([temp_df,new_df])
new_df
new_df=new_df.sort_values(by=['Similarity','Count'],ascending=False)
new_df.head(20)['Name']
labels=new_df.head(20)['Name']

sizes=new_df.head(20)['Count']

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
target_name[0]
matching_with_start_char_df=new_df[new_df['Name'].str.match(target_name[0])]



        
matching_with_start_char_df
list(matching_with_start_char_df['Name'])
plt.bar(matching_with_start_char_df['Name'],matching_with_start_char_df['Count'])
matching_with_start_char_df=matching_with_start_char_df[matching_with_start_char_df['Count']>500]
plt.bar(matching_with_start_char_df['Name'],matching_with_start_char_df['Count'])
x = list(matching_with_start_char_df['Similarity'])

y = list(matching_with_start_char_df['Count'])

plt.scatter(x, y)



labels=list(matching_with_start_char_df['Name'])

for i, txt in enumerate(labels):

    plt.annotate(txt, (x[i], y[i]))

# plt.rcParams["figure.figsize"] = (20,4)

plt.xlabel("similarity")

plt.ylabel("popularity")