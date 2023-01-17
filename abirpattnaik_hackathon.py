# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pprint as pp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

master_data = pd.read_csv('../input/Suicides in India 2001-2012.csv')

master_data.head()

# Any results you write to the current directory are saved as output.
master_data.isnull().sum()

#no null values present in the dataset



obj_type_variables = [column for column in master_data.columns if master_data[column].dtype in ['object']]

print(obj_type_variables)

for column in obj_type_variables:

    master_data[column] = master_data[column].astype('category')

master_data.info()



#import sklearn

#from sklearn.cluster import KMeans

#kmeans=KMeans(n_clusters=5)

#kmeans.fit(master_data)

#labels = kmeans.predict(master_data)

#print(labels)

#centroids = kmeans.cluster_centers_

#print(centroids)
from kmodes.kmodes import KModes

#Cleaning data as required

suicides_data=master_data

suicides_data = suicides_data.drop(suicides_data[suicides_data.Total == 0].index)



suicides_data=suicides_data.drop(['Type'], axis=1)

suicides_data=suicides_data.drop(['Type_code'], axis=1)



suicides_data.head()

#Selecting values with Age Group as 0-100+ only

suicides_data_selected_by_age=suicides_data.loc[suicides_data['Age_group']=="0-100+"]



suicides_data_selected_by_age = suicides_data_selected_by_age.drop(suicides_data_selected_by_age[suicides_data_selected_by_age.State.isin(

                                                                                                            ['Total (Uts)', 

                                                                                                             'Total (States)',

                                                                                                             'Total (All India)'])].index)

suicides_data_selected_by_age.head(20)

suicide_list = suicides_data_selected_by_age.values.tolist()

suicide_list = sorted(suicide_list, key=lambda x: x[0])

suicide_list = sorted(suicide_list, key=lambda x: x[2])
#State VS Gender Data

count = 0

final_list = list()

total_sum = 0 

for i in range(len(suicide_list)):

    if i!=0:

        if suicide_list[i-1][0] == suicide_list[i][0] and suicide_list[i-1][2] == suicide_list[i][2] and i != len(suicide_list)-1:

            total_sum+=suicide_list[i][4] 

        elif suicide_list[i-1][0] == suicide_list[i][0] and suicide_list[i-1][2] == suicide_list[i][2] and i == len(suicide_list)-1:

            total_sum+=suicide_list[i][4]

            final_list.append([suicide_list[i][0], suicide_list[i][2], total_sum])    

        else:

            final_list.append([suicide_list[i-1][0], suicide_list[i-1][2], total_sum])

            total_sum = 0

    

    else:

        total_sum+=suicide_list[i][4]          

final_list = sorted(final_list, key=lambda x: x[0])

#pp.pprint(final_list)
#calculating percentage and finding top 10 states vs genders based on percentage suicide

total_sum=master_data[master_data['State'] == 'Total (All India)'].Total.sum()

print("Total Number of Suicides:", total_sum)

final_list1 = list()

for row in final_list:

    if row[-1]!=((row[2]/total_sum)*100):

        row.append((row[2]/total_sum)*100)

    final_list1.append(row)

print("Top Regions w.r.t. Gender for Suicide Numbers")    

top_ten = sorted(final_list1, key = lambda x:x[3], reverse=True)[:11]

pp.pprint(top_ten)
###Since we have identified top states --> we are planning to drill down more to find out what are the core causes of this state and how 

### can we go about it.

States=['Maharashtra','Andhra Pradesh','Tamil Nadu','West Bengal','Karnataka','Kerala','Madhya Pradesh']

suicide_by_state = master_data

suicide_by_state = suicide_by_state.drop(suicide_by_state[suicide_by_state.Total == 0].index)

#print(suicide_by_state['State'])

suicide_by_state = suicide_by_state[suicide_by_state['State'].isin(States)]
#States vs Causes



suicide_by_state_causes = suicide_by_state[suicide_by_state['Type_code']=="Causes"]

suicide_by_state_causes = suicide_by_state_causes.drop(['Type_code', 'Age_group'], axis=1)

suicide_by_state_causes.head(15)
#Creating 2 different data frames 1. on the basis of year

#2.On the basis of Type



suicide_list = suicide_by_state_causes.values.tolist()

suicide_list = sorted(suicide_list, key=lambda x: x[0])

suicide_list = sorted(suicide_list, key=lambda x: x[2])

count = 0

final_list = list()

total_sum = 0 

for i in range(len(suicide_list)):

    if i!=0:

        if suicide_list[i-1][0] == suicide_list[i][0] and suicide_list[i-1][2] == suicide_list[i][2] and i != len(suicide_list)-1:

            total_sum+=suicide_list[i][4] 

        elif suicide_list[i-1][0] == suicide_list[i][0] and suicide_list[i-1][2] == suicide_list[i][2] and i == len(suicide_list)-1:

            total_sum+=suicide_list[i][4]

            final_list.append([suicide_list[i][0], suicide_list[i][2], total_sum])    

        else:

            final_list.append([suicide_list[i-1][0], suicide_list[i-1][2], total_sum])

            total_sum = 0

    

    else:

        total_sum+=suicide_list[i][4]  

        

        

final_list = sorted(final_list, key=lambda x: x[0])

pp.pprint(final_list)
#Calculating Totals, state wise

def calculate_total_suicide_per_state(my_list, state):

    total_sum = 0

    for i in range(len(my_list)):

        if my_list[i][0] == state:

            total_sum+=my_list[i][2]

    return total_sum



pop_dict = dict()



for state in States:

    value = calculate_total_suicide_per_state(final_list, state)

    pop_dict[state] = value

print(pop_dict)    
final_list1 = list()

for row in final_list:

    total_sum = pop_dict[row[0]]

    if row[-1]!=((row[2]/total_sum)*100):

        row.append((row[2]/total_sum)*100)

    final_list1.append(row)

print("Top Causes w.r.t. Selected Regions for Suicide Numbers")    

pp.pprint(final_list1)

from pandas import DataFrame

state_cause_df=DataFrame.from_records(final_list1)

#state_cause_df.head()
reason_by_state = sorted(final_list1, key = lambda x:x[0])

state_wise_dict = dict()



for state in States:

    state_wise_dict[state] = list()

    for row in final_list1:

        if row[0] == state:

            state_wise_dict[state].append(row)



pp.pprint(state_wise_dict)
#Top 3 Reasons per State



def top_three_reasons(my_list):

    top_three = sorted(my_list, key = lambda x:x[3], reverse=True)[:3]

    return top_three



for state in States:

    pp.pprint(top_three_reasons(state_wise_dict[state]))
#State vs Gender Unsupervised Learning

km = KModes(n_clusters=7, init='Huang', n_init=5, verbose=1)

#

clusters = km.fit_predict(suicides_data)



#Print the cluster centroids

print(km.cluster_centroids_)
#State vs Gender Unsupervised Learning

km1 = KModes(n_clusters=10, init='Huang', n_init=10, verbose=1)

#



clusters = km1.fit_predict(suicide_by_state_causes)



#Print the cluster centroids

print(km1.cluster_centroids_)
#State vs Gender Unsupervised Learning

km1 = KModes(n_clusters=10, init='Huang', n_init=10, verbose=1)

#



clusters = km1.fit_predict(state_cause_df)



#Print the cluster centroids

print(km1.cluster_centroids_)
total_sum=master_data[master_data['State'] == 'Total (All India)'].Total.sum()

print(total_sum)

master_data['Chance %']=(master_data.Total/total_sum)*100

master_data.describe()





#total_data.head()

#list(master_data.groupby(['State'],as_index=False))

#suicides_data = suicides_data.drop(suicides_data[suicides_data.Total == 0].index)



#master_data['Chance']=master_data.Total/
#Top 3 reasons for Selected States

#from sklearn.decomposition import FactorAnalysis

##X, _ = load_digits(return_X_y=True)

#transformer = FactorAnalysis(n_components=7, random_state=0)

#X_transformed = transformer.fit_transform(master_data[['Total','Chance %']])

#X_transformed.shape

'''

# Label ncoder

import keras

from keras.models import Sequential

from keras.layers import Dense

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 8))

# Adding the second hidden layer

classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))



# Compiling Neural Network

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



y_train=master_data['Total']

x_train=master_data.drop(['Total'],axis=1)

#df.drop(['B', 'C'], axis=1)

classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)

'''