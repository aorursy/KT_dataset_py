# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#matplotlib inline
matplotlib.rcParams.update({'font.size': 12})

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


def age_to_days(item):
    # convert item to list if it is one string
    if type(item) is str:
        item = [item]
    ages_in_days = np.zeros(len(item))
    for i in range(len(item)):
        # check if item[i] is str
        if type(item[i]) is str:
            if 'day' in item[i]:
                ages_in_days[i] = int(item[i].split(' ')[0])
            if 'week' in item[i]:
                ages_in_days[i] = int(item[i].split(' ')[0])*7
            if 'month' in item[i]:
                ages_in_days[i] = int(item[i].split(' ')[0])*30
            if 'year' in item[i]:
                ages_in_days[i] = int(item[i].split(' ')[0])*365    
        else:
            # item[i] is not a string but a nan
            ages_in_days[i] = 0
    return ages_in_days


df = pd.read_csv('../input/train.csv', sep=',')

feature = 'AgeuponOutcome'
feature_values_dog = np.array(df.loc[df['AnimalType'] == 'Dog',feature])
outcome_dog = np.array(df.loc[df['AnimalType'] == 'Dog','OutcomeType'])

feature_values_cat = np.array(df.loc[df['AnimalType'] == 'Cat',feature])
outcome_cat = np.array(df.loc[df['AnimalType'] == 'Cat','OutcomeType'])

ages_dog = age_to_days(feature_values_dog)
ages_cat = age_to_days(feature_values_cat)

unique_ages = np.unique(np.append(ages_dog,ages_cat))
unique_outcomes = np.unique(np.append(outcome_dog,outcome_cat))

print(unique_outcomes)

fractions_cat = np.zeros([len(unique_ages),len(unique_outcomes)])
fractions_dog = np.zeros([len(unique_ages),len(unique_outcomes)])
nr_animals_with_age_dog = np.zeros(len(unique_ages))
nr_animals_with_age_cat = np.zeros(len(unique_ages))
for i in range(len(unique_ages)):
    for j in range(len(unique_outcomes)):
        sublist_dog = outcome_dog[ages_dog == unique_ages[i]]  
        if len(sublist_dog) > 0:
            fractions_dog[i,j] = 1e0*len(sublist_dog[sublist_dog == unique_outcomes[j]]) / len(sublist_dog)
        else:
            fractions_dog[i,j] = 0e0
        sublist_cat = outcome_cat[ages_cat == unique_ages[i]]        
        fractions_cat[i,j] = 1e0*len(sublist_cat[sublist_cat == unique_outcomes[j]]) / len(sublist_cat)
        
    nr_animals_with_age_dog[i] = len(sublist_dog)
    nr_animals_with_age_cat[i] = len(sublist_cat)
# nr of animals vs age
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.title('Dog')
plt.plot(unique_ages,nr_animals_with_age_dog,'+',markersize=10,mew=2)
plt.plot(unique_ages,nr_animals_with_age_dog)
plt.xlim([0.7,1e4])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('age [days]')
plt.ylabel('number of animals in train.csv')
plt.tight_layout(w_pad=0, h_pad=0)

plt.subplot(1, 2, 2)
plt.title('Cat')
plt.plot(unique_ages,nr_animals_with_age_cat,'+',markersize=10,mew=2)
plt.plot(unique_ages,nr_animals_with_age_cat)
plt.xlim([0.7,1e4])
plt.xscale('log')
plt.yscale('log')
plt.xlabel('age [days]')
plt.ylabel('number of animals in train.csv')
plt.tight_layout(w_pad=0, h_pad=0)
plt.savefig('age-vs-nr_points.jpg',dpi=150)
plt.show()
plt.close()
# fraction of outcomes

ages_for_axis = np.append(unique_ages,age_to_days('20 years'))

left = (ages_for_axis[1:-1] + ages_for_axis[:-2])/2e0
right = (ages_for_axis[1:-1] + ages_for_axis[2:])/2e0
width = right-left

plt.figure(figsize=(10,4))

plt.subplot(1, 2, 1)
plt.title('Dog')
plt.xlabel('age [days]')
plt.ylabel('fraction outcomes')
plt.xscale('log')
plt.xlim([0.7,1e4])
plt1 = plt.bar(left, fractions_dog[1:,0], width,color='#5A8F29',edgecolor='none')
plt2 = plt.bar(left, fractions_dog[1:,1], width,color='k',bottom = np.sum(fractions_dog[1:,:1],axis=1),edgecolor='none')
plt3 = plt.bar(left, fractions_dog[1:,2], width,color='#FF8F00',bottom = np.sum(fractions_dog[1:,:2],axis=1),edgecolor='none')
plt4 = plt.bar(left, fractions_dog[1:,3], width,color='#FFF5EE',bottom = np.sum(fractions_dog[1:,:3],axis=1),edgecolor='none')
plt5 = plt.bar(left, fractions_dog[1:,4], width,color='#3C7DC4',bottom = np.sum(fractions_dog[1:,:4],axis=1),edgecolor='none')
plt.legend([plt1,plt2,plt3,plt4,plt5],unique_outcomes,loc=2,fontsize=10)
plt.tight_layout(w_pad=0, h_pad=0)
plt.tick_params(axis='x', length=6, which='major',width=2)
plt.tick_params(axis='x', length=4, which='minor',width=1)
plt.minorticks_on()
plt.tight_layout(w_pad=0, h_pad=0)

plt.subplot(1, 2, 2)
plt.title('Cat')
plt.xlabel('age [days]')
plt.ylabel('fraction outcomes')
plt.xscale('log')
plt.xlim([0.7,1e4])
plt1 = plt.bar(left, fractions_cat[1:,0], width,color='#5A8F29',edgecolor='none')
plt2 = plt.bar(left, fractions_cat[1:,1], width,color='k',bottom = np.sum(fractions_cat[1:,:1],axis=1),edgecolor='none')
plt3 = plt.bar(left, fractions_cat[1:,2], width,color='#FF8F00',bottom = np.sum(fractions_cat[1:,:2],axis=1),edgecolor='none')
plt4 = plt.bar(left, fractions_cat[1:,3], width,color='#FFF5EE',bottom = np.sum(fractions_cat[1:,:3],axis=1),edgecolor='none')
plt5 = plt.bar(left, fractions_cat[1:,4], width,color='#3C7DC4',bottom = np.sum(fractions_cat[1:,:4],axis=1),edgecolor='none')
plt.legend([plt1,plt2,plt3,plt4,plt5],unique_outcomes,loc=2,fontsize=10)
plt.tight_layout(w_pad=0, h_pad=0)
plt.tick_params(axis='x', length=6, which='major',width=2)
plt.tick_params(axis='x', length=4, which='minor',width=1)
plt.minorticks_on()
plt.tight_layout(w_pad=0, h_pad=0)
plt.savefig('age-vs-outcome.jpg',dpi=150)
plt.show()
plt.close()

feature = 'SexuponOutcome'

feature_values_dog = np.array(df.loc[df['AnimalType'] == 'Dog',feature])
outcome_dog = np.array(df.loc[df['AnimalType'] == 'Dog','OutcomeType'])

feature_values_cat = np.array(df.loc[df['AnimalType'] == 'Cat',feature])
outcome_cat = np.array(df.loc[df['AnimalType'] == 'Cat','OutcomeType'])

unique_sexes = np.unique(feature_values_cat)
unique_outcomes = np.unique(np.append(outcome_dog,outcome_cat))

fractions_cat = np.zeros([len(unique_sexes),len(unique_outcomes)])
fractions_dog = np.zeros([len(unique_sexes),len(unique_outcomes)])
nr_animals_with_sex_dog = np.zeros(len(unique_sexes))
nr_animals_with_sex_cat = np.zeros(len(unique_sexes))

for i in range(len(unique_sexes)):
    for j in range(len(unique_outcomes)):
        sublist_dog = outcome_dog[feature_values_dog == unique_sexes[i]]  
        if len(sublist_dog) > 0:
            fractions_dog[i,j] = 1e0*len(sublist_dog[sublist_dog == unique_outcomes[j]]) / len(sublist_dog)
        else:
            fractions_dog[i,j] = 0e0
        sublist_cat = outcome_cat[feature_values_cat == unique_sexes[i]]        
        fractions_cat[i,j] = 1e0*len(sublist_cat[sublist_cat == unique_outcomes[j]]) / len(sublist_cat)
        
    nr_animals_with_sex_dog[i] = len(sublist_dog)
    nr_animals_with_sex_cat[i] = len(sublist_cat)
    
