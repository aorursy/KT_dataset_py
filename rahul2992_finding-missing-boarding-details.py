import pandas as pd

import numpy as np

import re 

import itertools 

import collections 

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import train_test_split
file = '../input/train.csv'

file_test = '../input/test.csv'

#Import data as a dataframe

traindata = pd.read_csv(file) 

test = pd.read_csv(file_test)

print (traindata.head(2))
#Convert to numpy array 

name_num= traindata["Name"].values

cabin_num = traindata["Cabin"].values



name_test = test["Name"].values



#Convert to list 

name_lists = name_num.tolist()

cabin_lists = cabin_num.tolist()

name_test = name_test.tolist()
#Regular expression to parse Surname

regex = r'([a-zA-Z]+),'

#Regular expression to parse salutation

regex_sal = r', ([a-zA-Z]+).'

#Regular expression to parse deck number

regex_deck = r'([A-Z])[0-9]'



#Initiate variables

surname1 = [] 

surname = []

salutation1 = []

salutation = []

deck = []

deck1 = []

salutation1t = []



#Extract traindata

for name in name_lists:

	temp = re.findall(regex, name)

	temp1 = re.findall(regex_sal, name)

	surname1.append(temp)

	salutation1.append(temp1)



#Extract test data

for name in name_test:

	temp1 = re.findall(regex_sal, name)

	salutation1t.append(temp1)



temp = ['AA']



#Extract cabin details from traindata

for cabin in cabin_lists:

	if type(cabin) == float:

		deck1.append(temp)

	else:

		temp2 = re.findall(regex_deck, cabin)

		deck1.append(temp2)



#Convert list of lists to lists. This allows us to parse the data easily

surname = list(itertools.chain.from_iterable(surname1)) 

salutation = list(itertools.chain.from_iterable(salutation1))

deck = list(itertools.chain.from_iterable(deck1))



salutationt = list(itertools.chain.from_iterable(salutation1t))



#Dataframes for surname & salutation created

surname_df = pd.DataFrame(surname)

surname_df.columns = ['surname'] 

salutation_df = pd.DataFrame(salutation)

salutation_df.columns = ['Salutation']

deck_df = pd.DataFrame(deck)

deck_df.columns = ['Deck']

family_size = traindata['SibSp'] + traindata['Parch'] + 1

family_size.columns = ['Family_size']



salutationt_df = pd.DataFrame(salutationt)

salutationt_df.columns = ['Salutation']

#Creating new features of family size

family_size_t = test['SibSp'] + traindata['Parch'] + 1

family_size_t.columns = ['Family_size']
#Appended traindata with a column for surnames

traindata = traindata.assign(surname = surname_df)

traindata = traindata.assign(Salutation = salutation_df)

traindata = traindata.assign(Family_size = family_size)

traindata = traindata.assign(Deck = deck_df)



print (traindata.head(5))



#Append test data with a column for salutation and family size

test = test.assign(Salutation = salutationt_df)

test = test.assign(Family_size = family_size_t)
print (traindata.isnull().any())

traindata_tmp = traindata[(traindata["Fare"]> 70) & (traindata["Fare"]<90.0)]

traindata_tmp.groupby(["Embarked"]).size()



#Create dataframes for C1,C2 ... 

df_C1 = traindata[(traindata['Embarked'] == "C") & (traindata["Pclass"] == 1)]

df_C2 = traindata[(traindata['Embarked'] == "C") & (traindata["Pclass"] == 2)]

df_C3 = traindata[(traindata['Embarked'] == "C") & (traindata["Pclass"] == 3)]



df_S1 = traindata[(traindata['Embarked'] == "S") & (traindata["Pclass"] == 1)]

df_S2= traindata[(traindata['Embarked'] == "S") & (traindata["Pclass"] == 2)]

df_S3 = traindata[(traindata['Embarked'] == "S") & (traindata["Pclass"] == 3)]



df_Q1 = traindata[(traindata['Embarked'] == "Q") & (traindata["Pclass"] == 1)]

df_Q2= traindata[(traindata['Embarked'] == "Q") & (traindata["Pclass"] == 2)]

df_Q3 = traindata[(traindata['Embarked'] == "Q") & (traindata["Pclass"] == 3)]



#To create a boxplot, we need to enter lists or numpy arrays. Making the required conversion

lst_C1 = df_C1["Fare"].values.tolist()

lst_C2 = df_C2["Fare"].values.tolist()

lst_C3 = df_C3["Fare"].values.tolist()



lst_S1 = df_S1["Fare"].values.tolist()

lst_S2 = df_S2["Fare"].values.tolist()

lst_S3 = df_S3["Fare"].values.tolist()



lst_Q1 = df_Q1["Fare"].values.tolist()

lst_Q2 = df_Q2["Fare"].values.tolist()

lst_Q3 = df_Q3["Fare"].values.tolist()



#Plotting the data in box plots

data_to_plot = [lst_C1, lst_C2, lst_C3, lst_S1, lst_S2, lst_S3, lst_Q1, lst_Q2, lst_Q3]

#print data_to_plot

fig = plt.figure(1, figsize = (9,9))

ax = fig.add_subplot(111)

bp = ax.boxplot(data_to_plot, patch_artist = True, showfliers = False)



#Format the boxplot

for box in bp['boxes']:

	box.set(color = '#7570b3', linewidth = 2)

	box.set(facecolor = '#1b9e77')



for whisker in bp['whiskers']:

	whisker.set(color = '#7570b3', linewidth = 2)

	

for cap in bp['caps']:

	cap.set(color = '#7570b3', linewidth =2)

	

## change color and linewidth of the medians

for median in bp['medians']:

    median.set(color='#b2df8a', linewidth=2)



## change the style of fliers and their fill

for flier in bp['fliers']:

    flier.set(marker='o', color='#e7298a', alpha=0.5)



ax.set_xticklabels(['C1', 'C2', 'C3', 'S1', 'S2', 'S3', 'Q1', 'Q2', 'Q3'])



plt.show()

#Replacing missing values in Embarked 

traindata.loc[traindata['Embarked'].isnull(), 'Embarked'] = "C"



test.loc[test['Embarked'].isnull(), 'Embarked'] = "C"



print (traindata.isnull().any())