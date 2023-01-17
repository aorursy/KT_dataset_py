%matplotlib inline

## Imports



# pandas, numpy

import pandas as pan

import numpy as np



# Gaussian Naive Bayes

from sklearn import datasets

from sklearn import metrics

from sklearn.svm import LinearSVC



# Matplotlib, seaborn

import matplotlib.pyplot as plt

import seaborn as sns



# Missingno

import missingno as mno
# Function to automap when there are a lot of column to map

def auto_df_mapping(dataframe_column):

    

    # Get the unique values (the NaN must be dealt before the call of this function)

    list_unique_value = dataframe_column.unique()

    

    # Initialize the increment

    i = 0

    

    # Initialise the dictionnary who'll be used to map the df

    dictionnary = {}

    

    # Loop the value and map it to i

    for value in list_unique_value:

        dictionnary[value] = i

        i = i + 1

        

    #print("Dictionnary : ", dictionnary)

    return dataframe_column.map(dictionnary)



# Load the train data

dataframe = pan.DataFrame(pan.read_csv('../input/mushrooms.csv'))



# Check missing data (all good !)

mno.bar(df=dataframe, figsize=(12, 5), color=(255/255, 83/255, 51/255), inline=True)



# Automatically map all the dataset (lazy mode)

for column in dataframe.columns:

    dataframe[column] = auto_df_mapping(dataframe[column]).astype(int)

    

#print(len(dataframe.index))

#print(dataframe.mean())   
# Look at the correlation between the data (Good correlation between the class and gill size/spacing)

plt.figure(figsize=(23,21))

plt.title('Pearson Correlation of Features on a heatmap', y=1.05, size=15)

sns.heatmap(dataframe.corr(),linewidths=0.1,vmax=1.0, square=True, linecolor='grey', annot=True, cmap="YlGnBu")



# We can see there is only one type of value in veil-type so we can put it out

dataframe = dataframe.drop('veil-type', axis=1)
## Model choice and train



# We cut the given dataframe to ~90% train dataframe and ~10% to verify the accuracy model

train_dataframe = dataframe.iloc[:6500]

test_dataframe =  dataframe.iloc[6500:]



# Cut the train result columns

train_target_dataframe = train_dataframe['class']

train_main_dataframe = train_dataframe.drop('class', axis=1)



# LinearSVC

clf = LinearSVC()

clf.fit(train_main_dataframe, train_target_dataframe)



# Cut the test result columns

test_target_dataframe = test_dataframe['class']

test_main_dataframe = test_dataframe.drop('class', axis=1)



# predict

predicted = clf.predict(test_main_dataframe)



# Check between the expected and predicted result

i=0

for index, row in pan.DataFrame({'expected':test_target_dataframe, 'predicted':predicted}).iterrows():

    if row['expected'] == row['predicted']:

        i+=1      

print("Accuracy :", (i/len(test_main_dataframe.index)) * 100, "%")