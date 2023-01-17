import pandas as pd

import os

import seaborn as sns
data_original = os.listdir('../input/the-car-connection-picture-dataset') #
#Finds index 

def FindIndexInString(string,desired_counter):

    counter = 0

    from_idx = 0

    to_idx = 0

    for idx, char in enumerate(string): #Goes through each char in string

        if char == '_':

            counter += 1

            if counter == desired_counter-1: #Finds the from_idx, to go from example: "this'_'is_a_string"

                from_idx = idx+1

        if counter == desired_counter: #Finds the to_idx "this_is'_'a_string"

            to_idx = idx

            return from_idx, to_idx
#Creates dataframe based on input data

def CreateDataFrame(data):

    df = pd.DataFrame(columns=['Manufacturer','Series','Year'])

    

    for i,column in enumerate(df):

        string_list = []

        for idx, string in enumerate(data):

            from_idx,to_idx = FindIndexInString(string,i+1)

            string = string[from_idx:to_idx]

            string_list.append(string) 

        df[column] = string_list

        

    image_df = pd.DataFrame(data,columns=['Image'])

    result_df = pd.concat([df,image_df],axis=1)

    return result_df
df = CreateDataFrame(data_original)

df
sns.set(rc={'figure.figsize':(30,10)})

sns.set(font_scale = 0.6)

sns.countplot(x='Manufacturer',data=df)
sns.countplot(x='Year',data=df)