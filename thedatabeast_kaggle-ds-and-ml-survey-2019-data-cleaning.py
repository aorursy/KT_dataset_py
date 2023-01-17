# Loading necessary libraries



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
# Load the relevant datasets



mcr = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv") # responses

ques = pd.read_csv("../input/kaggle-survey-2019/questions_only.csv") # questions

txtr = pd.read_csv("../input/kaggle-survey-2019/other_text_responses.csv") # text responses

ss = pd.read_csv("../input/kaggle-survey-2019/survey_schema.csv")
# Inspect mcr using head()



mcr.head()
# Function to print the contents of all cells in a given row of a dataframe in their entire length



def print_all(df, r_num):

    """

    Task:

    To print all the content of the cells in a given row without truncation

    

    Input :

    df <- Dataframe

    r_num <- Row number (starts at 0)

    

    Output :

    Prints out all values for the input row across all attributes with corresponding headers

    """

    row = df.iloc[r_num]

    pd.set_option('display.max_rows', len(row))

    pd.set_option('display.max_colwidth', -1)

    print(row)

    pd.reset_option('display.max_rows')
# Inspecting Multiple Choice Responses (We need Questions, so that's why we are passing 0 as the row number)

print_all(mcr,0)
# Function to remove columns from a dataset based on an expression in the column name



def col_rem_by_exp(df, exp):

    """

    Task:

    To remove columns from a dataset based on an expression in the column name

    

    Input :

    df <- Dataframe

    exp <- The string expression that is common to all columns that need to be removed

    

    Output :

    Returns a dataframe with the removed columns

    """

    removable_cols = []

    for i in df.columns:

        if (exp in i):

            removable_cols.append(i)

    return (df.drop(removable_cols, axis=1))
# Removing "OTHER_TEXT" columns



mcr = col_rem_by_exp(mcr, "_OTHER_TEXT")

print_all(mcr,0)
# Function to segregate all the questions that have the "Select all that apply" option

def select_features(df, exp):

    """

    Task:

    To group columns from a dataset based on an expression in the column name

    

    Input :

    df <- Dataframe

    exp <- The string expression that is common to all columns that need to be aggregated

    

    Output :

    Returns the list of all features with the common expression

    """

    feature_list = []

    for i in mcr.columns:

        if ("_Part_" in i):

            q = i

            pos_ = q.index('_')

            q_no = int(q[:pos_][1:])

            if(q_no not in feature_list):

                feature_list.append(q_no)

    return feature_list

select_all_ques = select_features(mcr, "_Part_")

print('"Select all that apply" questions :')

print(select_all_ques)
def response_combine(df, q_num):

    """

    Task:

    To combine responses of "Select all that apply" questions

    

    Input :

    df <- Multiple choice response survey

    q_num <- Question number whose responses need to be combined

    

    Output :

    > List of lists...each list corresponds to a row and all the options selected by that respondent are grouped together in it

    > Leave out the first list (it just groups the headers) once you get the output

    """

    # Identify the PARTS of the given question number

    resp_cols = []

    for i in df.columns:

        if (('Q'+str(q_num)) in i):

            resp_cols.append(i)

            

    # Aggregate all the responses of a given respondent

    responses = []

    for i in range(df.shape[0]):

        l = list(df[resp_cols].iloc[i])

        cleaned_responses = [choice for choice in l if str(choice) != 'nan']

        responses.append(cleaned_responses)

    

    # Create a dataframe of these aggregated responses, merge them with the original dataframe and delete the PARTS

    header = ("Q"+str(q_num))

    temp_df = pd.DataFrame(dict({header:responses}))

    df = df.drop(resp_cols, axis=1)

    final_df = pd.concat([df, temp_df], axis=1, sort=False)

    

    return (final_df)
# Cleaning the complete dataframe

clean_mcr = response_combine(mcr,select_all_ques[0])

for q in select_all_ques[1:]:

    clean_mcr = response_combine(clean_mcr,q)

    

print("The shape of the cleaned dataframe is :",clean_mcr.shape)
"""Fixing the Column Headers"""



# list of all questions whose positions have changed after response_combine()

pos_changed_ques = [("Q"+str(x)) for x in select_all_ques]



# dropping the position-changed-questions from the main dataframe

questions = list(ques.loc[0,pos_changed_ques])

new_ques = ques.drop(pos_changed_ques, axis=1)



# using the concept of dataframes concatenation to create "new_ques" from "ques"

## new_ques is a modified version of ques that orders questions like how they have been modified in clean_mcr

temp_dict = {}

for i in range(len(pos_changed_ques)):

    temp_dict[pos_changed_ques[i]] = questions[i]

temp_ques = pd.DataFrame(temp_dict, index=[0])

new_ques = pd.concat([new_ques,temp_ques], axis=1)



# new_ques

new_ques
""" Final Clean Up """



# Rename columns

clean_mcr.columns = list(new_ques.iloc[0,:])



# Drop the first row

clean_mcr = clean_mcr.drop([0])



# Drop the first column (it's not needed)

# clean_mcr = clean_mcr.drop(clean_mcr.columns[[0]], axis=1)



# Save clean_mcr as an output dataset

clean_mcr.to_csv("clean_multiple_choice_responses.csv")
