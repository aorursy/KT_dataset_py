# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



#Read in the dataset

full_dataset = pd.read_csv("/kaggle/input/pew-research-american-trends-technology-2018/Pew ATP 2018.csv")

full_dataset
#CLEAN DATA



#get dataframe with only responses to question "How much power and influence do you think each of the following have in today’s economy?"

df1 =full_dataset.iloc[ : , 6:13 ] 



#simplify column names

column_names=['Technology companies', 'The energy industry', 'Labor unions', 'Banks and other financial institutions', 'Advertisers', 'Pharmaceutical companies', 'The farming and agriculture industry']

df1.columns =column_names



#define row names

row_names=['Too much power and influence','About the right amount', 'Not enough power and influence', 'Refused'] #row names



#Question we are looking at, title of graph

question="How much power and influence do you think each of the following have in today’s economy?"



#CREATE NEW DATAFRAME TO CONTAIN SUMMARY DATA (in percentages)

summary_df = pd.DataFrame(columns = column_names, index=row_names) #create empty dataframe



#iterate over each column in main dataframe and put the summary values into summary dataframe

for c in column_names:

    for r in row_names:

        #store percentage of total number of people who answered this way

        #normalized parameter stores percentage of total answers in the row rather than raw number

        summary_df.at[r,c]= df1[c].value_counts(normalize=True)[r] 



#rename column for clarity

summary_df= summary_df.rename({"Refused":"Refused to answer"})



#SET UP PLOT

plt=summary_df.T.plot.barh(stacked=True, title=question)#transpose (T) so that it is easier to read with percentages on the x-axis and categories on the y-axis







#FORMAT PLOT

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) #move legend out of the way of data

#format the x-axis labels as percentages rather than raw numbers

import matplotlib.ticker as mtick

plt.xaxis.set_major_formatter(mtick.PercentFormatter(1.0)) 





#print summaries

print("Most Americans think technology companies have too much power.")

print("Institutions >50% of Americans say have \"too much power and influence\":")

for c in column_names:

    if summary_df.at["Too much power and influence",c] > .5:

        print("\t"+c)



print("Institutions >25% of Americans say have \"Not enough power and influence\":")

for c in column_names:

    if summary_df.at["Not enough power and influence",c] > .25:

        print("\t"+c)
#CLEAN DATA



#get dataframe with only responses to question "How much power and influence do you think each of the following have in today’s economy?"

df2 =full_dataset[{"SM8A. Do you think it is acceptable or not acceptable for social media sites to do the following things? Change the look and feel of their site for some users, but not others",

                  "SM8B. Do you think it is acceptable or not acceptable for social media sites to do the following things? Remind some users, but not others, to vote on election day",

                 "SM8C. Do you think it is acceptable or not acceptable for social media sites to do the following things? Show some users, but not others, more of their friends’ happy posts and fewer of their sad posts"}] 



#simplify column names

column_names2=['Change the look and feel of their site for some users, but not others', 

              'Remind some users, but not others, to vote on election day', 

              'Show some users, but not others, more of their friends’ happy posts and fewer of their sad posts']

df2.columns =column_names2



#define answers

answers=['Acceptable','Not acceptable', 'Refused'] #row names



#Question we are looking at, title of graph

question="Do you think it is acceptable or not acceptable for social media sites to do the following things? "



#CREATE NEW DATAFRAME TO CONTAIN SUMMARY DATA (in percentages)

summary_df2 = pd.DataFrame(columns = column_names2, index=answers) #create empty dataframe



#iterate over each column in main dataframe and put the summary values into summary dataframe

for c in column_names2:

    for r in answers:

        #store percentage of total number of people who answered this way

        #normalized parameter stores percentage of total answers in the row rather than raw number

        summary_df2.at[r,c]= df2[c].value_counts(normalize=True)[r] 



#rename column for clarity

summary_df2= summary_df2.rename({"Refused":"Refused to answer"})



#SET UP PLOT

plt2=summary_df2.T.plot.barh(stacked=True, title=question)#transpose (T) so that it is easier to read with percentages on the x-axis and categories on the y-axis



#FORMAT PLOT

plt2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) #move legend out of the way of data

#format the x-axis labels as percentages rather than raw numbers

import matplotlib.ticker as mtick

plt2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0)) 



print("The vast majority of Americans don't think it's acceptable for social media sites to manipulate the emotional content of posts for a select group of users, remind only some users to vote on election day, or change the look and feel for some users but not others (A/B test).")
#CLEAN DATA

#Get new dataframe with only data relevant to influencing or changing your newsfeed

df3 =full_dataset[{"FB3B. Have you ever intentionally tried to influence or change the content you see on your Facebook News Feed?"}] 





#simplify column name

column_name="Have you ever intentionally tried to influence or change the content you see on your Facebook News Feed?"

df3.columns= [column_name]



#Get summary data

summary = df3[column_name].value_counts() #use value_counts built in data to get totals

summary =summary.rename({"Refused":"Refused to answer"}) #rename column for better understanding



#Make Pie Chart

import matplotlib.pyplot as plt

labels = summary.index #labels for chart (i.e. Yes, No, etc)

sizes = summary #values

fig1, ax1 = plt.subplots() #create a figure and a set of subplots

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

plt.title(column_name) #add title

plt.show()



print("About 36% of Americans have tried to influence or change the content they see in their Facebook News Feed.")
#CLEAN DATA

#Get new dataframe with only data relevant to influencing or changing your newsfeed

df4 =full_dataset[{"FB3B. Have you ever intentionally tried to influence or change the content you see on your Facebook News Feed?", 

                   "FB3C1. What actions have you taken to try and influence what shows up in your Facebook News Feed? Friended or unfriended certain people",

                  "FB3C2. What actions have you taken to try and influence what shows up in your Facebook News Feed? Liked, shared or commented on certain types of content",

                   "FB3C3. What actions have you taken to try and influence what shows up in your Facebook News Feed? Indicated that you wanted to see less of certain people or types of content",

                  "FB3C4. What actions have you taken to try and influence what shows up in your Facebook News Feed? Changed your privacy settings or ad preferences",

                  "FB3C5. What actions have you taken to try and influence what shows up in your Facebook News Feed? Followed or unfollowed certain groups or organizations",

                  "FB3C6. What actions have you taken to try and influence what shows up in your Facebook News Feed? Something else"}] 



#Keep only rows where peole answered 'Yes' to first question

df4=df4.loc[df4['FB3B. Have you ever intentionally tried to influence or change the content you see on your Facebook News Feed?'] == 'Yes']

#drop first column, since all values are now Yes

df4=df4.iloc[ : , 1: ] 



#simplify column names

column_names=['Friended or unfriended certain people', 

              'Liked, shared or commented on certain types of content', 

              'Indicated that you wanted to see less of certain people or types of content', 

              'Changed your privacy settings or ad preferences', 

              'Followed or unfollowed certain groups or organizations', 

              'Something else']

df4.columns =column_names



#get the number of total responses in dataset

total_responses=df4.count()[1]



#count the number of people who selected each technique

techniques=pd.Series(index=column_names) #set up an empty dictionary



#iterate over dataset and count up people who said they use that technique

for c in column_names:

    count=0; #set up counter for each column

    for i in df4[c]: #go through a single column

        if i != "Not selected": #if that technique was selected

            count +=1 #increment count

    techniques[c]=count/total_responses #add percentage to dictionary



techniques=techniques.sort_values(ascending=True) #sort from highest to lowest



#plot them on a bar chart

# Build the plot

plt = techniques.plot(kind="barh", title="What actions have you taken to try and influence what shows up in your Facebook News Feed?")



#format the x-axis labels as percentages rather than raw numbers

import matplotlib.ticker as mtick

plt.xaxis.set_major_formatter(mtick.PercentFormatter(1.0)) 
