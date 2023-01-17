import numpy as np # linear algebra
import pandas as pd # data processing 
import seaborn as sns # to plot good graphs
import matplotlib.pyplot as plt # to render our plots and set some parameters of the box
import squarify 

path = "../input/" # seting the Kaggle path 

#Importing the datasets
# First I will work with a multiple Choice Responses that I think is more insightful 
df_multiChoice = pd.read_csv(path + 'multipleChoiceResponses.csv', low_memory=False, header=[0,1])
#df_freedom = pd.read_csv(path + 'freeFormResponses.csv', low_memory=False, header=[0,1])
# Formating the DF
df_multiChoice.columns = df_multiChoice.columns.map('_'.join)

# Printing the data shape
print('The Kaggle Survey has {} rows and {} columns.'.format(df_multiChoice.shape[0], df_multiChoice.shape[1]))
# seting the function to show 
def knowningData(df, data_type=object, limit=3): #seting the function with df, 
    n = df.select_dtypes(include=data_type) #selecting the desired data type
    for column in n.columns: #initializing the loop
        print("#########################################################")
        print("Name of column ", column, ': \n', "Uniques: ", df[column].unique()[:limit], "\n",
              " | ## Total nulls: ", (round(df[column].isnull().sum() / len(df[column]) * 100,2)),
              " | ## Total unique values: ", df.nunique()[column]) #print the data and % of nulls)
        print("Percentual of top 5 of: ", column)
        print(round(df[column].value_counts()[:5] / df[column].value_counts().sum() * 100,2))
# Calling 
knowningData(df_multiChoice)
#Seting size to figure that we will plot
plt.figure(figsize=(10,5))

sns.countplot(df_multiChoice["Q1_What is your gender? - Selected Choice"]) #ploting the first Question
plt.title("Question 1 - What is your gender?", fontsize=20) # Adding Title and seting the size
plt.xlabel("Genders in Survey", fontsize=16) # Adding x label and seting the size
plt.ylabel("Count", fontsize=16) # Adding y label and seting the size

# printing the descriptive values in % of respondents

print("Question 1 - What is your gender? ")
print(round(df_multiChoice['Q1_What is your gender? - Selected Choice'].value_counts() \
            / len(df_multiChoice['Q1_What is your gender? - Selected Choice']) * 100,2))

#rendering the graph
plt.show()
#Seting size to figure that we will plot
plt.figure(figsize=(10,5))

sns.countplot(df_multiChoice["Q2_What is your age (# years)?"], # ploting the second question
              order=['18-21', '22-24', '25-29', '30-34', '35-39', '40-44', #seting order to our data
                     '45-49', '50-54', '55-59','60-69', '70-79', '80+']) 
plt.title("Question 2 - What is your age (# years)", fontsize=20) # Adding Title and seting the size
plt.xlabel("Age Range", fontsize=16) # Adding x label and seting the size
plt.ylabel("Count", fontsize=16) # Adding y label and seting the size

#rendering the plot
plt.show()
#Doing some treatment to the data, to a best visual when we plot it
# Modifying some names to a short 
df_multiChoice.loc[df_multiChoice["Q3_In which country do you currently reside?"] == 
                   'United Kingdom of Great Britain and Northern Ireland', "Q3_In which country do you currently reside?"] = "UK & N. Ireland"
df_multiChoice.loc[df_multiChoice["Q3_In which country do you currently reside?"] == 
                   'United States of America', "Q3_In which country do you currently reside?"] = "USA"
df_multiChoice.loc[df_multiChoice["Q3_In which country do you currently reside?"] == 
                   'I do not wish to disclose my location', "Q3_In which country do you currently reside?"] = "Not set"

# Doing the count of the columns and liming it to 15
countrys = df_multiChoice['Q3_In which country do you currently reside?'].value_counts()

#Seting size to figure that we will plot
plt.figure(figsize=(14,6))

# Now I will plot the Countrys 
sns.countplot(df_multiChoice[df_multiChoice["Q3_In which country do you currently reside?"]\
                             .isin(countrys[:15].index.values)]['Q3_In which country do you currently reside?'], 
              order=countrys[:15].index
             )
plt.title("Question 3 - In which country do you currently reside?", fontsize=20) # Adding Title and seting the size
plt.xlabel("Countrys", fontsize=16) # Adding x label and seting the size
plt.ylabel("Count", fontsize=16) # Adding y label and seting the size
plt.xticks(rotation=45) # Adjust the xticks, rotating the labels

plt.show()

crosstab_eda = pd.crosstab(columns=df_multiChoice["Q2_What is your age (# years)?"], 
                           # at this line, I am using the isin to select just the top 5 of browsers
                           index=df_multiChoice[df_multiChoice["Q3_In which country do you currently reside?"]\
                                                .isin(countrys[:15].index.values)]["Q3_In which country do you currently reside?"], normalize= "index")

# Ploting the crosstab that we did above
crosstab_eda.plot(kind="bar",    # select the bar to plot the count of categoricals
                 figsize=(16,7), # adjusting the size of graphs
                 stacked=True)   # code to unstack

plt.title("Age distribuition by Country", fontsize=22) # adjusting title and fontsize
plt.ylabel("Age Distribuition (in %)", fontsize=19) # adjusting y label and fontsize
plt.xlabel("Country Names", fontsize=19) # adjusting x label and fontsize
plt.legend(title="% of time exploring model insights") # 
plt.xticks(rotation=0) # seting the rotation to zero
plt.legend(title="Age Range", # Some parameters to legend of our graphs
          loc='best', bbox_to_anchor=(1, .9), #seting location and bbox anchor
          fancybox=True, shadow=True)
plt.show() # rendering

print("DESCRIPTIVE VALUES OF AGE DISTRIBUITION BY COUNTRYS: ")
country_repayment = ["Q3_In which country do you currently reside?", 
                     "Q2_What is your age (# years)?"] #seting the desired 

cm = sns.light_palette("green", as_cmap=True)
round(pd.crosstab(df_multiChoice[df_multiChoice["Q3_In which country do you currently reside?"]\
                                                .isin(countrys[:25].index.values)][country_repayment[0]],\
                                 df_multiChoice[country_repayment[1]], margins=True, margins_name="Totals",
                 rownames=["Country's"],colnames=["Age of Kagglers"]),2).drop("Totals", axis=0).style.background_gradient(cmap = cm)
# changing the name to get a short name in our crosstab
df_multiChoice.rename(columns={
    "Q46_Approximately what percent of your data projects involve exploring model insights?":
    "Q46_Approx percent involved exploring model insights?"}, 
                      inplace=True)

crosstab_eda = pd.crosstab(columns=df_multiChoice["Q46_Approx percent involved exploring model insights?"], 
                           # at this line, I am using the isin to select just the top 5 of browsers
                           index=df_multiChoice["Q2_What is your age (# years)?"], normalize= "index")

# Ploting the crosstab that we did above
crosstab_eda.plot(kind="bar",    # select the bar to plot the count of categoricals
                 figsize=(16,6), # adjusting the size of graphs
                 stacked=False)   # code to unstack 
plt.title("The percent of your data projects involve exploring model insights by Year of Users", fontsize=22) # adjusting title and fontsize
plt.xlabel("What is the age (# years)", fontsize=19) # adjusting x label and fontsize
plt.ylabel("% of Exploring Model Insights", fontsize=19) # adjusting y label and fontsize
plt.legend(title="% of time exploring model insights") # 
plt.xticks(rotation=0) # seting the rotation to zero

# Some parameters to legend of our graphs
plt.legend(title="% of time exploring model insights", 
          loc='upper right', bbox_to_anchor=(.7, .9),
          ncol=4, fancybox=True, shadow=True)


plt.show() # rendering

country_repayment = ["Q2_What is your age (# years)?",
                     "Q46_Approx percent involved exploring model insights?" ] #seting the desired 

print("DESCRIPTIVE VALUES BY EACH AGE RANGE AND % OF TIME CONSUMING: ")
cm = sns.light_palette("green", as_cmap=True)
round(pd.crosstab(df_multiChoice[country_repayment[0]],\
                                 df_multiChoice[country_repayment[1]], margins=True, margins_name="Totals",
                  rownames=["Age of Kagglers"],colnames=["% time exploring insights"]),2).drop("Totals", axis=0).style.background_gradient(cmap = cm)
crosstab_eda = pd.crosstab(columns=df_multiChoice["Q46_Approx percent involved exploring model insights?"], 
                           # at this line, I am using the isin to select just the top 5 of browsers
                           index=df_multiChoice[df_multiChoice["Q3_In which country do you currently reside?"]\
                                                .isin(countrys[:15].index.values)]["Q3_In which country do you currently reside?"], 
                           normalize= "index")
# Ploting the crosstab that we did above
crosstab_eda.plot(kind="bar",    # select the bar to plot the count of categoricals
                 figsize=(15,7), # adjusting the size of graphs
                 stacked=False)   # code to stack 
plt.title("Time exploring models to insights ", fontsize=22) # adjusting title and fontsize
plt.xlabel("Country Names", fontsize=19) # adjusting x label and fontsize
plt.ylabel("% of Exploring Model Insights", fontsize=19) # adjusting y label and fontsize
plt.xticks(rotation=0) # seting the rotation to zero
# Some parameters to legend of our graphs
plt.legend(title="% of time exploring model insights", 
          loc='upper right', bbox_to_anchor=(.97, 1.02),
          ncol=11, fancybox=True, shadow=False)

plt.show() # rendering

print("DESCRIPTIVE VALUES OF COUNTRYS BY TIME CONSUMING EXPLORING INSIGHTS IN MODELS: ")

country_repayment = ["Q3_In which country do you currently reside?", 
                     "Q46_Approx percent involved exploring model insights?"] #seting the desired 

cm = sns.light_palette("green", as_cmap=True) #seting the colormap of our table
round(pd.crosstab(df_multiChoice[df_multiChoice["Q3_In which country do you currently reside?"]\
                                                .isin(countrys[:25].index.values)][country_repayment[0]],\
                                 df_multiChoice[country_repayment[1]]),2).style.background_gradient(cmap = cm)
crosstab_eda = pd.crosstab(columns=df_multiChoice["Q1_What is your gender? - Selected Choice"], 
                           # at this line, I am using the isin to select just the top 5 of browsers
                           index=df_multiChoice[df_multiChoice["Q3_In which country do you currently reside?"]\
                                                .isin(countrys[:15].index.values)]["Q3_In which country do you currently reside?"], 
                           normalize= "index")

# Ploting the crosstab that we did above
crosstab_eda.plot(kind="bar",    # select the bar to plot the count of categoricals
                 figsize=(15,7), # adjusting the size of graphs
                 stacked=True)   # code to stack 
plt.title("Genders Distribuition by TOP 15 Country's", fontsize=22) # adjusting title and fontsize
plt.xlabel("Country Names", fontsize=19) # adjusting x label and fontsize
plt.ylabel("% each Gender by Country", fontsize=19) # adjusting y label and fontsize
plt.xticks(rotation=0) # seting the rotation to zero
# Some parameters to legend of our graphs
plt.legend(title="Gender of DS", 
          loc='best', bbox_to_anchor=(1, .9),
          ncol=1, fancybox=True, shadow=True)

plt.show() # rendering

print("DESCRIPTIVE VALUES OF GENDER BY COUNTRYS: ")

country_repayment = ["Q3_In which country do you currently reside?", 
                      "Q1_What is your gender? - Selected Choice"] #seting the desired 

cm = sns.light_palette("green", as_cmap=True) #seting the colormap of our table
round(pd.crosstab(df_multiChoice[df_multiChoice["Q3_In which country do you currently reside?"]\
                                                .isin(countrys[:25].index.values)][country_repayment[0]],\
                                 df_multiChoice[country_repayment[1]], normalize="index"),4).style.background_gradient(cmap = cm)
country_repayment = ["Q2_What is your age (# years)?", 
                     "Q1_What is your gender? - Selected Choice"] #seting the desired 

cm = sns.light_palette("green", as_cmap=True)
round(pd.crosstab(df_multiChoice[country_repayment[0]],\
                                 df_multiChoice[country_repayment[1]], normalize="index"),2).style.background_gradient(cmap = cm)
import random

number_of_colors = 20 # total number of different collors that we will use

# Here I will generate a bunch of hexadecimal colors 
color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
#Changing the name of some values in our category's to a short name
df_multiChoice.loc[df_multiChoice['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'] == 
                   'Some college/university study without earning a bachelor’s degree', 
                   'Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'] = "Some college-No Bachelor's"

df_multiChoice.loc[df_multiChoice['Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'] == 
                   'No formal education past high school', 
                   'Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?'] = 'No formal High S.'

# changing the name to get a short name in our crosstab
df_multiChoice.rename(columns={
    "Q4_What is the highest level of formal education that you have attained or plan to attain within the next 2 years?":
    "Q4_Highest formal education or plan to within the next 2 years?"}, 
                      inplace=True)

EducOrder = ['I prefer not to answer', 'No formal High S.', 'Professional degree', 
             "Some college-No Bachelor's", 'Bachelor’s degree','Master’s degree','Doctoral degree']

FormEduc = df_multiChoice["Q4_Highest formal education or plan to within the next 2 years?"].value_counts() #counting the values of Country

print("Description most frequent countrys: ")
print(FormEduc[:15]) #printing the top most frequent

country_tree = round((df_multiChoice["Q4_Highest formal education or plan to within the next 2 years?"].value_counts()[:30] \
                       / len(df_multiChoice["Q4_Highest formal education or plan to within the next 2 years?"]) * 100),2)

plt.figure(figsize=(15,7))
g = squarify.plot(sizes=country_tree.values, label=country_tree.index, 
                  value=country_tree.values,
                  alpha=.4, color=color)
g.set_title("Formal Education Badge of Users - % size of total",fontsize=20)
g.set_axis_off()
plt.show()
crosstab_eda = pd.crosstab(columns=df_multiChoice["Q4_Highest formal education or plan to within the next 2 years?"], 
                           # at this line, I am using the isin to select just the top 5 of browsers
                           index=df_multiChoice[df_multiChoice["Q3_In which country do you currently reside?"]\
                                                .isin(countrys[:15].index.values)]["Q3_In which country do you currently reside?"], 
                           normalize= "index").reindex(columns=EducOrder) # Seting the correct order to crosstab

# Ploting the crosstab that we did above
crosstab_eda.plot(kind="barh",    # select the bar to plot the count of categoricals
                 figsize=(9,15), # adjusting the size of graphs
                 stacked=True)   # code to stack 
plt.title("Education Badges by TOP 15 Country's", fontsize=22) # adjusting title and fontsize
plt.ylabel("Country Names", fontsize=19) # adjusting y label and fontsize
plt.xlabel("% each Badge by Country", fontsize=19) # adjusting x label and fontsize
plt.xticks(rotation=0) # seting the rotation to zero
# Some parameters to legend of our graphs
plt.legend(title="Education Badge", 
          loc='best', bbox_to_anchor=(1, .9),
          ncol=1, fancybox=True, shadow=True)

plt.show() # rendering

print("DESCRIPTIVE VALUES OF BADGES BY COUNTRYS: ")

country_repayment = ["Q3_In which country do you currently reside?", 
                     "Q4_Highest formal education or plan to within the next 2 years?"] #seting the desired 

cm = sns.light_palette("green", as_cmap=True) #seting the colormap of our table
round(pd.crosstab(df_multiChoice[df_multiChoice["Q3_In which country do you currently reside?"]\
                                                .isin(countrys[:25].index.values)][country_repayment[0]],\
                                 df_multiChoice[country_repayment[1]], normalize="index"),4).reindex(columns=EducOrder).style.background_gradient(cmap = cm)
crosstab_eda = pd.crosstab(index=df_multiChoice["Q4_Highest formal education or plan to within the next 2 years?"], 
                           # at this line, I am using the isin to select just the top 5 of browsers
                           columns=df_multiChoice["Q46_Approx percent involved exploring model insights?"], 
                           normalize= "index").reindex(index=EducOrder)
# Ploting the crosstab that we did above
crosstab_eda.plot(kind="bar",    # select the bar to plot the count of categoricals
                 figsize=(15,7), # adjusting the size of graphs
                 stacked=False)   # code to stack 
plt.title("Time exploring models to insights by Formal Education Badge's ", fontsize=22) # adjusting title and fontsize
plt.xlabel("Formal Education Badge Name", fontsize=19) # adjusting x label and fontsize
plt.ylabel("% of Time consuming to find insights", fontsize=19) # adjusting y label and fontsize
plt.xticks(rotation=15) # seting the rotation to zero
# Some parameters to legend of our graphs
plt.legend(title="% of time exploring model insights", 
          loc='upper right', bbox_to_anchor=(1, 1.01),
          ncol=4, fancybox=True, shadow=False)
plt.show() # rendering

print("DESCRIPTIVE VALUES OF COUNTRYS BY TIME CONSUMING EXPLORING INSIGHTS IN MODELS: ")
print("Using index as normalizer")
country_repayment = ["Q4_Highest formal education or plan to within the next 2 years?", 
                     "Q46_Approx percent involved exploring model insights?"] #seting the desired 

cm = sns.light_palette("green", as_cmap=True) #seting the colormap of our table
round(pd.crosstab(df_multiChoice[country_repayment[0]],\
                                 df_multiChoice[country_repayment[1]], normalize="index"),2).reindex(index=EducOrder).style.background_gradient(cmap = cm)
# changing the name to get a short name in our crosstab
df_multiChoice.rename(columns={
    'Q6_Select the title most similar to your current role (or most recent title if retired): - Selected Choice':
    'Q6_The title most similar to your current role'}, 
                      inplace=True)

TitleRole = df_multiChoice['Q6_The title most similar to your current role'].value_counts() #counting the values of Country

print("Description most frequent Title Role: ")
print(TitleRole[:10]) #printing the top most frequent

country_tree = round((df_multiChoice['Q6_The title most similar to your current role'].value_counts()[:15] \
                       / len(df_multiChoice['Q6_The title most similar to your current role']) * 100),2)

plt.figure(figsize=(15,7))
g = squarify.plot(sizes=country_tree.values, label=country_tree.index, 
                  value=country_tree.values,
                  alpha=.4, color=color)
g.set_title("TOP 15 title most similar to current role of Kaggle Users",fontsize=20)
g.set_axis_off()
plt.show()
crosstab_eda = pd.crosstab(columns=df_multiChoice["Q4_Highest formal education or plan to within the next 2 years?"], 
                           # at this line, I am using the isin to select just the top 5 of browsers
                           index=df_multiChoice['Q6_The title most similar to your current role'], 
                           normalize= "index")
# Ploting the crosstab that we did above
crosstab_eda.plot(kind="barh",    # select the bar to plot the count of categoricals
                 figsize=(12,18), # adjusting the size of graphs
                 stacked=True)   # code to stack 
plt.title("Title of current role by Education Badges", fontsize=22) # adjusting title and fontsize
plt.xlabel("% of each Educational Badge", fontsize=19) # adjusting x label and fontsize
plt.ylabel("Title the most represents the current role", fontsize=19) # adjusting y label and fontsize
plt.xticks(rotation=0) # seting the rotation to zero
# Some parameters to legend of our graphs
plt.legend(title="Formal Education", 
          loc='best', bbox_to_anchor=(1, .9),
          ncol=1, fancybox=True, shadow=True)

plt.show() # rendering

country_repayment = ['Q6_The title most similar to your current role',
                     "Q4_Highest formal education or plan to within the next 2 years?"] #seting the desired 

cm = sns.light_palette("green", as_cmap=True) #seting the colormap of our table
round(pd.crosstab(df_multiChoice[country_repayment[0]],\
                                 df_multiChoice[country_repayment[1]],
                 rownames=["Title Role"],
                 colnames=["Formal Education"],
                 margins=True,
                 margins_name = "Total"),2).drop("Total", axis=0).style.background_gradient(cmap = cm)
crosstab_eda = pd.crosstab(columns=df_multiChoice["Q46_Approx percent involved exploring model insights?"], 
                           # at this line, I am using the isin to select just the top 5 of browsers
                           index=df_multiChoice['Q6_The title most similar to your current role'], 
                           normalize= "index")
# Ploting the crosstab that we did above
crosstab_eda.plot(kind="barh",    # select the bar to plot the count of categoricals
                 figsize=(12,18), # adjusting the size of graphs
                 stacked=True)   # code to stack 
plt.title("Current role by Time exploring Insights", fontsize=22) # adjusting title and fontsize
plt.xlabel("% in time exploring (in %)", fontsize=19) # adjusting x label and fontsize
plt.ylabel("Title the most represents the current role", fontsize=19) # adjusting y label and fontsize
plt.xticks(rotation=0) # seting the rotation to zero
# Some parameters to legend of our graphs
plt.legend(title="% Time Exploring Insights", 
          loc='best', bbox_to_anchor=(1, .9),
          ncol=1, fancybox=True, shadow=True)

plt.show() # rendering

country_repayment = ['Q6_The title most similar to your current role',
                     "Q46_Approx percent involved exploring model insights?"] #seting the desired 

cm = sns.light_palette("green", as_cmap=True) #seting the colormap of our table
round(pd.crosstab(df_multiChoice[country_repayment[0]],\
                                 df_multiChoice[country_repayment[1]], 
                  margins=True, margins_name="Total",
                  rownames=["Title Role"],colnames=["Time Exploring Insights"]),2).drop("Total", axis=0).style.background_gradient(cmap = cm)
country_repayment = ['Q6_The title most similar to your current role',
                     "Q2_What is your age (# years)?"] #seting the desired 

cm = sns.light_palette("green", as_cmap=True) #seting the colormap of our table
round(pd.crosstab(df_multiChoice[country_repayment[0]],\
                                 df_multiChoice[country_repayment[1]], 
                  rownames=["Title Role"], colnames=["Age of Kagglers"],
                  margins=True, margins_name="Totals"),2).drop("Totals", axis=0).style.background_gradient(cmap = cm)
import squarify
 ## Getting the percentual of each category
FormEduc = round(df_multiChoice["Q17_What specific programming language do you use most often? - Selected Choice"].value_counts() /\
                 df_multiChoice["Q17_What specific programming language do you use most often? - Selected Choice"].value_counts().sum() * 100,2) #counting the values of Country

print("Description most frequent countrys: ")
print(FormEduc[:10]) #printing the top most frequent

country_tree = round((df_multiChoice["Q17_What specific programming language do you use most often? - Selected Choice"].value_counts()[:10] \
                       / df_multiChoice["Q17_What specific programming language do you use most often? - Selected Choice"].value_counts().sum() * 100),2)

plt.figure(figsize=(15,7))
g = squarify.plot(sizes=country_tree.values, label=country_tree.index, 
                  value=country_tree.values,
                  alpha=.4, color=color)
g.set_title("Formal Education Badge of Users - % size of total",fontsize=20)
g.set_axis_off()
plt.show()
Q17_Q4 = ["Q17_What specific programming language do you use most often? - Selected Choice",
                     "Q4_Highest formal education or plan to within the next 2 years?"] #seting the desired 

cm = sns.light_palette("green", as_cmap=True) #seting the colormap of our table
round(pd.crosstab(df_multiChoice[Q17_Q4[0]],\
                                 df_multiChoice[Q17_Q4[1]], normalize='index'),3).style.background_gradient(cmap = cm)
df_multiChoice.loc[df_multiChoice['Q23_Approximately what percent of your time at work or school is spent actively coding?'] == "0% of my time", 
                   "Q23_Approximately what percent of your time at work or school is spent actively coding?"] = "0%"
df_multiChoice.loc[df_multiChoice['Q23_Approximately what percent of your time at work or school is spent actively coding?'] == "1% to 25% of my time", 
                   "Q23_Approximately what percent of your time at work or school is spent actively coding?"] = "1% to 25%"
df_multiChoice.loc[df_multiChoice['Q23_Approximately what percent of your time at work or school is spent actively coding?'] == "25% to 49% of my time", 
                   "Q23_Approximately what percent of your time at work or school is spent actively coding?"] = "25% to 49%"
df_multiChoice.loc[df_multiChoice['Q23_Approximately what percent of your time at work or school is spent actively coding?'] == "50% to 74% of my time", 
                   "Q23_Approximately what percent of your time at work or school is spent actively coding?"] = "50% to 74%"
df_multiChoice.loc[df_multiChoice['Q23_Approximately what percent of your time at work or school is spent actively coding?'] == "75% to 99% of my time",
                   "Q23_Approximately what percent of your time at work or school is spent actively coding?"] = "75% to 99%"
df_multiChoice.loc[df_multiChoice['Q23_Approximately what percent of your time at work or school is spent actively coding?'] == "100% of my time", 
                   "Q23_Approximately what percent of your time at work or school is spent actively coding?"] = "100%"
import squarify

TimeCoding = df_multiChoice["Q23_Approximately what percent of your time at work or school is spent actively coding?"].value_counts() #counting the values of Country

print("Description most frequent: ")
print(TimeCoding[:10]) #printing the top most frequent

country_tree = round((df_multiChoice["Q23_Approximately what percent of your time at work or school is spent actively coding?"].value_counts()[:10] \
                       / df_multiChoice["Q23_Approximately what percent of your time at work or school is spent actively coding?"].value_counts().sum() * 100),2)

plt.figure(figsize=(15,7))
g = squarify.plot(sizes=country_tree.values, label=country_tree.index, 
                  value=country_tree.values,
                  alpha=.4, color=color)
g.set_title("Approximately what percent of your time is spent actively coding?",fontsize=20)
g.set_axis_off()
plt.show()
order_time =['0%', '1% to 25%', '25% to 49%', '50% to 74%', '75% to 99%', '100%']

EducOrder = ['I prefer not to answer', 'No formal High S.', 'Professional degree', 
             "Some college-No Bachelor's", 'Bachelor’s degree','Master’s degree','Doctoral degree']

crosstab_eda = pd.crosstab(columns=df_multiChoice["Q4_Highest formal education or plan to within the next 2 years?"], 
                           # at this line, I am using the isin to select just the top 5 of browsers
                           index=df_multiChoice["Q23_Approximately what percent of your time at work or school is spent actively coding?"], 
                           normalize= "index").reindex(columns=EducOrder)

crosstab_eda = crosstab_eda.reindex(index=order_time)

# Ploting the crosstab that we did above
crosstab_eda.plot(kind="bar",  # select the bar to plot the count of categoricals
                 figsize=(15,7), # adjusting the size of graphs
                 stacked=False)   # code to stack 
plt.title("Time coding by Formal Education Badge's ", fontsize=22) # adjusting title and fontsize
plt.xlabel("% of time expent coding", fontsize=19) # adjusting x label and fontsize
plt.ylabel("Prob of Formal Education", fontsize=19) # adjusting y label and fontsize
plt.xticks(rotation=0) # seting the rotation to zero
# Some parameters to legend of our graphs
plt.legend(title="% time coding", 
          loc='best', bbox_to_anchor=(1, 1.01),
          ncol=1, fancybox=True, shadow=False)

plt.show() # rendering

print("DESCRIPTIVE VALUES OF COUNTRYS BY TIME CONSUMING EXPLORING INSIGHTS IN MODELS: ")
print("Using columns as normalizer")
Q4_Q23 = ["Q4_Highest formal education or plan to within the next 2 years?", 
                     "Q23_Approximately what percent of your time at work or school is spent actively coding?"] #seting the desired 

cm = sns.light_palette("green", as_cmap=True) #seting the colormap of our table
round(pd.crosstab(df_multiChoice[Q4_Q23[1]],\
                             df_multiChoice[Q4_Q23[0]],  
                  margins=True, margins_name="Total",
                  rownames=["Percent Time Coding"],
                  colnames=["Formal Education"]),2)\
.reindex(index=order_time,columns=EducOrder)\
       .style.background_gradient(cmap = cm)
def CleaningToPlot(df, column):
    question = df.filter(like=(column))
    range_cols = len(question.columns)
    mapping = dict()
    for i in range(range_cols):
        old_index = question.columns[i]
        string = column + '_part_' + str(i+1)
        mapping.update({old_index : string})
    question = question.rename(columns=mapping)
    return question
Q15 = CleaningToPlot(df_multiChoice, "Q15")
#Dict to get short names to show the information
TypeDict = {"Audio Data":"Audio", "Categorical Data":"Categorical", "Genetic Data":"Genetic",
            "Geospatial Data": "Geospatial", "Image Data":"Image", "Numerical Data":"Numerical",
            "Sensor Data":"Sensor", "Tabular Data":"Tabular", "Text Data":"Text", 
            "Time Series Data":"Time Series", "Video Data":"Video"}

df_multiChoice["Q32_What is the type of data that you currently interact with most often at work or school? - Selected Choice"].replace(TypeDict, inplace=True)
crosstab_eda = pd.crosstab(columns=df_multiChoice["Q4_Highest formal education or plan to within the next 2 years?"], 
                           index=df_multiChoice['Q32_What is the type of data that you currently interact with most often at work or school? - Selected Choice'], 
                           normalize= "columns").reindex(columns=EducOrder)


# Ploting the crosstab that we did above
crosstab_eda.plot(kind="bar",  # select the bar to plot the count of categoricals
                 figsize=(15,7), # adjusting the size of graphs
                 stacked=False)   # code to stack 
plt.title("Data Types by Formal Education", fontsize=22) # adjusting title and fontsize
plt.xlabel("Data Types", fontsize=19) # adjusting x label and fontsize
plt.ylabel("Prob of Formal Education", fontsize=19) # adjusting y label and fontsize
plt.xticks(rotation=0) # seting the rotation to zero
# Some parameters to legend of our graphs
plt.legend(title="% time coding", 
           loc='best', bbox_to_anchor=(1, 1.01),
           ncol=1, fancybox=True, shadow=False)

plt.show() # rendering

print("DESCRIPTIVE VALUES OF DATA TYPES BY FORMAL EDUCATION:  ")

Q32_Q4 = ['Q32_What is the type of data that you currently interact with most often at work or school? - Selected Choice', 
          "Q4_Highest formal education or plan to within the next 2 years?"]
                                      
cm = sns.light_palette("green", as_cmap=True)

round(pd.crosstab(df_multiChoice[Q32_Q4[0]], 
                  df_multiChoice[Q32_Q4[1]],
                 rownames=["Most Often Data Type"], colnames=["Formal Education"]
                 ),2).reindex(columns=EducOrder).style.background_gradient(cmap = cm)
print("DESCRIPTIVE VALUES OF DATA TYPES BY TITLE OF CURRENT ROLE: ")

Q32_Q6 = ['Q32_What is the type of data that you currently interact with most often at work or school? - Selected Choice', 
          'Q6_The title most similar to your current role']
                                      
cm = sns.light_palette("green", as_cmap=True)

round(pd.crosstab(df_multiChoice[Q32_Q6[1]], 
                  df_multiChoice[Q32_Q6[0]],
                 rownames=["Title Role"], colnames=["Data Types"],
                 margins=True, margins_name="Totals"),2).drop("Totals", axis=0).style.background_gradient(cmap = cm)
salaryDict = {'0-10,000':'0-10','10-20,000':'10-20','20-30,000':'20-30', '30-40,000':'30-40', '40-50,000':'40-50', 
              '50-60,000':'50-60', '60-70,000':'60-70', '70-80,000':'70-80','80-90,000':'80-90',
              '90-100,000':'90-100','100-125,000':"100-125",'125-150,000':'125-150','150-200,000':'150-200',
              '200-250,000':'200-250','250-300,000':'250-300','300-400,000':'300-400',
              '400-500,000':'400-500', '500,000+':'500+','I do not wish to disclose my approximate yearly compensation': "not-inf" }
              
RevenueOrder = ["not-inf",'0-10','10-20','20-30', '30-40', '40-50', '50-60', '60-70',
                '70-80','80-90','90-100','100-125','125-150','150-200',
                '200-250','250-300','300-400','400-500', '500+']

df_multiChoice['Q9_What is your current yearly compensation (approximate $USD)?'].replace(salaryDict, inplace=True)

print(round(df_multiChoice['Q9_What is your current yearly compensation (approximate $USD)?'].value_counts()[:10] / \
     df_multiChoice['Q9_What is your current yearly compensation (approximate $USD)?'].value_counts().sum() * 100,2))

#Seting size to figure that we will plot
plt.figure(figsize=(15,6))

sns.countplot(df_multiChoice["Q9_What is your current yearly compensation (approximate $USD)?"], # ploting the nine question
              order=RevenueOrder) 
plt.title("Question 9 - What is your current yearly compensation (approximate $USD)?", fontsize=20) # Adding Title and seting the size
plt.xlabel("Compensation Range", fontsize=16) # Adding x label and seting the size
plt.ylabel("Count", fontsize=16) # Adding y label and seting the size
plt.xticks(rotation=35)

#rendering the plot
plt.show()
print("DESCRIPTIVE VALUES OF YEARLY COMPENSATION BY TITLE OF CURRENT ROLE: ")

Q32_Q6 = ["Q9_What is your current yearly compensation (approximate $USD)?", 
          'Q6_The title most similar to your current role']
                                      
cm = sns.light_palette("green", as_cmap=True)

round(pd.crosstab(df_multiChoice[Q32_Q6[1]], 
                  df_multiChoice[Q32_Q6[0]],
                  rownames=["Title Roles"], colnames=["Compensation in MUSD(K)"], 
                  margins=True, margins_name="Totals"),2).drop("Totals", axis=0)\
.reindex(columns=RevenueOrder)\
.style.background_gradient(cmap = cm) #Mapping the colors

crosstab_eda = pd.crosstab(columns=df_multiChoice["Q4_Highest formal education or plan to within the next 2 years?"], 
                           index=df_multiChoice['Q9_What is your current yearly compensation (approximate $USD)?'], 
                           normalize= "index").reindex(index=RevenueOrder, columns=EducOrder)


# Ploting the crosstab that we did above
crosstab_eda.plot(kind="bar",  # select the bar to plot the count of categoricals
                 figsize=(15,7), # adjusting the size of graphs
                 stacked=True)   # code to stack 
plt.title("Compensation Range by Formal Education", fontsize=22) # adjusting title and fontsize
plt.xlabel("Compensation Range", fontsize=19) # adjusting x label and fontsize
plt.ylabel("Prob of each Formal Education(in %)", fontsize=19) # adjusting y label and fontsize
plt.xticks(rotation=0) # seting the rotation to zero
# Some parameters to legend of our graphs
plt.legend(title="Formal Education", 
           loc='best', bbox_to_anchor=(1, 1.01),
           ncol=1, fancybox=True, shadow=False)

plt.show() # rendering

print("DESCRIPTIVE OF YEARLY COMPENSATION BY FORMAL EDUCATION  ")

Q9_Q4 = ['Q9_What is your current yearly compensation (approximate $USD)?', 
          "Q4_Highest formal education or plan to within the next 2 years?"]
                                      
cm = sns.light_palette("green", as_cmap=True)

round(pd.crosstab(df_multiChoice[Q9_Q4[0]], 
                  df_multiChoice[Q9_Q4[1]],
                  rownames=["Yearly Compensation"], colnames=["Formal Education"],
                  margins=True, margins_name="Total", normalize='index'
                 ),2).reindex(index=RevenueOrder, columns=EducOrder).style.background_gradient(cmap = cm)
print(round(df_multiChoice['Q37_On which online platform have you spent the most amount of time? - Selected Choice'].value_counts()[:10] / \
     df_multiChoice['Q37_On which online platform have you spent the most amount of time? - Selected Choice'].value_counts().sum() * 100,2))

#Seting size to figure that we will plot
plt.figure(figsize=(12,6))

sns.countplot(df_multiChoice["Q37_On which online platform have you spent the most amount of time? - Selected Choice"], # ploting the nine question
              ) 
plt.title("Question 37 - Which online platform users have spent the most time?", fontsize=20) # Adding Title and seting the size
plt.xlabel("Online Platform that users most spent time", fontsize=18) # Adding x label and seting the size
plt.ylabel("Count", fontsize=18) # Adding y label and seting the size
plt.xticks(rotation=35)

#rendering the plot
plt.show()
crosstab_eda = pd.crosstab(columns=df_multiChoice["Q4_Highest formal education or plan to within the next 2 years?"], 
                           index=df_multiChoice['Q37_On which online platform have you spent the most amount of time? - Selected Choice'], 
                           normalize= "index").reindex(columns=EducOrder)


# Ploting the crosstab that we did above
crosstab_eda.plot(kind="bar",  # select the bar to plot the count of categoricals
                 figsize=(13,7), # adjusting the size of graphs
                 stacked=True)   # code to stack 
plt.title("Online Platform by Formal Education", fontsize=22) # adjusting title and fontsize
plt.xlabel("Online Platform", fontsize=19) # adjusting x label and fontsize
plt.ylabel("Prob of each Formal Education(in %)", fontsize=19) # adjusting y label and fontsize
plt.xticks(rotation=35) # seting the rotation to zero
# Some parameters to legend of our graphs
plt.legend(title="Formal Education", 
           loc='best', bbox_to_anchor=(1, 1.01),
           ncol=1, fancybox=True, shadow=False)

plt.show() # rendering

print("MOST USED MOOC'S BY FORMAL EDUCATION  ")

Q9_Q4 = ['Q37_On which online platform have you spent the most amount of time? - Selected Choice', 
          "Q4_Highest formal education or plan to within the next 2 years?"]
                                      
cm = sns.light_palette("green", as_cmap=True)
round(pd.crosstab(df_multiChoice[Q9_Q4[0]], 
                  df_multiChoice[Q9_Q4[1]],
                  rownames=["Online Platforms"], colnames=["Formal Education"],
                  margins=True, margins_name="Total"
                 ),2).drop("Total", axis=0).reindex(columns=EducOrder).style.background_gradient(cmap = cm)
print("MOST USED MOOC'S BY YEARLY COMPENSATION: ")

Q37_Q9 = ['Q37_On which online platform have you spent the most amount of time? - Selected Choice', 
          "Q9_What is your current yearly compensation (approximate $USD)?"]
                                      
cm = sns.light_palette("green", as_cmap=True)
round(pd.crosstab(df_multiChoice[Q37_Q9[0]], 
                  df_multiChoice[Q37_Q9[1]],
                  rownames=["Online Platforms"], colnames=["Yearly Compensation"],
                  margins=True, margins_name="Total"),2).drop("Total", axis=0).reindex(columns=RevenueOrder).style.background_gradient(cmap = cm)
print("MOST USED MOOC'S BY TITLE ROLE ")
Q37_Q6 = ['Q37_On which online platform have you spent the most amount of time? - Selected Choice', 
          'Q6_The title most similar to your current role']
                                      
cm = sns.light_palette("green", as_cmap=True)
round(pd.crosstab(df_multiChoice[Q37_Q6[1]], 
                  df_multiChoice[Q37_Q6[0]],
                  colnames=["Online Platforms"], rownames=["Title role"],
                  margins=True, margins_name="Total"),2).drop("Total", axis=0).style.background_gradient(cmap = cm)
print("MOST USED MOOC'S BY TIME ANALYSING: ")

Q37_Q23 = ['Q37_On which online platform have you spent the most amount of time? - Selected Choice', 
          "Q23_Approximately what percent of your time at work or school is spent actively coding?"]
                                      
cm = sns.light_palette("green", as_cmap=True)
round(pd.crosstab(df_multiChoice[Q37_Q23[1]], 
                  df_multiChoice[Q37_Q23[0]],
                  colnames=["Online Platforms"], rownames=["% of Time Coding "], normalize="columns", 
                 ),2).reindex(order_time).style.background_gradient(cmap = cm)

codTimeDict = {'I have never written code but I want to learn': "NeverCoded - Want Learn",
               'I have never written code and I do not want to learn': "NeverCoded - Don't want Learn"}

df_multiChoice["Q24_How long have you been writing code to analyze data?"].replace(codTimeDict, inplace=True)

order_coding = ['< 1 year', '1-2 years', '3-5 years', '5-10 years', '10-20 years', '20-30 years',
                '30-40 years','40+ years', 'NeverCoded - Want Learn', "NeverCoded - Don't want Learn"]
print(round(df_multiChoice['Q24_How long have you been writing code to analyze data?'].value_counts()[:10] / \
     df_multiChoice['Q24_How long have you been writing code to analyze data?'].value_counts().sum() * 100,2).reindex(order_coding))

#Seting size to figure that we will plot
plt.figure(figsize=(15,6))

sns.countplot(df_multiChoice["Q24_How long have you been writing code to analyze data?"], # ploting the nine question
              order=order_coding) 
plt.title("Question 24 - How long have you been writing code to analyze data", fontsize=20) # Adding Title and seting the size
plt.xlabel("Years Range", fontsize=16) # Adding x label and seting the size
plt.ylabel("Count", fontsize=16) # Adding y label and seting the size
plt.xticks(rotation=35)

#rendering the plot
plt.show()
order_coding = ['< 1 year', '1-2 years', '3-5 years', '5-10 years',
                '10-20 years', '20-30 years', '30-40 years','40+ years',]

print("MOST USED MOOC'S BY TIME CODING: ")

Q37_Q24 = ['Q37_On which online platform have you spent the most amount of time? - Selected Choice', 
           "Q24_How long have you been writing code to analyze data?"]
                                      
cm = sns.light_palette("green", as_cmap=True)
round(pd.crosstab(df_multiChoice[Q37_Q24[0]], 
                  df_multiChoice[Q37_Q24[1]],
                  colnames=["Time coding"], rownames=["Online Platforms"], normalize="index", 
                 ),2).reindex(columns=order_coding).style.background_gradient(cmap = cm)