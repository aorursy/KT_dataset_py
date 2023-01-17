# Import Python packages
import numpy as np 
import pandas as pd 
import os
import math

import matplotlib.pyplot as plt

%matplotlib inline
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 5000)


# Change when no longer running notebook locally
base_dir = os.path.join(os.getcwd(), 'data') 
base_dir = '../input/'
fileName = 'multipleChoiceResponses.csv'
filePath = os.path.join(base_dir,fileName)
df_full = pd.read_csv(filePath) 
# Remove the first row which contains the question
df = df_full[1:]
#plt.style.use('ggplot')

# Update plot Parameters
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 24})
plt.rcParams.update({'figure.titlesize': 30})

# Generic labels used throughout the plots
respondents_count_title = "# of Respondents"
# Select the first part of the question
question_id = 'Q19'
# Describe the essence of this question in as few words as possible (will be used in plot labels)
short_question_description = 'ML Frameworks'
# DATA PREP

# Set up for working this question 
question_aggregate_col = question_id + '_Total'

# Get a list of all the column names in the df
full_col_lst = list(df.columns)

# Select a subset of columns (excluding the free response)
short_col_lst =  [q for q in full_col_lst if (question_id in q) and ('OTHER' not in q)]
# Remove the column for None (ie. Q19_Part_18)
short_col_lst_minus_none = short_col_lst[:-2] + short_col_lst[-1:]
# Remove the free response (ie. 'Q19_Part_19')
short_col_lst_minus_free = short_col_lst[:-1]

# Create a new dataframe consisting of answer choices except "None"
df_question = df[short_col_lst_minus_none]

## Add a column for total answer choices selected to the main df
# Summary: Replace all NaN values with 0, and string values with 1 (excluding "None" selection), then get a total count of all string fields
# Steps: (1) Identify all NaN Values (2) Invert True for False (3) Multiply by 1 to convert to float (4) Sum over rows (5) convert to int
df_plus = df.copy()
df_plus[question_aggregate_col] = ((df_plus[short_col_lst_minus_none].isnull() == False)*1).sum(axis = 1).astype(int)


## Add a column for whether this question was actually answered to the main df
answered_q = ('Answered ' + question_id + ' Question')
df_plus[answered_q] = ((df_plus[short_col_lst].isnull() == False)*1).sum(axis = 1).astype(int)
df_plus[answered_q] = df_plus[answered_q] > 0

# Create a dataframe with only respondents who answered this question
df_plus_answered = df_plus[df_plus[answered_q] == True]  
question_heading = question_id + '_Part_1'
question_string = df_full[[question_heading]].iloc[0,0].split('-')[0]
print(question_string)
# An Aside: What are all the unique answers?
choices = [ df_question[~df_question[col].isnull()][col].max() for col in short_col_lst_minus_none]
print("Below are the {} options for this question.\n".format(len(choices)))
print(choices)
total = (df_plus[answered_q].count())
answered = (df_plus[df_plus[answered_q] == True][answered_q].count())
no_response = (total - answered)/total

print('{0:.2f}% of survey participants did NOT answer this question.'.format(no_response*100))
# Specificy Dataframe
d = df[short_col_lst]

# Series
column_totals = ((d[short_col_lst].isnull() == False)*1).sum(axis = 0)
all_choice_options = [ d[~d[col].isnull()][col].max() for col in short_col_lst]

# Label Names
x_label = respondents_count_title
y_label = short_question_description
plot_title = 'Survey Participants Selected the Following\n' + short_question_description

# Plot Chart with Inputs
ax = sns.barplot(column_totals,
                 all_choice_options,
                 palette = 'Set2',
                 edgecolor = 'black'
                )
plt.setp(ax.get_xticklabels(), rotation=45)

plt.subplots_adjust(top=0.87)
plt.suptitle(plot_title, fontweight='bold', color = '#1c1d20' )
plt.xlabel(x_label, fontweight='bold')
plt.ylabel(y_label, fontweight='bold')

plt.show()
col_none_index = [i for i,x in enumerate(all_choice_options) if 'None' in x]
question_with_none_responses = short_col_lst[col_none_index[0]]
selected_option_none = (d[question_with_none_responses ].count())
none_response = (selected_option_none)/answered

print('Of the {0:.1f}% participants that answered this question, {1:.1f}% of the "respondents" selected the option: None.'.format((1-no_response)*100, none_response*100))
# Specificy Dataframe
d = df_plus_answered

# Series
x_col = question_aggregate_col

# Label Names
x_label = '# of ' + short_question_description
y_label = respondents_count_title
plot_title = 'Distribution of the Number of\n' + short_question_description + ' Selected by Respondents'

# Plot Chart with Inputs
ax = sns.countplot(x=x_col,
                   data=d,
                   palette = ['#00bcb5'],
                   edgecolor = 'black'
                  )

plt.subplots_adjust(top=0.87)
plt.suptitle(plot_title, fontweight='bold', color = '#1c1d20' )
plt.xlabel(x_label, fontweight='bold')
plt.ylabel(y_label, fontweight='bold')
plt.show()

print("The average number of ML frameworks that respondents know is {0:.2f}.".format(d[question_aggregate_col].mean()))
# Specificy Dataframe
d = df_plus_answered

# Years of experience without nan
years_of_experience = list(d['Q8'].unique())[2:]
# Re-order (manual)
years_of_experience = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30 +']

# Series
x_col = 'Q8'
y_col = question_aggregate_col

# Label Names
x_label = 'Years of Experience'
y_label = '# of ' + short_question_description
plot_title = 'Distribution of the Number of\n' + short_question_description + ' by ' + x_label

# Plot Chart with Inputs
ax = sns.boxplot(d[x_col], 
                 d[y_col], 
                 order = years_of_experience)

plt.subplots_adjust(top=0.87)
plt.suptitle(plot_title, fontweight='bold', color = '#1c1d20' )
plt.xlabel(x_label, fontweight='bold')
plt.ylabel(y_label, fontweight='bold')
plt.show()
# Add a column to isolate the data scientist profession
df_plus_answered['is_data_scientist'] = (df_plus_answered['Q6']== 'Data Scientist')*1

# Specificy Dataframe
d = df_plus_answered

# Series
x_col = 'Q8'
y_col = question_aggregate_col

# Label Names
x_label = 'Years of Experience'
y_label = '# of ' + short_question_description
plot_title = 'Distribution of the Number of\n' + short_question_description + ' by ' + x_label + '\n (Splitting out Data Scientists)'

# Plot Chart with Inputs
ax = sns.boxplot(x=x_col, 
                 y=y_col, 
                 hue='is_data_scientist', 
                 order=years_of_experience,
                 data=d, 
                 palette="Set2")

plt.subplots_adjust(top=0.80)
plt.suptitle(plot_title, fontweight='bold', color = '#1c1d20' )
plt.xlabel(x_label, fontweight='bold')
plt.ylabel(y_label, fontweight='bold')
plt.show()
# Specificy Dataframe
d = df_plus_answered[df_plus_answered['is_data_scientist'] == 1]

# Series
x_col = 'Q8'
y_col = question_aggregate_col

# Label Names
x_label = 'Years of Experience'
y_label = '# of ' + short_question_description
plot_title = 'Distribution of the Number of\n' + short_question_description + ' by ' + x_label + '\n (for Data Scientists)'

# Plot Chart with Inputs
ax = sns.violinplot(x=x_col, 
                    y=y_col, 
                    data=d, 
                    cut =0, 
                    order = years_of_experience,
                    scale = "count"
                   )   
plt.subplots_adjust(top=0.80)
plt.suptitle(plot_title, fontweight='bold', color = '#1c1d20' )
plt.xlabel(x_label, fontweight='bold')
plt.ylabel(y_label, fontweight='bold')
plt.show()
# Select a subset of the top roles by number of respondents with that title
profession = ['Student', 'Data Scientist', 'Software Engineer', 'Data Analyst', 
              'Research Scientist', 'Business Analyst', 'Data Engineer', 'Research Assistant']

# Series
x_col = 'Q8'  # Years of experience
y_col = question_aggregate_col
category = 'Q6' # Profession

# Specificy Dataframe
# Reduce the data frame to look at the most number of years of experience and subset my selected profession
d = df_plus_answered[(df_plus_answered[x_col].isin(years_of_experience[:5])) & (df_plus_answered[category].isin(profession))]


# Label Names
x_label = 'Years of Experience'
y_label = '# of ' + short_question_description
plot_title = 'Distribution of the Number of\n' + short_question_description + ' by ' + x_label + '\n (for Common Professions)'

# Plot Chart with Inputs
ax = sns.boxplot(x=x_col, 
                 y=y_col, 
                 hue=category, 
                 order=years_of_experience[:5],
                 data=d,
                 palette="Set3")
plt.legend(bbox_to_anchor=(1.2,0.5), loc="center")
plt.subplots_adjust(top=0.80)
plt.suptitle(plot_title, fontweight='bold', color = '#1c1d20' )
plt.xlabel(x_label, fontweight='bold')
plt.ylabel(y_label, fontweight='bold')
plt.show()
# Reduce the data frame to look at the most number of years of experience and subset my selected profession
d = df_plus_answered[(df_plus_answered[x_col].isin(years_of_experience[:5])) & (df_plus_answered[category].isin(profession))]

# Label Names
x_label = 'Years of Experience'
y_label = respondents_count_title
plot_title = 'Number of Respondents per profession, \nper years of experience'
category = 'Q6' # Profession


# Plot Chart with Inputs
ax = sns.countplot(x=x_col, 
                   hue=category, 
                   order=years_of_experience[:5],
                   data=d,
                   palette="Set3",
                   edgecolor = 'black'
             )
plt.legend(bbox_to_anchor=(1.2,0.5), loc="center")
plt.subplots_adjust(top=0.80)
plt.suptitle(plot_title, fontweight='bold', color = '#1c1d20' )
plt.xlabel(x_label, fontweight='bold')
plt.ylabel(y_label, fontweight='bold')
plt.show()
# Specificy Dataframe
d = df_plus_answered[df_plus_answered['is_data_scientist'] == 1]

# Shorten one of the options description
d['Q4'].replace("Some college/university study without earning a bachelor’s degree", "Some Undergraduate Education", inplace = True)

# Series
x_col = 'Q4'  # Highest Education Level
y_col = question_aggregate_col
#category = 'is_data_scientist' #'Q6' # Profession


# Label Names
x_label = 'Highest Education Level'
y_label = '# of ' + short_question_description
plot_title = 'Data Scientist\n' + 'Distribution of the Number of\n' + short_question_description + ' by ' + x_label 


# Plot Chart with Inputs
plt.figure(figsize=(20,16))
ax = sns.boxplot( d[y_col],
                 d[x_col],
                    #hue=category,
                 data=d, 
                 palette="Set2",
                   #scale = "count",
                    #cut = True
                   )


plt.subplots_adjust(top=0.87)
plt.suptitle(plot_title, fontweight='bold', color = '#1c1d20' )
plt.xlabel(y_label, fontweight='bold')
plt.ylabel(x_label, fontweight='bold')
#ax.xaxis.grid(True)

plt.show()
