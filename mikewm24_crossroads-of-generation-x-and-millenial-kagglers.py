import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import isnan  
import warnings
warnings.filterwarnings("ignore") 

mc_survey = pd.read_csv('../input/multipleChoiceResponses.csv')
free_form = pd.read_csv('../input/freeFormResponses.csv')

# Organize and clean
# Organize the target group. This also eliminates the first row, where the question is listed. 
free_form.drop(free_form.index[0], inplace = True)
mc_survey.drop(mc_survey.index[0], inplace = True)
tgt_gp = mc_survey[(mc_survey['Q2'] == '35-39') | (mc_survey['Q2'] == '40-44')] 
tgt_gp.drop(tgt_gp.columns[0], axis = 1, inplace = True)
tgt_gp['Q9'] = tgt_gp['Q9'].str.replace('I do not wish to disclose my approximate yearly compensation','Compensation Undisclosed') # Simplified this answer

#Ordinal Categorical Data Mapping for Analysis
edu_dict = {'I prefer not to answer':0, 'No formal education past high school':1, 'Some college/university study without earning a bachelor’s degree':2, 
          'Professional degree':3, "Bachelor’s degree":4, "Master’s degree":5, 'Doctoral degree': 6}
comp_dict = {'Compensation Undisclosed':0, '0-10,000':1, '10-20,000':2, '20-30,000':3, '30-40,000':4, '40-50,000':5, '50-60,000':6,
           '60-70,000':7, '70-80,000':8, '80-90,000':9, '90-100,000':10, '100-125,000':11, '125-150,000':12, '150-200,000':13, 
           '200-250,000':14, '250-300,000':15, '300-400,000':16, '400-500,000':17, '500,000+':18}
exp_dict = {'0-1':0, '1-2':1, '2-3':2, '3-4':3, '4-5':4, '5-10':5, '10-15':6, '15-20':7, '20-25':8, '25-30':9, '30 +':10}
DS_dict = {'Definitely not':1, 'Probably not':2, 'Maybe':3, 'Probably yes':4, 'Definitely yes':5}
inv_exp = {v: k for k, v in exp_dict.items()} 
inv_comp = {v: k for k, v in comp_dict.items()}
inv_edu = {v: k for k, v in edu_dict.items()}

# Create columns for target categorical-ordinal data if needed
tgt_gp['edu_cat'] = tgt_gp['Q4'].map(edu_dict)
tgt_gp['comp_cat'] = tgt_gp['Q9'].map(comp_dict)
tgt_gp['exp_cat'] = tgt_gp['Q8'].map(exp_dict)
tgt_gp['DS_cat'] = tgt_gp['Q26'].map(DS_dict)

# Functions for analysis
def hl_analyze(df, col): # high-level analysis function to calculate percentages for attributes
    output = df[col].value_counts(normalize = True) * 100
    output.sort_values()
    return output

def multi_quest(df,lst,dct): # Function to aggregate the data into a dictionary from multiple option questions
    for col in lst:
        for item in df[col]:
            if type(item) != str:
                continue
            elif item not in dct:
                dct[item] = 1
            else:
                dct[item] += 1
    rslts = pd.Series(dct).sort_values(ascending = False) # Convert dictionary to series
    rslts = rslts/(rslts.sum()) * 100 
    return rslts

def clean_series(df, col): # Function specific to the questions regarding time spent learning ML/DS
    df[col] = df[col].fillna(-1)
    df[col] = df[col].astype(float)
    df[col] = df[col].astype(str)
    df[col] = df[col].replace('-1', np.nan)
    series = pd.Series(df[col]).value_counts()
    series.drop(['-1.0'], axis=0, inplace = True)
    series = series/(series.sum()) * 100
    return series
tot_respondents = len(tgt_gp.index)
print('For reference purposes, there are {} survey participants in the target group (35-44 year olds). \nThe %\'s calculated in the corresponding questions represent the % of valid responses (null values are excluded).'.format(tot_respondents))
# Gender breakdown
gender = hl_analyze(tgt_gp, 'Q1')
gender.columns = ['Gender']
gender_plt = gender.plot(kind = 'barh', figsize = (7,4), title= '35-44 Year-Old Respondents by Gender', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
# Locations breakdown
locs = hl_analyze(tgt_gp,'Q3')
locs = locs[0:10]
locs.plot(kind = 'barh', title = 'Top 10 Countries of 35-44 Year-Old Respondents', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
# Breakdown of roles
roles = hl_analyze(tgt_gp, 'Q6')
roles.plot(kind = 'barh', figsize = (8,6), title = 'Current Roles of 35-44 Year Old Respondents', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
#Heatmap of countries/roles
top_10 = ['United States of America', 'India', 'Other', 'Japan', 'United Kingdom of Great Britain and Northern Ireland', 
         'Germany', 'Brazil', 'Russia', 'Spain', 'Canada']
cty_role = tgt_gp[['Q3','Q6']]
cty_role = cty_role[(cty_role['Q3'].isin(top_10))]
cty_role.columns = ('Country', 'Role')
cty_v_role = pd.pivot_table(cty_role, index='Country', columns = 'Role', aggfunc='size', fill_value = 0)
plt.figure(figsize=(10,8))
sns.heatmap(cty_v_role, annot=True, fmt="d", linewidths=.5, cmap = 'Blues')
plt.title('Top 10 Represented Countries and Roles')
plt.xticks(rotation = 40, ha = 'right')
plt.show()
# Analysis of experience, using the ordinal data, next cell plots the data
exp = hl_analyze(tgt_gp, 'Q8')
exp.plot(kind = 'barh', figsize = (7,7), title = 'Experience in Current Roles of 35-44 Year Old Respondents', color = 'Blue')
plt.xlabel('% of respondents')
plt.ylabel('Years')
plt.show()
# Summarize those new to the field. 
print('{}% of 35-44 year old respondents have 4 years or less experience in their current roles.'.format(round(exp['0-1'] + exp['1-2'] + exp['2-3'] + exp['3-4'], 2)))
# Breakdown by sectors
sectors = hl_analyze(tgt_gp,'Q7')
sectors.plot(kind = 'barh', figsize = (7,7), title = 'Sectors Represented', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
# Select any activities that make up an important part of your role at work
q11_cols = ['Q11_Part_1', 'Q11_Part_2', 'Q11_Part_3', 'Q11_Part_4', 'Q11_Part_5',
       'Q11_Part_6', 'Q11_Part_7']
q11 = {}
q11_rslts = multi_quest(tgt_gp, q11_cols, q11)
q11_rslts.plot.barh(color = 'Blue', figsize = (5,5), title = 'Important Activities in Role at Work')
plt.xlabel('% of respondents')
plt.show()
#Time at work spent coding
q23 = hl_analyze(tgt_gp, 'Q23')
q23.plot(kind = 'barh', figsize = (5,5), title = '% of Time at Work/School Spent Coding', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
# Experience in writing code to analyze data
q24 = hl_analyze(tgt_gp, 'Q24')
# Years using ML methods
q25 = hl_analyze(tgt_gp, 'Q25')

fig = plt.figure(figsize = (10,12))
ax1 = fig.add_subplot(2,1,1)
q24.plot(kind = 'barh', ax = ax1, title = 'Coding Experience for Analyzing Data', color = 'Blue')
plt.xlabel('% of respondents')

ax2 = fig.add_subplot(2,1,2)
q25.plot(kind = 'barh', ax = ax2, title = 'Years Using ML Methods', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
code_low = q24[['< 1 year', '1-2 years', '3-5 years', 'I have never written code but I want to learn', 'I have never written code and I do not want to learn']].sum()
code_low = round(code_low, 2)
ml_high = q25[['5-10 years', '10-15 years', '20 + years']].sum()
ml_low = round(100 - ml_high, 2)
print('Of the respondents, {}% have less than 5 years of coding experience and {}% have less than 5 years of using ML methods.'.format(code_low, ml_low))
# Which types of data do you currently interact with most often at work or school?
q31_cols = ['Q31_Part_1', 'Q31_Part_2', 'Q31_Part_3',
       'Q31_Part_4', 'Q31_Part_5', 'Q31_Part_6', 'Q31_Part_7', 'Q31_Part_8',
       'Q31_Part_9', 'Q31_Part_10', 'Q31_Part_11', 'Q31_Part_12']
q31 = {}
q31_rslts = multi_quest(tgt_gp, q31_cols, q31)
# What is the type of data that you currently interact with most often at work or school?
q32 = hl_analyze(tgt_gp, 'Q32')

fig = plt.figure(figsize = (12,12))
ax1 = fig.add_subplot(2,1,1)
q31_rslts.plot(kind = 'barh', ax = ax1, title = 'Data Types with Most Interaction', color = 'Blue')
plt.xlabel('% of respondents')

ax2 = fig.add_subplot(2,1,2)
q32.plot(kind = 'barh', ax = ax2, title = 'Data Types Currently Interacting with Most Often', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
# Degree breakdown
edu = hl_analyze(tgt_gp, 'Q4')
edu.plot(kind = 'barh', title = 'Education Levels of 35-44 Year Old Respondents', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
high_ed = round(edu[['Bachelor’s degree', 'Doctoral degree', 'Master’s degree']].sum(), 2)
print('Bachelor\'s degrees or higher are held by {}% of the respondents.'.format(high_ed))
# Undergrad specialization breakdown
undergrad = hl_analyze(tgt_gp,'Q5')
undergrad.plot(kind = 'barh', figsize = (7,7), title = 'Undergrad Specialization of 35-44 Year Old Respondents', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
# What percentage of your current machine learning/data science training falls under each category? 
q35_pt1 = clean_series(tgt_gp, 'Q35_Part_1')
q35_pt2 = clean_series(tgt_gp, 'Q35_Part_2')
q35_pt3 = clean_series(tgt_gp, 'Q35_Part_3')
q35_pt4 = clean_series(tgt_gp, 'Q35_Part_4')
q35_pt5 = clean_series(tgt_gp, 'Q35_Part_5')
q35_pt6 = clean_series(tgt_gp, 'Q35_Part_6')

fig = plt.figure(figsize = (12,24))
ax1 = fig.add_subplot(3,2,1)
q35_pt1.plot(kind = 'barh', ax = ax1, title = '% of ML/DS Training: Self-taught', color = 'Blue')
plt.xlabel('% of respondents')
ax2 = fig.add_subplot(3,2,2)
q35_pt2.plot(kind = 'barh', ax = ax2, title = '% of ML/DS Training: Online Courses (Coursera, Udemy, edX, etc.)', color = 'Blue')
plt.xlabel('% of respondents')
ax3 = fig.add_subplot(3,2,3)
q35_pt3.plot(kind = 'barh', ax = ax3, title = '% of ML/DS Training: Work', color = 'Blue')
plt.xlabel('% of respondents')
ax4 = fig.add_subplot(3,2,4)
q35_pt4.plot(kind = 'barh', ax = ax4, title = '% of ML/DS Training: University', color = 'Blue')
plt.xlabel('% of respondents')
ax5 = fig.add_subplot(3,2,5)
q35_pt5.plot(kind = 'barh', ax = ax5, title = '% of ML/DS Training: Kaggle Competitions', color = 'Blue')
plt.xlabel('% of respondents')
ax6 = fig.add_subplot(3,2,6)
q35_pt6.plot(kind = 'barh', ax = ax6, title = '% of ML/DS Training: Other', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
st_high = round(q35_pt1[['50.0', '89.0', '85.0','55.0', '95.0', '65.0', '75.0', '90.0', '80.0', '70.0', '60.0', '100.0']].sum(), 2)
ol_high = round(q35_pt2[['50.0', '63.0', '59.0', '99.0', '73.0', '84.0', '85.0', '95.0', '55.0', '65.0', '75.0', '90.0', '100.0', '70.0', '60.0', '80.0']].sum(), 2)
print('{}% of respondents reported  50% or more of their ML/DS learning as self-taught and {}% reported 50% or more through online courses.'.format(st_high, ol_high))
# On which online platforms have you begun or completed data science courses? 
q36_cols = ['Q36_Part_1', 'Q36_Part_2', 'Q36_Part_3',
       'Q36_Part_4', 'Q36_Part_5', 'Q36_Part_6', 'Q36_Part_7', 'Q36_Part_8',
       'Q36_Part_9', 'Q36_Part_10', 'Q36_Part_11', 'Q36_Part_12',
       'Q36_Part_13']
q36 = {}
q36_rslts = multi_quest(tgt_gp, q36_cols, q36)
# Learning platform with most time spent on
q37 = hl_analyze(tgt_gp, 'Q37')
# How do you perceive the quality of online learning platforms and in-person bootcamps as compared to the quality of the education provided by traditional brick and mortar
q39_cols = ['Q39_Part_1', 'Q39_Part_2']
q39 = {}
q39_rslts = multi_quest(tgt_gp, q39_cols, q39)

fig = plt.figure(figsize = (6,18))
ax1 = fig.add_subplot(3,1,1)
q36_rslts.plot.barh(color = 'Blue', ax = ax1, title = 'Platforms Utilized for DS Courses')
plt.xlabel('% of respondents')
ax2 = fig.add_subplot(3,1,2)
q37_plt = q37.plot(kind = 'barh', ax = ax2, title = 'Learning Platform with Most Time Spent On', color = 'Blue')
plt.xlabel('% of respondents')
ax3 = fig.add_subplot(3,1,3)
q39_rslts.plot.barh(color = 'Blue', ax = ax3, title = 'Perceived Quality of Online Learning/In-Person Bootcamps as opposed to Brick and mortar Education')
plt.xlabel('% of respondents')
plt.show()
# Do you consider yourself to be a data scientist
q26 = pd.Series(tgt_gp['Q26']).value_counts(normalize = True).sort_values() * 100 #Did not use hl_analyze to match barplot with boxplot category codes

# Education vs considered a data scientist
ds_vs_edu = tgt_gp[['edu_cat','DS_cat']]
ds_vs_edu = ds_vs_edu[(ds_vs_edu['edu_cat'] > 0) & (ds_vs_edu['DS_cat'] > 0)]
ds_vs_edu.sort_values('edu_cat')

fig = plt.figure(figsize = (12,6))
ax1 = fig.add_subplot(1,2,1)
q26.plot(kind = 'barh', ax = ax1, title = 'Do you consider yourself to be a data scientist', color = 'Blue')
plt.xlabel('% of respondents')
ax2 = fig.add_subplot(1,2,2)
sns.boxplot(x=ds_vs_edu['edu_cat'], y=ds_vs_edu['DS_cat'], data=ds_vs_edu, ax = ax2)
plt.title('Data Scientist vs Education')
plt.ylabel('Considered a Data Scientist')
plt.xlabel('Education Level')
plt.show()
# What data scientists do
data_scientists = tgt_gp[(tgt_gp['Q6'] == 'Data Scientist')]
ug_spec = hl_analyze(data_scientists, 'Q5')

# What data scientists do opposed to others
others = tgt_gp[~(tgt_gp['Q6'] == 'Data Scientist')]
q11_for_DS = multi_quest(data_scientists, q11_cols, q11)
q11_for_others = multi_quest(others, q11_cols, q11)

# Declared data scientists roles
declared_DS = tgt_gp[(tgt_gp['Q26'] == 'Definitely yes')]
roles_for_dDS = hl_analyze(declared_DS, 'Q6')

fig = plt.figure(figsize = (10,20))
ax1 = fig.add_subplot(3,1,1)
ug_spec.plot(kind='barh', ax = ax1, color='blue', title = 'Data Scientists\' Undergrad Specializations')
plt.xlabel('% of respondents')

ax2 = fig.add_subplot(3,1,2)
q11_for_DS.plot(kind='barh', ax = ax2, color='red', width=0.2, position=1, label = 'Data Scientists')
q11_for_others.plot(kind='barh', color='blue', width=0.2, position=0, label = 'Others')
plt.title('Important Activities at Work')
plt.xlabel('% of respondents')
plt.legend()

ax3 = fig.add_subplot(3,1,3)
roles_for_dDS.plot(kind='barh', ax = ax3, color='blue', title = 'Declared Data Scientists\' Roles')
plt.xlabel('% of respondents')
plt.show()

# Compensation breakdown
comp = pd.Series(tgt_gp['comp_cat']).value_counts(normalize = True).sort_index() * 100
comp = comp.rename(inv_comp)
comp_plt = comp.plot(kind = 'barh', title = 'Compensation of 35-44 Year Old Respondents',figsize = (7,7), color = 'Blue')
plt.xlabel('% of respondents')
plt.ylabel('USD')
plt.show()
# Capture those earning 100k and higher
high_earners = round(comp['100-125,000'] + comp['125-150,000'] + comp['150-200,000'] + comp['200-250,000'] + comp['250-300,000'] + comp['300-400,000'] + comp['400-500,000'] + comp['500,000+'], 2)
print('{}% earn over 100k among 35-44 year olds, while {}% chose not to disclose their compensation.'.format(high_earners, round(comp['Compensation Undisclosed'], 2)))
# Top 10 countries by compensation
cty_comp = tgt_gp[['Q3','comp_cat']]
cty_comp = cty_comp[(cty_comp['comp_cat'] > 0)]
cty_comp.columns = ('Country', 'Compensation')
cty_v_comp = cty_comp.groupby(['Country'],as_index= True).mean().sort_values('Compensation', ascending = False)
cty_v_comp = cty_v_comp[0:10]
cty_v_comp.plot(kind = 'barh', figsize = (5,5), legend = False, color = 'blue')
plt.title('Top 10 Countries for Average Compensation in USD (excluding respondents that did not disclose compensation)')
plt.xlabel('Compensation Range Category')
plt.show()
# Prepare analysis of education vs. compensation
edu_vs_comp = tgt_gp[['edu_cat','comp_cat']]
edu_vs_comp.columns = ('Education', 'Compensation')
edu_vs_comp = edu_vs_comp[(edu_vs_comp['Education'] > 0) & (edu_vs_comp['Compensation'] > 0)]
plt.figure(figsize=(6,4))
e_v_c = sns.swarmplot(x=edu_vs_comp['Education'], y=edu_vs_comp['Compensation'], data=edu_vs_comp)
plt.title('Education vs Compensation')
plt.show()
# Compare roles and compensation
pos_comp = tgt_gp[(tgt_gp['comp_cat'] > 0)]
gp_role_comp = pos_comp[['Q6', 'comp_cat']]
gp_role_comp.columns = ['Role', 'Comp']
sns.boxplot(x=gp_role_comp['Role'], y=gp_role_comp['Comp'], data=gp_role_comp)
plt.title('Roles vs Compensation')
plt.ylabel('Compensation Level')
plt.xlabel('Role')
plt.xticks(rotation = 40, ha = 'right')
plt.show()
# Data scientists' earnings
ds_earnings = data_scientists[(data_scientists['comp_cat'] > 0)]
ds_earnings = ds_earnings['comp_cat'].median()
ds_earnings = inv_comp[ds_earnings]
print('The overall median salary for respondents 35-44 years old in a data scientist role is {} USD (excluding compensation undisclosed cases).'.format(ds_earnings))
# Compare experience and compensation
gp_ex_comp = pos_comp[['exp_cat', 'comp_cat']]
gp_ex_comp.columns = ['Experience', 'Comp']
sns.boxplot(x=gp_ex_comp['Experience'], y=gp_ex_comp['Comp'], data=gp_ex_comp)
plt.title('Experience vs Compensation')
plt.ylabel('Compensation Level')
plt.xlabel('Experience')
plt.show()
# What is the primary tool that you use at work or school to analyze data?
q12_mc = hl_analyze(tgt_gp, 'Q12_MULTIPLE_CHOICE')
q12_mc.plot(kind = 'barh', figsize = (5,5), title = 'Primary Tool used to Analyze Data', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
# Which of the following integrated development environments (IDE's) have you used at work or school in the last 5 years? 
q13_cols = ['Q13_Part_1', 'Q13_Part_2', 'Q13_Part_3', 'Q13_Part_4', 'Q13_Part_5', 'Q13_Part_6', 'Q13_Part_7', 'Q13_Part_8', 'Q13_Part_9', 'Q13_Part_10', 'Q13_Part_11', 'Q13_Part_12', 'Q13_Part_13', 'Q13_Part_14', 'Q13_Part_15']
q13 = {}
q13_rslts = multi_quest(tgt_gp, q13_cols, q13)
q13_rslts.plot.barh(figsize = (5,5), title = 'IDEs used in the Past 5 Years', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
# What programming languages do you use on a regular basis?
q16_cols = ['Q16_Part_1', 'Q16_Part_2', 'Q16_Part_3', 'Q16_Part_4', 'Q16_Part_5', 'Q16_Part_6', 'Q16_Part_7', 'Q16_Part_8',
       'Q16_Part_9', 'Q16_Part_10', 'Q16_Part_11', 'Q16_Part_12',
       'Q16_Part_13', 'Q16_Part_14', 'Q16_Part_15', 'Q16_Part_16',
       'Q16_Part_17', 'Q16_Part_18']
q16 = {}
q16_rslts = multi_quest(tgt_gp, q16_cols, q16)
#Programming language most often used
q17 = hl_analyze(tgt_gp, 'Q17')


fig = plt.figure(figsize = (12,14))
ax1 = fig.add_subplot(2,1,1)
q16_rslts.plot(kind = 'barh', ax = ax1, title = 'Programming Languages used on a Regular Basis', color = 'Blue')
plt.xlabel('% of respondents')
ax2 = fig.add_subplot(2,1,2)
q17.plot(kind = 'barh', ax = ax2, title = 'Programming Languages used Most Often', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
#Machine learning frameworks in the past 5 years
q19_cols = ['Q19_Part_1', 'Q19_Part_2', 'Q19_Part_3', 'Q19_Part_4', 'Q19_Part_5', 'Q19_Part_6', 'Q19_Part_7', 'Q19_Part_8', 'Q19_Part_9', 'Q19_Part_10', 'Q19_Part_11', 'Q19_Part_12', 'Q19_Part_13', 'Q19_Part_14', 'Q19_Part_15', 'Q19_Part_16', 'Q19_Part_17', 'Q19_Part_18', 'Q19_Part_19']
q19 = {}
q19_rslts = multi_quest(tgt_gp, q19_cols, q19)
# ML library used most
q20 = hl_analyze(tgt_gp, 'Q20')


fig = plt.figure(figsize = (16,6))
ax1 = fig.add_subplot(1,2,1)
q19_rslts.plot.barh(color = 'Blue', ax = ax1, title = 'ML Frameworks used in the Past 5 Years')
plt.xlabel('% of respondents')
ax2 = fig.add_subplot(1,2,2)
q20_plt = q20.plot(kind = 'barh', ax = ax2, title = 'ML Library Most Used', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
# Which of the following machine learning products have you used at work or school in the last 5 years?
q28_cols = ['Q28_Part_1', 'Q28_Part_2', 'Q28_Part_3',
       'Q28_Part_4', 'Q28_Part_5', 'Q28_Part_6', 'Q28_Part_7', 'Q28_Part_8',
       'Q28_Part_9', 'Q28_Part_10', 'Q28_Part_11', 'Q28_Part_12',
       'Q28_Part_13', 'Q28_Part_14', 'Q28_Part_15', 'Q28_Part_16',
       'Q28_Part_17', 'Q28_Part_18', 'Q28_Part_19', 'Q28_Part_20',
       'Q28_Part_21', 'Q28_Part_22', 'Q28_Part_23', 'Q28_Part_24',
       'Q28_Part_25', 'Q28_Part_26', 'Q28_Part_27', 'Q28_Part_28',
       'Q28_Part_29', 'Q28_Part_30', 'Q28_Part_31', 'Q28_Part_32',
       'Q28_Part_33', 'Q28_Part_34', 'Q28_Part_35', 'Q28_Part_36',
       'Q28_Part_37', 'Q28_Part_38', 'Q28_Part_39', 'Q28_Part_40',
       'Q28_Part_41', 'Q28_Part_42', 'Q28_Part_43']
q28 = {}
q28_rslts = multi_quest(tgt_gp, q28_cols, q28)
q28_rslts.plot.barh(color = 'Blue', figsize = (8,10), title = 'ML Products used in the Past 5 Years')
plt.xlabel('% of respondents')
plt.show()
# Which of the following relational database products have you used at work or school in the last 5 years?
q29_cols = ['Q29_Part_1', 'Q29_Part_2', 'Q29_Part_3', 'Q29_Part_4', 'Q29_Part_5',
       'Q29_Part_6', 'Q29_Part_7', 'Q29_Part_8', 'Q29_Part_9', 'Q29_Part_10',
       'Q29_Part_11', 'Q29_Part_12', 'Q29_Part_13', 'Q29_Part_14',
       'Q29_Part_15', 'Q29_Part_16', 'Q29_Part_17', 'Q29_Part_18',
       'Q29_Part_19', 'Q29_Part_20', 'Q29_Part_21', 'Q29_Part_22',
       'Q29_Part_23', 'Q29_Part_24', 'Q29_Part_25', 'Q29_Part_26',
       'Q29_Part_27', 'Q29_Part_28']
q29 = {}
q29_rslts = multi_quest(tgt_gp, q29_cols, q29)
q29_rslts.plot.barh(color = 'Blue', figsize = (7,8), title = 'Relational DB Products used in the Past 5 Years')
plt.xlabel('% of respondents')
plt.show()
#Which of the following big data and analytics products have you used at work or school in the last 5 years? 
q30_cols = ['Q30_Part_1',
       'Q30_Part_2', 'Q30_Part_3', 'Q30_Part_4', 'Q30_Part_5', 'Q30_Part_6',
       'Q30_Part_7', 'Q30_Part_8', 'Q30_Part_9', 'Q30_Part_10', 'Q30_Part_11',
       'Q30_Part_12', 'Q30_Part_13', 'Q30_Part_14', 'Q30_Part_15',
       'Q30_Part_16', 'Q30_Part_17', 'Q30_Part_18', 'Q30_Part_19',
       'Q30_Part_20', 'Q30_Part_21', 'Q30_Part_22', 'Q30_Part_23',
       'Q30_Part_24', 'Q30_Part_25']
q30 = {}
q30_rslts = multi_quest(tgt_gp, q30_cols, q30)
q30_rslts.plot.barh(color = 'Blue', figsize = (7,8), title = 'Big Data and Analytics Products used in the Past 5 Years')
plt.xlabel('% of respondents')
plt.show()
#Visualization libraries used the most
q22 = hl_analyze(tgt_gp, 'Q22')
q22.plot(kind = 'barh', figsize = (5,5), title = 'Visualization Libraries most used', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
#What programming language would you recommend an aspiring data scientist to learn first?
q18 = hl_analyze(tgt_gp, 'Q18')
q18.plot(kind = 'barh', figsize = (5,5), title = 'Recommended Language for an Aspiring Data Scientist to Learn First', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
# Preferred Media Sources for DS
q38_cols = ['Q38_Part_1',
       'Q38_Part_2', 'Q38_Part_3', 'Q38_Part_4', 'Q38_Part_5', 'Q38_Part_6',
       'Q38_Part_7', 'Q38_Part_8', 'Q38_Part_9', 'Q38_Part_10', 'Q38_Part_11',
       'Q38_Part_12', 'Q38_Part_13', 'Q38_Part_14', 'Q38_Part_15',
       'Q38_Part_16', 'Q38_Part_17', 'Q38_Part_18', 'Q38_Part_19',
       'Q38_Part_20', 'Q38_Part_21', 'Q38_Part_22']
q38 = {}
q38_rslts = multi_quest(tgt_gp, q38_cols, q38)
q38_rslts.plot.barh(color = 'Blue', figsize = (5,5), title = 'Preferred DS Media Sources')
plt.xlabel('% of respondents')
plt.show()
# Which better demonstrates expertise in data science: academic achievements or independent projects?
q40 = hl_analyze(tgt_gp, 'Q40')
q40.plot(kind = 'barh', figsize = (5,5), title = 'Which better Demonstrates Expertise in DS: Academic Acheivements of Independent Projects?', color = 'Blue')
plt.xlabel('% of respondents')
plt.show()
# Experienced in ML vs Q48
expMLk = tgt_gp[['Q25','Q48']]
expMLk.columns = ['Experience with ML','View on ML']
expMLk = pd.pivot_table(expMLk, index='View on ML', columns='Experience with ML', aggfunc='size', fill_value = 0)
plt.figure(figsize=(5,5))
sns.heatmap(expMLk, annot=True, fmt="d", linewidths=.5, cmap = 'Blues')
plt.xticks(rotation = 40, ha = 'right')
plt.title('Experience and Perspectives on ML (by count)')
plt.show()
