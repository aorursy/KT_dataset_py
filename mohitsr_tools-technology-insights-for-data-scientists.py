#Importing required packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.graph_objects as go

import plotly.express as px

import copy

%matplotlib inline
#Import the raw data

survey_schema_raw = pd.read_csv('/kaggle/input/kaggle-survey-2019/survey_schema.csv')

responses_raw = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv', low_memory = False)
#Deleting the row beneath header and creating a new data frame

responses = responses_raw.drop(0)

#rename the time column

responses = responses.rename(columns = {'Time from Start to Finish (seconds)' : 'Time'})

#Make the time column numeric

responses['Time']=pd.to_numeric(responses['Time'])
#Respondants from Age and Gender perspective

plt.figure(figsize=(18,6))

sns.countplot(responses.Q1,hue=responses.Q2)

plt.xlabel('Age Groups')

plt.ylabel('Number of Responses')

plt.title('Age and Gender wise Responses')

plt.show()
#Number of Respondents from geaographies

plt.figure(figsize=(18,6))

chart=(sns.countplot(responses.Q3))

chart.set_xticklabels(chart.get_xticklabels(),rotation=45,horizontalalignment='right')

plt.xlabel('Countries')

plt.ylabel('Number of Responses')

plt.title('Country wise Responses')

plt.show()
plt.figure(figsize=(18,6))

chart=(sns.countplot(responses.Q4,hue=responses.Q5))

chart.set_xticklabels(chart.get_xticklabels(),rotation=45,horizontalalignment='right')

plt.xlabel('Qualifications')

plt.ylabel('Number of Responses')

plt.title('Qualification and Role wise Responses')

plt.show()
plt.figure(figsize=(18,6))

chart = (sns.countplot(responses.Q10))

chart.set_xticklabels(chart.get_xticklabels(),rotation=45,horizontalalignment='right')

plt.xlabel('Salary Categories')

plt.ylabel('Number of Responses')

plt.title('Salary category wise Responses')

plt.show()
#Data Preparation for analysis

#Removing other columns from dataframe

col_list = list(responses.columns)

other_cols = []

for i in range(len(col_list)):

    if ("OTHER" in col_list[i]):

        other_cols.append(col_list[i])

responses = responses.drop(other_cols,axis =1)
# Create dictionary for all columns

all_cols = {'Q1': 'AGE_GROUP', 'Q2': 'Gender', 'Q3':'Country','Q4' : 'Qualification','Q5' : 'Role', 'Q10' : 'Salary', 'Q15': 'Experience'}

dict_df = {'Q1': 'AGE_GROUP_df', 'Q2': 'Gender_df', 'Q3':'Country_df','Q4' : 'Qualification_df','Q5' : 'Role_df', 'Q10' : 'Salary_df', 'Q15': 'Experience_df'}
Base_members = ['Q1','Q2','Q3','Q4','Q5','Q10','Q15']
for mem in Base_members:

    temp_df = pd.DataFrame()

    temp_list = list(responses[mem].unique())

    for val in temp_list:

        if mem == 'Q1':

            temp_df = temp_df.append(pd.DataFrame(responses.query("Q1==@val").count()).transpose(),ignore_index=True)

        if mem == 'Q2':

            temp_df = temp_df.append(pd.DataFrame(responses.query("Q2==@val").count()).transpose(),ignore_index=True)

        if mem == 'Q3':

            temp_df = temp_df.append(pd.DataFrame(responses.query("Q3==@val").count()).transpose(),ignore_index=True)

        if mem == 'Q4':

            temp_df = temp_df.append(pd.DataFrame(responses.query("Q4==@val").count()).transpose(),ignore_index=True)

        if mem == 'Q5':

            temp_df = temp_df.append(pd.DataFrame(responses.query("Q5==@val").count()).transpose(),ignore_index=True)

        if mem == 'Q10':

            temp_df = temp_df.append(pd.DataFrame(responses.query("Q10==@val").count()).transpose(),ignore_index=True)

        if mem == 'Q15':

            temp_df = temp_df.append(pd.DataFrame(responses.query("Q15==@val").count()).transpose(),ignore_index=True)

    if mem == 'Q1':

        AGE_GROUP_df= copy.deepcopy(temp_df)

        AGE_GROUP_df.apply(pd.to_numeric).info()

        for i in range(len(AGE_GROUP_df.index)):

            AGE_GROUP_df.iloc[i,8:] = AGE_GROUP_df.iloc[i,8:].div(AGE_GROUP_df['Q2'].iloc[i])

        AGE_GROUP_df.iloc[:,8:] = AGE_GROUP_df.iloc[:,8:]*100

        AGE_GROUP_df['Val'] = temp_list

    if mem == 'Q2':

        Gender_df = copy.deepcopy(temp_df)

        Gender_df.apply(pd.to_numeric).info()

        for i in range(len(Gender_df.index)):

            Gender_df.iloc[i,8:] = Gender_df.iloc[i,8:].div(Gender_df['Q2'].iloc[i])

        Gender_df.iloc[:,8:] = Gender_df.iloc[:,8:]*100

        Gender_df['Val'] = temp_list

    if mem == 'Q3':

        Country_df = copy.deepcopy(temp_df)

        Country_df.apply(pd.to_numeric).info()

        for i in range(len(Country_df.index)):

            Country_df.iloc[i,8:] = Country_df.iloc[i,8:].div(Country_df['Q2'].iloc[i])

        Country_df.iloc[:,8:] = Country_df.iloc[:,8:]*100

        Country_df['Val'] = temp_list

    if mem == 'Q4':

        Qualification_df= copy.deepcopy(temp_df)

        Qualification_df.apply(pd.to_numeric).info()

        for i in range(len(Qualification_df.index)):

            Qualification_df.iloc[i,8:] = Qualification_df.iloc[i,8:].div(Qualification_df['Q2'].iloc[i])

        Qualification_df.iloc[:,8:] = Qualification_df.iloc[:,8:]*100

        Qualification_df['Val'] = temp_list

    if mem == 'Q5':

        Role_df= copy.deepcopy(temp_df)

        Role_df.apply(pd.to_numeric).info()

        for i in range(len(Role_df.index)):

            Role_df.iloc[i,8:] = Role_df.iloc[i,8:].div(Role_df['Q2'].iloc[i])

        Role_df.iloc[:,8:] = Role_df.iloc[:,8:]*100

        Role_df['Val'] = temp_list

    if mem == 'Q10':

        Salary_df= copy.deepcopy(temp_df)

        Salary_df.apply(pd.to_numeric).info()

        for i in range(len(Salary_df.index)):

            Salary_df.iloc[i,8:] = Salary_df.iloc[i,8:].div(Salary_df['Q2'].iloc[i])

        Salary_df.iloc[:,8:] = Salary_df.iloc[:,8:]*100

        Salary_df['Val'] = temp_list

    if mem == 'Q15':

        Experience_df = copy.deepcopy(temp_df)

        Experience_df.apply(pd.to_numeric).info()

        for i in range(len(Experience_df.index)):

            Experience_df.iloc[i,8:] = Experience_df.iloc[i,8:].div(Experience_df['Q2'].iloc[i])

        Experience_df.iloc[:,8:] = Experience_df.iloc[:,8:]*100

        Experience_df['Val'] = temp_list
#Finding Strong Associations (More than 40% of same category resposes)

assoc_df = pd.DataFrame(columns=['Entity', 'Val', 'Col_name','Association' ])

for mem in Base_members:

    if mem == 'Q1':

        for i in range(len(AGE_GROUP_df.index)):

            for j in range(len(AGE_GROUP_df.columns)):

                if ((j > 8 ) & (AGE_GROUP_df.columns[j] != 'Val')):

                    if AGE_GROUP_df.iloc[i,j] > 50:

                        assoc_df = assoc_df.append({'Entity' :mem, 'Val':AGE_GROUP_df.Val[i],'Col_name':AGE_GROUP_df.columns[j], 'Association':AGE_GROUP_df.iloc[i,j]} ,ignore_index=True)

    if mem == 'Q2':

        for i in range(len(Gender_df.index)):

            for j in range(len(Gender_df.columns)):

                 if ((j > 8 ) & (Gender_df.columns[j] != 'Val')):

                        if Gender_df.iloc[i,j] > 50:

                            assoc_df = assoc_df.append({'Entity' :mem,'Val':Gender_df.Val[i], 'Col_name':Gender_df.columns[j], 'Association':Gender_df.iloc[i,j] },ignore_index=True)

    if mem == 'Q3':

        for i in range(len(Country_df.index)):

            for j in range(len(Country_df.columns)):

                 if ((j > 8 ) & (Country_df.columns[j] != 'Val')):

                        if Country_df.iloc[i,j] > 50:

                            assoc_df = assoc_df.append({'Entity' :mem, 'Val':Country_df.Val[i],'Col_name':Country_df.columns[j], 'Association':Country_df.iloc[i,j] },ignore_index=True)

    

    if mem == 'Q4':

        for i in range(len(Qualification_df.index)):

            for j in range(len(Qualification_df.columns)):

                 if ((j > 8 ) & (Qualification_df.columns[j] != 'Val')):

                        if Qualification_df.iloc[i,j] > 50:

                            assoc_df = assoc_df.append({'Entity' :mem, 'Val':Qualification_df.Val[i],'Col_name':Qualification_df.columns[j], 'Association':Qualification_df.iloc[i,j] },ignore_index=True)

    if mem == 'Q5':

        for i in range(len(Role_df.index)):

            for j in range(len(Role_df.columns)):

                 if ((j > 8 ) & (Role_df.columns[j] != 'Val')):

                        if Role_df.iloc[i,j] > 50:

                            assoc_df = assoc_df.append({'Entity' :mem,'Val':Role_df.Val[i], 'Col_name':Role_df.columns[j], 'Association':Role_df.iloc[i,j] },ignore_index=True)

    if mem == 'Q10':

        for i in range(len(Salary_df.index)):

            for j in range(len(Salary_df.columns)):

                 if ((j > 8 ) & (Salary_df.columns[j] != 'Val')):

                        if Salary_df.iloc[i,j] > 50:

                            assoc_df = assoc_df.append({'Entity' :mem, 'Val':Salary_df.Val[i], 'Col_name':Salary_df.columns[j], 'Association':Salary_df.iloc[i,j] },ignore_index=True)

    if mem == 'Q15':

        for i in range(len(Experience_df.index)):

            for j in range(len(Experience_df.columns)):

                 if ((j > 8 ) & (Experience_df.columns[j] != 'Val')):

                        if Experience_df.iloc[i,j] > 50:

                            assoc_df = assoc_df.append({'Entity' :mem, 'Val':Experience_df.Val[i],'Col_name':Experience_df.columns[j], 'Association':Experience_df.iloc[i,j] },ignore_index=True)

    

    
#Removing columns which are not relevant for this relationship

col_remove_list = ['Q10', 'Q11',  'Q14', 'Q14_Part_1_TEXT', 'Q14_Part_2_TEXT', 'Q14_Part_3_TEXT', 'Q14_Part_4_TEXT',

 'Q14_Part_5_TEXT', 'Q15', 'Q19', 'Q22', 'Q23']

assoc_df = assoc_df.query("Col_name not in @col_remove_list") 
assoc_df.query("Entity == 'Q1'").count()

#assoc_df.query("Col_name == 'Q12_Part_4'")
#Analysis for most popular 

assoc_df.sort_values(by=['Association'],ascending=False).head(5)
python_age_grp= pd.DataFrame(assoc_df.query("Entity == 'Q1' and Col_name == 'Q18_Part_1'"))
plt.figure(figsize=(18,6))

chart = (sns.barplot(y=python_age_grp.Association,x=python_age_grp.Val))

chart.set_xticklabels(chart.get_xticklabels(),rotation=45,horizontalalignment='right')

plt.xlabel('Age Groups')

plt.ylabel('% of Popularity')

plt.title('Popularity of Python amongst various Age Groups')

plt.show()
python_role_grp= pd.DataFrame(assoc_df.query("Entity == 'Q5' and Col_name == 'Q18_Part_1'"))
plt.figure(figsize=(18,6))

chart = (sns.barplot(y=python_role_grp.Association,x=python_role_grp.Val))

chart.set_xticklabels(chart.get_xticklabels(),rotation=45,horizontalalignment='right')

plt.xlabel('Roles')

plt.ylabel('% of Popularity')

plt.title('Popularity of Python amongst people playing various Roles')

plt.show()
python_Sal_grp= pd.DataFrame(assoc_df.query("Entity == 'Q10' and Col_name == 'Q18_Part_1'"))
plt.figure(figsize=(18,6))

chart = (sns.barplot(y=python_Sal_grp.Association,x=python_Sal_grp.Val))

chart.set_xticklabels(chart.get_xticklabels(),rotation=45,horizontalalignment='right')

plt.xlabel('Salary Groups')

plt.ylabel('% of Popularity')

plt.title('Popularity of Python amongst people with various salary ranges')

plt.show()
python_Exp_grp= pd.DataFrame(assoc_df.query("Entity == 'Q15' and Col_name == 'Q18_Part_1'"))
plt.figure(figsize=(18,6))

chart = (sns.barplot(y=python_Exp_grp.Association,x=python_Exp_grp.Val))

chart.set_xticklabels(chart.get_xticklabels(),rotation=45,horizontalalignment='right')

plt.xlabel('Experience Levels')

plt.ylabel('% of Popularity')

plt.title('Popularity of Python amongst people with various experience levels')

plt.show()
ML_age_grp= pd.DataFrame(assoc_df.query("Entity == 'Q1' and Col_name == 'Q24_Part_1'"))
plt.figure(figsize=(18,6))

chart = (sns.barplot(y=ML_age_grp.Association,x=ML_age_grp.Val))

chart.set_xticklabels(chart.get_xticklabels(),rotation=45,horizontalalignment='right')

plt.xlabel('Age Groups')

plt.ylabel('% of Popularity')

plt.title('Popularity of ML Linear Regression amongst various age groups')

plt.show()
ML_role_grp= pd.DataFrame(assoc_df.query("Entity == 'Q5' and Col_name == 'Q24_Part_1'"))
plt.figure(figsize=(18,6))

chart = (sns.barplot(y=ML_role_grp.Association,x=ML_role_grp.Val))

chart.set_xticklabels(chart.get_xticklabels(),rotation=45,horizontalalignment='right')

plt.xlabel('Roles')

plt.ylabel('% of Popularity')

plt.title('Popularity of Linear Regression Algorithm amongst people playing various Roles')

plt.show()
ML_Sal_grp= pd.DataFrame(assoc_df.query("Entity == 'Q10' and Col_name == 'Q24_Part_1'"))
plt.figure(figsize=(18,6))

chart = (sns.barplot(y=ML_Sal_grp.Association,x=ML_Sal_grp.Val))

chart.set_xticklabels(chart.get_xticklabels(),rotation=45,horizontalalignment='right')

plt.xlabel('Salary Groups')

plt.ylabel('% of Popularity')

plt.title('Popularity of Linear Regression amongst people with various salary ranges')

plt.show()
ML_Exp_grp= pd.DataFrame(assoc_df.query("Entity == 'Q15' and Col_name == 'Q24_Part_1'"))
plt.figure(figsize=(18,6))

chart = (sns.barplot(y=ML_Exp_grp.Association,x=ML_Exp_grp.Val))

chart.set_xticklabels(chart.get_xticklabels(),rotation=45,horizontalalignment='right')

plt.xlabel('Experience Levels')

plt.ylabel('% of Popularity')

plt.title('Popularity of Linear Regression amongst people with various experience levels')

plt.show()
jupyter_age_grp= pd.DataFrame(assoc_df.query("Entity == 'Q1' and Col_name == 'Q16_Part_1'"))

plt.figure(figsize=(18,6))

chart = (sns.barplot(y=jupyter_age_grp.Association,x=jupyter_age_grp.Val))

chart.set_xticklabels(chart.get_xticklabels(),rotation=45,horizontalalignment='right')

plt.xlabel('Age Groups')

plt.ylabel('% of Popularity')

plt.title('Popularity of jupyter amongst various Age Groups')

plt.show()
jupyter_role_grp= pd.DataFrame(assoc_df.query("Entity == 'Q5' and Col_name == 'Q16_Part_1'"))

plt.figure(figsize=(18,6))

chart = (sns.barplot(y=jupyter_role_grp.Association,x=jupyter_role_grp.Val))

chart.set_xticklabels(chart.get_xticklabels(),rotation=45,horizontalalignment='right')

plt.xlabel('Roles')

plt.ylabel('% of Popularity')

plt.title('Popularity of jupyter amongst people playing various Roles')

plt.show()
jupyter_Sal_grp= pd.DataFrame(assoc_df.query("Entity == 'Q10' and Col_name == 'Q16_Part_1'"))

plt.figure(figsize=(18,6))

chart = (sns.barplot(y=jupyter_Sal_grp.Association,x=jupyter_Sal_grp.Val))

chart.set_xticklabels(chart.get_xticklabels(),rotation=45,horizontalalignment='right')

plt.xlabel('Salary Groups')

plt.ylabel('% of Popularity')

plt.title('Popularity of jupyter amongst people with various salary ranges')

plt.show()
SQL_grp= pd.DataFrame(assoc_df.query("Col_name == 'Q18_Part_3'"))



plt.figure(figsize=(18,6))

chart = (sns.barplot(y=SQL_grp.Association,x=SQL_grp.Val))

chart.set_xticklabels(chart.get_xticklabels(),rotation=45,horizontalalignment='right')

plt.xlabel('Various Categories')

plt.ylabel('% of Popularity')

plt.title('Popularity of SQL amongst various categories')

plt.show()