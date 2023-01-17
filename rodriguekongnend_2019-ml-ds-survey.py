import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from locale import atof

%matplotlib inline



#set ggplot style

#plt.style.use('ggplot')



# Load dataset



multiple_choice_responses = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')
len(multiple_choice_responses) - 1
multiple_choice_responses = multiple_choice_responses.drop(0, axis=0)

multiple_choice_responses = multiple_choice_responses.reset_index(drop=True)

multiple_choice_responses["Q5"].value_counts().plot.barh()
# Get a dataframe with responding who make at least one ML activities

df = multiple_choice_responses.copy()

only_ml_activities = multiple_choice_responses.copy()

for rows in range(len(multiple_choice_responses)):

    if(pd.isna(df.iloc[rows]['Q9_Part_1']) & pd.isna(df.iloc[rows]['Q9_Part_2']) 

       & pd.isna(df.iloc[rows]['Q9_Part_3']) & pd.isna(df.iloc[rows]['Q9_Part_4']) 

       & pd.isna(df.iloc[rows]['Q9_Part_5']) & pd.isna(df.iloc[rows]['Q9_Part_6'])

       & pd.isna(df.iloc[rows]['Q9_Part_7']) & pd.isna(df.iloc[rows]['Q9_Part_8'])):

        only_ml_activities.drop(rows, axis=0, inplace=True)

only_ml_activities = only_ml_activities.reset_index(drop=True)

len(only_ml_activities)
fig, ax = plt.subplots(figsize=(15,10))

ax = sns.countplot(x="Q2", hue="Q5", data=only_ml_activities, palette="muted", order=only_ml_activities["Q2"].unique())
def question_column(column, df):

    count = 0

    for col in range(len(df.columns)):

        if column in df.columns[col]:

            new=df[[df.columns[col]]]

            if count==0:

                new_df = pd.DataFrame(new)

                count=+1

            else:

                new_df = new_df.join(new)

    return new_df
def change_column_name(df):

    name = []

    dfs = df.copy()

    for column in df.columns:

        for row in range(len(df)):

            if pd.isna(df.iloc[row][column]):

                continue

            else:

                dfs=df.rename(columns = {column : df.iloc[row][column]}, inplace=True)

                break

    return df 
def change_value(df):

    for column in df.columns:

        for row in range(len(df)):

            if pd.isna(df.iloc[row][column]):

                df.iloc[row][column] = 0

            else:

                df.iloc[row][column] = 1

    return df 
# Get activity columns from DataFrame only_ml_activitiabses

columns = question_column("Q9", only_ml_activities)

# Change columns name

change_name = change_column_name(columns)

# Change cell value 0 or 1

Q9 = change_value(change_name)

# Drop the last column

Q9_drop = Q9.drop(Q9.columns[-1], axis=1)

# Concatenate activity columns with respondent role columns

Q5_Q9 = pd.concat([only_ml_activities["Q5"], Q9_drop], axis=1)
# Group by role

groupe_role = Q5_Q9.groupby("Q5").sum().transpose()

# Plotting for data scientist role

groupe_role["Data Scientist"].plot.barh()
# Plotting all role whose make least a machine learning activities

only_ml_activities["Q5"].value_counts().plot.barh()
# Plotting activities and total of role for each activity

fig, ax = plt.subplots(figsize=(15,10))

Q5_Q9.groupby("Q5").sum().plot(ax=ax)

ax.set_xlabel("ROLE")

ax.set_ylabel("TOTAL")
# Concatenate role and responsible for data science workloads columns

concat_Q5_Q7 = pd.concat([only_ml_activities["Q5"], only_ml_activities["Q7"]], axis=1)

group_Q5_Q7 = concat_Q5_Q7.groupby(["Q5", "Q7"]).size()

# Get DataFrame for data scientist only 

data_scientist = concat_Q5_Q7[concat_Q5_Q7["Q5"]=="Data Scientist"]

# Get DataFrame for all who work in a team

in_team = concat_Q5_Q7[concat_Q5_Q7["Q7"]!="0"]

# Get for data scientist only who work in a team

data_scientist_in_team = in_team[in_team["Q5"]=="Data Scientist"]

# Compute the percent of data scientist who work in a team

percent_data_scientist =(len(data_scientist_in_team) / len(data_scientist)) * 100

print(percent_data_scientist)
group_Q5_Q7["Data Scientist"].plot.barh()
# Plotting for all role

fig, ax = plt.subplots(figsize=(15,5))

concat_Q5_Q7.groupby(["Q7","Q5"])["Q7"].count().unstack().plot(ax=ax)

ax.set_xlabel("ROLE")

ax.set_ylabel("TOTAL")
group_level = pd.concat([only_ml_activities["Q5"], only_ml_activities["Q4"]], axis=1)

group = group_level.groupby(["Q5", "Q4"]).size()



group["Data Scientist"].plot.barh()
def compensation_remove_carac(df):

    for x in range(len(df)):

        compensation = df.get_value(x,'Q10')

        if(type(compensation)!=float):

            value = compensation.replace("$", "")

            df.at[x, "Q10"] = value.replace("> ", "")

        else:

            continue

    return df
def get_min_max_compensation(df):

    df2 = pd.DataFrame(columns=["Min_compensation","Max_compensation"])

    group_compensation = pd.concat([df, df2], axis=1)

    for x in range(len(df)):

        compensation2 = df.get_value(x,'Q10')

        if(pd.isna(compensation2)):

            group_compensation.at[x, "Min_compensation"] = float(0)

            group_compensation.at[x, "Max_compensation"] = float(0)

        else:

            if(compensation2=="500,000"):

                group_compensation.at[x, "Min_compensation"] = float(500000)

                group_compensation.at[x, "Max_compensation"] = float(500000)

            else:

                group_compensation.at[x, "Min_compensation"] = float(compensation2.split('-', 1)[0].replace(',', ''))

                group_compensation.at[x, "Max_compensation"] = float(compensation2.split('-', 1)[1].replace(',', ''))

    group_compensation=group_compensation.drop(group_compensation.columns[0], axis=1)

    return group_compensation.astype(float)
# Get compensation column

compensation = pd.DataFrame(only_ml_activities["Q10"])

df = compensation_remove_carac(compensation)

compensation_min_max = get_min_max_compensation(df)

# Concatenate role, yearly compensation, professional experiences columns 

role_compensation_experience = pd.concat([only_ml_activities["Q5"], only_ml_activities["Q23"], compensation_min_max], axis = 1)

# Plotting data scientist yearly compensation with experience < 1 year

group_role_compensation_experience = role_compensation_experience[role_compensation_experience["Q23"]=="< 1 years"]

group_role_compensation_experience = group_role_compensation_experience[group_role_compensation_experience["Q5"]=="Data Scientist"]



mean_min = group_role_compensation_experience["Min_compensation"].mean()  

mean_max = group_role_compensation_experience["Max_compensation"].mean()



print("$" + str("{:.2f}".format(mean_min)) + "-" + str("{:.2f}".format(mean_max)))
group_role_compensation_experience = role_compensation_experience[role_compensation_experience["Q23"]=="5-10 years"]

group_role_compensation_experience = group_role_compensation_experience[group_role_compensation_experience["Q5"]=="Data Scientist"]



mean_min = group_role_compensation_experience["Min_compensation"].mean()  

mean_max = group_role_compensation_experience["Max_compensation"].mean()  



print("$" + str("{:.2f}".format(mean_min)) + "-" + str("{:.2f}".format(mean_max)))
group_role_compensation_experience = role_compensation_experience[role_compensation_experience["Q23"]=="10-15 years"]#.groupby(["Q5", "Min_compensation"]).size()

group_role_compensation_experience = group_role_compensation_experience[group_role_compensation_experience["Q5"]=="Data Scientist"]



mean_min = group_role_compensation_experience["Min_compensation"].mean()  

mean_max = group_role_compensation_experience["Max_compensation"].mean()  



print("$" + str("{:.2f}".format(mean_min)) + "-" + str("{:.2f}".format(mean_max)))
group_role_compensation_experience = role_compensation_experience[role_compensation_experience["Q23"]=="20+ years"]

group_role_compensation_experience = group_role_compensation_experience[group_role_compensation_experience["Q5"]=="Data Scientist"]



mean_min = group_role_compensation_experience["Min_compensation"].mean()  

mean_max = group_role_compensation_experience["Max_compensation"].mean()  



print("$" + str("{:.2f}".format(mean_min)) + "-" + str("{:.2f}".format(mean_max)))
# Get activity columns from DataFrame only_ml_activitiabses

columns = question_column("Q13", only_ml_activities)

# Change columns name

change_name = change_column_name(columns)

# Change cell value 0 or 1

Q13 = change_value(change_name)

# Drop the last column

Q13_drop = Q13.drop(Q13.columns[-1], axis=1)

# Concatenate activity columns with respondent role columns

Q5_Q13 = pd.concat([only_ml_activities["Q5"], Q13_drop], axis=1)
fig, ax = plt.subplots(figsize=(15,7))

group_platform = Q5_Q13.groupby("Q5").sum().transpose()

group_platform.plot.barh(ax= ax)
# Get activity columns from DataFrame only_ml_activitiabses

columns = question_column("Q12", only_ml_activities)

# Change columns name

change_name = change_column_name(columns)

# Change cell value 0 or 1

Q12 = change_value(change_name)

# Drop the last column

Q12_drop = Q12.drop(Q12.columns[-1], axis=1)

# Concatenate activity columns with respondent role columns

Q5_Q12 = pd.concat([only_ml_activities["Q5"], Q12_drop], axis=1)
fig, ax = plt.subplots(figsize=(15,5))

Q5_Q12 = Q5_Q12.groupby("Q5").sum()

Q5_Q12.plot(ax=ax)

ax.set_xlabel("ROLE")

ax.set_ylabel("TOTAL")
Q14 = pd.get_dummies(only_ml_activities["Q14"])

Q14 = Q14.reset_index(drop=True)

Q14_concat = pd.concat([only_ml_activities["Q5"], Q14], axis=1)

Q14_concat_group = Q14_concat.groupby("Q5").sum()

fig, ax = plt.subplots(figsize=(15,5))

Q14_concat_group.plot(ax=ax)

ax.set_xlabel("ROLE")

ax.set_ylabel("TOTAL")
only_ml_activities["Q19"].value_counts().plot.barh()
expense = pd.concat([only_ml_activities["Q5"], only_ml_activities["Q11"]], axis=1)

group_expense = expense.groupby(["Q5","Q11"])["Q11"].count()

group_expense["Data Scientist"].plot.barh()
age = pd.concat([only_ml_activities["Q5"], only_ml_activities["Q1"]], axis=1)

group_age_interval = age.groupby(["Q5","Q1"])["Q1"].count()

group_age_interval["Data Scientist"].plot.barh()
# Get activity columns from DataFrame only_ml_activitiabses

columns = question_column("Q24", only_ml_activities)

# Change columns name

change_name = change_column_name(columns)

# Change cell value 0 or 1

Q24 = change_value(change_name)

# Drop the last column

Q24_drop = Q24.drop(Q24.columns[-1], axis=1)

# Concatenate activity columns with respondent role columns

Q5_Q24 = pd.concat([only_ml_activities["Q5"], Q24_drop], axis=1)
fig, ax = plt.subplots(figsize=(15,5))

group_Q5_Q24 = Q5_Q24.groupby("Q5").sum()

group_Q5_Q24.plot(ax=ax)

ax.set_xlabel("ROLE")

ax.set_ylabel("TOTAL")
# Get activity columns from DataFrame only_ml_activitiabses

columns = question_column("Q16", only_ml_activities)

# Change columns name

change_name = change_column_name(columns)

# Change cell value 0 or 1

Q16 = change_value(change_name)

# Drop the last column

Q16_drop = Q16.drop(Q16.columns[-1], axis=1)

# Concatenate activity columns with respondent role columns

Q5_Q16 = pd.concat([only_ml_activities["Q5"], Q16_drop], axis=1)
fig, ax = plt.subplots(figsize=(15,5))

Q5_Q16.groupby("Q5").sum().plot(ax=ax)

ax.set_xlabel("ROLE")

ax.set_ylabel("TOTAL")
def small_multiples_for_line_chart(df, title):

    # create a color palette

    palette = plt.get_cmap('Set1')

    

    plt.figure(figsize=(10,15))

    # multiple line plot

    num=0

    for column in df[df.columns[1:]]:

        num+=1

        # Find the right spot on the plot

        

        plt.subplot(6,2, num)

         # plot every groups, but discreet

        for v in df[df.columns[1:]]:

            plt.plot(df.index, df[v], marker='', color='grey', linewidth=1.6, alpha=0.2)

 

        # Plot the lineplot

        plt.plot(df.index, df[column], marker='', color=palette(num), linewidth=2.4, alpha=1, label=column)



        # Same limits for everybody!

        plt.xlim(0,10)

        plt.ylim(-2,2500)



        # Not ticks everywhere

        if num in range(11) :

            plt.tick_params(labelbottom='off')

        if num not in [1,3,5,7,9,11] :

            plt.tick_params(labelleft='off')



        # Add title

        plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num) )



    # general title

    plt.suptitle(title, fontsize=13, fontweight=0, color='black', style='italic', y=1.02)



    # Axis title

    plt.text(0.5, 0.02, 'Time', ha='center', va='center')

    plt.text(0.06, 0.5, 'Note', ha='center', va='center', rotation='vertical')
# Get activity columns from DataFrame only_ml_activitiabses

columns = question_column("Q17", only_ml_activities)

# Change columns name

change_name = change_column_name(columns)

# Change cell value 0 or 1

Q17 = change_value(change_name)

# Drop the last column

Q17_drop = Q17.drop(Q17.columns[-1], axis=1)

# Concatenate activity columns with respondent role columns

Q5_Q17 = pd.concat([only_ml_activities["Q5"], Q17_drop], axis=1)
group_concat_Q5_Q17 = Q5_Q17.groupby("Q5").sum()

df = pd.DataFrame(group_concat_Q5_Q17.index)

#df2= group_concat_Q5_Q17[group_concat_Q5_Q17.columns]

#small_multiples_for_line_chart(df)

df2 = group_concat_Q5_Q17.reset_index(drop=True)

concat = pd.concat([df, df2], axis=1)

#concat[concat.columns[1:]]

title = "Hosted notebook products from survey"

small_multiples_for_line_chart(concat, title)
# Get activity columns from DataFrame only_ml_activitiabses

columns = question_column("Q20", only_ml_activities)

# Change columns name

change_name = change_column_name(columns)

# Change cell value 0 or 1

Q20 = change_value(change_name)

# Drop the last column

Q20_drop = Q20.drop(Q20.columns[-1], axis=1)

# Concatenate activity columns with respondent role columns

Q5_Q20 = pd.concat([only_ml_activities["Q5"], Q20_drop], axis=1)
fig, ax = plt.subplots(figsize=(15,5))

Q5_Q20.groupby("Q5").sum().plot(ax=ax)

ax.set_xlabel("ROLE")

ax.set_ylabel("TOTAL")
# Get activity columns from DataFrame only_ml_activitiabses

columns = question_column("Q21", only_ml_activities)

# Change columns name

change_name = change_column_name(columns)

# Change cell value 0 or 1

Q21 = change_value(change_name)

# Drop the last column

Q21_drop = Q21.drop(Q21.columns[-1], axis=1)

# Concatenate activity columns with respondent role columns

Q5_Q21 = pd.concat([only_ml_activities["Q5"], Q21_drop], axis=1)
fig, ax = plt.subplots(figsize=(15,5))

Q5_Q21.groupby("Q5").sum().plot(ax=ax)

ax.set_xlabel("ROLE")

ax.set_ylabel("TOTAL")
tpu = pd.get_dummies(only_ml_activities["Q22"])

tpu = tpu.reset_index(drop=True)

tpu_concat = pd.concat([only_ml_activities["Q5"], tpu], axis=1)

tpu_concat_group = tpu_concat.groupby("Q5").sum()

fig, ax = plt.subplots(figsize=(15,5))

tpu_concat_group.plot(ax=ax)

ax.set_xlabel("ROLE")

ax.set_ylabel("TOTAL")
# Get activity columns from DataFrame only_ml_activitiabses

columns = question_column("Q28", only_ml_activities)

# Change columns name

change_name = change_column_name(columns)

# Change cell value 0 or 1

Q28 = change_value(change_name)

# Drop the last column

Q28_drop = Q28.drop(Q28.columns[-1], axis=1)

# Concatenate activity columns with respondent role columns

Q5_Q28 = pd.concat([only_ml_activities["Q5"], Q28_drop], axis=1)
fig, ax = plt.subplots(figsize=(15,5))

Q5_Q28.groupby("Q5").sum().plot(ax=ax)

ax.set_xlabel("ROLE")

ax.set_ylabel("TOTAL")
group_Q5 = Q5_Q28.groupby("Q5").sum().transpose()

group_Q5["Data Scientist"].plot.barh()
# Get activity columns from DataFrame only_ml_activitiabses

columns = question_column("Q25", only_ml_activities)

# Change columns name

change_name = change_column_name(columns)

# Change cell value 0 or 1

Q25 = change_value(change_name)

# Drop the last column

Q25_drop = Q25.drop(Q25.columns[-1], axis=1)

# Concatenate activity columns with respondent role columns

Q5_Q25 = pd.concat([only_ml_activities["Q5"], Q25_drop], axis=1)
fig, ax = plt.subplots(figsize=(15,5))

Q5_Q25.groupby("Q5").sum().plot(ax=ax)

ax.set_xlabel("ROLE")

ax.set_ylabel("TOTAL")
# Get activity columns from DataFrame only_ml_activitiabses

columns = question_column("Q33", only_ml_activities)

# Change columns name

change_name = change_column_name(columns)

# Change cell value 0 or 1

Q33 = change_value(change_name)

# Drop the last column

Q33_drop = Q33.drop(Q33.columns[-1], axis=1)

# Concatenate activity columns with respondent role columns

Q5_Q33 = pd.concat([only_ml_activities["Q5"], Q33_drop], axis=1)
fig, ax = plt.subplots(figsize=(15,5))

Q5_Q33.groupby("Q5").sum().plot(ax=ax)

ax.set_xlabel("ROLE")

ax.set_ylabel("TOTAL")
# Get activity columns from DataFrame only_ml_activitiabses

columns = question_column("Q32", only_ml_activities)

# Change columns name

change_name = change_column_name(columns)

# Change cell value 0 or 1

Q32 = change_value(change_name)

# Drop the last column

Q32_drop = Q32.drop(Q32.columns[-1], axis=1)

# Concatenate activity columns with respondent role columns

Q5_Q32 = pd.concat([only_ml_activities["Q5"], Q32_drop], axis=1)
fig, ax = plt.subplots(figsize=(15,5))

Q5_Q32.groupby("Q5").sum().plot(ax=ax)

ax.set_xlabel("ROLE")

ax.set_ylabel("TOTAL")
user = 0

for col in Q5_Q32.columns[1:]:

    s = Q5_Q32[col].sum()

    if((col !="None") & (col != "Other")):

        user = user + s

    print(col + " : " + str(s) + "\n")

print("Total users" + " : " + str(user))
# Get activity columns from DataFrame only_ml_activitiabses

columns = question_column("Q29", only_ml_activities)

# Change columns name

change_name = change_column_name(columns)

# Change cell value 0 or 1

Q29 = change_value(change_name)

# Drop the last column

Q29_drop = Q29.drop(Q29.columns[-1], axis=1)

# Concatenate activity columns with respondent role columns

Q5_Q29 = pd.concat([only_ml_activities["Q5"], Q29_drop], axis=1)
fig, ax = plt.subplots(figsize=(15,5))

Q5_Q29.groupby("Q5").sum().plot(ax=ax)

ax.set_xlabel("ROLE")

ax.set_ylabel("TOTAL")
# Get activity columns from DataFrame only_ml_activitiabses

columns = question_column("Q30", only_ml_activities)

# Change columns name

change_name = change_column_name(columns)

# Change cell value 0 or 1

Q30 = change_value(change_name)

# Drop the last column

Q30_drop = Q30.drop(Q30.columns[-1], axis=1)

# Concatenate activity columns with respondent role columns

Q5_Q30 = pd.concat([only_ml_activities["Q5"], Q30_drop], axis=1)
fig, ax = plt.subplots(figsize=(15,5))

Q5_Q30.groupby("Q5").sum().plot(ax=ax)

ax.set_xlabel("ROLE")

ax.set_ylabel("TOTAL")
# Get activity columns from DataFrame only_ml_activitiabses

columns = question_column("Q31", only_ml_activities)

# Change columns name

change_name = change_column_name(columns)

# Change cell value 0 or 1

Q31 = change_value(change_name)

# Drop the last column

Q31_drop = Q31.drop(Q31.columns[-1], axis=1)

# Concatenate activity columns with respondent role columns

Q5_Q31 = pd.concat([only_ml_activities["Q5"], Q31_drop], axis=1)
fig, ax = plt.subplots(figsize=(15,5))

Q5_Q31.groupby("Q5").sum().plot(ax=ax)

ax.set_xlabel("ROLE")

ax.set_ylabel("TOTAL")
for x in range(len(Q9_drop.columns)-1):

    column = pd.DataFrame(Q9_drop[Q9_drop.columns[x]])

    df=pd.concat([column, Q17_drop], axis=1)

    df.groupby(Q9_drop.columns[x]).sum().transpose().plot.barh()