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
# inviting pandas and friends to the party

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



sns.set_style("whitegrid") # set seaborn styles
df = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv")
# drop questions

df = df.drop(df.index[0])



# check for and drop duplicate rows

df = df.drop_duplicates()



# dropping columns based on string pattern

drop_all = ["Q26", "Time from Start to Finish", "Q11", "Q19"]



for col in drop_all:

    df = df.loc[:,~df.columns.str.contains(col)]



# rename columns for better readability

rename = {"Q1" : "age",

          "Q2" : "gender",

          "Q3" : "country",

          "Q4" : "degree",

          "Q5" : "position",

          "Q6" : "company_size",

          "Q10": "salary"

          }

df = df.rename(columns=rename)
# preparing 2017 and 2018 datasets for comparison



# import 2017

df_2017 = pd.read_csv("../input/kaggle-survey-2017/multipleChoiceResponses.csv", encoding = "ISO-8859-1")



# drop duplicates

df_2017 = df_2017.drop_duplicates()



# rename columns

rename_2017 = {"Age" : "age",

                "GenderSelect" : "gender",

                "Country" : "country",

                "FormalEducation" : "degree",

                "EmployerSize" : "company size",

                "LearningDataScienceTime" : "experience",

                "CompensationAmount" : "salary",

                "CompensationCurrency" : "currency",

                }



# select columns

df_2017 = df_2017[list(rename_2017.keys())]



# rename columns

df_2017 = df_2017.rename(columns=rename_2017)



# select countries

df_2017 = df_2017[(df_2017["country"] == "India") | (df_2017["country"] == "United States")]
# import 2018

df_2018 = pd.read_csv("../input/kaggle-survey-2018/multipleChoiceResponses.csv", encoding = "UTF-8")



# drop duplicates

df_2018 = df_2018.drop_duplicates()



# rename columns

rename_2018 = {"Q2" : "age",

                "Q1" : "gender",

                "Q3" : "country",

                "Q4" : "degree",

                "Q8" : "experience",

                "Q9" : "salary",

                }

# select columns

df_2018 = df_2018[list(rename_2018.keys())]



# rename columns

df_2018 = df_2018.rename(columns=rename_2018)



# select countries

df_2018 = df_2018[(df_2018["country"] == "India") | (df_2018["country"] == "United States of America")]
# creating seperate dataframes for India and the US to make normalization more easy

india = df[df["country"] == "India"]

usa = df[df["country"] == "United States of America"]
# creating comparison dataframe containing both India and the US

df_comp = df[(df["country"] == "India") | (df["country"] == "United States of America")].copy()

df_country_comp = df_comp[["country"]]
# data manipulation functions



# function to reverse dummy valiables based on column string pattern for whole dataframe

# may return list of unique values when labels_only is set to True



def undummy(df, col_pattern, labels_only=False):

    df_tmp = df.loc[:,df.columns.str.contains(col_pattern)]

    cols_to_bool = list(df_tmp.columns)



    df_tmp = df_tmp.fillna(0)



    labels = []

    for col in cols_to_bool:

        labels.append(df_tmp[col].value_counts().keys()[1])





    for col in cols_to_bool:

        df_tmp[col] = np.where(df_tmp[col] != 0, 1, 0)

        df_tmp[col] = df_tmp[col].astype(int)



    resources = []

    for col in cols_to_bool:

        resources.append(df_tmp[col].sum())



    labels



    resources = pd.DataFrame(data=resources, index=labels)

    resources = resources.rename(columns={0: "value"})

    resources = resources.sort_values(by="value", ascending=False)

    

    if labels_only == True:

        return labels

    return resources
# function to reverse dummy valiables based on column string pattern for single column



def undummy_single_col(df, col_name):

    sr_tmp = df[col_name]

    sr_tmp = sr_tmp.fillna(0)



    labels = sr_tmp.value_counts().keys().tolist()

    labels = pd.Series(labels).astype(str)

    labels = labels.str.replace("0", "None")



    resources = sr_tmp.value_counts().tolist()

    

    resources = pd.DataFrame(data=resources, index=labels)

    resources = resources.rename(columns={0: "value"})

    resources = resources.sort_values(by="value", ascending=False)



    return resources
# functions for calculating value percentages within given column

def perc(df, col):

    return round(100 * df[col].value_counts(normalize=True),2).to_frame()
# function for normalizing value distribution within given column for both India and the US

# return dataframe containing normalized values for given columns



def total_to_perc(df, df1, df2, col1, col2):

    df_perc = df.copy()

    df_perc = df_perc.drop(columns=[col1, col2])

    df_perc = df_perc.assign(india = df[col1].apply(lambda x: (x/len(df1))*100))

    df_perc = df_perc.assign(usa = df[col2].apply(lambda x: (x/len(df2))*100))

    return df_perc
# global plotting variables



# global figure size to use in the plots

figure_size = (12,6)



colors_pie_3 = ["yellowgreen", "coral", "cornflowerblue"]

colors_pie_2 = ["coral", "cornflowerblue"]

colors_bar_3 = ["cornflowerblue", "coral", "yellowgreen"]

colors_bar_2 = ["cornflowerblue", "coral"]



colors_years_2_in = ["peachpuff", "coral"]

colors_years_2_us = ["lightsteelblue", "cornflowerblue"]

colors_years_3_in = ["peachpuff", "coral", "sienna"]

colors_years_3_us = ["lightsteelblue", "cornflowerblue", "royalblue"]
# plotting global survey participant ratio by country (countries other than India and US are aggregated in

# "Other countries")



labels = ["Other countries", "India", "US"]

sizes = [len(df), len(india), len(usa)]

colors = colors_pie_3



fig1, ax1 = plt.subplots(figsize=figure_size)

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=90, colors=colors)

ax1.axis('equal')

ax1.set_title("India / USA as a subset")

plt.show()
# plotting ratio of survey participants for India and the US



labels = df_country_comp["country"].value_counts().keys().tolist()

sizes = [len(india), len(usa)]

colors = colors_pie_2



fig1, ax1 = plt.subplots(figsize=figure_size)

ax1.pie(sizes, labels=labels, autopct='%1.1f%%',

        shadow=False, startangle=90, colors=colors_pie_2)

ax1.axis('equal')

ax1.set_title("India / US Proportion 2019")

plt.show()
# creating dataframes for India and the US for the years 2017 and 2018

india_2017 = df_2017[df_2017["country"] == "India"]

usa_2017 = df_2017[df_2017["country"] == "United States"]

india_2018 = df_2018[df_2018["country"] == "India"]

usa_2018 = df_2018[df_2018["country"] == "United States of America"]
# plotting ratio of participants from India and the US over the last three years



labels = ["India", "US"]



sizes17 = [len(india_2017), len(usa_2017)]

sizes18 = [len(india_2018), len(usa_2018)]

sizes19 = [len(india), len(usa)]

colors = colors_pie_2



fig = plt.figure()



ax1 = fig.add_axes([.1, .3, 1, 1], aspect=1)

ax1.pie(sizes17, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)



ax2 = fig.add_axes([.8, .3, 1, 1], aspect=1)

ax2.pie(sizes18, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)



ax3 = fig.add_axes([1.5, .3, 1, 1], aspect=1)

ax3.pie(sizes19, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)



ax1.set_title("2017")

ax2.set_title("2018")

ax3.set_title("2019")



plt.show()
# preparing data for plotting: combining gender distribution of India and the US in dataframe

global_gender_dist = perc(df, "gender").rename(columns={"gender" : "global"})

india_gender_dist = perc(india, "gender").rename(columns={"gender" : "india"})

usa_gender_dist = perc(usa, "gender").rename(columns={"gender" : "usa"})



df_gender = usa_gender_dist.merge(right=india_gender_dist, how="inner", left_index=True, right_index=True)

df_gender= df_gender.merge(right=global_gender_dist, how="inner", left_index=True, right_index=True)
# plotting gender distribution

fig, ax = plt.subplots(figsize=figure_size)

df_gender.plot(kind="bar", ax=ax, rot=45, color=colors_bar_3)

ax.set_ylabel('Distribution [%]')

ax.set_title("Gender Distribution");
# preparing age dataframes

global_age_dist = perc(df, "age").rename(columns={"age" : "global"})

india_age_dist = perc(india, "age").rename(columns={"age" : "india"})

usa_age_dist = perc(usa, "age").rename(columns={"age" : "usa"})



# merge age dataframes

df_age = usa_age_dist.merge(right=india_age_dist, how="inner", left_index=True, right_index=True)

df_age = df_age.merge(right=global_age_dist, how="inner", left_index=True, right_index=True)

df_age = df_age.reset_index()

df_age = df_age.rename(columns={"index": "age"}).sort_values(by="age")

df_age = df_age.set_index("age")
# plotting age distribution

fig, ax = plt.subplots(figsize=figure_size)

df_age.plot(kind="bar", ax=ax, stacked=True, rot=45, color=colors_bar_3);

ax.set_ylabel('Distribution [%]')

ax.set_title("Age Distribution");
#get data 2018

age_in = (df_2018[df_2018["country"]=="India"]["age"]

           .value_counts(normalize=True)

           .to_frame()

           .rename(columns={"age" : "2018"}))

age_us = (df_2018[df_2018["country"]=="United States of America"]["age"]

           .value_counts(normalize=True)

           .to_frame()

           .rename(columns={"age" : "2018"}))



age_in = age_in.reindex(index = ["18-21", "22-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-69", "70-79", "80+"])

age_us = age_us.reindex(index = ["18-21", "22-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-69", "70-79", "80+"])



#get data 2019

age_in = age_in.assign(age2019=df[df["country"]=="India"]["age"].value_counts(normalize=True))

age_in.rename(columns={"age2019" : "2019"}, inplace=True)



age_us = age_us.assign(age2019=df[df["country"]=="United States of America"]["age"].value_counts(normalize=True))

age_us.rename(columns={"age2019" : "2019"}, inplace=True)
# plotting age distribution over time for US and India

fig, ax = plt.subplots(ncols=2, figsize=(18,8))

plt.rcParams.update({'font.size': 10})

data1 = age_in*100

data2 = age_us*100

ax1 = data1.plot(kind="bar", ax=ax[0], grid=True, rot=45, color=colors_years_3_in)

ax2 = data2.plot(kind="bar", ax=ax[1], grid=True, rot=45, color=colors_years_3_us)

ax1.set_title("Distribution of participants age (India)")

ax2.set_title("Distribution of participants age (USA)")

ax1.set_ylabel('Distribution [%]')

ax2.set_ylabel('Distribution [%]')

ax1.get_yticks()

ax2.get_yticks();
# preparing dataframes for educational degrees for US and India

global_degree_dist = perc(df, "degree").rename(columns={"degree" : "global"})

india_degree_dist = perc(india, "degree").rename(columns={"degree" : "india"})

usa_degree_dist = perc(usa, "degree").rename(columns={"degree" : "usa"})



df_degree = usa_degree_dist.merge(right=india_degree_dist, how="inner", left_index=True, right_index=True)

df_degree = df_degree.merge(right=global_degree_dist, how="inner", left_index=True, right_index=True)
# plotting educational degrees for India and the US

ax = df_degree.plot(kind="bar", figsize=figure_size, rot=45, color=colors_bar_3);

labels = [item.get_text() for item in ax.get_xticklabels()]

labels[3] = 'College/University study without degree'

ax.set_xticklabels(labels)

ax.set_ylabel('Distribution [%]')

ax.set_title("Distribution of Educational Degree");
# get data 2017

degree_in = (df_2017[df_2017["country"]=="India"]["degree"]

           .value_counts(normalize=True)

           .to_frame()

           .rename(columns={"degree" : "2017"}))

degree_us = (df_2017[df_2017["country"]=="United States"]["degree"]

           .value_counts(normalize=True)

           .to_frame()

           .rename(columns={"degree" : "2017"}))



# uniform degrees 

replace_degree = {"Bachelor's degree" : "Bachelor’s degree",

                  "Master's degree" : "Master’s degree",

                  "Some college/university study without earning a bachelor's degree" : "Some college/university study without earning a bachelor’s degree",

                  "I did not complete any formal education past high school" : "No formal education past high school",

                 }

degree_in.rename(index=replace_degree, inplace=True)

degree_us.rename(index=replace_degree, inplace=True)
# get data 2018

degree_in = degree_in.assign(deg2018=df_2018[df_2018["country"]=="India"]["degree"]

                               .value_counts(normalize=True))

degree_in.rename(columns={"deg2018" : "2018"}, inplace=True)

degree_us = degree_us.assign(deg2018=df_2018[df_2018["country"]=="United States of America"]["degree"]

                               .value_counts(normalize=True))

degree_us.rename(columns={"deg2018" : "2018"}, inplace=True)
# get data 2019

degree_in = degree_in.assign(deg2019=df[df["country"]=="India"]["degree"]

                               .value_counts(normalize=True))

degree_in.rename(columns={"deg2019" : "2019"}, inplace=True)

degree_us = degree_us.assign(deg2019=df[df["country"]=="United States of America"]["degree"]

                               .value_counts(normalize=True))

degree_us.rename(columns={"deg2019" : "2019"}, inplace=True)



# uniform degrees 

rename_values = {"Some college/university study without earning a bachelor’s degree" : "Studied without earning degree",

                "Bachelor’s degree" : "Bachelor",

                "Master’s degree" : "Master",

                "Professional degree" : "Professional",

                "Doctoral degree" : "Doctor",

                "No formal education past high school" : "No formal education",

                "I prefer not to answer" : "No answer"

                }

degree_us.rename(index=rename_values, inplace=True)

degree_in.rename(index=rename_values, inplace=True)
#reorder the index

index_degree = (['Bachelor', 'Master', 'Doctor', 'Professional', 'Studied without earning degree', 'No formal education', 'No answer'])

degree_us = degree_us.reindex(index_degree)

degree_in = degree_in.reindex(index_degree)
# plotting educational degrees over time for US and India

fig, ax = plt.subplots(ncols=2, figsize=(18,8))

plt.rcParams.update({'font.size': 10})

data1 = degree_in*100

data2 =data2 = degree_us*100

ax1 = data1.plot(kind="bar", ax=ax[0], grid=True, rot=45, color=colors_years_3_in)

ax2 = data2.plot(kind="bar", ax=ax[1], grid=True, rot=45, color=colors_years_3_us)

ax1.set_title("Distribution of educational degree of participants (India)")

ax2.set_title("Distribution of educational degree of participants (USA)")

ax1.set_ylabel('Distribution [%]')

ax2.set_ylabel('Distribution [%]')

ax1.get_yticks()

ax2.get_yticks();
# preparing dataframes for job position names for India and the US

global_position_dist = perc(df, "position").rename(columns={"position" : "global"})

india_position_dist = perc(india, "position").rename(columns={"position" : "india"})

usa_position_dist = perc(usa, "position").rename(columns={"position" : "usa"})



df_position = usa_position_dist.merge(right=india_position_dist, how="inner", left_index=True, right_index=True)

df_position = df_position.merge(right=global_position_dist, how="inner", left_index=True, right_index=True)
ax = df_position.plot(kind="bar", figsize=figure_size, rot=45, color=colors_bar_3)

ax.set_ylabel('Distribution [%]')

ax.set_title("Distribution of Current Job Position");

# preparing dataframes for salary distribution within Data Science for India and the US

global_salary_dist = perc(df, "salary").rename(columns={"salary" : "global"})

india_salary_dist = perc(india, "salary").rename(columns={"salary" : "india"})

usa_salary_dist = perc(usa, "salary").rename(columns={"salary" : "usa"})



df_salary = usa_salary_dist.merge(right=india_salary_dist, how="inner", left_index=True, right_index=True)

df_salary = df_salary.merge(right=global_salary_dist, how="inner", left_index=True, right_index=True)

df_salary = df_salary.reset_index().rename(columns={"index" : "salary"}).sort_values("salary").set_index("salary")



# reordering salaries

df_salary = df_salary.reindex(index = ["$0-999", "1,000-1,999", "2,000-2,999",  "3,000-3,999", "4,000-4,999", 

                                       "5,000-7,499", "7,500-9,999", "10,000-14,999", "15,000-19,999", "20,000-24,999", 

                                       "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", 

                                       "60,000-69,999", "70,000-79,999", "80,000-89,999", "90,000-99,999", 

                                       "100,000-124,999", "125,000-149,999", "150,000-199,999", "200,000-249,999",

                                       "250,000-299,999", "300,000-500,000", "> $500,000"])
ax = df_salary.plot(kind="line", figsize=figure_size, rot=45, color=colors_bar_3)

ax.set_ylabel('Distribution [%]')

ax.set_title("Distribution of Yearly Salary");
# preparing dataframes for company size distribution

global_company_size_dist = perc(df, "company_size").rename(columns={"company_size" : "global"})

india_company_size_dist = perc(india, "company_size").rename(columns={"company_size" : "india"})

usa_company_size_dist = perc(usa, "company_size").rename(columns={"company_size" : "usa"})



df_company_size = usa_company_size_dist.merge(right=india_company_size_dist, how="inner", left_index=True, right_index=True)

df_company_size = df_company_size.merge(right=global_company_size_dist, how="inner", left_index=True, right_index=True)



# reordering index

df_company_size = df_company_size.reset_index().rename(columns={"index" : "company_size"}).sort_values("company_size").set_index("company_size")

df_company_size = df_company_size.reindex(index = ["0-49 employees", "50-249 employees", "250-999 employees", "1000-9,999 employees", "> 10,000 employees"])
ax = df_company_size.plot(kind="line", figsize=figure_size, rot=45, color=colors_bar_3)

ax.set_ylabel('Distribution [%]')

ax.set_title("Distribution of Company Size");
# preparing dataframes for the use of programming languages in India and the US

languages_india = undummy(india, "Q18")

languages_india = languages_india.drop("-1", axis=0)

languages_india = languages_india.rename(index={0: "Python"})



languages_usa = undummy(usa, "Q18")

languages_usa = languages_usa.drop("-1", axis=0)

languages_usa = languages_usa.rename(index={0: "Python"})



# merging dataframes

languages = languages_india.merge(right=languages_usa, how="inner", left_index=True, right_index=True)

languages = languages.rename(columns={"value_x": "india", "value_y": "usa"})



# normalizing

languages_perc = total_to_perc(languages, india, usa, "india", "usa")

languages_perc = languages_perc[['usa', 'india']]
# plotting programming languages percentages

fig, ax = plt.subplots(figsize=figure_size)

languages_perc.plot(kind="bar", ax=ax, rot=45, color=colors_bar_2)

ax.set_ylabel('Distribution [%]')

ax.set_title("Distribution of the Use of Programming Languages");
# preparing dataframes for programming experience in India and the US

india_exp_dist = perc(india, "Q15").rename(columns={"Q15": "india"})

usa_exp_dist = perc(usa, "Q15").rename(columns={"Q15": "usa"})

df_exp = usa_exp_dist.merge(right=india_exp_dist, how="inner", left_index=True, right_index=True)



# reordering index

df_exp = df_exp.reindex(["I have never written code", "< 1 years", "1-2 years", "3-5 years", "5-10 years", "10-20 years", "20+ years"])
# plotting programming experience percentages

fig, ax = plt.subplots(figsize=figure_size)

df_exp.plot(kind="line", ax=ax, color=colors_bar_2)

ax.set_ylabel('Distribution [%]')

ax.set_title("Distribution of Programming Experience in Years");
# preparing dataframes for machine learning experience in India and the US

india_exp_ml_dist = perc(india, "Q23").rename(columns={"Q23": "india"})

usa_exp_ml_dist = perc(usa, "Q23").rename(columns={"Q23": "usa"})

df_exp_ml = usa_exp_ml_dist.merge(right=india_exp_dist, how="inner", left_index=True, right_index=True)
# plotting machine learning experience percentages

fig, ax = plt.subplots(figsize=figure_size)

df_exp_ml.plot(kind="line", ax=ax, color=colors_bar_2)

ax.set_ylabel('Distribution [%]')

ax.set_title("Distribution of Machine Learning Experience in Years");
# preparing dataframes for machine learning algorithms used in India and the US

ml_tools_india = undummy(india, "Q24")

ml_tools_india = ml_tools_india.rename(index={0: "Linear or Logistic Regression"})

ml_tools_usa = undummy(usa, "Q24")

ml_tools_usa = ml_tools_usa.rename(index={0: "Linear or Logistic Regression"})



ml_tools = ml_tools_india.merge(right=ml_tools_usa, how="inner", left_index=True, right_index=True)

ml_tools = ml_tools.rename(columns={"value_x": "india", "value_y": "usa"})

ml_tools = ml_tools.drop("-1", axis=0)
ml_tools_perc = total_to_perc(ml_tools, india, usa, "india", "usa")

ml_tools_perc = ml_tools_perc[['usa', 'india']]
# plotting machine learning algorithms percentage

fig, ax = plt.subplots(figsize=figure_size)

ml_tools_perc.plot(kind="bar", ax=ax, color=colors_bar_2)

ax.set_ylabel('Distribution [%]')

ax.set_title("Distribution of the Use of Machine Learning Algorithms");
# preparing dataframes for Data Science learning resources used in India and the US

resources_india = undummy(india, "Q13")

resources_usa = undummy(usa, "Q13")



resources = resources_india.merge(right=resources_usa, how="inner", left_index=True, right_index=True)

resources = resources.rename(columns={"value_x": "india", "value_y": "usa"})

resources = resources.drop("-1", axis=0)



resources_perc = total_to_perc(resources, india, usa, "india", "usa")



resources_perc = resources_perc[['usa', 'india']]
# plotting learning resources percentages

fig, ax = plt.subplots(figsize=figure_size)

resources_perc.plot(kind="bar", ax=ax, color=colors_bar_2)   

ax.set_ylabel('Distribution [%]')

ax.set_title("Distribution of the Use of Learning Resources");