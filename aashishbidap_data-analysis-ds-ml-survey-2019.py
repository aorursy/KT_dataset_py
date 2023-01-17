#libraries

import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot
#configs

pd.set_option("display.max_columns",999)

pd.set_option("display.max_rows",999)



#files importing

data1=pd.read_csv("/kaggle/input/kaggle-survey-2019/survey_schema.csv")

data2=pd.read_csv("/kaggle/input/kaggle-survey-2019/other_text_responses.csv")

data3=pd.read_csv("/kaggle/input/kaggle-survey-2019/questions_only.csv")

data4=pd.read_csv("/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv")               

               
data_degree = data4[data4["Q4"].notna()]



#figure 1 highest level of education.



my_dataframe = pd.DataFrame(data_degree.loc[1:,"Q4"])

my_dataframe["Q4"] = my_dataframe["Q4"].astype('category')

figure1 = sns.countplot(x="Q4",data=my_dataframe)

figure1.set_xticklabels(figure1.get_xticklabels(), rotation=75)

pyplot.title("Educational Analysis")

pyplot.xlabel("Degree")
data4["Q4"].unique()

Masters_data = data4[data4["Q4"]=="Master’s degree"]

Masters_data = Masters_data[Masters_data != 'Student']

Masters_data = Masters_data[Masters_data != 'Not employed']
Masters_data["Q5"].unique()

figure2 = sns.countplot(x='Q5',data=Masters_data,order = Masters_data['Q5'].value_counts().index)

figure2.set_xticklabels(figure2.get_xticklabels(), rotation=75)

figure2.set(xlabel='Job Profile',ylabel='Count')

pyplot.title("Job Profile for the Surveyors with a Master's Degree")
bachelors_data = data4[data4["Q4"]=="Bachelor’s degree"]

bachelors_data = bachelors_data[bachelors_data != 'Student']

bachelors_data = bachelors_data[bachelors_data != 'Not employed']



bachelors_data["Q5"].unique()



figure3 = sns.countplot(x='Q5',data=bachelors_data,order = bachelors_data['Q5'].value_counts().index)

figure3.set_xticklabels(figure3.get_xticklabels(), rotation=75)

figure3.set(xlabel='Job Profile',ylabel='Count')

pyplot.title("Job Profile for the Surveyors with a Bachelors's Degree")
Masters_data["Q3"].unique()
a4_dims = (11.7, 8.27)

fig, ax = pyplot.subplots(figsize=a4_dims)

figure4=sns.countplot(x="Q3",data=Masters_data,palette="Greens_d",order = Masters_data['Q3'].value_counts().index)

figure4.set_xticklabels(figure4.get_xticklabels(), rotation=90)

figure4.set(xlabel="Countries",ylabel="Count")

pyplot.title("Countries of Surveyors with Master's Degree")
bachelors_data["Q3"].unique()
a4_dims = (11.7, 8.27)

fig, ax = pyplot.subplots(figsize=a4_dims)

figure5=sns.countplot(x="Q3",data=bachelors_data,palette="Greens_d",order = bachelors_data['Q3'].value_counts().index)

figure5.set_xticklabels(figure5.get_xticklabels(), rotation=90)

figure5.set(xlabel="Countries",ylabel="Count")

pyplot.title("Countries of Surveyors with Bachelor's Degree")
# Gender check for the surveyors.



column=['Q2']

data4_gender = pd.DataFrame(data_degree.loc[1:,'Q2'],columns=column)

gend = data4_gender["Q2"].value_counts()

gender_data = pd.DataFrame((gend.values/len(data4_gender["Q2"]) ) * 100)

gender_data.insert(1,"Gender",['Male', 'Female', 'Prefer not to say', 'Prefer to self-describe'],True)

columns=["Percent","Gender"]

gender_data.columns=columns

labels = gender_data["Gender"]

percent= gender_data["Percent"]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

explode = (0,0.1, 0, 0)  # explode 2nd slice

pyplot.pie(percent, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=180)

pyplot.axis('equal')

pyplot.show()
Masters_data["Q1"].unique()
a4_dims = (11.7, 8.27)

fig, ax = pyplot.subplots(figsize=a4_dims)

figure4=sns.countplot(x="Q1",data=Masters_data,order = Masters_data['Q1'].value_counts().index)

figure4.set_xticklabels(figure4.get_xticklabels())

pyplot.title("Age Group for Surveyors with Master's Degree")

figure4.set(xlabel="Age",ylabel="Count")
bachelors_data["Q1"].unique()
a4_dims = (11.7, 8.27)

fig, ax = pyplot.subplots(figsize=a4_dims)

figure4=sns.countplot(x="Q1",data=bachelors_data,order = bachelors_data['Q1'].value_counts().index)

figure4.set_xticklabels(figure4.get_xticklabels())

pyplot.title("Age Group for Surveyors with Bachelors's Degree")

figure4.set(xlabel="Age",ylabel="Count")
job_data = data_degree.loc[1:,:]
job_data['Q10'].value_counts().mean
Job_Sal_data = pd.crosstab(job_data['Q10'],job_data['Q5'])
figure5 = Job_Sal_data.plot.bar(stacked=True,figsize=(10,10))
my_data = data_degree.loc[1:,]
my_data["Q19"].unique()
my_data["Q19"].value_counts()
data_job_skill = pd.crosstab(my_data["Q5"],my_data["Q19"])
figure6 = data_job_skill.plot.bar(figsize=(10,10),stacked=True,grid=True)

pyplot.title("Recommended Skills by the Kaggle Surveyors")

figure6.set(xlabel="Job Role",ylabel="Count")
def my_function_plot(mydataframe):

    list_notebook_sources=[]

    for i in mydataframe:

        list_notebook_sources.append(mydataframe[i].value_counts())



    test =pd.DataFrame(list_notebook_sources)



    l1=[]

    for i in test:

        l1.append(i)

    l2=[]

    for i in range(0,12):

        l2.append(test.iloc[i,i])



    data_1= {}

    data_1 = {

        'Source' : l1,

        'Value' : l2

    } 



    data_source_df = pd.DataFrame(data_1,index=[0,1,2,3,4,5,6,7,8,9,10,11])

    figure9 = sns.barplot(x='Source',y='Value',data=data_source_df,palette="Greens_d")

    figure9.set_xticklabels(figure9.get_xticklabels(), rotation=90)

data_sources = my_data.loc[:,"Q12_Part_1":"Q12_Part_12"]

my_function_plot(data_sources)

pyplot.title("Favorite media sources that report on data science topics.")

pyplot.xlabel("Media Sources")

pyplot.ylabel("Count")
data_online_sources = my_data.loc[:,"Q13_Part_1":"Q13_Part_12"]



my_function_plot(data_online_sources)

pyplot.title("On which platforms have you begun or completed data science courses")

pyplot.xlabel("Data Science Course")

pyplot.ylabel("Count")
my_data["Q14"].value_counts()

figure8=sns.countplot(x="Q14",data=my_data)

figure8.set_xticklabels(figure8.get_xticklabels(), rotation=90)

pyplot.title("Primary tool that you use at work or school to analyze data")

pyplot.xlabel("Analysing Tools")
Masters_data["Q15"].value_counts()

figure9=sns.countplot(x="Q15",data=Masters_data,order = Masters_data['Q15'].value_counts().index)

figure9.set_xticklabels(figure9.get_xticklabels(), rotation=90)

pyplot.title("Masters data having experience in coding using any of the programming languages")

pyplot.xlabel("Analysing Tools")

#Masters_data["Q15"].value_counts()

figure9=sns.countplot(x="Q15",data=bachelors_data,order = bachelors_data['Q15'].value_counts().index)

figure9.set_xticklabels(figure9.get_xticklabels(), rotation=90)

pyplot.title("Bachelors data having experience in coding using any of the programming languages")

pyplot.xlabel("Analysing Tools")

my_data["Q15"].value_counts()

figure9=sns.countplot(x="Q15",data=my_data)

figure9.set_xticklabels(figure9.get_xticklabels(), rotation=90)

pyplot.title("Experience with Coding")

pyplot.xlabel("In Years")
data_IDE_sources = my_data.loc[:,"Q16_Part_1":"Q16_Part_12"]

my_function_plot(data_IDE_sources)

pyplot.title("Integrated development environments (IDE's) do you use on a regular basis")

pyplot.xlabel("DIfferent IDE's")
data_notebook_sources = my_data.loc[:,"Q17_Part_1":"Q17_Part_12"]



my_function_plot(data_notebook_sources)

pyplot.title("Notebook's Do you use on a regular basis")

pyplot.xlabel("DIfferent Notebook's")
data_language_sources = my_data.loc[:,"Q18_Part_1":"Q18_Part_12"]



my_function_plot(data_language_sources)

pyplot.title("Popular Programming Languages")

pyplot.xlabel("Programming Language")

pyplot.ylabel("Count")
data_visualization_sources = my_data.loc[:,"Q20_Part_1":"Q20_Part_12"]



my_function_plot(data_visualization_sources)

pyplot.title("Popular Data Visualization Packages")

pyplot.xlabel("Visualization Package")

pyplot.ylabel("Count")
data_modeling_sources = my_data.loc[:,"Q24_Part_1":"Q24_Part_12"]



my_function_plot(data_modeling_sources)

pyplot.title("Popular ML algoirthms")

pyplot.xlabel("Algorithm")

pyplot.ylabel("Count")
data_cat_sources = my_data.loc[:,"Q28_Part_1":"Q28_Part_12"]



my_function_plot(data_cat_sources)

pyplot.title(" Machine Learning Frameworks")

pyplot.xlabel("Frameworks")

pyplot.ylabel("Count")
data_cloud_sources = my_data.loc[:,"Q29_Part_1":"Q29_Part_12"]



my_function_plot(data_cloud_sources)

pyplot.title(" Popular cloud computing platforms")

pyplot.xlabel("Cloud Computing platforms")

pyplot.ylabel("Count")
data_computing_sources = my_data.loc[:,"Q30_Part_1":"Q30_Part_12"]



my_function_plot(data_computing_sources)

pyplot.title(" Popular cloud computing platforms")

pyplot.xlabel("cloud computing platforms")

pyplot.ylabel("Count")
data_computing_sources = my_data.loc[:,"Q31_Part_1":"Q31_Part_12"]



my_function_plot(data_computing_sources)

pyplot.title(" Popular big data / analytics products")

pyplot.xlabel("big data / analytics products")

pyplot.ylabel("Count")
data_ML_sources = my_data.loc[:,"Q32_Part_1":"Q32_Part_12"]



my_function_plot(data_ML_sources)

pyplot.title("machine learning products")

pyplot.xlabel("big data / analytics products")

pyplot.ylabel("Count")
data_ML_sources = my_data.loc[:,"Q33_Part_1":"Q33_Part_12"]



my_function_plot(data_ML_sources)

pyplot.title("Automated machine learning tools")

pyplot.xlabel("Automated machine learning tools")

pyplot.ylabel("Count")
data_DB_sources = my_data.loc[:,"Q34_Part_1":"Q34_Part_12"]



my_function_plot(data_DB_sources)

pyplot.title("Popular Databases for Data Analysis.")

pyplot.xlabel("Databases")

pyplot.ylabel("Count")