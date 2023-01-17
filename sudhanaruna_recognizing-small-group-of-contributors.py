import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec

%matplotlib inline

import seaborn as sns



# loading 2019 survey dataset

mcq_data_2019 = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')

# loading 2018 survey dataset

mcq_data_2018 = pd.read_csv("../input/kaggle-survey-2018/multipleChoiceResponses.csv")

# removing questions column

mcq_data_2018 = mcq_data_2018.drop([0])

mcq_data_2019 = mcq_data_2019.drop([0])


countries_2018 = pd.DataFrame( mcq_data_2018.Q3.value_counts() )

countries_2018 = countries_2018[:10]

countries_2019 = pd.DataFrame( mcq_data_2019.Q3.value_counts() )

countries_2019 = countries_2019[:10]



plt.figure(2, figsize=(20,6) )

grid = GridSpec(1,2)

sns.set( style="whitegrid" )



countries_2018 =countries_2018.rename(index={"United Kingdom of Great Britain and Northern Ireland" : "UK", 

                              "United States of America": "USA"})

countries_2019 =countries_2019.rename(index={"United Kingdom of Great Britain and Northern Ireland" : "UK", 

                              "United States of America": "USA"})



plt.subplot( grid[0,0], title="Top 10 countries in 2018" )

chart2018 = sns.barplot( x=countries_2018.index, y=countries_2018.Q3, palette="GnBu_d" )



plt.subplot( grid[0,1], title="Top 10 countries in 2019" )

chart2019 = sns.barplot( x=countries_2019.index, y=countries_2019.Q3, palette="GnBu_d" )



chart2018.set_ylabel("Involvement")

chart2018.set_xlabel("Countries")

chart2018.set_xticklabels( countries_2018.index, rotation="45" )

chart2019.set_ylabel("Involvement")

chart2019.set_xlabel("Countries")

chart2019.set_xticklabels( countries_2019.index, rotation="45" )

plt.show()
involvement = pd.DataFrame( { "Difference": (countries_2019.Q3 - countries_2018.Q3) }, index=countries_2019.index ).dropna()

print(involvement)
# dataset focusing on countries mentioned as "Other"

age_dataset = pd.DataFrame( { "year19" : mcq_data_2019.loc[ mcq_data_2019.Q3 == "Other" ].Q1.value_counts(),

                            "year18" : mcq_data_2018.loc[ mcq_data_2018.Q3 == "Other" ].Q2.value_counts() },

                          index=['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-60','60-69','70+'])

age_dataset = age_dataset.dropna()



plt.figure(2, figsize=(20,6) )

grid = GridSpec(1,2)

sns.set( style="whitegrid" )



plt.subplot( grid[0,0], title="Popularity in 2018" )

chart_18 = sns.barplot( x=age_dataset.index, y=age_dataset.year18, palette="GnBu_d" )



plt.subplot( grid[0,1], title="Popularity in 2019" )

chart_19 = sns.barplot( x=age_dataset.index, y=age_dataset.year19, palette="GnBu_d" )



chart_18.set_ylabel("Involvement")

chart_18.set_xlabel("Age")

chart_18.set_xticklabels( age_dataset.index, rotation="45" )

chart_19.set_ylabel("Involvement")

chart_19.set_xlabel("Age")

chart_19.set_xticklabels( age_dataset.index, rotation="45" )

plt.show()
# dataset focusing on countries mentioned as "India"

age_dataset_india = pd.DataFrame( { "year19" : mcq_data_2019.loc[ mcq_data_2019.Q3 == "India" ].Q1.value_counts(),

                            "year18" : mcq_data_2018.loc[ mcq_data_2018.Q3 == "India" ].Q2.value_counts() },

                          index=['18-21','22-24','25-29','30-34','35-39','40-44','45-49','50-54','55-60','60-69','70+'])

age_dataset_india = age_dataset_india.dropna()



plt.figure(2, figsize=(20,4) )

grid = GridSpec(1,2)

sns.set( style="whitegrid" )



plt.subplot( grid[0,0], title="Popularity in 2018 India" )

chart_18_india = sns.barplot( x=age_dataset_india.index, y=age_dataset_india.year18, palette="GnBu_d" )



plt.subplot( grid[0,1], title="Popularity in 2019 India" )

chart_19_india = sns.barplot( x=age_dataset_india.index, y=age_dataset_india.year19, palette="GnBu_d" )



chart_18_india.set_ylabel("Involvement")

chart_18_india.set_xlabel("Age")

chart_18_india.set_xticklabels( age_dataset.index, rotation="45" )

chart_19_india.set_ylabel("Involvement")

chart_19_india.set_xlabel("Age")

chart_19_india.set_xticklabels( age_dataset.index, rotation="45" )

plt.show()
#  Education qualification hold by the contributors

edu_quali = pd.DataFrame( mcq_data_2019.loc[ mcq_data_2019.Q3 == "Other" ].Q4.value_counts() )

edu_quali = edu_quali.rename(index={"Some college/university study without earning a bachelor’s degree" : "Other college study",

                                   "No formal education past high school": "High School"})

# edu_quali

# Job titles hold by the contributors

job_title = pd.DataFrame( mcq_data_2019.loc[ mcq_data_2019.Q3 == "Other" ].Q5.value_counts() )



plt.figure(2, figsize=(25,7),  )

plt.tight_layout()

grid = GridSpec(1,2)

sns.set( style="whitegrid" )



plt.subplot( grid[0,0], title="Edu qualification hold by the contributor") 

edus = sns.barplot( x=edu_quali.Q4, y=edu_quali.index, palette='GnBu_d' )



plt.subplot(grid[0,1], title="Job titles hold by the contributors" )

jobs = sns.barplot( x=job_title.Q5, y=job_title.index, palette='GnBu_d' )



plt.show()

jobs_edu = pd.DataFrame({ 

                        "Data_Scientist" :  mcq_data_2019.loc[ (mcq_data_2019.Q5 == "Data Scientist") & (mcq_data_2019.Q3 =="Other") ].Q4.value_counts(),

                        "Software Engineer" :  mcq_data_2019.loc[ (mcq_data_2019.Q5 == "Software Engineer") & (mcq_data_2019.Q3 =="Other") ].Q4.value_counts(),

                        "Data Analyst" : mcq_data_2019.loc[ (mcq_data_2019.Q5 == "Data Analyst") & (mcq_data_2019.Q3 =="Other") ].Q4.value_counts(),

                        "Research Scientist": mcq_data_2019.loc[ (mcq_data_2019.Q5 == "Research Scientist") & (mcq_data_2019.Q3 =="Other") ].Q4.value_counts(),

                        "Statistician": mcq_data_2019.loc[ (mcq_data_2019.Q5 == "Statistician") & (mcq_data_2019.Q3 =="Other") ].Q4.value_counts(),

                        "Data Engineer": mcq_data_2019.loc[ (mcq_data_2019.Q5 == "Data Engineer") & (mcq_data_2019.Q3 =="Other") ].Q4.value_counts(),

                        "Business Analyst": mcq_data_2019.loc[ (mcq_data_2019.Q5 == "Business Analyst") & (mcq_data_2019.Q3 =="Other") ].Q4.value_counts(),

                        "Student": mcq_data_2019.loc[ (mcq_data_2019.Q5 == "Student") & (mcq_data_2019.Q3 =="Other") ].Q4.value_counts(),

                        "Other": mcq_data_2019.loc[ (mcq_data_2019.Q5 == "Other") & (mcq_data_2019.Q3 =="Other") ].Q4.value_counts(),

                        "Not employed": mcq_data_2019.loc[ (mcq_data_2019.Q5 == "Not employed") & (mcq_data_2019.Q3 =="Other") ].Q4.value_counts()

                        })

jobs_edu = jobs_edu.fillna(0)

jobs_edu = jobs_edu.rename(index={"Some college/university study without earning a bachelor’s degree" : "Other college study",

                                  "No formal education past high school": "High School"})

plt.figure(figsize=(15, 7))

sns.heatmap(jobs_edu, annot=True, linewidths=.5, cmap="Blues")

plt.show()
salary_datascientist_2018 = pd.DataFrame(mcq_data_2018.loc[ (mcq_data_2018.Q3 == "Other") & (mcq_data_2018.Q6 == "Data Scientist") ].Q9.value_counts().drop(index="I do not wish to disclose my approximate yearly compensation" ))

salary_datascientist_2018 = salary_datascientist_2018.drop(index=["300-400,000","200-250,000"])



salary2019_dataset =mcq_data_2019.loc[ (mcq_data_2019.Q3 == "Other") & (mcq_data_2019.Q5 == "Data Scientist") ].Q10.value_counts()



# altering range of 2019 salary range

rangelist = [

    ["$0-999","1,000-1,999",'2,000-2,999','3,000-3,999','4,000-4,999','5,000-7,499','7,500-9,999'],

    ['10,000-14,999','15,000-19,999'],

    ['20,000-24,999','25,000-29,999'],

    ['30,000-39,999'],

    ['40,000-49,999'],

    ['50,000-59,999'],

    ['60,000-69,999'],

    ['70,000-79,999'],

    ['80,000-89,999'],

    ['90,000-99,999'] ]



def range_alt( lst ):

    tot = 0

    for limit in lst:

        tot += salary2019_dataset[limit]

    return tot



salary_datascientist_2019 = pd.DataFrame( {

    "0-10,000" : range_alt(rangelist[0]),

    "10-20,000" : range_alt(rangelist[1]),

    "20-30,000" : range_alt(rangelist[2]),

    "30-40,000" : range_alt(rangelist[3]),

    "40-50,000" : range_alt(rangelist[4]),

    "50-60,000" : range_alt(rangelist[5]),

    "60-70,000" : range_alt(rangelist[6]),

    "70-80,000" : range_alt(rangelist[7]),

    "80-90,000" : range_alt(rangelist[8]),

    "90-100,000" : range_alt(rangelist[9]),

}, index=[0] )

 

salary_datascientist_2019 = salary_datascientist_2019.transpose()



# rename column name

salary_datascientist_2019 = salary_datascientist_2019.rename( columns=({0:"Count"}) )

salary_datascientist_2018 = salary_datascientist_2018.rename( columns=({'Q9':"Count"}) )



# rearrange the index

salary_datascientist_2018 = pd.DataFrame(salary_datascientist_2018, index=[ "0-10,000",  

    "10-20,000", 

    "20-30,000", 

    "30-40,000", 

    "40-50,000", 

    "50-60,000", 

    "60-70,000", 

    "70-80,000",

    "80-90,000", 

    "90-100,000"])



# ploting the graphs



plt.figure(2, figsize=(20,6) )

grid = GridSpec(1,2)

sns.set( style="whitegrid" )



# print(salary_datascientist_2018)

# print(salary_datascientist_2019)

plt.subplot( grid[0,0], title="Salary of Data scientist 2018" )

bargraph_2018_dataScientist = sns.barplot( x=salary_datascientist_2018.index, y=salary_datascientist_2018["Count"], palette="GnBu_d" )



plt.subplot( grid[0,1], title="Salary of Data scientist 2019" )

bargraph_2019_dataScientist = sns.barplot( x=salary_datascientist_2019.index, y=salary_datascientist_2019.Count, palette="GnBu_d" )



bargraph_2018_dataScientist.set_ylabel("Counts")

bargraph_2018_dataScientist.set_xlabel("Salary range")

bargraph_2018_dataScientist.set_xticklabels( salary_datascientist_2018.index, rotation="45" )

bargraph_2019_dataScientist.set_ylabel("Counts")

bargraph_2019_dataScientist.set_xlabel("Salary range")

bargraph_2019_dataScientist.set_xticklabels( salary_datascientist_2019.index, rotation="45" )

plt.show()
salary_datascientist_2018 = pd.DataFrame(mcq_data_2018.loc[ (mcq_data_2018.Q3 == "Other") & (mcq_data_2018.Q6 == "Software Engineer") ].Q9.value_counts().drop(index="I do not wish to disclose my approximate yearly compensation" ))

salary2019_dataset =mcq_data_2019.loc[ (mcq_data_2019.Q3 == "Other") & (mcq_data_2019.Q5 == "Software Engineer") ].Q10.value_counts()



# altering range of 2019 salary range

rangelist = [

    ["$0-999","1,000-1,999",'2,000-2,999','3,000-3,999','4,000-4,999','5,000-7,499','7,500-9,999'],

    ['10,000-14,999','15,000-19,999'],

    ['20,000-24,999','25,000-29,999'],

    ['30,000-39,999'],

    ['40,000-49,999'],

    ['50,000-59,999'],

    ['60,000-69,999'],

    ['70,000-79,999'],

    ['80,000-89,999'],

    ['90,000-99,999'] ]



def range_alt( lst ):

    tot = 0

    for limit in lst:

        tot += salary2019_dataset[limit]

    return tot



salary_datascientist_2019 = pd.DataFrame( {

    "0-10,000" : range_alt(rangelist[0]),

    "10-20,000" : range_alt(rangelist[1]),

    "20-30,000" : range_alt(rangelist[2]),

    "30-40,000" : range_alt(rangelist[3]),

    "40-50,000" : range_alt(rangelist[4]),

    "50-60,000" : range_alt(rangelist[5]),

    "60-70,000" : range_alt(rangelist[6]),

    "70-80,000" : range_alt(rangelist[7]),

    "80-90,000" : range_alt(rangelist[8]),

    "90-100,000" : range_alt(rangelist[9]),

}, index=[0] )

 

salary_datascientist_2019 = salary_datascientist_2019.transpose()



# rename column name

salary_datascientist_2019 = salary_datascientist_2019.rename( columns=({0:"Count"}) )

salary_datascientist_2018 = salary_datascientist_2018.rename( columns=({'Q9':"Count"}) )



# rearrange the index

salary_datascientist_2018 = pd.DataFrame(salary_datascientist_2018, index=[ "0-10,000",  

    "10-20,000", 

    "20-30,000", 

    "30-40,000", 

    "40-50,000", 

    "50-60,000", 

    "60-70,000", 

    "70-80,000",

    "80-90,000", 

    "90-100,000"])



# ploting the graphs



plt.figure(2, figsize=(20,6) )

grid = GridSpec(1,2)

sns.set( style="whitegrid" )



# print(salary_datascientist_2018)

# print(salary_datascientist_2019)

plt.subplot( grid[0,0], title="Salary of Software Engineer 2018" )

bargraph_2018_dataScientist = sns.barplot( x=salary_datascientist_2018.index, y=salary_datascientist_2018["Count"], palette="GnBu_d" )



plt.subplot( grid[0,1], title="Salary of Software Engineer 2019" )

bargraph_2019_dataScientist = sns.barplot( x=salary_datascientist_2019.index, y=salary_datascientist_2019.Count, palette="GnBu_d" )



bargraph_2018_dataScientist.set_ylabel("Counts")

bargraph_2018_dataScientist.set_xlabel("Salary range")

bargraph_2018_dataScientist.set_xticklabels( salary_datascientist_2018.index, rotation="45" )

bargraph_2019_dataScientist.set_ylabel("Counts")

bargraph_2019_dataScientist.set_xlabel("Salary range")

bargraph_2019_dataScientist.set_xticklabels( salary_datascientist_2019.index, rotation="45" )

plt.show()


def range_alt_dataset( lst , data_set  ):

    tot = 0

    for limit in lst:

        tot += data_set[limit]

    return tot



def genDataset(name):

    data_set = mcq_data_2019.loc[ (mcq_data_2019.Q3 == str(name)) ].Q10.value_counts()

    pre_dataset =  pd.DataFrame( {

        "0-10,000" : range_alt_dataset(rangelist[0], data_set),

        "10-20,000" : range_alt_dataset(rangelist[1], data_set),

        "20-30,000" : range_alt_dataset(rangelist[2], data_set),

        "30-40,000" : range_alt_dataset(rangelist[3], data_set),

        "40-50,000" : range_alt_dataset(rangelist[4], data_set),

        "50-60,000" : range_alt_dataset(rangelist[5], data_set),

        "60-70,000" : range_alt_dataset(rangelist[6], data_set),

        "70-80,000" : range_alt_dataset(rangelist[7], data_set),

        "80-90,000" : range_alt_dataset(rangelist[8], data_set),

        "90-100,000" : range_alt_dataset(rangelist[9], data_set),

        }, index=[0] )

    pre_dataset = pre_dataset.transpose()

    return list(pre_dataset[0])





other = genDataset("Other")

india = genDataset("India")

usa = genDataset("United States of America")

japan = genDataset("Japan")

brazil = genDataset("Brazil")



salary2019 = pd.DataFrame( {

                "other": other,

                "india":india,

                "usa":usa,

                "japan":japan,

                "brazil": brazil

}, index=[ "0-10,000","10-20,000","20-30,000","30-40,000","40-50,000","50-60,000","60-70,000","70-80,000","80-90,000","90-100,000"]

)



fig = plt.figure(figsize=(15, 7))

sns.heatmap(data=salary2019,cmap="Blues", annot=True);
# Specific related question columns

specific_col =['Q13_Part_1','Q13_Part_2','Q13_Part_3','Q13_Part_4','Q13_Part_5','Q13_Part_6','Q13_Part_7','Q13_Part_8'

              ,'Q13_Part_9','Q13_Part_10','Q13_Part_11','Q13_Part_12']

considered_dataSet = mcq_data_2019.loc[ mcq_data_2019.Q3 == "Other" ]

focusing_dataset = considered_dataSet.loc[:,specific_col]

lst = [ considered_dataSet[ col ].value_counts() for col in focusing_dataset ]



courses_dataset = pd.DataFrame(lst)

course_index = list(courses_dataset.columns)

values_course =[ l[0] for l in lst ]

courseDataSet = pd.DataFrame( {"Courses": course_index, "Involvement": values_course} )



# Plotting the graph

sns.set( style='whitegrid' ) 

plt.tight_layout(5)

plt.figure(figsize=(16,6))



# replacing values for clear display of graph

courseDataSet.Courses.replace("Kaggle Courses (i.e. Kaggle Learn)","Kaggle", inplace=True)

courseDataSet.Courses.replace("University Courses (resulting in a university degree)","University", inplace=True)

courseDataSet.Courses.replace("LinkedIn Learning","LinkedIn", inplace=True)



plt.subplot( title="Data Science and ML course used") 

courses_bar_chart = sns.barplot( x=courseDataSet.Courses, y=courseDataSet.Involvement, palette='GnBu_d')



plt.show(courses_bar_chart)
specific_col =['Q18_Part_1','Q18_Part_2','Q18_Part_3','Q18_Part_4','Q18_Part_5','Q18_Part_6','Q18_Part_7','Q18_Part_8'

              ,'Q18_Part_9','Q18_Part_10','Q18_Part_11','Q18_Part_12']



focusing_dataset = considered_dataSet.loc[:,specific_col]

lst = [ considered_dataSet[col].value_counts() for col in focusing_dataset ]

pl_dataset = pd.DataFrame(lst)

course_index = list(pl_dataset.columns)

values_course =[ l[0] for l in lst ]

courseDataSet = pd.DataFrame( {"ProgrammingLanguages": course_index, "Usage": values_course} )



# Cleaning not specified data ( 'None' and 'Other' ) 

courseDataSet = courseDataSet.drop(10)

courseDataSet = courseDataSet.drop(11)



plt.tight_layout(5.0)

sns.set( style='whitegrid' ) 

plt.figure(2,figsize=(20,5))

grid = GridSpec(1,2)



plt.subplot(grid[0,0], title="Programming Lang. used in regular basis") 

first_graph = sns.barplot( x=courseDataSet.ProgrammingLanguages, y=courseDataSet.Usage, palette='GnBu_d' )



recommended_pl  = pd.DataFrame( considered_dataSet.loc[ considered_dataSet.Q3 == "Other" ].Q19.value_counts() )



plt.subplot( grid[0,1], title="Recommended Programming Language for DS and ML" )

second_graph = sns.barplot( x=recommended_pl.index, y=recommended_pl.Q19, palette='GnBu_d' )

second_graph.set_ylabel("Recommandation")

second_graph.set_xlabel("Programming Languages")

plt.show(first_graph,second_graph)