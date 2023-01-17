# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_major = pd.read_csv('../input/utbk-clean-dataset-fixed/majors_fix.csv')

df_score_all = pd.read_csv('../input/utbk-clean-dataset-fixed/scores_all.csv')

df_universities = pd.read_csv('../input/utbk-clean-dataset-fixed/universities.csv')



df_major.set_index("id_major", inplace=True)

df_universities.set_index("id_university", inplace=True)
#all student

#we visualize top 10 from most favorite universities

df_fav_univs = df_score_all.groupby(by="id_university").count().sort_values(by='choice', ascending=False)[0:10]

df_fav_univs = df_fav_univs.reset_index().merge(df_universities, on="id_university", how="left")



#make a new dataframe

df_fav_univs.columns

columns_to_keep = ['choice', 'university_name']

summary_fav_univs = df_fav_univs[columns_to_keep]

summary_fav_univs
#visualise summary_fav_univs

plt.figure(figsize=(20,8))

sns.set_style("whitegrid")

graph = sns.barplot(data=summary_fav_univs, x='choice', y='university_name')

plt.xlim(8000,20000)

plt.ylabel('UNIVERSITIES', color='w')

plt.xlabel('COUNTS', color='w')

plt.title('TOP 10 MOST FAVORITE UNIVERSITY')

plt.rcParams['font.size']=20



for p in graph.patches:

    width = p.get_width()

    plt.text(400+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.0f}'.format(width),

             ha='center', va='center')

        

plt.show()
#by humanities student

#we visualize top 10 from most favorite universities by humanities student

df_fav_univs_hums = df_score_all[df_score_all.type == "humanity"]

df_fav_univs_hums = df_fav_univs_hums.groupby(by="id_university").count().sort_values(by='choice', ascending=False)[0:10]

df_fav_univs_hums = df_fav_univs_hums.reset_index().merge(df_universities, on="id_university", how="left")



#make a new dataframe

df_fav_univs_hums.columns

columns_to_keep = ['choice', 'university_name']

summary_fav_univs_hums = df_fav_univs_hums[columns_to_keep]

summary_fav_univs_hums
#visualize summary_fav_univs_hums

plt.figure(figsize=(20,8))

sns.set_style("whitegrid")

graph = sns.barplot(data=summary_fav_univs_hums, x='choice', y='university_name')

plt.xlim(4500,7000)

plt.ylabel('UNIVERSITIES', color='w')

plt.xlabel('COUNTS', color='w')

plt.title('TOP 10 MOST FAVORITE UNIVERSITY BY HUMANITIES STUDENT')

plt.rcParams['font.size']=20



for p in graph.patches:

    width = p.get_width()

    plt.text(75+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.0f}'.format(width),

             ha='center', va='center')

        

plt.show()
#by science student

#we visualize top 10 from most favorite universities by science student

df_fav_univs_sci = df_score_all[df_score_all.type == "science"]

df_fav_univs_sci = df_fav_univs_sci.groupby(by="id_university").count().sort_values(by='choice', ascending=False)[0:10]

df_fav_univs_sci= df_fav_univs_sci.reset_index().merge(df_universities, on="id_university", how="left")



#make a new dataframe

df_fav_univs_sci.columns

columns_to_keep = ['choice', 'university_name']

summary_fav_univs_sci = df_fav_univs_sci[columns_to_keep]

summary_fav_univs_sci
#vusualize summary_fav_univs_sci

plt.figure(figsize=(20,8))

sns.set_style("whitegrid")

graph = sns.barplot(data=summary_fav_univs_sci, x='choice', y='university_name')

plt.xlim(5500,12500)

plt.ylabel('UNIVERSITIES', color='w')

plt.xlabel('COUNTS', color='w')

plt.title('TOP 10 MOST FAVORITE UNIVERSITY BY SCIENCES STUDENT')

plt.rcParams['font.size']=20



for p in graph.patches:

    width = p.get_width()

    plt.text(250+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.0f}'.format(width),

             ha='center', va='center')

        

plt.show()
#all student

#we visualize top 10 from most favorite majors

df_fav_major = df_score_all.groupby(by="id_major").count().sort_values(by='choice', ascending=False)[0:10]

df_fav_major = df_fav_major.reset_index().merge(df_major, on="id_major", how="left")



#make a new column, contain major_name and university_name

df_fav_major['major_univ'] = df_fav_major['major_name'] + ', ' + df_fav_major['university_name']



#make a new dataframe

df_fav_major.columns

columns_to_keep = ['major_univ', 'choice']

summary_fav_major = df_fav_major[columns_to_keep]

summary_fav_major
#visualize summary_fav_major

plt.figure(figsize=(20,8))

sns.set_style("whitegrid")

graph = sns.barplot(data=summary_fav_major, x='choice', y='major_univ')

plt.xlim(680,1200)

plt.ylabel('MAJOR', color='w')

plt.xlabel('COUNTS', color='w')

plt.title('TOP 10 MOST FAVORITE MAJOR')

plt.rcParams['font.size']=20



for p in graph.patches:

    width = p.get_width()

    plt.text(15+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.0f}'.format(width),

             ha='center', va='center')

        

plt.show()
#by humanities student

#we visualize top 10 from most favorite majors by humanities student

df_fav_major_hums = df_score_all[df_score_all.type == "humanity"]

df_fav_major_hums = df_fav_major_hums.groupby(by="id_major").count().sort_values(by='choice', ascending=False)[0:10]

df_fav_major_hums = df_fav_major_hums.reset_index().merge(df_major, on="id_major", how="left")



#make a new column, contain major_name and university_name

df_fav_major_hums['major_univ'] = df_fav_major_hums['major_name'] + ', ' + df_fav_major_hums['university_name']



#make a new dataframe

df_fav_major_hums.columns

columns_to_keep = ['major_univ', 'choice']

summary_fav_major_hums = df_fav_major_hums[columns_to_keep]

summary_fav_major_hums
#visualize summary_fav_major_hums

plt.figure(figsize=(20,8))

sns.set_style("whitegrid")

graph = sns.barplot(data=summary_fav_major_hums, x='choice', y='major_univ')

plt.xlim(540,1200)

plt.ylabel('MAJOR', color='w')

plt.xlabel('COUNTS', color='w')

plt.title('TOP 10 MOST FAVORITE MAJOR BY HUMANITIES STUDENT')

plt.rcParams['font.size']=20



for p in graph.patches:

    width = p.get_width()

    plt.text(15+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.0f}'.format(width),

             ha='center', va='center')

        

plt.show()
#by sciences student

#we visualize top 10 from most favorite majors by sciences student

df_fav_major_sci = df_score_all[df_score_all.type == "science"]

df_fav_major_sci = df_fav_major_sci.groupby(by="id_major").count().sort_values(by='choice', ascending=False)[0:10]

df_fav_major_sci = df_fav_major_sci.reset_index().merge(df_major, on="id_major", how="left")



#make a new column, contain major_name and university_name

df_fav_major_sci['major_univ'] = df_fav_major_sci['major_name'] + ', ' + df_fav_major_sci['university_name']



#make a new dataframe

df_fav_major_sci.columns

columns_to_keep = ['major_univ', 'choice']

summary_fav_major_sci = df_fav_major_sci[columns_to_keep]

summary_fav_major_sci
#visualize summary_fav_major_sci

plt.figure(figsize=(16,8))

sns.set_style("whitegrid")

graph = sns.barplot(data=summary_fav_major_sci, x='choice', y='major_univ')

plt.xlim(620,820)

plt.ylabel('MAJOR', color='w')

plt.xlabel('COUNTS', color='w')

plt.title('TOP 10 MOST FAVORITE SCIENCES MAJOR')

plt.rcParams['font.size']=20



for p in graph.patches:

    width = p.get_width()

    plt.text(5+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.0f}'.format(width),

             ha='center', va='center')

        

plt.show()
#all students

#we visualize top 10 from least favorite universities

df_lst_univs = df_score_all.groupby(by="id_university").count().sort_values(by='choice', ascending=True)[0:10]

df_lst_univs = df_lst_univs.reset_index().merge(df_universities, on="id_university", how="left")



#make a new dataframe

df_lst_univs.columns

columns_to_keep = ['choice', 'university_name']

summary_lst_univs = df_lst_univs[columns_to_keep]

summary_lst_univs = summary_lst_univs.sort_values(by='choice', ascending=False)

summary_lst_univs
#visualize summary_lst_univs

plt.figure(figsize=(20,8))

sns.set_style("whitegrid")

graph = sns.barplot(data=summary_lst_univs, x='choice', y='university_name')

plt.xlim(0,100)

plt.ylabel('UNIVERSITIES', color='w')

plt.xlabel('COUNTS', color='w')

plt.title('TOP 10 MOST LEAST FAVORITE UNIVERSITY')

plt.rcParams['font.size']=20



for p in graph.patches:

    width = p.get_width()

    plt.text(2+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.0f}'.format(width),

             ha='center', va='center')

        

plt.show()
#by humanities student

#we visualize top 10 from least favorite universities by humanities student

df_lst_univs_hums = df_score_all[df_score_all.type == "humanity"]

df_lst_univs_hums = df_lst_univs_hums.groupby(by="id_university").count().sort_values(by='choice', ascending=True)[0:10]

df_lst_univs_hums = df_lst_univs_hums.reset_index().merge(df_universities, on="id_university", how="left")



#make a new dataframe

df_lst_univs_hums.columns

columns_to_keep = ['choice', 'university_name']

summary_lst_univs_hums = df_lst_univs_hums[columns_to_keep]

summary_lst_univs_hums = summary_lst_univs_hums.sort_values(by='choice', ascending=False)

summary_lst_univs_hums
#visualize summary_lst_univs_hums

plt.figure(figsize=(20,8))

sns.set_style("whitegrid")

graph = sns.barplot(data=summary_lst_univs_hums, x='choice', y='university_name')

plt.xlim(0,50)

plt.ylabel('UNIVERSITIES', color='w')

plt.xlabel('COUNTS', color='w')

plt.title('TOP 10 MOST LEAST FAVORITE UNIVERSITY OF HUMANITIES MAJOR')

plt.rcParams['font.size']=20



for p in graph.patches:

    width = p.get_width()

    plt.text(1+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.0f}'.format(width),

             ha='center', va='center')

        

plt.show()
#by sciences student

#we visualize top 10 from least favorite universities by sciences student

df_lst_univs_sci = df_score_all[df_score_all.type == "science"]

df_lst_univs_sci = df_lst_univs_sci.groupby(by="id_university").count().sort_values(by='choice', ascending=True)[0:10]

df_lst_univs_sci= df_lst_univs_sci.reset_index().merge(df_universities, on="id_university", how="left")



#make a new dataframe

df_lst_univs_sci.columns

columns_to_keep = ['choice', 'university_name']

summary_lst_univs_sci = df_lst_univs_sci[columns_to_keep]

summary_lst_univs_sci = summary_lst_univs_sci.sort_values(by='choice', ascending=False)

summary_lst_univs_sci
#visualize summary_lst_univs_sci

plt.figure(figsize=(20,8))

sns.set_style("whitegrid")

graph = sns.barplot(data=summary_lst_univs_sci, x='choice', y='university_name')

plt.xlim(0,110)

plt.ylabel('UNIVERSITIES', color='w')

plt.xlabel('COUNTS', color='w')

plt.title('TOP 10 MOST LEAST FAVORITE UNIVERSITY OF SCIENCES MAJOR')

plt.rcParams['font.size']=20



for p in graph.patches:

    width = p.get_width()

    plt.text(2.5+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.0f}'.format(width),

             ha='center', va='center')

        

plt.show()
#all student

#we visualize top 10 from least favorite majors

df_lst_major = df_score_all.groupby(by="id_major").sum().sort_values(by='choice', ascending=True)[0:10]

df_lst_major = df_lst_major.reset_index().merge(df_major, on="id_major", how="left")



#make a new column, contain major_name and university_name

df_lst_major['major_univ'] = df_lst_major['major_name'] + ', ' + df_lst_major['university_name']



#make a new dataframe

df_lst_major.columns

columns_to_keep = ['major_univ', 'choice']

summary_lst_major = df_lst_major[columns_to_keep]

summary_lst_major
#visualize summary_lst_major

plt.figure(figsize=(16,8))

sns.set_style("whitegrid")

graph = sns.barplot(data=summary_lst_major, x='choice', y='major_univ')

plt.xlim(0,10)

plt.ylabel('MAJOR', color='w')

plt.xlabel('COUNTS', color='w')

plt.title('TOP 10 MOST LEAST FAVORITE MAJOR')

plt.rcParams['font.size']=20



for p in graph.patches:

    width = p.get_width()

    plt.text(0.25+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.0f}'.format(width),

             ha='center', va='center')

        

plt.show()
#by humanities student

#we visualize top 10 from least favorite majors by humanities student

df_lst_major_hums = df_score_all[df_score_all.type == "humanity"]

df_lst_major_hums = df_lst_major_hums.groupby(by="id_major").count().sort_values(by='choice', ascending=True)[0:10]

df_lst_major_hums = df_lst_major_hums.reset_index().merge(df_major, on="id_major", how="left")



#make a new column, contain major_name and university_name

df_lst_major_hums['major_univ'] = df_lst_major_hums['major_name'] + ', ' + df_lst_major_hums['university_name']



#make a new dataframe

df_lst_major_hums.columns

columns_to_keep = ['major_univ', 'choice']

summary_lst_major_hums = df_lst_major_hums[columns_to_keep]

summary_lst_major_hums
#visualize summary_lst_major_hums

plt.figure(figsize=(16,8))

sns.set_style("whitegrid")

graph = sns.barplot(data=summary_lst_major_hums, x='choice', y='major_univ')

plt.xlim(0,10)

plt.ylabel('MAJOR', color='w')

plt.xlabel('COUNTS', color='w')

plt.title('TOP 10 MOST LEAST FAVORITE MAJOR BY HUMANITIES STUDENT')

plt.rcParams['font.size']=20



for p in graph.patches:

    width = p.get_width()

    plt.text(0.25+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.0f}'.format(width),

             ha='center', va='center')

        

plt.show()
#by sciences student

#we visualize top 10 from least favorite majors by sciences student

df_lst_major_sci = df_score_all[df_score_all.type == "science"]

df_lst_major_sci = df_lst_major_sci.groupby(by="id_major").count().sort_values(by='choice', ascending=True)[0:10]

df_lst_major_sci = df_lst_major_sci.reset_index().merge(df_major, on="id_major", how="left")



#make a new column, contain major_name and university_name

df_lst_major_sci['major_univ'] = df_lst_major_sci['major_name'] + ', ' + df_lst_major_sci['university_name']



#make a new dataframe

df_lst_major_sci.columns

columns_to_keep = ['major_univ', 'choice']

summary_lst_major_sci = df_lst_major_sci[columns_to_keep]

summary_lst_major_sci
#visualize summary_lst_major_sci

plt.figure(figsize=(16,8))

sns.set_style("whitegrid")

graph = sns.barplot(data=summary_lst_major_sci, x='choice', y='major_univ')

plt.xlim(0,10)

plt.ylabel('MAJOR', color='w')

plt.xlabel('COUNTS', color='w')

plt.title('TOP 10 MOST LEAST FAVORITE SCIENCES MAJOR')

plt.rcParams['font.size']=20



for p in graph.patches:

    width = p.get_width()

    plt.text(0.25+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.0f}'.format(width),

             ha='center', va='center')

        

plt.show()
#calculate passing grade

def get_passing_grade(index, percent = 20):

    capacity = df_major.loc[[index]].capacity.values[0] * percent // 100

    return round(df_score_all[df_score_all.id_major == index].nlargest(capacity, "avg_score").avg_score.min(),2)



#make a new column

df_major['passing_grade'] = df_major.index.to_series().apply(lambda x: get_passing_grade(x, 25))
#calculate highest grade

def get_highest_score(index):

    return round(df_score_all[df_score_all.id_major == index].avg_score.max(), 2)



#make a new column

df_major['highest_score'] = df_major.index.to_series().apply(lambda x: get_highest_score(x))
#calculate acceptance ratio

def get_acceptance_ratio(index, percent = 20):

    capacity = df_major.loc[[index]].capacity.values[0] * percent // 100

    applicants = df_score_all[df_score_all.id_major == index].shape[0]    

    return 0 if applicants==0 else round(capacity / applicants , 2)



df_major['acceptance_ratio'] = df_major.index.to_series().apply(lambda x: get_acceptance_ratio(x, 25))
#calculate applicant

def get_applicant(index):

    applicants = df_score_all[df_score_all.id_major == index].shape[0]

    return applicants



#make a new column

df_major['applicants'] = df_major.index.to_series().apply(lambda x: get_applicant(x))
#calculate score range

def get_score_range(index):

    score_max = df_score_all[df_score_all.id_major == index].avg_score.max()

    score_min = df_score_all[df_score_all.id_major == index].avg_score.min()

    return round(score_max - score_min, 2)



#make a new column

df_major['score_range'] = df_major.index.to_series().apply(lambda x: get_score_range(x))
#calculate lowest score

def get_min_score(index):

    score_min = df_score_all[df_score_all.id_major == index].avg_score.min()

    return round(score_min, 2)



#make a new column

df_major['lowest_score'] = df_major.index.to_series().apply(lambda x: get_min_score(x))
#highest passing grade

pass_grade = df_major[[ 'major_name', 'university_name', 'capacity', 'applicants', 'passing_grade', 'lowest_score', 'highest_score', 'score_range', 'acceptance_ratio']]



#make a copy

pass_grade1 = pass_grade.copy()

pass_grade1['major_univ'] = pass_grade1['major_name'] + ', ' + pass_grade1['university_name']



#sort

highest_pass_grade = pass_grade1.nlargest(10, 'passing_grade').loc[:,'major_name':]



#visualize it

plt.figure(figsize=(18,10))

sns.set_style("whitegrid")

graph = sns.barplot(data=highest_pass_grade, x='passing_grade', y='major_univ')

plt.xlim(720,770)

plt.ylabel('MAJOR', color='w')

plt.xlabel('COUNTS', color='w')

plt.title('TOP 10 HIGHEST PASSING GRADE')

plt.rcParams['font.size']=20



for p in graph.patches:

    width = p.get_width()

    plt.text(2+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.0f}'.format(width),

             ha='center', va='center')

        

plt.show()
#highest passing grade

#sort

highest_acc_ratio = pass_grade1.nlargest(10, 'acceptance_ratio').loc[:,'major_name':]



#visualize it

plt.figure(figsize=(18,10))

sns.set_style("whitegrid")

graph = sns.barplot(data=highest_acc_ratio, x='acceptance_ratio', y='major_univ')

plt.xlim(20,50)

plt.ylabel('MAJOR', color='w')

plt.xlabel('COUNTS', color='w')

plt.title('TOP 10 HIGHEST PASSING GRADE')

plt.rcParams['font.size']=20



for p in graph.patches:

    width = p.get_width()

    plt.text(1+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.1f}'.format(width),

             ha='center', va='center')

        

plt.show()
#lowest passing grade

#sort

lowest_pass_grade = pass_grade1.nsmallest(10, 'passing_grade').sort_values(by='passing_grade', ascending=False)



#visualize it

plt.figure(figsize=(18,10))

sns.set_style("whitegrid")

graph = sns.barplot(data=lowest_pass_grade, x='passing_grade', y='major_univ')

plt.xlim(340, 390)

plt.ylabel('MAJOR', color='w')

plt.xlabel('COUNTS', color='w')

plt.title('TOP 10 LOWEST PASSING GRADE')

plt.rcParams['font.size']=20



for p in graph.patches:

    width = p.get_width()

    plt.text(2+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.1f}'.format(width),

             ha='center', va='center')

        

plt.show()
#lowest acceptance ratio

#dropna values

pass_grade1.dropna(inplace=True)



#sort

lowest_acc_ratio = pass_grade1.nsmallest(10, 'acceptance_ratio').sort_values(by='acceptance_ratio', ascending=False)



#visualize it

plt.figure(figsize=(18,8))

sns.set_style("whitegrid")

graph = sns.barplot(data=lowest_acc_ratio, x='acceptance_ratio', y='major_univ')

plt.xlim(0, 1)

plt.ylabel('MAJOR', color='w')

plt.xlabel('COUNTS', color='w')

plt.title('TOP 10 LOWEST ACCEPTANCE RATIO')

plt.rcParams['font.size']=20



for p in graph.patches:

    width = p.get_width()

    plt.text(0.03+p.get_width(), p.get_y()+0.55*p.get_height(),

             '{:.2f}'.format(width),

             ha='center', va='center')

        

plt.show()