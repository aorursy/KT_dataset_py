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
data = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
col = ["gender","group","parental_level_of_education","lunch","test_preparation_course","math_score","reading_score","writing_score"]
data.columns = col
data.head()
data.info()


data[["gender","group","parental_level_of_education","lunch","test_preparation_course"]] = data[["gender","group","parental_level_of_education","lunch","test_preparation_course"]].apply(lambda x : x.astype("category"))
data.test_preparation_course.value_counts()

data.test_preparation_course.replace("none","unknown",inplace = True)
data.group.unique()
group_a = data[data.group =="group A"].iloc[:15,:]

group_b = data[data.group =="group B"].iloc[:15,:]

group_c = data[data.group =="group C"].iloc[:15,:]

group_d = data[data.group =="group D"].iloc[:15,:]

group_e = data[data.group =="group E"].iloc[:15,:]
new_data = pd.concat([group_a,group_b,group_c,group_d,group_e])
new_data.info()
new_data.test_preparation_course=new_data.test_preparation_course.astype("category")
new_data.info()
new_data.parental_level_of_education.unique()
some_high_school = []

high_school = []

some_college =[]

bachelors_degree = []

masters_degree = []

edu_list = list(new_data.parental_level_of_education.unique())

for i in edu_list:

    x = new_data[new_data.parental_level_of_education == i]

    some_high_school.append(x[x.parental_level_of_education == "some high school"])

    high_school.append(x[x.parental_level_of_education == "high school"])

    some_college.append(x[x.parental_level_of_education == "some college"])

    bachelors_degree.append(x[x.parental_level_of_education == "bachelor's degree"])

    masters_degree.append(x[x.parental_level_of_education == "master's degree"])
some_hs = x[x.parental_level_of_education == "some high school"].append(some_high_school)

hs = x[x.parental_level_of_education == "high school"].append(high_school)

sc =  x[x.parental_level_of_education == "some college"].append(some_college)

bd =  x[x.parental_level_of_education == "bachelor's degree"].append(bachelors_degree)

md = x[x.parental_level_of_education =="master's degree"].append(masters_degree)
import matplotlib.pyplot as plt

import seaborn as sns 
di = {"group": bd.group,"math_score": bd.math_score}

df = pd.DataFrame(di)

ni = (df.group.sort_values(ascending = True)).index

sort_data = df.reindex(ni)









plt.figure(figsize = (5,5))

sns.barplot(x = sort_data.group,y = sort_data.math_score)

plt.xticks(rotation = 90)

plt.title("parental level of education factor by childs")

plt.xlabel("math_score")

plt.ylabel("bachelors degree")

plt.show()
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly as py

import plotly.graph_objs as go



init_notebook_mode(connected=True)
import plotly.graph_objs as go

trace1=go.scatter(

    x = hs.group,

    y = hs.math_score,

    mode = "lines",

    name = "group by score",

    marker = dict(color = "rgba(15,25,45,78)"),

    text = hs.parental_level_of_education

)

trace2 = go.scatter(

    x = sc.group,

    y = sc.math_score,

    mode = "lines",

    marker = dict(color = "rgba(20,35,22,90)"),

    text = sc.parental_level_of_education

)

vi = [trace1,trace2];

layout = dict(title = 'family education',

              xaxis= dict(title= 'group',ticklen= 5,zeroline= False)

             )

fig = dict(data = vi, layout = layout)

iplot(fig)