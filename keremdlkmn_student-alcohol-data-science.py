# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import plotly.plotly as py
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from wordcloud import WordCloud

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
student_mat_dataframe = pd.read_csv("../input/student-mat.csv") #read to csv file
print(student_mat_dataframe.columns)
print(student_mat_dataframe.info())
newStudentDataFrame = student_mat_dataframe.loc[:,["Walc","Dalc"]]
import missingno as msno
msno.matrix(newStudentDataFrame)
plt.show()
msno.bar(newStudentDataFrame)
plt.show()
student_mat_dataframe.head() #TOP10 data
student_mat_dataframe.describe()
trace1 = go.Box(
    y = student_mat_dataframe.Dalc,
    name = "Daily alcohol consumption",
    marker = dict(color = "red")
)
trace2 = go.Box(
    y = student_mat_dataframe.Walc,
    name = "Weekly  alcohol consumption",
    marker = dict(color = "blue")
)
concatTrace = [trace1,trace2]
iplot(concatTrace)
trace = go.Scatter3d(
    x = student_mat_dataframe.age,
    y = student_mat_dataframe.Dalc,
    z = student_mat_dataframe.Walc,
    mode = 'markers',
    marker = dict(
        size = 10,
        color = student_mat_dataframe.age,
        colorscale = 'Rainbow'
    )
)
dataTrace = [trace]
layoutTrace = go.Layout(
    margin = dict(
    l = 0,
    r = 0,
    b = 0,
    t = 0 )
)
figTrace = go.Figure(data=dataTrace, layout=layoutTrace)
iplot(figTrace)
#correlation for dataset
f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(student_mat_dataframe.corr(),annot=True,fmt=".1f",linewidths=1,ax=ax)
plt.show()
school_list = list(student_mat_dataframe.school.unique())
print(school_list)
school_list = list(student_mat_dataframe.school.unique())
age_mean = []
for i in school_list:
    x = student_mat_dataframe[student_mat_dataframe.school == i]
    age_sum = sum(x.age)/len(x)
    age_mean.append(age_sum)

#sorting
data = pd.DataFrame({'school_list':school_list,'age_mean':age_mean})
new_index = (data['age_mean'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

#visualization
plt.figure(figsize=(10,10))
sns.barplot(x=sorted_data['school_list'],y=sorted_data['age_mean'])
plt.xticks(rotation=90)
plt.xlabel("School")
plt.ylabel("Age Mean")
plt.show()
school_list = list(student_mat_dataframe.school.unique())
free_time = []
for i in school_list:
    x = student_mat_dataframe[student_mat_dataframe.school == i]
    freeTime = sum(x.freetime)/len(x)
    free_time.append(freeTime)

#sorting
data = pd.DataFrame({'school_list':school_list,'free_time':free_time})
new_index = (data['free_time'].sort_values(ascending=False)).index.values
sorted_data2 = data.reindex(new_index)

#visualization
plt.figure(figsize=(10,10))
sns.barplot(x=sorted_data2['school_list'],y=sorted_data2['free_time'])
plt.xticks(rotation=90)
plt.xlabel("School")
plt.ylabel("Free Time")
plt.show()
fatherJobs_Unique = list(student_mat_dataframe.Fjob.unique())
print(fatherJobs_Unique)
fatherJobs_xAxis = list(student_mat_dataframe.Fjob.unique())
fatherJobs_list = list(student_mat_dataframe.Fjob)
fJobsCounter = Counter(fatherJobs_list)
most_common_jobs = fJobsCounter.most_common(5)
x,y = zip(*most_common_jobs)
x,y = list(x),list(y)

plt.figure(figsize=(15,10))
sns.barplot(x=x,y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel("Father Jobs")
plt.ylabel("Frequency")
plt.title("The profession of all fathers in two schools")
plt.show()
motherJobs_Unique = list(student_mat_dataframe.Mjob.unique())
print(motherJobs_Unique)
motherJobs_xAxis = list(student_mat_dataframe.Mjob.unique())
motherJobs_list = list(student_mat_dataframe.Mjob)
mJobsCounter = Counter(motherJobs_list)
most_common_jobs = mJobsCounter.most_common(5)
x,y = zip(*most_common_jobs)
x,y = list(x),list(y)

plt.figure(figsize=(15,10))
sns.barplot(x=x,y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel("Mother Jobs")
plt.ylabel("Frequency")
plt.title("The profession of all mothers in two schools")
plt.show()
ageList = list(student_mat_dataframe.age.unique())

parentStatusT = []
parentStatusA = []

for i in ageList:
    x = student_mat_dataframe[student_mat_dataframe.age == i]
    parentStatusT.append(sum(x.Pstatus == "T"))
    parentStatusA.append(sum(x.Pstatus == "A"))
    
#sorting
sort_data2 = pd.DataFrame({'student_age':ageList,'student_status':parentStatusA})
new_index4 = (sort_data2['student_status'].sort_values(ascending=False)).index
sorted_data4 = sort_data2.reindex(new_index4)

f,ax = plt.subplots(figsize=(15,10))
sns.barplot(x=ageList,y=parentStatusT,color="red",alpha=0.6,label="Living together")
sns.barplot(x=ageList,y=parentStatusA,color="blue",alpha=0.8,label="Living apart")
ax.legend(loc="upper right",frameon=True)
ax.set(xlabel="Living Together Sum and Living Part Sum ",ylabel="Age",title="Where do children between 15 and 22 live")
plt.show()
print(sorted_data.head())
print(sorted_data2.head())
#Standardization
sorted_data['age_mean'] = sorted_data['age_mean']/max(sorted_data['age_mean'])
sorted_data2['free_time'] = sorted_data2['free_time']/max(sorted_data2['free_time'])

#Data Concat
data = pd.concat([sorted_data,sorted_data2['free_time']],axis=1)
data.sort_values('age_mean',inplace=True)


#Visualization
f,ax1 = plt.subplots(figsize=(20,10))
sns.pointplot(x='school_list',y='age_mean',data=data,color='lime',alpha=0.6)
sns.pointplot(x='school_list',y='free_time',data=data,color='red',alpha=0.8)
plt.xlabel("School",fontsize=15,color='blue')
plt.ylabel("Values",fontsize=15,color='blue')
plt.title("Age average vs. Time spent outside")
plt.grid()

#Notice That: Red: Free Time, Lime: Age Mean
#g = sns.jointplot(data.age_mean,data.free_time,size=7)
#plt.savefig('graph.png')
#plt.show()
#labels = student_mat_dataframe.health.value_counts()
#print(labels)
#index  1 2 3 4 5
#values 146 91 66 47 45
labelsSex = student_mat_dataframe.sex.value_counts().index
colorsSex = ['blue','red']
explodeSex = [0,0]
valuesSex = student_mat_dataframe.sex.value_counts().values

#Visualitizion
plt.figure(figsize=(10,10))
plt.pie(valuesSex,explode=explodeSex,labels=labelsSex,colors=colorsSex,autopct='%1.1f%%')
plt.title("Male and female ratios",color="black",fontsize=15)
plt.show()
labels = student_mat_dataframe.health.value_counts().index 
colors = ['lime','red','blue','green','brown']
explode = [0,0,0,0,0]
values = student_mat_dataframe.health.value_counts().values 

#Visualitizion
plt.figure(figsize=(10,10))
plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title("Health status of students",color="blue",fontsize=15)
plt.show()
fatherJobsList = list(student_mat_dataframe.Fjob.unique())
FstudentDalc = []
for each in fatherJobsList:
    x = student_mat_dataframe[student_mat_dataframe.Fjob == each]
    FstudentDalc.append(sum(x.Dalc))

#sorting
sort_data = pd.DataFrame({'father_jobs':fatherJobsList,'student_dalc':FstudentDalc})
new_index3 = (sort_data['student_dalc'].sort_values(ascending=False)).index
sorted_data3 = sort_data.reindex(new_index3)

#Visualitizon    
plt.figure(figsize=(15,15))
sns.barplot(x=FstudentDalc,y=fatherJobsList)
plt.xticks(rotation=90)
plt.xlabel("Weekly Alcohol Consumption")
plt.ylabel("Father Jobs")
plt.show()
#Standardization
sorted_data3['student_dalc'] = sorted_data3['student_dalc']/max(sorted_data3['student_dalc'])
sorted_data4['student_status'] = sorted_data4['student_status']/max(sorted_data4['student_status'])

#Data Concat
data = pd.concat([sorted_data3,sorted_data4['student_status']],axis=1)
data.sort_values('student_dalc',inplace=True)

sns.lmplot(x="student_status",y="student_dalc",data=data)
plt.show()
pal = sns.cubehelix_palette(2,rot=-5,dark=.5)
sns.violinplot(data=data,palette=pal,inner='points')
plt.show()
f,ax = plt.subplots(figsize=(20,20))
sns.heatmap(data.corr(),annot=True,fmt=".1f",linewidths=1,ax=ax)
plt.show()
plt.figure(figsize=(15,15))
sns.boxplot(x="school",y="age",hue="internet",data=student_mat_dataframe,palette="PRGn")
plt.show()
plt.figure(figsize=(15,15))
sns.swarmplot(x="school",y="age",hue="higher",data=student_mat_dataframe)
plt.show()
student_mat_dataframe2 = student_mat_dataframe.loc[:,["school","age","Dalc","Walc"]]

#Kalıp1 oluşturduk
trace1 = go.Scatter(
    x = student_mat_dataframe2.age,
    y = student_mat_dataframe2.Walc,
    mode = "lines",
    name="Weekly Alcohol Consumption",
    marker = dict(color = 'green'),
    text = student_mat_dataframe2.school
)
#Kalıp2 oluşturduk
trace2 = go.Scatter(
    x = student_mat_dataframe2.age,
    y = student_mat_dataframe2.Dalc,
    mode = "lines+markers",
    name="Daily Alcohol Consumption",
    marker = dict(color = 'blue'),
    text = student_mat_dataframe2.school
)
#Concat Trace
newData = [trace1,trace2]

layout = dict(title="Weekly and daily alcohol consumption of students aged 15 - Line Plot",
             xaxis=dict(title="Age",ticklen=5,zeroline=False))
fig = dict(data = newData,layout=layout)
iplot(fig)