# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib.gridspec import GridSpec
import plotly.graph_objs as go
import warnings
from plotly.offline import init_notebook_mode,iplot
import plotly.graph_objs as go
import plotly.plotly as py
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)
pd.set_option('display.max_columns', 5000)
# Any results you write to the current directory are saved as output.
multiple_data = pd.read_csv("../input/multipleChoiceResponses.csv")
#What is the age distribution participating survey? 
ages = multiple_data.Q2.value_counts()
x = list(ages.index)
y = list(ages.values)
chart = go.Pie(labels=x, values=y)
layout = go.Layout(title='Age Distribution Participating Survey')
fig = go.Figure(data=[chart], layout=layout)
iplot(fig, filename="age")


#Gender based IDE usage ratios
gender_female = multiple_data[multiple_data.Q1 == "Female"]
gender_male = multiple_data[multiple_data.Q1 == "Male"]

female_dict = {"Python":gender_female['Q16_Part_1'].value_counts().values, "R":gender_female['Q16_Part_2'].value_counts().values,"SQL":gender_female['Q16_Part_3'].value_counts().values,"Bash":gender_female['Q16_Part_4'].value_counts().values,
                   "Java":gender_female['Q16_Part_5'].value_counts().values,"JS":gender_female['Q16_Part_6'].value_counts().values,"VBA":gender_female['Q16_Part_7'].value_counts().values,"CPlusPlus":gender_female['Q16_Part_8'].value_counts().values,
                   "MATLAB":gender_female['Q16_Part_9'].value_counts().values,"Scala":gender_female['Q16_Part_10'].value_counts().values,"Julia":gender_female['Q16_Part_11'].value_counts().values,"Go":gender_female['Q16_Part_12'].value_counts().values,
                   "CSharp":gender_female['Q16_Part_13'].value_counts().values,"PHP":gender_female['Q16_Part_14'].value_counts().values,"Ruby":gender_female['Q16_Part_15'].value_counts().values,"SAS":gender_female['Q16_Part_16'].value_counts().values,
                   "None":gender_female['Q16_Part_17'].value_counts().values}
male_dict = {"Python":gender_male['Q16_Part_1'].value_counts().values, "R":gender_male['Q16_Part_2'].value_counts().values,"SQL":gender_male['Q16_Part_3'].value_counts().values,"Bash":gender_male['Q16_Part_4'].value_counts().values,
                   "Java":gender_male['Q16_Part_5'].value_counts().values,"JS":gender_male['Q16_Part_6'].value_counts().values,"VBA":gender_male['Q16_Part_7'].value_counts().values,"CPlusPlus":gender_male['Q16_Part_8'].value_counts().values,
                   "MATLAB":gender_male['Q16_Part_9'].value_counts().values,"Scala":gender_male['Q16_Part_10'].value_counts().values,"Julia":gender_male['Q16_Part_11'].value_counts().values,"Go":gender_male['Q16_Part_12'].value_counts().values,
                   "CSharp":gender_male['Q16_Part_13'].value_counts().values,"PHP":gender_male['Q16_Part_14'].value_counts().values,"Ruby":gender_male['Q16_Part_15'].value_counts().values,"SAS":gender_male['Q16_Part_16'].value_counts().values,
                   "None":gender_male['Q16_Part_17'].value_counts().values}

labels = list(female_dict.keys())
sizes1 = list(female_dict.values())
sizes2 = list(male_dict.values())
data1 = pd.DataFrame({'labels': labels,'sizes': sizes1})
data2 = pd.DataFrame({'labels': labels,'sizes': sizes2})
data1['sizes'] = [each[0] for each in data1['sizes']]
data2['sizes'] = [each[0] for each in data2['sizes']]
new_index = (data1['sizes'].sort_values(ascending=False)).index.values
sorted_data1 = data1.reindex(new_index)
new_index = (data2['sizes'].sort_values(ascending=False)).index.values
sorted_data2 = data2.reindex(new_index)

f,axes = plt.subplots(figsize = (20,8),nrows=1,ncols=2)
g1=sns.barplot(x=sorted_data1["labels"],y=sorted_data1["sizes"],color='red',alpha = 0.5,label='Female', ax=axes[0])
g1.set_xticklabels(sorted_data1["labels"],rotation=45)
g2=sns.barplot(x=sorted_data2["labels"],y=sorted_data2["sizes"],color='blue',alpha = 0.7,label='Male',ax=axes[1])
g2.set_xticklabels(sorted_data1["labels"],rotation=45)

axes[0].set(xlabel='Programming Languages', ylabel='Count',title = "Female Programming Language Usage ")
axes[1].set(xlabel='Programming Languages', ylabel='Count',title = "Male Programming Language Usage ")

plt.show()
#which countries participating survey
countries = multiple_data.Q3
x = list(countries.index)
x = [str(x) for each in x]
plt.subplots(figsize=(10,10))
wordcloud = WordCloud(
                          background_color='white',
                          width=1200,
                          height=800
                         ).generate(" ".join(countries))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
programming_dict = {"Python":multiple_data['Q16_Part_1'].value_counts().values, "R":multiple_data['Q16_Part_2'].value_counts().values,"SQL":multiple_data['Q16_Part_3'].value_counts().values,"Bash":multiple_data['Q16_Part_4'].value_counts().values,
                   "Java":multiple_data['Q16_Part_5'].value_counts().values,"JS":multiple_data['Q16_Part_6'].value_counts().values,"VBA":multiple_data['Q16_Part_7'].value_counts().values,"CPlusPlus":multiple_data['Q16_Part_8'].value_counts().values,
                   "MATLAB":multiple_data['Q16_Part_9'].value_counts().values,"Scala":multiple_data['Q16_Part_10'].value_counts().values,"Julia":multiple_data['Q16_Part_11'].value_counts().values,"Go":multiple_data['Q16_Part_12'].value_counts().values,
                   "CSharp":multiple_data['Q16_Part_13'].value_counts().values,"PHP":multiple_data['Q16_Part_14'].value_counts().values,"Ruby":multiple_data['Q16_Part_15'].value_counts().values,"SAS":multiple_data['Q16_Part_16'].value_counts().values,
                   "None":multiple_data['Q16_Part_17'].value_counts().values}
labels_prog = list(programming_dict.keys())
sizes_prog = list(programming_dict.values())
data = pd.DataFrame({'labels': labels_prog,'sizes': sizes_prog})
data['sizes'] = [each[0] for each in data['sizes']]
new_index = (data['sizes'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

plt.figure(figsize=(12,9))
sns.barplot(x=sorted_data['labels'], y=sorted_data['sizes'])
plt.xticks(rotation= 45)
plt.xlabel('Programming Language')
plt.ylabel('Frequency')
plt.title('Programming Language Usage')
plt.show()
ide_dict = {"IPython":multiple_data['Q13_Part_1'].value_counts().values, "RStudio":multiple_data['Q13_Part_2'].value_counts().values,"PyCharm":multiple_data['Q13_Part_3'].value_counts().values,"VisualStudioCode":multiple_data['Q13_Part_4'].value_counts().values,
            "Nteract":multiple_data['Q13_Part_5'].value_counts().values,"Atom":multiple_data['Q13_Part_6'].value_counts().values,"Matlab":multiple_data['Q13_Part_7'].value_counts().values,"VisualStudio":multiple_data['Q13_Part_8'].value_counts().values,
            "NotepadPlusPlus":multiple_data['Q13_Part_9'].value_counts().values,"SublimeText":multiple_data['Q13_Part_10'].value_counts().values,"Vim":multiple_data['Q16_Part_11'].value_counts().values,"IntelliJ":multiple_data['Q13_Part_12'].value_counts().values,
            "Spyder":multiple_data['Q13_Part_13'].value_counts().values,"None":multiple_data['Q13_Part_14'].value_counts().values}

labels_prog = list(ide_dict.keys())
sizes_prog = list(ide_dict.values())
data = pd.DataFrame({'labels': labels_prog,'sizes': sizes_prog})
data['sizes'] = [each[0] for each in data['sizes']]
new_index = (data['sizes'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

plt.figure(figsize=(12,9))
sns.barplot(x=sorted_data['labels'], y=sorted_data['sizes'])
plt.xticks(rotation= 45)
plt.xlabel('IDE')
plt.ylabel('Frequency')
plt.title('IDE Usage')
plt.show()
# industry based programming language distribution
#top three industry and their programming language distribution
#First - Computers/Technology
computers_technology = multiple_data[multiple_data.Q7 == "Computers/Technology"]
students = multiple_data[multiple_data.Q7 == "I am a student"]
academics = multiple_data[multiple_data.Q7 == "Academics/Education"]

programming_dict1 = {"Python":computers_technology['Q16_Part_1'].value_counts().values, "R":computers_technology['Q16_Part_2'].value_counts().values,"SQL":computers_technology['Q16_Part_3'].value_counts().values,"Bash":computers_technology['Q16_Part_4'].value_counts().values,
                   "Java":computers_technology['Q16_Part_5'].value_counts().values,"JS":computers_technology['Q16_Part_6'].value_counts().values,"VBA":computers_technology['Q16_Part_7'].value_counts().values,"CPlusPlus":computers_technology['Q16_Part_8'].value_counts().values,
                   "MATLAB":computers_technology['Q16_Part_9'].value_counts().values,"Scala":computers_technology['Q16_Part_10'].value_counts().values,"Julia":computers_technology['Q16_Part_11'].value_counts().values,"Go":computers_technology['Q16_Part_12'].value_counts().values,
                   "CSharp":computers_technology['Q16_Part_13'].value_counts().values,"PHP":computers_technology['Q16_Part_14'].value_counts().values,"Ruby":computers_technology['Q16_Part_15'].value_counts().values,"SAS":computers_technology['Q16_Part_16'].value_counts().values,
                   "None":computers_technology['Q16_Part_17'].value_counts().values}
programming_dict2 = {"Python":students['Q16_Part_1'].value_counts().values, "R":students['Q16_Part_2'].value_counts().values,"SQL":students['Q16_Part_3'].value_counts().values,"Bash":students['Q16_Part_4'].value_counts().values,
                   "Java":students['Q16_Part_5'].value_counts().values,"JS":students['Q16_Part_6'].value_counts().values,"VBA":students['Q16_Part_7'].value_counts().values,"CPlusPlus":students['Q16_Part_8'].value_counts().values,
                   "MATLAB":students['Q16_Part_9'].value_counts().values,"Scala":students['Q16_Part_10'].value_counts().values,"Julia":students['Q16_Part_11'].value_counts().values,"Go":students['Q16_Part_12'].value_counts().values,
                   "CSharp":students['Q16_Part_13'].value_counts().values,"PHP":students['Q16_Part_14'].value_counts().values,"Ruby":students['Q16_Part_15'].value_counts().values,"SAS":students['Q16_Part_16'].value_counts().values,
                   "None":students['Q16_Part_17'].value_counts().values}
programming_dict3 = {"Python":academics['Q16_Part_1'].value_counts().values, "R":academics['Q16_Part_2'].value_counts().values,"SQL":academics['Q16_Part_3'].value_counts().values,"Bash":academics['Q16_Part_4'].value_counts().values,
                   "Java":academics['Q16_Part_5'].value_counts().values,"JS":academics['Q16_Part_6'].value_counts().values,"VBA":academics['Q16_Part_7'].value_counts().values,"CPlusPlus":academics['Q16_Part_8'].value_counts().values,
                   "MATLAB":academics['Q16_Part_9'].value_counts().values,"Scala":academics['Q16_Part_10'].value_counts().values,"Julia":academics['Q16_Part_11'].value_counts().values,"Go":academics['Q16_Part_12'].value_counts().values,
                   "CSharp":academics['Q16_Part_13'].value_counts().values,"PHP":academics['Q16_Part_14'].value_counts().values,"Ruby":academics['Q16_Part_15'].value_counts().values,"SAS":academics['Q16_Part_16'].value_counts().values,
                   "None":academics['Q16_Part_17'].value_counts().values}
labels = list(programming_dict1.keys())
sizes1 = list(programming_dict1.values())
sizes2 = list(programming_dict2.values())
sizes3 = list(programming_dict3.values())
data = pd.DataFrame({'labels': labels,'sizes1': sizes1,'sizes2':sizes2,'sizes3':sizes3})
data['sizes1'] = [each[0] for each in data['sizes1']]
data['sizes2'] = [each[0] for each in data['sizes2']]
data['sizes3'] = [each[0] for each in data['sizes3']]

f,ax = plt.subplots(figsize = (9,12))
sns.barplot(x=data["sizes1"],y=data["labels"],color='green',alpha = 0.5,label='Computers/Technology' )
sns.barplot(x=data["sizes2"],y=data["labels"],color='blue',alpha = 0.7,label='Students')
sns.barplot(x=data["sizes3"],y=data["labels"],color='cyan',alpha = 0.6,label='Academics')


ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu
ax.set(xlabel='Count', ylabel='Programming Languages',title = "Industry Based Programming Language Rates")
plt.show()
computers_technology = multiple_data[multiple_data.Q7 == "Computers/Technology"]
students = multiple_data[multiple_data.Q7 == "I am a student"]
academics = multiple_data[multiple_data.Q7 == "Academics/Education"]

programming_dict1 = {"IPython":computers_technology['Q13_Part_1'].value_counts().values, "RStudio":computers_technology['Q13_Part_2'].value_counts().values,"PyCharm":computers_technology['Q13_Part_3'].value_counts().values,"VisualStudioCode":computers_technology['Q13_Part_4'].value_counts().values,
                   "Nteract":computers_technology['Q13_Part_5'].value_counts().values,"Atom":computers_technology['Q13_Part_6'].value_counts().values,"Matlab":computers_technology['Q13_Part_7'].value_counts().values,"VisualStudio":computers_technology['Q13_Part_8'].value_counts().values,
                   "NotepadPlusPlus":computers_technology['Q13_Part_9'].value_counts().values,"SublimeText":computers_technology['Q13_Part_10'].value_counts().values,"Vim":computers_technology['Q16_Part_11'].value_counts().values,"IntelliJ":computers_technology['Q13_Part_12'].value_counts().values,
                   "Spyder":computers_technology['Q13_Part_13'].value_counts().values,"None":computers_technology['Q13_Part_14'].value_counts().values}
programming_dict2 = {"IPython":students['Q13_Part_1'].value_counts().values, "RStudio":students['Q13_Part_2'].value_counts().values,"PyCharm":students['Q13_Part_3'].value_counts().values,"VisualStudioCode":students['Q13_Part_4'].value_counts().values,
                   "Nteract":students['Q13_Part_5'].value_counts().values,"Atom":students['Q13_Part_6'].value_counts().values,"Matlab":students['Q13_Part_7'].value_counts().values,"VisualStudio":students['Q13_Part_8'].value_counts().values,
                   "NotepadPlusPlus":students['Q13_Part_9'].value_counts().values,"SublimeText":students['Q13_Part_10'].value_counts().values,"Vim":students['Q13_Part_11'].value_counts().values,"IntelliJ":students['Q13_Part_12'].value_counts().values,
                   "Spyder":students['Q13_Part_13'].value_counts().values,"None":students['Q13_Part_14'].value_counts().values}
programming_dict3 = {"IPython":academics['Q13_Part_1'].value_counts().values, "RStudio":academics['Q13_Part_2'].value_counts().values,"PyCharm":academics['Q13_Part_3'].value_counts().values,"VisualStudioCode":academics['Q13_Part_4'].value_counts().values,
                   "Nteract":academics['Q13_Part_5'].value_counts().values,"Atom":academics['Q13_Part_6'].value_counts().values,"Matlab":academics['Q13_Part_7'].value_counts().values,"VisualStudio":academics['Q13_Part_8'].value_counts().values,
                   "NotepadPlusPlus":academics['Q13_Part_9'].value_counts().values,"SublimeText":academics['Q13_Part_10'].value_counts().values,"Vim":academics['Q13_Part_11'].value_counts().values,"IntelliJ":academics['Q13_Part_12'].value_counts().values,
                   "Spyder":academics['Q13_Part_13'].value_counts().values,"None":academics['Q13_Part_14'].value_counts().values}


labels = list(programming_dict1.keys())
sizes1 = list(programming_dict1.values())
sizes2 = list(programming_dict2.values())
sizes3 = list(programming_dict3.values())

data = pd.DataFrame({'labels': labels,'sizes1': sizes1,'sizes2':sizes2,'sizes3':sizes3})
data['sizes1'] = [each[0] for each in data['sizes1']]
data['sizes2'] = [each[0] for each in data['sizes2']]
data['sizes3'] = [each[0] for each in data['sizes3']]

f,ax = plt.subplots(figsize = (9,12))
sns.barplot(x=data["sizes1"],y=data["labels"],color='gray',alpha = 0.5,label='Computers/Technology' )
sns.barplot(x=data["sizes2"],y=data["labels"],color='yellow',alpha = 0.7,label='Students')
sns.barplot(x=data["sizes3"],y=data["labels"],color='black',alpha = 0.6,label='Academics')


ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu
ax.set(xlabel='Count', ylabel='IDE')
ax.set_title(label = "Industry Based IDE Usage Rates")
plt.show()

# General ML Framework Usage
multiple_data_copy = multiple_data[1:]
mlframework_1 = pd.concat([multiple_data_copy["Q19_Part_1"],multiple_data_copy["Q19_Part_2"],multiple_data_copy["Q19_Part_3"],multiple_data_copy["Q19_Part_4"],multiple_data_copy["Q19_Part_5"],
                           multiple_data_copy["Q19_Part_6"],multiple_data_copy["Q19_Part_7"],multiple_data_copy["Q19_Part_8"],multiple_data_copy["Q19_Part_9"],multiple_data_copy["Q19_Part_10"]])
mlframework_2 = pd.concat([multiple_data_copy["Q19_Part_11"],multiple_data_copy["Q19_Part_12"],multiple_data_copy["Q19_Part_13"],multiple_data_copy["Q19_Part_14"],multiple_data_copy["Q19_Part_15"],
                           multiple_data_copy["Q19_Part_16"],multiple_data_copy["Q19_Part_17"]])
mlframework = pd.concat([mlframework_1,mlframework_2])
ml_counts = mlframework.value_counts()

plt.figure(figsize=(12,9))
sns.barplot(x=ml_counts.index, y=ml_counts.values)
plt.xticks(rotation= 45)
plt.xlabel('ML Framework')
plt.ylabel('Frequency')
plt.title('ML Framework Usage')
plt.show()
# Top 3 Industry Based ML Framework Distribution
ml_computers_data = multiple_data_copy[multiple_data_copy.Q7 == "Computers/Technology"]
ml_students_data = multiple_data_copy[multiple_data_copy.Q7 == "I am a student"]
ml_academics_data = multiple_data_copy[multiple_data_copy.Q7 == "Academics/Education"]

computers_dict1 = {"Scikit-Learn":ml_computers_data['Q19_Part_1'].count(), "TensorFlow":ml_computers_data['Q19_Part_2'].count(),"Keras":ml_computers_data['Q19_Part_3'].count(),"PyTorch":ml_computers_data['Q19_Part_4'].count(),
                   "Spark MLlib":ml_computers_data['Q19_Part_5'].count(),"H20":ml_computers_data['Q19_Part_6'].count(),"Fastai":ml_computers_data['Q19_Part_7'].count(),"Mxnet":ml_computers_data['Q19_Part_8'].count(),
                   "Caret":ml_computers_data['Q19_Part_9'].count(),"Xgboost":ml_computers_data['Q19_Part_10'].count(),"mlr":ml_computers_data['Q19_Part_11'].count(),"Prophet":ml_computers_data['Q19_Part_12'].count(),
                   "randomForest":ml_computers_data['Q19_Part_13'].count(),"lightgbm":ml_computers_data['Q19_Part_14'].count(),"catboost":ml_computers_data['Q19_Part_15'].count(),"CNTK":ml_computers_data['Q19_Part_16'].count(),
                   "Caffe":ml_computers_data['Q19_Part_17'].count()}
students_dict2 = {"Scikit-Learn":ml_students_data['Q19_Part_1'].count(), "TensorFlow":ml_students_data['Q19_Part_2'].count(),"Keras":ml_students_data['Q19_Part_3'].count(),"PyTorch":ml_students_data['Q19_Part_4'].count(),
                   "Spark MLlib":ml_students_data['Q19_Part_5'].count(),"H20":ml_students_data['Q19_Part_6'].count(),"Fastai":ml_students_data['Q19_Part_7'].count(),"Mxnet":ml_students_data['Q19_Part_8'].count(),
                   "Caret":ml_students_data['Q19_Part_9'].count(),"Xgboost":ml_students_data['Q19_Part_10'].count(),"mlr":ml_students_data['Q19_Part_11'].count(),"Prophet":ml_students_data['Q19_Part_12'].count(),
                   "randomForest":ml_students_data['Q19_Part_13'].count(),"lightgbm":ml_students_data['Q19_Part_14'].count(),"catboost":ml_students_data['Q19_Part_15'].count(),"CNTK":ml_students_data['Q19_Part_16'].count(),
                   "Caffe":ml_students_data['Q19_Part_17'].count()}
academics_dict3 = {"Scikit-Learn":ml_academics_data['Q19_Part_1'].count(), "TensorFlow":ml_academics_data['Q19_Part_2'].count(),"Keras":ml_academics_data['Q19_Part_3'].count(),"PyTorch":ml_academics_data['Q19_Part_4'].count(),
                   "Spark MLlib":ml_academics_data['Q19_Part_5'].count(),"H20":ml_academics_data['Q19_Part_6'].count(),"Fastai":ml_academics_data['Q19_Part_7'].count(),"Mxnet":ml_academics_data['Q19_Part_8'].count(),
                   "Caret":ml_academics_data['Q19_Part_9'].count(),"Xgboost":ml_academics_data['Q19_Part_10'].count(),"mlr":ml_academics_data['Q19_Part_11'].count(),"Prophet":ml_academics_data['Q19_Part_12'].count(),
                   "randomForest":ml_academics_data['Q19_Part_13'].count(),"lightgbm":ml_academics_data['Q19_Part_14'].count(),"catboost":ml_academics_data['Q19_Part_15'].count(),"CNTK":ml_academics_data['Q19_Part_16'].count(),
                   "Caffe":ml_academics_data['Q19_Part_17'].count()}

labels = list(computers_dict1.keys())
sizes1 = list(computers_dict1.values())
sizes2 = list(students_dict2.values())
sizes3 = list(academics_dict3.values())

f,ax = plt.subplots(figsize = (9,12))
sns.barplot(x=sizes1,y=labels,color='orange',alpha = 0.5,label='Computers/Technology' )
sns.barplot(x=sizes2,y=labels,color='gray',alpha = 0.7,label='Students')
sns.barplot(x=sizes3,y=labels,color='black',alpha = 0.6,label='Academics')


ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu
ax.set(xlabel='Count', ylabel='ML Framework')
ax.set_title(label = "Computers/Technology, Students and Academics Based ML Framework Usage Distribution")
plt.show()


working_periods = multiple_data_copy.Q24.value_counts()
plt.figure(figsize=(12,9))
sns.barplot(y=working_periods.index, x=working_periods.values)
plt.ylabel('Time Periods')
plt.xlabel('Frequency')
plt.title('How Long Working on Analyzing Data Issues?')
plt.show()

data_spend = multiple_data_copy[(multiple_data_copy.Q24 == "1-2 years") | (multiple_data_copy.Q24 =="< 1 year")]
spend_times = data_spend.Q23.value_counts()
spend_list = ("50%-74%","25%-49%","1%-25%","75%-99%","100%","0%")

fig = {
  "data": [
    {
      "values": spend_times.values,
      "labels": spend_list,
      "domain": {"x": [0, 1]},
      "name": "Spending Time on Data Analyzing",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Spending Time on Data Analyzing",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Percantage of Time Spent",
                "x": 5,
                "y": 1
            },
        ]
    }
}
iplot(fig)

# New people anf their jobs
new_people_jobs = data_spend.Q6.value_counts()
plt.figure(figsize=(15,9))
sns.barplot(x=new_people_jobs.index, y=new_people_jobs.values)
plt.xticks(rotation= 45)
plt.xlabel('Title')
plt.ylabel('Frequency')
plt.title('Title of New People')
plt.show()
#50% - 100% Jobs
new_people_75_100 = data_spend.Q6[(data_spend.Q23 == "75% to 99% of my time") | (data_spend.Q23 == "100% of my time") | (data_spend.Q23 == "50% to 74% of my time")]
new_people_75_100_counts = new_people_75_100.value_counts()
plt.figure(figsize=(15,9))
sns.barplot(x=new_people_75_100_counts.index, y=new_people_75_100_counts.values)
plt.xticks(rotation= 45)
plt.xlabel('Title')
plt.ylabel('Count of People')
plt.title('Jobs of People Spending Time 50% - 100%')
plt.show()
# Major of Data analyst and Data scientist
data_of_scientist = multiple_data_copy[(multiple_data_copy.Q6 == "Data Scientist") | (multiple_data_copy.Q6 == "Data Analyst")]
majors_list = data_of_scientist.Q5.value_counts()
plt.figure(figsize=(15,9))
sns.barplot(x=majors_list.values, y=majors_list.index)
plt.xlabel('People Graduated')
plt.ylabel('Undergraduate Majors')
plt.title('Undergraduate Major of Data Scientist or Data Analyst')
plt.show()
country_list = data_of_scientist.Q3.unique()
count_list = []
for each in country_list:
    count_list.append(len(data_of_scientist[data_of_scientist.Q3 == each]))

dc = pd.Series(country_list)
dv = pd.Series(count_list)
data = [ dict(
        type = 'choropleth',
        autocolorscale = False,
        reversescale = True,
        colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        showscale = False,
        locations = dc,
        z = dv,
        locationmode = 'country names',
        text = dv,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            tickprefix = 'Count',
            title = 'Data Scientists/Analysts Count'),)   
    ]

layout = dict(
    height=800,
    title = 'Country Distribution of Data Scientists/Analysts',
    geo = dict(
        showframe = True,
        showocean = True,
        oceancolor = 'darkgray',
        projection = dict(
        type = 'mercator'
        )),
        )
fig = dict(data=data, layout=layout)
iplot(fig)
df = pd.concat([data_of_scientist["Q3"],data_of_scientist["Q9"]],axis=1)
count_USA = df.Q9[df.Q3 == "United States of America"].value_counts()
count_India = df.Q9[df.Q3 == "India"].value_counts()
count_China = df.Q9[df.Q3 == "China"].value_counts()

f,ax = plt.subplots(figsize = (9,12))
sns.barplot(x=count_USA.values,y=count_USA.index,color='blue',alpha = 0.5,label='United States Of America' )
sns.barplot(x=count_India.values,y=count_India.index,color='gray',alpha = 0.7,label='India')
sns.barplot(x=count_China.values,y=count_China.index,color='yellow',alpha = 0.6,label='China')


ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu
ax.set(xlabel='Number of People', ylabel='Salary')
ax.set_title(label = "Salary Of Data Scientists/Analysts(USA, India and China)")
plt.show()

online_platforms_dict = {"Udacity":data_of_scientist.Q36_Part_1.count(),"Coursera":data_of_scientist.Q36_Part_2.count(),
                        "edX":data_of_scientist.Q36_Part_3.count(),"DataCamp":data_of_scientist.Q36_Part_4.count(),
                        "DataQuest":data_of_scientist.Q36_Part_5.count(),"Kaggle Learn":data_of_scientist.Q36_Part_6.count(),
                        "Fast.AI ":data_of_scientist.Q36_Part_7.count(),"developers.google.com":data_of_scientist.Q36_Part_8.count(),
                        "Udemy":data_of_scientist.Q36_Part_9.count(),"TheSchool":data_of_scientist.Q36_Part_10.count(),
                        "OnlineUniversityCourses":data_of_scientist.Q36_Part_11.count(),"None":data_of_scientist.Q36_Part_12.count(),
                        "Other ":data_of_scientist.Q36_Part_13.count()}

online_platforms = pd.Series(online_platforms_dict)
online_platforms.sort_values(ascending=False, inplace=True)
fig = {
  "data": [
    {
      "y": online_platforms.values,
      "x": online_platforms.index,
      "marker":dict(color='darkblue',),
      "type": "bar"
    },],
  "layout": {
        "title":"Favorite Online Learning Platforms for Data Scientists/Anlaysts",
    }
}
iplot(fig)