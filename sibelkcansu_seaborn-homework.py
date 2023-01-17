# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
student_mat=pd.read_csv("../input/student-mat.csv",encoding="windows-1252")
student_por=pd.read_csv("../input/student-por.csv",encoding="windows-1252")
student_mat.head(10)
student_mat.info()
student_mat.corr()
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(student_mat.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
student_mat.school.value_counts()
# alcohol consumption by school 
school_list=list(student_mat.school.unique())
school_dav=[]
school_wav=[]
for i in school_list:
    x=student_mat[student_mat["school"]==i]
    workday_avg=sum(x.Dalc)/len(x)
    weekend_avg=sum(x.Walc)/len(x)
    school_dav.append(workday_avg)
    school_wav.append(weekend_avg)
    
data=pd.DataFrame({"school":school_list,"school_workday_avg":school_dav,"school_weekend_avg":school_wav})

new_index = (data['school_workday_avg'].sort_values(ascending=True)).index.values
sorted_data=data.reindex(new_index)
sorted_data.head()
# visualization1
plt.figure(figsize=(10,10))
sns.barplot(x=sorted_data["school"],y=sorted_data["school_workday_avg"])
plt.xticks(rotation= 360)
plt.xlabel("School")
plt.ylabel("Workday Alcohol Average")
plt.title("Workday Alcohol Consumption by School")
plt.show()
# visualization2
plt.figure(figsize=(10,10))
sns.barplot(x=sorted_data["school"],y=sorted_data["school_weekend_avg"])
plt.xticks(rotation= 360)
plt.xlabel("School")
plt.ylabel("Weekend Alcohol Average")
plt.title("Weekend Alcohol Consumption by School")
plt.show()
student_mat.head(10)
data2=student_mat.loc[:,["Dalc","Walc","goout"]]
data2.plot(subplots=True)
plt.show()
#Going Out vs Working Day Alcohol Consumption
plt.figure(figsize=(5,5))
sns.barplot(x=student_mat.goout,y=student_mat.Dalc,)
plt.xticks(rotation= 360)
plt.xlabel("goout")
plt.ylabel("Dalc")
plt.title("Going Out vs Working Day Alcohol Consumption")
plt.show()
#Going Out vs Weekend Alcohol Consumption
plt.figure(figsize=(5,5))
sns.barplot(x=student_mat.goout,y=student_mat.Walc,)
plt.xticks(rotation= 360)
plt.xlabel("goout")
plt.ylabel("Dalc")
plt.title("Going Out vs Weekend Alcohol Consumption")
plt.show()
goout_av=[]
for i in school_list:
    goout_av.append(sum(student_mat[student_mat["school"]==i].goout)/len(student_mat[student_mat["school"]==i].goout))
data3=pd.DataFrame({"school":school_list,"school_workday_avg":school_dav,"school_weekend_avg":school_wav,"goout_avg":goout_av})
new_index3 = (data3['goout_avg'].sort_values(ascending=True)).index.values 
sorted_data3 = data3.reindex(new_index3)
sorted_data3.head()

sorted_data.head()
#visualization

f,ax1=plt.subplots(figsize=(9,15))
sns.barplot(x=school_dav,y=school_list,color='green',alpha = 0.5,label='School Working Day Average')
sns.barplot(x=school_wav,y=school_list, color="blue",alpha=0.6,label="School Weekend Average")
sns.barplot(x=goout_av,y=school_list,color="yellow",alpha=0.7,label="School Going Out Average")

ax1.legend(loc="upper left",frameon=True)
ax1.set(xlabel='Percentage of Average', ylabel='Schools',title = "Percentage of Alcohol Consumption According to Schools ")
plt.show()
student_mat.head()
f,ax2=plt.subplots(figsize=(10,10))
sns.pointplot(x="school",y='goout',data=student_mat,color='lime',alpha=0.8)
sns.pointplot(x="school",y='Dalc',data=student_mat,color='red',alpha=0.8)
sns.pointplot(x="school",y='Walc',data=student_mat,color='blue',alpha=0.8)

plt.xlabel('Schools',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Going Out vs  Working Day Consumption',fontsize = 20,color='blue')
plt.grid()
plt.show()

sorted_data3["goout_avg"]=sorted_data3["goout_avg"]/max(sorted_data3["goout_avg"])
sorted_data3["school_workday_avg"]=sorted_data3["school_workday_avg"]/max(sorted_data3["school_workday_avg"])
sorted_data3["school_weekend_avg"]=sorted_data3["school_weekend_avg"]/max(sorted_data3["school_weekend_avg"])

f,ax3 = plt.subplots(figsize =(10,10))
sns.pointplot(x='school',y='goout_avg',data=sorted_data3,color='lime',alpha=0.8)
sns.pointplot(x='school',y='school_workday_avg',data=sorted_data3,color='red',alpha=0.8)
plt.text(0.9,0.9,'working day average',color='red',fontsize = 17,style = 'italic')
plt.text(0.9,0.92,'going out average',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('Schools',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Going Out Average vs  Working Day Consumption Average',fontsize = 20,color='blue')
plt.grid()
plt.show()

f,ax4 = plt.subplots(figsize =(10,10))
sns.pointplot(x='school',y='goout_avg',data=sorted_data3,color='lime',alpha=0.8)
sns.pointplot(x='school',y='school_weekend_avg',data=sorted_data3,color='red',alpha=0.8)
plt.text(0.9,0.95,'weekend average',color='red',fontsize = 17,style = 'italic')
plt.text(0.9,0.92,'going out average',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('Schools',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Going Out Average vs  Weekend Consumption Average',fontsize = 20,color='blue')
plt.grid()
plt.show()
data3.head()
#joint plot1
g=sns.jointplot(student_mat.Dalc,student_mat.Walc,kind="kde",size=7,ratio=3)
plt.savefig("graph.png")
plt.show()
#joint plot2
g=sns.jointplot(student_mat.Dalc,student_mat.goout,kind="kde",size=7,ratio=3)
plt.savefig("graph.png")
plt.show()
#joint plot3
g=sns.jointplot(student_mat.Walc,student_mat.goout,kind="kde",size=7,ratio=3)
plt.savefig("graph.png")
plt.show()
#joint plot4
g=sns.jointplot(student_mat.Walc,student_mat.goout,kind="reg",size=7,ratio=3)
plt.savefig("graph.png")
plt.show()
student_mat.head(10)
student_mat.age.unique()
student_mat.age.value_counts()
#pie chart1
student_mat.age.dropna(inplace=True)
labels=student_mat.age.value_counts().index
colors=["red","green","blue","purple","yellow","grey","brown","lime"]
explode = [0,0,0,0,0,0,0,0]
sizes=student_mat.age.value_counts().values

# visualization
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Students According to Ages',color = 'blue',fontsize = 15)


student_mat.Dalc.value_counts()
#pie chart2
student_mat.Dalc.dropna(inplace=True)
labels=student_mat.Dalc.value_counts().index
colors=["red","green","blue","purple","grey"]
explode = [0,0,0,0,0]
sizes=student_mat.Dalc.value_counts().values

# visualization
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Students According to WorkDay Alcohol Consumption',color = 'blue',fontsize = 15)
plt.hist(student_mat.Dalc,color="r",label="Workday Alcohol Consumption")
plt.xlabel("Dalc")
plt.legend(loc="upper right")
plt.show()
#pie chart3
student_mat.Walc.dropna(inplace=True)
labels=student_mat.Walc.value_counts().index
colors=["red","green","blue","purple","grey"]
explode = [0,0,0,0,0]
sizes=student_mat.Walc.value_counts().values

# visualization
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Students According to Weekend Alcohol Consumption',color = 'blue',fontsize = 15)
plt.hist(student_mat.Walc,color="r",label="Weekend Alcohol Consumption")
plt.xlabel("Walc")
plt.legend(loc="upper right")
plt.show()
# Alcohol Consumption Rate between Genders
gender_list=list(student_mat.sex.unique())
workd_av=[]
wd_av=[]
for i in gender_list:
    x=student_mat[student_mat["sex"]==i]
    davg=sum(x.Dalc)/len(x)
    wd_avg=sum(x.Walc)/len(x)
    workd_av.append(davg)
    wd_av.append(wd_avg)
    
data4=pd.DataFrame({"sex":gender_list,"workday_avg":workd_av,"weekend_avg":wd_av})
new_index4 = (data4['sex'].sort_values(ascending=True)).index.values
sorted_data4=data4.reindex(new_index4)
sorted_data4

plt.figure(figsize=(15,10)) #yeni bir figure ac ve boyutu da (15,10) olsun. 
sns.barplot(x=sorted_data4['sex'], y=sorted_data4['workday_avg'],palette = sns.cubehelix_palette(len(x)))
plt.xticks(rotation= 45) # seklin altına isimleri 45 derecelik acı ile yerleştir.
plt.xlabel("Gender")
plt.ylabel('Workday Average')
plt.title('Workday Alcohol Consumption Given Genders')
# lmplot 
# Show the results of a linear regression within each dataset
sns.lmplot(x="Dalc", y="Walc", data=student_mat)
plt.show()
# lmplot 
# Show the results of a linear regression within each dataset
sns.lmplot(x="Dalc", y="goout", data=student_mat)
plt.show()
data5=student_mat.copy()
data5["Dalc"]=data5["Dalc"]/max(data5["Dalc"])
data5["absences"]=data5["absences"]/max(data5["absences"])

sns.lmplot(x="Dalc",y="absences",data=data5)
plt.show()

# cubehelix plot
sns.kdeplot(student_mat.Dalc, student_mat.Walc, shade=True, cut=3) #shade=True: grafikteki sekillerin ici dolu olsun. cut=3: cıkan sekillerin buyuklugunu ayarlar.
plt.show()
# cubehelix plot
sns.kdeplot(student_mat.goout, student_mat.Walc, shade=True, cut=3) #shade=True: grafikteki sekillerin ici dolu olsun. cut=3: cıkan sekillerin buyuklugunu ayarlar.
plt.show()
# cubehelix plot
sns.kdeplot(student_mat.goout, student_mat.Dalc, shade=True, cut=3) #shade=True: grafikteki sekillerin ici dolu olsun. cut=3: cıkan sekillerin buyuklugunu ayarlar.
plt.show()
#volin plot1
# Show each distribution with both violins and points

# iki feature arasındaki iliskiye bakmak yerine her bir featuren icindeki degerlerin dagılımına bakar.

data6=student_mat.loc[:,["school","Dalc","Walc"]]
# Use cubehelix to get a custom sequential palette
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=data6, palette=pal, inner="points")
plt.show()
#volin plot2
data7=student_mat.loc[:,["school","Walc","goout"]]
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=data7, palette=pal, inner="points")
plt.show()
sorted_data.head()
sorted_data.corr()
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
#annot=True map uzernde sayısal degerleri goster.
#fmt= '.1f' degerleri virgulden sonra 1 basamak seklinde goster.
plt.show()
school_list=list(student_mat.school.unique())
school_dav=[]
school_wav=[]
school_goout=[]
for i in school_list:
    x=student_mat[student_mat["school"]==i]
    workday_avg=sum(x.Dalc)/len(x)
    weekend_avg=sum(x.Walc)/len(x)
    gooutavg=sum(x.goout)/len(x)
    school_dav.append(workday_avg)
    school_wav.append(weekend_avg)
    school_goout.append(gooutavg)
    
data8=pd.DataFrame({"school":school_list,"school_workday_avg":school_dav,"school_weekend_avg":school_wav,"goout_av":school_goout})

new_index8 = (data8['goout_av'].sort_values(ascending=True)).index.values
sorted_data8=data8.reindex(new_index8)
sorted_data8.head()
data8.corr()
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data8.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
#annot=True map uzernde sayısal degerleri goster.
#fmt= '.1f' degerleri virgulden sonra 1 basamak seklinde goster.
plt.show()
student_mat.head()
#box plot1
sns.boxplot(x="sex", y="age", hue="guardian", data=student_mat, palette="PRGn") #hue=class.
plt.show()
#swarm plot1
sns.swarmplot(x="sex", y="age",hue="guardian", data=student_mat)
plt.show()
#box plot2
sns.boxplot(x="sex", y="age", hue="famsize", data=student_mat, palette="PRGn") #hue=class.
plt.show()
#box plot3
sns.boxplot(x="sex", y="age", hue="Mjob", data=student_mat, palette="PRGn") #hue=class.
plt.show()
#swarm plot2
sns.swarmplot(x="sex", y="age",hue="Mjob", data=student_mat)
plt.show()
#box plot4
sns.boxplot(x="sex", y="age", hue="Fjob", data=student_mat, palette="PRGn") #hue=class.
plt.show()
#swarm plot
sns.swarmplot(x="sex", y="age",hue="Fjob", data=student_mat)
plt.show()
data2.head()
# pair plot
sns.pairplot(data2)
plt.show()
data5.head()
#count plot
# father job
# mother job
#sns.countplot(student_mat.Fjob)
sns.countplot(student_mat.Mjob)
plt.title("mother job",color = 'blue',fontsize=15)


#count plot
# gender
sns.countplot(student_mat.sex)
plt.title("gender",color = 'blue',fontsize=15)

sns.countplot(student_mat.G1)
plt.title("G1",color = 'blue',fontsize=15)
plt.ylabel("Number of Scores")
plt.xlabel("G1 score")
student_mat.head()
student_mat.absences.unique()
#number of absences
gret20=["high" if i >20 else "low" for i in student_mat.absences]
df=pd.DataFrame({"absence":gret20})
sns.countplot(x=df.absence)
plt.ylabel('Number of Students')
plt.title('Absences of students',color = 'blue',fontsize=15)

#study time
sns.countplot(student_mat.studytime)
plt.title("Study Time of Students",color = 'red',fontsize=15)
plt.ylabel("Number of Students")
plt.xlabel("Study Time")
plt.show()
student_mat.head(10)
student_mat.G3.value_counts()
#10 highest G3 score
score=student_mat.G3.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=score[:10].index,y=score[:10].values)
plt.xticks(rotation=45)
plt.ylabel("Scores")
plt.xlabel("Number of Scores")
plt.title('10 Highest G3 Score',color = 'blue',fontsize=15)
#10 highest G3 score
score=student_mat.G3.value_counts()
score.index = np.arange(len(score)) # indexleri büyük değerden küçük değere göre ayarlar.
plt.figure(figsize=(10,7))
sns.barplot(x=score[:10].index,y=score[:10].values)
plt.xticks(rotation=45)
plt.ylabel("Scores")
plt.xlabel("Indexes")
plt.title('10 Highest G3 Score',color = 'blue',fontsize=15)

# Having romantic relation for students
sns.countplot(student_mat.romantic)
plt.xlabel('Romantic Relationship')
plt.ylabel('Number of Students')
plt.title('Having Romantic Relationship or not',color = 'blue', fontsize = 15)
#health situtation of students
sns.countplot(student_mat.health)
plt.title("Health Situtation of Students",color = 'red',fontsize=15)
plt.ylabel("Number of Students")
plt.xlabel("Health Situtation")
plt.show()
#reason for choosing school
sns.countplot(student_mat.reason)
plt.title("Choosing Reason of School",color = 'red',fontsize=15)
plt.ylabel("Number of Students")
plt.xlabel("Reason")
plt.show()
