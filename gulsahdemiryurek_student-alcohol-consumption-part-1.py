import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

mat_data= pd.read_csv("../input/student-mat.csv")
por_data= pd.read_csv("../input/student-por.csv")
student = pd.merge(por_data, mat_data, how='outer', on=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet","guardian","traveltime","studytime","famsup","activities","higher","romantic","famrel","freetime","goout","Dalc","Walc","health","schoolsup"])
student
student.info()
student.head(20)
student.describe()
student.corr()

f,ax=plt.subplots(figsize=(18,18))
sns.heatmap(student.corr(),annot=True,linewidth=0.5,fmt='.3f',ax=ax)
plt.show()
l=[1,2,3,4,5] #Alcohol consumption level
labels="1-Very Low","2-Low","3-Medium","4-High","5-Very High"
student.age.unique()
#Age
student.age.unique()
plt.figure(figsize=(10,5))
plt.hist(student.age,bins=7,color="mediumpurple",width=0.8,density=True)
plt.xlabel("Age")
plt.ylabel("Percentage")
plt.show()
age15=student[(student.age==15)]
age15.describe()
age15_workday=list(map(lambda l: list(age15.Dalc).count(l),l))
age15_weekend=list(map(lambda l: list(age15.Walc).count(l),l))
plt.style.use("bmh")
plt.figure(figsize=(10,5))
plt.plot(labels,age15_workday,color="dodgerblue",linestyle="--",marker="X", markersize=10,label="Workday")
plt.plot(labels,age15_weekend,color="darkmagenta",linestyle="--",marker="X", markersize=10,label="Weekend")
plt.title("Age 15 Student Alcohol Consumption")
plt.ylabel("Number of Students")
plt.legend()


plt.show()
age16=student[(student.age==16)]
age16.describe()
age16_workday=list(map(lambda l: list(age16.Dalc).count(l),l))
age16_weekend=list(map(lambda l: list(age16.Walc).count(l),l))
plt.style.use("bmh")
plt.figure(figsize=(10,5))
plt.plot(labels,age16_workday,color="sandybrown",linestyle=":",marker="s", markersize=10,label="Workday")
plt.plot(labels,age16_weekend,color="seagreen",linestyle=":",marker="s", markersize=10,label="Weekend")
plt.title("Age 16 Student Alcohol Consumption")
plt.ylabel("Number of Students")
plt.legend()
plt.show()
age17=student[(student.age==17)]
age17.describe()
age17_workday=list(map(lambda l: list(age17.Dalc).count(l),l))
age17_weekend=list(map(lambda l: list(age17.Walc).count(l),l))
plt.style.use("bmh")
plt.figure(figsize=(10,5))
plt.plot(labels,age17_workday,color="lawngreen",linestyle="dotted",marker="*", markersize=12,label="Workday")
plt.plot(labels,age17_weekend,color="rebeccapurple",linestyle="dotted",marker="*", markersize=12,label="Weekend")
plt.legend()
plt.ylabel("Number of Students")
plt.title("Age 17 Student Alcohol Consumption")
plt.show()
age18=student[(student.age==18)]
age18.describe()
age18_workday=list(map(lambda l: list(age18.Dalc).count(l),l))
age18_weekend=list(map(lambda l: list(age18.Walc).count(l),l))
plt.style.use("bmh")
plt.figure(figsize=(10,5))
plt.plot(labels,age18_workday,color="darkgoldenrod",linestyle="dotted",marker="P", markersize=10,label="Workday")
plt.plot(labels,age18_weekend,color="mediumblue",linestyle="dotted",marker="P", markersize=10,label="Weekend")
plt.legend()
plt.ylabel("Number of Students")
plt.title("Age 18 Student Alcohol Consumption")
plt.show()
age19=student[(student.age==19)]
age19.describe()
age19_workday=list(map(lambda l: list(age19.Dalc).count(l),l))
age19_weekend=list(map(lambda l: list(age19.Walc).count(l),l))
plt.style.use("bmh")
plt.figure(figsize=(10,5))
plt.plot(labels,age19_workday,color="dodgerblue",marker="H", markersize=10,label="Workday")
plt.plot(labels,age19_weekend,color="darkmagenta",marker="H", markersize=10,label="Weekend")
plt.legend()
plt.ylabel("Number of Students")
plt.title("Age 19 Student Alcohol Consumption")
plt.show()

age20=student[(student.age==20)]
age20.describe()
age20_workday=list(map(lambda l: list(age20.Dalc).count(l),l))
age20_weekend=list(map(lambda l: list(age20.Walc).count(l),l))
plt.style.use("bmh")
plt.figure(figsize=(15,5))
plt.plot(labels,age20_workday,color="darkorange",marker="d", markersize=10,label="Workday")
plt.plot(labels,age20_weekend,color="deepskyblue",marker="d", markersize=10,label="Weekend")
plt.legend()
plt.ylabel("Number of Students")
plt.title("Age 20 Student Alcohol Consumption")
plt.show()
age21=student[(student.age==21)]
age21.describe()
age21_workday=list(map(lambda l: list(age21.Dalc).count(l),l))
age21_weekend=list(map(lambda l: list(age21.Walc).count(l),l))
plt.figure(figsize=(10,5))
plt.plot(labels,age21_workday,color="chocolate",linestyle=":",linewidth=2,marker="X", markersize=10,label="Workday")
plt.plot(labels,age21_weekend,color="indigo",linestyle=":",linewidth=2,marker="X", markersize=10,label="Weekend")
plt.legend()
plt.style.use("bmh")
plt.ylabel("Number of Students")
plt.title("Age 21 Student Alcohol Consumption")
plt.show()
#sex
female=student[student.sex=="F"]
male= student[student.sex=="M"]
female.describe()
female_workday= list(map(lambda l: list(female.Dalc).count(l),l))
female_weekend= list(map(lambda l: list(female.Walc).count(l),l))

n = 5
fig, ax = plt.subplots(figsize=(10,5))
i = np.arange(n)    
w = 0.4   

plot1= plt.bar(i, female_workday, w, color="g")
plot2= plt.bar(i+w, female_weekend, w, color="r" )

plt.ylabel('Number of Student')
plt.title('Female Student Alcohol Consumption')
plt.xticks(i+w/2, labels)
plt.legend((plot1[0],plot2[0]),("Workday","Weekend"))
plt.tight_layout()
plt.style.use("bmh")
plt.show()

male.describe()
male_workday= list(map(lambda l: list(male.Dalc).count(l),l))
male_weekend= list(map(lambda l: list(male.Walc).count(l),l))

n = 5
fig, ax = plt.subplots(figsize=(10,5))
i = np.arange(n)    
w = 0.4   

plot1= plt.bar(i, male_workday, w, color="cadetblue")
plot2= plt.bar(i+w,male_weekend, w, color="b" )

plt.ylabel("Number of Student")
plt.title("Male Student Alcohol Consumption")
plt.xticks(i+w/2, labels)
plt.legend((plot1[0],plot2[0]),("Workday","Weekend"))
plt.tight_layout()
plt.style.use("bmh")
plt.show()

#school
student.school.unique()
GP=student[student.school=="GP"]
MS= student[student.school=="MS"]
GP.describe()
GP_workday= list(map(lambda l: list(GP.Dalc).count(l),l))

colors="mediumspringgreen","orchid","orangered","darkgoldenrod","aqua"
plt.figure(figsize=(8,8))
plt.pie(GP_workday,colors=colors,autopct='%1.1f%%', startangle=90)
plt.title("GP Students Workday Alcohol Consumptions")
plt.legend(labels)
plt.show()
GP_weekend= list(map(lambda l: list(GP.Walc).count(l),l))

plt.figure(figsize=(8,8))
plt.pie(GP_weekend,colors=colors,autopct='%1.1f%%', startangle=90)
plt.title("GP Students Weekend Alcohol Consumptions")
plt.legend(labels)
plt.show()
MS.describe()
MS_workday= list(map(lambda l: list(MS.Dalc).count(l),l))
colors2="sandybrown","springgreen","tomato","grey","pink"
plt.figure(figsize=(8,8))
plt.pie(MS_workday,colors=colors2,autopct='%1.1f%%', startangle=90)
plt.title("MS Students Workday Alcohol Consumptions")
plt.legend(labels)
plt.show()
MS_weekend= list(map(lambda l: list(MS.Walc).count(l),l))
plt.figure(figsize=(8,8))
plt.pie(MS_weekend,colors=colors2,autopct='%1.1f%%', startangle=90)
plt.title("MS Students Weekend Alcohol Consumptions")
plt.legend(labels)
plt.show()
student.address.unique()
urban=student[student.address=="U"]
rural= student[student.address=="R"]
urban.describe()
urban_workday=list(map(lambda l: list(urban.Dalc).count(l),l))
urban_weekend=list(map(lambda l: list(urban.Walc).count(l),l))

n = 5
fig, ax = plt.subplots(figsize=(10,5))
i = np.arange(n)   
w = 0.4   

plot1= plt.bar(i, urban_workday, w, color="peachpuff")
plot2= plt.bar(i+w, urban_weekend, w, color="skyblue" )

plt.ylabel('Number of Student')
plt.title('Urban Student Alcohol Consumption')
plt.xticks(i+w/2, labels)
plt.legend((plot1[0],plot2[0]),("Workday","Weekend"))
plt.tight_layout()
plt.grid()
plt.show()
rural.describe()
rural_workday=list(map(lambda l: list(rural.Dalc).count(l),l))
rural_weekend=list(map(lambda l: list(rural.Walc).count(l),l))

n = 5
fig, ax = plt.subplots(figsize=(10,5))
i = np.arange(n)    
w = 0.4 
p1= plt.bar(i, rural_workday, w, color="lightsalmon")
p2= plt.bar(i+w,rural_weekend, w, color="cornflowerblue" )

plt.ylabel('Number of Student')
plt.title('Rural Student Alcohol Consumption')
plt.xticks(i+w/2, labels)
plt.legend((p1[0],p2[0]),("Workday","Weekend"))
plt.tight_layout()
plt.grid()
plt.show()

d= {"Feature": ["All Students","Age 15","Age 16","Age 17","Age 18", "Age 19", "Age 20", "Age 21", "Female","Male","GP","MS","Urban","Rural"],
    "Count": [student.shape[0],age15.shape[0],age16.shape[0],age17.shape[0],age18.shape[0],age19.shape[0],age20.shape[0],age21.shape[0],female.shape[0],male.shape[0],GP.shape[0],MS.shape[0],urban.shape[0],rural.shape[0]],
    "Average Working Day Alcohol Consumption": [student.Dalc.mean(),age15.Dalc.mean(),age16.Dalc.mean(),age17.Dalc.mean(),age18.Dalc.mean(),age19.Dalc.mean(),age20.Dalc.mean(),age21.Dalc.mean(),female.Dalc.mean(),male.Dalc.mean(),GP.Dalc.mean(),MS.Dalc.mean(),urban.Dalc.mean(),rural.Dalc.mean()],
    "Average Weekend Alcohol Consumption": [student.Walc.mean(),age15.Walc.mean(),age16.Walc.mean(),age17.Walc.mean(),age18.Walc.mean(),age19.Walc.mean(),age20.Walc.mean(),age21.Walc.mean(),female.Walc.mean(),male.Walc.mean(),GP.Walc.mean(),MS.Walc.mean(),urban.Walc.mean(),rural.Walc.mean()]}

df=pd.DataFrame(d)


df
