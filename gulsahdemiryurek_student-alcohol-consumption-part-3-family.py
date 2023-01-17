import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
mat_data= pd.read_csv("../input/student-mat.csv")
por_data= pd.read_csv("../input/student-por.csv")
student = pd.merge(por_data, mat_data, how='outer', on=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet","guardian","traveltime","studytime","famsup","activities","higher","romantic","famrel","freetime","goout","Dalc","Walc","health","schoolsup"])
l=[1,2,3,4,5] #Alcohol consumption level
labels="1-Very Low","2-Low","3-Medium","4-High","5-Very High"
student.famsize.value_counts(dropna=False)
assert student.famsize.notnull().all()
id_=np.arange(1,675,1)
student["id"]=id_
m=pd.melt(frame=student,id_vars="id",value_vars=["famsize","Dalc","Walc"])
pivot_table=m.pivot(index="id",columns="variable",values="value")
famsize_table=pd.concat([student.famsize,student.Dalc,student.Walc],axis=1)
print(m)
print(pivot_table)
print(famsize_table)
famsize_table.boxplot(column=["Dalc","Walc"],by="famsize",rot=45, fontsize=10,figsize=(15,5),grid=True)
plt.show()
GT3=famsize_table[(famsize_table.famsize=="GT3")]
GT3.describe()
GT3.boxplot()
plt.show()
l=[1,2,3,4,5] #Alcohol consumption levels from 1 - very low to 5 - very high
labels= "1-Very Low","2-Low","3-Medium","4-High","5-Very High"
plt.figure(figsize=(10,5))
plt.plot(labels,list(map(lambda l: list(GT3.Dalc).count(l),l)),color="palevioletred",linestyle="--",marker="o", markersize=10,label="Workday")
plt.plot(labels,list(map(lambda l: list(GT3.Walc).count(l),l)),color="mediumaquamarine",linestyle="--",marker="o", markersize=10,label="Weekend")
plt.title("GT3 Student Alcohol Consumption")
plt.grid()
plt.ylabel("Number of Students")
plt.legend()
plt.show()
LE3=famsize_table[(famsize_table.famsize=="LE3")]

LE3.describe()
plt.figure(figsize=(10,5))
plt.plot(labels,list(map(lambda l: list(LE3.Dalc).count(l),l)),color="darkslategrey",linestyle="--",marker="X", markersize=10,label="Workday")
plt.plot(labels,list(map(lambda l: list(LE3.Walc).count(l),l)),color="orangered",linestyle="--",marker="X", markersize=10,label="Weekend")
plt.title("LE3 Student Alcohol Consumption")
plt.grid()
plt.ylabel("Number of Students")
plt.legend()
plt.show()
student.Pstatus.value_counts(dropna=False)
pstatus=pd.concat([student.Pstatus,student.Dalc,student.Walc],axis=1)
pstatus.Pstatus=pstatus.Pstatus.astype("category")
pstatus.dtypes
t=pstatus[pstatus.Pstatus=="T"]
t.describe()
plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.plot(t.Dalc,"ro",color="chocolate")
plt.grid()
plt.legend()
plt.ylabel("Alcohol Consumption Level")
plt.title("Working Day Alcohol Consumption of Students who Parents Living Together")
plt.subplot(2,1,2)
plt.plot(t.Walc,"ro",color="blueviolet")
plt.grid()
plt.legend()
plt.ylabel("Alcohol Consumption Level")
plt.xlabel("Student Id")
plt.title("Weekend Alcohol Consumption of Students who Parents Living Together")
plt.show()
a=pstatus[(pstatus.Pstatus=="A")]
a.describe()
plt.figure(figsize=(20,10))
plt.subplot(2,1,1)
plt.plot(a.Dalc,"bs",color="red")
plt.legend()
plt.grid()
plt.ylabel("Alcohol Consumption Level")
plt.xlabel("Student Id")
plt.title("Working Day Alcohol Consumption of Students who Parents Living Apart")
plt.subplot(2,1,2)
plt.plot(a.Walc,"g^",color="g")
plt.ylabel("Alcohol Consumption Level")
plt.xlabel("Student Id")
plt.legend()
plt.title("Weekend Alcohol Consumption of Students who Parents Living Apart")
plt.grid()
plt.show()
parent_education=pd.concat([student.Medu,student.Fedu,student.Dalc,student.Walc],axis=1)
parent_education.describe()
le=[0,1,2,3,4]
labelse=["0 - none", "1 - primary education (4th grade)", "2 – 5th to 9th grade", "3 – secondary education", "4 – higher education"]
mother= list(map(lambda le: list(student.Medu).count(le),le))
father= list(map(lambda le: list(student.Fedu).count(le),le))

n = 5
fig, ax = plt.subplots(figsize=(15,5))
i = np.arange(n)    
w = 0.4   

plot1= plt.bar(i, mother, w, color="blue")
plot2= plt.bar(i+w, father, w, color="skyblue" )

plt.ylabel('Number of Student')
plt.title("Parent's Educational Level")
plt.xticks(i+w/2, labelse)
plt.legend((plot1[0],plot2[0]),("Mother","Father"))
plt.tight_layout()
plt.show()
student.boxplot(column=["Dalc","Walc"],by="Medu",rot=100, fontsize=10,figsize=(15,5),grid=False)
plt.show()
student.boxplot(column=["Dalc","Walc"],by="Fedu",rot=100, fontsize=10,figsize=(15,5),grid=False)
plt.show()
student.Mjob.unique()
student.Mjob.value_counts(dropna=False)
jobs="other","services","at home","teacher","health"
colors="mediumspringgreen","orchid","orangered","darkgoldenrod","aqua"
plt.figure(figsize=(8,8))
plt.pie(student.Mjob.value_counts(),colors=colors,autopct='%1.1f%%', startangle=90)
plt.title("Mother's job")
plt.legend(jobs)
plt.show()
student.boxplot(column=["Dalc","Walc"],by="Mjob",rot=45, fontsize=10,figsize=(15,5),grid=False)
plt.show()
Mat_home=student[(student.Mjob=="at_home")]
Mservice=student[(student.Mjob=="services")]
Mteacher=student[(student.Mjob=="teacher")]
Mhealth=student[(student.Mjob=="health")]
Mother=student[(student.Mjob=="other")]
DalcMjob= {"Workday\Mjob": labels,
    "other":list(map(lambda l :list(Mother.Dalc).count(l)/Mother.Dalc.count()*100,l))  ,
    "services":list(map(lambda l :list(Mservice.Dalc).count(l)/Mservice.Dalc.count()*100,l)) ,
    "at_home": list(map(lambda l :list(Mat_home.Dalc).count(l)/Mat_home.Dalc.count()*100,l)) ,
    "teacher":list(map(lambda l :list(Mteacher.Dalc).count(l)/Mteacher.Dalc.count()*100,l)) ,
    "health":list(map(lambda l :list(Mhealth.Dalc).count(l)/Mhealth.Dalc.count()*100,l))}
DalcMjob=pd.DataFrame(DalcMjob)
DalcMjob
colorset= "m","firebrick","lightcoral","gold","lightsteelblue"
DalcMjob.plot(kind='bar',x= "Workday\Mjob",grid=True, title="percentage of alcohol consumption on working days according to the mother's jobs",figsize=(15,5),
        sharex=True, sharey=False, legend=True,color=colorset)
plt.ylabel("Percentage")
plt.xlabel("Alcohol Consumption Level")
plt.show()
WalcMjob= {"Weekend\Mjob": labels,
    "other":list(map(lambda l :list(Mother.Walc).count(l)/Mother.Walc.count()*100,l))  ,
    "services":list(map(lambda l :list(Mservice.Walc).count(l)/Mservice.Walc.count()*100,l)) ,
    "at_home": list(map(lambda l :list(Mat_home.Walc).count(l)/Mat_home.Walc.count()*100,l)) ,
    "teacher":list(map(lambda l :list(Mteacher.Walc).count(l)/Mteacher.Walc.count()*100,l)) ,
    "health":list(map(lambda l :list(Mhealth.Walc).count(l)/Mhealth.Walc.count()*100,l))}
WalcMjob=pd.DataFrame(WalcMjob)
WalcMjob
WalcMjob.plot(kind='bar',x= "Weekend\Mjob",grid=True, title="percentage of alcohol consumption on weekends according to the mother's jobs",figsize=(15,5),
        sharex=True, sharey=False, legend=True,color=colorset)
plt.ylabel("Percentage")
plt.xlabel("Alcohol Consumption Level")
plt.show()
student.Fjob.value_counts(dropna=False)
jobs=("other","services","at home","teacher","health")
colors="mediumspringgreen","orchid","orangered","darkgoldenrod","aqua"
plt.figure(figsize=(8,8))
plt.pie(student.Fjob.value_counts(),colors=colors,autopct='%1.1f%%', startangle=90)
plt.title("Father's job")
plt.legend(jobs)
plt.show()
student.boxplot(column=["Dalc","Walc"],by="Fjob",rot=45, fontsize=10,figsize=(15,5),grid=False)
plt.show()
Fat_home=student[(student.Fjob=="at_home")]
Fservice=student[(student.Fjob=="services")]
Fteacher=student[(student.Fjob=="teacher")]
Fhealth=student[(student.Fjob=="health")]
Fother=student[(student.Fjob=="other")]
DalcFjob= {"Workday\Fjob": labels,
    "other":list(map(lambda l :list(Fother.Dalc).count(l)/Fother.Dalc.count()*100,l))  ,
    "services":list(map(lambda l :list(Fservice.Dalc).count(l)/Fservice.Dalc.count()*100,l)) ,
    "at_home": list(map(lambda l :list(Fat_home.Dalc).count(l)/Fat_home.Dalc.count()*100,l)) ,
    "teacher":list(map(lambda l :list(Fteacher.Dalc).count(l)/Fteacher.Dalc.count()*100,l)) ,
    "health":list(map(lambda l :list(Fhealth.Dalc).count(l)/Fhealth.Dalc.count()*100,l))}
DalcFjob=pd.DataFrame(DalcFjob)
DalcFjob
colorset2="salmon","lightseagreen","olivedrab","orange","darkslategrey"
DalcFjob.plot(kind='bar',x= "Workday\Fjob",grid=True, title="Percentage of Alcohol Consumption on Working Days According to Father's Jobs",figsize=(15,5),
        sharex=True, sharey=False, legend=True,color=colorset2)
plt.ylabel("Percentage")
plt.xlabel("Alcohol Consumption Level")
plt.show()
WalcFjob= {"Weekend\Fjob": labels,
    "other":list(map(lambda l :list(Fother.Walc).count(l)/Fother.Walc.count()*100,l))  ,
    "services":list(map(lambda l :list(Fservice.Walc).count(l)/Fservice.Walc.count()*100,l)) ,
    "at_home": list(map(lambda l :list(Fat_home.Walc).count(l)/Fat_home.Walc.count()*100,l)) ,
    "teacher":list(map(lambda l :list(Fteacher.Walc).count(l)/Fteacher.Walc.count()*100,l)) ,
    "health":list(map(lambda l :list(Fhealth.Walc).count(l)/Fhealth.Walc.count()*100,l))}
WalcFjob=pd.DataFrame(WalcFjob)
WalcFjob

WalcFjob.plot(kind='bar',x= "Weekend\Fjob",grid=True, title="Percentage of Alcohol Consumption on Weekends According to Father's Jobs",figsize=(15,5),
        sharex=True, sharey=False,color=colorset2, legend=True)
plt.ylabel("Percentage")
plt.xlabel("Alcohol Consumption Level")
plt.show()
student.guardian.value_counts(dropna=False)
student.boxplot(column=["Dalc","Walc"],by="guardian",rot=45, fontsize=10,figsize=(15,5),grid=True)
plt.show()
student.famrel.describe()
student.boxplot(column=["Dalc","Walc"],by="famrel",rot=45, fontsize=10,figsize=(15,5),grid=False)
plt.show()
d= {"Feature": ["All Students","Famsize GT3","Famsize LE3","Together","Apart","Mother None Ed.","Mother Primary Ed.", "Mother 5th to 9th Grade Ed.", "Mother Secondary Ed.", "Mother Higher Ed.","Father None Ed.","Father Primary Ed.", "Father 5th to 9th Grade Ed.", "Father Secondary Ed.", "Father Higher Ed.","Mother Other","Mother Service", "Mother At Home","Mother Teacher","Mother Health","Father Other","Father Service", "Father At Home","Father Teacher","Father Health","Mother Guardian","Father Guardian","Other Guardian","Family Relationship 1","Family Relationship 2","Family Relationship 3","Family Relationship 4","Family Relationship 5"],
    "Count": [student.shape[0],GT3.shape[0],LE3.shape[0],t.shape[0],a.shape[0],student[(student.Medu==0)].shape[0],student[(student.Medu==1)].shape[0],student[(student.Medu==2)].shape[0],student[(student.Medu==3)].shape[0],student[(student.Medu==4)].shape[0],student[(student.Fedu==0)].shape[0],student[(student.Fedu==1)].shape[0],student[(student.Fedu==2)].shape[0],student[(student.Fedu==3)].shape[0],student[(student.Fedu==4)].shape[0],Mother.shape[0],Mservice.shape[0],Mat_home.shape[0],Mteacher.shape[0],Mhealth.shape[0],Fother.shape[0],Fservice.shape[0],Fat_home.shape[0],Fteacher.shape[0],Fhealth.shape[0],student[(student.guardian=="mother")].shape[0],student[(student.guardian=="father")].shape[0],student[(student.guardian=="other")].shape[0],student[(student.famrel==1)].shape[0],student[(student.famrel==2)].shape[0],student[(student.famrel==3)].shape[0],student[(student.famrel==4)].shape[0],student[(student.famrel==5)].shape[0]],
    "Average Working Day Alcohol Consumption": [student.Dalc.mean(),GT3.Dalc.mean(),LE3.Dalc.mean(),t.Dalc.mean(),a.Dalc.mean(),student[(student.Medu==0)].Dalc.mean(),student[(student.Medu==1)].Dalc.mean(),student[(student.Medu==2)].Dalc.mean(),student[(student.Medu==3)].Dalc.mean(),student[(student.Medu==4)].Dalc.mean(),student[(student.Fedu==0)].Dalc.mean(),student[(student.Fedu==1)].Dalc.mean(),student[(student.Fedu==2)].Dalc.mean(),student[(student.Fedu==3)].Dalc.mean(),student[(student.Fedu==4)].Dalc.mean(),Mother.Dalc.mean(),Mservice.Dalc.mean(),Mat_home.Dalc.mean(),Mteacher.Dalc.mean(),Mhealth.Dalc.mean(),Fother.Dalc.mean(),Fservice.Dalc.mean(),Fat_home.Dalc.mean(),Fteacher.Dalc.mean(),Fhealth.Dalc.mean(),student[(student.guardian=="mother")].Dalc.mean(),student[(student.guardian=="father")].Dalc.mean(),student[(student.guardian=="other")].Dalc.mean(),student[(student.famrel==1)].Dalc.mean(),student[(student.famrel==2)].Dalc.mean(),student[(student.famrel==3)].Dalc.mean(),student[(student.famrel==4)].Dalc.mean(),student[(student.famrel==5)].Dalc.mean()],
    "Median of Working Day Alcohol Consumption": [student.Dalc.median(),GT3.Dalc.median(),LE3.Dalc.median(),t.Dalc.median(),a.Dalc.median(),student[(student.Medu==0)].Dalc.median(),student[(student.Medu==1)].Dalc.median(),student[(student.Medu==2)].Dalc.median(),student[(student.Medu==3)].Dalc.median(),student[(student.Medu==4)].Dalc.median(),student[(student.Fedu==0)].Dalc.median(),student[(student.Fedu==1)].Dalc.median(),student[(student.Fedu==2)].Dalc.median(),student[(student.Fedu==3)].Dalc.median(),student[(student.Fedu==4)].Dalc.median(),Mother.Dalc.median(),Mservice.Dalc.median(),Mat_home.Dalc.median(),Mteacher.Dalc.median(),Mhealth.Dalc.median(),Fother.Dalc.median(),Fservice.Dalc.median(),Fat_home.Dalc.median(),Fteacher.Dalc.median(),Fhealth.Dalc.median(),student[(student.guardian=="mother")].Dalc.median(),student[(student.guardian=="father")].Dalc.median(),student[(student.guardian=="other")].Dalc.median(),student[(student.famrel==1)].Dalc.median(),student[(student.famrel==2)].Dalc.median(),student[(student.famrel==3)].Dalc.median(),student[(student.famrel==4)].Dalc.median(),student[(student.famrel==5)].Dalc.median()],
    "Average Weekend Alcohol Consumption": [student.Walc.mean(),GT3.Walc.mean(),LE3.Walc.mean(),t.Walc.mean(),a.Walc.mean(),student[(student.Medu==0)].Walc.mean(),student[(student.Medu==1)].Walc.mean(),student[(student.Medu==2)].Walc.mean(),student[(student.Medu==3)].Walc.mean(),student[(student.Medu==4)].Walc.mean(),student[(student.Fedu==0)].Walc.mean(),student[(student.Fedu==1)].Walc.mean(),student[(student.Fedu==2)].Walc.mean(),student[(student.Fedu==3)].Walc.mean(),student[(student.Fedu==4)].Walc.mean(),Mother.Walc.mean(),Mservice.Walc.mean(),Mat_home.Walc.mean(),Mteacher.Walc.mean(),Mhealth.Walc.mean(),Fother.Walc.mean(),Fservice.Walc.mean(),Fat_home.Walc.mean(),Fteacher.Walc.mean(),Fhealth.Walc.mean(),student[(student.guardian=="mother")].Walc.mean(),student[(student.guardian=="father")].Walc.mean(),student[(student.guardian=="other")].Walc.mean(),student[(student.famrel==1)].Walc.mean(),student[(student.famrel==2)].Walc.mean(),student[(student.famrel==3)].Walc.mean(),student[(student.famrel==4)].Walc.mean(),student[(student.famrel==5)].Walc.mean()],
    "Median of Weekend Alcohol Consumption": [student.Walc.median(),GT3.Walc.median(),LE3.Walc.median(),t.Walc.median(),a.Walc.median(),student[(student.Medu==0)].Walc.median(),student[(student.Medu==1)].Walc.median(),student[(student.Medu==2)].Walc.median(),student[(student.Medu==3)].Walc.median(),student[(student.Medu==4)].Walc.median(),student[(student.Fedu==0)].Walc.median(),student[(student.Fedu==1)].Walc.median(),student[(student.Fedu==2)].Walc.median(),student[(student.Fedu==3)].Walc.median(),student[(student.Fedu==4)].Walc.median(),Mother.Walc.median(),Mservice.Walc.median(),Mat_home.Walc.median(),Mteacher.Walc.median(),Mhealth.Walc.median(),Fother.Walc.median(),Fservice.Walc.median(),Fat_home.Walc.median(),Fteacher.Walc.median(),Fhealth.Walc.median(),student[(student.guardian=="mother")].Walc.median(),student[(student.guardian=="father")].Walc.median(),student[(student.guardian=="other")].Walc.median(),student[(student.famrel==1)].Walc.median(),student[(student.famrel==2)].Walc.median(),student[(student.famrel==3)].Walc.median(),student[(student.famrel==4)].Walc.median(),student[(student.famrel==5)].Walc.median()]}

df=pd.DataFrame(d)
df
df.plot(kind="line",y="Average Working Day Alcohol Consumption",grid=True,figsize=(15,5),marker="o")
plt.axhline(y=1.5,color="r",label="All student Average")
plt.plot(df["Median of Working Day Alcohol Consumption"],color="green",linestyle="--",marker="o")
plt.title("Working Day Alcohol Consumtion")
plt.ylabel("Alcohol Consumption Level")
plt.xlabel("Features")
plt.legend()
plt.show()

df.plot(kind="line",y="Average Weekend Alcohol Consumption",grid=True,figsize=(15,5),marker="o")
plt.axhline(y=2.278932,color="red",label="All student Average")
plt.plot(df["Median of Weekend Alcohol Consumption"],color="orchid",linestyle="--",marker="o")
plt.title("Weekends Alcohol Consumtion")
plt.ylabel("Alcohol Consumption Level")
plt.xlabel("Features")
plt.legend()
plt.show()