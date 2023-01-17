# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
mat_data= pd.read_csv("../input/student-mat.csv")
por_data= pd.read_csv("../input/student-por.csv")
student = pd.merge(por_data, mat_data, how='outer', on=["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet","guardian","traveltime","studytime","famsup","activities","higher","romantic","famrel","freetime","goout","Dalc","Walc","health","schoolsup"])
l=[1,2,3,4,5] #Alcohol consumption level
labels="1-Very Low","2-Low","3-Medium","4-High","5-Very High"
colorset="darkorange","chartreuse","seagreen","slateblue","firebrick"
def barplot(value1,value2,yLabel,Title,Legend1,Legend2):  #gives 2 barchart
    """
    parameter: value1,value2,ylabel,Title,Legend1,Legend2
    return 2 barchart 
    """
    
    n = 5
    fig, ax = plt.subplots(figsize=(10,5))

    i = np.arange(n)    # the x locations for the groups
    w = 0.4   # the width of the bars: can also be len(x) sequence
    
    plot1= plt.bar(i,value1, w, color="teal")
    plot2= plt.bar(i+w,value2, w, color="darkmagenta" )

    plt.ylabel(yLabel)
    plt.title(Title)
    plt.xticks(i, labels)
    plt.legend((plot1[0],plot2[0]),(Legend1,Legend2))
    plt.tight_layout()
    plt.show()  
def piechart(value,colorset,Title,labels):    
    """
    parameter: value, colorset, Title
    return piechart
    """
    plt.figure(figsize=(8,8))
    plt.pie(value,colors=colorset,autopct='%1.1f%%', startangle=90)
    plt.legend(labels)
    plt.title(Title)
def linechart(value1,label1,value2,label2,Ylabel,Title):
    """
    parameter:value1,label1,value2,label2,Ylabel,Title
    label 1:label of value1
    label 2: label of value2
    return 2 linechart
    """
    plt.figure(figsize=(10,5))
    plt.plot(labels,value1,color="blue",marker="o", linestyle="dashed", markersize=10,label=label1)
    plt.plot(labels,value2,color="red",marker="o", linestyle="dashed", markersize=10,label=label2)
    plt.ylabel(Ylabel)
    plt.title(Title)
    plt.grid()
    plt.legend()
   
student.activities.unique()
def a(alc,answer):  #alc="Walc" or "Dalc", answer= "yes" or "no"
    if alc=="Dalc":
        y= list(map(lambda l :list(student[(student.activities==answer)].Dalc).count(l),l)) 
        percent=[i/sum(y)*100 for i in y]
        #print("1-Vey Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        #print("1-Very Low:",percent[0],"2-Low:",percent[1],"3-Medium",percent[2] ,"4-High",percent[3], "5-Very High:",percent[4])
        return percent
    elif alc=="Walc":
        y= list(map(lambda l :list(student[(student.activities==answer)].Walc).count(l),l)) 
        percent=[i/sum(y)*100 for i in y]
        #print("1-Very Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        #print("1-Very Low:",percent[0],"2-Low:",percent[1],"3-Medium",percent[2] ,"4-High",percent[3], "5-Very High:",percent[4])
        return percent
a("Dalc","yes")
piechart(a("Dalc","yes"),colorset,"Percentage of alcohol consumed by students participating in activities on workdays",labels)
piechart(a("Dalc","no"),colorset,"Percentage of alcohol consumed by students not participating in activities on workdays",labels)
linechart(a("Walc","yes"),"yes",a("Walc","no"),"no","Percentage","the percentage of alcohol consumption on the weekends according to the activities of the students")

student.internet.unique()
def i(alc,answer): #alc="Dalc" or Walc , answer "yes" or "no"
    if alc=="Dalc":
        y= list(map(lambda l :list(student[(student.internet==answer)].Dalc).count(l),l)) 
        #print("1-Vey Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        return y
    elif alc=="Walc":
        y= list(map(lambda l :list(student[(student.internet==answer)].Walc).count(l),l)) 
        #print("1-Very Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        return y
print(i("Dalc","yes"))
print(sum(i("Dalc","yes")))
print(i("Dalc","no"))
print(sum(i("Dalc","no")))
barplot(i("Dalc","yes"),i("Dalc","no"),"Number of Students","Alcohol Consumption Levels on Working days According to Internet Connection","yes","no")
barplot(i("Dalc","yes"),i("Walc","yes"),"Number of Students","Students who have internet connection Alcohol consumption levels on workdays or on weekends ","working day","weekend")
barplot(i("Dalc","no"),i("Walc","no"),"Number of Students","Students who have not internet connection Alcohol consumption levels on workdays or on weekends ","working day","weekend")
def r(alc,answer):   #alc="Dalc" or Walc , answer "yes" or "no"
    if alc=="Dalc":
        y= list(map(lambda l :list(student[(student.romantic==answer)].Dalc).count(l),l)) 
        print("1-Very Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        print("sum of students:", sum(y))
        percent=[i/sum(y) for i in y]
        return percent
    elif alc=="Walc":
        y= list(map(lambda l :list(student[(student.romantic==answer)].Walc).count(l),l))
        print("1-Very Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        print("sum of students:",sum(y))
        percent=[i/sum(y) for i in y]
        return percent
print(r("Dalc","yes"))
print(r("Dalc","no"))
barplot(r("Dalc","yes"),r("Dalc","no"),"Percentage","Percentage of alcohol consumed by students on working days according to have relationship" ,"yes","no")
barplot(r("Walc","yes"),r("Walc","no"),"Percentage","Percentage of alcohol consumed by students on weekends according to have relationship" ,"yes","no")
student.freetime.describe()
plt.hist(student.freetime,bins=5)
def f(alc,l1): #alc ="yes" or "no" , l1= 1,2,3,4,5 (level of freetime)(1- very low-5 very high)
    if alc=="Dalc":
        y= list(map(lambda l :list(student[(student.freetime==l1)].Dalc).count(l),l)) 
        percent=[i/sum(y)*100 for i in y]
        #print("1-Vey Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        #print("1-Very Low:",percent[0],"2-Low:",percent[1],"3-Medium",percent[2] ,"4-High",percent[3], "5-Very High:",percent[4])
        return percent
    elif alc=="Walc":
        y= list(map(lambda l :list(student[(student.freetime==l1)].Walc).count(l),l)) 
        percent=[i/sum(y)*100 for i in y]
        #print("1-Very Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        #print("1-Very Low:",percent[0],"2-Low:",percent[1],"3-Medium",percent[2] ,"4-High",percent[3], "5-Very High:",percent[4])
        return percent
DalcFreetime= {"Workday\Free Time": ["1-Very Low","2-Low","3-Medium","4-High","5-Very High"],
    "1-Very Low": f("Dalc",1),"2-Low": f("Dalc",2), "3-Medium": f("Dalc",3),"4-High":f("Dalc",4),"5-Very High": f("Dalc",5)}
dfDalcFreetime=pd.DataFrame(DalcFreetime)
dfDalcFreetime

dfDalcFreetime.plot(kind='bar',x="Workday\Free Time" ,grid=True, title="percentage of alcohol consumption on working days according to the students' free time levels",figsize=(15,5),
        sharex=True, sharey=False, legend=True)
plt.ylabel("Percentage")
plt.xlabel("Alcohol Consumption Level")
plt.show()

WalcFreetime= {"Weekend\Free Time": ["1-Very Low","2-Low","3-Medium","4-High","5-Very High"],
    "1-Very Low": f("Walc",1),"2-Low": f("Walc",2), "3-Medium": f("Walc",3),"4-High":f("Walc",4),"5-Very High": f("Walc",5)}
dfWalcFreetime=pd.DataFrame(WalcFreetime)
dfWalcFreetime
dfWalcFreetime.plot(kind='bar',x="Weekend\Free Time" ,grid=True, title="percentage of alcohol consumption on weekends according to the students' free time levels",figsize=(15,5), legend=True)
plt.xlabel("Alcohol Consumption Level")
plt.ylabel("Percentage")
plt.show()
student.goout.describe()
plt.hist(student.goout,bins=5)
def g(alc,l1):   #alc="Dalc" or "Walc", l1= 1,2,3,4,5 (level of going out)(1- very low-5 very high)
    if alc=="Dalc":
        y= list(map(lambda l :list(student[(student.goout==l1)].Dalc).count(l),l)) 
        percent=[i/sum(y)*100 for i in y]
        #print("1-Vey Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        #print("1-Very Low:",percent[0],"2-Low:",percent[1],"3-Medium",percent[2] ,"4-High",percent[3], "5-Very High:",percent[4])
        return percent
    elif alc=="Walc":
        y= list(map(lambda l :list(student[(student.goout==l1)].Walc).count(l),l)) 
        percent=[i/sum(y)*100 for i in y]
        #print("1-Very Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        #print("1-Very Low:",percent[0],"2-Low:",percent[1],"3-Medium",percent[2] ,"4-High",percent[3], "5-Very High:",percent[4])
        return percent
DalcGoout= {"Workday\Go out": ["1-Very Low","2-Low","3-Medium","4-High","5-Very High"],
    "1-Very Low": g("Dalc",1),"2-Low": g("Dalc",2), "3-Medium": g("Dalc",3),"4-High":g("Dalc",4),"5-Very High": g("Dalc",5)}
DalcGoout=pd.DataFrame(DalcGoout)
DalcGoout
DalcGoout.plot(kind='bar',x="Workday\Go out" ,grid=True, title="percentage of alcohol consumption on working days according to the students' go out levels",figsize=(15,5), legend=True)
plt.xlabel("Alcohol Consumption Level")
plt.ylabel("Percentage")
plt.show()
WalcGoout= {"Weekend\Go out": ["1-Very Low","2-Low","3-Medium","4-High","5-Very High"],
    "1-Very Low": g("Walc",1),"2-Low": g("Walc",2), "3-Medium": g("Walc",3),"4-High":g("Walc",4),"5-Very High": g("Walc",5)}
WalcGoout=pd.DataFrame(WalcGoout)
WalcGoout
WalcGoout.plot(kind='bar',x="Weekend\Go out" ,grid=True, title="percentage of alcohol consumption on weekends according to the students' go out levels",figsize=(15,5), legend=True)
plt.xlabel("Alcohol Consumption Level")
plt.ylabel("Percentage")
plt.show()
student.health.describe()
plt.hist(student.health,bins=5)
def h(alc,l1):   #alc="Dalc" or "Walc", l1= 1,2,3,4,5 (level of health)(1- very low-5 very high)
    if alc=="Dalc":
        y= list(map(lambda l :list(student[(student.health==l1)].Dalc).count(l),l)) 
        percent=[i/sum(y)*100 for i in y]
        #print("1-Vey Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        #print("1-Very Low:",percent[0],"2-Low:",percent[1],"3-Medium",percent[2] ,"4-High",percent[3], "5-Very High:",percent[4])
        return percent
    elif alc=="Walc":
        y= list(map(lambda l :list(student[(student.health==l1)].Walc).count(l),l)) 
        percent=[i/sum(y)*100 for i in y]
        #print("1-Very Low:",y[0],"2-Low:",y[1],"3-Medium",y[2] ,"4-High",y[3], "5-Very High:",y[4])
        #print("1-Very Low:",percent[0],"2-Low:",percent[1],"3-Medium",percent[2] ,"4-High",percent[3], "5-Very High:",percent[4])
        return percent
DalcHealth= {"Workday\Health": ["1-Very Low","2-Low","3-Medium","4-High","5-Very High"],
    "1-Very Low": h("Dalc",1),"2-Low": h("Dalc",2), "3-Medium": h("Dalc",3),"4-High":h("Dalc",4),"5-Very High": h("Dalc",5)}
DalcHealth=pd.DataFrame(DalcHealth)
DalcHealth
DalcHealth.plot(kind='bar',x="Workday\Health" ,grid=True, title="percentage of alcohol consumption on working days according to the students' health levels",figsize=(15,5), legend=True)
plt.xlabel("Alcohol Consumption Level")
plt.ylabel("Percentage")
plt.show()
WalcHealth= {"Weekend\Health": ["1-Very Low","2-Low","3-Medium","4-High","5-Very High"],
    "1-Very Low": h("Walc",1),"2-Low": h("Walc",2), "3-Medium": h("Walc",3),"4-High":h("Walc",4),"5-Very High": h("Walc",5)}
WalcHealth=pd.DataFrame(WalcHealth)
WalcHealth

WalcHealth.plot(kind='bar',x="Weekend\Health" ,grid=True, title="percentage of alcohol consumption on working days according to the students' health levels",figsize=(15,5), legend=True)
plt.xlabel("Alcohol Consumption Level")
plt.ylabel("Percentage")
plt.show()