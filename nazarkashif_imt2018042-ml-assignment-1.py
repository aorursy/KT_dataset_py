import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
alcdata = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/alcoholism/student-mat.csv")
fifadata = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/fifa18/data.csv")
accidata1 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2005_to_2007.csv")
accidata2 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2009_to_2011.csv")
accidata3 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2012_to_2014.csv")
G_avg=(alcdata.G1+alcdata.G2+alcdata.G3)/3
alcdata["G_avg"]=G_avg
alcdata.drop(["G1","G2","G3"],axis=1,inplace=True)

plt.figure(figsize=(16,10))
sns.color_palette("ocean")
sns.heatmap(alcdata.corr(),annot=True,cmap="coolwarm",vmin=-1)
alcdata.columns
sns.boxplot(x=alcdata["G_avg"],y=alcdata["sex"])
sns.boxplot(x=alcdata["G_avg"],y=alcdata["romantic"],hue=alcdata["romantic"])
sns.barplot(x=alcdata["traveltime"],y=alcdata["G_avg"])
a=alcdata.select_dtypes(include=["object"])
a=a.columns
alcdata.isnull().sum()


p=alcdata[a].nunique().apply(lambda x: x==2)
my_label=p[p].index
my_label=alcdata[my_label]
my_label=my_label.apply(LabelEncoder().fit_transform)
k=alcdata[a].nunique().apply(lambda x: x!=2)
my_onehot=k[k].index
my_onehot=alcdata[my_onehot]
my_onehot
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

dumm_bin = pd.get_dummies(my_onehot)
dumm_bin
new_alcdata=alcdata.drop(p.index,axis=1)
new_alcdata=pd.concat([new_alcdata,dumm_bin,my_label],axis=1)

new_alcdata
# my_onehot
# alcdata

new_alcdata.info()
sns.barplot(x=alcdata["Pstatus"],y=alcdata["G_avg"])
sns.barplot(x=alcdata["famrel"],y=alcdata["G_avg"])
alcdata.skew()
sns.distplot(new_alcdata.absences,bins=100)
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

def min_max_transform(x):
    return (x-x.min())/(x.max()-x.min())
sns.distplot(min_max_transform(new_alcdata.absences)**0.5,bins=100)

fifadata["Release Clause"]=fifadata["Release Clause"].replace('[\€,]', '', regex=True).replace('M','e06' , regex=True).replace('K','e03' , regex=True).astype(float)
fifadata["Value"]= fifadata["Value"].replace('[\€,]', '', regex=True).replace('M','e06' , regex=True).replace('K','e03' , regex=True).astype(float)
fifadata["Wage"]= fifadata["Wage"].replace('[\€,]', '', regex=True).replace('M','e06' , regex=True).replace('K','e03' , regex=True).astype(float)


club_economy=fifadata[["Wage","Value","Club","Release Clause","Overall"]].groupby(["Club"]).sum()
economical=(club_economy["Release Clause"]+club_economy.Value-club_economy.Wage)/club_economy["Overall"]
economical.sort_values(ascending = False)
sns.lineplot(fifadata.Age,fifadata.Potential)
# sns.barplot(fifadata.Age,fifadata.Potential)
sns.lineplot(fifadata.Age,fifadata.Value)
sns.lineplot(fifadata.Age,fifadata.SprintSpeed)

fifadata.corr()["Potential"]
sns.lineplot(fifadata["Penalties"],fifadata["Potential"])
sns.lineplot(fifadata["HeadingAccuracy"],fifadata["Potential"])
sns.lineplot(fifadata["Crossing"],fifadata["Potential"])
sns.lineplot(fifadata["ShotPower"],fifadata["Potential"])
sns.lineplot(fifadata["Reactions"],fifadata["Potential"])
sns.lineplot(fifadata["Weak Foot"],fifadata["Potential"])
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
fifadata.corr()["Wage"]
sns.lineplot(fifadata["Value"],fifadata["Wage"])
sns.lineplot(fifadata["Overall"],fifadata["Wage"])
sns.lineplot(fifadata["International Reputation"],fifadata["Wage"])
sns.lineplot(fifadata["Release Clause"],fifadata["Wage"])
tp1=fifadata[["Age","Club"]].groupby(["Club"]).describe()
tp1.sort_values([('Age',  'mean')])

accidents=pd.concat([accidata1,accidata2,accidata3],axis=0)


accidents
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
week_casualty={"Mon":0,"Tue":0,"Wed":0,"Thurs":0,"Fri":0,"Sat":0,"Sun":0}
week_casualty["Mon"]=accidents[accidents.Day_of_Week==1].Number_of_Casualties.sum()
week_casualty["Tue"]=accidents[accidents.Day_of_Week==2].Number_of_Casualties.sum()
week_casualty["Wed"]=accidents[accidents.Day_of_Week==3].Number_of_Casualties.sum()
week_casualty["Thurs"]=accidents[accidents.Day_of_Week==4].Number_of_Casualties.sum()
week_casualty["Fri"]=accidents[accidents.Day_of_Week==5].Number_of_Casualties.sum()
week_casualty["Sat"]=accidents[accidents.Day_of_Week==6].Number_of_Casualties.sum()
week_casualty["Sun"]=accidents[accidents.Day_of_Week==7].Number_of_Casualties.sum()
week_casualty
# {k: v for k, v in sorted(l.items(), key=lambda item: item[1])}
# l=sorted(l,key=lambda item:item[1])
{k: v for k, v in sorted(week_casualty.items(), key=lambda item: item[1],reverse=True)}

#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 

day_speed={"Mon":(0,0),"Tue":(0,0),"Wed":(0,0),"Thurs":(0,0),"Fri":(0,0),"Sat":(0,0),"Sun":(0,0)}
day_speed["Mon"]=(accidents[accidents.Day_of_Week==1].Speed_limit.min(),accidents[accidents.Day_of_Week==1].Speed_limit.max())
day_speed["Tue"]=(accidents[accidents.Day_of_Week==2].Speed_limit.min(),accidents[accidents.Day_of_Week==1].Speed_limit.max())
day_speed["Wed"]=(accidents[accidents.Day_of_Week==3].Speed_limit.min(),accidents[accidents.Day_of_Week==1].Speed_limit.max())
day_speed["Thurs"]=(accidents[accidents.Day_of_Week==4].Speed_limit.min(),accidents[accidents.Day_of_Week==1].Speed_limit.max())
day_speed["Fri"]=(accidents[accidents.Day_of_Week==5].Speed_limit.min(),accidents[accidents.Day_of_Week==1].Speed_limit.max())
day_speed["Sat"]=(accidents[accidents.Day_of_Week==6].Speed_limit.min(),accidents[accidents.Day_of_Week==1].Speed_limit.max())
day_speed["Sun"]=(accidents[accidents.Day_of_Week==7].Speed_limit.min(),accidents[accidents.Day_of_Week==1].Speed_limit.max())
day_speed
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
plt.figure(figsize=(16,10))
accidents.Accident_Severity.value_counts()
sns.barplot(x=accidents.Light_Conditions,y=accidents.Accident_Severity)
plt.figure(figsize=(16,10))
sns.lineplot(x=accidents.Light_Conditions,y=accidents.Accident_Severity)
plt.figure(figsize=(16,10))
sns.barplot(x=accidents.Weather_Conditions,y=accidents.Accident_Severity)
plt.figure(figsize=(16,10))
sns.lineplot(x=accidents.Weather_Conditions,y=accidents.Accident_Severity)
sns.lineplot(accidents["1st_Road_Class"],accidents["Accident_Severity"])
number_of_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
for i in range(1, len(number_of_days)):
    number_of_days[i] += number_of_days[i-1]

print(number_of_days)

month = accidents.Date.apply(lambda x: int(x[3:5]))
date = accidents.Date.apply(lambda x: int(x[:2]))

accidents.Date = month.apply(lambda x: number_of_days[x-1])+date
accidents.Date
sns.lineplot(accidents["Date"],accidents["Accident_Severity"])
def convert_time(time):
    if pd.isnull(time):
        return pd.NA
    
    hours, minutes = time.split(":")
    hours, minutes = int(hours), int(minutes)
    return int(hours + minutes/60)

accidents.Time = accidents.Time.apply(convert_time)
accidents.Time.fillna(accidents.Time.mean(), inplace=True)
accidents.Time
sns.lineplot(accidents["Time"],accidents["Accident_Severity"])
sns.lineplot(accidents["Year"],accidents["Accident_Severity"])
sns.lineplot(accidents["Day_of_Week"],accidents["Accident_Severity"])
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
new_accidents=accidents.drop("Accident_Severity",axis=1)
y=accidents["Accident_Severity"]
new_accidents.corr()
new_accidents["LSOA_of_Accident_Location"]
sns.lineplot(new_accidents["1st_Road_Number"],y)
new_accidents["2nd_Road_Number"].nunique()
new_accidents["1st_Road_Number"].nunique()
sns.boxplot(new_accidents["Local_Authority_(Highway)"],y)
new_accidents1 = new_accidents.drop(["Accident_Index","Location_Easting_OSGR","Location_Northing_OSGR","Local_Authority_(District)","Junction_Detail","Year","Date","Time","LSOA_of_Accident_Location","Local_Authority_(Highway)","1st_Road_Number","2nd_Road_Number"],axis=1)
objects=new_accidents1.select_dtypes(include=["object"])
not_objects=new_accidents1.select_dtypes(include=["float64","int64"])


objects=objects.fillna("NA")
not_objects["Longitude"]=not_objects["Longitude"].fillna(not_objects["Longitude"].mean())
not_objects["Latitude"]=not_objects["Latitude"].fillna(not_objects["Latitude"].mean())
not_objects.isnull().sum()

objects1=pd.get_dummies(objects)
accidata1=pd.concat([objects1,not_objects],axis=1)
accidata1.dtypes.value_counts()

accidata1.isnull().sum()
#enter code/answer in this cell. You can add more code/markdown cells below for your answer. 
import sklearn
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
y1=[]
y2=[]
y3=[]
for i in y:
    if i==1:
        y1.append(1)
        y2.append(0)
        y3.append(0)
    if i==2:
         y1.append(0)
         y2.append(1)
         y3.append(0)
    if i==3:
         y1.append(0)
         y2.append(0)
         y3.append(1)
        

model1=LogisticRegression(max_iter=1000)
model2=LogisticRegression(max_iter=1000)
model3=LogisticRegression(max_iter=1000)
X_train1,X_test1,Y_train1,Y_test1=train_test_split(accidata1,y1,test_size=0.2)
X_train2,X_test2,Y_train2,Y_test2=train_test_split(accidata1,y2,test_size=0.2)
X_train3,X_test3,Y_train3,Y_test3=train_test_split(accidata1,y3,test_size=0.2)


model1.fit(X_train1,Y_train1)

model2.fit(X_train2,Y_train2)
model3.fit(X_train3,Y_train3)
y_pred1=model1.predict(accidata1)
y_pred2=model2.predict(accidata1)
y_pred3=model3.predict(accidata1)

y_pred=[]
for i in range(len(y_pred1)):
    if y_pred1[i]==1:
        y_pred.append(1)
    elif y_pred2[i]==1:
        y_pred.append(2)
    else:
        y_pred.append(3)
accuracy_score(y_pred,y)
model=LogisticRegressionCV(cv=5,multi_class="multinomial",n_jobs=-1,max_iter=1000)

scaler=StandardScaler()
accidata1=scaler.fit_transform(accidata1)
accidata1
X_train,X_test,Y_train,Y_test=train_test_split(accidata1,y,test_size=0.2)
model.fit(X_train,Y_train)
y_pred_2=model.predict(X_test)
accuracy_score(y_pred_2,Y_test)