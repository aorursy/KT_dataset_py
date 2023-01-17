# import relative python libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 
import datetime as dt
%matplotlib inline 
import warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# load dataset into Pandas
mydata = pd.read_csv('../input/My Uber Drives - 2016.csv')
mydata.info()
mydata.head()
mydata.tail()
mydata.isnull().sum()
# Copy a dataset
datacopy = mydata.copy()
# delete the last line
datacopy = datacopy.drop(datacopy.index[1155])
start_list = [info.split(' ') for info in datacopy['START_DATE*'].tolist()]
stop_list  = [info.split(' ') for info in datacopy['END_DATE*'].tolist()]

start  = pd.DataFrame(start_list,columns=['Start_Date','Start_Time'])
end    = pd.DataFrame(stop_list,columns=['End_Date','End_Time'])

sub_data = datacopy[['CATEGORY*','START*','STOP*','MILES*','PURPOSE*']]
start_end_total = pd.concat([start,end,],axis=1)
datacopy = pd.concat([ start_end_total, sub_data],axis=1)
datacopy.head(10)


ml_dis=datacopy["MILES*"]
ml_range_lst=["<=5","5-10","10-15","15-20",">20"]
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.03*height, '%s' % int(height))
ml_dic=dict()
for item in ml_range_lst:
    ml_dic[item]=0
for mile in ml_dis.values:
    if mile<=5:
        ml_dic["<=5"]+=1
    elif mile<=10:
        ml_dic["5-10"]+=1
    elif mile<=15:
        ml_dic["10-15"]+=1
    elif mile<=20:
        ml_dic["15-20"]+=1
    else:
        ml_dic[">20"]+=1
ml_dis=pd.Series(ml_dic)
ml_dis.sort_values(inplace=True,ascending=False)
print("Miles:\n",ml_dis)
#figure
rects=plt.bar(range(1,len(ml_dis.index)+1),ml_dis.values)
plt.title("Miles")
plt.xlabel("Miles")
plt.ylabel("Quantity")
plt.xticks(range(1,len(ml_dis.index)+1),ml_dis.index)
plt.grid()
autolabel(rects)
plt.savefig("./ml_dis_fig")
datacopy['PURPOSE*'].value_counts()
plt.figure(figsize=(15,8))
sns.countplot(datacopy['PURPOSE*'])
plt.figure(figsize=(12,12))
datacopy['PURPOSE*'].value_counts()[:11].plot(kind='pie',autopct='%1.1f%%',shadow=True,legend = True)
plt.show()
datacopy['CATEGORY*'].value_counts()
#plot 
plt.figure(figsize=(15,5))
sns.countplot(datacopy['CATEGORY*'])
datacopy["Start_Date"]=pd.to_datetime(datacopy["Start_Date"],format="%m/%d/%Y")
per_month=datacopy['Start_Date'].dt.month.value_counts()
per_month=per_month.sort_index()
per_month_mean = per_month.mean()
print("Month Distribute:\n",per_month)
plt.figure(figsize=(20,10))
sns.countplot(datacopy['Start_Date'].dt.month)
datacopy["Start_Time"]=pd.to_datetime(datacopy["Start_Time"],format="%H:%M")
per_hour = datacopy['Start_Time'].dt.hour.value_counts()
per_hour =per_hour.sort_index()
per_hour_mean = per_hour.mean()
print("Month Distribute:\n",per_hour)
plt.figure(figsize=(20,10))
sns.countplot(datacopy['Start_Time'].dt.hour)

Pur_Mil = datacopy.groupby('PURPOSE*').mean()
Pur_Mil
plt.figure(figsize=(15,10))
Pur_Mil['PURPOSE*']=Pur_Mil.index.tolist()
ax = sns.barplot(x='MILES*',y='PURPOSE*',data=Pur_Mil,order=Pur_Mil.sort_values('MILES*',ascending=False)['PURPOSE*'].tolist())
ax.set(xlabel='Avrg Miles', ylabel='Purpose')
plt.show()

rides_per_month = datacopy.groupby('Start_Date').sum()
rides_per_month['Month']=pd.to_datetime(rides_per_month.index.tolist()) #converting dates to a python friendly format
rides_per_month['Month']= rides_per_month['Month'].dt.to_period("M") #grouping dates by month
rides_per_month= rides_per_month.sort_values(by= 'Month',ascending=True)
total_miles_per_month =rides_per_month.groupby('Month').sum()
total_miles_per_month['MONTH'] = total_miles_per_month.index.tolist()
total_miles_per_month['MONTH'] = total_miles_per_month['MONTH'].astype(str) #converting the time stamp format to string
plt.figure(figsize=(15,10))
ax = sns.barplot(x='MILES*',y='MONTH',data=total_miles_per_month,order=total_miles_per_month.sort_values('MONTH',ascending=False)['MONTH'].tolist())
ax.set(xlabel='Total Miles', ylabel='Month')
plt.show()
CAT_Mil_SUM = datacopy.groupby('PURPOSE*').sum()
CAT_Mil_SUM 
plt.figure(figsize=(15,10))
CAT_Mil_SUM ['PURPOSE*']=CAT_Mil_SUM .index.tolist()
ax = sns.barplot(x='MILES*',y='PURPOSE*',data=CAT_Mil_SUM ,order=CAT_Mil_SUM.sort_values('MILES*',ascending=False)['PURPOSE*'].tolist())
ax.set(xlabel='Avrg Miles', ylabel='Purpose')
plt.show()
CAT_Mil_Mean = datacopy.groupby('PURPOSE*').mean()
CAT_Mil_Mean
plt.figure(figsize=(15,10))
CAT_Mil_Mean['PURPOSE*']=CAT_Mil_Mean.index.tolist()
ax = sns.barplot(x='MILES*',y='PURPOSE*',data=CAT_Mil_Mean ,order=CAT_Mil_Mean.sort_values('MILES*',ascending=False)['PURPOSE*'].tolist())
ax.set(xlabel='Avrg Miles', ylabel='Purpose')
plt.show()
datacopy["End_Time"]=pd.to_datetime(datacopy["End_Time"],format="%H:%M")
datacopy["Start_Time"]=pd.to_datetime(datacopy["Start_Time"],format="%H:%M")
speed=datacopy["MILES*"]/((datacopy["End_Time"]-datacopy["Start_Time"]).dt.seconds/60)
print(speed)
datacopy["SPEED*"]=speed
datacopy["START_HOUR*"]=datacopy["Start_Time"].dt.hour
print(datacopy[datacopy["SPEED*"]!=np.inf])
spd_df=datacopy[datacopy["SPEED*"]!=np.inf].groupby(["START_HOUR*"])["SPEED*"].mean()
print(spd_df)
#rects=plt.bar(range(0,len(spd_df.index)),spd_df.values)
rects=plt.bar(spd_df.index,spd_df.values)
plt.title("Speed")
plt.xlabel("Time(Hour)")
plt.ylabel("Speed[Mile(s)/min]")
plt.xticks(spd_df.index)
plt.grid()
plt.savefig("./speed_fig")

