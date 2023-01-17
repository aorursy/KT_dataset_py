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
# Change 'START_DATE*','END_DATE*' to time format
datacopy['START_DATE*'] = pd.to_datetime(datacopy['START_DATE*'])
datacopy['END_DATE*'] = pd.to_datetime(datacopy['END_DATE*'])
# Extract 'Hour','Month','Day of Week','Date' from 'START_DATE*'
datacopy['Hour'] = datacopy['START_DATE*'].apply(lambda time: time.hour)
datacopy['Month'] = datacopy['START_DATE*'].apply(lambda time: time.month)
datacopy['Day of Week'] = datacopy['START_DATE*'].apply(lambda time: time.dayofweek)
datacopy['Date'] = datacopy['START_DATE*'].apply(lambda time: time.date())
datacopy.head()
# Convert 'Day of Week' from numerical to text(that we can understand)
daymap ={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
datacopy['Day of Week'] = datacopy['Day of Week'].map(daymap)
datacopy.head()
# Try to find the hiden relationship between the missing value and 'Day of Week'
plt.figure(figsize=(20,8))
sns.countplot(x='Day of Week',data = datacopy,hue = 'PURPOSE*')
plt.legend(bbox_to_anchor = (1.05,1),loc=2,borderaxespad=0.)
# Try to find the hiden relationship between the missing value and day of 'Hour'
plt.figure(figsize=(20,8))
sns.countplot(x='Hour',data = datacopy,hue = 'PURPOSE*')
plt.legend(bbox_to_anchor = (1.05,1),loc=2,borderaxespad=0.)
datacopy.head()
datacopy['Hour'].unique()
#Fill the missing value
datacopy[(datacopy['Hour'] >= 1) & (datacopy['Hour'] <= 14)] = datacopy[(datacopy.Hour >= 1) & (datacopy.Hour <= 14)].fillna({'PURPOSE*':'Meeting'})
datacopy[(datacopy['Hour'] >= 15) & (datacopy['Hour'] <= 21)] = datacopy[(datacopy['Hour'] >= 15) & (datacopy['Hour'] <= 21)].fillna({'PURPOSE*':'Meal/Entertain'})
datacopy[(datacopy['Hour'] >= 22) | (datacopy['Hour'] == 0)] = datacopy[(datacopy['Hour'] >= 22) | (datacopy['Hour'] == 0 )].fillna({'PURPOSE*':'Meeting'})
#datacopy[(datacopy['Hour'] == 0)] = datacopy[(datacopy['Hour'] == 0)].fillna({'PURPOSE*':'Meeting'})

datacopy.isnull().sum()
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
# Combine 'Charity ($)','Commute','Moving','Airport/Travel' into 'Others'
dp = datacopy
dp.replace(['Charity ($)', 'Commute','Moving','Airport/Travel'],'Others',inplace = True)
plt.figure(figsize=(12,12))
dp['PURPOSE*'].value_counts()[:11].plot(kind='pie',autopct='%1.1f%%',shadow=True,legend = True)
plt.show()
datacopy['CATEGORY*'].value_counts()
#plot 
plt.figure(figsize=(15,5))
sns.countplot(datacopy['CATEGORY*'])
per_month =pd.DataFrame()
per_month =datacopy.groupby('Month').sum()
plt.figure(figsize=(20,8))
sns.barplot(x='Month',y='MILES*',data=per_month.reset_index())

ByHour =pd.DataFrame()
ByHour =datacopy.groupby('Hour').sum()
plt.figure(figsize=(20,8))
sns.barplot(x='Hour',y='MILES*',data=ByHour.reset_index())

Pur_Mil = datacopy.groupby('PURPOSE*')['MILES*'].sum()
Pur_Mil

plt.figure(figsize=(20,8))
sns.barplot(x='PURPOSE*',y='MILES*',data=Pur_Mil.reset_index())
CAT_Mil_Mean = datacopy.groupby('PURPOSE*').mean()
CAT_Mil_Mean
plt.figure(figsize=(15,10))
CAT_Mil_Mean['PURPOSE*']=CAT_Mil_Mean.index.tolist()
ax = sns.barplot(x='MILES*',y='PURPOSE*',data=CAT_Mil_Mean ,order=CAT_Mil_Mean.sort_values('MILES*',ascending=False)['PURPOSE*'].tolist())
ax.set(xlabel='Avrg Miles', ylabel='Purpose')
plt.show()

MilPurMon = datacopy.groupby('Month')['MILES*'].sum()

plt.figure(figsize=(20,8))
sns.barplot(x='Month',y='MILES*',data=MilPurMon.reset_index())
plt.tight_layout()
MilPurMon = datacopy.groupby('Month').count()['MILES*'].plot()
#Month purpose regression
sns.lmplot(x='Month',y='PURPOSE*',data=datacopy.groupby('Month').count().reset_index())
#Heatmap
dayHour = datacopy.groupby(by=['Day of Week','Hour']).count()['PURPOSE*'].unstack()
plt.figure(figsize=(20,12))
sns.heatmap(dayHour,cmap='coolwarm',linecolor='white',linewidth=1)
CAT_Mil_SUM = datacopy.groupby('CATEGORY*').sum()
plt.figure(figsize=(10,8))
sns.barplot(x='CATEGORY*',y='MILES*',data=CAT_Mil_SUM.reset_index())
plt.tight_layout()

datacopy["END_DATE*"]=pd.to_datetime(datacopy["END_DATE*"],format="%m/%d/%Y %H:%M")
speed=datacopy["MILES*"]/((datacopy["END_DATE*"]-datacopy["START_DATE*"]).dt.seconds/60)
#print(speed)
datacopy["SPEED*"]=speed
datacopy["START_HOUR*"]=datacopy["START_DATE*"].dt.hour
spd_df=datacopy[datacopy["SPEED*"]!=np.inf].groupby(["START_HOUR*"])["SPEED*"].mean()
datacopy.head()
plt.figure(figsize=(20,8))
sns.barplot(x="START_HOUR*",y="SPEED*",data=spd_df.reset_index())
plt.title("Speed")
plt.xlabel("Time(Hour)")
plt.ylabel("Speed[Mile(s)/min]")
plt.xticks(spd_df.index)
plt.grid()