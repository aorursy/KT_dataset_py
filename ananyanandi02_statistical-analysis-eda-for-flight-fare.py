import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
import seaborn as sns
train = pd.read_excel('../input/flight-fare-prediction-mh/Data_Train.xlsx')
train.shape
train.head(5)
print(train.isnull().sum()) #checking for null values
train=train.dropna()
train['Journey_Day'] = pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.day
train['Journey_Month'] = pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.month
train['weekday']= pd.to_datetime(train.Date_of_Journey, format='%d/%m/%Y').dt.weekday


train.drop(labels = 'Date_of_Journey', axis = 1, inplace = True)

train.columns
def duration(test):
    test = test.strip()
    total=test.split(' ')
    to=total[0]
    hrs=(int)(to[:-1])*60
    if((len(total))==2):
        mint=(int)(total[1][:-1])
        hrs=hrs+mint
    test=str(hrs)
    return test
train['Duration']=train['Duration'].apply(duration)

train['Duration'].nunique()
def deparrtime(x):
    x=x.strip()
    tt=(int)(x.split(':')[0])
    if(tt>=16 and tt<21):
        x='Evening'
    elif(tt>=21 or tt<5):
        x='Night'
    elif(tt>=5 and tt<11):
        x='Morning'
    elif(tt>=11 and tt<16):
        x='Afternoon'
    return x
train['Dep_Time']=train['Dep_Time'].apply(deparrtime)
train['Arrival_Time']=train['Arrival_Time'].apply(deparrtime)

def stops(x):
    if(x=='non-stop'):
        x=str(0)
    else:
        x.strip()
        stps=x.split(' ')[0]
        x=stps
    return x
train['Total_Stops']=train['Total_Stops'].apply(stops)
pd.options.mode.chained_assignment = None 
for i in range(train.shape[0]):
    if(train.iloc[i]['Additional_Info']=='No info'):
        train.iloc[i]['Additional_Info']='No Info' 
train=train.drop(['Route'], axis=1) #we don't need it as we already have total_stops
train.head(2)
train.info()
train["Duration"] = train["Duration"].astype(int)
train["Journey_Day"] = train["Journey_Day"].astype(object)
train["Journey_Month"] = train["Journey_Month"].astype(object)
train["weekday"] = train["weekday"].astype(object)
df1 =train.copy() 
df1["Journey_Month"]=df1["Journey_Month"].replace({3:"March",4:"April",5:"May",6:"June"}) #assigning month names
df1["Journey_Month"]=df1["Journey_Month"].astype(object)
df1.info()
#Journey month v/s total fare
v1=sns.barplot(x='Journey_Month', y='Price', data=df1,estimator=sum)
v1.set_title('Monthv/sPrice')
v1.set_ylabel('Price')
v1.set_xlabel('Month of booking')
v1.set_xticklabels(v1.get_xticklabels(), rotation=80)
#count of flights per month
top_month=df1.Journey_Month.value_counts().head(10)
top_month
monthly_avg=df1.groupby(['Journey_Month']).agg({'Price':np.mean}).reset_index()
#Journey month v/s Averagefare
monthly_avg.plot(x='Journey_Month',y='Price',figsize=(6,6))
# Destination vs AveragePrice
sns.catplot(y='Price',x='Destination',data= df1.sort_values('Price',ascending=False),kind="boxen",height=6, aspect=3)
plt.show
# Source vs AveragePrice
sns.catplot(y='Price',x='Source',data= train.sort_values('Price',ascending=False),kind="boxen",height=6, aspect=3)
plt.show
#Count of flights v/s Airline
plt.figure(figsize = (15, 10))
plt.title('Count of flights with different Airlines')
ax=sns.countplot(x = 'Airline', data =train)
plt.xlabel('Airline')
plt.ylabel('Count of flights')
plt.xticks(rotation = 90)
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',
                    color= 'black')
# Airline vs AveragePrice
sns.catplot(y='Price',x='Airline',data= train.sort_values('Price',ascending=False),kind="boxen",height=6, aspect=3)
plt.show
#duration v/s AveragePrice
sns.scatterplot(data=train, x='Duration', y='Price')
#Deptarure time v/s AveragePrice
v2=sns.barplot(x='Dep_Time', y='Price', data=train)
v2.set_ylabel('Price')
v2.set_xlabel('Time of dept')
v2.set_xticklabels(v2.get_xticklabels(), rotation=80)
# time of departure v/s count of flights
top_time=train.Dep_Time.value_counts().head(10)
top_time
#TIME OF ARRIVAL V/S average price
v3=sns.barplot(x='Arrival_Time', y='Price', data=train)
v3.set_title('TIME OF ARRIVALV/S PRICE')
v3.set_ylabel('Price')
v3.set_xlabel('Arrival_time')
v3.set_xticklabels(v3.get_xticklabels(), rotation=80)
#total stops v/s average price
v4=sns.barplot(x='Total_Stops', y='Price', data=train)
v4.set_title('NO. OF STOPS V/S PRICE')
v4.set_ylabel('Price')
v4.set_xlabel('Total_Stops')
v4.set_xticklabels(v4.get_xticklabels(), rotation=80)
#WEEKDAY V/S average price
v4=sns.barplot(x='weekday', y='Price', data=train)
v4.set_title('WEEKDAY V/S PRICE')
v4.set_ylabel('Price')
v4.set_xlabel('WEEKDAY')
v4.set_xticklabels(v4.get_xticklabels(), rotation=80)
train["Journey_Day"].unique()
#Count of flights with different dates
plt.figure(figsize = (15, 10))
plt.title('Count of flights with different dates')
ax=sns.countplot(x = 'Journey_Day', data =train)
plt.xlabel('Journey_Day')
plt.ylabel('Count of flights')
plt.xticks(rotation = 90)
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',
                    color= 'black')
#Journey_Day v/s Average price
v5=sns.barplot(x='Journey_Day', y='Price', data=train)
v5.set_title('Price of flights with different datess')
v5.set_ylabel('Price')
v5.set_xlabel('date')
v5.set_xticklabels(v5.get_xticklabels(), rotation=80)
print(train.dtypes)
ncol=["Duration"]          
for i in ncol:
    q75, q25 = np.percentile(train.loc[:,i], [75 ,25])
    iqr = q75 - q25
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    train = train.drop(train[train.loc[:,i] <= min].index)
    train = train.drop(train[train.loc[:,i] >= max].index)
train.shape
import scipy.stats as stats
_ = sns.jointplot(x='Duration',y='Price',data=train,kind = 'reg')
_.annotate(stats.pearsonr)
plt.show()
# Import label encoder 
colnames = list(train.columns)
from sklearn import preprocessing 

# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
for col in colnames:
    if train[col].dtype==object:
        train[col]= label_encoder.fit_transform(train[col]) 
cat_var=["Airline","Source","Destination","Dep_Time","Arrival_Time","Total_Stops","Additional_Info","Journey_Day","Journey_Month","weekday"] 
catdf=train[cat_var]
catdf.head(2)
from sklearn.feature_selection import chi2
n= 10
for i in range(0,9):
    X=catdf.iloc[:,i+1:n]
    y=catdf.iloc[:,i]
    chi_scores = chi2(X,y)
    p_values = pd.Series(chi_scores[1],index = X.columns)
    print("for",i)
    print(p_values)
    for j in range (0, len(p_values)):
        if (p_values[j]<0.01):
            print(p_values[j])
train=train.drop(["Airline","Source","Destination","Total_Stops","Journey_Month","Journey_Day","Arrival_Time"],axis=1)
train.info()
import statsmodels.api as sm
from statsmodels.formula.api import ols
model = ols('Price ~ C(Dep_Time)+C(weekday)+C(Additional_Info)',data=train).fit()
aov_table = sm.stats.anova_lm(model)
aov_table
probanova=list(aov_table["PR(>F)"])
for i in range(0,4):
    if probanova[i]>0.05:
        print(i)
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)
df1=train.drop(["Price"],axis=1)
calc_vif(df1)
train = train.drop(["Additional_Info"],axis=1)
train["weekday"] = train["weekday"].astype(object)
train["Dep_Time"] = train["Dep_Time"].astype(object)
train.head(2)
train = pd.get_dummies(train, drop_first=True)
train.info()
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
x = train.drop('Price',axis=1).values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)
y = train['Price'].values
model = sm.OLS(y,X).fit()
model.summary()
