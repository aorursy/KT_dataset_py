import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import statistics 
from datetime import datetime
sns.set(style="darkgrid")
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_org=pd.read_csv('/kaggle/input/us-accidents/US_Accidents_Dec19.csv')
df=df_org.sample(500000, random_state =21)
df.info()
plt.figure(figsize=(20, 5))
plt.title('Distribution of recordings per state')
sns.countplot(x=df['State'], data=df)
plt.figure(figsize=(10, 4))
plt.subplot(111)
sns.distplot(df['Severity'])
plt.title('Severity distribution as histogram')
plt.figure(figsize=(10, 4))
plt.subplot(111)
sns.distplot(df['Distance(mi)'])
plt.title('Distance distribution as histogram')
bool_features=['Amenity',
              'Bump',
              'Crossing',
              'Give_Way',
              'Junction',
              'No_Exit',
              'Railway',
              'Roundabout',
              'Station',
              'Stop',
              'Traffic_Calming',
              'Traffic_Signal',
              'Turning_Loop']

for i in bool_features:
    df_temp=df[i].copy()
    df_temp[df_temp==False]=0
    df_temp[df_temp==True]=1
    df[i] = df_temp
    
plt.figure(figsize=(20, 10))
plt.subplot(431)
sns.countplot(df['Amenity'])
plt.title('Amenity')
plt.xlabel('')
plt.subplot(432)
sns.countplot(df['Bump'])
plt.title('Bump')
plt.xlabel('')
plt.subplot(433)
sns.countplot(df['Crossing'])
plt.title('Crossing')
plt.xlabel('')
plt.subplot(434)
sns.countplot(df['Give_Way'])
plt.title('Give_Way')
plt.xlabel('')
plt.subplot(435)
sns.countplot(df['Junction'])
plt.title('Junction')
plt.xlabel('')
plt.subplot(436)
sns.countplot(df['No_Exit'])
plt.title('No_Exit')
plt.xlabel('')
plt.subplot(437)
sns.countplot(df['Railway'])
plt.title('Railway')
plt.xlabel('')
plt.subplot(438)
sns.countplot(df['Roundabout'])
plt.title('Roundabout')
plt.xlabel('')
plt.subplot(439)
sns.countplot(df['Station'])
plt.title('Station')
plt.figure(figsize=(20, 10))
plt.subplot(431)
sns.countplot(df['Stop'])
plt.title('Stop')
plt.xlabel('')
plt.subplot(432)
sns.countplot(df['Traffic_Calming'])
plt.title('Traffic_Calming')
plt.xlabel('')
plt.subplot(433)
sns.countplot(df['Traffic_Signal'])
plt.title('Traffic_Signal')
plt.xlabel('')
plt.subplot(434)
sns.countplot(df['Turning_Loop'])
plt.title('Turning_Loop')
plt.xlabel('')
weather_features=['Weather_Timestamp',
                  'Temperature(F)',
                  'Wind_Chill(F)',
                  'Humidity(%)',
                  'Pressure(in)',
                  'Visibility(mi)',
                  'Wind_Direction',
                  'Wind_Speed(mph)',
                  'Precipitation(in)',
                  'Weather_Condition']
print('Types')  
print(df[weather_features].info())
print('\n')
print('Count nan-values')
print(df[weather_features].isna().sum())
df['Wind_Direction'].unique()
df['Wind_Direction']=df['Wind_Direction'].replace('E', 'East')
df['Wind_Direction']=df['Wind_Direction'].replace('N', 'North')
df['Wind_Direction']=df['Wind_Direction'].replace('W', 'West')
df['Wind_Direction']=df['Wind_Direction'].replace('S', 'South')
df['Wind_Direction']=df['Wind_Direction'].replace('CALM', 'Calm')
df['Wind_Direction'].unique()
factor_wd = pd.factorize(df['Wind_Direction'])
df['Wind_Direction'] = factor_wd[0]
df['Weather_Condition'].unique()
factor_wc = pd.factorize(df['Weather_Condition'])
df['Weather_Condition'] = factor_wc[0]
acc_features=['Temperature(F)',
              'Wind_Chill(F)',
              'Humidity(%)',
              'Pressure(in)',
              'Visibility(mi)',
              'Wind_Speed(mph)',
              'Precipitation(in)']

for feature in acc_features:
    df[feature]=df[feature].fillna(df[feature].median())
plt.figure(figsize=(20, 12))
plt.subplot(341)
sns.boxplot(x=df['Temperature(F)'])
plt.title('Temperature(F)')
plt.xlabel('')
plt.subplot(342)
sns.boxplot(x=df['Wind_Chill(F)'])
plt.title('Wind_Chill(F)')
plt.xlabel('')
plt.subplot(343)
sns.boxplot(x=df['Humidity(%)'])
plt.title('Humidity(%)')
plt.xlabel('')
plt.subplot(344)
sns.boxplot(x=df['Pressure(in)'])
plt.title('Pressure(in)')
plt.xlabel('')
plt.subplot(345)
sns.boxplot(x=df['Visibility(mi)'])
plt.title('Visibility(mi)')
plt.xlabel('')
plt.subplot(346)
sns.boxplot(x=df['Wind_Speed(mph)'])
plt.title('Wind_Speed(mph)')
plt.xlabel('')
plt.subplot(347)
sns.boxplot(x=df['Precipitation(in)'])
plt.title('Precipitation(in)')
plt.xlabel('')

df=df[df['Wind_Speed(mph)'] <= 253]
df=df[df['Temperature(F)'] <= 134]
plt.figure(figsize=(20, 12))
plt.subplot(211)
g=sns.countplot(x=df['Wind_Direction'])
plt.title('Wind_Direction')
plt.xlabel('')
x_list=[]
for i in factor_wd[1]:
    x_list.append(i)
g.set_xticklabels(x_list)
plt.subplot(212)
o=sns.countplot(df['Weather_Condition'])
plt.title('Weather_Condition')
plt.xlabel('')
x_list=[]
for i in factor_wc[1]:
    x_list.append(i)
o.set_xticklabels(x_list, size=10, rotation=90)
plt.show()
df['Start_Time']=pd.to_datetime(df['Start_Time'])
df['End_Time']=pd.to_datetime(df['End_Time'])
df['Start_Time_weekday']=df['Start_Time'].dt.dayofweek
df['Start_Time_hour']=df['Start_Time'].dt.hour
df['Timezone'].unique()
for row in df.index:
    if df.loc[row,'Timezone']=='US/Eastern':
        df.loc[row,'Start_Time_hour']=df.loc[row,'Start_Time_hour']-1
    elif df.loc[row,'Timezone']=='US/Pacific':
        df.loc[row,'Start_Time_hour']=df.loc[row,'Start_Time_hour']+2
    elif df.loc[row,'Timezone']=='US/Mountain':
        df.loc[row,'Start_Time_hour']=df.loc[row,'Start_Time_hour']+1
plt.figure(figsize=(20,8))
plt.subplot(211)
sns.distplot(df['Start_Time_weekday'])
plt.title('Start_Time_weekday')
plt.xlabel('')
plt.subplot(212)
sns.distplot(df['Start_Time_hour'])
plt.title('Start_Time_hour')
plt.xlabel('')
print('Number of records wth day')
print(df[df['Sunrise_Sunset']=='Day']['ID'].count())
print('Number of records wth night')
print(df[df['Sunrise_Sunset']=='Night']['ID'].count())

# Filling the nans with day

df['Sunrise_Sunset']=df['Sunrise_Sunset'].fillna('Day')
print('nans left')
print(df['Sunrise_Sunset'].isna().sum())
df['Sunrise_Sunset']=df['Sunrise_Sunset'].replace('Day',0)
df['Sunrise_Sunset']=df['Sunrise_Sunset'].replace('Night',1)
plt.figure(figsize=(10, 6))
plt.subplot(211)
g=sns.countplot(x=df['Sunrise_Sunset'])
plt.title('Sunrise_Sunset')
plt.xlabel('')
g.set_xticklabels(['Day','Night'])
feat_columns=['State',
              'Severity',
              'Distance(mi)', 
              'Temperature(F)',
              'Wind_Chill(F)',
              'Humidity(%)',
              'Wind_Direction',
              'Weather_Condition',
              'Visibility(mi)',
              'Wind_Speed(mph)',
              'Precipitation(in)',
              'Start_Time_hour',
              'Start_Time_weekday',
              'Sunrise_Sunset',
              'Amenity',
              'Bump',
              'Crossing',
              'Give_Way',
              'Junction',
              'No_Exit',
              'Railway',
              'Roundabout',
              'Station',
              'Stop',
              'Traffic_Calming',
              'Traffic_Signal',
              'Turning_Loop']

# accident focused features 
featO_columns=['State',
              'Severity',
              'Distance(mi)',
              'Start_Time_hour',
              'Start_Time_weekday',
              'Sunrise_Sunset',
              'Weather_Condition',
              'Amenity',
              'Bump',
              'Crossing',
              'Give_Way',
              'Junction',
              'No_Exit',
              'Railway',
              'Roundabout',
              'Station',
              'Stop',
              'Traffic_Calming',
              'Traffic_Signal',
              'Turning_Loop']




# Last check    
df_feat=df[feat_columns]    
#print(df_feat.isna().sum())
print(df_feat.isna().sum())
print(df_feat.info())

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
df_feat=df[feat_columns]
# Convert state to numbers
factor = pd.factorize(df_feat['State'])
df_feat['State'] = factor[0]
df_feat['State'].unique()
#Splitting the data into independent and dependent variables
target='State'
y = df_feat[target]
X = df_feat.drop(columns=target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score

def multiclass_roc_auc_score(y_test, y_pred, average="weighted"):
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    print(roc_auc_score(y_test, y_pred, average=average))
    return

def multiclass_f1_score(y_test, y_pred, average="weighted"):
    f1=f1_score(y_test, y_pred, average=average)
    print(f1)
    return
    
def multiclass_classification_report(y_test, y_pred):
    list=[]
    for i in factor[1]:
        list.append(i)
    print(classification_report(y_test,y_pred,target_names=list))
    return
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)
definitions=factor[1]
state_num= len(definitions)
y_pred = classifier.predict(X_test)
feature_imp = pd.Series(classifier.feature_importances_,index=X.columns).sort_values(ascending=False)

k=10
sns.barplot(x=feature_imp[:10], y=feature_imp.index[:k])
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
print('ROC AUC Score')
print(multiclass_roc_auc_score(y_test, y_pred))
print('F1 Score')
print(multiclass_f1_score(y_test, y_pred))
list=[]
for i in factor[1]:
    list.append(i)

report = classification_report(y_test, y_pred, output_dict=True, target_names=list)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('dataset_whole_report.csv')
df_feat=df[featO_columns]
factor = pd.factorize(df_feat['State'])
df_feat['State'] = factor[0]
target='State'
y = df_feat[target]
X = df_feat.drop(columns=target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 21)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)
definitions=factor[1]
state_num= len(definitions)
y_pred = classifier.predict(X_test)
feature_imp = pd.Series(classifier.feature_importances_,index=X.columns).sort_values(ascending=False)

k=10
sns.barplot(x=feature_imp[:10], y=feature_imp.index[:k])
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
print('ROC AUC Score')
print(multiclass_roc_auc_score(y_test, y_pred))
print('F1 Score')
print(multiclass_f1_score(y_test, y_pred))
list=[]
for i in factor[1]:
    list.append(i)

report = classification_report(y_test, y_pred, output_dict=True, target_names=list)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv('dataset_acc_report.csv')
df_prop_state=pd.DataFrame(columns=['State','Count','Prop'])
df_prop_state
list_state=df['State'].unique()
list_state
i=0
for state in list_state:
    count=len(df[df['State']== state].index)
    prop=count/len(df.index)
    df_prop_state.loc[i,'State']=state
    df_prop_state.loc[i,'Count']=count
    df_prop_state.loc[i,'Prop']=prop
    i+=1
df_prop_state=df_prop_state.set_index('State')
df_prop_state
df_report_whole=pd.read_csv('/kaggle/working/dataset_whole_report.csv', index_col=0, header=0)
df_report_acc=pd.read_csv('/kaggle/working/dataset_acc_report.csv', index_col=0, header=0)
df_report_whole=df_report_whole.rename(columns={'precision':'whole_df_precision'})
df_report_acc=df_report_acc.rename(columns={'precision':'acc_df_precision'})
df_compare = pd.concat([df_report_whole, df_report_acc,df_prop_state], axis=1, join='inner')
df_compare=df_compare[['whole_df_precision','acc_df_precision','Prop']]
df_compare['DELTA whole_df vs prop']=df_compare['whole_df_precision']-df_compare['Prop']
df_compare['DELTA acc vs prop']=df_compare['acc_df_precision']-df_compare['Prop']
df_compare
df_compare['DELTA acc vs prop'].mean()