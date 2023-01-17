import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv('../input/capstone-car-accident-serveity/Data_Collisions.csv')
df.head()
print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n     :',df.columns.tolist())
print('\nMissing values    :',df.isnull().values.sum())
print('\nUnique values :  \n',df.nunique())
df.dtypes
df= df[['SEVERITYCODE','ADDRTYPE','COLLISIONTYPE','PERSONCOUNT','PEDCOUNT','VEHCOUNT','INCDATE','JUNCTIONTYPE','WEATHER'
       ,'ROADCOND','LIGHTCOND','SPEEDING','INATTENTIONIND','UNDERINFL']]
df.select_dtypes(exclude=['int','float']).columns
df.select_dtypes(exclude=['object']).columns
print('Unique entries of SEVERITYCODE:\n',df.SEVERITYCODE.unique())
print('Datatype of SEVERITYCODE:\n',df.SEVERITYCODE.dtypes)
print('Null values in SEVERITYCODE:\n',df.SEVERITYCODE.isnull().any())
print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n     :',df.columns.tolist())
print('\nMissing values    :',df.isnull().values.sum())
print('\nUnique values :  \n',df.nunique())
df.isnull().sum()
print('Unique Values of SPEEDING: ',df.SPEEDING.unique(),'\n\n')
print('Unique Values of UNDERINFL: ',df.UNDERINFL.unique(),'\n\n')
print('Unique Values of INATTENTIONIND: ',df.INATTENTIONIND.unique())
df.SPEEDING.fillna(value=0,axis=0,inplace=True)
df.SPEEDING.replace(to_replace='Y',value=1,inplace=True)

df.INATTENTIONIND.fillna(value=0,axis=0,inplace=True)
df.INATTENTIONIND.replace(to_replace='Y',value=1,inplace=True)

df.UNDERINFL.replace(to_replace=('Y','N','1','0'),value=(1,0,1,0),inplace=True)

print('SPEEDING unique values: ',df.SPEEDING.unique(),'\n\n')
print('INATTENTIONIND unique values: ',df.INATTENTIONIND.unique(),'\n\n')
print('UNDERINFL unique values:',df.UNDERINFL.unique())
df.isnull().sum()
print('Unique vaues for ADDRTYPE:','\n',df['ADDRTYPE'].unique(),'\n\n\n')
print('Unique vaues for COLLISIONTYPE:','\n',df['COLLISIONTYPE'].unique(),'\n\n\n')
print('Unique vaues for JUNCTIONTYPE:','\n',df['JUNCTIONTYPE'].unique(),'\n\n\n')
print('Unique vaues for WEATHER:','\n',df['WEATHER'].unique(),'\n\n\n')
print('Unique vaues for ROADCOND:','\n',df['ROADCOND'].unique(),'\n\n\n')
print('Unique vaues for LIGHTCOND:','\n',df['LIGHTCOND'].unique(),'\n\n\n')
df.dropna(axis=0,inplace=True)
print('Any null values?','\n', df.isnull().any(),'\n\n')
print('Rows:', df.shape[0])
print('Columns:',df.shape[1])
df['INCDATE']=pd.to_datetime(df['INCDATE'],format='%Y-%m-%d %H:%M:%S')
df['YEAR']=df['INCDATE'].dt.year
df['MONTH']=df['INCDATE'].dt.month
df['DAY']=df['INCDATE'].dt.weekday

df.drop(labels='INCDATE',axis=1,inplace=True)
df.drop(labels='JUNCTIONTYPE',axis=1,inplace=True)

df.head()
%matplotlib inline

df['WEATHER'].value_counts().plot.bar()

plt.title('Number of accidents occured according to the type of Weather')
plt.xlabel('WEATHER Type')
plt.ylabel('Number of accidents')

plt.show()
sns.catplot(x="WEATHER", hue="SEVERITYCODE", kind="count", data=df,height=5,aspect=3)
%matplotlib inline

df['ROADCOND'].value_counts().plot.bar()

plt.title('Number of accidents occured according to road conditions')
plt.xlabel('Road Condition')
plt.ylabel('Number of accidents')

plt.show()
sns.catplot(x="ROADCOND", hue="SEVERITYCODE", kind="count", data=df,height=5,aspect=3)
%matplotlib inline

df['LIGHTCOND'].value_counts().plot.bar()

plt.title('Number of accidents occured according to lighting conditions')
plt.xlabel('Light Condition')
plt.ylabel('Number of accidents')

plt.show()
sns.catplot(x="LIGHTCOND", hue="SEVERITYCODE", kind="count", data=df,height=5,aspect=3)
%matplotlib inline

df['ADDRTYPE'].value_counts().plot.bar()

plt.title('Number of accidents occured according to location')
plt.xlabel('Location')
plt.ylabel('Number of accidents')

plt.show()
%matplotlib inline

df['UNDERINFL'].value_counts().plot.bar()

plt.title('Number of accidents occured due to being under influence')
plt.xlabel('Under Influence')
plt.ylabel('Number of accidents')

plt.show()
%matplotlib inline

df['INATTENTIONIND'].value_counts().plot.bar()

plt.title('Number of accidents occured due to being inattentive')
plt.xlabel('Inattentive')
plt.ylabel('Number of accidents')

plt.show()
%matplotlib inline

df['SPEEDING'].value_counts().plot.bar()

plt.title('Number of accidents occured due to Speeding')
plt.xlabel('Speeding')
plt.ylabel('Number of accidents')

plt.show()
plt.figure(figsize=(20,5))
sns.countplot(x='YEAR',data=df)
plt.title('No. of Accidents per Year')
plt.show()
plt.figure(figsize=(20,5))
sns.countplot(x='MONTH',data=df)
plt.title('No. of Accidents per Month')
plt.show()
plt.figure(figsize=(20,5))
sns.countplot(x='DAY',data=df)
plt.title('No. of Accidents per Day of the Week')
plt.show()
df['UNDERINFL']=df['UNDERINFL'].astype('int64')
df.dtypes
data=pd.get_dummies(df,dtype='int64')
data.columns
#data.drop(['PERSONCOUNT','PEDCOUNT','VEHCOUNT'],axis=1,inplace=True)
data.rename(columns={'ADDRTYPE_Alley':'ALLEY','ADDRTYPE_Block':'BLOCK','ADDRTYPE_Intersection':'INTERSECTION',
                    'COLLISIONTYPE_Angles':'ANGLES','COLLISIONTYPE_Cycles':'CYCLES','COLLISIONTYPE_Head On':'HEAD ON',
                    'COLLISIONTYPE_Left Turn':'LEFT TURN','COLLISIONTYPE_Other':'COLLISION OTHER',
                    'COLLISIONTYPE_Parked Car':'PARKED CAR', 'COLLISIONTYPE_Pedestrian':'PEDESTRIAN',
                    'COLLISIONTYPE_Rear Ended':'REAR ENDED', 'COLLISIONTYPE_Right Turn':'RIGHT TURN',
                    'COLLISIONTYPE_Sideswipe':'SIDESWIPE', 'WEATHER_Blowing Sand/Dirt':'BLOWING SAND/DIRT', 'WEATHER Clear':'CLEAR',
                    'WEATHER_Fog/Smog/Smoke':'FOG/SMOG/SMOKE', 'WEATHER_Other':'WEATHER_OTHER', 'WEATHER_Overcast':'OVERCAST',
                    'WEATHER_Partly Cloudy':'CLOUDY', 'WEATHER_Raining':'RAINING', 'WEATHER_Severe Crosswind':'SEVERE CROSSWIND',
                    'WEATHER_Sleet/Hail/Freezing Rain':'SLEET/HAIL/FREEZING RAIN', 'WEATHER_Snowing':'SNOWING',
                    'WEATHER_Unknown':'WEATHER UNKNOWN', 'ROADCOND_Dry':'DRY', 'ROADCOND_Ice':'ICE', 'ROADCOND_Oil':'OIL',
                    'ROADCOND_Other':'ROADCOND OTHER', 'ROADCOND_Sand/Mud/Dirt':'SAND/MUD/DIRT', 'ROADCOND_Snow/Slush':'SNOW/SLUSH',
                    'ROADCOND_Standing Water':'STANDING WATER', 'ROADCOND_Unknown':'ROADCOND UNKNOWN', 'ROADCOND_Wet':'WET',
                    'LIGHTCOND_Dark - No Street Lights':'DARK - NO STREET LIGHTS',
                    'LIGHTCOND_Dark - Street Lights Off':'DARK - STREET LIGHTS OFF',
                    'LIGHTCOND_Dark - Street Lights On':'DARK - STREET LIGHTS ON',
                    'LIGHTCOND_Dark - Unknown Lighting':'DARK - LIGHTNING', 'LIGHTCOND_Dawn':'DAWN',
                    'LIGHTCOND_Daylight':'DAYLIGHT', 'LIGHTCOND_Dusk':'DUSK', 'LIGHTCOND_Other':'LIGHTCOND OTHER',
                    'LIGHTCOND_Unknown':'LIGHTCOND UNKOWN'},inplace=True)

data.columns
data.SEVERITYCODE.value_counts()
from sklearn.utils import resample

# Separate majority and minority classes
majority = data[data.SEVERITYCODE==1]
minority = data[data.SEVERITYCODE==2]
 
# Downsample majority class
majority_downsampled = resample(majority, 
                                replace=False,    # sample without replacement
                                n_samples=56625,     # to match minority class
                                random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
downsampled = pd.concat([majority_downsampled, minority])
 
# Display new class counts
print('Display downsampled SEVERITYCODE data:','\n',downsampled.SEVERITYCODE.value_counts())
x= np.asarray(downsampled[['SPEEDING', 'INATTENTIONIND', 'UNDERINFL', 'YEAR',
                           'ALLEY', 'BLOCK', 'INTERSECTION', 'ANGLES', 'CYCLES',
                           'HEAD ON', 'LEFT TURN', 'COLLISION OTHER', 'PARKED CAR', 'PEDESTRIAN',
                           'REAR ENDED', 'RIGHT TURN', 'SIDESWIPE', 'BLOWING SAND/DIRT',
                           'WEATHER_Clear', 'FOG/SMOG/SMOKE', 'WEATHER_OTHER', 'OVERCAST',
                           'CLOUDY', 'RAINING', 'SEVERE CROSSWIND', 'SLEET/HAIL/FREEZING RAIN',
                           'SNOWING', 'WEATHER UNKNOWN', 'DRY', 'ICE', 'OIL', 'ROADCOND OTHER',
                           'SAND/MUD/DIRT', 'SNOW/SLUSH', 'STANDING WATER', 'ROADCOND UNKNOWN',
                           'WET', 'DARK - NO STREET LIGHTS', 'DARK - STREET LIGHTS OFF',
                           'DARK - STREET LIGHTS ON', 'DARK - LIGHTNING', 'DAWN', 'DAYLIGHT',
                            'DUSK', 'LIGHTCOND OTHER', 'LIGHTCOND UNKOWN']])

x[:5]
y= np.asarray(downsampled['SEVERITYCODE'])
y[:5]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.1)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression().fit(X_train,y_train)
LR_yhat = LR.predict(X_test)
LR_yhat
from sklearn.metrics import f1_score
import sklearn.metrics as metrics

LR_acc= round(metrics.accuracy_score(y_test, LR_yhat),2)
f1= round(f1_score(y_test,LR_yhat),2)

print('Logistic Regression accuracy: ',LR_acc)
print('Logistic Regression f1 score: ',f1)
from sklearn import metrics,tree
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

Tree = DecisionTreeClassifier()
Tree= Tree.fit(X_train,y_train)

# tree.plot_tree(Tree)
Tree_yhat = Tree.predict(X_test)

f2 = round(f1_score(y_test, Tree_yhat),2)
Tree_acc= round(metrics.accuracy_score(y_test,Tree_yhat),2)

print('Tree accuracy:', Tree_acc)
print('Tree f1 score: ', f2)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors = 20).fit(X_train,y_train)
KNN_yhat = neigh.predict(X_test)

KNN_acc = round(metrics.accuracy_score(y_test, KNN_yhat),2)
f3= round(f1_score(y_test,KNN_yhat),2)

print('KNN accuracy: ', KNN_acc)
print('KNN f1 score: ',f3)
from sklearn.neighbors import KNeighborsClassifier
Ks = 20
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test,yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc
plt.plot(range(1,20),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()
from sklearn import svm
clf = svm.SVC(kernel='rbf')

clf.fit(X_train, y_train) 

SVM_yhat = clf.predict(X_test)

SVM_acc = metrics.accuracy_score(y_test,SVM_yhat)
f4= f1_score(y_test,SVM_yhat)

print('SVM accuracy: ', SVM_acc)
print('SVM f1 score: ',f4)
from sklearn.metrics import jaccard_score
# Logistic Regression
jss1 = round(jaccard_score(y_test, LR_yhat), 2)

# Decision Tree
jss2 = round(jaccard_score(y_test, Tree_yhat), 2)

# KNN
jss3 = round(jaccard_score(y_test, KNN_yhat), 2)

# Support Vector Machine
jss4 = round(jaccard_score(y_test, SVM_yhat), 2)

jss_list = [jss1, jss2, jss3, jss4]
jss_list
f1_list= [f1,f2,f3,round(f4,2)]
acc_list= [LR_acc,Tree_acc,KNN_acc,round(SVM_acc,2)]

columns= ['Logistic Regression','Decision Tree','KNN','SVM']
index= ['Jaccard Score','Model Accuracy','F1 Score']

accuracy_df= pd.DataFrame([jss_list,acc_list,f1_list],index= index,columns=columns)
accuracy_df.head()
accuracy_df1= accuracy_df.transpose()
accuracy_df1.columns.name= 'Algorithm'
accuracy_df1
