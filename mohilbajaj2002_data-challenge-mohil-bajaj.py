import numpy as np
import pandas as pd
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import display

pd.options.display.max_columns = None
# Pre-processing 1: Dataset contains both German and English Text. Converting everything to English.
# Google translate module could also be used for this.

col_eng = ['Master Number','Target Variable', 'Day', 'Month', 'Duration', 'Call Id', 'Age', 'Gender', 
           'Type of Employment', 'Marital Status', 'Education level', 'Credit Failure', 'Bank Balance',
          'House', 'Credit', 'Contact Type', 'No of calls to customer', 'Days since last call',
          'No of calls in last campaign', 'Result of last campaign']

df = pd.read_csv('../input/TrainData.csv',sep=';', names=col_eng , encoding='latin-1')
df = df.drop(df.index[0])
df.head()
print (df['Target Variable'].unique())
print(df['Type of Employment'].unique())
print(df['Marital Status'].unique())
print(df['Education level'].unique())
print(df['Credit Failure'].unique())
print(df['House'].unique())
print(df['Credit'].unique())
print(df['Contact Type'].unique())
print(df['Days since last call'].unique())
print(df['Result of last campaign'].unique())
df = df.replace('nein', 'No')
df = df.replace('ja', 'Yes')

df['Type of Employment'] = df['Type of Employment'].replace('Arbeiter', 'Worker')
df['Type of Employment'] = df['Type of Employment'].replace('Dienstleistung', 'Service')
df['Type of Employment'] = df['Type of Employment'].replace('Arbeitslos', 'Unemployed')
df['Type of Employment'] = df['Type of Employment'].replace('Technischer Beruf', 'Technical profession')
df['Type of Employment'] = df['Type of Employment'].replace('Rentner', 'Pensioner')
df['Type of Employment'] = df['Type of Employment'].replace('Verwaltung', 'Administration')
df['Type of Employment'] = df['Type of Employment'].replace('Gründer', 'Entrepreneur')
df['Type of Employment'] = df['Type of Employment'].replace('Hausfrau', 'Housewife')
df['Type of Employment'] = df['Type of Employment'].replace('Selbständig', 'Independent')
df['Type of Employment'] = df['Type of Employment'].replace('Unbekannt', 'Unknown')

df['Marital Status'] = df['Marital Status'].replace('verheiratet', 'Married')
df['Marital Status'] = df['Marital Status'].replace('geschieden', 'Divorced')

df['Education level'] = df['Education level'].replace('Abitur', 'High School')
df['Education level'] = df['Education level'].replace('Studium', 'Education')
df['Education level'] = df['Education level'].replace('Real-/Hauptschule', 'Middle School Career / secondary school')
df['Education level'] = df['Education level'].replace('Unbekannt', 'Unknown')

df['Contact Type'] = df['Contact Type'].replace('Unbekannt', 'Unknown')
df['Contact Type'] = df['Contact Type'].replace('Handy', 'Handy')
df['Contact Type'] = df['Contact Type'].replace('Festnetz', 'Landline')

df['Result of last campaign'] = df['Result of last campaign'].replace('Unbekannt', 'Unknown')
df['Result of last campaign'] = df['Result of last campaign'].replace('Kein Erfolg', 'No success')
df['Result of last campaign'] = df['Result of last campaign'].replace('Sonstiges', 'Miscellaneous')
df['Result of last campaign'] = df['Result of last campaign'].replace('Erfolg', 'Success')

df.head(10)
# Pre-processing 2: Tackling unknown values + Normalizing numerical values + Making dummy variables for categorical values 
# Base Model - Replace unknown numerical values with mean and unknown categorical values with mode
# Base with Class Imbalance correction - Remvoing columns with high unknown values + Replacing unknown values with 
# mean/mode + Class imbalance correction (at data splitting stage)
# Advanced Model - Using Knn clustering to find clusters and replacing unknown values with mean/mode value of cluster
df2 = df.replace('Unknown', np.nan)
df2 = df2.reset_index(drop=True)
df2.count()
# Base Model
# Tackling unknown values
df_base = df2
df_base2 = df2

col = ['Type of Employment', 'Education level', 'Contact Type', 'Result of last campaign']

for c in col:
    mode = df_base2[c].mode()
    df_base2[c] = df_base2[c].fillna(mode[0])
    
mean = df_base2['Days since last call'].astype(float).mean()
df_base2['Days since last call'] = df_base2['Days since last call'].replace(np.nan, mean)

df_base2.head(10)
# Normalizing numerical values and making dummy variables. Some feature engineering is also performed.

def processing(df2):
    scaler = preprocessing.StandardScaler()

    x = df2['Duration'].values.reshape(-1,1)
    x_scaled = scaler.fit_transform(x)
    df2['Duration'] = x_scaled

    y = df2['Age'].values.reshape(-1,1)
    y_scaled = scaler.fit_transform(y)
    df2['Age'] = y_scaled
    
    u = df2['Bank Balance'].values.reshape(-1,1)
    u_scaled = scaler.fit_transform(u)
    df2['Bank Balance'] = u_scaled
    
    v = df2['No of calls to customer'].values.reshape(-1,1)
    v_scaled = scaler.fit_transform(v)
    df2['No of calls to customer'] = v_scaled
    
    w = df2['No of calls in last campaign'].values.reshape(-1,1)
    w_scaled = scaler.fit_transform(w)
    df2['No of calls in last campaign'] = w_scaled
    
    z = df2['Days since last call'].values.reshape(-1,1)
    z_scaled = scaler.fit_transform(z)
    df2['Days since last call'] = z_scaled

# Making a new feature, Week No. This will also reduce dimensionality by converting days into week.

    df2.Day = df2.Day.astype(float)

    for i in range(0,len(df2)):
        if(df2.loc[i,'Day'] <= 7):
            df2.loc[i,'Week_No'] = 1
        elif(df2.loc[i,'Day'] > 7 and df2.loc[i,'Day'] <= 14):
            df2.loc[i,'Week_No'] = 2
        elif(df2.loc[i,'Day'] > 14 and df2.loc[i,'Day'] <= 21):
            df2.loc[i,'Week_No'] = 3
        elif(df2.loc[i,'Day'] > 21 and df2.loc[i,'Day'] <= 28):
            df2.loc[i,'Week_No'] = 4
        else:
            df2.loc[i,'Week_No'] = 5
    
# Making dummy variables for categorical data

    columns = ['Week_No','Month', 'Gender','Type of Employment', 'Marital Status', 'Education level', 'Credit Failure',
          'House', 'Credit', 'Contact Type', 'Result of last campaign']

    for c in columns:
            dummies = pd.get_dummies(df2[c], prefix = c) 
            df2 = pd.concat([df2, dummies], axis=1)
    
    target = df2['Target Variable'].apply(lambda x: 0 if x=='No' else 1)    
    df2 = df2.drop(['Day','Week_No','Month', 'Gender','Type of Employment', 'Marital Status', 'Education level',
                    'Credit Failure', 'House', 'Credit','Master Number','Target Variable', 'Call Id',
                   'Contact Type', 'Result of last campaign'], axis = 1)
    
    return df2, target
df_base2, target = processing(df_base2)
df_base2.head(10)
# Base model with Class Imbalance correction
# Dropping columns 'Days since last call' and 'Result of last campaign' as they contain too many unknown values. 
# Tackling unknown values inn the rest of the columns similarly to the base model

df_class = df2
df_class = df_class.drop(['Days since last call', 'Result of last campaign'], axis = 1)

col = ['Type of Employment', 'Education level', 'Contact Type']

for c in col:
    mode = df_class[c].mode()
    df_class[c] = df_class[c].fillna(mode[0])
    
df_class.head(10)
scaler = preprocessing.StandardScaler()

x = df_class['Duration'].values.reshape(-1,1)
x_scaled = scaler.fit_transform(x)
df_class['Duration'] = x_scaled

y = df_class['Age'].values.reshape(-1,1)
y_scaled = scaler.fit_transform(y)
df_class['Age'] = y_scaled
    
u = df_class['Bank Balance'].values.reshape(-1,1)
u_scaled = scaler.fit_transform(u)
df_class['Bank Balance'] = u_scaled
    
v = df_class['No of calls to customer'].values.reshape(-1,1)
v_scaled = scaler.fit_transform(v)
df_class['No of calls to customer'] = v_scaled
    
w = df_class['No of calls in last campaign'].values.reshape(-1,1)
w_scaled = scaler.fit_transform(w)
df_class['No of calls in last campaign'] = w_scaled
    
   
 # Feature Engineering - Creating a new feature Week_No from Day.  

df_class.Day = df_class.Day.astype(float)

for i in range(0,len(df_class)):
    if(df_class.loc[i,'Day'] <= 7):
        df_class.loc[i,'Week_No'] = 1
    elif(df_class.loc[i,'Day'] > 7 and df_class.loc[i,'Day'] <= 14):
        df_class.loc[i,'Week_No'] = 2
    elif(df_class.loc[i,'Day'] > 14 and df_class.loc[i,'Day'] <= 21):
        df_class.loc[i,'Week_No'] = 3
    elif(df_class.loc[i,'Day'] > 21 and df_class.loc[i,'Day'] <= 28):
        df_class.loc[i,'Week_No'] = 4
    else:
        df_class.loc[i,'Week_No'] = 5
    
# Making dummy variables for categorical data

columns = ['Week_No','Month', 'Gender','Type of Employment', 'Marital Status', 'Education level', 'Credit Failure',
          'House', 'Credit', 'Contact Type']
          

for c in columns:
        dummies = pd.get_dummies(df_class[c], prefix = c) 
        df_class = pd.concat([df_class, dummies], axis=1)
    
df_class = df_class.drop(['Day','Week_No','Month', 'Gender','Type of Employment', 'Marital Status', 'Education level',
                          'Credit Failure', 'House', 'Credit', 'Contact Type', 'Master Number', 
                          'Target Variable', 'Call Id'], axis = 1)
    
df_class.head() 
# Advanced Model
# As can be seen, a few important features have lots of unknown/nan values. Using clustering technique to divide the
# dataset into clusters and then using avg./most common values in those clusters to fill the unknown/nan values.

df2 = df2.drop(['Master Number','Target Variable','Call Id','Contact Type', 'Days since last call',
               'Result of last campaign'], axis = 1)

#df2.dropna(how='any', inplace = True)
#df2 = df2.reset_index(drop=True)

df2 = df2.replace(np.nan,'Unknown')
print('Total Nan values in the dataset: ', df2.isnull().sum().sum())

df2.head(10)
scaler = preprocessing.StandardScaler()

x = df2['Duration'].values.reshape(-1,1)
x_scaled = scaler.fit_transform(x)
df2['Duration'] = x_scaled

y = df2['Age'].values.reshape(-1,1)
y_scaled = scaler.fit_transform(y)
df2['Age'] = y_scaled
    
u = df2['Bank Balance'].values.reshape(-1,1)
u_scaled = scaler.fit_transform(u)
df2['Bank Balance'] = u_scaled
    
v = df2['No of calls to customer'].values.reshape(-1,1)
v_scaled = scaler.fit_transform(v)
df2['No of calls to customer'] = v_scaled
    
w = df2['No of calls in last campaign'].values.reshape(-1,1)
w_scaled = scaler.fit_transform(w)
df2['No of calls in last campaign'] = w_scaled
    
   
 # Knn works well with low dimensional data. Converting days into a new feature, Week No. This will hopefully,
#  also help with our main classification model. 

df2.Day = df2.Day.astype(float)

for i in range(0,len(df2)):
    if(df2.loc[i,'Day'] <= 7):
        df2.loc[i,'Week_No'] = 1
    elif(df2.loc[i,'Day'] > 7 and df2.loc[i,'Day'] <= 14):
        df2.loc[i,'Week_No'] = 2
    elif(df2.loc[i,'Day'] > 14 and df2.loc[i,'Day'] <= 21):
        df2.loc[i,'Week_No'] = 3
    elif(df2.loc[i,'Day'] > 21 and df2.loc[i,'Day'] <= 28):
        df2.loc[i,'Week_No'] = 4
    else:
        df2.loc[i,'Week_No'] = 5
    
# Making dummy variables for categorical data

columns = ['Week_No','Month', 'Gender','Type of Employment', 'Marital Status', 'Education level', 'Credit Failure',
          'House', 'Credit']

for c in columns:
        dummies = pd.get_dummies(df2[c], prefix = c) 
        df2 = pd.concat([df2, dummies], axis=1)
    
df2 = df2.drop(['Day','Week_No','Month', 'Gender','Type of Employment', 'Marital Status', 'Education level',
                    'Credit Failure', 'House', 'Credit'], axis = 1)
    
df2.head()     
# Developing knn model and finding the right k

distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k, max_iter=2500, algorithm = 'auto')
    kmeanModel.fit(df2.values)
    distortions.append(sum(np.min(cdist(df2.values, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df2.values.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
# As can be seen from the graph above, k=6 looks like the optimal value of k. Proceeding with using this value for the 
# Knn model. The model is used to develop a new feature 'Data Group' and fill the unknown values.

theModel = KMeans(n_clusters=6, max_iter=2500, algorithm = 'auto')
theModel.fit(df2.values)

group = []
centroid = []

group = theModel.predict(df2.values)
df2['Data Group'] = group

centroids = theModel.cluster_centers_
     
df2.head()
# Filling up the unknown values

df_adv = df
df_adv = df_adv.reset_index(drop=True)
df_adv = df_adv.replace('Unknown', np.nan)
df_adv['Data Group'] = df2['Data Group']

grp0 = df_adv[df_adv['Data Group']==0]
grp1 = df_adv[df_adv['Data Group']==1]
grp2 = df_adv[df_adv['Data Group']==2]
grp3 = df_adv[df_adv['Data Group']==3]
grp4 = df_adv[df_adv['Data Group']==4]
grp5 = df_adv[df_adv['Data Group']==5]

col = ['Type of Employment', 'Education level', 'Contact Type', 'Result of last campaign']

for i in range(0,len(df_adv)):
    for c in col:
        if (df_adv.loc[i,c]!= df_adv.loc[i,c]):
            if (df_adv.loc[i,'Data Group'] == 0):
                mode = grp0[c].mode()
                df_adv.loc[i,c] = mode[0]
            elif (df_adv.loc[i,'Data Group'] == 1):
                mode = grp1[c].mode()
                df_adv.loc[i,c] = mode[0]
            elif (df_adv.loc[i,'Data Group'] == 2):
                mode = grp2[c].mode()
                df_adv.loc[i,c] = mode[0]
            elif (df_adv.loc[i,'Data Group'] == 3):
                mode = grp3[c].mode()
                df_adv.loc[i,c] = mode[0]
            elif (df_adv.loc[i,'Data Group'] == 4):
                mode = grp4[c].mode()
                df_adv.loc[i,c] = mode[0]
            elif (df_adv.loc[i,'Data Group'] == 5):
                mode = grp5[c].mode()
                df_adv.loc[i,c] = mode[0]
    
    
for i in range(0,len(df_adv)):
    if (df_adv.loc[i,'Days since last call']!= df_adv.loc[i,'Days since last call']):
        if (df_adv.loc[i,'Data Group'] == 0):
            df_adv.loc[i,'Days since last call'] = grp0['Days since last call'].astype(float).mean()
        elif (df_adv.loc[i,'Data Group'] == 1):
            df_adv.loc[i,'Days since last call'] = grp1['Days since last call'].astype(float).mean()
        elif (df_adv.loc[i,'Data Group'] == 2):
            df_adv.loc[i,'Days since last call'] = grp2['Days since last call'].astype(float).mean()
        elif (df_adv.loc[i,'Data Group'] == 3):
            df_adv.loc[i,'Days since last call'] = grp3['Days since last call'].astype(float).mean()
        elif (df_adv.loc[i,'Data Group'] == 4):
            df_adv.loc[i,'Days since last call'] = grp4['Days since last call'].astype(float).mean()
        elif (df_adv.loc[i,'Data Group'] == 5):
            df_adv.loc[i,'Days since last call'] = grp5['Days since last call'].astype(float).mean()
    
print('Total Nan values in the dataset: ', df_adv.isnull().sum().sum())

df_adv.head(10)
df_adv.head(10)
scaler = preprocessing.StandardScaler()

x = df_adv['Duration'].values.reshape(-1,1)
x_scaled = scaler.fit_transform(x)
df_adv['Duration'] = x_scaled

y = df_adv['Age'].values.reshape(-1,1)
y_scaled = scaler.fit_transform(y)
df_adv['Age'] = y_scaled
    
u = df_adv['Bank Balance'].values.reshape(-1,1)
u_scaled = scaler.fit_transform(u)
df_adv['Bank Balance'] = u_scaled
    
v = df_adv['No of calls to customer'].values.reshape(-1,1)
v_scaled = scaler.fit_transform(v)
df_adv['No of calls to customer'] = v_scaled
    
w = df_adv['No of calls in last campaign'].values.reshape(-1,1)
w_scaled = scaler.fit_transform(w)
df_adv['No of calls in last campaign'] = w_scaled

z = df_adv['Days since last call'].values.reshape(-1,1)
z_scaled = scaler.fit_transform(z)
df_adv['Days since last call'] = z_scaled
    
   
 # Converting days into a new feature, Week No.  

df_adv.Day = df_adv.Day.astype(float)

for i in range(0,len(df_adv)):
    if(df_adv.loc[i,'Day'] <= 7):
        df_adv.loc[i,'Week_No'] = 1
    elif(df_adv.loc[i,'Day'] > 7 and df_adv.loc[i,'Day'] <= 14):
        df_adv.loc[i,'Week_No'] = 2
    elif(df_adv.loc[i,'Day'] > 14 and df_adv.loc[i,'Day'] <= 21):
        df_adv.loc[i,'Week_No'] = 3
    elif(df_adv.loc[i,'Day'] > 21 and df_adv.loc[i,'Day'] <= 28):
        df_adv.loc[i,'Week_No'] = 4
    else:
        df_adv.loc[i,'Week_No'] = 5
    
# Making dummy variables for categorical data

columns = ['Week_No','Month', 'Gender','Type of Employment', 'Marital Status', 'Education level', 'Credit Failure',
          'House', 'Credit', 'Contact Type', 'Result of last campaign', 'Data Group']

for c in columns:
        dummies = pd.get_dummies(df_adv[c], prefix = c) 
        df_adv = pd.concat([df_adv, dummies], axis=1)
    
df_adv = df_adv.drop(['Day','Week_No','Month', 'Gender','Type of Employment', 'Marital Status', 'Education level',
                'Credit Failure', 'House', 'Credit','Master Number','Target Variable', 'Call Id',
                'Contact Type', 'Result of last campaign', 'Data Group'], axis = 1)
    
df_adv.head() 
df_adv.shape
# Exploring the dataset to understand the inherent features in the data. 
df_base['Target Variable'] = df_base['Target Variable'].apply(lambda x: 0 if x=='No' else 1)
# Checking if the dataset is imbalanced

data_balance = 100*len(df_base[df_base['Target Variable']==1])/len(df_base)
print('Dataset Imbalance: ', data_balance, '%')
# Next, checking the correlation between various features

corr = df_base.corr()
plot1 = sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)
plot1.figure.set_size_inches(12,8)
plot1.axes.set_title("Pearson's Correlation Matrix", fontsize=24,color="r",alpha=0.5)
# As can be seen from the above heat map, apart from Duration there is almost no direct correlation between the target  
# variable and the features. This suggests a more complex relationship between features. To understand more about this 
# complex nature, graphs are plotted taking two features at a time.
df_base.head()
df_base['Duration'] = df_base['Duration'].astype(float)
df_base['Duration'].describe()
df_base['Age'] = df_base['Age'].astype(float)
df_base['Age'].describe()
df_base['Bank Balance'] = df_base['Bank Balance'].astype(float)
df_base['Bank Balance'].describe()
def week(x):
    x = int(x)

    if(x <= 7):
        return 1
    elif(x > 7 and x <= 14):
        return 2
    elif(x > 14 and x <= 21):
        return 3
    elif(x > 21 and x <= 28):
        return 4
    else:
        return 5
    
df_base['Week_No'] = df_base['Day'].apply(lambda x: week(x))
def duration(x):
    
    if(x <= 30):
        return 'Less than 30 sec'
    elif(x > 30 and x <= 60):
        return 'Less than 1 min'
    elif(x > 60 and x <= 180):
        return 'Less than 3 min'
    elif(x > 180 and x <= 300):
        return 'Less than 5 min'
    elif(x > 300 and x <= 600):
        return 'Less than 10 min'
    else:
        return 'Greater than 10 min'
    
df_base['Duration_Group'] = df_base['Duration'].apply(lambda x: duration(x))
def age(x):
    
    if(x <= 20):
        return 'Less than 20'
    elif(x > 20 and x <= 30):
        return '20-30'
    elif(x > 30 and x <= 40):
        return '30-40'
    elif(x > 40 and x <= 50):
        return '40-50'
    elif(x > 50 and x <= 60):
        return '50-60'
    else:
        return 'Greater than 60'
    
df_base['Age_Group'] = df_base['Age'].apply(lambda x: age(x))
def bankbalance(x):
    
    if(x <= -250):
        return 'Less than -250'
    elif(x > -250 and x <= 250):
        return 'Less than 250'
    elif(x > 250 and x <= 750):
        return 'Less than 750'
    elif(x > 750 and x <= 1250):
        return 'Less than 1250'
    elif(x > 1250 and x <= 1750):
        return 'Less than 1750'
    elif(x > 1750 and x <= 2500):
        return 'Less than 2500'
    else:
        return 'Greater than 2500'
    
df_base['Bank_Balance_Group'] = df_base['Bank Balance'].apply(lambda x: bankbalance(x))
df_base.head(10)
# Gender and Week_No

df_gender = df_base['Target Variable'].groupby([df_base['Gender'], df_base['Week_No']])
a = df_gender.sum().unstack()
b = a.loc[['w']].T
c = a.loc[['m']].T

width = .35
index = np.arange(len(b))
plt.bar(index,b['w'],width, label='Female')
plt.bar (index+width,c['m'],width, label='Male')
plt.xlabel('Week_No', fontsize=10)
plt.ylabel('Total Positive Outcomes', fontsize=10)
plt.xticks(index+width/2, index, fontsize=10)
plt.title('Total positive outcomes for different gender types over the month')
plt.legend(loc = 'best')
plt.show()
# Gender and Contact Type

df_gender = df_base['Target Variable'].groupby([df_base['Gender'], df_base['Contact Type']])
a = df_gender.sum().unstack()
b = a.loc[['w']].T
c = a.loc[['m']].T

width = .35
index = np.arange(len(b))
plt.bar(index,b['w'],width, label='Female')
plt.bar (index+width,c['m'],width, label='Male')
plt.xlabel('Contact Type', fontsize=10)
plt.ylabel('Total Positive Outcomes', fontsize=10)
plt.xticks(index+width/2, index, fontsize=10)
plt.title('Total positive outcomes for different gender types with different contact methods')
plt.legend(loc = 'best')
plt.show()
# Gender and Call Duration Group

df_gender = df_base['Target Variable'].groupby([df_base['Gender'], df_base['Duration_Group']])
a = df_gender.sum().unstack()
b = a.loc[['w']].T
c = a.loc[['m']].T

width = .35
index = np.arange(len(b))
plt.bar(index,b['w'],width, label='Female')
plt.bar (index+width,c['m'],width, label='Male')
plt.xlabel('Duration_Group', fontsize=10)
plt.ylabel('Total Positive Outcomes', fontsize=10)
plt.xticks(index+width/2, index, fontsize=10)
plt.title('Total positive outcomes for different gender types over different call duration')
plt.legend(loc = 'best')
plt.show()
# Gender and Education background

df_gender = df_base['Target Variable'].groupby([df_base['Gender'], df_base['Education level']])
a = df_gender.sum().unstack()
b = a.loc[['w']].T
c = a.loc[['m']].T

width = .35
index = np.arange(len(b))
plt.bar(index,b['w'],width, label='Female')
plt.bar (index+width,c['m'],width, label='Male')
plt.xlabel('Education level', fontsize=10)
plt.ylabel('Total Positive Outcomes', fontsize=10)
plt.xticks(index+width/2, index, fontsize=10)
plt.title('Total positive outcomes for different gender types with different education levels')
plt.legend(loc = 'best')
plt.show()
# Gender and Employment Types

df_gender = df_base['Target Variable'].groupby([df_base['Gender'], df_base['Type of Employment']])
a = df_gender.sum().unstack()
b = a.loc[['w']].T
c = a.loc[['m']].T

width = .35
index = np.arange(len(b))
plt.bar(index,b['w'],width, label='Female')
plt.bar (index+width,c['m'],width, label='Male')
plt.xlabel('Type of Employment', fontsize=10)
plt.ylabel('Total Positive Outcomes', fontsize=10)
plt.xticks(index+width/2, index, fontsize=10)
plt.title('Total positive outcomes for different gender types with different employment types')
plt.legend(loc = 'best')
plt.show()
# Gender and Age Group

df_gender = df_base['Target Variable'].groupby([df_base['Gender'], df_base['Age_Group']])
a = df_gender.sum().unstack()
b = a.loc[['w']].T
c = a.loc[['m']].T

width = .35
index = np.arange(len(b))
plt.bar(index,b['w'],width, label='Female')
plt.bar (index+width,c['m'],width, label='Male')
plt.xlabel('Age_Group', fontsize=10)
plt.ylabel('Total Positive Outcomes', fontsize=10)
plt.xticks(index+width/2, index, fontsize=10)
plt.title('Total positive outcomes for different gender types over different age groups')
plt.legend(loc = 'best')
plt.show()
# Gender and Bank Balance Group

df_gender = df_base['Target Variable'].groupby([df_base['Gender'], df_base['Bank_Balance_Group']])
a = df_gender.sum().unstack()
b = a.loc[['w']].T
c = a.loc[['m']].T

width = .35
index = np.arange(len(b))
plt.bar(index,b['w'],width, label='Female')
plt.bar (index+width,c['m'],width, label='Male')
plt.xlabel('Bank_Balance_Group', fontsize=10)
plt.ylabel('Total Positive Outcomes', fontsize=10)
plt.xticks(index+width/2, index, fontsize=10)
plt.title('Total positive outcomes for different gender types over different bank balance groups')
plt.legend(loc = 'best')
plt.show()
# Marital Status and Credit Failure

df_marital = df_base['Target Variable'].groupby([df_base['Marital Status'], df_base['Credit Failure']])
a = df_marital.sum().unstack()
b = a.loc[['single']].T
c = a.loc[['Married']].T
d = a.loc[['Divorced']].T

width = .2
empty = [0,0]
index = np.arange(len(b))
plt.figure(figsize=(25, 12))
plt.bar(index, b['single'], width, align='edge',label='Single')
plt.bar(index+width, c['Married'], width, align='edge', label='Married')
plt.bar(index+2*width, d['Divorced'], width, align='edge', label='Divorced')
plt.bar(index+3*width, empty, width, align='edge')

plt.xlabel('Credit Failure', fontsize=10)
plt.ylabel('Total Positive Outcomes', fontsize=10)
plt.xticks(index+1.5*width, index, fontsize=10)
plt.title('Total positive outcomes for different marital status types and credit failure history')
plt.legend(loc = 'best')
plt.show()
# Marital Status and Home

df_marital = df_base['Target Variable'].groupby([df_base['Marital Status'], df_base['House']])
a = df_marital.sum().unstack()
b = a.loc[['single']].T
c = a.loc[['Married']].T
d = a.loc[['Divorced']].T

width = .2
empty = [0,0]
index = np.arange(len(b))
plt.figure(figsize=(25, 12))
plt.bar(index, b['single'], width, align='edge',label='Single')
plt.bar(index+width, c['Married'], width, align='edge', label='Married')
plt.bar(index+2*width, d['Divorced'], width, align='edge', label='Divorced')
plt.bar(index+3*width, empty, width, align='edge')

plt.xlabel('Home', fontsize=10)
plt.ylabel('Total Positive Outcomes', fontsize=10)
plt.xticks(index+1.5*width, index, fontsize=10)
plt.title('Total positive outcomes for different marital status types and history of home purchase')
plt.legend(loc = 'best')
plt.show()
# Marital Status and Credit

df_marital = df_base['Target Variable'].groupby([df_base['Marital Status'], df_base['Credit']])
a = df_marital.sum().unstack()
b = a.loc[['single']].T
c = a.loc[['Married']].T
d = a.loc[['Divorced']].T

width = .2
empty = [0,0]
index = np.arange(len(b))
plt.figure(figsize=(25, 12))
plt.bar(index, b['single'], width, align='edge',label='Single')
plt.bar(index+width, c['Married'], width, align='edge', label='Married')
plt.bar(index+2*width, d['Divorced'], width, align='edge', label='Divorced')
plt.bar(index+3*width, empty, width, align='edge')

plt.xlabel('Credit', fontsize=10)
plt.ylabel('Total Positive Outcomes', fontsize=10)
plt.xticks(index+1.5*width, index, fontsize=10)
plt.title('Total positive outcomes for different marital status types and history of credit')
plt.legend(loc = 'best')
plt.show()
# Marital Status and Month

df_marital = df_base['Target Variable'].groupby([df_base['Marital Status'], df_base['Month']])
a = df_marital.sum().unstack()
b = a.loc[['single']].T
c = a.loc[['Married']].T
d = a.loc[['Divorced']].T

width = .2
empty = [0,0,0,0,0,0,0,0,0,0,0,0]
index = np.arange(len(b))
plt.figure(figsize=(25, 12))
plt.bar(index, b['single'], width, align='edge',label='Single')
plt.bar(index+width, c['Married'], width, align='edge', label='Married')
plt.bar(index+2*width, d['Divorced'], width, align='edge', label='Divorced')
plt.bar(index+3*width, empty, width, align='edge')

plt.xlabel('Month', fontsize=10)
plt.ylabel('Total Positive Outcomes', fontsize=10)
plt.xticks(index+1.5*width, index, fontsize=10)
plt.title('Total positive outcomes for different marital status types over the course of the year')
plt.legend(loc = 'best')
plt.show()


# Building Base models from Logistic Regression, Random Forest and XGBoost. Ensembling tried as well. 

train_features, test_features, train_labels, test_labels = train_test_split(df_base2, target, test_size = 0.30, 
                                                                            random_state = 42)

test_labels = test_labels.values
# Logistic Regression 

param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

lr = LogisticRegression(penalty='l2')

grid_search_lr = GridSearchCV(estimator = lr, param_grid = param_grid_lr, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search_lr.fit(train_features, train_labels)

print(grid_search_lr.best_params_)

best_base_lr = grid_search_lr.best_estimator_

predictions_base_lr = best_base_lr.predict(test_features)


print ('Logistic Regression (Base Model)')
print ("Validation Accuracy: ", accuracy_score(test_labels, predictions_base_lr))
print ('Auc Score: ', roc_auc_score(test_labels, predictions_base_lr))
print ('F1 score: ', f1_score(test_labels, predictions_base_lr, average='micro'))
# Random Forest 

param_grid = {
    'max_depth': [1, 5, 25, 50],
    'max_features': ['auto'],
    'min_samples_leaf': [300, 500, 1000],
    'n_estimators': [100, 250, 500, 1000]}

rf = RandomForestClassifier()

grid_search_base_rf = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search_base_rf.fit(train_features, train_labels)

print(grid_search_base_rf.best_params_)

best_base_rf = grid_search_base_rf.best_estimator_

predictions_base_rf = best_base_rf.predict(test_features)

print ('Random Forest (Base Model)')
print ("Validation Accuracy: ", accuracy_score(test_labels, predictions_base_rf))
print ('Auc Score: ', roc_auc_score(test_labels, predictions_base_rf))
print ('F1 score: ', f1_score(test_labels, predictions_base_rf, average='micro'))
# XGBoost

model_xgb = xgb.XGBClassifier(colsample_bytree=0.2, gamma=0.0, 
                            learning_rate=0.05, max_depth=6, 
                            min_child_weight=1.5, n_estimators=7200,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2,seed=42, silent=1,
                            random_state =7)

model_xgb.fit(train_features, train_labels)

predictions_base_xgb = model_xgb.predict(test_features)


print ('XGBoost (Base Model)')
print ("Validation Accuracy: ", accuracy_score(test_labels, predictions_base_xgb))
print ('Auc Score: ', roc_auc_score(test_labels, predictions_base_xgb))
print ('F1 score: ', f1_score(test_labels, predictions_base_xgb, average='micro'))
# Understanding the features which had the most impact on the XGB model result

#best_feature = pd.DataFrame()
#best_feature['Column Name'] = df_base.columns
#best_feature['Importance_xgb'] = model_xgb.feature_importances_
#best_feature = best_feature.sort_values('Importance_xgb', ascending=False )
#best_feature.head(20)
# Ensembling - Simple Average of above models

ensemble = pd.DataFrame()
ensemble['LR'] = best_base_lr.predict_proba(test_features)[:,1]
ensemble['RF'] = best_base_rf.predict_proba(test_features)[:,1]
ensemble['XGB'] = model_xgb.predict_proba(test_features)[:,1]

ensemble['ENS'] = (ensemble['LR'] + ensemble['RF'] + ensemble['XGB'])/3
ensemble['PRED'] = ensemble['ENS'].apply(lambda x: 0 if x<0.5 else 1)
predictions_base_ens = ensemble['PRED'].values

print ('Ensemble (Base Model)')
print ("Validation Accuracy: ", accuracy_score(test_labels, predictions_base_ens))
print ('Auc Score: ', roc_auc_score(test_labels, predictions_base_ens))
print ('F1 score: ', f1_score(test_labels, predictions_base_ens, average='micro'))
# Building Base models with Class Imbalance rectification (using SMOTE)
train_features2, test_features, train_labels2, test_labels = train_test_split(df_class, target, test_size = 0.30, 
                                                                            random_state = 42)

test_labels = test_labels.values
sm = SMOTE(random_state=12, ratio = 1.0)
train_features, train_labels = sm.fit_sample(train_features2, train_labels2)
len(train_features)
# Logistic Regression 

param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

lr = LogisticRegression(penalty='l2')

grid_search_class_lr = GridSearchCV(estimator = lr, param_grid = param_grid_lr, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search_class_lr.fit(train_features, train_labels)

print(grid_search_class_lr.best_params_)

best_class_lr = grid_search_class_lr.best_estimator_

predictions_class_lr = best_class_lr.predict(test_features)


print ('Logistic Regression (Base with Class Rec. Model)')
print ("Validation Accuracy: ", accuracy_score(test_labels, predictions_class_lr))
print ('Auc Score: ', roc_auc_score(test_labels, predictions_class_lr))
print ('F1 score: ', f1_score(test_labels, predictions_class_lr, average='micro'))
# Random Forest 

param_grid = {
    'max_depth': [1, 5, 25, 50],
    'max_features': ['auto'],
    'min_samples_leaf': [300, 500, 1000],
    'n_estimators': [100, 250, 500, 1000]}

rf = RandomForestClassifier()

grid_search_class_rf = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search_class_rf.fit(train_features, train_labels)

print(grid_search_class_rf.best_params_)

best_class_rf = grid_search_class_rf.best_estimator_

predictions_class_rf = best_class_rf.predict(test_features)

print ('Random Forest (Base with Class Rec. Model)')
print ("Validation Accuracy: ", accuracy_score(test_labels, predictions_class_rf))
print ('Auc Score: ', roc_auc_score(test_labels, predictions_class_rf))
print ('F1 score: ', f1_score(test_labels, predictions_class_rf, average='micro'))
# Understanding the features which had the most impact on the Random Forest model result

best_feature_class = pd.DataFrame()
best_feature_class['Column Name'] = df_class.columns
best_feature_class['Importance_rf'] = best_class_rf.feature_importances_
best_feature_class = best_feature_class.sort_values('Importance_rf', ascending=False )
best_feature_class.head(20)
# XGBoost

model_class_xgb = xgb.XGBClassifier(colsample_bytree=0.2, gamma=0.0, 
                            learning_rate=0.05, max_depth=6, 
                            min_child_weight=1.5, n_estimators=7200,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2,seed=42, silent=1,
                            random_state =7)

train_features1 = pd.DataFrame(train_features)
train_features1.columns = test_features.columns

model_class_xgb.fit(train_features1, train_labels)

predictions_class_xgb = model_class_xgb.predict(test_features)


print ('XGBoost (Base with Class Rec. Model)')
print ("Validation Accuracy: ", accuracy_score(test_labels, predictions_class_xgb))
print ('Auc Score: ', roc_auc_score(test_labels, predictions_class_xgb))
print ('F1 score: ', f1_score(test_labels, predictions_class_xgb, average='micro'))


# Ensembling - Simple Average of above models

ensemble_class = pd.DataFrame()
ensemble_class['LR'] = best_class_lr.predict_proba(test_features)[:,1]
ensemble_class['RF'] = best_class_rf.predict_proba(test_features)[:,1]
ensemble_class['XGB'] = model_class_xgb.predict_proba(test_features)[:,1]

ensemble_class['ENS'] = (ensemble_class['LR'] + ensemble_class['RF'] + ensemble_class['XGB'])/3
ensemble_class['PRED'] = ensemble_class['ENS'].apply(lambda x: 0 if x<0.5 else 1)
predictions_class_ens = ensemble_class['PRED'].values

print ('Ensemble (Base with Class Rec. Model)')
print ("Validation Accuracy: ", accuracy_score(test_labels, predictions_class_ens))
print ('Auc Score: ', roc_auc_score(test_labels, predictions_class_ens))
print ('F1 score: ', f1_score(test_labels, predictions_class_ens, average='micro'))

# Building Advanced models from Logistic Regression, Random Forest and XGBoost along with Ensembling.
# Building Advanced models from Logistic Regression, Random Forest and XGBoost along with Ensembling.

train_features, test_features, train_labels, test_labels = train_test_split(df_adv, target, test_size = 0.30, 
                                                                            random_state = 42)

test_labels = test_labels.values
# Logistic Regression 

param_grid_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

lr = LogisticRegression(penalty='l2')

grid_search_adv_lr = GridSearchCV(estimator = lr, param_grid = param_grid_lr, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search_adv_lr.fit(train_features, train_labels)

print(grid_search_adv_lr.best_params_)

best_adv_lr = grid_search_adv_lr.best_estimator_

predictions_adv_lr = best_adv_lr.predict(test_features)


print ('Logistic Regression (Advance Model)')
print ("Validation Accuracy: ", accuracy_score(test_labels, predictions_adv_lr))
print ('Auc Score: ', roc_auc_score(test_labels, predictions_adv_lr))
print ('F1 score: ', f1_score(test_labels, predictions_adv_lr, average='micro'))
# Random Forest 

param_grid = {
    'max_depth': [1, 5, 25, 50],
    'max_features': ['auto'],
    'min_samples_leaf': [300, 500, 1000],
    'n_estimators': [100, 250, 500, 1000]}

rf = RandomForestClassifier()

grid_search_adv_rf = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

grid_search_adv_rf.fit(train_features, train_labels)

print(grid_search_adv_rf.best_params_)

best_adv_rf = grid_search_adv_rf.best_estimator_

predictions_adv_rf = best_adv_rf.predict(test_features)

print ('Random Forest (Advance Model)')
print ("Validation Accuracy: ", accuracy_score(test_labels, predictions_adv_rf))
print ('Auc Score: ', roc_auc_score(test_labels, predictions_adv_rf))
print ('F1 score: ', f1_score(test_labels, predictions_adv_rf, average='micro'))
# XGBoost

model_adv_xgb = xgb.XGBClassifier(colsample_bytree=0.2, gamma=0.0, 
                            learning_rate=0.05, max_depth=6, 
                            min_child_weight=1.5, n_estimators=7200,
                             reg_alpha=0.9, reg_lambda=0.6,
                             subsample=0.2,seed=42, silent=1,
                            random_state =7)

model_adv_xgb.fit(train_features, train_labels)

predictions_adv_xgb = model_adv_xgb.predict(test_features)


print ('XGBoost (Advance Model)')
print ("Validation Accuracy: ", accuracy_score(test_labels, predictions_adv_xgb))
print ('Auc Score: ', roc_auc_score(test_labels, predictions_adv_xgb))
print ('F1 score: ', f1_score(test_labels, predictions_adv_xgb, average='micro'))
# Ensembling - Simple Average of above models

ensemble_adv = pd.DataFrame()
ensemble_adv['LR'] = best_adv_lr.predict_proba(test_features)[:,1]
ensemble_adv['RF'] = best_adv_rf.predict_proba(test_features)[:,1]
ensemble_adv['XGB'] = model_adv_xgb.predict_proba(test_features)[:,1]

ensemble_adv['ENS'] = (ensemble_adv['LR'] + ensemble_adv['RF'] + ensemble_adv['XGB'])/3
ensemble_adv['PRED'] = ensemble_adv['ENS'].apply(lambda x: 0 if x<0.5 else 1)
predictions_adv_ens = ensemble_adv['PRED'].values

print ('Ensemble (Advance Model)')
print ("Validation Accuracy: ", accuracy_score(test_labels, predictions_adv_ens))
print ('Auc Score: ', roc_auc_score(test_labels, predictions_adv_ens))
print ('F1 score: ', f1_score(test_labels, predictions_adv_ens, average='micro'))
# It can be seen from the above that Random Forest (Base Model with Class Rect) performed the best in our analysis. 
# Using it to make predictions on our test data.  
col_eng = ['Master Number','Target Variable', 'Day', 'Month', 'Duration', 'Call Id', 'Age', 'Gender', 
           'Type of Employment', 'Marital Status', 'Education level', 'Credit Failure', 'Bank Balance',
          'House', 'Credit', 'Contact Type', 'No of calls to customer', 'Days since last call',
          'No of calls in last campaign', 'Result of last campaign']

test = pd.read_csv('../input/TestData.csv',sep=';', names=col_eng , encoding='latin-1')
test = test.drop(test.index[0])
test.head()
test = test.replace('nein', 'No')
test = test.replace('ja', 'Yes')

test['Type of Employment'] = test['Type of Employment'].replace('Arbeiter', 'Worker')
test['Type of Employment'] = test['Type of Employment'].replace('Dienstleistung', 'Service')
test['Type of Employment'] = test['Type of Employment'].replace('Arbeitslos', 'Unemployed')
test['Type of Employment'] = test['Type of Employment'].replace('Technischer Beruf', 'Technical profession')
test['Type of Employment'] = test['Type of Employment'].replace('Rentner', 'Pensioner')
test['Type of Employment'] = test['Type of Employment'].replace('Verwaltung', 'Administration')
test['Type of Employment'] = test['Type of Employment'].replace('Gründer', 'Entrepreneur')
test['Type of Employment'] = test['Type of Employment'].replace('Hausfrau', 'Housewife')
test['Type of Employment'] = test['Type of Employment'].replace('Selbständig', 'Independent')
test['Type of Employment'] = test['Type of Employment'].replace('Unbekannt', 'Unknown')

test['Marital Status'] = test['Marital Status'].replace('verheiratet', 'Married')
test['Marital Status'] = test['Marital Status'].replace('geschieden', 'Divorced')

test['Education level'] = test['Education level'].replace('Abitur', 'High School')
test['Education level'] = test['Education level'].replace('Studium', 'Education')
test['Education level'] = test['Education level'].replace('Real-/Hauptschule', 'Middle School Career / secondary school')
test['Education level'] = test['Education level'].replace('Unbekannt', 'Unknown')

test['Contact Type'] = test['Contact Type'].replace('Unbekannt', 'Unknown')
test['Contact Type'] = test['Contact Type'].replace('Handy', 'Handy')
test['Contact Type'] = test['Contact Type'].replace('Festnetz', 'Landline')

test['Result of last campaign'] = test['Result of last campaign'].replace('Unbekannt', 'Unknown')
test['Result of last campaign'] = test['Result of last campaign'].replace('Kein Erfolg', 'No success')
test['Result of last campaign'] = test['Result of last campaign'].replace('Sonstiges', 'Miscellaneous')
test['Result of last campaign'] = test['Result of last campaign'].replace('Erfolg', 'Success')

test.head(10)
test = test.replace('Unknown', np.nan)
test = test.reset_index(drop=True)
test.count()
test = test.drop(['Days since last call', 'Result of last campaign'], axis = 1)

col = ['Type of Employment', 'Education level', 'Contact Type']

for c in col:
    mode = test[c].mode()
    test[c] = test[c].fillna(mode[0])

test.head(10)
test2 = test

scaler = preprocessing.StandardScaler()

x = test2['Duration'].values.reshape(-1,1)
x_scaled = scaler.fit_transform(x)
test2['Duration'] = x_scaled

y = test2['Age'].values.reshape(-1,1)
y_scaled = scaler.fit_transform(y)
test2['Age'] = y_scaled
    
u = test2['Bank Balance'].values.reshape(-1,1)
u_scaled = scaler.fit_transform(u)
test2['Bank Balance'] = u_scaled
    
v = test2['No of calls to customer'].values.reshape(-1,1)
v_scaled = scaler.fit_transform(v)
test2['No of calls to customer'] = v_scaled
    
w = test2['No of calls in last campaign'].values.reshape(-1,1)
w_scaled = scaler.fit_transform(w)
test2['No of calls in last campaign'] = w_scaled
    
   
 # Feature Engineering - Creating a new feature Week_No from Day.  

test2.Day = test2.Day.astype(float)

for i in range(0,len(test2)):
    if(test2.loc[i,'Day'] <= 7):
        test2.loc[i,'Week_No'] = 1
    elif(test2.loc[i,'Day'] > 7 and test2.loc[i,'Day'] <= 14):
        test2.loc[i,'Week_No'] = 2
    elif(test2.loc[i,'Day'] > 14 and test2.loc[i,'Day'] <= 21):
        test2.loc[i,'Week_No'] = 3
    elif(test2.loc[i,'Day'] > 21 and test2.loc[i,'Day'] <= 28):
        test2.loc[i,'Week_No'] = 4
    else:
        test2.loc[i,'Week_No'] = 5
    
# Making dummy variables for categorical data

columns = ['Week_No','Month', 'Gender','Type of Employment', 'Marital Status', 'Education level', 'Credit Failure',
          'House', 'Credit', 'Contact Type']
          

for c in columns:
        dummies = pd.get_dummies(test2[c], prefix = c) 
        test2 = pd.concat([test2, dummies], axis=1)
    
test2 = test2.drop(['Day','Week_No','Month', 'Gender','Type of Employment', 'Marital Status', 'Education level',
                          'Credit Failure', 'House', 'Credit', 'Contact Type', 'Master Number', 
                          'Target Variable', 'Call Id'], axis = 1)

test2.head(10)
# Making predictions and saving it in file

final_predictions = best_class_rf.predict_proba(test2)

final = pd.DataFrame()
final['ID'] = test['Master Number']
final['Expected'] = final_predictions[:,1]
final.to_csv('prediction_Mohil_Bajaj_from_kaggle_kernel.csv',index = False)
