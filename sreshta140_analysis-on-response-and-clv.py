# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('/kaggle/input/ibm-watson-marketing-customer-value-data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv')

df.head()
df.columns
Response=df.groupby('Response')['Customer'].count()
Response
df.groupby('Response')['Customer'].count().plot(kind='bar', grid=True,
figsize=(10, 7),
title='Marketing Responses').set_ylabel('Count')
categorical_var=['State', 'Response', 'Coverage',
       'Education', 'EmploymentStatus', 'Gender',
       'Marital Status', 'Number of Policies', 'Policy Type',
       'Policy', 'Renew Offer Type', 'Sales Channel',
       'Vehicle Class', 'Vehicle Size']
for i in categorical_var:
    df.groupby(i)['Customer'].count().plot(kind='bar', grid=True,
    figsize=(10, 7)).set_ylabel('Count')
    plt.show()
categorical_var.remove('Response')
categorical_var
for i in categorical_var:
    cat_vs_cat=df.groupby(['Response', i])['Customer'].count()
    cat_vs_cat = cat_vs_cat.unstack().fillna(0)
    ax = (cat_vs_cat).plot(
    kind='bar',
    figsize=(10, 7),
    grid=True
    )
    ax.set_ylabel('Count')
    plt.show()
Education_vs_Emp=df[df['Response']=='Yes'].groupby(['Education', 'EmploymentStatus'])['Customer'].count()
Education_vs_Emp
Education_vs_Emp = Education_vs_Emp.unstack().fillna(0)
Education_vs_Emp
# Visualize this data in bar plot
ax = (Education_vs_Emp).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
Offer_vs_Channel=df[df['Response']=='Yes'].groupby(['Renew Offer Type', 'Sales Channel'])['Customer'].count()
Offer_vs_Channel
Offer_vs_Channel = Offer_vs_Channel.unstack().fillna(0)
Offer_vs_Channel
# Visualize this data in bar plot
ax = (Offer_vs_Channel).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
Vehicle_vs_Size=df[df['Response']=='Yes'].groupby(['Vehicle Class', 'Vehicle Size'])['Customer'].count()
Vehicle_vs_Size
Vehicle_vs_Size = Vehicle_vs_Size.unstack().fillna(0)
Vehicle_vs_Size
# Visualize this data in bar plot
ax = (Vehicle_vs_Size).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
offer_vs_vclass=df[df['Response']=='Yes'].groupby(['Renew Offer Type', 'Vehicle Class'])['Customer'].count()
offer_vs_vclass
offer_vs_vclass = offer_vs_vclass.unstack().fillna(0)
offer_vs_vclass
# Visualize this data in bar plot
ax = (offer_vs_vclass).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
channel_vs_vclass=df[df['Response']=='Yes'].groupby(['Sales Channel', 'Vehicle Class'])['Customer'].count()
channel_vs_vclass
channel_vs_vclass = channel_vs_vclass.unstack().fillna(0)
channel_vs_vclass
# Visualize this data in bar plot
ax = (channel_vs_vclass).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
channel_vs_gender=df[df['Response']=='Yes'].groupby(['Sales Channel', 'Gender'])['Customer'].count()
channel_vs_gender
channel_vs_gender = channel_vs_gender.unstack().fillna(0)
channel_vs_gender
# Visualize this data in bar plot
ax = (channel_vs_gender).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
offer_vs_gender=df[df['Response']=='Yes'].groupby(['Renew Offer Type', 'Gender'])['Customer'].count()
offer_vs_gender
offer_vs_gender = offer_vs_gender.unstack().fillna(0)
offer_vs_gender
# Visualize this data in bar plot
ax = (offer_vs_gender).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
channel_vs_mstatus=df[df['Response']=='Yes'].groupby(['Sales Channel', 'Marital Status'])['Customer'].count()
channel_vs_mstatus
channel_vs_mstatus = channel_vs_mstatus.unstack().fillna(0)
channel_vs_mstatus
# Visualize this data in bar plot
ax = (channel_vs_mstatus).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
offer_vs_mstatus=df[df['Response']=='Yes'].groupby(['Renew Offer Type', 'Marital Status'])['Customer'].count()
offer_vs_mstatus
offer_vs_mstatus = offer_vs_mstatus.unstack().fillna(0)
offer_vs_mstatus
# Visualize this data in bar plot
ax = (offer_vs_mstatus).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
gender_vs_mstatus=df[df['Response']=='Yes'].groupby(['Gender', 'Marital Status'])['Customer'].count()
gender_vs_mstatus
gender_vs_mstatus = gender_vs_mstatus.unstack().fillna(0)
gender_vs_mstatus
# Visualize this data in bar plot
ax = (gender_vs_mstatus).plot(
kind='bar',
figsize=(10, 7),
grid=True
)
ax.set_ylabel('Count')
plt.show()
df.describe()
corrmat = df.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True, annot = True)
plt.show()
num_list=['Customer Lifetime Value', 'Income', 'Monthly Premium Auto',
       'Months Since Last Claim', 'Months Since Policy Inception',
       'Number of Open Complaints', 'Number of Policies', 'Total Claim Amount']
num_list
for i in num_list:
    plt.figure(figsize = (12, 9))
    plt.hist(df[i], bins = 100)
    plt.xlabel(i)
    plt.ylabel('Count')
    plt.show()
import matplotlib.pyplot as plt
plt.figure(figsize = (12, 9))
plt.scatter(df['Monthly Premium Auto'], df['Customer Lifetime Value'])
plt.ylabel('Customer Lifetime Value')
plt.xlabel('Monthly Premium Auto')
plt.show()
categorical_var=['State', 'Response', 'Coverage',
       'Education', 'EmploymentStatus', 'Gender',
       'Marital Status', 'Number of Policies', 'Policy Type',
       'Policy', 'Renew Offer Type', 'Sales Channel',
       'Vehicle Class', 'Vehicle Size']
for i in categorical_var:
    cats=df.groupby(i)['Customer'].count()
    indx=cats.index
    print(i)
    for j in indx:
        print(j)
        plt.figure(figsize = (12, 9))
        plt.scatter(df[df[i]==j]['Customer Lifetime Value'], df[df[i]==j]['Monthly Premium Auto'])
        plt.xlabel('Customer Lifetime Value')
        plt.ylabel('Monthly Premium Auto')
        plt.show()  
nop=df.groupby('Number of Policies')['Customer'].count()
nop
nopi=nop.index
plt.figure(figsize = (12, 9))
for i in nopi:
    plt.scatter(df[df['Number of Policies']==i]['Monthly Premium Auto'], df[df['Number of Policies']==i]['Customer Lifetime Value'])
plt.xlabel('Customer Lifetime Value')
plt.ylabel('Monthly Premium Auto')
plt.show()
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
X = df['Monthly Premium Auto'].values.reshape(-1,1)
y = df['Customer Lifetime Value'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm

print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)
y_pred = regressor.predict(X_test)
df_ = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df_
df1 = df_.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
df_pol_1=df[df['Number of Policies']==1]
df_pol_1

X = df_pol_1['Monthly Premium Auto'].values.reshape(-1,1)
y = df_pol_1['Customer Lifetime Value'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
#To retrieve the intercept:
print(regressor.intercept_)
#For retrieving the slope:
print(regressor.coef_)
y_pred = regressor.predict(X_test)
df_ = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df_
df1 = df_.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))