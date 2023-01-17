# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/creditcard.csv')

df.head()
df.shape,df.columns
import matplotlib.pyplot as plt

time = df['Time'].values

arr = np.arange(time.shape[0])

plt.plot(arr,time)

plt.show()
plt.figure(figsize=(5, 3), dpi=80)

plt.subplots()

for i in range(28):

    time = df['V'+str(i+1)].values

    arr = np.arange(time.shape[0])

    plt.scatter(arr,time)

    plt.title("Ploting V"+str(i+1)+" vs index", fontsize=15)

    plt.show()
from sklearn import preprocessing

train_np = df.values

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(train_np)

train_df = pd.DataFrame(x_scaled)

train_df.columns = df.columns 
import seaborn as sns

for i in range(28):

    color = sns.color_palette()

    ##train_df['V'+str(i+1)] = train_df['V'+str(i+1)]/np.max(train_df['V'+str(i+1)].values) #truncation for better visuals

    plt.figure(figsize=(4,4))

    sns.violinplot(x='Class', y='V'+str(i+1), data=train_df)

    plt.xlabel('Class', fontsize=12)

    plt.ylabel('V'+str(i+1), fontsize=12)

    plt.title("V"+str(i+1)+" by category", fontsize=15)

    plt.show()
class_0 = [ df['Amount'][x] for x in range(df.shape[0]) if df['Class'][x]==0]

time_0 =  [ df['Time'][x] for x in range(df.shape[0]) if df['Class'][x]==0]

dict1 = {0:'r',1:'g'}

color_0 = [dict1[x] for x in df['Class'].values if x==0]

plt.scatter(time_0,class_0,color=color_0)

plt.title("Ploting Amount vs time", fontsize=15)

plt.show()
class_1 = [ df['Amount'][x] for x in range(df.shape[0]) if df['Class'][x]==1]

time_1 =  [ df['Time'][x] for x in range(df.shape[0]) if df['Class'][x]==1]

dict1 = {0:'r',1:'g'}

color_1 = [dict1[x] for x in df['Class'].values if x==1]

plt.scatter(time_1,class_1,color=color_1)

plt.title("Ploting Amount vs time", fontsize=15)

plt.show()
x_np = np.array(class_1).reshape(len(class_1),1)

x_np = np.append(x_np,np.array(time_1).reshape(len(class_1),1),axis=1)

df_1 = pd.DataFrame(x_np)

df_1.columns = ['Amount','Time']

df_1.head(),df_1.shape
color = sns.color_palette()

##train_df['Amount'] = train_df['V'+str(i+1)]/np.max(train_df['V'+str(i+1)].values) #truncation for better visuals

plt.figure(figsize=(4,4))

sns.violinplot( y='Time', data=df_1)

plt.xlabel('Class_1', fontsize=12)

plt.ylabel('Time_1', fontsize=12)

plt.title("Times for fraud Transaction", fontsize=15)

plt.show()
x_np = np.array(class_0).reshape(len(class_0),1)

x_np = np.append(x_np,np.array(time_0).reshape(len(class_0),1),axis=1)

df_0 = pd.DataFrame(x_np)

df_0.columns = ['Amount','Time']

df_0.head(),df_0.shape
color = sns.color_palette()

##train_df['Amount'] = train_df['V'+str(i+1)]/np.max(train_df['V'+str(i+1)].values) #truncation for better visuals

plt.figure(figsize=(4,4))

sns.violinplot( y='Time', data=df_0)

plt.xlabel('Class_1', fontsize=12)

plt.ylabel('Time_1', fontsize=12)

plt.title("Times for non-fraud Transaction", fontsize=15)

plt.show()