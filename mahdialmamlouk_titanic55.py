
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.model_selection import train_test_split
import pandas as pd
pd.options.display.max_columns = 100
from matplotlib import pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
pd.options.display.max_rows = 100

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/titanic/train.csv')
data

#data.drop(columns=['zero', 'zero.1','zero.2', 'zero.3', 'zero.4', 'zero.5', 'zero.6',  'zero.7',
#                   'zero.8', 'zero.9', 'zero.10', 'zero.11', 'zero.12', 'zero.13','zero.14',
#                   'zero.15', 'zero.16', 'zero.17' , 'zero.18'], inplace=True)
#data
data.describe()
data.columns
data.info()
data.sample(5)
X = data.iloc[:,1:]
y = data.iloc[:,0]
X.columns
X['Ticket'].mode
y
X.sample(5)
X =X.drop(columns =['Name'])
data['Age'].fillna(data['Age'].median(), inplace=True)
data.describe()
data[data['Survived']==1]['Sex'].value_counts()
data[data['Survived']==0]['Sex'].value_counts()
survived_sex = data[data['Survived']==1]['Sex'].value_counts()
dead_sex = data[data['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))
figure = plt.figure(figsize=(18,9))
plt.hist([data[data['Survived']==1]['Age'], data[data['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
figure = plt.figure(figsize=(15,8))
plt.hist([data[data['Survived']==1]['Fare'],data[data['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(data[data['Survived']==1]['Age'],data[data['Survived']==1]['Fare'],c='green',s=40)
ax.scatter(data[data['Survived']==0]['Age'],data[data['Survived']==0]['Fare'],c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)
category1 = ["Survived", "Sex", "Pclass", "Embarked","SibSp", "Parch"]
for c in category1:
    bar_plot(c)