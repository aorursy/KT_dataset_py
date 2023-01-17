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
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex7 import *
my_filepath = "../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv"
my_data = pd.read_csv(my_filepath,index_col = 'sl_no')
my_data.head()
my_data.info()
my_data['salary'] = my_data['salary'].replace(np.nan,0)
plt.figure(figsize = (12,8))
sns.lineplot(x = 'ssc_p',y = 'salary',data = my_data,label = 'SSC')
sns.lineplot(x = 'hsc_p',y = 'salary',data = my_data,label = 'HSC')
sns.lineplot(x = 'degree_p',y = 'salary',data = my_data,label = 'Degree')
sns.lineplot(x = 'mba_p',y = 'salary',data = my_data,label = 'MBA')
# Check that a figure appears below
step_4.check()
my_data['gender'].value_counts().plot(kind = 'bar',alpha = 0.7)
plt.xlabel('Gender')
plt.ylabel('Appeared in Placements')
my_data['status'].value_counts().plot(kind = 'bar',alpha = 0.7)
plt.xlabel('Status')
f,axes = plt.subplots(1,2,figsize = (12,5))
my_data['status'][my_data['gender']=='M'].value_counts().plot(kind = 'bar',alpha = 1,ax = axes[0])
my_data['status'][my_data['gender']=='F'].value_counts().plot(kind = 'bar',alpha = 1,ax = axes[1],color = 'pink')
f,axes = plt.subplots(1,3,figsize = (12,5))
my_data['hsc_s'].value_counts().plot(kind = 'bar',alpha = 1,ax = axes[0])
my_data['degree_t'].value_counts().plot(kind = 'bar',alpha = 1,ax = axes[1],color = 'cyan')
my_data['specialisation'].value_counts().plot(kind = 'bar',alpha = 1,ax = axes[2],color = 'green')
f,axes = plt.subplots(1,5,figsize = (25,6))

sns.swarmplot(x = 'status',y = 'hsc_p',data = my_data,ax = axes[0])
sns.swarmplot(x = 'status',y = 'ssc_p',data = my_data,ax = axes[1])
sns.swarmplot(x = 'status',y = 'degree_p',data = my_data,ax = axes[2])
sns.swarmplot(x = 'status',y = 'etest_p',data = my_data,ax = axes[3])
sns.swarmplot(x = 'status',y = 'mba_p',data = my_data,ax = axes[4])
sns.set_style('darkgrid')
sns.set_style('ticks')
features = ['hsc_p','ssc_p','degree_p','etest_p','mba_p']
f, ax = plt.subplots(5,2,figsize = (10,30))
x,y = 0,0
for i in range(4):
    for j in range(i+1,5):
        sns.scatterplot(x = my_data[features[i]],y = my_data[features[j]],hue = my_data['status'],ax = ax[x,y])
        sns.set_style('darkgrid')
        if y==1:
            x+=1
            y = 0
        else:
            y+=1
features = ['hsc_p','ssc_p','degree_p','etest_p','mba_p','salary']
fig, ax = plt.subplots(2,3,figsize = (25,15))
x,y = 0,0
for item in features:
    label = 'Distribution of ' + item
    d = np.array(my_data[item][my_data.status=='Placed'])
    sns.kdeplot(data = d,label = 'Placed',ax = ax[x,y],shade = True)
    d = np.array(my_data[item][my_data.status=='Not Placed'])
    sns.kdeplot(data = d,label = 'Not Placed',ax = ax[x,y])
    ax[x,y].set_title(label)
    if y==2:
        x+=1
        y = 0
    else:y+=1
    #label = 'Distribution of ' + item
    #plt.title.set_text(label)
    sns.set_style('darkgrid')
features =['gender',
 'ssc_p',
 'hsc_p',
 'degree_p',
 'mba_p',
 'etest_p',          
 'salary',
 'status',]
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
newdata = {}
for item in features:
    if item!='status' and item!='gender':
        newdata[item] = my_data[item]
        continue
    ffs = encoder.fit_transform(my_data[item])
    newdata[item] = ffs
newdata = pd.DataFrame(newdata)
newdata.head()
plt.figure(figsize = (12,10))
sns.heatmap(newdata.corr(),annot = True)
newdata.drop('salary',axis = 1,inplace = True)
X = newdata.drop('status',1)
Y = newdata['status']
from sklearn.model_selection import train_test_split
X,xtest,Y,ytest = train_test_split(X,Y,test_size = 0.1)
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(C = 1e5)
log_model.fit(X,Y)
log_model.score(xtest,ytest)
from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X,Y)
svm_model.score(xtest,ytest)