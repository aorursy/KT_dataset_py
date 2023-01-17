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
import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

import warnings

warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier 

from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

heartdf = pd.read_csv('../input/heart-disease-dataset/Heart Disease Dataset.csv')

heartdf.head()
heartdf.shape
heartdf.info()
heartdf.isnull().sum()
heartdf.describe()
heartdf.columns
num_cols = list(heartdf.columns[0:len(heartdf.columns)-1])

num_cols.remove('sex')
plt.figure(figsize=(30,15))

for i in enumerate(num_cols):

    plt.subplot(3,4,i[0]+1)

    ax = sns.boxplot(heartdf[i[1]])

    ax.set_xlabel(i[1],fontsize=20)



plt.tight_layout()

plt.show()
heartdf.nunique(axis=0)
fig = plt.figure(figsize = (25, 8))





# ----------------------------------------------------------------------------------------------------

# plot the data

# the idea is to iterate over each class

# extract their data ad plot a sepate density plot

for i in heartdf["target"].unique():

    # extract the data

    x = heartdf[heartdf["target"] == i]["chol"]

    # plot the data using seaborn

    plt.subplot(1,4,1)

    sns.kdeplot(x, shade=True, label = "{} target".format(i))



# set the title of the plot

plt.title("Density Plot of chol by target")



# ----------------------------------------------------------------------------------------------------

# plot the data

# the idea is to iterate over each class

# extract their data ad plot a sepate density plot

for i in heartdf["target"].unique():

    # extract the data

    x = heartdf[heartdf["target"] == i]["trestbps"]

    # plot the data using seaborn

    plt.subplot(1,4,2)

    sns.kdeplot(x, shade=True, label = "{} target".format(i))



# set the title of the plot

plt.title("Density Plot of trestbps by target")



# ----------------------------------------------------------------------------------------------------

# plot the data

# the idea is to iterate over each class

# extract their data ad plot a sepate density plot

for i in heartdf["target"].unique():

    # extract the data

    x = heartdf[heartdf["target"] == i]["thalach"]

    # plot the data using seaborn

    plt.subplot(1,4,3)

    sns.kdeplot(x, shade=True, label = "{} target".format(i))



# set the title of the plot

plt.title("Density Plot of thalach by target")



# ----------------------------------------------------------------------------------------------------

# plot the data

# the idea is to iterate over each class

# extract their data ad plot a sepate density plot

for i in heartdf["target"].unique():

    # extract the data

    x = heartdf[heartdf["target"] == i]["oldpeak"]

    # plot the data using seaborn

    plt.subplot(1,4,4)

    sns.kdeplot(x, shade=True, label = "{} target".format(i))



# set the title of the plot

plt.title("Density Plot of oldpeak by target")



plt.tight_layout()

plt.show()
## plot the data based on different target to show the ditribution of chol and trestbps as per different sex



plt.figure(figsize=(25,8),dpi=80)

plt.subplot(1,2,1)

ax = sns.violinplot(x = "sex", y = "chol", hue = "target", split = True, data = heartdf)

ax.set_title('Distribution of chol for different target by sex', fontsize = 15)



plt.subplot(1,2,2)

ay = sns.violinplot(x = "sex", y = "trestbps", hue = "target", split = True, data = heartdf)

ay.set_title('Distribution of trestbps for different target by sex', fontsize = 15)



plt.tight_layout()

plt.show()
## plot the data based on different target to show the ditribution of thalach and oldpeak as per different sex





plt.figure(figsize=(25,8),dpi=80)

plt.subplot(1,2,1)

ax = sns.violinplot(x = "sex", y = "thalach", hue = "target", split = True, data = heartdf)

ax.set_title('Distribution of thalach for different target by sex', fontsize = 15)



plt.subplot(1,2,2)

ay = sns.violinplot(x = "sex", y = "oldpeak", hue = "target", split = True, data = heartdf)

ay.set_title('Distribution of oldpeak for different target by sex', fontsize = 15)



plt.tight_layout()

plt.show()
cat_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
heart_des = heartdf[heartdf['target']==1]

heart_notdes = heartdf[heartdf['target']==0]
plt.figure(figsize=(30,15))

for i in enumerate(cat_cols):

    plt.subplot(2,4,i[0]+1)

    ax = heart_des[i[1]].value_counts(normalize=True).plot.barh()

    ax.set_title("Deceased showing by "+i[1],fontsize=15)

plt.show()
plt.figure(figsize=(30,15))

for i in enumerate(cat_cols):

    plt.subplot(2,4,i[0]+1)

    ax = heart_notdes[i[1]].value_counts(normalize=True).plot.barh()

    ax.set_title("Not deceased showing by "+i[1],fontsize=15)

plt.show()
df_train,df_test = train_test_split(heartdf,train_size=0.7,random_state=50)
y_train = df_train.pop('target')

X_train = df_train
y_test = df_test.pop('target')

X_test = df_test
## creat function for check train and test set

def check_model(dt):

    print("train confusion matrix : ",confusion_matrix(y_train,dt.predict(X_train)))

    print("train accuracy score : ",accuracy_score(y_train,dt.predict(X_train)))

    print("__"*50)

    print("test confusion matrix : ",confusion_matrix(y_test,dt.predict(X_test)))

    print("test accuracy score : ",accuracy_score(y_test,dt.predict(X_test)))    

    
dt_default = DecisionTreeClassifier(random_state=0)

dt_res = dt_default.fit(X_train,y_train)
check_model(dt_res)
## create function to visualize graphs



def tree_graph(dt):



    fig = plt.figure(figsize=(25,20))



    dt_plot = tree.plot_tree(dt,feature_names=X_train.columns,class_names=['Not Deceased','Deceased'],filled=True)
tree_graph(dt_res)
params = {'max_depth':[2,3,4,5,6,7,8,9,10],

          'min_samples_split':[5,10,25,50,75,100,150]}
grid_search = GridSearchCV(estimator=dt_default,param_grid=params,scoring='accuracy',n_jobs=-1,verbose=1) ## create grid search object
grid_search.fit(X_train,y_train)
grid_search.best_estimator_
best_dt = grid_search.best_estimator_
check_model(best_dt)
tree_graph(best_dt)