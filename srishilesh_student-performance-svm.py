import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model,preprocessing,model_selection
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
# Lets assume the passmark to be 40
passmark = 40
# Input the CSV file
data = pd.read_csv("../input/StudentsPerformance.csv")
# Printing the first 5 rows
data.head()
# Describing the data
data.describe()
# Renaming Columns
data.rename(columns={'race/ethnicity':'race','parental level of education':'parent','test preparation course':'test',
                    'math score':'math','reading score':'reading','writing score':'writing'},inplace=True)
data.head()
data.test.unique()
# Visualizing the Male and Female count
plt.rcParams['figure.figsize'] = (5,5)
plt.style.use('_classic_test')
sns.countplot(data['gender'],palette='bone')
plt.title('Comparison of Males and Females', fontweight = 30)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()
# To check for missing values
data.isnull().sum()
# Handling the Categorical data

data.loc[data["gender"]=="male","gender"]=0
data.loc[data["gender"]=="female","gender"]=1
data.loc[data["race"]=="group A","race"] = 0
data.loc[data["race"]=="group B","race"] = 1
data.loc[data["race"]=="group C","race"] = 2
data.loc[data["race"]=="group D","race"] = 3
data.loc[data["race"]=="group E","race"] = 4
data.loc[data["parent"]=="bachelor's degree","parent"] = 0
data.loc[data["parent"]=="some college","parent"] = 1
data.loc[data["parent"]=="master's degree","parent"] = 2
data.loc[data["parent"]=="associate's degree","parent"] = 3
data.loc[data["parent"]=="high school","parent"] = 4
data.loc[data["parent"]=="some high school","parent"] = 5
data.loc[data["lunch"]=="standard","lunch"] = 0
data.loc[data["lunch"]=="free/reduced","lunch"] = 1
data.loc[data["test"]=="none","test"] = 0
data.loc[data["test"]=="completed","test"] = 1

data.head()
# To find how many have passed and failed
data["pass_math"] = np.where(data["math"]>passmark,1,0)
data.pass_math.value_counts()
# Comparison of Parental education
plt.rcParams['figure.figsize'] = (5,5)
plt.style.use('dark_background')
sns.countplot(data['parent'],palette='Blues')
plt.title('Comparison of Parental Education', fontweight = 30, fontsize = 20)
plt.xlabel('Degree')
plt.ylabel('count')
plt.show()
# Graph plotting the pass and failure of Math scores based on Parents level of education
p = sns.countplot(x='parent', data = data, hue='pass_math', palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90) # To set colors for different plots
plt.xlabel("Parent level of education")
plt.ylabel("Maths scores")
data["pass_read"] = np.where(data["reading"]>passmark,1,0)
data.pass_read.value_counts()
r = sns.countplot(x='parent',data=data,hue='pass_read',palette='bright')
_ = plt.setp(p.get_xticklabels(), rotation=90) # To set colors for different plots
data['pass_write'] = np.where(data['writing']>passmark,1,0)
data.pass_write.value_counts()
w = sns.barplot(x='parent',y='writing',data=data,hue='pass_write')
data['total_score'] = data['math'] + data['reading'] + data['writing']
data.sort_values(by=['total_score'],ascending=False)
#data['total_gender'] = np.where(data['total_score'])
data.head()
male_count = (data['gender']==0).value_counts()
female_count = (data['gender']==1).value_counts()
#male_count
female_count
features = ['gender','race','parent','lunch','test']
target = ['pass_math']
X = data[features]
y = data[target]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)
# Printing Confusion matrix as a metric 

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_pred,y_test))


