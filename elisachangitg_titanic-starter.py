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
test_labels = pd.read_csv("../input/gendermodel.csv",index_col =[0])

test_attr = pd.read_csv("../input/test.csv",index_col =[0])



train_labels = pd.read_csv("../input/train.csv", usecols=[0,1], index_col=[0])

train_attr = pd.read_csv("../input/train.csv", usecols=np.append(0,[x+2 for x in range(10)]), index_col =[0])
#from IPython.display import display

#display(train_labels,10)

train_attr.head(10)
# Convert categorical to numerical

data= pd.get_dummies(data=train_attr,columns={"Sex","Embarked"})

data.head(10)
import matplotlib.pyplot as plt

sumData = data.sum(axis=0)

print(sumData)
#####

### Distribution of passengers from different cities ###

#####



# Graph Distribution per class

p1 = plt.barh(np.arange(3),sumData[9:12],tick_label=sumData[9:12].index)

# Graph Survivors per class

classSurvivors = [train_labels[data["Embarked_C"] == 1].sum(),

                  train_labels[data["Embarked_Q"] == 1].sum(),

                 train_labels[data["Embarked_S"] == 1].sum()]

p2 = plt.barh(np.arange(3),classSurvivors,tick_label=sumData[9:12].index, color = '#d62728')

plt.tight_layout()



plt.legend((p1[0],p2[0]),("Count", "Survived"))


#####

### Distribution of passengers by Gender ###

### Observation: Women more likely to survive

### Improve by adding percentage

#####



# Graph Distribution per class

p1 = plt.barh(np.arange(2),sumData[7:9],tick_label=sumData[7:9].index)

# Graph Survivors per class

genderSurvivors = [train_labels[data["Sex_female"] == 1].sum(),

                  train_labels[data["Sex_male"] == 1].sum()]

p2 = plt.barh(np.arange(2),genderSurvivors,tick_label=sumData[7:9].index, color = '#d62728')



plt.tight_layout()



plt.legend((p1[0],p2[0]),("Count", "Survived"))
# Histogram of passengers by age

import matplotlib.pyplot as plt

plt.figure()

data['Age'].plot.hist(alpha = 1)
# Count of passengers by class

print(data["Pclass"].value_counts())
# Histogram of passengers by fare

import matplotlib.pyplot as plt

plt.figure()

data['Fare'].plot.hist(alpha = 0.5)
# Drop qualitative data

data.drop(["Name","Ticket","Cabin"],1,inplace=True)
# Process Test Attributes Similarly

test_data= pd.get_dummies(data=test_attr,columns={"Sex","Embarked"})

test_data.drop(['Name', "Ticket", "Cabin"],1,inplace=True)



# Replace NaN with 0

data.fillna(0,inplace=True)

test_data.fillna(0,inplace=True)
cols_with_missing = [col for col in data.columns 

                                 if data[col].isnull().any()]

print(cols_with_missing)

print(data['Age'])
print(data.shape)

print(train_labels.shape)

print(test_data.shape)

print(test_labels.shape)
test_data.head()
# Train a model

from sklearn import linear_model

logistic = linear_model.LogisticRegression()



#print(test_data)

print("Logistic Regression score: %f: " % logistic.fit(data,train_labels).score(test_data, test_labels))