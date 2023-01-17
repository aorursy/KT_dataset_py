# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#load data and seperate labels
data = pd.read_csv('../input/train.csv')
labels_train = data['Survived']
features_train = data.drop(['Survived'], axis=1)
print(features_train.head())
# axis set to 0 as we are talking about the columns
#Belive me its the worst plotting skills
#Data Exploration(I know i could use ratios, but its a genetic problem of using graphs:))
plt.bar([0], [len(labels_train) - sum(labels_train)], color = 'r')
plt.bar([1], [sum(labels_train)], color='g')
plt.xticks([0+0.35,1+0.35], ['Not Survived', 'Survived'])
plt.show()
#No of females and males dying
plt.bar([0], [len(data[(data['Survived']==0)&(data['Sex']=='male')])], color='b')
plt.bar([1], [len(data[(data['Survived']==0)&(data['Sex']=='female')])],color='r')
plt.xticks([0+0.35, 1+0.35],['Males Not Survived','Females Not Survived'])
plt.show()
#Thus a considerable amount of men die and many ladies were saved!!
#Let us see how travelling class decided Death in the Titanic
plt.bar([0],[len(data[(data['Survived']==0)&(data['Pclass']==3)])], color='r')
plt.bar([1],[len(data[(data['Survived']==0)&(data['Pclass']==2)])], color='b')
plt.bar([2],[len(data[(data['Survived']==0)&(data['Pclass']==1)])], color='g')
plt.xticks([0+0.45, 1+0.45, 2+0.45], ['Lower', 'Middle', 'Upper'])
plt.show()
#Let us see how lower class females did
plt.bar([0],[len(data[(data['Survived']==0)&(data['Pclass']==3)&(data['Sex']=='female')])],color='r')
plt.bar([1],[len(data[(data['Survived']==0)&(data['Pclass']==2)&(data['Sex']=='female')])],color='b')
plt.bar([2],[len(data[(data['Survived']==0)&(data['Pclass']==1)&(data['Sex']=='female')])],color='g')
plt.xticks(np.arange(3)+0.45, ['LFNS','MFNS','UFNS'])
plt.show()
#Age can be an important feature too. Thus let us explore that too.
plt.bar([0],[len(data[(data['Survived']==0)&(data['Age']<=10)&(data['Sex']=='male')])],color='r')
plt.bar([1],[len(data[(data['Survived']==0)&((data['Age']>10)&(data['Age']<=30))&(data['Sex']=='male')])],color='b')
plt.bar([2],[len(data[(data['Survived']==0)&(data['Age']>30)&(data['Sex']=='male')])],color='g')
plt.xticks(np.arange(3)+0.45, ['Age<=10','10<Age<=30','Age>30'])
plt.ylabel("Men died plotted against age")
plt.show()
plt.bar([0],[len(data[(data['Survived']==0)&(data['Age']<=10)&(data['Sex']=='female')])],color='r')
plt.bar([1],[len(data[(data['Survived']==0)&((data['Age']>10)&(data['Age']<=30))&(data['Sex']=='female')])],color='b')
plt.bar([2],[len(data[(data['Survived']==0)&(data['Age']>30)&(data['Sex']=='female')])],color='g')
plt.xticks(np.arange(3)+0.45, ['Age<=10','10<Age<=30','Age>30'])
plt.ylabel("Women")
plt.show()
plt.bar([0],[len(data[(data['Survived']==0)&(data['Embarked']=='C')])],color='r')
plt.bar([1],[len(data[(data['Survived']==0)&(data['Embarked']=='S')])],color='b')
plt.bar([2],[len(data[(data['Survived']==0)&(data['Embarked']=='Q')])],color='g')
plt.xticks(np.arange(3)+0.5, ['C','S','Q'])
plt.ylabel("People died")
plt.show()
def accuracy_measure(labels_test, features_test):
    predicted_survival = classify(features_test)
    true_classified = float(sum((labels_test == predicted_survival)))/len(labels_test)
    return true_classified
def classify(features_set):
    predicted_survival = []
    for _,feature_data in features_set.iterrows():
        if(feature_data['Sex'] == 'male'):
            if((feature_data['Age']<10)and(feature_data['Pclass']!=3)):
                predicted_survival.append(1)
            else:
                predicted_survival.append(0)
        else:
            if(feature_data['Embarked'] == 'S') and (feature_data['Pclass'] == 3):
                predicted_survival.append(0)
            else:
                predicted_survival.append(1)
    return predicted_survival
data_test = pd.read_csv('../input/test.csv')
#labels_test = data_test['Survived']
#features_test = data_test.drop(['Survived'], axis=1)
print(accuracy_measure(labels_train, features_train))
# On the train set
def makeFinalAnswer(test_set):
    predictions = classify(test_set)
    a = pd.DataFrame({'PassengerId':test_set['PassengerId'], 'Survived':predictions})
    return(a)
makeFinalAnswer(data_test)