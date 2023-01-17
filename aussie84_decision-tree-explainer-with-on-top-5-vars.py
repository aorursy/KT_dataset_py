# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.ticker as ticker

sns.set_style("whitegrid")

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Setting up the dataframe

train = pd.read_csv('../input/train.csv')

train.info()

train.head()
# Creaet violin and swarm plots

sns.violinplot(x='Pclass',y='Fare',data=train,inner=None)

sns.swarmplot(x='Pclass',y='Fare',data=train,color='w',alpha=0.5)

plt.title("Violin and Swarm Plot to compare fare distribution among Pclass groups")

# Note: This takes a long time to run
# Create kde plot

sns.kdeplot(train['Fare'][train['Pclass']==1],shade=True)

sns.kdeplot(train['Fare'][train['Pclass']==2],shade=True)

sns.kdeplot(train['Fare'][train['Pclass']==3],shade=True)

plt.title("Distribution of fares across Pclass groups")
# Create distribution plot again for pclass=1, because the scale is hard to read in previous one

fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(12,4))

fig.suptitle("PDF and CDF of fares among Pclass=1")

sns.distplot(train['Fare'][train['Pclass']==1],rug=True,ax=axis1)

axis1.set_title("PDF")

sns.kdeplot(train['Fare'][train['Pclass']==1],shade=True,cumulative=True,ax=axis2)

axis2.set_title("CDF")

x_forplot = train['Fare'][train['Pclass']==1]

axis2 = plt.xticks(np.arange(min(x_forplot),max(x_forplot)+1,40.0))
# Create distribution plot again for pclass=2 and 3, because the scale is hard to read in previous one

sns.kdeplot(train['Fare'][train['Pclass']==2],shade=True)

sns.kdeplot(train['Fare'][train['Pclass']==3],shade=True)

plt.title("PDF of fares among Pclass= 2 and 3")



# Note: You need to add this line of code for the code above to work - %matplotlib inline
# Create the 2 new segmentation columns

train['PaxclassA'] = train['Pclass']

train.loc[(train['PaxclassA'] == 1) & (train['Fare']>100),'PaxclassA'] = 0

train['PaxclassB'] = train['Pclass']

train.loc[(train['Fare']>60),'PaxclassB'] = 0
# Group fare into bins to analyze survival rate across brackets. The brackets are informed by the dist plot above

bins = [0,20,40,60,80,100,200,400,800]

train['Fare_Groups'] = pd.cut(train['Fare'],bins)
# Create plots to compare survival between the 2 new segmentation columns. We also show similar plot based on original Pclass and Fare buckets

fig, ((axis1,axis2),(axis3,axis4)) = plt.subplots(2,2,sharey=True,figsize=(12,4))

sns.factorplot("PaxclassA","Survived",data=train,ax=axis1)

sns.factorplot("PaxclassB","Survived",data=train,ax=axis2)

sns.factorplot("Pclass","Survived",data=train,ax=axis3)

sns.factorplot("Fare_Groups","Survived",data=train,ax=axis4)

fig.suptitle("Survival Rate across Segments, based on 2 new segments and 2 original vars")



# Note: I still don't know why these line of codes produce the blank charts on the bottom....
train['Fare_Groups2'] = train['Fare_Groups'].astype("object") # Need this conversion for heatmap to work

sns.heatmap(pd.crosstab(train['Pclass'],train['Fare_Groups2'],values=train['Survived'],aggfunc=np.mean).T,annot=True,cmap="Blues")

plt.title("Crosstab Heatmap of Pclass x Fares")



# Ideally, I should add 1 more heatmap to show the count. But let me put it in backburner, as the count is quite large for the regular pclass=1
# Analyzing cross-tab of age and sex on survival

bins = [0,12,18,35,50,70,100]  # General age group breakdown

train['Age_Groups'] = pd.cut(train['Age'],bins)

sns.heatmap(pd.crosstab(train['Sex'],train['Age_Groups'],values=train['Survived'],aggfunc=np.mean).T,annot=True,cmap="Blues")

plt.title("Crosstab Heatmap of Sex x Age: Children (<12yo) seems prioritized, but elderly were not")
# Create combined variable and show the survival rate

train['SexAge'] = train['Sex']

train.loc[(train['Age']<=12),'SexAge'] = 'children'

sns.factorplot("SexAge","Survived",data=train)
# Crosstab and heatmap on the impact of having parents / children

print(pd.crosstab(train['SexAge'],train['Parch']))

crosstab1 = pd.crosstab(train['SexAge'],train['Parch'],values=train['Survived'],aggfunc=np.mean)

sns.heatmap(crosstab1.T,annot=True,cmap="Blues")

plt.title("Crosstab Heatmap of SexAge x Parch")
print(pd.crosstab(train['SexAge'],train['SibSp']))

crosstab1 = pd.crosstab(train['SexAge'],train['SibSp'],values=train['Survived'],aggfunc=np.mean)

sns.heatmap(crosstab1.T,annot=True,cmap="Blues")

plt.title("Crosstab Heatmap of SexAge x SibSp")
# We need to convert categorical variables into binary variable

train['Female'] = 0

train.loc[(train['SexAge']=="female"),'Female'] = 1

train['Children'] = 0

train.loc[(train['SexAge']=="children"),'Children'] = 1

train['Class1_Premium'] = 0

train.loc[(train['PaxclassA']==0),'Class1_Premium'] = 1

train['Class1'] = 0

train.loc[(train['PaxclassA']==1),'Class1'] = 1

train['Class2'] = 0

train.loc[(train['PaxclassA']==2),'Class2'] = 1
# Define the variables for training

from sklearn import tree

Xtrain = train[['Female','Children','Parch','SibSp','Class1_Premium','Class1','Class2']]

Ytrain = train['Survived']
# Set up and fit the decision tree model. Then export as graphviz

Tree1 = tree.DecisionTreeClassifier(max_depth=4,min_samples_split=50,random_state=1)

Tree1.fit(Xtrain,Ytrain)

Tree1_dot = tree.export_graphviz(Tree1,out_file=None,feature_names=Xtrain.columns,class_names=['Not Survived','Survived'],proportion=True,filled=True)

print(Tree1_dot)
# Check the score of prediction accuracy

Tree1.score(Xtrain,Ytrain)
test = pd.read_csv('../input/test.csv')



# Create new combined variables

test['SexAge'] = test['Sex']

test.loc[(test['Age']<=12),'SexAge'] = 'children'

test['PaxclassA'] = test['Pclass']

test.loc[(test['PaxclassA'] == 1) & (test['Fare']>100),'PaxclassA'] = 0



# Create binary variables out of categorical variables

test['Female'] = 0

test.loc[(test['SexAge']=="female"),'Female'] = 1

test['Children'] = 0

test.loc[(test['SexAge']=="children"),'Children'] = 1

test['Class1_Premium'] = 0

test.loc[(test['PaxclassA']==0),'Class1_Premium'] = 1

test['Class1'] = 0

test.loc[(test['PaxclassA']==1),'Class1'] = 1

test['Class2'] = 0

test.loc[(test['PaxclassA']==2),'Class2'] = 1



# Create the prediction

Xtest = test[['Female','Children','Parch','SibSp','Class1_Premium','Class1','Class2']]

Ytest_pred = Tree1.predict(Xtest)

submission = pd.DataFrame({

    "PassengerId":test['PassengerId'],

    "Survived":Ytest_pred

})

submission.to_csv('titanic.csv',index=False)
# Set up and fit the decision tree model. Then export as graphviz

Tree2 = tree.DecisionTreeClassifier(max_depth=6,min_samples_split=50,random_state=1,min_impurity_decrease=0.0003)

Tree2.fit(Xtrain,Ytrain)

Tree2_dot = tree.export_graphviz(Tree1,out_file=None,feature_names=Xtrain.columns,class_names=['Not Survived','Survived'],proportion=True,filled=True)

print(Tree2_dot)
# Create the prediction

Xtest = test[['Female','Children','Parch','SibSp','Class1_Premium','Class1','Class2']]

Ytest_pred = Tree2.predict(Xtest)



submission = pd.DataFrame({

    "PassengerId":test['PassengerId'],

    "Survived":Ytest_pred

})

submission.to_csv('titanic2.csv',index=False)