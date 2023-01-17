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
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



print ('sum of train data', df_train.isnull().sum())

print ('')

print ('sum of test data', df_test.isnull().sum())
###################################################################

# delete feature,who misses too many data

# delete feature, who doesn't contain the meaningful information

###################################################################

#1 - train data

# too many data miss for feature Cabin, 

df_train = df_train.drop('Cabin', 1)

# feature - Ticket, Name, doesn't contain the meaningful information for the algorithm

df_train = df_train.drop('Name', 1)

df_train = df_train.drop('Ticket', 1)



#2 - test data

# too many data miss for feature Cabin, 

df_test = df_test.drop('Cabin', 1)

# feature - Ticket, Name, doesn't contain the meaningful information for the algorithm

df_test = df_test.drop('Name', 1)

df_test = df_test.drop('Ticket', 1)



print ('sum of train data', df_train.isnull().sum())

print ('')

print ('sum of test data', df_test.isnull().sum())
#########################################

# prepare 2 temp data fram

# df_train_Non_Surived - data frame for all not survived

# df_train_Survived - data frame for all survived

df_train_Non_Survived = df_train.loc[(df_train.Survived == 0)]

df_train_Survived = df_train.loc[df_train.Survived == 1]

#df_train_Survived.describe()
##########################################

# 1. plot Fare vs Survived

##########################################



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#matplotlib.style.use('ggplot') 

#print(df_train['Fare'].values.max())



fig, axes = plt.subplots(nrows=2, ncols=2)



#hist for fare

ax1 = df_train.hist(column="Fare",        # Column to plot

              figsize=(8,8),         # Plot size

              color="blue",          # Plot color

              bins = 50, 

              ax=axes[0,0])

axes[0, 0].set_title('Fare Hist')



#box diagram - Fare by Survived

ax2 =df_train.boxplot(column="Fare",        # Column to plot

                 by= "Survived",       # Column to split upon

                 figsize= (8,8),       # Figure size

                 ax=axes[0,1])

ax2.set_ylim(-1, 125)

axes[0, 1].set_title('Fare by Survived')





#hist for fare when Survived = 0

ax3 = df_train_Non_Survived.hist(column="Fare",        # Column to plot

                                 figsize=(8,8),         # Plot size

                                 color="blue",          # Plot color

                                 bins = 50, 

                                 ax=axes[1,0])

axes[1, 0].set_title('Fare who Not Survived')



ax4 = df_train_Survived.hist(column="Fare",        # Column to plot

                             figsize=(8,8),         # Plot size

                             color="blue",          # Plot color

                             bins = 50,

                             ax = axes[1,1])

axes[1, 1].set_title('Fare who Survived')



#plt.show()
##########################################

# 2. plot Embarked vs Survived

##########################################



import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



fig, axes = plt.subplots(nrows=2, ncols=2)



#first plot: bar (freq) of Embarked

ax1 = df_train.Embarked.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[0,0])

axes[0, 0].set_title('Embarked Hist')



#second plot: Embarked vs Servived

sns.set(style="whitegrid", color_codes=True)

ax2 = sns.pointplot(x="Embarked", y="Survived", data=df_train, ax=axes[0,1]);

axes[0, 1].set_title('Embarked vs Survived')



#hist for fare when Survived = 0

ax3 = df_train_Non_Survived.Embarked.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,0])

axes[1, 0].set_title('Fare who Not Survived')



ax4 = df_train_Survived.Embarked.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,1])

axes[1, 1].set_title('Fare who Survived')
##########################################

# 3. plot Pclass vs Survived

##########################################



fig, axes = plt.subplots(nrows=2, ncols=2)



#first plot: bar (freq) of Pclass

ax1 = df_train.Pclass.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[0,0])

axes[0, 0].set_title('Pclass Hist')



#second plot: Pclass vs Servived

sns.set(style="whitegrid", color_codes=True)

#ax2 = sns.pointplot(x="Sex", y="Survived", data=df_train, ax=axes[0,1]);

ax2 = sns.barplot(x ="Pclass", y = "Survived", data=df_train, ax = axes[0,1])

axes[0, 1].set_title('Pclass vs Survived')



#hist for Pclass when Survived = 0

ax3 = df_train_Non_Survived.Pclass.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,0])

axes[1, 0].set_title('Pclass who Not Survived')



ax4 = df_train_Survived.Pclass.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,1])

axes[1, 1].set_title('Pclass who Survived')
##########################################

# 4. plot Sex vs Survived

##########################################



fig, axes = plt.subplots(nrows=2, ncols=2)



#first plot: bar (freq) of Gender

ax1 = df_train.Sex.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[0,0])

axes[0, 0].set_title('Gender Hist')



#second plot: Gender vs Servived

sns.set(style="whitegrid", color_codes=True)

ax2 = sns.barplot(x ="Sex", y = "Survived", data=df_train, ax = axes[0,1])

axes[0, 1].set_title('Gender vs Survived')



#hist for Sex when Survived = 0

ax3 = df_train_Non_Survived.Sex.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,0])

axes[1, 0].set_title('Pclass who Not Survived')



ax4 = df_train_Survived.Sex.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,1])

axes[1, 1].set_title('Pclass who Survived')
##########################################

# 5. plot Parch vs Survived

##########################################

#plt.tight_layout()

#plt.show()



fig, axes = plt.subplots(nrows=2, ncols=2)



#first plot: bar (freq) of Parch

ax1 = df_train.Parch.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[0,0])

axes[0, 0].set_title('Parch Hist')



#second plot: Parch vs Servived

sns.set(style="whitegrid", color_codes=True)

#ax2 = sns.pointplot(x="Sex", y="Survived", data=df_train, ax=axes[0,1]);

ax2 = sns.pointplot(x ="Parch", y = "Survived", data=df_train, ax = axes[0,1])

axes[0, 1].set_title('Parch vs Survived')



#hist for Parch when Survived = 0

ax3 = df_train_Non_Survived.Parch.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,0])

axes[1, 0].set_title('Parch who Not Survived')

#hist for Parch when Survived = 1

ax4 = df_train_Survived.Parch.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,1])

axes[1, 1].set_title('Parch who Survived')
##########################################

# 6. plot SibSp vs Survived

##########################################

fig, axes = plt.subplots(nrows=2, ncols=2)



#first plot: bar (freq) of SibSp

ax1 = df_train.SibSp.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[0,0])

axes[0, 0].set_title('SibSp Hist')



#second plot: SibSp vs Servived

sns.set(style="whitegrid", color_codes=True)

ax2 = sns.pointplot(x ="SibSp", y = "Survived", data=df_train, ax = axes[0,1])

axes[0, 1].set_title('SibSp vs Survived')



#hist for SibSp when Survived = 0

ax3 = df_train_Non_Survived.SibSp.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,0])

axes[1, 0].set_title('SibSp who Not Survived')

#hist for SibSp when Survived = 1

ax4 = df_train_Survived.SibSp.value_counts(sort = False).plot(kind='bar', stacked=True, ax=axes[1,1])

axes[1, 1].set_title('SibSp who Survived')
##########################################

# 7. plot Age vs Survived

##########################################

fig, axes = plt.subplots(nrows=2, ncols=2)



#hist for Age

ax1 = df_train.hist(column="Age",        # Column to plot

              figsize=(8,8),         # Plot size

              color="blue",          # Plot color

              bins = 50, 

              ax=axes[0,0])

axes[0, 0].set_title('Age Hist')



#box diagram - Age by Survived

ax2 =df_train.boxplot(column="Age",        # Column to plot

                 by= "Survived",       # Column to split upon

                 figsize= (8,8),       # Figure size

                 ax=axes[0,1])

#ax2.set_ylim(-1, 125)

axes[0, 1].set_title('Age by Survived')





#hist for Age when Survived = 0

ax3 = df_train_Non_Survived.hist(column="Age",        # Column to plot

                                 figsize=(8,8),         # Plot size

                                 color="blue",          # Plot color

                                 bins = 50, 

                                 ax=axes[1,0])

axes[1, 0].set_title('Age who Not Survived')



ax4 = df_train_Survived.hist(column="Age",        # Column to plot

                             figsize=(8,8),         # Plot size

                             color="blue",          # Plot color

                             bins = 50,

                             ax = axes[1,1])

axes[1, 1].set_title('Age who Survived')
#################################################

# Get correlation between features

# Delete passengerID, not useful for Correlation

#################################################

df_train_clean = df_train.drop('PassengerId', 1)

print(df_train_clean.describe())

print(df_train_clean.corr())
######################################

#Age >65, P(Survived) --> 0

#Fare > 100, P(Surived) is increased

######################################

ax1 = df_train_Survived.plot.scatter(x='Age', y='Fare', c='r', s= 100, marker = '+');

ax2 = df_train_Non_Survived.plot.scatter(x='Age', y='Fare', c='b', s= 10,  marker = 'x', ax=ax1);

plt.show()
##################################################

# Imputing the missing data for age and embarked

##################################################

from sklearn.preprocessing import Imputer



#1 - train data

# mean - stragety for imput missing Age

tmp =df_train['Age'].values

tmp = tmp.reshape(-1,1)

imr = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)

imr = imr.fit(tmp)

tmp = imr.transform(tmp)

df_train['Age'] = tmp



#most_frequent - strategy for imput missing Embarded

#Embarked is categorical data, mapping to Integer for imputting



embarked_mapping = {'S':3, 'C':2, 'Q':1}

inv_embarked_mapping = {v: k for k, v in embarked_mapping.items()}

df_train['Embarked']=df_train['Embarked'].map(embarked_mapping)



tmp =df_train['Embarked'].values

tmp = tmp.reshape(-1,1)

imr = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis=0)

imr = imr.fit(tmp)

tmp = imr.transform(tmp)

df_train['Embarked'] = tmp



df_train['Embarked']=df_train['Embarked'].map(inv_embarked_mapping)



#2 - test data

# mean - stragety for imput missing Age

tmp =df_test['Age'].values

tmp = tmp.reshape(-1,1)

imr = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)

imr = imr.fit(tmp)

tmp = imr.transform(tmp)

df_test['Age'] = tmp



# mean - stragety for imput missing Fare

tmp =df_test['Fare'].values

tmp = tmp.reshape(-1,1)

imr = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)

imr = imr.fit(tmp)

tmp = imr.transform(tmp)

df_test['Fare'] = tmp



#3 - show the summary of data

print ('sum of train data', df_train.isnull().sum())

print ('')

print ('sum of test data', df_test.isnull().sum())
test = df_train.tail()

print(test)



#Extract the data from the train.csv

#indx: contain the label of each columun

#X: contain all the passenager infor, except the label of 'Survived'

#y: the list of 'Survived'

index =[]

for i in range (len(df_train.axes[1])):

    if df_train.axes[1][i] != 'Survived':

        index.append(df_train.axes[1][i])



X = df_train.drop('Survived', 1).values

y = df_train.values[:,1]



print(index)
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import norm



#########################################

#fit the fare to the normal distribution

#########################################

#1. Fit for Fare - Non survived 

#find the minimum and maximum and the line space

xmin, xmax = min(df_train_Non_Survived.Fare), max(df_train_Non_Survived.Fare)

lnspc = np.linspace(xmin, xmax, len(df_train_Non_Survived.Fare))

#get mean and standared deviation

m_fare_no, s_fare_no = norm.fit(df_train_Non_Survived.Fare)

pdf_fare_Non_Survived = norm.pdf(lnspc, m_fare_no, s_fare_no) # now get theoretical values in our interval 

plt.subplot(2, 2, 3)

plt.plot(lnspc, pdf_fare_Non_Survived, 'r') # plot it



#2. FIt for Fare - survived

#find the minimum and maximum and the line space

xmin, xmax = min(df_train_Survived.Fare), max(df_train_Survived.Fare)

lnspc = np.linspace(xmin, xmax, len(df_train_Survived.Fare))

#get mean and standared deviation

m_fare_yes, s_fare_yes = norm.fit(df_train_Survived.Fare)

pdf_fare_Survived = norm.pdf(lnspc, m_fare_yes, s_fare_yes) # now get theoretical values in our interval 

plt.subplot(2, 2, 3)

plt.plot(lnspc, pdf_fare_Survived, 'g') # plot it

##########################################



##########################################

# Define Probability fun for Fare

##########################################

def Fare_prob(fare):

    Prob_no = norm.pdf(fare, m_fare_no, s_fare_no)

    Prob_yes = norm.pdf(fare, m_fare_yes, s_fare_yes)

    return [Prob_no, Prob_yes]



##########################################

# Define Probability fun for Pclass 

##########################################

def Pclass_prob(Pclass):

    Pclass_cnt_no= df_train_Non_Survived.Pclass.value_counts()

    Pclass_cnt_yes= df_train_Survived.Pclass.value_counts()

        

    Prop_yes = Pclass_cnt_yes[Pclass]/Pclass_cnt_yes.sum()

    Prop_no = Pclass_cnt_no[Pclass]/Pclass_cnt_no.sum()

    return [Prop_no, Prop_yes]

        

print(Pclass_prob(1))

print(Pclass_prob(2))

print(Pclass_prob(3))
##########################################

# Define Probability fun for Sex

##########################################

def Sex_prob(Sex):

    Sex_cnt_no= df_train_Non_Survived.Sex.value_counts()

    Sex_cnt_yes= df_train_Survived.Sex.value_counts()

        

    Prop_yes = Sex_cnt_yes[Sex]/Sex_cnt_yes.sum()

    Prop_no = Sex_cnt_no[Sex]/Sex_cnt_no.sum()

    return [Prop_no, Prop_yes]



print(Sex_prob('male'))

print(Sex_prob('female'))
##########################################

# Define Probability fun for Survived

##########################################

def Survived_prob():

    Survived_total = df_train.Survived.value_counts()

    total = Survived_total[0] + Survived_total[1]

           

    Prop_yes = Survived_total[1]/total

    Prop_no = Survived_total[0]/total

    return [Prop_no, Prop_yes]



print(Survived_prob())
#################################################

#Caluate predication from the training data

#################################################

#init pred vector

pred = []



for i in range(df_train.Pclass.size):

    #read the feature values from each sample

    f_pclass = df_train.Pclass[i]

    f_fare = df_train.Fare[i]

    f_sex = df_train.Sex[i]

    

    #calculate the prob

    p_pclass = Pclass_prob(f_pclass)

    p_fare = Fare_prob(f_fare)

    p_sex = Sex_prob (f_sex)

    p_survived = Survived_prob()

    

    p = np.multiply(p_pclass,p_fare)

    p = np.multiply(p, p_sex)

    p = np.multiply(p, p_survived)

    

    #calulate pred

    tmp=1 #set the default to 1: survived

    if (p[0] > p[1]):

        tmp = 0

    pred.append(tmp)

from sklearn.metrics import confusion_matrix



CM = confusion_matrix(df_train.Survived, pred)



TN = CM[0][0]

FN = CM[1][0]

TP = CM[1][1]

FP = CM[0][1]



# Overall accuracy

ACC = (TP+TN)/(TP+FP+FN+TN)

print(TN, FN, TP, FP)

print(ACC)

#################################################

#Caluate predication from the training data

#################################################

#init pred vector

pred_test = []



for i in range(df_test.Pclass.size):

    #read the feature values from each sample

    f_pclass_test = df_test.Pclass[i]

    f_fare_test = df_test.Fare[i]

    f_sex_test = df_test.Sex[i]

    

    #calculate the prob

    p_pclass_test = Pclass_prob(f_pclass_test)

    p_fare_test = Fare_prob(f_fare_test)

    p_sex_test = Sex_prob (f_sex_test)

    p_survived_test = Survived_prob()

    

    p = np.multiply(p_pclass_test,p_fare_test)

    p = np.multiply(p, p_sex_test)

    p = np.multiply(p, p_survived_test)

    

    #calulate pred

    tmp=1 #set the default to 1: suvived

    if (p[0] > p[1]):

        tmp = 0

    pred_test.append(tmp)   
output = pd.DataFrame({ 'PassengerId': df_test["PassengerId"], "Survived": pred_test})

output.to_csv('prediction.csv', index=False)