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
# linear algebra

import numpy as np 

from statistics import mode



# data processing

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



# ML models

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import OneHotEncoder
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')



# concatenated train and test

dfs = [df_train, df_test]



df_train.name = 'Training Set'

df_test.name = 'Test Set'





print('Number of Training Examples = {}'.format(df_train.shape[0]))

print('Number of Test Examples = {}\n'.format(df_test.shape[0]))

print('Training X Shape = {}'.format(df_train.shape))

print('Training y Shape = {}\n'.format(df_train['Survived'].shape[0]))

print('Test X Shape = {}'.format(df_test.shape))

print('Test y Shape = {}\n'.format(df_test.shape[0]))

print(df_train.columns)

print(df_test.columns)
df_train.info()
df_test.info()
df_train.describe()
# box plot



df_train.Age.plot.box(grid='True')



'''Age distribution lies between 14 and 80 and we can easily visualise by below box plot'''
df_train.Fare.plot.box(grid='True')



'''more than 80% of fare values falls below 31'''
survived = df_train['Survived'].value_counts()[1]

not_survived = df_train['Survived'].value_counts()[0]

survived_per = survived / df_train.shape[0] * 100

not_survived_per = not_survived / df_train.shape[0] * 100



print('{} of {} passengers survived '.format(survived, df_train.shape[0], survived_per))

print('{} of {} passengers didnt survive '.format(not_survived, df_train.shape[0], not_survived_per))



plt.figure(figsize=(10, 8))

sns.countplot(df_train['Survived'])



plt.xlabel('Survival', size=15, labelpad=15)

plt.ylabel('Passenger Count', size=15, labelpad=15)

plt.xticks((0, 1), ['Not Survived ({0:.2f}%)'.format(not_survived_per), 'Survived ({0:.2f}%)'.format(survived_per)])

plt.tick_params(axis='x', labelsize=13)

plt.tick_params(axis='y', labelsize=13)



plt.title('Training Set Survival Distribution', size=15, y=1.05)



plt.show();
# train set correlation



df_train_corr = df_train.drop(['PassengerId'], axis=1).corr()

df_train_corr
sns.heatmap(df_train_corr,annot = True,cmap='coolwarm',vmin=-1, vmax=1, center= 0)

plt.title("Training Set correlation");
# test set correlation



sns.heatmap(df_test.corr(),annot = True,cmap='coolwarm',vmin=-1, vmax=1, center= 0)

plt.title("Test Set correlation");
df_train_corr1 = df_train.drop(['PassengerId'], axis=1).corr()#.abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()

#df_train_corr1['highly_correlated'] = df_train_corr1.T.apply(lambda x: x.nlargest(2).idxmin())

df_train_corr1['high_corr'] = df_train_corr1.columns[df_train_corr1.values.argsort(1)[:, -2]]

df_train_corr1
df_train_corr1 = df_train.drop(['PassengerId'], axis=1).corr()



df_train_corr1['high_corr']  = None

# Iterate over each row 

for index, rows in df_train_corr1.iterrows(): 

    # Create list for the current row 

    my_list =[rows.Survived, rows.Pclass, rows.Age ,rows.SibSp,rows.Parch,rows.Fare ] 

    my_list.sort()

    df_train_corr1.high_corr[index] = my_list[-2]

    

df_train_corr1
df_train_corr3 = df_train.drop(['PassengerId'], axis=1).corr()

sns.heatmap(df_train_corr3[((df_train_corr3 >= 0.3) | (df_train_corr3 <= -0.3)) & (df_train_corr3 != 1)],cmap='coolwarm', annot=True, linewidths=.5, fmt= '.2f')

plt.title('Configured Corelation Matrix');
df_train_corr2 = df_train.drop(['PassengerId'], axis=1).corr().abs().unstack().sort_values(ascending=False).reset_index()

df_train_corr2.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

df_train_corr2 = df_train_corr2.drop(df_train_corr2[df_train_corr2['Correlation Coefficient'] == 1.0].index)

df_train_corr2.drop(df_train_corr2.iloc[1::2].index, inplace=True)

df_train_corr2


# Training set high correlations

corr = df_train_corr2['Correlation Coefficient'] > 0.1

df_train_corr2[corr]
# test set correlation



df_test_corr = df_test.corr().abs().unstack().sort_values(ascending=False).reset_index()

df_test_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

df_test_corr = df_train_corr2.drop(df_train_corr2[df_train_corr2['Correlation Coefficient'] == 1.0].index)

df_test_corr.drop(df_train_corr2.iloc[1::2].index, inplace=True)



corr_test = df_test_corr['Correlation Coefficient'] > 0.1

df_test_corr[corr_test]
def display_missing(df):    

    for col in df.columns.tolist():          

        print('{} missing values: {}'.format(col, df[col].isnull().sum()))

    print('\n')

    

for df in dfs:

    print (df.name)

    display_missing(df)
total = df_train.isnull().sum().sort_values(ascending=False)

percent_1 = df_train.isnull().sum()/df_train.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

missing_data
#train set

df_train.loc[df_train.Age.isnull(), 'Age'] = df_train.groupby("Pclass").Age.transform('median')





#test set

df_test.loc[df_test.Age.isnull(), 'Age'] = df_test.groupby("Pclass").Age.transform('median')
df_train.Embarked.value_counts()
df_train['Embarked'] = df_train['Embarked'].fillna(mode(df_train['Embarked']))
med_fare = df_test.groupby(['Pclass']).Fare.transform('median')

# Filling the missing value in Fare with the median Fare of 3rd class alone passenger

df_test['Fare'] = df_test['Fare'].fillna(med_fare)
# generate binary values using get_dummies

dum_df = pd.get_dummies(df_train, columns=["Sex"] )

'''# merge with main df bridge_df on key values

df_train = df_train.join(dum_df)

df_train'''

dum_df
survived = 'survived'

not_survived = 'not survived'





ax = sns.distplot(df_train[df_train['Survived']==1].Age.dropna(), bins=20, label = survived, kde =False)

ax = sns.distplot(df_train[df_train['Survived']==0].Age.dropna(), bins=40, label = not_survived, kde =False)

_= ax.legend()
survived = 'survived'

not_survived = 'not survived'

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = df_train[df_train['Sex']=='female']

men = df_train[df_train['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=25, label = survived, ax = axes[0], kde =False)

ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=35, label = not_survived, ax = axes[0], kde =False)

ax.legend()

ax.set_title('Female')

ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=25, label = survived, ax = axes[1], kde = False)

ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=35, label = not_survived, ax = axes[1], kde = False)

ax.legend()

_ = ax.set_title('Male')

sns.countplot(x='Embarked', hue='Survived', data=df_train)

    

plt.xlabel('Embarked', size=20, labelpad=15)

plt.ylabel('Passenger Count', size=20, labelpad=15)    

plt.tick_params(axis='x', labelsize=20)

plt.tick_params(axis='y', labelsize=20)

plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})

plt.title('Count of Survival in Embarked Feature', size=20, y=1.05)

_ = ax.set_title('')
# Passenger count of each feature



category_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp']



fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 20))

plt.subplots_adjust(right=1.5, top=1.25)



for i, feature in enumerate(category_features, 1):    

    plt.subplot(3, 2, i)

    sns.countplot(x=feature,data=df_train)

    plt.xlabel('{}'.format(feature), size=20, labelpad=15)

    plt.ylabel('Passenger Count', size=20, labelpad=15) 

    plt.title('Count of  {} Feature'.format(feature), size=20, y=1.05)



    #sns.countplot(train['Embarked'])

    #plt.title('Number of Port of embarkation');
category_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp']



fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 20))

plt.subplots_adjust(right=1.5, top=1.25)



for i, feature in enumerate(category_features, 1):    

    plt.subplot(2, 3, i)

    sns.countplot(x=feature, hue='Survived', data=df_train)

    

    plt.xlabel('{}'.format(feature), size=20, labelpad=15)

    plt.ylabel('Passenger Count', size=20, labelpad=15)    

    plt.tick_params(axis='x', labelsize=20)

    plt.tick_params(axis='y', labelsize=20)

    

    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})

    plt.title('Count of Survival in {} Feature'.format(feature), size=20, y=1.05)



plt.show()
sns.catplot(x="Pclass", y="Age",hue = "Survived", kind="swarm", data=df_train, height = 6)

plt.title("Passenger_class Vs Age ");

class_1 = df_train[df_train['Pclass']==1] 

survival_1 = class_1[class_1['Survived']==1]

class_1_survival_rate = (len(survival_1)/len(class_1))*100



class_2 = df_train[df_train['Pclass']==2] 

survival_2 = class_2[class_2['Survived']==1]

class_2_survival_rate = (len(survival_2)/len(class_2))*100



class_3 = df_train[df_train['Pclass']==3] 

survival_3 = class_3[class_3['Survived']==1]

class_3_survival_rate = (len(survival_3)/len(class_3))*100



print("First class passengers survival rate {} ".format(class_1_survival_rate))

print('Number of survival in first class {}'.format(len(survival_1)))

print("Second class passengers survival rate {} ".format(class_2_survival_rate))

print('Number of survival in second class {}'.format(len(survival_2)))

print("Third class passengers survival rate {} ".format(class_3_survival_rate))

print('Number of survival in third class {}'.format(len(survival_3)))
ax = sns.boxplot(x="Pclass", y="Fare", data=df_train)

ax = sns.swarmplot(x="Pclass", y="Fare", data=df_train, color=".25")



medians = df_train.groupby(['Pclass'])['Fare'].median().values

median_labels = [str(np.round(s, 2)) for s in medians]



pos = range(len(medians))

for tick,label in zip(pos,ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick] + 0.5, median_labels[tick], 

            horizontalalignment='center', size='medium', color='w', weight='semibold')



#sns.catplot(x="Pclass", y="Fare",hue = "Survived", kind="swarm", data=df_train, height = 6)



plt.title("Passenger_class Vs Fare ");
sns.catplot(x="SibSp", y="Age",hue = 'Survived', kind="swarm", data=df_train, height = 6)

plt.title("SipSp Vs Age ");
sibsp_0 = df_train[df_train['SibSp']==0] 

survival_0 = sibsp_0[sibsp_0['Survived']==1]

sibsp_0_survival_rate = (len(survival_0)/len(sibsp_0))*100





sibsp_1 = df_train[df_train['SibSp']==1] 

survival_1 = sibsp_1[sibsp_1['Survived']==1]

sibsp_1_survival_rate = (len(survival_1)/len(sibsp_1))*100



sibsp_2 = df_train[df_train['SibSp']==2] 

survival_2 = sibsp_2[sibsp_2['Survived']==1]

sibsp_2_survival_rate = (len(survival_2)/len(sibsp_2))*100



sibsp_3 = df_train[df_train['SibSp']==3] 

survival_3 = sibsp_3[sibsp_3['Survived']==1]

sibsp_3_survival_rate = (len(survival_3)/len(sibsp_3))*100



sibsp_4 = df_train[df_train['SibSp']==4] 

survival_4 = sibsp_4[sibsp_4['Survived']==1]

sibsp_4_survival_rate = (len(survival_4)/len(sibsp_4))*100



sibsp_5 = df_train[df_train['SibSp']==5] 

survival_5 = sibsp_5[sibsp_5['Survived']==1]

sibsp_5_survival_rate = (len(survival_5)/len(sibsp_5))*100



print("0-SibSp passengers survival rate {} ".format(sibsp_0_survival_rate))



print("1-SibSp passengers survival rate {} ".format(sibsp_1_survival_rate))



print("2-SibSp class passengers survival rate {} ".format(sibsp_2_survival_rate))



print("3-SibSp class passengers survival rate {} ".format(sibsp_3_survival_rate))



print("4-SibSp class passengers survival rate {} ".format(sibsp_4_survival_rate))



print("5-SibSp class passengers survival rate {} ".format(sibsp_5_survival_rate))

fig, ax = plt.subplots(figsize=(8,8))

sns.boxplot(x="SibSp", y="Age", data=df_train,palette="Set3",ax=ax)

#sns.swarmplot(x="SibSp", y="Age", data=df_train, color=".25")



medians = df_train.groupby(['SibSp'])['Age'].median().values

median_labels = [str(np.round(s, 2)) for s in medians]



pos = range(len(medians))

for tick,label in zip(pos,ax.get_xticklabels()):

    ax.text(pos[tick], medians[tick] + 0.5, median_labels[tick], 

            horizontalalignment='center', size='medium', color='b', weight='semibold')

plt.title("SipSp Vs Age ");
plt.subplots(figsize=(10,5))



sns.countplot(x='Parch', hue='SibSp', data=df_train)



plt.title("Parch Vs Age ");
'''df_train.groupby('Parch')['SibSp']\

    .value_counts()\

    .unstack(level=1)\

    .plot.bar(stacked=True)'''



ct = pd.crosstab(df_train.Parch, df_train.SibSp)



ax = ct.plot.bar(stacked=True)



plt.legend(title='Parch vs SibSp')



plt.show()



print (ct)
'''def feature_interactions(df,feature1, feature2,continuous_col):

    group = df.groupby([feature1,feature2],as_index=False)[continuous_col].mean().reset_index(drop=True)

    pivot = group.pivot(index=feature1, columns=feature2, values=continuous_col)

    pivot.fillna(0, inplace=True)

    plt.figure(figsize=(10,6))

    sns.heatmap(pivot,cmap='Reds')

    plt.show()



feature_interactions(df_train,'Parch','SibSp','Age')'''
all_data = [df_train,df_test]



for i in all_data:

    i['relatives'] = i['SibSp'] + i['Parch']

    i['family_size'] = i['SibSp'] + i['Parch'] +1

    i.loc[i['relatives'] > 0, 'not_alone'] = 0

    i.loc[i['relatives'] == 0, 'not_alone'] = 1

    i['not_alone'] = i['not_alone'].astype(int)

    

df_train['not_alone'].value_counts()



sns.barplot(x="relatives",y = 'Survived', data=df_train)



plt.title("relatives Vs Survived ");

# Drop redundant features

df_train = df_train.drop(['SibSp', 'Parch', 'Ticket','relatives','not_alone'], axis = 1)

df_test = df_test.drop(['SibSp', 'Parch', 'Ticket','relatives','not_alone'], axis = 1)
# Creating Deck column from the first letter of the Cabin column (M stands for Missing)

df_all = [df_train,df_test]

    

#df_all['Deck'] = df_all['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')



df_train['Deck'] = df_train['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

df_train.drop(columns = 'Cabin')



df_test['Deck'] = df_test['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

df_test.drop(columns = 'Cabin')



ct = pd.crosstab(df_train.Deck, df_train.Pclass)





ct.plot.bar(stacked=True,figsize=(10,5))

#plt.figure(figsize=(20,20))

plt.legend(title='Deck vs Pclass');



#print (ct)
# calculate percentage of Classes in each decks



def calc_pclass_deck_percentage(df,i):

    

    class_1 = df[df['Pclass'] == i] 

    

    Deck_Ai = class_1[class_1['Deck']== 'A']

    Deck_Bi = class_1[class_1['Deck']== 'B']

    Deck_Ci = class_1[class_1['Deck']== 'C']

    Deck_Di = class_1[class_1['Deck']== 'D']

    Deck_Ei = class_1[class_1['Deck']== 'E']

    Deck_Fi = class_1[class_1['Deck']== 'F']

    Deck_Gi = class_1[class_1['Deck']== 'G']

    Deck_Mi = class_1[class_1['Deck']== 'M']

    Deck_Ti = class_1[class_1['Deck']== 'T']

    

    Deck_A = df[df['Deck'] == 'A']

    Deck_B = df[df['Deck'] == 'B']

    Deck_C = df[df['Deck'] == 'C']

    Deck_D = df[df['Deck'] == 'D']

    Deck_E = df[df['Deck'] == 'E']

    Deck_F = df[df['Deck'] == 'F']

    Deck_G = df[df['Deck'] == 'G']

    Deck_M = df[df['Deck'] == 'M']

    Deck_T = df[df['Deck'] == 'T']

    

    

    percent_Ai = (len(Deck_Ai) / len(Deck_A) ) *100

    percent_Bi = (len(Deck_Bi) / len(Deck_B) ) *100

    percent_Ci = (len(Deck_Ci) / len(Deck_C) ) *100

    percent_Di = (len(Deck_Di) / len(Deck_D) ) *100

    percent_Ei = (len(Deck_Ei) / len(Deck_E) ) *100

    percent_Fi = (len(Deck_Fi) / len(Deck_F) ) *100

    percent_Gi = (len(Deck_Gi) / len(Deck_G) ) *100

    percent_Mi = (len(Deck_Mi) / len(Deck_M) ) *100

    percent_Ti = (len(Deck_Ti) / len(Deck_T) ) *100

    

    print('Class ' + str(i) +  '---Deck-A :',percent_Ai)

    print('Class ' + str(i) +  '---Deck-B :' ,percent_Bi)

    print('Class ' + str(i) +  '---Deck-C :' ,percent_Ci)

    print('Class ' + str(i) +  '---Deck-D :' ,percent_Di)

    print('Class ' + str(i) +  '---Deck-E :',percent_Ei)

    print('Class ' + str(i) +  '---Deck-F :' ,percent_Fi)

    print('Class ' + str(i) +  '---Deck-G :' ,percent_Gi)

    print('Class ' + str(i) +  '---Deck-M :' ,percent_Mi)

    print('Class ' + str(i) +  '---Deck-T :' ,percent_Ti)

    

    

    

    return percent_Ai,percent_Bi,percent_Ci,percent_Di,percent_Ei,percent_Fi,percent_Gi,percent_Mi,percent_Ti

    

print ("Percentage of Class-1 passengers in Decks")

Pclass_1 = calc_pclass_deck_percentage(df_train,1)

print (Pclass_1)



print ("Percentage of Class-2 passengers in Decks")

Pclass_2 = calc_pclass_deck_percentage(df_train,2)

print (Pclass_2)



print ("Percentage of Class-3 passengers in Decks")

Pclass_3 = calc_pclass_deck_percentage(df_train,3)

print (Pclass_3)







pclass_1 = []

for i in Pclass_1:

    pclass_1.append(i)





pclass_2 = []

for i in Pclass_2:

    pclass_2.append(i)

    

pclass_3 = []

for i in Pclass_3:

    pclass_3.append(i)

    

# Heights of bars1 + bars2

bars = np.add(pclass_1, pclass_2).tolist()

 

# The position of the bars on the x-axis

r = [0,1,2,3,4,5,6,7,8]

 

# Names of group and bar width

names = ['A','B','C','D','E','F','G','M','T']

barWidth = 1

 

plt.figure(figsize = (10,5))

    

# Create brown bars

plt.bar(r, pclass_1, color='lightcoral', edgecolor='white', width=barWidth,label = 'Pclass-1')

# Create green bars (middle), on top of the first ones

plt.bar(r, pclass_2, bottom=pclass_1, color='paleturquoise', edgecolor='white', width=barWidth,label = 'Pclass-2')

# Create green bars (top)

plt.bar(r, pclass_3, bottom=bars, color='mediumpurple', edgecolor='white', width=barWidth,label = 'Pclass-3')

 

# Custom X axis



plt.xticks(r, names, fontweight='bold')

plt.xlabel("Deck")

plt.ylabel("percentage of passengers")



#plt.legend(title='Passenger Class')

plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})

plt.title('Passenger class Percentage in Decks', size=18, y=1.05)

    

# Show graphic

plt.show()

# Passenger in the T deck is changed to A

idx = df_train[df_train['Deck'] == 'T'].index

df_train.loc[idx, 'Deck'] = 'A'
def survival_percentage_deck(df,deck):

    temp_df =  df[df['Deck'] == deck] 

    len_temp_surv = len(temp_df[temp_df['Survived'] == 1])

    percentage = (len_temp_surv/len(temp_df))*100

    return percentage,(100-percentage)



deck_A = survival_percentage_deck(df_train,'A')

deck_B = survival_percentage_deck(df_train,'B')

deck_C = survival_percentage_deck(df_train,'C')

deck_D = survival_percentage_deck(df_train,'D')

deck_E = survival_percentage_deck(df_train,'E')

deck_F = survival_percentage_deck(df_train,'F')

deck_G = survival_percentage_deck(df_train,'G')

deck_M = survival_percentage_deck(df_train,'M')





# values of each group

survival_perc = [deck_A[0],deck_B[0],deck_C[0],deck_D[0],deck_E[0],deck_F[0],deck_G[0],deck_M[0]]

non_survival_perc = [deck_A[1],deck_B[1],deck_C[1],deck_D[1],deck_E[1],deck_F[1],deck_G[1],deck_M[1]]



 

# Heights of bars1 + bars2

bars = np.add(survival_perc, non_survival_perc).tolist()

 

# The position of the bars on the x-axis

r = [0,1,2,3,4,5,6,7]

 

# Names of group and bar width

names = ['A','B','C','D','E','F','G','M']

barWidth = 1

 

plt.figure(figsize = (10,5))

 

# Create stacked bars



plt.bar(r, survival_perc, color='#b5ffb9', edgecolor='white', width=barWidth, label="Survived")

plt.bar(r, non_survival_perc, bottom=survival_perc, color='#f9bc86', edgecolor='white', width=barWidth, label="Non_Survived")



#plt.bar(r, survival_perc, color='#7f6d5f', edgecolor='white', width=barWidth)



#plt.bar(r, non_survival_perc, bottom=bars, color='#2d7f5e', edgecolor='white', width=barWidth)

 

# Custom X axis

plt.xticks(r, names, fontweight='bold')

plt.xlabel("Deck")

 

# Show graphic

plt.title('Survival Percentage in Decks', size=18, y=1.05)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})

plt.show()



# Dropping the Cabin feature

df_train.drop(['Cabin'], inplace=True, axis=1)

df_test.drop(['Cabin'], inplace=True, axis=1)





dfs = [df_train, df_test]

for df in dfs:

    display_missing(df)
df_train['Name'] = df_train.Name.str.extract(' ([A-Za-z]+)\.', expand = False)

df_test['Name'] = df_test.Name.str.extract(' ([A-Za-z]+)\.', expand = False)



df_train.Name.value_counts()
# combine similar titles



df_train.rename(columns={'Name' : 'Title'}, inplace=True)

df_train['Title'] = df_train['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 

                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')

                                      

df_test.rename(columns={'Name' : 'Title'}, inplace=True)

df_test['Title'] = df_test['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'Countess', 

                                       'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')



df_train['Title'].value_counts(normalize = True) * 100
# plotting title feature



sns.barplot(x=df_train['Title'].value_counts().index, y=df_train['Title'].value_counts().values)

plt.title('Title Feature Value Counts After Grouping', size=20, y=1.05)



plt.show()

df_train.info()
# transforming title to numeric



encoder = OneHotEncoder()

temp = pd.DataFrame(encoder.fit_transform(df_train[['Title']]).toarray(),columns=encoder.get_feature_names(['Title']))

df_train = df_train.join(temp)

df_train.drop(columns='Title', inplace=True)



temp_test = pd.DataFrame(encoder.transform(df_test[['Title']]).toarray(),columns=encoder.get_feature_names(['Title']))

df_test = df_test.join(temp_test)

df_test.drop(columns='Title', inplace=True)
# transforming Embarked



encoder = OneHotEncoder()

temp = pd.DataFrame(encoder.fit_transform(df_train[['Embarked']]).toarray(), columns=['S', 'C', 'Q'])

df_train = df_train.join(temp)

df_train.drop(columns='Embarked', inplace=True)



temp = pd.DataFrame(encoder.transform(df_test[['Embarked']]).toarray(), columns=['S', 'C', 'Q'])

df_test = df_test.join(temp)

df_test.drop(columns='Embarked', inplace=True)
# transforming deck to numeric



deck_category = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'M':8}

df_train['Deck'] = df_train['Deck'].map(deck_category)

df_test['Deck'] = df_test['Deck'].map(deck_category)
# transfroming Sex to numeric



df_train['Sex'][df_train['Sex'] == 'male'] = 0

df_train['Sex'][df_train['Sex'] == 'female'] = 1



df_test['Sex'][df_test['Sex'] == 'male'] = 0

df_test['Sex'][df_test['Sex'] == 'female'] = 1
df_train
#train test split





from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df_train.drop(['Survived', 'PassengerId'], axis=1), df_train['Survived'], test_size = 0.2, random_state=2)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()



X_train_scaled = scaler.fit_transform(X_train)



# we must apply the scaling to the test set that we computed for the training set

X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression



linreg = LinearRegression()

linreg.fit(X_train, y_train)



#R-Squared Score

print("R-Squared for Train set: {:.3f}".format(linreg.score(X_train, y_train)))

print("R-Squared for test set: {:.3f}" .format(linreg.score(X_test, y_test)))



acc_linreg_train = round(linreg.score(X_train, y_train) * 100, 2)

acc_linreg_test = round(linreg.score(X_test, y_test) * 100, 2)



acc_linreg_train_scaled = round(logreg.score(X_train_scaled, y_train) * 100, 2)

acc_linreg_test_scaled = round(logreg.score(X_test_scaled, y_test) * 100, 2)
# logistic regression on raw data



from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(max_iter=10000, C=50)

logreg.fit(X_train, y_train)



#R-Squared Score

print("R-Squared for Train set: {:.3f}".format(logreg.score(X_train, y_train)))

print("R-Squared for test set: {:.3f}" .format(logreg.score(X_test, y_test)))
# logistic regression on scaled data



logreg = LogisticRegression(max_iter=10000)

logreg.fit(X_train_scaled, y_train)



#R-Squared Score

print("R-Squared for Train set: {:.3f}".format(logreg.score(X_train_scaled, y_train)))

print("R-Squared for test set: {:.3f}" .format(logreg.score(X_test_scaled, y_test)))



acc_log_train = round(logreg.score(X_train, y_train) * 100, 2)

acc_log_test = round(logreg.score(X_test, y_test) * 100, 2)



acc_log_train_scaled = round(logreg.score(X_train_scaled, y_train) * 100, 2)

acc_log_test_scaled = round(logreg.score(X_test_scaled, y_test) * 100, 2)
from sklearn.neighbors import KNeighborsClassifier



knnclf = KNeighborsClassifier(n_neighbors=7)



# Train the model using the training sets

knnclf.fit(X_train, y_train)

y_pred = knnclf.predict(X_test)



from sklearn.metrics import accuracy_score



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",accuracy_score(y_test, y_pred))
# scaled data



knnclf = KNeighborsClassifier(n_neighbors=7)



# Train the model using the scaled training sets

knnclf.fit(X_train_scaled, y_train)

y_pred = knnclf.predict(X_test_scaled)



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",accuracy_score(y_test, y_pred))



acc_knn_train = round(knnclf.score(X_train, y_train) * 100, 2)

acc_knn_test = round(knnclf.score(X_test, y_test) * 100, 2)



acc_knn_train_scaled = round(knnclf.score(X_train_scaled, y_train) * 100, 2)

acc_knn_test_scaled = round(knnclf.score(X_test_scaled, y_test) * 100, 2)
from sklearn.svm import LinearSVC



svmclf = LinearSVC(C=50)

svmclf.fit(X_train, y_train)



print('Accuracy of Linear SVC classifier on training set: {:.2f}'

     .format(svmclf.score(X_train, y_train)))

print('Accuracy of Linear SVC classifier on test set: {:.2f}'

     .format(svmclf.score(X_test, y_test)))
#Scaled data



svmclf = LinearSVC()

svmclf.fit(X_train_scaled, y_train)



print('Accuracy of Linear SVC classifier on scaled-training set: {:.2f}'

     .format(svmclf.score(X_train_scaled, y_train)))

print('Accuracy of Linear SVC classifier on scaled-test set: {:.2f}'

     .format(svmclf.score(X_test_scaled, y_test)))





acc_linear_svc_train = round(svmclf.score(X_train, y_train) * 100, 2)

acc_linear_svc_test = round(svmclf.score(X_test, y_test) * 100, 2)



acc_linear_svc_train_scaled = round(svmclf.score(X_train_scaled, y_train) * 100, 2)

acc_linear_svc_test_scaled = round(svmclf.score(X_test_scaled, y_test) * 100, 2)

from sklearn.tree import DecisionTreeClassifier



dtclf = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)



print('Accuracy of Decision Tree classifier on training set: {:.2f}'

     .format(dtclf.score(X_train, y_train)))

print('Accuracy of Decision Tree classifier on test set: {:.2f}'

     .format(dtclf.score(X_test, y_test)))
#scaled data



from sklearn.tree import DecisionTreeClassifier



dtclf = DecisionTreeClassifier(max_depth = 3).fit(X_train, y_train)



print('Accuracy of Decision Tree classifier on scaled-training set: {:.2f}'

     .format(dtclf.score(X_train_scaled, y_train)))

print('Accuracy of Decision Tree classifier on scaled-test set: {:.2f}'

     .format(dtclf.score(X_test_scaled, y_test)))





acc_decision_tree_train = round(dtclf.score(X_train, y_train) * 100, 2)

acc_decision_tree_test = round(dtclf.score(X_test, y_test) * 100, 2)



acc_decision_tree_train_scaled = round(dtclf.score(X_train_scaled, y_train) * 100, 2)

acc_decision_tree_test_scaled = round(dtclf.score(X_test_scaled, y_test) * 100, 2)
from sklearn.ensemble import RandomForestClassifier

rfclf = RandomForestClassifier(random_state=2).fit(X_train, y_train)



print('Accuracy of Random Forest classifier on training set: {:.2f}'

     .format(rfclf.score(X_train, y_train)))

print('Accuracy of Random Forest classifier on test set: {:.2f}'

     .format(rfclf.score(X_test, y_test)))



print('Accuracy of Random Forest classifier on scaled-training set: {:.2f}'

     .format(rfclf.score(X_train_scaled, y_train)))

print('Accuracy of Random Forest classifier on scaled-test set: {:.2f}'

     .format(rfclf.score(X_test_scaled, y_test)))



acc_random_forest_train = round(rfclf.score(X_train, y_train) * 100, 2)

acc_random_forest_test = round(rfclf.score(X_test, y_test) * 100, 2)



acc_random_forest_train_scaled = round(rfclf.score(X_train_scaled, y_train) * 100, 2)

acc_random_forest_test_scaled = round(rfclf.score(X_test_scaled, y_test) * 100, 2)
from sklearn.linear_model import SGDClassifier



sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(X_train, y_train)

Y_pred = sgd.predict(X_test)





print('Accuracy of Random Forest classifier on training set: {:.2f}'

     .format(sgd.score(X_train, y_train)))

print('Accuracy of Random Forest classifier on test set: {:.2f}'

     .format(sgd.score(X_test, y_test)))



print('Accuracy of Random Forest classifier on scaled-training set: {:.2f}'

     .format(sgd.score(X_train_scaled, y_train)))

print('Accuracy of Random Forest classifier on scaled-test set: {:.2f}'

     .format(sgd.score(X_test_scaled, y_test)))



acc_stochastic_gd_train = round(sgd.score(X_train, y_train) * 100, 2)

acc_stochastic_gd_test = round(sgd.score(X_test, y_test) * 100, 2)



acc_stochastic_gd_train_scaled = round(sgd.score(X_train_scaled, y_train) * 100, 2)

acc_stochastic_gd_test_scaled = round(sgd.score(X_test_scaled, y_test) * 100, 2)
# cell highlighting



def highlight_max(data, color='yellow'):

    '''

    highlight the maximum in a Series or DataFrame

    '''

    attr = 'background-color: {}'.format(color)

    #remove % and cast to float

    data = data.replace('%','', regex=True).astype(float)

    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1

        is_max = data == data.max()

        return [attr if v else '' for v in is_max]

    else:  # from .apply(axis=None)

        is_max = data == data.max().max()

        return pd.DataFrame(np.where(is_max, attr, ''),

                            index=data.index, columns=data.columns)


results_df = pd.DataFrame({

    'Model': ['Linear Regression', 'Logistic Regression', 'KNN','Support Vector Machines',

             'Decision Tree', 'Random Forest','Stochastic Gradient Decent'],

    'train_score': [acc_linreg_train,acc_log_train,acc_knn_train,acc_linear_svc_train,acc_decision_tree_train,

             acc_random_forest_train,acc_stochastic_gd_train],

    'test_score': [acc_linreg_test,acc_log_test,acc_knn_test,acc_linear_svc_test,acc_decision_tree_test,

             acc_random_forest_test,acc_stochastic_gd_test],

    'scaled_train_score': [acc_linreg_train_scaled,acc_log_train_scaled,acc_knn_train_scaled,acc_linear_svc_train_scaled,acc_decision_tree_train_scaled,

             acc_random_forest_train_scaled,acc_stochastic_gd_train_scaled],

    'scaled_test_score': [acc_linreg_test_scaled,acc_log_test_scaled,acc_knn_test_scaled,acc_linear_svc_test_scaled,acc_decision_tree_test_scaled,

             acc_random_forest_test_scaled,acc_stochastic_gd_test_scaled]})

results_df = results_df.set_index('Model')

results_df.head(10)
results_df.style.apply(highlight_max)
# Set our parameter grid

param_grid = { 

    'criterion' : ['gini', 'entropy'],

    'n_estimators': [100, 300, 500],

    'max_features': ['auto', 'log2'],

    'max_depth' : [3, 5, 7]    

}



from sklearn.model_selection import GridSearchCV



randomForest_CV = GridSearchCV(estimator = rfclf, param_grid = param_grid, cv = 5)

randomForest_CV.fit(X_train, y_train)
# our optimal parameters



randomForest_CV.best_params_
rf_clf = RandomForestClassifier(random_state = 2, criterion = 'entropy', max_depth = 7, max_features = 'auto', n_estimators = 500)



rf_clf.fit(X_train, y_train)



acc_rf_clf_test = round(rf_clf.score(X_test, y_test) * 100, 2)





rf_clf.fit(X_train_scaled, y_train)



acc_scaled_rf_clf = round(rf_clf.score(X_test_scaled, y_test) * 100, 2)



print ('Accuracy RF on test set: ' ,acc_rf_clf)



print ('Accuracy RF on scaled test set: ' ,acc_scaled_rf_clf)
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(rf_clf.feature_importances_,3)})

importances = importances.sort_values('importance',ascending=False).set_index('feature')

importances.head(15)
importances.plot.bar()

plt.title('Feature importance');
df_train_new  = df_train.drop(['S', 'C','Q','Title_Master','Title_Other'], axis=1)

df_test_new  = df_test.drop(['S', 'C','Q','Title_Master','Title_Other'], axis=1)
df_train_new
# Re-training



X_train, X_test, y_train, y_test = train_test_split(df_train_new.drop(['Survived', 'PassengerId'], axis=1), df_train_new['Survived'], test_size = 0.2, random_state=2)



rf_clf.fit(X_train, y_train)



acc_rf_clf_test = round(rf_clf.score(X_test, y_test) * 100, 2)



scaler = MinMaxScaler()



X_train_scaled = scaler.fit_transform(X_train)



# we must apply the scaling to the test set that we computed for the training set

X_test_scaled = scaler.transform(X_test)



rf_clf.fit(X_train_scaled, y_train)



acc_scaled_rf_clf = round(rf_clf.score(X_test_scaled, y_test) * 100, 2)



print ('Accuracy RF on test set: ' ,acc_rf_clf)



print ('Accuracy RF on scaled test set: ' ,acc_scaled_rf_clf)
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



predictions = cross_val_predict(rf_clf, X_train, y_train, cv=3)



cf = confusion_matrix(y_train, predictions)



print(cf)



sns.heatmap(cf,annot=True,cmap='Blues');
import seaborn as sns

sns.heatmap(cf/np.sum(cf), annot=True, fmt='.2%', cmap='Blues');
scaler = MinMaxScaler()



train_submission = scaler.fit_transform(df_train_new.drop(['Survived', 'PassengerId'], axis=1))

test_submission = scaler.transform(df_test_new.drop(['PassengerId'], axis = 1))



rf_clf.fit(train_submission, df_train_new['Survived'])

df_test_new['Survived'] = rf_clf.predict(test_submission)



df_test_new[['PassengerId', 'Survived']].to_csv('Submission.csv', index = False)