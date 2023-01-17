import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%pylab inline



# Get a glimpse of data

titanic_df = pd.read_csv('../input/train.csv')

titanic_df.head()
# Check the information of our data

titanic_df.info()
#Af first fill lost data, fill age data 



# Get avarage, std to calculate the limitaton of random number

# Get NAN number to determine how many data need to generate

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))



# Plot original age data

titanic_df['Age'].dropna().hist(bins=70, ax=axis1, ls='solid', lw=0.2, ec='black')



average_age = titanic_df.Age.mean()

std_age = titanic_df.Age.std()

nan_age_number = titanic_df.Age.isnull().sum()



# Generate 

rand_age = np.random.randint(average_age - std_age, average_age + std_age, 

                             size = nan_age_number)

# Fill in 

titanic_df.loc[np.isnan(titanic_df['Age']), 'Age'] = rand_age



# Plot result

titanic_df['Age'].hist(bins=70, ax=axis2, ls='solid', lw=0.2, ec='black')

axis1.set_title('Before Fill In')

axis1.set_xlabel('Age')

axis1.set_ylabel('People Number')

axis2.set_title('After Fill In')

axis2.set_xlabel('Age')

axis2.set_ylabel('People Number')
# At first drop data it seems useless for this analysis

# they are ID, name, ticket number, embark place, cabin, SibSp, and Parch

titanic_df = titanic_df.drop(['PassengerId','Name','Ticket','Embarked','Cabin','SibSp','Parch'],axis = 1)

titanic_df.head()
# At first let's analyse from sex and age view

# Divide children from male and female type

titanic_df.loc[titanic_df['Age'] <= 16, 'Sex'] = 'child'

titanic_df = titanic_df.drop(['Age'],axis=1)

titanic_df.head()

# Give mroe descriptive labels for Survived and Pclass

titanic_df['Survival'] = titanic_df.Survived.map({0:'Died',1:'Survived'})

titanic_df['Class'] = titanic_df.Pclass.map({1:'1st Class',2:'2nd Class',3:'3rd Class'})



# Child and not child

titanic_df['Child'] = titanic_df.Sex.map({'child':'Is Child','female':'Not Child','male':'Not Child'})

titanic_df.head()
# Draw pictures to see more clearly of the relations

# about sex and age factor



sns.factorplot(data=titanic_df,x='Sex',y='Survived',kind="violin",size=4,aspect=3)

plt.yticks([0,1], ['Died', 'Survived'])



# Plot basic information about sex and age

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))

sns.countplot(data=titanic_df, x='Sex',ax=axis1)

sns.countplot(data=titanic_df,x='Survived',hue='Sex',order=[0,1],ax=axis2)

plt.xticks([0,1], ['Died', 'Survived'])



fig, (axis3,axis4) = plt.subplots(1,2,figsize=(15,5))

# Group data by sex and whether child

sex_survi_groups = titanic_df[['Sex','Survived']].groupby(['Sex'],as_index=True)



#Divide into three groups

men_group = sex_survi_groups.get_group('male')

women_group = sex_survi_groups.get_group('female')

children_group = sex_survi_groups.get_group('child')



# Plot survive rate between different sex

sns.barplot(data=titanic_df[['Sex','Survived']],x='Sex',y='Survived',order=['male','female'],ax=axis3)

axis3.set_ylabel("Survival Rate")



# Draw Child and Non-Child plot

sns.barplot(data=titanic_df[['Child', 'Survived']],x='Child',y='Survived',order=['Is Child','Not Child'],ax=axis4)

axis4.set_ylabel("Survival Rate")



axis3.set_title('Survive rate compare by Sex')

axis4.set_title('Survive rate compare by whether child')
# Statistic Hypothesis Test

# Chi-Square Test for Independence

# State the hypothesis: H0: Gender and survival rate are independent

from scipy.stats import chi2_contingency



men_women_group = pd.concat([men_group, women_group])

gender_pivot = pd.pivot_table(data=men_women_group[['Survived','Sex']],index='Survived',columns=['Sex'],

                      aggfunc=len)

chi2, p_value, dof, expected = chi2_contingency(gender_pivot)

print("Results of Chi-Squared test on Sex to Survival.")

print("Chi-Square Score = %s"%str(chi2))

print("Pvalue = %s\n"%str(p_value))
# Test for child and non-child

child_pivot = pd.pivot_table(data=titanic_df[['Survived','Child']],index='Survived',columns=['Child'],

                      aggfunc=len)

chi2, p_value, dof, expected = chi2_contingency(child_pivot)

print("Results of Chi-Squared test on Child to Survival.")

print("Chi-Square Score = %s"%str(chi2))

print("Pvalue = %s\n"%str(p_value))
# Then let's analyze class factor

sns.factorplot(data=titanic_df,x='Class',y='Survived',kind="violin", \

               order=['1st Class','2nd Class','3rd Class'],size=4,aspect=3)

plt.yticks([0,1],['Died','Survived'])



# Group by class and take mean

class_survi_prec = titanic_df[['Class','Survived']].groupby(['Class'],as_index=False).mean()



# Compare number and survived rate between three classes

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))

sns.countplot(data=titanic_df, x='Class',order=['1st Class','2nd Class','3rd Class'],ax=axis1)

sns.barplot(data=class_survi_prec,x='Class',y='Survived', \

            order=['1st Class','2nd Class','3rd Class'],ax=axis2)

axis2.set_ylabel('Survival Rate')
# Statistic Hypothesis Test:

# H0: Class and Survival rate are independent

class_pivot = pd.pivot_table(data=titanic_df[['Survived','Class']],index='Survived',columns=['Class'],

                            aggfunc=len)

chi2, p_value, dof, expected = chi2_contingency(class_pivot)

print("Results of Chi-Squared test on Class to Survival.")

print("Chi-Square Score = %s"%str(chi2))

print("Pvalue = %s\n"%str(p_value))
# Last let's analyze fare factor



# Try to plot on a logarithmic x-axis as comment suggests, but it not looks so good 

# fig = titanic_df['Fare'].plot(kind='hist', figsize=(15,3), bins=100, logx=True, 

#                               ls='solid', lw=1, ec='black')



fig = titanic_df['Fare'].plot(kind='hist', figsize=(15,3), bins=100, \

                             ls='solid', lw=0.5, ec='black')

ax = fig.axes

ax.set_xlabel('Fare')

ax.set_ylabel('People Number')

ax.set_title('People Distribution with Fare')



# We clear out people have very high fare

normal_people = titanic_df[['Fare','Survived']][titanic_df['Fare']<200]

fare_survi_group = normal_people[['Fare','Survived']].groupby(['Survived'],as_index=False)



# Survive condition for people with normal fare

figure(2)

sns.factorplot(data=normal_people,x='Survived',y='Fare',aspect=2)

plt.xticks([0,1],['Died','Survived'])
# Statitic Test, variable is continuous, so we choose T-test

# H0: People survived and not survived have same fare, mean(survive_fare)=mean(non_survive_fare)

from scipy.stats import ttest_ind



ttest_ind(fare_survi_group.get_group(0)['Fare'],fare_survi_group.get_group(1)['Fare'])
# Obviously We can guess fare is related to passenger class

# from scatter Plot we can see only first class have very high fare

titanic_df.plot.scatter(x='Pclass',y='Fare')

plt.xticks([1,2,3],['1st Class','2nd Class','3rd Class'])



# We calculate their correlation to confirm

titanic_df[['Fare', 'Pclass']].corr(method='spearman')
# To explore more details

# let's see sex distrubution in different classes

figure(figsize=(8,5))

sns.countplot(data=titanic_df,x='Class',hue='Sex',order=['1st Class','2nd Class','3rd Class'])
# From above we could see class 3 have large percent of men

# So we can guess the low survived rate of men is caused by class3 men

# the survive rate in higher class between sex may not very distinct



# Draw chart of different classes's survive rate detail

class_sex_group = titanic_df[['Sex','Class','Survived']].groupby(['Sex','Class'],as_index=False)

class_sex_survive_prec = class_sex_group.mean()



figure(figsize=(8,5))

fig = sns.barplot(data=class_sex_survive_prec, x='Sex',y='Survived',hue='Class', \

                  order=['male','female','child'])

fig.axes.set_ylabel('Survival Rate')
# Between class1 and class2 women they have similar survive rates

# Chi-Square test

# H0 = For Class1 and Class2 female, the survive rate and class is independent

female_class1_class2 = titanic_df[(titanic_df['Sex']=='female') \

                                  & ((titanic_df['Class']=='1st Class') \

                                     | (titanic_df['Class']=='2nd Class') )]



class_pivot = pd.pivot_table(data=female_class1_class2[['Survived','Class']],index='Survived',columns=['Class'],

                            aggfunc=len)

chi2, p_value, dof, expected = chi2_contingency(class_pivot)

print("Results of Chi-Squared test on Class to Survival on upper two classes female.")

print("Chi-Square Score = %s"%str(chi2))

print("Pvalue = %s\n"%str(p_value))
# Also between class1 and class2 child they have much similar survive rates

# Do test

child_class1_class2 = titanic_df[(titanic_df['Sex']=='child') \

                                  & ((titanic_df['Class']=='1st Class') \

                                     | (titanic_df['Class']=='2nd Class') )]



class_pivot = pd.pivot_table(data=child_class1_class2[['Survived','Class']],index='Survived',columns=['Class'],

                            aggfunc=len)

chi2, p_value, dof, expected = chi2_contingency(class_pivot)

print("Results of Chi-Squared test on Class to Survival on upper two classes child.")

print("Chi-Square Score = %s"%str(chi2))

print("Pvalue = %s\n"%str(p_value))
# And class2 and class3 male they also have similar survive rate

male_class2_class3 = titanic_df[(titanic_df['Sex']=='male') \

                                  & ((titanic_df['Class']=='3rd Class') \

                                     | (titanic_df['Class']=='2nd Class') )]



class_pivot = pd.pivot_table(data=male_class2_class3[['Survived','Class']],index='Survived',columns=['Class'],

                            aggfunc=len)

print("Results of Chi-Squared test on Class to Survival on lower two classes male.")

print("Chi-Square Score = %s"%str(chi2))

print("Pvalue = %s\n"%str(p_value))