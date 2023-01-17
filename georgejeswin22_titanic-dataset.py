# Importing all the necessary packages.

%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Loading the csv file as a pandas dataframe

titanic_df = pd.read_csv('../input/train.csv')
# Studying the Titanic data set

titanic_df.head()
# To get the idea of total number of rows and columns in the data frame.

titanic_df.shape
# To get details such as count, maximum, minimum, mean , etc., which helps in improving the analysis.

titanic_df.describe()
titanic_df["Age"]= titanic_df["Age"].fillna(titanic_df["Age"].median())

titanic_df["Embarked"] = titanic_df["Embarked"].fillna('S')
sex_count = titanic_df['Sex'].value_counts()

total_males_onboard = sex_count['male']

total_females_onboard = sex_count['female']

Passengers_onboard = titanic_df.groupby('Sex').count()['PassengerId']



print("Total number of males onboard the Titanic : {}.".format(total_males_onboard))

print("Total number of females onboard the Titanic : {}.".format(total_females_onboard))
# To count the number of males and females who survived



survived_sex = titanic_df.groupby('Sex').sum()['Survived']

survived_males = survived_sex['male']

survived_females = survived_sex['female']



print("Number of males survived : {}.".format(survived_males))

print("Number of females survived : {}.".format(survived_females))
print("Proportion of people suvived : ",(survived_males + survived_females)/(total_males_onboard + total_females_onboard))

print("Proportion of males survived : ", survived_males/total_males_onboard)

print("Proportion of females survived : ",survived_females/total_females_onboard)
survived_passengers_sex = pd.concat([Passengers_onboard,survived_sex],axis='columns')

survived_passengers_sex.columns.values[0] = 'Passengers onboard'

survived_passengers_sex.columns.values[1] = 'Survived passengers'

survived_passengers_sex
survived_passengers_sex.plot(kind='bar')

plt.xlabel('People classified based on sex')

plt.ylabel('Total number')

plt.title('Compare the survival of passengers based on their sex')
# Creating an additional column in the data frame

# Creating a list to store data

adult_child = []



for i in range(len(titanic_df['Sex'])):

    if titanic_df['Sex'].iloc[i] == 'male':

        if titanic_df['Age'].iloc[i] > 18:

            adult_child.append("Male_adult")

        else:

            adult_child.append("Male_child")

            

    else:

        if titanic_df['Age'].iloc[i] > 18:

            adult_child.append("Female_adult")

        else:

            adult_child.append("Female_child")



titanic_df['adult_child'] = adult_child  # Additional column added to the dataframe
# The modified dataframe

titanic_df.head()
adult_and_children_onboard = titanic_df['adult_child'].value_counts()

adult_and_children_onboard
print("Total number of male adults on the Titanic : {}.".format(adult_and_children_onboard['Male_adult']))

print("Total number of male children on the Titanic : {}.".format(adult_and_children_onboard['Male_child']))

print("Total number of females adults on the Titanic : {}.".format(adult_and_children_onboard['Female_adult']))

print("Total number of females children on the Titanic : {}.".format(adult_and_children_onboard['Female_child']))
a = titanic_df['adult_child']

b = titanic_df['Survived']

Survivors = a[b==1]  # Finding the survived people with help of indexing feature in pandas series

print('Clssifying Survivors based on their age and sex')

survived_people = Survivors.value_counts()

survived_people
survivors_people_df = pd.concat([adult_and_children_onboard, survived_people], axis=1)

survivors_people_df.columns.values[0] = 'Boarded the Titanic'

survivors_people_df.columns.values[1] = 'Survived'

survivors_people_df
# Finding ot the proportion of people (based on the above classifcation) who survived

classification_proportion = survived_people/adult_and_children_onboard

print('Proportion of the people who survived')

classification_proportion
survivors_people_df.plot(kind='bar')

plt.xlabel('People classified based on age and sex')

plt.ylabel('Total number')

plt.title('To compare the total people and survivors')
# Here I have used the Survived column in the dataframe as the index array to obtain only the sex and age details of each person.

Survivors_age = titanic_df['Age'][titanic_df['Survived']==1]

Survivors_sex = titanic_df['Sex'][titanic_df['Survived']==1]
sns.set(style="whitegrid")

sns.swarmplot(x=Survivors_sex, y=Survivors_age)

plt.title('Age of each sex which survived the most')
passengers_from_port = titanic_df.groupby('Embarked').count()['PassengerId']

Survivors_each_port = titanic_df.groupby('Embarked').sum()['Survived']

compare_port_survivors = pd.concat([passengers_from_port, Survivors_each_port], axis=1)

compare_port_survivors.columns.values[0] = 'Total_people'   # Rename the 1st column

compare_port_survivors 
print("Number of passengers onboard from Cherbourg : ",passengers_from_port['C'])

print("Number of survivors from Cherbourg : ",Survivors_each_port['C'])

print('\n')

print("Number of passengers onboard from Queenstown : ",passengers_from_port['Q'])

print("Number of survivors from Queenstown : ",Survivors_each_port['Q'])

print('\n')

print("Number of passengers onboard from Southampton : ",passengers_from_port['S'])

print("Number of survivors from Southampton : ",Survivors_each_port['S'])
Proportion_of_survivors_each_port = Survivors_each_port/passengers_from_port

print('Proportion of people survived from port Cherbourg : ',Proportion_of_survivors_each_port['C'] )

print('Proportion of people survived from port Queenstown : ',Proportion_of_survivors_each_port['Q'] )

print('Proportion of people survived from port Southampton : ',Proportion_of_survivors_each_port['S'] )
compare_port_survivors.plot(kind='bar')

plt.xlabel('Ports of Embarkation')

plt.ylabel('Total number')

plt.title('To compare the total people boarded and survivors from each port')
Passengers_onboard_perclass = titanic_df['Pclass'].value_counts()

Passengers_survived_perclass = titanic_df.groupby('Pclass').sum()['Survived']

compare_survivors_pclass = pd.concat([Passengers_onboard_perclass, Passengers_survived_perclass], axis=1)

compare_survivors_pclass.columns.values[0] = 'Total_people'   # Rename the 1st column

compare_survivors_pclass
Proportion_of_people_survived_perclass = Passengers_survived_perclass/Passengers_onboard_perclass

print('Proportion of people surviving from Class 1 : ',Proportion_of_people_survived_perclass[1])

print('Proportion of people surviving from Class 2 : ',Proportion_of_people_survived_perclass[2])

print('Proportion of people surviving from Class 3 : ',Proportion_of_people_survived_perclass[3])
compare_survivors_pclass.plot(kind='bar')

plt.xlabel('Passenger Class')

plt.ylabel('Total number')

plt.ylabel('Total number')

plt.title('To compare the total people boarded and survivors from each passenger class')
titanic_df['Fare'].hist(bins=[0,50,100,200,300,550])

plt.xlabel('Passenger Fares for Titanic')

plt.ylabel('Total number')

plt.title('Histogram of pasenger fares on Titanic')
t = []

for i in range(len(titanic_df)):

    if titanic_df['Survived'].iloc[i]:

            t.append(titanic_df['Fare'].iloc[i])

survivors_only_fares = pd.Series(t)

survivors_only_fares.hist()

plt.xlabel('Passenger Fares for Titanic')

plt.ylabel('Total number')

plt.title('Histogram of pasenger fares on Titanic')
Total_children_onboard = adult_and_children_onboard['Male_child'] + adult_and_children_onboard['Female_child']

Total_survived_children = survived_people['Male_child'] + survived_people['Female_child']

children_with_parents = 0

children_without_parents = 0

Total_survived_children_with_parents = 0

Total_survived_children_without_parents = 0

for i in range(len(titanic_df)):

    if titanic_df['adult_child'].iloc[i] == 'Female_child' or titanic_df['adult_child'].iloc[i] == 'Male_child':

        if titanic_df['Parch'].iloc[i]!=0:

            children_with_parents += 1

            if titanic_df['Survived'].iloc[i]==1:

                Total_survived_children_with_parents += 1

        else:

            children_without_parents += 1

            if titanic_df['Survived'].iloc[i]==1:

                Total_survived_children_without_parents += 1

                



        

print('Total Children onboard : ',Total_children_onboard)

print('Total survived children : ',Total_survived_children)

print('\n')

print('Children with parents onboard : ',children_with_parents)

print('Survived children who had parents onboard : ',Total_survived_children_with_parents)

print('\n')

print('Children without parents onboard : ',children_without_parents)

print('Survived children without parents onboard : ',Total_survived_children_without_parents)    

print('Proportion of children survived who had parents onboard : ',Total_survived_children_with_parents/children_with_parents)

print('Proportion of children survived without parents onboard : ',

      Total_survived_children_without_parents/children_without_parents)
people_with_sibsp = 0

survived_sibsp = 0

survived_without_sibsp = 0

for i in range(len(titanic_df)):

    if titanic_df['SibSp'].iloc[i]:

        people_with_sibsp += 1

        if titanic_df['Survived'].iloc[i]:

            survived_sibsp += 1

    else:

        if titanic_df['Survived'].iloc[i]:

            survived_without_sibsp += 1

            

            

            

people_without_sibsp = titanic_df['PassengerId'].count() - people_with_sibsp

print("Total people with siblings/spouse onboard : ",people_with_sibsp)

print("Survived people with siblings/spouse onboard : ",survived_sibsp)

print('\n')

print("Total people without any siblings/spouse onboard : ",people_without_sibsp)

print("Survived people without siblings/spouse onboard : ",survived_without_sibsp)

print('\n')



print("Proportion of survived people having siblings/spouse on board : ", survived_sibsp/people_with_sibsp)

print("Proportion of survived people without siblings/spouse on board : ", survived_without_sibsp/people_without_sibsp)


