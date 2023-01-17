# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import statistics

import statsmodels.api as sm

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.linear_model import LogisticRegression

from statsmodels.stats.outliers_influence import variance_inflation_factor
# Loading the dataset

train_dataset = pd.read_csv("../input/train.csv")

test_dataset = pd.read_csv("../input/test.csv")



# putting both the dataframe in one list

containers = [train_dataset,test_dataset]
containers[0].head()
containers[1].head()
# Finding the number of emply cells in training and test data



for container in containers:

    print(container.isnull().sum())

    print("\n\n\n")
for container in containers:

    print(container.info())

    print("\n\n\n")
# Writing functions to create derived columns



def age_group_creation(x):

    if x>=71 and x<=80:

        return '71-80'

    elif x>=61:

        return '61-70'

    elif x>=51:

        return '51-60'

    elif x>=41:

        return '41-50'

    elif x>=31:

        return '31-40'

    elif x>=21:

        return '21-30'

    elif x>=11:

        return '11-20'

    elif x>=0:

        return '0-10'

    else:

        return 'None'



def fare_group_creation(x):

    if x>500 and x<=550:

        return '501-550'

    elif x>=451:

        return '451-500'

    elif x>=401:

        return '401-450'

    elif x>=351:

        return '351-400'

    elif x>=301:

        return '301-350'

    elif x>=251:

        return '251-300'

    elif x>=201:

        return '201-250'

    elif x>=151:

        return '151-200'

    elif x>=101:

        return '101-150'

    elif x>=51:

        return '51-100'

    elif x>=0:

        return '0-50'

    else:

        return 'None'



def title_extractor(x):

    return (x[x.find(',')+1:x.find('.')]).strip()



# Creating derived columns for training and test data



def create_derived(containers):

    for container in containers:

        container['age_cat'] = container['Age'].apply(age_group_creation)

        container['fare_cat'] = container['Fare'].apply(fare_group_creation)

        container['Travel_alone'] = (container['SibSp'] + container['Parch']).apply(lambda x: 0 if x>0 else 1)

        container['title'] = container['Name'].apply(title_extractor)

        container['Ticket_init'] = container['Ticket'].apply(lambda x: x[0])

        

create_derived(containers)

containers[1].head()
containers[0]['Embarked'].value_counts()
containers[0][containers[0]['Embarked'].isnull() == True]
containers[0].loc[containers[0]['fare_cat'] == '51-100',:]['Embarked'].value_counts()
containers[0].loc[containers[0]['Ticket'].apply(lambda x: x.startswith('1135')),:]['Embarked'].value_counts()


for container in containers:

    print(container.loc[(container['fare_cat'] == '51-100') & (container['Pclass'] == 1) & (container['Sex'] == 'female') 

                      & (container['Travel_alone'] == 1) & (container['Ticket_init'] == '1'),:]['Embarked'].value_counts())

    print("\n\n\n")
# As we find that frequency of Embarkment 'S' is the most, we filled the emply cell with 'S'



containers[0].loc[containers[0]['Embarked'].isnull(), 'Embarked'] = 'S'
# Checking if there is a relationship between age and fare



print(pd.crosstab(containers[0]['fare_cat'], containers[0]['age_cat'], margins = True)) # train data

print("\n\n\n")

print(pd.crosstab(containers[1]['fare_cat'], containers[0]['age_cat'], margins = True)) # test data
# # Checking if there is a relationship between travling alone and age



print(pd.crosstab(containers[0]['age_cat'], containers[0]['Travel_alone'], margins = True)) # train data

print("\n\n\n")

print(pd.crosstab(containers[1]['age_cat'], containers[0]['Travel_alone'], margins = True)) # test data
# Checking the relationship between title and age category



print(pd.crosstab(containers[0]['age_cat'], containers[0]['title'], margins = True))

print("\n\n\n")

print(pd.crosstab(containers[1]['age_cat'], containers[0]['title'], margins = True))



print("A clear relationship exists as we see that 'Master' which is used for young male, hold true in our data")
# Observing the age data with respect to title (train_dataset)

for title in containers[0]['title'].unique():

    print(title+"\n")

    print(containers[0].loc[(containers[0]['title'] == title) & (containers[0]['Age'].notnull()),['Age']].describe())

    print("\n\n\n")

# Observing the age data with respect to title (test_dataset)

for title in containers[1]['title'].unique():

    print(title+"\n")

    print(containers[1].loc[(containers[1]['title'] == title) & (containers[1]['Age'].notnull()),['Age']].describe())

    print("\n\n\n")
# From above observation it is clear that 'Age' in both train and test data are comparable with respect to age.

# Therefore to fill the missing values, a dictionary is created using train data with median value in each age category



age_impute = {}



for title in containers[0]['title'].unique():

    age_impute[title] = containers[0].loc[(containers[0]['title'] == title) & (containers[0]['Age'].notnull()),['Age']].median()[0]



# Filing the missing age value in both train and test data

for container in containers:

    for title in age_impute.keys():

        container.loc[(container['title'] == title) & (container['Age'].isnull()),['Age']] = age_impute[title]
age_impute
containers[1].loc[containers[1]['Fare'].isnull(),:]


for container in containers:

    print(container.loc[(container['Pclass'] == 3) & (container['title'] == 'Mr') & (container['Sex'] == 'male') & 

                      (container['Embarked'] == 'S') & (container['age_cat'] == '51-60') & (container['Travel_alone'] == 1) &

                       (container['Ticket_init'] == '3'), :]['Fare'])

    print("\n\n\n")
containers[0].loc[(containers[0]['Pclass'] == 3) & (containers[0]['title'] == 'Mr') & (containers[0]['Sex'] == 'male') & 

                      (containers[0]['Embarked'] == 'S') & (containers[0]['age_cat'] == '51-60') & (containers[0]['Travel_alone'] == 1) &

                       (containers[0]['Ticket_init'] == '3'), :]['Fare'].describe()
# Using median value to impute the missing fare data

containers[1].loc[containers[1]['Fare'].isnull(),['Fare']] = containers[0].loc[(containers[0]['Pclass'] == 3) & (containers[0]['title'] == 'Mr') & (containers[0]['Sex'] == 'male') & 

                      (containers[0]['Embarked'] == 'S') & (containers[0]['age_cat'] == '51-60') & (containers[0]['Travel_alone'] == 1) &

                       (containers[0]['Ticket_init'] == '3'), :]['Fare'].median()
# Droping Cabin column as more than 75% of the data in that columns are null



for container in containers:

    container.drop(['Cabin'], axis = 1, inplace = True)
# Finding the number of emply cells in training and test data



for container in containers:

    print(container.isnull().sum() / len(container.index))

    print("\n\n\n")
# Calling function create_derived() to fill the missing age_cat



create_derived(containers)



#containers[0].info()
# Finding the number of emply cells in training and test data



for container in containers:

    print(container.isnull().sum() / len(container.index))

    print("\n\n\n")
containers[0].head()
plt.figure(figsize = (10,12))



sns.set(font_scale = 1.2)

sns.countplot(x = 'Pclass', hue = 'Survived', data = containers[0])

plt.suptitle("Number of people survived and not survived in each class")

plt.xlabel("Passenger's class")



plt.show()



temp = containers[0].groupby(['Pclass'])['Survived'].value_counts()



xper = round((temp[1,1] / temp.loc[[1],[0,1]].sum()) * 100,2)

print("Percentage of people survived from 1st class ticket is "+str(xper))



xper = round((temp[2,1] / temp.loc[[2],[0,1]].sum()) * 100,2)

print("Percentage of people survived from 2nd class ticket is "+str(xper))



xper = round((temp[3,1] / temp.loc[[3],[0,1]].sum()) * 100,2)

print("Percentage of people survived from 2nd class ticket is "+str(xper))



print("\n\nWe see that 3rd class passengers were very less likely to survive")


plt.figure(figsize = (15,10))



# for 1st class

plt.subplot(1,3,1)



temp = containers[0].loc[containers[0]['Pclass'] == 1,:]

age_cat = containers[0].loc[containers[0]['Pclass'] == 1,:]['age_cat'].unique()

age_cat.sort()



sns.countplot(x = 'age_cat', hue = 'Survived', order = age_cat, data = temp)

plt.xticks(rotation = 90)

plt.xlabel("First class passengers")





# for second class

plt.subplot(1,3,2)



temp = containers[0].loc[containers[0]['Pclass'] == 2,:]

age_cat = containers[0].loc[containers[0]['Pclass'] == 2,:]['age_cat'].unique()

age_cat.sort()



sns.countplot(x = 'age_cat', hue = 'Survived', order = age_cat, data = temp)

plt.xticks(rotation = 90)

plt.xlabel("Second class passengers")





# for third class

plt.subplot(1,3,3)



temp = containers[0].loc[containers[0]['Pclass'] == 3,:]

age_cat = containers[0].loc[containers[0]['Pclass'] == 3,:]['age_cat'].unique()

age_cat.sort()



sns.countplot(x = 'age_cat', hue = 'Survived', order = age_cat, data = temp)

plt.xticks(rotation = 90)

plt.xlabel("Third class passengers")



plt.show()



print("Person with age between 0-10 ,with second class ticket survived the most")


plt.figure(figsize = (15,10))



# for 1st class

plt.subplot(1,3,1)



temp = containers[0].loc[containers[0]['Pclass'] == 1,:]



sns.countplot(x = 'Sex', hue = 'Survived', data = temp)

plt.xlabel("First class passengers")





# for second class

plt.subplot(1,3,2)



temp = containers[0].loc[containers[0]['Pclass'] == 2,:]



sns.countplot(x = 'Sex', hue = 'Survived', data = temp)

plt.xlabel("Second class passengers")





# for third class

plt.subplot(1,3,3)



temp = containers[0].loc[containers[0]['Pclass'] == 3,:]



sns.countplot(x = 'Sex', hue = 'Survived', data = temp)

plt.xlabel("Third class passengers")



plt.show()



print("A female person with 1st class ticket survived the most")
plt.figure(figsize = (15,10))



# for 1st class

plt.subplot(1,3,1)



temp = containers[0].loc[containers[0]['Pclass'] == 1,:]



sns.countplot(x = 'Travel_alone', hue = 'Survived', data = temp)

plt.xticks([0,1],['With someone', 'Alone'])

plt.xlabel("First class passengers")





# for second class

plt.subplot(1,3,2)



temp = containers[0].loc[containers[0]['Pclass'] == 2,:]



sns.countplot(x = 'Travel_alone', hue = 'Survived', data = temp)

plt.xticks([0,1],['With someone', 'Alone'])

plt.xlabel("Second class passengers")





# for third class

plt.subplot(1,3,3)



temp = containers[0].loc[containers[0]['Pclass'] == 3,:]



sns.countplot(x = 'Travel_alone', hue = 'Survived', data = temp)

plt.xticks([0,1],['With someone', 'Alone'])

plt.xlabel("Third class passengers")



plt.show()



print("There is a higher chance of surviving if you are with someone")
# Finding the relationship between survived and other data



temp = pd.crosstab(containers[0]['Survived'], containers[0]['Sex'], margins = True)

x1 = round(((temp['female'][1] / temp['female'][2]) * 100), 2)

x2 = round(((temp['male'][1] / temp['male'][2]) * 100), 2)



plt.figure(figsize = (8,10))



sns.set(font_scale = 1.2)

sns.countplot(x = 'Sex', hue = 'Survived', data = containers[0])

plt.suptitle("Male and Female survival rate")

plt.xlabel("Gender")

plt.ylabel("survived")



plt.show()



print("Percentage of male survuved is "+str(x2))

print("Percentage of femal survuved is "+str(x1))

print("\n\nA female person had a higher chance of survival")
plt.figure(figsize = (10,12))



sns.set(font_scale = 1.2)



age_cat = containers[0]['age_cat'].unique()

age_cat.sort()

sns.countplot(x = 'age_cat', hue = 'Survived', order = age_cat, data = containers[0])



plt.show()



temp = pd.crosstab(containers[0]['age_cat'], containers[0]['Survived'], margins = True)

temp



for age in age_cat:

    per_age_cat = round((temp.loc[age,1] / temp.loc[age,'All']) * 100 , 2)

    print("Percentage of people survived in age category "+age+" is "+str(per_age_cat))



print("\n\nTherefore we see that childern between age 0-10 has the highest percentage of being saved")

print("\nAlso person above 60 years of age has less percentage of being saved")
plt.figure(figsize = (8,10))



sns.set(font_scale = 1.2)

sns.countplot(x = 'Travel_alone', hue = 'Survived', data = containers[0])

plt.xticks([0,1],['With someone', 'Alone'])



plt.show()



temp = pd.crosstab(containers[0]['Travel_alone'], containers[0]['Survived'], margins = True)

x1 = round((temp.loc[0,1] / temp.loc[0,'All']) * 100, 2)

x2 = round((temp.loc[1,1] / temp.loc[1,'All']) * 100, 2)



print("Percentage of people survived who were travelling with someone "+str(x1))

print("Percentage of people survived who were travelling alone "+str(x2))



print("\n\nChances of survival are higher if a person is not travelling alone")
plt.figure(figsize = (8,10))



sns.set(font_scale = 1.2)

sns.countplot(x = 'Embarked', hue = 'Survived', data = containers[0])

#

plt.show()



temp = pd.crosstab(containers[0]['Embarked'], containers[0]['Survived'], margins = True)

x1 = round((temp.loc['C', 1] / temp.loc['C', 'All']) * 100, 2)

x2 = round((temp.loc['S', 1] / temp.loc['S', 'All']) * 100, 2)

x3 = round((temp.loc['Q', 1] / temp.loc['Q', 'All']) * 100, 2)



print("% of people survived, who boarded from Cherbourg is "+str(x1))

print("% of people survived, who boarded from Southampton is "+str(x2))

print("% of people survived, who boarded from Queenstown is "+str(x3))

# We see that women had a better chance of surviving than man.

# People with class 1 ticket had better chance of surviving then the people from other class.

# Childern aged between 0-10 had higher chances of surviving.

# Also second class passengers aged between 0-10 years had higher chances of surviving.

# Ppeople with age between 60-80 years were less likely to survive

# Poeple who boarded from Cherbourg survived more.

# Person is traveling with someone, had igher percentage of surviving members.
temp = containers[0].loc[((containers[0]['Pclass'] == 1) & (containers[0]['Sex'] == 'female')

                   & (containers[0]['Embarked'] == 'C')& (containers[0]['Travel_alone'] == 0)),:]



print(temp['Survived'].value_counts())



print("\n\nTherefore we see that all first class passenger female who boarded from Cherbourg and traveled with someone survived")
containers[0].head()
# Converting categorical variables



for container in containers:

    container['Sex'] = container['Sex'].apply(lambda x : 0 if x == 'female' else 1)



# For train data

# Creating a dummy variable for the variable 'Pclass' and dropping the first one.

temp = pd.get_dummies(containers[0]['Pclass'], prefix='Pclass', drop_first=True)

#Adding the results to the master dataframe

containers[0] = pd.concat([containers[0],temp],axis=1)



# Creating a dummy variable for the variable 'Embarked' and dropping the first one.

temp = pd.get_dummies(containers[0]['Embarked'], prefix='Embarked', drop_first=True)

#Adding the results to the master dataframe

containers[0] = pd.concat([containers[0],temp],axis=1)



# Creating a dummy variable for the variable 'title' and dropping the first one.

#temp = pd.get_dummies(containers[0]['title'], prefix='title', drop_first=True)

#Adding the results to the master dataframe

#containers[0] = pd.concat([containers[0],temp],axis=1)



# For test data

# Creating a dummy variable for the variable 'Pclass' and dropping the first one.

temp = pd.get_dummies(containers[1]['Pclass'], prefix='Pclass', drop_first=True)

#Adding the results to the master dataframe

containers[1] = pd.concat([containers[1],temp],axis=1)



# Creating a dummy variable for the variable 'Embarked' and dropping the first one.

temp = pd.get_dummies(containers[1]['Embarked'], prefix='Embarked', drop_first=True)

#Adding the results to the master dataframe

containers[1] = pd.concat([containers[1],temp],axis=1)



# Creating a dummy variable for the variable 'title' and dropping the first one.

#temp = pd.get_dummies(containers[1]['title'], prefix='title', drop_first=True)

#Adding the results to the master dataframe

#containers[1] = pd.concat([containers[1],temp],axis=1)
# Dropping the categorical variables for which dummy varible has been created.

# Also droping few variables which we will not require in prediction.



containers[0].drop(['Pclass','Embarked','title','PassengerId','Name','Ticket','age_cat','fare_cat','Ticket_init'], axis = 1, inplace = True)



containers[1].drop(['Pclass','Embarked','title','Name','Ticket','age_cat','fare_cat','Ticket_init'], axis = 1, inplace = True)
containers[0].head()
plt.figure(figsize = (20,8))



sns.set(font_scale = 1.2)



plt.subplot(1,2,1)



sns.boxplot(x = 'Age', data = containers[0])



plt.subplot(1,2,2)



sns.boxplot(x = 'Fare', data = containers[0])



plt.show()



print("We see both age an fare data has outliers")
# Making headmap

plt.figure(figsize = (30,20))



corr_mat = containers[0].corr()

sns.heatmap(corr_mat, annot = True)



plt.show()
# Feature scaling



containers[0].head()
# Feature scaling



# For train data

df = containers[0][['Age','SibSp','Parch','Fare']]

normalized_df=(df-df.mean())/df.std()

containers[0] = containers[0].drop(['Age','SibSp','Parch','Fare'], 1)

containers[0] = pd.concat([containers[0],normalized_df],axis=1)



# For test data

df = containers[1][['Age','SibSp','Parch','Fare']]

normalized_df=(df-df.mean())/df.std()

containers[1] = containers[1].drop(['Age','SibSp','Parch','Fare'], 1)

containers[1] = pd.concat([containers[1],normalized_df],axis=1)
containers[1].head()
# Dividing the data in dependent and independent values



X = containers[0].drop(['Survived'],axis=1)

y = containers[0]['Survived']
# Splitting the data



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)
# Logistic regression model

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.sort_values(by = 'VIF', axis=0, ascending=False, inplace=True)



vif
# Removing Embarked_Q as it has a very high p-value



X = containers[0].drop(['Survived','Embarked_Q'],axis=1)

y = containers[0]['Survived']



# dividing the data into training data and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)
# Logistic regression model

logm2 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm2.fit().summary()
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.sort_values(by = 'VIF', axis=0, ascending=False, inplace=True)



vif
# Removing Fare as it has a very high p-value



X = containers[0].drop(['Survived','Embarked_Q','Fare'],axis=1)

y = containers[0]['Survived']



# dividing the data into training data and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)
# Logistic regression model

logm3 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm3.fit().summary()
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.sort_values(by = 'VIF', axis=0, ascending=False, inplace=True)



vif
# Removing Embarked_S as it has a very high p-value



X = containers[0].drop(['Survived','Embarked_Q','Embarked_S','Fare'],axis=1)

y = containers[0]['Survived']



# dividing the data into training data and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)
# Logistic regression model

logm4 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm4.fit().summary()
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.sort_values(by = 'VIF', axis=0, ascending=False, inplace=True)



vif
# Removing Travel_alone as it has a very high VIF



X = containers[0].drop(['Survived','Embarked_Q','Embarked_S','Fare','Travel_alone'],axis=1)

y = containers[0]['Survived']



# dividing the data into training data and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)
# Logistic regression model

logm5 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm5.fit().summary()
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.sort_values(by = 'VIF', axis=0, ascending=False, inplace=True)



vif
# Removing Parch and zdding Travel_alone as Parch has a very high p-value



X = containers[0].drop(['Survived','Embarked_Q','Embarked_S','Fare','Parch'],axis=1)

y = containers[0]['Survived']



# dividing the data into training data and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)
# Logistic regression model

logm6 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm6.fit().summary()
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.sort_values(by = 'VIF', axis=0, ascending=False, inplace=True)



vif
# Removing Travel_alone as it has a very high VIF



X = containers[0].drop(['Survived','Embarked_Q','Embarked_S','Fare','Parch','Travel_alone'],axis=1)

y = containers[0]['Survived']



# dividing the data into training data and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 100)
# Logistic regression model

logm7 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm7.fit().summary()
# For each X, calculate VIF and save in dataframe

vif = pd.DataFrame()

vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif["features"] = X.columns

vif.sort_values(by = 'VIF', axis=0, ascending=False, inplace=True)



vif
# Making model using 7th model columns



col = ['Sex', 'Pclass_2', 'Pclass_3', 'Age', 'SibSp']

logsk = LogisticRegression()

logsk.fit(X_train[col], y_train)
# Predicted probabilities

y_pred = logsk.predict_proba(X_test[col])



# Converting y_pred to a dataframe which is an array

y_pred_df = pd.DataFrame(y_pred)



# Converting to column dataframe

y_pred_1 = y_pred_df.iloc[:,[1]]



# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)



# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)



# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df,y_pred_1],axis=1)



# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 1 : 'Survive_Prob'})



# Let's see the head of y_pred_final

y_pred_final.head()
# Let's see the head

y_pred_final.head()
from sklearn import metrics
# Checking with different cut-off percentage to imporve accuracy



num = [float(x)/10 for x in range(10)]

for i in num:

    y_pred_final[i] = y_pred_final['Survive_Prob'].map( lambda x: 1 if x > i else 0)

y_pred_final.head()
# Checking different accuracy for different cut-off predicted values



cutoff_df = pd.DataFrame(columns = ['Cutoff_prob', 'Accuracy', 'Sensitivity', 'Specificity'])

num = [float(x)/10 for x in range(10)]

for i in num:

    conff_mat = metrics.confusion_matrix(y_pred_final['Survived'], y_pred_final[i])

    tot = sum(sum(conff_mat))

    accur = (conff_mat[0,0] + conff_mat[1,1])/tot

    sensi = conff_mat[0,0] / (conff_mat[0,0] + conff_mat[0,1]) # TP/(TP+FN)

    speci = conff_mat[1,1] / (conff_mat[1,1] + conff_mat[1,0]) # TN/(TN + FP)

    cutoff_df.loc[i] = [i, accur, sensi, speci]

    

cutoff_df
# From the above table it is clear that cutoff probability of 0.4 will give high accuracy.

# Therefore using that and predicting



# Creating new column 'predicted' with 1 if Churn_Prob>0.4 else 0

y_pred_final['predicted'] = y_pred_final.Survive_Prob.map( lambda x: 1 if x > 0.4 else 0)



# Let's see the head

y_pred_final.head()
# Confusion matrix 

confusion = metrics.confusion_matrix( y_pred_final.Survived, y_pred_final.predicted )

confusion
# Predicted     not_Survived    Survived

# Actual

# not_Survived        87           17

# Survived            15           60  



#Let's check the overall accuracy.

metrics.accuracy_score( y_pred_final.Survived, y_pred_final.predicted)
# ROC curve





fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_final['Survive_Prob'])

roc_auc = metrics.auc( fpr, tpr )

plt.figure(figsize=(6, 4))

plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc )

plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()



    
TP = confusion[0,0] # True Positive

TN = confusion[1,1] # True Negative

FP = confusion[0,1] # False Positive

FN = confusion[1,0] # False Negative
# Sencitivity

sensitivity = TP / (TP + FN) # percentage of people correctly identified as survived (True positive rate)

specificity = TN / (TN + FP) # percentage of people correctly identified as not survived

precision = TP / (TP + FP)

recall = TP / (TP + FN)

f1score = (2 * precision * recall) / (precision + recall)



print("Sensitivity : "+str(sensitivity))

print("Specificity : "+str(specificity))

print("Precision : "+str(precision))

print("Recall : "+str(recall))

print("f1 score : "+str(f1score))
# Applying K-fold cross validation 



# Creating folds object

folds = KFold(n_splits = 5, shuffle = True, random_state = 4)



# instanciating a model

model = LogisticRegression()



# Computing cross Validation score

cv_results = cross_val_score(model, X_train, y_train, cv = folds, scoring = 'accuracy')
print(cv_results)

print(cv_results.mean())
# Building final model



col = ['Sex', 'Pclass_2', 'Pclass_3', 'Age', 'SibSp']



# Dividing the data in dependent and independent values



X_train = containers[0].drop(['Survived'],axis=1)

y_train = containers[0]['Survived']



X_test = containers[1]



logsk = LogisticRegression()

logsk.fit(X_train[col], y_train)
# Predicted probabilities

y_pred = logsk.predict_proba(X_test[col])



# Converting y_pred to a dataframe which is an array

y_pred_df = pd.DataFrame(y_pred)



# Converting to column dataframe

y_pred_1 = y_pred_df.iloc[:,[1]]



# Converting index to dataframe

y_id = pd.DataFrame(X_test['PassengerId'])



# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_id.reset_index(drop=True, inplace=True)



# Appending y_id and y_pred_1

y_pred_final = pd.concat([y_id,y_pred_1],axis=1)



# Renaming the column 

y_pred_final= y_pred_final.rename(columns={ 1 : 'Survive_Prob'})



# Let's see the head of y_pred_final

y_pred_final.head()
# Predicting with 0.4, 0.5, 0.6, 0.7, 0.8 as cut-off



num = [0.4, 0.5, 0.6, 0.7, 0.8]



for i in num:

    y_pred_final[i] = y_pred_final['Survive_Prob'].map(lambda x: 1 if x>i else 0)

    

y_pred_final.head()
# Here 0.6 was used as it was giving the maximum accuracy

d = pd.DataFrame({'PassengerId': y_pred_final['PassengerId'], 'Survived': y_pred_final[0.6]})

d.set_index('PassengerId', inplace = True)

d.to_csv('predictions.csv', sep=",")