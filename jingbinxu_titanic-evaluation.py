import pandas as pd

import numpy as np

import re



import matplotlib.pyplot as plt

from numpy.random import rand

from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor

from sklearn import tree



# Load the train and test datasets to create two DataFrames

base_url = "../input"

train_url = base_url + "/train.csv"

train = pd.read_csv(train_url)



test_url = base_url + "/test.csv"

test = pd.read_csv(test_url)



# Combine test and training to facilitate cleanup and pre-processing

full_data = pd.concat([train, test], axis=0)



print ("Full data {}\n".format(full_data.shape))



# Lets see how we are doing with missing values

print("Full data missing \n{}\n".format(full_data.isnull().sum()))
# Remove Warnings from Pandas

pd.options.mode.chained_assignment = None  # default='warn'



# Lets fill the missing 'Embarked' values with the most occurred value, which is "S".

# 72.5% of people left from Southampton.

full_data.Embarked.fillna('S', inplace=True)



# There is only one missing 'Fare' and its in the test dataset

# Let's just go ahead and fill the fare value in with the median fare

full_data.Fare.fillna(full_data.Fare.median(), inplace=True)



# Lets take a look at Cabins...

# It looks like 77% and 78% of these fields are empty so I'm going to ignore

# this column for now.

#print("Null Cabins in training {:.4f}".format(1-(train["Cabin"].value_counts().sum()/ len(train["Cabin"]))))

#print("Null Cabins in test {:.4f}".format(1-(test["Cabin"].value_counts().sum()/len(test["Cabin"]))))



# Also ignoring Ticket for now as it is not clear to me what to do with it.
# Create categories for Sex and Embarked

full_data = pd.concat([full_data, pd.get_dummies(full_data['Sex'], prefix='Sex')], axis=1)



full_data = pd.concat([full_data, pd.get_dummies(full_data['Embarked'], prefix='Embarked')], axis=1)
# Extract the title from the name

def get_title(name):

    index_comma = name.index(',') + 2

    title = name[index_comma:]

    index_space = title.index('.') + 1

    title = title[0:index_space]

    return title



# Helper method to show unique_titles

unique_titles = {}

def get_unique_titles(name):

    title = get_title(name)

    if title in unique_titles:

        unique_titles[title] += 1

    else:

        unique_titles[title] = 1



# Uncomment to show the unique titles in the data set

#full_data["Name"].apply(get_unique_titles)

#print(unique_titles)



#Upon review of the unique titles we consolidate on the below mappings as optimal

def map_title(name):

    title = get_title(name)

    #should add no key found exception

    title_mapping = {"Mr.": 1, "Miss.": 2, "Ms.": 10, "Mrs.": 3, "Master.": 4, "Dr.": 5,

                     "Rev.": 6, "Major.": 7, "Col.": 7, "Don.": 7, "Sir.": 7, "Capt.": 7,

                     "Mlle.": 8, "Mme.": 8, "Dona.": 9, "Lady.": 9, "the Countess.": 9,

                     "Jonkheer.": 9}

    return title_mapping[title]





# Create a new field with a Title

full_data["Title"] = full_data["Name"].apply(map_title)



# Extract the last name from the title

def get_last_name(name):

    index_comma = name.index(',')

    last_name = name[0:index_comma:]

    #print(last_name)

    return last_name



# Helper method to show unique_last_names

unique_last_names = {}

def get_unique_last_names(name):

    last_name = get_last_name(name)

    if last_name in unique_last_names:

        unique_last_names[last_name] += 1

    else:

        unique_last_names[last_name] = 1





# Create a new field with last names

full_data["LastName"] = full_data["Name"].apply(get_last_name)



# Create a category by grouping like last names 

full_data["Name"].apply(get_unique_last_names)

full_data["LastNameCount"] = full_data["Name"].apply(lambda x: unique_last_names[get_last_name(x)])
# To set the missing ages we will find the median age for the persons title and use that

# as the age of the person

def map_missing_ages1(df):

    avg_title_age = {}

    # Find median age for all non null passengers

    avg_age_all= df['Age'].dropna().median()

    # Iterate all the titles and set a median age for each title

    for title in range(1,11):

        avg_age = df['Age'][(df["Title"] == title)].dropna().median()

         # If the average age is null for a title defualt back to average for all passengers

        if pd.isnull(avg_age):

            avg_age = avg_age_all

        avg_title_age[title] = avg_age



    # Now that we have a list with average age by title we apply it to all our null passengers

    # Map Ages without data

    for title in range(1,11):

        # print("title code:",title," avg age:",avg_title_age[title])

        df["Age"][(df["Title"] == title) & df["Age"].isnull()] = avg_title_age[title]





# Set the  missing ages by createing a classifier based on the below criteria

def map_missing_ages2(df):

    feature_list = [

                "Fare",

                "Pclass",

                "Parch",

                "SibSp",

                "Title",

                "Sex_female",

                "Sex_male",

                "Embarked_C",

                "Embarked_Q",

                "Embarked_S"

                ]



    etr = ExtraTreesRegressor(n_estimators=200,random_state = 42)



    train = df.loc[df.Age.notnull(),feature_list]

    target = df.loc[df.Age.notnull(),['Age']]



    test = df.loc[df.Age.isnull(),feature_list]

    etr.fit(train,np.ravel(target))



    age_preds = etr.predict(test)

    df.loc[df.Age.isnull(),['Age']] = age_preds



map_missing_ages2(full_data)
#Exploring some of the fare data lookign for patterns



#print(full_data["Fare"].round().value_counts().sort_index())

#www = full_data[(full_data["Fare"]==0.0)]

#www.loc[:,['Survived','Name','Sex','Age','Fare']]
fig, ((axis1, axis2),(axis3, axis4)) = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))



###########################################################

pclass_survived = train['Pclass'][train['Survived']==1].value_counts().sort_index()

pclass_died =     train['Pclass'][train['Survived']==0].value_counts().sort_index()



width = 0.30

x_pos = np.arange(len(pclass_survived))



axis1.bar(x_pos,pclass_survived, width, color='blue', label='Survived')

axis1.bar(x_pos + width, pclass_died, width, color='red', label='Died')

axis1.set_xlabel('Passenger Classes', fontsize=12)

axis1.set_ylabel('Number of people', fontsize=10)

axis1.legend(loc='upper center')

axis1.set_xticklabels(('','First Class','','Second Class','','Third Class'))

axis1.yaxis.grid(True)



###########################################################

embarked_survived = train['Embarked'][train['Survived']==1].value_counts().sort_index()

embarked_died     = train['Embarked'][train['Survived']==0].value_counts().sort_index()



#print(embarked_died)

#print(embarked_survived)

x_pos = np.arange(len(embarked_survived))

axis2.bar(x_pos,embarked_survived, width, color='blue', label='Survived')

axis2.bar(x_pos + width, embarked_died, width, color='red', label='Died')

axis2.set_xlabel('Embarked From', fontsize=12)

axis2.set_ylabel('Number of people', fontsize=10)

axis2.legend(loc='upper center')

axis2.set_xticklabels(('','Cherbourg','','Queenstown','','Southamton'))

axis2.yaxis.grid(True)



###########################################################

# Age fill has an interesting spike based on the above fill of empty ages

age_survived = train['Age'][train['Survived']==1].value_counts().sort_index()

age_died     = train['Age'][train['Survived']==0].value_counts().sort_index()



minAge, maxAge = min(train.Age), max(train.Age)

bins = np.linspace(minAge, maxAge, 100)



# You can squash the distribution with a log function but I prefered to see the outliers

#axis3.bar(np.arange(len(age_survived)), np.log10(age_survived), color='blue', label='Survived')

#axis3.bar(np.arange(len(age_died)), -np.log10(age_died), color='red', label='Died')



axis3.bar(np.arange(len(age_survived)), age_survived, color='blue', label='Survived')

axis3.bar(np.arange(len(age_died)), -(age_died), color='red', label='Died')

#axis3.set_yticks(range(-3,4), (10**abs(k) for k in range(-3,4)))

axis3.legend(loc='upper right',fontsize="x-small")

axis3.set_xlabel('Age', fontsize=12)

axis3.set_ylabel('Number of people', fontsize=10)



###########################################################

# Chart Fare by Survived and Perished

fair_survived = train['Fare'][train['Survived']==1].value_counts().sort_index()

fair_died     = train['Fare'][train['Survived']==0].value_counts().sort_index()



minAge, maxAge = min(train.Age), max(train.Age)

bins = np.linspace(minAge, maxAge, 100)



axis4.bar(np.arange(len(fair_survived)), fair_survived, color='blue', label='Survived')

axis4.bar(np.arange(len(fair_died)), -(fair_died), color='red', label='Died')

#axis4.set_yticks(range(-3,4), (10**abs(k) for k in range(-3,4)))

axis4.legend(loc='upper right',fontsize="x-small")

axis4.set_xlabel('Fair', fontsize=12)

axis4.set_ylabel('Number of people', fontsize=10)



plt.show()



###########################################################

fig, (axis1) = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))



single_male_survived = train['Sex'][train['Survived']==1].value_counts().sort_index()

single_male_died =     train['Sex'][train['Survived']==0].value_counts().sort_index()



#print(single_male_survived)

#print(single_male_died)



width = 0.30

x_pos = np.arange(len(single_male_survived))



axis1.bar(x_pos, single_male_survived, width, color='b', label='Survived')

axis1.bar(x_pos + width, single_male_died, width, color='r', label='Died')

axis1.set_xlabel('Gender', fontsize=12)

axis1.set_ylabel('Number of people', fontsize=10)

axis1.set_xticklabels(('','Females','','','','','Males'))

axis1.legend(loc="upper left", fontsize="xx-small",

           ncol=2, shadow=True, title="Legend")

axis1.yaxis.grid(True)



plt.show()
# Lets do some feature engineering



# Assign 1 to passengers under 14, 0 to those 14 or older.

full_data["Child"] = 0

full_data["Child"][full_data["Age"] < 14] = 1



# Create a Mother field (It seems Mother had a pretty high survival rate)

# Note that Title "Miss." = 2 in our mappings

full_data["Mother"] = 0

full_data["Mother"][(full_data["Parch"] > 0) & (full_data["Age"] > 18) &

                    (full_data["Sex"] == 'female') & (full_data["Title"] != 2)] = 1



full_data["FamilySize"] = full_data["SibSp"] + full_data["Parch"]



# Create a Family category none, small, large

full_data["FamilyCat"] = 0

full_data["FamilyCat"][ (full_data["Parch"] + full_data["SibSp"]) == 0] = 0

full_data["FamilyCat"][((full_data["Parch"] + full_data["SibSp"]) > 0) & ((full_data["Parch"] + full_data["SibSp"]) <= 3)] = 1

full_data["FamilyCat"][ (full_data["Parch"] + full_data["SibSp"]) > 3 ] = 2



full_data["SingleMale"] = 0 #0 -- Other ends up being females

full_data["SingleMale"][((full_data["Parch"] + full_data["SibSp"]) == 0) & (full_data["Sex"] == 'male')] = 2

full_data["SingleMale"][((full_data["Parch"] + full_data["SibSp"]) >  0) & (full_data["Sex"] == 'male')] = 1



full_data["AdultFemale"] = 0

full_data["AdultFemale"][(full_data["Age"] > 18) & (full_data["Sex"] == 'female')] = 1



full_data["AdultMale"] = 0

full_data["AdultMale"][(full_data["Age"] > 18) & (full_data["Sex"] == 'male')] = 1
# Create some plots of our New Features

fig, ((axis1, axis2),(axis3, axis4)) = plt.subplots(nrows=2, ncols=2, figsize=(9, 7))

train = full_data.iloc[:891,:]



width = 0.30



###########################################################

single_male_survived = train['SingleMale'][train['Survived']==1].value_counts().sort_index()

single_male_died =     train['SingleMale'][train['Survived']==0].value_counts().sort_index()



x_pos = np.arange(len(single_male_survived))



axis1.bar(x_pos, single_male_survived, width, color='b', label='Survived')

axis1.bar(x_pos + width, single_male_died, width, color='r', label='Died')

axis1.set_xlabel('Male Maritail Status', fontsize=12)

axis1.set_ylabel('Number of people', fontsize=10)

axis1.set_xticklabels(('','Females','','Single Male','','Married Male'))

axis1.legend(loc="upper left", fontsize="xx-small",

           ncol=2, shadow=True, title="Legend")



axis1.annotate('Single Males survive \nbetter than Married', xy=(1.2, 100),xytext=(1, 150),

              arrowprops=dict(facecolor='black', shrink=0.05),)



###########################################################

mother_survived = train['Mother'][train['Survived']==1].value_counts().sort_index()

mother_died =     train['Mother'][train['Survived']==0].value_counts().sort_index()

    

x_pos = np.arange(len(mother_survived))



axis2.bar(x_pos, mother_survived, width, color='b', label='Survived')

axis2.bar(x_pos + width, mother_died, width, color='r', label='Died')

axis2.set_xlabel('Mother Status', fontsize=12)

axis2.set_ylabel('Number of people', fontsize=10)

axis2.set_xticklabels(('','All others','','','','','Mothers'))

axis2.legend(loc="upper right", fontsize="xx-small",

           ncol=2, shadow=True, title="Legend")



###########################################################

family_survived = train['FamilyCat'][train['Survived']==1].value_counts().sort_index()

family_died =     train['FamilyCat'][train['Survived']==0].value_counts().sort_index()

    

x_pos = np.arange(len(family_survived))



axis3.bar(x_pos, family_survived, width, color='b', label='Survived')

axis3.bar(x_pos + width, family_died, width, color='r', label='Died')

axis3.set_xlabel('Family Status', fontsize=12)

axis3.set_ylabel('Number of people', fontsize=10)

axis3.set_xticklabels(('','No Kids','','1 to 3 kids','','> 3 Kids',''))

axis3.legend(loc="upper right", fontsize="xx-small",

           ncol=2, shadow=True, title="Legend")



###########################################################

title_survived = train['Title'][train['Survived']==1].value_counts().sort_index()

title_died =     train['Title'][train['Survived']==0].value_counts().sort_index()



width = 0.40

x_pos_s = np.arange(len(title_survived))

x_pos_d = np.arange(len(title_died))



axis4.bar(x_pos_s, title_survived, width, color='b', label='Survived')

axis4.bar(x_pos_d + width, title_died, width, color='r', label='Died')

axis4.set_xlabel('Title Status', fontsize=12)

axis4.set_ylabel('Number of people', fontsize=10)

axis4.set_xticklabels(('Mr','Miss','Mrs','Mst','Dr','Rev','Sir','Ml','Lady','Ms'))

axis4.legend(loc="upper right", fontsize="xx-small",

           ncol=2, shadow=True, title="Legend")



plt.show()
# Setup calassifier and predict

print(full_data.describe())



feature_list = [

                "Age",

                "Fare",

                "Pclass",

                "Parch",

                "SibSp",

                "Sex_female",

                "Sex_male",

                "Embarked_C",

                "Embarked_Q",

                "Embarked_S",



                #"Title",

                #"Mother",

                #"FamilySize",

                #"FamilyCat",

                #"SingleMale",

                #"AdultFemale",

                #"AdultMale",

                #"LastNameCount",

                "Child"

                ]



# Building and fitting the Random forest

forest_classifier = RandomForestClassifier(max_depth = 6,

                                           min_samples_split=2,

                                           n_estimators = 450,

                                           random_state = 1)





train_all      = full_data.iloc[:891,:]

train_features = train_all.loc[:,feature_list]

train_target   = train_all.loc[:,['Survived']]



test_data      = full_data.iloc[891:,:]

test_features  = test_data.loc[:,feature_list]



my_forest = forest_classifier.fit(train_features, np.ravel(train_target))



pred_forest = my_forest.predict(test_features)



# Create a data frame with two columns: PassengerId & Survived.

PassengerId =np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])



# Write solution to a csv file with the name my_solution.csv

my_solution.to_csv("predict_random_forest.csv", index_label = ["PassengerId"])



# Print the score of the fitted random forest

print(my_forest.feature_importances_)

print(my_forest.score(train_features, train_target))



# Check that the data frame has 418 entries

print(my_solution.shape)