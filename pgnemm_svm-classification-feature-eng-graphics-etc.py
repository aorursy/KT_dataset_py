%matplotlib inline

## Imports



# pandas, numpy

import pandas as pan

import numpy as np



# Gaussian Naive Bayes

from sklearn import datasets

from sklearn import metrics

from sklearn import svm



# Matplotlib, seaborn

import matplotlib.pyplot as plt

import seaborn as sns

## Feature engineering on the dataset



# Function to get the title only in the Name column

def getTitle(name):

    firstSplit = name.split(",")[1].split(" ")

    if firstSplit[1] == 'the' :

        return firstSplit[2]

    else :

        return firstSplit[1]





# Function to clean data

def clearifyDataFrame(dataframe):

    

    # Drop the tickets as they are id

    dataframe = dataframe.drop('Ticket', axis=1)



    # Use of a quantile-based discretization function to make equal sized buckets based on the price

    dataframe['Fare'] = dataframe['Fare'].fillna(dataframe['Fare'].median())

    dataframe['Fare'] = pan.qcut(dataframe['Fare'],5,labels=[0,1,2,3,4])



    # Fill the NaN Embarked (There are 2) row as if they embarked in Cherbourg

    dataframe['Embarked'] = dataframe['Embarked'].fillna('C')



    # Map the Embarked city

    dataframe['Embarked'] = dataframe['Embarked'].map( {'C': 0, 'Q': 1, 'S':2} ).astype(int)



    # Map the sex column

    dataframe['Sex'] = dataframe['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



    # Modify and rename the Cabin column to a easier to understand : "Does this person had a cabin ?"

    #dataframe['HasACabin'] = np.where(dataframe['Cabin'].isnull(), 0, 1)

    dataframe = dataframe.drop('Cabin', axis=1)



    # Fill NaN age and use quantile based discretization function as with 'Fare'

    dataframe['Age'] = dataframe['Age'].fillna(dataframe['Age'].median())

    dataframe['Age'] = pan.cut(dataframe['Age'],5,labels=[0,1,2,3,4]).astype(int)



    # Get the title only of a name

    dataframe['Name'] = dataframe['Name'].apply(lambda x : getTitle(x))

    # Concatenate the rare titles

    dataframe['Name'] = dataframe['Name'].replace(['Don.', 'Dr.', 'Rev.', 'Major.', 'Col.', 'Sir.', 'Lady.', 'Capt.', 'Jonkheer.', 'Countess.', 'Dona.'], 0)

    # Concatenate Mrs, Miss and their french alike

    dataframe['Name'] = dataframe['Name'].replace(['Mrs.', 'Miss.', 'Mlle.', 'Mme.', 'Ms.'], 1)

    # Concatenate the Mr and Master

    dataframe['Name'] = dataframe['Name'].replace(['Mr.', 'Master.'], 2)



    # We create a new column representing the total family as SibSp is the # of siblings/spouses and Parch is the # of parents/children

    dataframe['FamilySize'] = dataframe['SibSp'].astype(int) + dataframe['Parch'].astype(int) + 1

    dataframe = dataframe.drop('SibSp', axis=1)

    dataframe = dataframe.drop('Parch', axis=1)



    # We drop the PassengerId column as it doesn't provide any usefull information

    dataframe = dataframe.drop('PassengerId', axis=1)

    

    return dataframe

      

# Load the train data

dataframe = pan.DataFrame(pan.read_csv('../input/train.csv'))



# Look at the correlation between the data

plt.figure(figsize=(8,5))

plt.title('Pearson Correlation of Features on a heatmap', y=1.05, size=15)

sns.heatmap(dataframe.corr(),linewidths=0.1,vmax=1.0, square=True, linecolor='grey', annot=True)
# Show with a boxplot the age difference between the class

sns.boxplot(x="Pclass", y="Age", hue="Survived", data=dataframe)

sns.despine(offset=10, trim=True)



# Prepare the data for the model

dataframe = clearifyDataFrame(dataframe)
## Model choice and train



# We cut the given dataframe to ~90% train dataframe and ~10% to verify the accuracy model

train_dataframe = dataframe.iloc[:800]

test_dataframe =  dataframe.iloc[800:]



# Cut the train result columns

train_target_dataframe = train_dataframe['Survived']

train_main_dataframe = train_dataframe.drop('Survived', axis=1)



clf = svm.SVC()

clf.fit(train_main_dataframe, train_target_dataframe)



# Cut the test result columns

test_target_dataframe = test_dataframe['Survived']

test_main_dataframe = test_dataframe.drop('Survived', axis=1)



# make predictions

expected = test_target_dataframe

predicted = clf.predict(test_main_dataframe)



# Output dataframe to compare results

outputDF = pan.DataFrame({'expected':expected, 'predicted':predicted})



# Check between the expected and predicted result

i=0

for index, row in outputDF.iterrows():

    if row['expected'] == row['predicted']:

        i+=1      

percentage = (i/len(outputDF)) * 100

print("Percentage on the 10% train : " + repr(percentage) + " %\n")        

print(outputDF.head(10))

print("...\t    ...\t       ...")
## Test of the model on the test data



# Load the train data and work on it

dataframe_test = pan.DataFrame(pan.read_csv('../input/test.csv'))

tempoDFTest_PassengerId = dataframe_test['PassengerId']

dataframe_test = clearifyDataFrame(dataframe_test)



# Predict the ouput

test_predicted = clf.predict(dataframe_test)



# Output dataframe to compare results

outputDFTest = pan.DataFrame({'PassengerId':tempoDFTest_PassengerId, 'Survived':test_predicted})

#outputDFTest.to_csv('../input/predicted_SVM.csv', columns = ["PassengerId", "Survived"], index=False)

print(outputDFTest)