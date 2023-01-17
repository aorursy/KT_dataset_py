import pandas as pd              # pandas is great for managing datasets as a single object

import numpy as np               # numpy has great matrix/math operations on arrays



import matplotlib.pyplot as plt  # allows you to print figures and charts

import seaborn as sns            # a fancy chart add-on for matplotlib



# This command creates the figures and charts within the Jupyter notebook

%matplotlib inline     
'''

This calls all of the preprocessing functions.

Preprocessing should be done exactly the same on both the training and testing data sets.

'''



def preprocessData(df):

    

    # Convert the embarkation field from categorical ("S", "C", "Q")

    # to numeric (0,1,2)

    df = convertEmbarked(df)

    

    # Convert sex. Female = 0, Male = 1

    df = convertSex(df)



    df = addFamilyFactor(df)

    

    df = addTitles(df)



    # Remove irrelevant and non-numeric columns (features)

    df = df.drop(['Name', 'Cabin', 'PassengerId', 'Ticket'], axis=1) 



    # Replace the missing values (NaN) with the mean value for that field

    df = replaceWithMean(df)



    return df
'''

Convert the sex field to numeric

'''

def convertSex(df):

    

    # Convert the 'male'/'female' strings to integer classifiers.

    # Female = 0, Male = 1

    # Create a new column called "Gender" which is a mapping of the "Sex" column into integer values

    df["Gender"] = df["Sex"].map( {"female": 0, "male": 1} ).astype(int)

    # Now drop the "Sex" column since we've already replaced it by the integer column "Gender"

    df = df.drop(['Sex'], axis=1)

    

    return df

    
'''

Scikit-learn can only handle numbers.

So let's replace the text values for the 'Embarked' field with numbers. 

For example, the embarkation port labeled 'S' becomes 0, 'C' becomes 1, and 'Q' becomes 2.

'''

def convertEmbarked(df):

    

    if ('Embarked' in df.columns) :  # If the field 'Embarked' is in the pandas dataframe df

        

        # missing value, fill na with most often occured value

        if (len(df[df["Embarked"].isnull()]) > 0):



            # We need to get rid of missing values (not-a-number, NaN)

            # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.mode.html

            # If you want to impute missing values with the mode in a dataframe df, you can just do this:

            df.loc[df["Embarked"].isnull(), 'Embarked'] = df["Embarked"].dropna().mode().iloc[0]



        ports = list(enumerate(np.unique(df["Embarked"])))  # Get the list of unique port IDs

        port_dict = { name: i for i, name in ports } # Create a dictionary of the different port IDs

        df["Embarked"] = df["Embarked"].map( lambda x: port_dict[x]).astype(int)  # Reassign the port IDs to numbers

        

    return df
def addTitles(df):

    

    # we extract the title from each name

    combined = df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

    # a map of more aggregated titles

    Title_Dictionary = {

                        "Capt":       1,

                        "Col":        1,

                        "Major":      1,

                        "Jonkheer":   3,

                        "Don":        3,

                        "Sir" :       3,

                        "Dr":         2,

                        "Rev":        1,

                        "the Countess":3,

                        "Dona":       3,

                        "Mme":        0,

                        "Mlle":       0,

                        "Ms":         0,

                        "Mr" :        0,

                        "Mrs" :       0,

                        "Miss" :      0,

                        "Master" :    1,

                        "Lady" :      3



                        }

    

    # we map each title

    df['Title'] = combined.map(Title_Dictionary)

    

    return df
'''

Replace any missing values (NaN) with the mean value for that column

'''

def replaceWithMean(df):

    

    # Replace all NaNs in a Dataframe with the mean of that column (field).

    # I think this only works for numbers (int, float) 

    # You should do this as the last step of pre-processing in case you want to replace the NaNs

    # with some other method. Once you run this, all of the NaNs in the dataframe are gone.

    df = df.fillna(df.mean())

    return df
'''

There are two "family" variables in the original data. This combines them both into one variable.

'''

def addFamilyFactor(df):

    # Add a category called FamilyFactor

    # Perhaps people with larger families had a greater probablity of rescue?

    # If I just add the two together, then the new catgegory is just a linear transform and

    # won't really add new info. So I add and then square the value. 

    df['FamilyFactor'] = np.power(df.SibSp + df.Parch, 2)

    

    return df
'''

Read the training data from csv file

'''

train_df = pd.read_csv('../input/train.csv', header=0, dtype={"Age": np.float64})  # Load the train file into a dataframe



# Get the basic info for the data in this file

train_df.info()
# Setup a plot with 3 subplots side by side

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))



# Count plot of how many people embarked at each location

# countplot is for categorial data, barplot for quantitative data

sns.countplot(x="Embarked", data=train_df, ax=axis1)

axis1.set_title("# passengers per embarkation site")



# Comparing survivors versus fatalities as a function of embarkation

sns.countplot(x="Survived", hue="Embarked", data=train_df, order=[1,0], ax=axis2)

axis2.set_title("Survivors versus Fatalities")



# group by embarked, and get the mean for survived passengers for each value in Embarked

embark_perc = train_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()

sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

axis3.set_title("Survival versus embarkation")
train_df = preprocessData(train_df)
train_df.describe()
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.svm import SVC



# The data is now ready to go. So lets fit to the train, then predict to the test!

# Convert back to a numpy array

train_data = train_df.values

train_features = train_data[0::,1::]   # The features to use for the prediction model (e.g. age, family size)

train_result = train_data[0::,0]       # The thing the model predicts (i.e. survived)



print('Training with model. Please WAIT ...')



# Adaboost using a bunch of RandomForest models

# SAMME — Stagewise Additive Modeling using a Multi-class Exponential loss function 

# Zhu, H. Zou, S. Rosset, T. Hastie, “Multi-class AdaBoost”, 2009.

# For more info, http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_multiclass.html

model = AdaBoostClassifier(RandomForestClassifier(n_estimators = 1000),

                         algorithm="SAMME",

                         n_estimators=500)



# Here's how to do a Random Forest

#model = RandomForestClassifier(n_estimators=1000).fit(train_features, train_result)



# Here's a support-vector machine model (SVM)

#model = SVC(probability=True,random_state=1)



# Fit the training data to the Adaboost model

model = model.fit(train_features, train_result)



print ('Ok. Finished training the model.')
from sklearn.metrics import accuracy_score



print ('Accuracy = {:0.2f}%'.format(100.0 * accuracy_score(train_result, model.predict(train_features))))
from sklearn.model_selection import cross_val_score



# Calculating cross-validation of the training model

print ('Calculating cross-validation of the training model. Please WAIT ...')



# Cross-validation with k-fold of 5. So this will randomly split the training data into two sets.

# It then fits a model to one set and tests it against the other to get an accuracy.

# It will do this 5 times and return the average accuracy.

scores = cross_val_score(model, train_features, train_result, cv=5)

print ( 'On average, this model is correct {:0.2f}% (+/- {:0.2f}%) of the time.'.format(

        scores.mean() * 100.0, scores.std() * 2 * 100.0))
# Import the test data into a Pandas dataframe

test_df = pd.read_csv('../input/test.csv', header=0, dtype={"Age": np.float64})        # Load the test file into a dataframe



test_df.info()
# Grab the passenger Ids first since they are removed by the pre-processing function

testIds = test_df['PassengerId']



test_df = preprocessData(test_df)
print('Predicting the survival from the test data. PLEASE WAIT... ', end='') # Supress newline with end=''



test_predictions = model.predict(test_df.values)



print('DONE')
# Create a new dataframe with only the columns Kaggle wants from the dataset.

submission = pd.DataFrame({"PassengerId" : testIds.astype(int),

                           "Survived" : test_predictions.astype(int)})



# Save the submission to CSV file

submission.to_csv("titanic_submission", index=False)