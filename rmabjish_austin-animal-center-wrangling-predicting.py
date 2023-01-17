import pandas as pd

train = pd.read_csv("../input/shelter-animal-outcomes/train.csv.gz")

test = pd.read_csv("../input/shelter-animal-outcomes/test.csv.gz")
train
# num of rows and columns

print( "records and columns in train dataset: ",train.shape)

print( "records and columns in test dataset:  ",test.shape)
# Four columns have nulls -

train.isnull().sum(axis = 0)
# All 9 columns are categorical - need to convert

train.info()
# values in target variable are not balanced, will make it challenging to predict "Died" or "Euthenized"

train.OutcomeType.value_counts()
# outcomeSubtype is dependant on OutcomeType, does not contribute to prediction

train = train.drop('OutcomeSubtype', axis=1)
# The actual name does not matter, what is important is whether the dog has a name or not

# has_name =1, no_name=0

train.Name = train.Name.apply(lambda x: 0 if pd.isnull(x) else 1)

test.Name = test.Name.apply(lambda x: 0 if pd.isnull(x) else 1)
# Now Name is numeric with no nulls

train.Name
# Out of 26729 records, 18 are nulls - we will replace them with zero

# Replace nulls with zeros, convert age from an ordinal variable to a numeric 

#age in weeks

#calculate Age in weeks

def age_in_weeks(x):

    if pd.isnull(x):

        return 0

    num = int(x.split(' ')[0])

    if 'year' in x:

        return num * 52

    elif 'month' in x:

        return num * 4.5

    elif 'week' in x:

        return num

    elif 'day' in x:

        return 1
train.AgeuponOutcome= train.AgeuponOutcome.apply(lambda x : age_in_weeks(x))

test.AgeuponOutcome= test.AgeuponOutcome.apply(lambda x : age_in_weeks(x))

#AgeuponOutcome is numeric with no nulls

train.AgeuponOutcome
# Breed Column - reduce unique values and convert to numeric. mixed breed=1 else 0



def breed_type(x):

    if "Mix" in x:

        return 1

    return 0
# Is animal of mix breed?

train.Breed = train.Breed.apply(breed_type)

test.Breed = test.Breed.apply(breed_type)

#Breed is now numeric with no nulls

train.Breed
# DateTime - Remove the timestamp then split date into three new fields.

train.DateTime=pd.to_datetime(train.DateTime)

train["dayofweek"] = train.DateTime.dt.dayofweek

train["month"] = train.DateTime.dt.month

train["year"] = train.DateTime.dt.year



test.DateTime=pd.to_datetime(test.DateTime)

test["dayofweek"] = test.DateTime.dt.dayofweek

test["month"] = test.DateTime.dt.month

test["year"] = test.DateTime.dt.year
train
#SexuponOutcome has one null value

train.isnull().sum(axis = 0)
# Drop record with null value

train = train.dropna()

train.isnull().sum(axis = 0)
# Use LabelEncoder to convert rest of features

from sklearn.preprocessing import LabelEncoder



train.OutcomeType = LabelEncoder().fit_transform(train.OutcomeType)

train.AnimalType = LabelEncoder().fit_transform(train.AnimalType)

train.SexuponOutcome = LabelEncoder().fit_transform(train.SexuponOutcome)

train.Color = LabelEncoder().fit_transform(train.Color)
test.AnimalType = LabelEncoder().fit_transform(test.AnimalType)

test.SexuponOutcome = LabelEncoder().fit_transform(test.SexuponOutcome)

test.Color = LabelEncoder().fit_transform(test.Color)
# ALl required columns has numeric values

train
# We will store the target variable in a dataset by itelf 

target = train.OutcomeType

target
# drop unused columns

train=train.drop(["AnimalID", "DateTime","OutcomeType"],axis=1)

train



testID=test.ID #keeping the IDs

test=test.drop(["ID","DateTime"],axis=1)
X=train

X
# we have 9 numeric columns with no nulls

train.isnull().sum(axis = 0)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, #train w/o target var

                                                    target, 

                                                    test_size=0.20, 

                                                    random_state=1)



print("Records & variables in X_train dataset: ", X_train.shape)

print("Records in training dataset for Target variable: ", y_train.shape)

print("Records & variables in X_test dataset: ", X_test.shape)

print("Records in testing dataset for Target variable: ", y_test.shape)
from sklearn.ensemble import RandomForestClassifier

# Do random forest

rf = RandomForestClassifier(n_estimators=1000)

rf.fit(X, target)

# Let's see the train accuracy

tra_score=rf.score(X, target)



print("Training accuracy for RandomForest: ",tra_score)
#Retraining with the complete training set

rf.fit(train, target)
#Getting predicted probabilities

pred = rf.predict_proba(test)
my_submission = pd.DataFrame({'ID':testID, 

                              'Adoption':pred[:,0], 

                              'Died':pred[:,1],'Euthanasia':pred[:,2],

                              'Return_to_owner':pred[:,3],'Transfer':pred[:,4] })



# you could use any filename

my_submission.to_csv('submission.csv', index=False)