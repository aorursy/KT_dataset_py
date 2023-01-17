import pandas
import math
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

# define a function for cleaning up data, we will call this on both the training and test data
def clean_data(data_frame):

    # There's a bunch of fields we don't need so go ahead and delete those
    # We're removing cabin family related fields because we discovered
    #   that they are not important based on results from our previous forest
    data_frame = data_frame.drop(["Embarked", "Cabin", "Ticket", "Name", "PassengerId", "SibSp", "Parch"],
                                     axis=1)

    # We're going to change sex into a numeric value
    data_frame["Gender"] = data_frame["Sex"].map({"female": 0, "male": 1}).astype(int)
    del data_frame["Sex"]

    # we need to do something about missing age, fare and gender values as these will the ones we use to predict with.
    data_frame = fill_in_missing_values(data_frame)

    return data_frame

# Before we can fill in missing values we need to generate educated guesses
def calibrate_guesses():
    # Go over initial_data and identify median for age, gender and fare based on class
    guesses = {"Age": [[0., 0., 0.], [0., 0., 0.]],
               "Fare": [[0., 0., 0.], [0., 0., 0.]],
               "Gender": [0., 0., 0.]}

    trimmed_data = initial_data.copy()
    trimmed_data = trimmed_data.dropna()
    trimmed_data["Gender"] = trimmed_data["Sex"].map({"female":0, "male":1}).astype(int)

    for i in range(0,2):
        for j in range(0,3):
            guesses["Age"][i][j] = trimmed_data[(trimmed_data['Gender'] == i) & \
                              (trimmed_data['Pclass'] == j+1)]['Age'].median()
            guesses["Fare"][i][j] = trimmed_data[(trimmed_data['Gender'] == i) & \
                                                (trimmed_data['Pclass'] == j + 1)]['Fare'].median()
            guesses["Gender"][j] = math.floor(trimmed_data[(trimmed_data['Pclass'] == j + 1)]['Gender'].median())

    return guesses

# Finally a function that will iterate on different genders and class and fill in missing or invalid values
# I'm quite new to Python and Panda so my code may look a bit clunky, I've seen the same function written in 
#   three lines by people who are more experienced with Panda functions.
def fill_in_missing_values(data_frame):

    # fill in missing genders
    for j in range(0,3):
        data_frame.loc[(data_frame.Gender.isnull() & data_frame.Pclass == (j+1)), 'Gender'] = default_values["Gender"][j]

    # fill in missing age and fare, consider fares below 1 as missing too.
    for i in range(0, 2):
        for j in range(0, 3):
            data_frame.loc[(data_frame.Age.isnull()) & (data_frame.Gender == i) & (data_frame.Pclass == j + 1), \
                   'Age'] = default_values["Age"][i][j]
            data_frame.loc[(data_frame.Fare.isnull()) & (data_frame.Gender == i) & (data_frame.Pclass == j + 1), \
                           'Fare'] = default_values["Fare"][i][j]
            data_frame.loc[(data_frame.Fare < 1) & (data_frame.Gender == i) & (data_frame.Pclass == j + 1), \
                           'Fare'] = default_values["Fare"][i][j]

    return data_frame

# Load csv data
print("Loading data")
initial_data = pandas.read_csv('../input/train.csv', header=0)
final_test_data = pandas.read_csv('../input/test.csv', header=0)

default_values = calibrate_guesses()

cleaned_data = clean_data(initial_data)
cleaned_final_test = clean_data(final_test_data)

print("Data cleaned")
print(cleaned_data.info())
print(cleaned_final_test.info())
# we will split our cleaned data frame into two test and train groups, just so we can evaluate our ML performance
# we will use the train_test_split function from SKLearn

train_data_df, eval_test_data_df = train_test_split(cleaned_data, test_size=0.25)

# we have to turn our data frames into numpy arrays since that's what ML libs work with;

train_data = train_data_df.values
eval_test_data = eval_test_data_df.values

final_test = cleaned_final_test.values

# we try the random forest
print("------")
print("Forest score:")

# create the random forest object which will include all the parameters for the fit
forest = RandomForestClassifier(n_estimators=100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])

# now we will evaluate our forest by comparing the output with actual survival values in our test_data
print(forest.score(eval_test_data[0::, 1::], eval_test_data[0::, 0]))

# run prediction on test data

output = forest.predict(final_test)

# create output file by creating data frame and writing it to csv file.
output_table = {"PassengerId": final_test_data["PassengerId"],
                "Survived": output.astype(int)}
output_data_frame = pandas.DataFrame(output_table)

output_data_frame.to_csv(path_or_buf="output.csv", index=False)