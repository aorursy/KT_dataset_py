# Kaggle's Python 3 environment comes with many helpful analytics libraries installed
# Below we import the modules we will need for our analysis, visualisation and submission of our results

import pandas as pd # data processing as a 'dataframe' (table) and .CSV file input/output
import matplotlib.pyplot as plt # module used for plotting graphs
# the below line ensures graphs display within this notebook
%matplotlib inline

from sklearn.model_selection import train_test_split # used for splitting the data ahead of applying machine learning models
from sklearn import tree # contains the `DecisionTreeClassifier()` machine learning model, as well as other useful tools
# We convert the provided data into Pandas dataframe objects (tables, essentially)

testandtrain_df = pd.read_csv('/kaggle/input/titanic/train.csv') # we know whether these passengers survived, we'll build our ML models on these data
validation_df = pd.read_csv('/kaggle/input/titanic/test.csv') # these are the passengers that we must estimate whether they survived or not

# We will not split the 'testandtrain_df' into the separate 'test' and 'train' sets until the data has been cleaned
print("Training dataset (rows, columns):", testandtrain_df.shape)
print("Validation dataset (rows, columns):", validation_df.shape)
testandtrain_df.sample(10) # takes a random sample of rows from the dataframe
validation_df.sample(10) # takes a random sample of rows from the dataframe
# first we look at the "train.csv" data, where 'Survived' is provided

testandtrain_df.info() # provides a basic summary of the data in the table

# we can see below we have data for all 891 passengers on most features, but not all (Age, Cabin and Embarked)
# and below we look at the "test.csv" data, where 'Survived' is missing (to be estimated by us)

validation_df.info() # provides a basic summary of the data in the table

# again we have data for all 418 passengers on most features, but not all (Age, Fare, and Cabin)
# the below ".describe()" method provides a statistical summary of the data in the table

testandtrain_df.describe(include ="all") # note we add 'include = "all"' to avoid ignoring null/non-numeric rows/cols
# the below ".describe()" method provides a statistical summary of the data in the table

validation_df.describe(include ="all") # note we add 'include = "all"' to avoid ignoring null/non-numeric rows/cols
validation_df[validation_df.Fare.isna()] # Filtering the "validation_df" by the rows with null/NaN values in the Fare column
validation_df.corr()["Fare"][:] # checks the Pearson correlation between 'Fare' and all other numeric columns

# Unsurprisingly, it shows a decent (negative) correlation with 'Pclass'.
# for completion, we check whether this relationship also holds for the "train.csv" data, which it does

testandtrain_df.corr()["Fare"][:]
# first we build a new concatenated dataframe from copies of the original two

temp1_df = pd.concat([testandtrain_df.copy(), validation_df.copy()], sort = False)


# then we delete all features except 'Fare' and 'Pclass' and delete rows with null data (should be just one)

temp1_df = temp1_df.filter(["Pclass", "Fare"])
temp1_df = temp1_df.dropna()

# to check this has worked correctly we look at the shape of the remaining dataframe
# it should have 1,308 (891 + 417) rows and 2 columns

print("Combined dataframe of all passengers except those with missing 'Fare' values\n", "Shape (rows, columns):", temp1_df.shape)
avg_fares = temp1_df.groupby('Pclass').mean()
avg_fares
validation_df.loc[152, "Fare"] = avg_fares.loc[3, 'Fare']
# to confirm success, we can look again at the passenger's data and see that Fare is now filled

validation_df[validation_df.index == 152]
testandtrain_df[testandtrain_df.Embarked.isna()] # Filtering the "testandtrain_df" by the rows with 'null' values in the Embarked column
# Below we observe the embarkation port of all the passengers (both datasets combined)
testandtrain_df.Embarked.value_counts() + validation_df.Embarked.value_counts()
testandtrain_df.loc[61, "Embarked"] = "S"
testandtrain_df.loc[829, "Embarked"] = "S"
# these columns would take significant work to be useful in our model

for df in [testandtrain_df, validation_df]:
    df.drop(columns = ['Cabin', 'Ticket'], inplace = True) # inplace = True ensures the original dataframes are updated
# this column is arbitrary and won't help training the ML model

testandtrain_df.drop(columns = ['PassengerId'], inplace = True) # inplace = True ensures the original dataframes are updated
testandtrain_df.info()
validation_df.info()
# first a reminder of the current dataframe format, looking at the 'Name' value in particular

testandtrain_df.sample(10)
# for each dataframe, we go row-by-row, extract the title, then add it to a new column 'Title'

for df in [testandtrain_df, validation_df]:
    for row in df.index:
        # take the text after ", " and before ". "
        df.loc[row, 'Title'] = df.loc[row, 'Name'].split(", ")[1].split(". ")[0]
# check both dataframes to see if this has had the desired effect...

testandtrain_df.sample(10)
# check both dataframes to see if this has had the desired effect...

validation_df.sample(10)
# again we create a temporary combined dataframe, then group-by the 'Title'
temp2_df = pd.concat([testandtrain_df.copy(), validation_df.copy()], sort = False)

print("Number of unique titles (in both dataframes):", temp2_df.Title.nunique()) # shows the number of unique values
print("Number of null titles (in both dataframes):", temp2_df.Title.isna().sum()) # shows the number of null values

# then we count the rows (by arbitrarily counting the 'Name' values)
temp2_df.groupby("Title").count().Name
prof_titles = ["Capt", "Col", "Dr", "Major", "Rev"] # military or professional titles
other_titles = ["Don", "Dona", "Jonkheer", "Lady", "Mlle", "Mme", "Ms", "Sir", "the Countess"] # nobility/foreign titles

for df in [testandtrain_df, validation_df]:
    for row in df.index:
        if df.loc[row, 'Title'] in prof_titles:
            df.loc[row, 'Title'] = "Professional"
        elif df.loc[row, 'Title'] in other_titles:
            df.loc[row, 'Title'] = "Other"
        # otherwise, leaves title unchanged (for Miss/Mrs/Master/Mr)
# Below we show the count and average survival rate (0 to 1) of passengers in each title group (out of 889 total)
# We utilise the aggregator function `.agg()` which applies the (name, function) to each row
testandtrain_df.groupby("Title").Survived.agg([("Count", "count"), ("Survival (mean)", "mean")], axis = "rows")
for df in [testandtrain_df, validation_df]:
    df.drop(columns = ['Name'], inplace = True)
# Again we create a temporary combined dataframe and group-by the 'Title'
# Then we calculate the mean age for each title and assign the result to "title_ages"

temp3_df = pd.concat([testandtrain_df.copy(), validation_df.copy()], sort = False)
title_ages = temp3_df.groupby("Title").Age.mean()
title_ages
# for both dataframes, we now fill any null 'Age' values using the above average "title_ages"

for df in [testandtrain_df, validation_df]:
    for row in df.index:
        if pd.isna(df.loc[row, 'Age']): # if 'Age' is null value ("NaN")
            df.loc[row, 'Age'] = title_ages[df.loc[row, 'Title']] # then set to average for that passenger's title, as above
# Below the first `sum()` counts columns with NaN values, the second sums the indiviual rows

print("Total null values in both dataframes:", testandtrain_df.isna().sum().sum() + validation_df.isna().sum().sum())
testandtrain_df.corr()["Survived"][:] # the Pearson correlation between survival and other numeric features

# for both SibSp and Parch the correlation is near 0
# for both dataframes, we create the 'Family_Onboard' feature
# as well as then dropping the 'SibSp' and 'Parch' features

for df in [testandtrain_df, validation_df]:
    df["Family_Onboard"] = df["SibSp"] + df["Parch"]
    df.drop(columns = ['SibSp', 'Parch'], inplace = True)
testandtrain_df.sample(10)
validation_df.sample(10)
testandtrain_df[testandtrain_df["Fare"] == 0]
validation_df[validation_df["Fare"] == 0]
print("Training dataset (rows, columns):", testandtrain_df.shape)
print("Validation dataset (rows, columns):", validation_df.shape)

# We still have 891 + 418 = 1,309 passengers in total.
temp4_df = pd.concat([testandtrain_df.copy(), validation_df.copy()], sort = False) # create a temporary combined dataframe
temp4_df = temp4_df[temp4_df.Fare != 0] # filter out zero fare rows
temp4_df.shape # we check we have 1,292 with non-zero fares (1309 minus the 17 zero fare passengers above)

# for those that notice we have an extra column, we should, this is just because the dataframes have 1 distinct column each
# so the combined dataframe takes all the unique columns and fills the rows with "NaN" where necessary
# below we use plt.hist to create a histogram with fare values colour-coded by passenger class
# we will not manually select 'bins' here, so they will be automatically chosen reflecting the data spread

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (20,10)) # we'll present charts in a 2x2 grid
ax0, ax1, ax2, ax3 = axes.flatten()

# First chart (all passenger classes together)
ax0.hist(temp4_df[temp4_df["Pclass"] == 1].Fare, color = "red", alpha = 0.4, label = "Pclass 1") # alpha <1 makes the bars partially transparent
ax0.hist(temp4_df[temp4_df["Pclass"] == 2].Fare, color = "green", alpha = 0.8, label = "Pclass 2")
ax0.hist(temp4_df[temp4_df["Pclass"] == 3].Fare, color = "blue", alpha = 0.6, label = "Pclass 3")

ax0.set_title("Histogram of fares (non-crew) - all passenger classes")
ax0.legend()
ax0.set_xlabel("Fare (US$)")
ax0.set_ylabel("Number of passengers")

# Second chart (passenger class 1)
ax1.hist(temp4_df[temp4_df["Pclass"] == 1].Fare, color = "red", bins = 40)
ax1.set_title("Histogram of fares (non-crew) - passenger class 1")
ax1.set_xlabel("Fare (US$)")
ax1.set_ylabel("Number of passengers")

# Third chart (passenger class 2)
ax2.hist(temp4_df[temp4_df["Pclass"] == 2].Fare, color = "green")
ax2.set_title("Histogram of fares (non-crew) - passenger class 2")
ax2.set_xlabel("Fare (US$)")
ax2.set_ylabel("Number of passengers")

# Fourth chart (passenger class 3)
ax3.hist(temp4_df[temp4_df["Pclass"] == 3].Fare, color = "blue")
ax3.set_title("Histogram of fares (non-crew) - passenger class 3")
ax3.set_xlabel("Fare (US$)")
ax3.set_ylabel("Number of passengers")

plt.show()
# Below we use a manual `for` loop to 'bin' the data so we can easily apply custom labelling, for clarity
# However note there do exist functions for automating this process

fareGroups = ["Crew", "0_10", "10_20", "20_50", "50_100", "over100"] # names for each 'bin'/group
fareRanges = [-1, 0, 10, 20, 50, 100, 1000] # limits for each 'bin' as decided above - e.g. "20_50" = (20, 50]

for df in [testandtrain_df, validation_df]:
    df["FareBinned"] = pd.cut(df.Fare, fareRanges, labels = fareGroups)
# Checking our new column has appeared as desired

testandtrain_df.sample(10)
# Checking our new column has appeared as desired
validation_df.sample(10)
temp4_df = pd.concat([testandtrain_df.copy(), validation_df.copy()], sort=False)
print("Number of passengers in each bin/grouping (out of 1,309 - both dataframes):-\n")
temp4_df.groupby("FareBinned").Fare.count() # we arbitrarily take Fare as both dataframes have values in those columns
# Next, we check whether there is a clear correlation between the fare groupings we've chosen and survival
# As we only have 'Survived' data for the training data, we do it just for that dataframe

testandtrain_df.groupby("FareBinned").Survived.agg([("Count", "count"), ("Survival (mean)", "mean")], axis = "rows")

# The data suggests quite clear correlations in the data - especially for the crew/low-fare passengers and the highest fare passengers
# As the above was successful, we can now drop the "Fare" column

for df in [testandtrain_df, validation_df]:
    df.drop(columns = ["Fare"], inplace = True)
plt.hist(temp4_df.Age) # we can re-use the temporary combined dataframe from above

plt.title("Histogram of passenger ages")
plt.xlabel("Age (years)")
plt.ylabel("Number of passengers")

plt.show()
# Next, we will look quickly at whether there is a clear correlation between age and survival
# As we only have Survived data for the training data, we do it just for that dataframe

plt.scatter(testandtrain_df.Age, testandtrain_df.Survived)

plt.title("Survival of passengers by Age (years)")
plt.xlabel("Age (years)")
plt.ylabel("Survival (1 = yes, 0 = no)")

plt.show()

# The chart below doesn't show a strong correlation
# Except perhaps for the very oldest passengers - significantly fewer survived
plt.hist(temp4_df.Family_Onboard) # we can re-use the temporary combined dataframe from above

plt.title("Histogram of number of family members onboard")
plt.xlabel("Total family members onboard")
plt.ylabel("Number of passengers")

plt.show()
# Next, we will check whether there is a clear correlation between the number of family members onboard and survival
# As we only have Survived data for the training data, we do it just for that dataframe

testandtrain_df.groupby("Family_Onboard").Survived.agg([("Count", "count"), ("Survival (mean)", "mean")], axis = "rows")
ageGroups = ["infant","child","teenager","young_adult","adult","senior"]
ageRanges = [0,5,12,19,30,58,100] # so an infant is <= 5, a child is > 5 and <=12, etc.
 
familyGroups = ["0","1","2","3","4+"] # the number of family members a passenger has onboard
familyRanges = [-1,0,1,2,3,20]

for df in [testandtrain_df, validation_df]:
    df["AgeBinned"] = pd.cut(df.Age, ageRanges, labels = ageGroups)
    df["FamilyBinned"] = pd.cut(df.Family_Onboard, familyRanges, labels = familyGroups)
# Checking our new column has appeared as desired
testandtrain_df.sample(10)
# Checking our new column has appeared as desired
validation_df.sample(10)
# As the above was successful, we can now drop the "Age" and "Family_Onboard" columns

for df in [testandtrain_df, validation_df]:
    df.drop(columns = ["Age", "Family_Onboard"], inplace = True)
testandtrain_df.sample(10)
validation_df.sample(10)
for df in [testandtrain_df, validation_df]:
    df['Sex'] = df['Sex'].map({"female": 1, "male": 0})
# the columns to be replaced with dummy variables for each unique possible value
dummy_cols = ["Pclass", "Embarked", "Title", "FareBinned", "AgeBinned", "FamilyBinned"]

# the prefixes for the respective dummy columns
dummy_prefixes = ["Pclass", "Port", "Title", "Fare", "Age", "Family"]

testandtrain_df = pd.get_dummies(testandtrain_df, columns = dummy_cols, prefix = dummy_prefixes)
validation_df = pd.get_dummies(validation_df, columns = dummy_cols, prefix = dummy_prefixes)
testandtrain_df.columns # the dataframes now have 31 columns, all binary values
testandtrain_df.sample(5)
validation_df.columns # the dataframes now have 31 columns, all binary values
# (except PassengerId in the validation dataframe)
validation_df.sample(5)
testandtrain_df.corr()["Survived"][:].sort_values(ascending = False)
print("Survival % split for train.csv data")
testandtrain_df.Survived.value_counts(normalize = True) # return the % of all values in this Series/column
X = testandtrain_df.iloc[:, 1:] # all rows, all columns except the first ('Survived')
y = testandtrain_df.iloc[:, 0] # all rows, only the first column ('Survived')

# we next (randomly) split the data into a 70%/30% split train/test
# random_state = (integer) fixes the random selection for later comparison (otherwise it will be a new random selection every time)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.3, random_state = 0)
# stratification on 'y' selected - as explained above
X_train # 70% of the passengers from "train.csv", all features/columns except 'Survived'
y_train # same passengers as above, a single feature/column showing whether they survived
X_test # remaining 30% of the passengers from "train.csv", all features/columns except 'Survived'
y_test # same passengers as above, a single feature/column showing whether they survived
# we create an instance of the model with random_state fixed (to any integer) so we can recreate the result if we make changes
dt_model = tree.DecisionTreeClassifier(random_state = 0)

# next we fit the model to our training data
dt_model.fit(X_train, y_train)
print("Train Accuracy:", dt_model.score(X_train, y_train))
print("Test Accuracy:", dt_model.score(X_test, y_test))
# changing the test:train split ratio to see impact to model accuracy

train_accuracy = []
test_accuracy = []
trial_range = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

for test_split in trial_range:
    # for each test_split we run a temporary Decision Tree
    temp_X_train, temp_X_test, temp_y_train, temp_y_test = train_test_split(X, y, stratify = y, test_size = test_split, random_state=1)
    temp_dt = tree.DecisionTreeClassifier(random_state = 0)
    temp_dt.fit(temp_X_train, temp_y_train)
    train_accuracy.append(temp_dt.score(temp_X_train, temp_y_train))
    test_accuracy.append(temp_dt.score(temp_X_test, temp_y_test))

testtrainsplit_df = pd.DataFrame({"test_split": trial_range, "train_acc": train_accuracy, "test_acc": test_accuracy})

plt.figure(figsize = (20,5))
plt.plot(testtrainsplit_df["test_split"], testtrainsplit_df["train_acc"], marker="o")
plt.plot(testtrainsplit_df["test_split"], testtrainsplit_df["test_acc"], marker="o")
plt.xlabel("Ratio of test data (vs train data)")
plt.xticks(trial_range)
plt.ylabel("Model accuracy score")
plt.title("Decision Tree model accuracy with different test:train ratios")
plt.legend()
train_accuracy = []
test_accuracy = []
trial_range = range(1,11)

for depth in trial_range:
    temp_dt = tree.DecisionTreeClassifier(random_state = 0, max_depth = depth)
    temp_dt.fit(X_train, y_train)
    train_accuracy.append(temp_dt.score(X_train, y_train))
    test_accuracy.append(temp_dt.score(X_test, y_test))
    
maxdepth_df = pd.DataFrame({"max_depth": trial_range, "train_acc": train_accuracy, "test_acc": test_accuracy})
maxdepth_df
plt.figure(figsize=(10,5))
plt.plot(maxdepth_df["max_depth"], maxdepth_df["train_acc"], marker = "o")
plt.plot(maxdepth_df["max_depth"], maxdepth_df["test_acc"], marker = "o")
plt.xlabel("Maximum depth of tree")
plt.xticks(range(0,max(trial_range) + 1))
plt.ylabel("Model accuracy score")
plt.title("Decision Tree model accuracy with varying 'max_depth'")
plt.legend()
# based on the above, we will fix max_depth for future iterations
max_depth_fixed = 4
train_accuracy = []
test_accuracy = []
trial_range = range(2,51)

for leaves in trial_range:
    # we use max_depth_fixed as decided above
    temp_dt = tree.DecisionTreeClassifier(random_state = 0, max_depth = max_depth_fixed, max_leaf_nodes = leaves)
    temp_dt.fit(X_train, y_train)
    train_accuracy.append(temp_dt.score(X_train, y_train))
    test_accuracy.append(temp_dt.score(X_test, y_test))
    
maxleaves_df = pd.DataFrame({"max_leaf_nodes": trial_range, "train_acc": train_accuracy, "test_acc": test_accuracy})

plt.figure(figsize = (20,5))
plt.plot(maxleaves_df["max_leaf_nodes"], maxleaves_df["train_acc"], marker = "o")
plt.plot(maxleaves_df["max_leaf_nodes"], maxleaves_df["test_acc"], marker = "o")
plt.xlabel("Maximum number of leaf nodes")
plt.xticks(range(0,max(trial_range) + 2, 2))
plt.ylabel("Model accuracy score")
plt.title("Decision Tree model accuracy with varying 'max_leaf_nodes'")
plt.legend()
# based on the above, we will not set max_leaf_nodes for future iterations, we will revert to default 'None'

train_accuracy = []
test_accuracy = []
trial_range = range(2,51)

for samples in trial_range:
    # we keep the max_depth_fixed set to the variable we decided above
    temp_dt = tree.DecisionTreeClassifier(random_state = 0, max_depth = max_depth_fixed, min_samples_leaf = samples)
    temp_dt.fit(X_train, y_train)
    train_accuracy.append(temp_dt.score(X_train, y_train))
    test_accuracy.append(temp_dt.score(X_test, y_test))
    
minsamples_df = pd.DataFrame({"min_samples_leaf": trial_range, "train_acc": train_accuracy, "test_acc": test_accuracy})

plt.figure(figsize = (20,5))
plt.plot(minsamples_df["min_samples_leaf"], minsamples_df["train_acc"], marker = "o")
plt.plot(minsamples_df["min_samples_leaf"], minsamples_df["test_acc"], marker = "o")
plt.xlabel("Minimum samples in terminal nodes")
plt.xticks(range(0, max(trial_range) + 2, 2))
plt.ylabel("Model accuracy score")
plt.title("Decision Tree model accuracy with varying 'min_samples_leaf'")
plt.legend()
dt_model = tree.DecisionTreeClassifier(random_state = 0, max_depth = max_depth_fixed, min_samples_leaf = 10)
dt_model.fit(X_train, y_train)

print("Train Accuracy:", dt_model.score(X_train, y_train))
print("Test Accuracy:", dt_model.score(X_test, y_test))
plt.figure(figsize = (25,10)) # the plot requires a decent amount of space, note if you have adjusted the max_depth beyond 3-4 it can get very hard to fit legibly!

# feature_names takes the columns from our dataframe
# class_names allows you to label the final classifications ('Survived' in this case)
# `filled = True` means the boxes will be coloured in line with it's classification
# (mostly survived = dark blue, mostly deceased = dark orange, 50:50 = white)

mytree = tree.plot_tree(dt_model,
                feature_names = testandtrain_df.columns[1:], 
                class_names = ["Deceased", "Survived"], 
                filled = True, fontsize = 12)
from sklearn.linear_model import LogisticRegression

regression = LogisticRegression(solver = 'lbfgs')
regression.fit(X_train, y_train)

print("Logistic Regression model")
print("="*25)
print("Train Accuracy:", regression.score(X_train, y_train))
print("Test Accuracy:", regression.score(X_test, y_test))
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(X_train, y_train)

print("Random Forest Classifier model")
print("="*30)
print("Train Accuracy:", random_forest.score(X_train, y_train))
print("Test Accuracy:", random_forest.score(X_test, y_test))
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)

print("XGB Classifier model")
print("="*20)
print("Train Accuracy:", xgb.score(X_train, y_train))
print("Test Accuracy:", xgb.score(X_test, y_test))
validation_df # the data is model-ready EXCEPT for the 'PassengerId' column - so we need to pass it into the model without that column
# so we create a copy of the dataframe without the first column, which we'll then feed into the model
X_validate = validation_df.drop("PassengerId", axis = 1)
# then we ask the model to predict the 'Survived' for each passenger based on those features (as learnt from the training data)
dt_model.fit(X, y)
prediction = dt_model.predict(X_validate)
prediction
# finally, we need to have a .csv file for submission to Kaggle
# so we create a dataframe adding the PassengerId data to the survival predicton - then convert to .csv
output = pd.DataFrame({'PassengerId': validation_df.PassengerId, 'Survived': prediction})
output.to_csv('my_submission.csv', index = False) # filename is used as required by the Kaggle competition