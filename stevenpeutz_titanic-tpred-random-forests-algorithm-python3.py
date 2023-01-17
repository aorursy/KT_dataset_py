import os

os.getcwd()
# os.chdir('/Users/steven/Documents/Kaggle/Titanic')
os.getcwd()
import pandas as pd
%pylab inline  

# the %pylab statement here is just something to make sure visualizations we will make later on 

# will be printed within this window..
train_df = pd.read_csv('../input/train.csv', header=0)

#above is the basic command to 'import' your csv data. Df is the new name for your 'imported' data 

#(df is short for dataframe, you can name this anyway you want, but including 'df' in your name is convention)



test_df = pd.read_csv('../input/test.csv', header=0)

#you don't have to use the test set but I am doing this to eveluate the model without uploading. You can slip this.



#Other options for this (splitting dataset to train part and test part) involve importing 'train_test_split' from sklearn. 

#I have not used this option, but perhaps it is 'easier'..



train_df.head(2)

#with df.head(2) we can 'test' by previewing the first(head) 2 rows(2) of the dataframe(df)

#You can see the final x by using 'df.tail(x)  (replace x with number of rows)
train_df

#show full dataset (df). (if very large this can be very inconvenient but with our trainset it's ok)

#notice it adds a rows total and columns total underneath (troubleshooting: if you do not see these totals

# you can seperatly create this by using 'df.shape')
#let's get slightly more meta. (data about the data, like what type is eacht variable ('column')?)

train_df.info()

#especially the information on the right is usefull at this point (the clomuns with values 'int64 and 'object' etc)

# These values describing each variable should be identical to that of the testset (which in this case being

# the Titanic datasets from Kaggle) they are. To test this you could repeat this procedure but use the test set instead

# of the train set.
# More, More more!

#Let's fully dive in this meta data description of the variables:

train_df.describe()



# Notice that the decribe function only gives back non-'object' (7 out of the 12) variables..
train_df.Survived.value_counts()

#the variable name (Survived) is with a capital letter because it has a capital letter in data set.

#'value_counts' is the 'smart' part, the function.
train_df.Sex.value_counts().plot(kind='bar')

#you can replace the variable with any of the 12 (for some with more visual succes than for others..)
train_df[train_df.Sex=='female']

# double == because were making a comparison not setting up for creating)
#before continuing let's do a quick check for 'missing values' (rows where gender is unknown)

# by using the 'isnull' function:

train_df[train_df.Sex.isnull()]
# Let's visualize the number of survival amongst woman and later the number of survival amongst men to campare.

train_df[train_df.Sex=='female'].Survived.value_counts().plot(kind='bar', title='Survival among female')
train_df[train_df.Sex=='male'].Survived.value_counts().plot(kind='bar', title='Survival among male')
# The same can be done for age. Here it can also be interesting to combine age with sex;

train_df[(train_df.Age<11) & (train_df.Sex=='female')].Survived.value_counts().plot(kind='bar')

# '11' is just an arbitrarily chosen number as value of age.
train_df[(train_df.Age<11)].Survived.value_counts().plot(kind='bar')
import seaborn as sns

# I don't know why seaborn is abbreviated as sns but you can choose anything you like as long as it is not used

# by anything else. Sns seems to be convention.
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train_df);
#If we don't mind stereotypes ;p we could change the colors so that we don't have to look at the legend

# to remind us of the colorcoding for Sex:

#Just use the same command but add ' palette={"male": "blue", "female": "pink"} '
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train_df, palette={"male": "blue", "female": "pink"});
# Let's firs remove the variables we don't want:

def drop_features(df):

    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)
# make bins for ages and name them for ease:

def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df
#keep only the first letter (similar effect as making bins/clusters):

def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df
# make bins for fare prices and name them:

def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = categories

    return df
# createa all in transform_features function to be called later:

def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = drop_features(df)

    return df
# create new dataframe with different name:

train_df2 = transform_features(train_df)

test_df2 = transform_features(test_df)
train_df2.head()
sns.barplot(x="Age", y="Survived", hue="Sex", data=train_df2, palette={"male": "blue", "female": "pink"});
sns.barplot(x="Cabin", y="Survived", hue="Sex", data=train_df2, palette={"male": "blue", "female": "pink"});
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train_df2, palette={'male': 'blue', 'female': 'pink'});
from sklearn import preprocessing

def encode_features(df_train, df_test):

    features = ['Fare', 'Cabin', 'Age', 'Sex']

    df_combined = pd.concat([df_train[features], df_test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test

    

train_df2, test_df2 = encode_features(train_df2, test_df2)

train_df2.head()
train_df2.info()
X_train = train_df2.drop(["Survived", "PassengerId"], axis=1)

Y_train = train_df2["Survived"]

X_test  = test_df2.drop("PassengerId", axis=1).copy()









# I initially did not drop PassengerID. Keeping 8 variables ('features') in x-train and x-test. However, later on

# (during the modelling part) this resulted in an accuracy (for the random forests and classification trees) of 1.00

# Most likely I think it keeping PassengerId in this manner caused some form of label leakage. 

# After dropping this in both sets accuracy results were more realistic..



X_train.shape, Y_train.shape , X_test.shape
X_train.head()
Y_train.head()
# Logistic Regression



# Import from the the scikit-learn library (sklearn is the abbreviation for scikit-learn)

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
# Decision Tree



# Import from the the scikit-learn library (sklearn is the abbreviation for scikit-learn)

from sklearn.tree import DecisionTreeClassifier



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest



# Import from the the scikit-learn library (sklearn is the abbreviation for scikit-learn)

from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)



acc_random_forest
#Creating a csv with the predicted scores (Y as 0 and 1's for survival)

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })



# But let's print it first to see if we don't see anything weird:
submission.describe()
os.getcwd()
#submission.to_csv('../pathhere../submission.csv', index=False)