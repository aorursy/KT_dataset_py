from matplotlib import pyplot as plt

import pandas as pd

import numpy as np 
holdout = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
def remove_border(plt, legend=None, frameon=False):

    if plt is None:

        return

    

    # Remove plot border

    for sphine in plt.gca().spines.values():

        sphine.set_visible(False)

        

    # Create legend and remove legend box

    if legend is not None:

        plt.legend(legend, frameon=False)

    

    # Remove ticks 

    plt.tick_params(left=False, bottom=False) 
def process_age(df, cut_point=None, label_names=None):

    # Fill missing age with negative values to denote missing category

    df["Age"] = df["Age"].fillna(-0.5)

    

    # If cut point and label names are not provided, default ranges used.

    if cut_point is None and label_names is None:

        age_ranges = [("Missing", -1, 0), ("Infant", 0, 5), ("Child", 5, 12), 

                      ("Teenager", 12, 18), ("Young Adult", 18, 35), ("Adult", 35, 60),

                      ("Senior", 60, 100)]

    

        cut_points = [x for _, x, _ in age_ranges]

        cut_points.append(age_ranges[-1][-1])

        label_names = [labels for labels, *_ in age_ranges]

    

    df["Age_categories"] = pd.cut(df["Age"], cut_points, labels=label_names)

    return df
train = process_age(train)

holdout = process_age(holdout)

train_original_columns = train.columns
def create_dummies(df, column):

    return pd.concat([df, pd.get_dummies(df[column], prefix = column)], axis=1)
for col in ["Age_categories", 'Pclass', "Sex"]:

    train = create_dummies(train, col)

    holdout = create_dummies(holdout, col)
from sklearn.preprocessing import minmax_scale



# The holdout set has a missing value in the Fare column which

# we'll fill with the mean.

holdout["Fare"] = holdout["Fare"].fillna(train["Fare"].mean())



# Fill missing Embarked values with S, where Titanic first started 

for df in [train, holdout]:

    col = "Embarked"

    df[col] = df[col].fillna("S")



# Create dummy columns for Embarked    

train = create_dummies(train, col)

holdout = create_dummies(holdout, col)

    

#This estimator scales and translates each feature individually 

# such that it is in the given range on the training set, i.e. between zero and one.

for col in ["SibSp", "Parch", "Fare"]:

    train[col+"_scaled"] = minmax_scale(train[col].astype("float"))

    holdout[col+"_scaled"] = minmax_scale(holdout[col].astype("float"))
print(train.columns)
target = "Survived"

features = train.drop(columns=train_original_columns).columns

features
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver='liblinear')

lr.fit(train[features], train[target])

coefficients = lr.coef_

feature_importance = pd.Series(coefficients[0], index=train[features].columns)

print(feature_importance)

feature_importance.plot.barh()

remove_border(plt)

plt.show()
ordered_feature_importance = feature_importance.abs().sort_values(ascending=True)

ordered_feature_importance.plot.barh()

remove_border(plt)

plt.show()



# Since index 0 is smallest value, we want index 0 to be biggest value

ordered_feature_importance = feature_importance.abs().sort_values(ascending=False)
# Using top 8 features

features = ordered_feature_importance[:8].index.tolist()

features
from sklearn.model_selection import cross_val_score

from numpy import mean 



lr = LogisticRegression(solver='liblinear')

scores = cross_val_score(lr, train[features], train[target], cv=10)

accuracy = mean(scores)

print(scores)

print(accuracy)
lr = LogisticRegression()

lr.fit(train[features], train[target])

holdout_predictions = lr.predict(holdout[features])

holdout_predictions
def submission(holdout, holdout_predictions):

    submission = pd.DataFrame({"PassengerId":holdout["PassengerId"], "Survived":holdout_predictions})

    submission.to_csv("gender_submission.csv", index=False)

submission(holdout, holdout_predictions)
survived = train[train["Survived"] == 1]

died = train.drop(survived.index)



col_hist = "Fare"

# Survived and died histogram in a single plot

for df, color in [(survived, "red"), (died,"blue")]:

    ax = df[col_hist].plot.hist(alpha=.5, color=color, bins=50)

    ax.set_xlabel(col_hist)

    ax.set_xlim([0, 250])



remove_border(plt, legend=["Survived", "Died"])



plt.title("Survived and Died Histogram")

plt.show()



print(train[col_hist].describe())
def process_fare(df, cut_point=None, label_names=None):

    

    # If cut point and label names are not provided, default ranges used.

    if cut_point is None and label_names is None:

        fare_ranges = [("0-12", 0, 12), ("12-50", 12, 50), ("50-100", 50, 100), 

                      ("100+", 100, 1000)]

    

        cut_points = [x for _, x, _ in fare_ranges]

        cut_points.append(fare_ranges[-1][-1])

        label_names = [labels for labels, *_ in fare_ranges]

    

    df["Fare_categories"] = pd.cut(df["Fare"], cut_points, labels=label_names)

    return df
train = process_fare(train)

holdout = process_fare(holdout)

# Create dummy columns for newly Fare bins    

train = create_dummies(train, "Fare_categories")

holdout = create_dummies(holdout, "Fare_categories")
train.columns
# To construct list of all possible feature columns

train_original_columns = train_original_columns.tolist()

train_original_columns.append("Fare_categories")



features = train.drop(columns=train_original_columns).columns

features
def create_titles_n_cabins(df):

    titles = {

        "Mr" :         "Mr",

        "Mme":         "Mrs",

        "Ms":          "Mrs",

        "Mrs" :        "Mrs",

        "Master" :     "Master",

        "Mlle":        "Miss",

        "Miss" :       "Miss",

        "Capt":        "Officer",

        "Col":         "Officer",

        "Major":       "Officer",

        "Dr":          "Officer",

        "Rev":         "Officer",

        "Jonkheer":    "Royalty",

        "Don":         "Royalty",

        "Sir" :        "Royalty",

        "Countess":    "Royalty",

        "Dona":        "Royalty",

        "Lady" :       "Royalty"

    }



    extracted_titles = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)

    df["Title"] = extracted_titles.map(titles)



    # Form a new feature from cabin by using the first letter 

    df["Cabin_type"] = df["Cabin"].str[0]

    df["Cabin_type"] = df["Cabin_type"].fillna("Unknown")

    

    df = create_dummies(df, "Title")

    df = create_dummies(df, "Cabin_type")

    return df 
train = create_titles_n_cabins(train)

holdout = create_titles_n_cabins(holdout)

train_original_columns.append("Cabin_type")

train_original_columns.append("Title")
train["Title"].value_counts()
pd.pivot_table(train, values="Survived", index="Title")
features = train.drop(columns=train_original_columns).columns

features
import numpy as np

import seaborn as sns



def plot_correlation_heatmap(df):

    corr = df.corr()

    

    sns.set(style="white")

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True



    f, ax = plt.subplots(figsize=(11, 9))

    cmap = sns.diverging_palette(220, 10, as_cmap=True)





    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.show()
plot_correlation_heatmap(train[features])
to_drop = ["Pclass_2", "Age_categories_Teenager", "Fare_categories_12-50", "Title_Master", "Cabin_type_A"]

features = features.drop(to_drop)

features
from sklearn.feature_selection import RFECV

original_features = features

lr = LogisticRegression(solver='liblinear')

selector = RFECV(estimator=lr, cv=10)

selector.fit(train[features], train[target])

optimized_features = train[features].columns[selector.support_]

print("{} features were removed.".format(len(original_features)-len(optimized_features)))

print("Features removed are:", [col for col in original_features if col not in optimized_features])

print("Optimized {} features are: {}".format(len(optimized_features), optimized_features))
lr = LogisticRegression(solver='liblinear')

scores = cross_val_score(lr, train[optimized_features], train[target], cv=10)

accuracy = mean(scores)

accuracy
lr = LogisticRegression(solver='liblinear')

lr.fit(train[optimized_features], train[target])

holdout_predictions = lr.predict(holdout[optimized_features])

submission(holdout, holdout_predictions)