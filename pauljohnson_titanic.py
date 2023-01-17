# Data Transformation

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Feature Engineering

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectFromModel



# Modeling

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score





# Visualisation

import matplotlib.pyplot as plt

import seaborn as sns



# Misc

import os
train = pd.read_csv("../input/train.csv")

train.head()
test = pd.read_csv("../input/test.csv")

test.head()
# Combine datasets for efficient transformation

combined_set = [train, test]
# Checking Dataset

train.describe()
train.shape
# Check for missing data

train.isnull().sum()
# Functions for visualising dataset



# Continuous Data Plot

def cont_plot(df, feature_name, target_name, palettemap, hue_order, feature_scale): 

    df['Counts'] = "" # A trick to skip using an axis (either x or y) on splitting violinplot

    fig, [viz0,viz1] = plt.subplots(1,2,figsize=(10,5))

    sns.distplot(df[feature_name], ax=viz0);

    sns.violinplot(x=feature_name, y="Counts", hue=target_name, hue_order=hue_order, data=df,

                   palette=palettemap, split=True, orient='h', ax=viz1)

    viz1.set_xticks(feature_scale)

    plt.show()

    # WARNING: This will leave Counts column in dataset if you continues to use this dataset



# Categorical/Ordinal Data Plot

def cat_plot(df, feature_name, target_name, palettemap): 

    fig, [viz0,viz1] = plt.subplots(1,2,figsize=(10,5))

    df[feature_name].value_counts().plot.pie(autopct='%1.1f%%',ax=viz0)

    sns.countplot(x=feature_name, hue=target_name, data=df,

                  palette=palettemap,ax=viz1)

    plt.show()



    

survival_palette = {0: "red", 1: "green"} # Color map for visualization
cat_plot(train, 'Pclass','Survived', survival_palette)
cat_plot(train, 'Sex','Survived', survival_palette)
#Drop NA as they cant be visualised

train_dropna = train.dropna()

cont_plot(train_dropna, 'Age', 'Survived', survival_palette, [1, 0], range(0,100,5))
cat_plot(train, 'SibSp','Survived', survival_palette)
cat_plot(train, 'Parch','Survived', survival_palette)
cont_plot(train, 'Fare', 'Survived', survival_palette, [1, 0], range(0,100,10))
# Whats missing

for dataset in combined_set:

    print(dataset.isnull().sum())
# Extract Title from Name



def get_title(dataset, feature_name):

    return dataset[feature_name].map(lambda name:name.split(',')[1].split('.')[0].strip())



for dataset in combined_set:

    dataset["Title"] = get_title(train, "Name")

    #Show values

    print(dataset["Title"].value_counts())

# Use the Median Age for a Title to populate missing ages    



titles=["Miss", "Mr", "Master", "Mrs", "Dr", "Rev"]



for title in titles:

    for dataset in combined_set:

        # Find the value we want to populate

        median = dataset["Age"][dataset["Title"] == title].median()

        # Return a list of indicies we want to consider

        indicies = dataset["Title"] == title

        # Use the list of indicies with the column to fill and fill nulls with the value calculated

        dataset.loc[indicies,"Age"] = dataset["Age"].fillna(median)



for dataset in combined_set:

    print(dataset["Title"][dataset["Age"].notnull() == False].value_counts)
# There are a lot of titles with only one occurence or few occurences so it makes sense to combine them.

title_dict = {

                "Mr" :        "Mr",

                "Miss" :      "Miss",

                "Mrs" :       "Mrs",

                "Master" :    "Master",

                "Dr":         "Other",

                "Rev":        "Other",

                "Col":        "Other",

                "Major":      "Other",

                "Mlle":       "Miss",

                "Don":        "Other",

                "the Countess":"Mrs",

                "Ms":         "Mrs",

                "Mme":        "Mrs",

                "Capt":       "Other",

                "Lady" :      "Mrs",

                "Sir" :       "Mr",

                "Jonkheer":   "Other"

            }

for dataset in combined_set:

    dataset["TitleGroup"] = dataset.Title.map(title_dict)

    dataset["TitleGroup"].value_counts()

    

cat_plot(train, "TitleGroup","Survived", survival_palette)
# Create new feature for people who are alone

for dataset in combined_set:

    dataset["Has_Cabin"] = np.where(dataset["Cabin"].isna(), 0, 1)

    

cat_plot(train, "Has_Cabin","Survived", survival_palette)
# The only missing Emabrked values are for those with first class tickets, the most common embarkemnt for first class was Southamption "S"

for dataset in combined_set:

    print(dataset["Embarked"][dataset["Pclass"] == 1].value_counts())

    dataset["Embarked"] = dataset["Embarked"].fillna("S")

    

cat_plot(train, "Embarked","Survived", survival_palette)
# Combine Parch & SibSp

for dataset in combined_set:

    dataset["FamilySize"] = dataset["Parch"] + dataset["SibSp"]



cat_plot(train, "FamilySize","Survived", survival_palette)
# Create new feature for people who are alone

for dataset in combined_set:

    dataset["Is_Alone"] = np.where(dataset["FamilySize"]== 0, 1, 0)

    

cat_plot(train, "Is_Alone","Survived", survival_palette)
#Pretty normal distribution for Age

sns.distplot(train["Age"])
#Define Quanitiles

quantile_list = [0, .25, .5, .75, 1.]

quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']
### This transformation was cut from final as splitting using domain knowledge performed better

# Create quantile features

#for dataset in combined_set:

#    dataset["Age_quantile_range"] = pd.qcut(dataset["Age"],q=quantile_list)

#    dataset["Age_quantile_label"] = pd.qcut(dataset["Age"],q=quantile_list, labels=quantile_labels)



#train.head()
# We know women and children were offered lifeboats, so binning based on this domain knowledge

age_bin_ranges = [0, 18, 200]

age_bin_labels = ['Child', 'Adult']



# Create binned features

for dataset in combined_set:

    dataset["Age Bins"] = pd.cut(dataset["Age"],bins=age_bin_ranges)

    dataset["Age Bin Label"] = pd.cut(dataset["Age"],bins=age_bin_ranges, labels=age_bin_labels)



train.head()
# Fare has a heavy right skew so will try a log transformation to make a Gaussian distribution

sns.distplot(train["Fare"])
for dataset in combined_set:

    dataset["Fare_log"] = np.log((1+dataset["Fare"]))

#Distribution is more normal like after transformation so we will keep the new values

sns.distplot(train["Fare_log"])
#Create Quantile features

for dataset in combined_set:

    dataset["Fare_log_quantile_range"] = pd.qcut(dataset["Fare_log"],q=quantile_list)

    dataset["Fare_log_quantile_label"] = pd.qcut(dataset["Fare_log"],q=quantile_list, labels=quantile_labels)

    

train.head()
X_train = train.copy()

X_test  = test.copy()

# Drop Columns we wont be using

X_train = X_train.drop(columns=["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Title", "TitleGroup", "Age Bins", "Age", "Fare", "Cabin", "Fare_log", "Fare_log_quantile_range", "Survived", "Counts",])

X_test = X_test.drop(columns=[    "PassengerId", "Name", "SibSp", "Parch", "Ticket",  "Title", "TitleGroup", "Age Bins", "Age", "Fare", "Cabin", "Fare_log", "Fare_log_quantile_range"])

# Add survived as our label

y_train = train["Survived"]



# Keep PassengerIds for the test set

X_predict = test["PassengerId"]



combined_X = [X_train, X_test]

for dataset in combined_X:

    # Binary encoding

    dataset["Sex"] = dataset["Sex"].map({'male': 0, 'female': 1}).astype(int)



# Dummy encoding (can't do this inside for loop)

X_train = pd.get_dummies(X_train, prefix_sep="_", drop_first=True)

X_test = pd.get_dummies(X_test, prefix_sep="_", drop_first=True)



#Update the combined set

combined_X = [X_train, X_test]



for index, dataset in enumerate(combined_X):

    print("Shape of dataset {} : {}".format(index, dataset.shape))

    

X_test.head()
#Scale Values

mm_scaler = MinMaxScaler()

X_train = pd.DataFrame(data=mm_scaler.fit_transform(X_train),columns = list(X_train))

X_test = pd.DataFrame(data=mm_scaler.fit_transform(X_test),columns = list(X_test))
# Feature Selection



# Build a forest and compute the feature importances

xt = ExtraTreesClassifier(n_estimators=250, random_state=0)



xt.fit(X_train, y_train)



#Map feature names to feature importance

features = zip(X_train.columns, xt.feature_importances_)

#Convert zip to sorted list

features_sorted = sorted(features, key = lambda x:x[1])



# Plot freature importance

plt.bar(range(len(features_sorted)), [val[1] for val in features_sorted])

plt.xticks(range(len(features_sorted)), [val[0] for val in features_sorted])

plt.xticks(rotation=90)

plt.show()
# Set a minimum threshold and select features

sfm = SelectFromModel(xt, threshold="0.5*mean")



sfm.fit(X_train, y_train)

X_train_sfm = pd.DataFrame(sfm.transform(X_train))

X_test_sfm = pd.DataFrame(sfm.transform(X_test))



n_features = sfm.transform(X_train).shape[1]

print(n_features)
# Hyperparamater grid

sgd_params = [

  {'loss': ['hinge'], 

   'penalty': ['none', 'l2','l1'],

   'max_iter':[100, 250, 500, 750, 1000],

   'tol':[0.0001,0.001, 0.01]}

]



# Train Model

sgd = SGDClassifier()

sgd_gs = GridSearchCV(sgd, sgd_params, cv=10, scoring="accuracy")

#sgd_gs.fit(X_train, y_train)

#print("Best Hyperparameters: {}".format(sgd_gs.best_params_))

#sgd_mean_score = sgd_gs.cv_results_['mean_test_score'].mean()

#print("Accuracy: {}".format(sgd_mean_score))
# Hyperparamater grid

knn_params = [

  {'n_neighbors': [2,3,5,7,9], 

   'algorithm': ['auto',],

   'leaf_size':[1,3,5,7,9]}

]



# Train Model

knn = KNeighborsClassifier()

knn_gs = GridSearchCV(knn, knn_params, cv=10, scoring="accuracy")

#knn_gs.fit(X_train, y_train)

#print("Best Hyperparameters: {}".format(knn_gs.best_params_))

#knn_mean_score = knn_gs.cv_results_['mean_test_score'].mean()

#print("Accuracy: {}".format(knn_mean_score))
# Hyperparamater grid

lsvc_params = [

  {'penalty': ['l2'], 

   'loss': ['hinge','squared_hinge'],

   'C':[1,2,3,4,5],

   'max_iter':[10,100,1000],

   'tol':[0.0001,0.001, 0.01]}

]



# Train Model

lsvc = LinearSVC()

lsvc_gs = GridSearchCV(lsvc, lsvc_params, cv=10, scoring="accuracy")

#lsvc_gs.fit(X_train, y_train)

#print("Best Hyperparameters: {}".format(lsvc_gs.best_params_))

#lsvc_mean_score = lsvc_gs.cv_results_['mean_test_score'].mean()

#print("Accuracy: {}".format(lsvc_mean_score))
# Hyperparamater grid

dt_params = [

  {'splitter': ['best', 'random'], 

   'min_samples_split': [2,4,6,8],

   'min_samples_leaf':[9,10,11,12],

   'max_features':[None, 'auto', 'sqrt', 'log2']}

]



dt = DecisionTreeClassifier()

dt_gs = GridSearchCV(dt, dt_params, cv=10, scoring="accuracy")

#dt_gs.fit(X_train, y_train)

#print("Best Hyperparameters: {}".format(dt_gs.best_params_))

#dt_mean_score = dt_gs.cv_results_['mean_test_score'].mean()

#print("Accuracy: {}".format(dt_mean_score))

# Hyperparamater grid

rf_params = [

  {'n_estimators': [10,100,1000,10000], 

   'min_samples_split': [2,4,6,8],

   'min_samples_leaf':[9,10,11,12],

   'max_features':[None, 'auto', 'sqrt', 'log2'],

   'bootstrap':[True, False]}

]



rf = RandomForestClassifier()

rf_gs = GridSearchCV(dt, dt_params, cv=10, scoring="accuracy")

#rf_gs.fit(X_train, y_train)

#print("Best Hyperparameters: {}".format(rf_gs.best_params_))

#rf_mean_score = dt_gs.cv_results_['mean_test_score'].mean()

#print("Accuracy: {}".format(rf_mean_score))
# Use max voting technique to select best estimator for prediction

base_learners = [

    ('sgd', sgd_gs), ('knn', knn_gs), ('lsvc', lsvc_gs), ('dt', dt_gs), ('rf', rf_gs)

]



model = VotingClassifier(estimators=base_learners, voting='hard')

model.fit(X_train_sfm,y_train)

print("Accuracy: {}".format(model.score(X_train_sfm,y_train)))

y_predict = pd.DataFrame(model.predict(X_test_sfm), columns=["Survived"])

# Concatonate X & y

prediction = pd.concat([X_predict, y_predict], axis=1)

print("Submission shape: {}".format(prediction.shape))



#Create submission file

prediction.to_csv("submission.csv", index=False)