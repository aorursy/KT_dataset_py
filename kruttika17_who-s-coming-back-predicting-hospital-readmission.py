# Importing libraries

import pandas as pd

import numpy as np

import os

import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

import re

from sklearn.model_selection import learning_curve

from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import ShuffleSplit

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from xgboost import XGBClassifier

from xgboost import to_graphviz

from sklearn.tree import export_graphviz

from sklearn.metrics import classification_report, confusion_matrix

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Reshape, MaxPooling2D, Flatten, Dropout

from numpy.random import seed

from tensorflow import set_random_seed

from keras.models import load_model

import matplotlib.pylab as py

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

# Reading in the training datasets

data1 = pd.read_csv("../input/Train.csv", na_values = ["NA", " ", "?"])

data2 = pd.read_csv("../input/Train_HospitalizationData.csv", na_values = ["NA", " ", "?"])

data3 = pd.read_csv("../input/Train_Diagnosis_TreatmentData.csv", na_values = ["NA", " ", "?"])

data4 = pd.read_csv("../input/IDs_mapping.csv", na_values = ["NA", " ", "?"], header = None)
# Checking how much training data we have

print("The number of rows and columns in the patient demographics data is", data1.shape)

print("The number of rows and columns in the patient hospitalization data is", data2.shape)

print("The number of rows and columns in the patient diagnosis and treatment data is", data3.shape)

print("The number of rows and columns in the ID mapping data is", data4.shape)
data1.head(3)
data2.head(3)
data3.head(3)
# Merging the first three datasets

# The sort argument will align data2's patient IDs with data1's and data3's

data = data1.merge(data2, on = "patientID", sort = True).merge(data3, on = "patientID", sort = True)
# Printing the number of rows and columns of the combined dataset

print("The numbers of rows and columns in the combined dataset is {}".format(data.shape))
# Separating the ID mapping info via row indexing

admission_type_id = data4[1:10]

discharge_disposition_id = data4[11:42]

admission_source_id = data4[43:68]
# Creating dictionaries for the ID mapping

admission_type_id_dict = dict(zip(admission_type_id[0], admission_type_id[1]))

discharge_disposition_id_dict = dict(zip(discharge_disposition_id[0], discharge_disposition_id[1]))

admission_source_id_dict = dict(zip(admission_source_id[0], admission_source_id[1]))
# Adding new columns to the main dataset that will contain mapped ID info

id_cols = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]

data[id_cols] = data[id_cols].astype(str)

# Mapping IDs to each patient using the dictionaries created above

data["admission_type"]= data["admission_type_id"].map(admission_type_id_dict)

data["discharge_disposition"]= data["discharge_disposition_id"].map(discharge_disposition_id_dict)

data["admission_source"]= data["admission_source_id"].map(admission_source_id_dict)
print(data["discharge_disposition"].value_counts())
print("The number of expired patients is {}".format(len(data[(data["discharge_disposition_id"] == "11") | (data["discharge_disposition_id"] == "19") | (data["discharge_disposition_id"] == "20")])))
# Dropping expired patients 

data = data[~((data["discharge_disposition_id"] == "11") | (data["discharge_disposition_id"] == "19") | (data["discharge_disposition_id"] == "20"))]
# Counting the missing values in each column

data.isnull().sum()
print("The percentage of missing values in the weight column is", int(33592/34650 * 100), "%")
print("The percentage of missing values in the payer_code column is", int(14719/34650 * 100), "%")

print("The percentage of missing values in the medical_specialty column is", int(16394/34650 * 100), "%")
# Getting the summary statistics

with pd.option_context("display.expand_frame_repr", True):

    with pd.option_context("display.max_columns", None):

        print(data.describe(include = "all"))

# Dropping AdmissionID, acetohexamide, weight and setting PatientID as the index

data.drop(["AdmissionID", "weight", "acetohexamide"], axis = 1, inplace = True)
# Filling NA values with "unknown" since they only occur in categorical columns

data.fillna("unknown", inplace = True)

data.set_index("patientID", inplace = True)
# Converting the date columns to datetime type

data["Admission_date"] = pd.to_datetime(data["Admission_date"], format = "%Y-%m-%d")

data["Discharge_date"] = pd.to_datetime(data["Discharge_date"], format = "%Y-%m-%d")



# Subtracting the admission_date and discharged_date columns to get a new column called days_hospitalized

data["days_hospitalized"] = ((data["Discharge_date"] - data["Admission_date"]).dt.days)

data["days_hospitalized"][:5]
# Writing a function to group ICD 9 codes based on online information



def grouper1(i):

    i = str(i) # to make the code subscriptable

    if (i[0].isnumeric() == True): # if the code begins with a number

        i = pd.to_numeric(i) # change the code from categorical to numeric dtype

        # assign a code group to each code

        if (1 >= i <= 139):

            return "001-139"

        elif (140 <= i <= 239):

            return "140-239"

        elif (240 <= i <= 279):

            return "240-279"

        elif (280 <= i <= 289):

            return "280-289"

        elif (290 <= i <= 319):

            return "290-319"

        elif (320 <= i <= 389):

            return "320-389"

        elif (390 <= i <= 459):

            return "390-459"

        elif (460 <= i <= 519):

            return "460-519"

        elif (520 <= i <= 579):

            return "520-579"

        elif (580 <= i <= 629):

            return "580-629"

        elif (630 <= i <= 679):

            return "630-679"

        elif (680 <= i <= 709):

            return "680-709"

        elif (710 <= i <= 739):

            return "710-739"

        elif (740 <= i <= 759):

            return "740-759"

        elif (760 <= i <= 779):

            return "760-779"

        elif (780 <= i <= 799):

            return "780-799"

        else:

            return "800-999"

    elif (i == "unknown"):  

        return "unknown" 

    else: # if the code does not begin with a number

        return "EV_code"

        
data["diag_1_grouped"] = data["diagnosis_1"].apply(grouper1)

data["diag_2_grouped"] = data["diagnosis_2"].apply(grouper1)

data["diag_3_grouped"] = data["diagnosis_3"].apply(grouper1)



# Dropping the original diagnosis columns

data.drop(["diagnosis_1", "diagnosis_2", "diagnosis_3"], axis = 1, inplace = True)
# Writing a function to group medical specialties

def grouper2(i):

    if ((i == "unknown") or (i == "InternalMedicine") or (i == "Family/GeneralPractice")

        or (i == "Emergency/Trauma") or (i == "Cardiology") or (i == "Surgery-General")):

        return i

    else:

        return "Other"
# Grouping medical specialty categories with under 1000 samples as Other

data["medical_specialty"] = data["medical_specialty"].apply(grouper2)

data["medical_specialty"].value_counts()
# Writing a function to group payer codes 

def grouper3(i):

    if ((i == "unknown") or (i == "MC") or (i == "HM")

        or (i == "BC") or (i == "SP") or (i == "MD") or (i == "CP") or (i == "UN")):

        return i

    else:

        return "Other"
# Grouping payer code categories with under 1000 samples as Other

data["payer_code"] = data["payer_code"].apply(grouper3)

data["payer_code"].value_counts()
# Creating barplots showing the distributions of some categorical features



temp = ["gender", "race", "age", "readmitted", "diabetesMed", "diag_3_grouped"]

for col in data[temp]:

    plt.figure(figsize=(10,7))

    ax = sns.countplot(y = col, data = data, order = data[col].value_counts().index, palette = "Set2");

    ax.set_alpha(0.8)

    ax.set_title("Bar plot : {}".format(col), fontsize=24)

    ax.set_xlabel("Number of patients", fontsize=14);

    ax.set_ylabel("");

    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)

    ax.set_xticks(range(0, 30000, 5000))

    ax.set_facecolor('xkcd:off white')

    ax.grid(alpha = 0.2)



# Add percentages to individual bars

    totals = []

    for i in ax.patches:

        totals.append(i.get_width())

    total = sum(totals)



    for i in ax.patches:

        ax.text(i.get_width()+.3, i.get_y()+.38, \

                str(round((i.get_width()/total)*100, 2))+'%', fontsize=12,

    color='black')



    plt.show()

    print()
# Printing histograms showing the distributions of numeric columns 

fig, ax = plt.subplots(nrows = 3, ncols = 2, figsize = (20, 22))

sns.set_style("darkgrid")

a = sns.distplot(a = data["num_lab_procedures"], kde = False, ax=ax[0, 0])

a.axes.set_title("Number of lab procedures", fontsize=16)

a.tick_params(labelsize=14)

a.set_ylabel("Count", fontsize = 12, rotation = 0)

b = sns.distplot(a = data["days_hospitalized"], kde = False, ax=ax[0, 1])

b.axes.set_title("Days hospitalized", fontsize=16)

b.tick_params(labelsize=14)

b.set_ylabel("Count", fontsize = 12, rotation = 0)

c = sns.distplot(a = data["num_medications"], kde = False, ax=ax[1, 0])

c.axes.set_title("Number of medications", fontsize=16)

c.tick_params(labelsize=14)

c.set_ylabel("Count", fontsize = 12, rotation = 0)

d = sns.distplot(a = data["num_diagnoses"], kde = False, ax=ax[1, 1])

d.axes.set_title("Number of diagnoses", fontsize=16)

d.tick_params(labelsize=14)

d.set_ylabel("Count", fontsize = 12, rotation = 0)

e = sns.distplot(a = data["num_procedures"], kde = False, ax=ax[2, 0])

e.axes.set_title("Number of procedures", fontsize=16)

e.tick_params(labelsize=14)

e.set_ylabel("Count", fontsize = 12, rotation = 0)

fig.delaxes(ax[2][1])

fig.show()
# Printing a correlation matrix for the numeric variables

data.corr()



# Original code for correlation plot (runs into bug while executing on Kaggle)

# corr = data.corr()

# cm = sns.light_palette("red", as_cmap=True) 

# corr.style.background_gradient(cmap = cm, axis = None)\

# .set_properties(**{"max-width": "100p", "font-size": "20pt"})\

# .set_precision(2)
sns.catplot(x = "age", y = "num_medications", data=data, jitter = "0.25")

plt.gcf().set_size_inches(12, 8)

plt.title("Age and Number of medications", fontsize = 22)

plt.show()
sns.catplot(x = "race", y = "num_medications", data=data, jitter = "0.25")

plt.gcf().set_size_inches(12, 8)

plt.title("Race and Number of medications", fontsize = 22)

plt.show()
sns.catplot(x = "gender", y = "num_medications", data=data, jitter = "0.25")

plt.gcf().set_size_inches(8, 5)

plt.title("Gender and Number of medications", fontsize = 22)

plt.show()
data[data["num_medications"] > 80]
sns.catplot(x = "A1Cresult", y = "num_medications", hue = "change", data=data, jitter = "0.25")

plt.gcf().set_size_inches(10, 5)

plt.title("A1Cresult and Number of medications", fontsize = 22)

plt.show()
sns.catplot(x = "discharge_disposition_id", y = "num_medications", hue = "readmitted", data=data) 

plt.gcf().set_size_inches(15, 8)

plt.title("Discharge disposition and readmitted status", fontsize = 22)

plt.show()
fig, ax = plt.subplots(nrows = 3, ncols = 2, figsize = (20, 30))

a = sns.boxplot(y = data["num_medications"], ax=ax[0, 0], palette = "Set2")

a.axes.set_title("Boxplot: number of medications", fontsize=16)

a.tick_params(labelsize=14)

a.set_ylabel("Count", fontsize = 12, rotation = 0)

b = sns.boxplot(y = data["num_procedures"], ax=ax[0, 1], palette = "Set2")

b.axes.set_title("Boxplot: number of procedures", fontsize=16)

b.tick_params(labelsize=14)

b.set_ylabel("Count", fontsize = 12, rotation = 0)

c = sns.boxplot(y = data["num_lab_procedures"], ax=ax[1, 0], palette = "Set2")

c.axes.set_title("Boxplot: number of lab procedures", fontsize=16)

c.tick_params(labelsize=14)

c.set_ylabel("Count", fontsize = 12, rotation = 0)

d = sns.boxplot(y = data["days_hospitalized"], ax=ax[1, 1], palette = "Set2")

d.axes.set_title("Boxplot: days in hospital", fontsize=16)

d.tick_params(labelsize=14)

d.set_ylabel("Count", fontsize = 12, rotation = 0)

e = sns.boxplot(y = data["num_diagnoses"], ax=ax[2, 0], palette = "Set2")

e.axes.set_title("Boxplot: number of diagnoses", fontsize=16)

e.tick_params(labelsize=14)

e.set_ylabel("Count", fontsize = 12, rotation = 0)

fig.delaxes(ax[2][1])

fig.show()
# Estimate the number of outliers

print("Approximate number of outliers in num_procedures is {}".format(len(data[data["num_procedures"] > 5])))

print("Approximate number of outliers in num_diagnoses is {}".format(len(data[data["num_diagnoses"] > 15])))

print("Approximate number of outliers in days_hospitalized is {}".format(len(data[data["days_hospitalized"] > 12])))

print("Approximate number of outliers in num_medications is {}".format(len(data[data["num_medications"] > 35])))

print("Approximate number of outliers in num_lab_procedures is {}".format(len(data[data["num_lab_procedures"] > 97])))
# Separating the predictors and target

predictors = data.loc[:, data.columns != "readmitted"]

target = data["readmitted"]
# Train/validation split in a 70:30 ratio

X_train, X_valid, y_train, y_valid = train_test_split(predictors, target, test_size=0.3, random_state=0)
# Preprocessing the training set



# Separating the numeric columns

num = ["num_medications", "num_procedures", "num_lab_procedures", "days_hospitalized", "num_diagnoses"]

num_cols = X_train[num]



# Separating categorical columns except ID mapping cols and target 

cat = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',

       'admission_source_id', 'payer_code', 'medical_specialty',

       'max_glu_serum', 'A1Cresult', 'metformin',

       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',

       'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',

       'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',

       'insulin', 'glyburide.metformin', 'glipizide.metformin',

       'metformin.rosiglitazone', 'metformin.pioglitazone', 'change',

       'diabetesMed', 'diag_1_grouped',

       'diag_2_grouped', 'diag_3_grouped']



# Dummifying (one-hot-encoding) the categorical columns

cat_cols = X_train[cat]

cat_dummies = pd.DataFrame(pd.get_dummies(cat_cols, drop_first = True))



# Merging the processed columns

X_train = pd.concat([cat_dummies, num_cols], axis = 1)



# Reformatting column names to avoid XGBoost model fitting errors 



regex = re.compile(r"\[|\]|<", re.IGNORECASE)

X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]



# Label encoding the target variable



label_encoder = LabelEncoder() 

y_train = label_encoder.fit_transform(y_train) 

# Defining a function to plot a learning curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    

    plt.figure(figsize = (10, 5))

    plt.title(title, fontsize = 14)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training samples", fontsize = 14)

    plt.ylabel("Recall", rotation = 0, fontsize = 14)

    plt.xticks(fontsize = 12)

    plt.yticks(fontsize = 12)

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring = 'recall')

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="b")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",

             label="Cross-validation score")



    plt.legend(loc="best")

    plt.grid()

    return plt
title = "Learning Curves (XGBoost) - imbalanced data"



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

estimator = XGBClassifier()

plot_learning_curve(estimator, title, X_train, y_train, ylim=(0, 1.01), cv=cv, n_jobs=1)



plt.show()
# Oversampling the positive class to balance the dataset



ros = RandomOverSampler(random_state=0)

X, y_train = ros.fit_resample(X_train, y_train)

X_train = pd.DataFrame(X, columns = X_train.columns)
title = "Learning Curves (Logistic Regression) - balanced data"



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

estimator = LogisticRegression()

plot_learning_curve(estimator, title, X_train, y_train, ylim=(0, 1.01), cv=cv, n_jobs=1)



plt.show()
title = "Learning Curves (Decision Tree) - balanced data"



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

estimator = DecisionTreeClassifier()

plot_learning_curve(estimator, title, X_train, y_train, ylim=(0, 1.01), cv=cv, n_jobs=1)



plt.show()
title = "Learning Curves (Random Forest) - balanced data"



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

estimator = RandomForestClassifier()

plot_learning_curve(estimator, title, X_train, y_train, ylim=(0, 1.01), cv=cv, n_jobs=1)



plt.show()
title = "Learning Curves (NaiveBayes) - balanced data"



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

estimator = GaussianNB()

plot_learning_curve(estimator, title, X_train, y_train, ylim=(0, 1.01), cv=cv, n_jobs=1)



plt.show()
title = "Learning Curves (Adaboost) - balanced data"



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

estimator = AdaBoostClassifier()

plot_learning_curve(estimator, title, X_train, y_train, ylim=(0, 1.01), cv=cv, n_jobs=1)



plt.show()
title = "Learning Curves (Gradient boosting) - balanced data"



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

estimator = GradientBoostingClassifier()

plot_learning_curve(estimator, title, X_train, y_train, ylim=(0, 1.01), cv=cv, n_jobs=1)



plt.show()
title = "Learning Curves (XGBoost) - balanced data"



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

estimator = XGBClassifier()

plot_learning_curve(estimator, title, X_train, y_train, ylim=(0, 1.01), cv=cv, n_jobs=1)



plt.show()
title = "Learning Curves (K Nearest Neighbor) - balanced data"



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

estimator = KNeighborsClassifier()

plot_learning_curve(estimator, title, X_train, y_train, ylim=(0, 1.01), cv=cv, n_jobs=1)



plt.show()
title = "Learning Curves (SVM) - balanced data"



cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

estimator = SVC()

plot_learning_curve(estimator, title, X_train, y_train, ylim=(0, 1.01), cv=cv, n_jobs=1)



plt.show()
# Preprocessing validation data 



num_cols = X_valid[num]

cat_cols = X_valid[cat]

cat_dummies = pd.DataFrame(pd.get_dummies(cat_cols, columns = cat_cols.columns, drop_first = True))

X_valid = pd.concat([cat_dummies, num_cols], axis = 1)



# Reformatting column names 



X_valid.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_valid.columns.values]



# Label encoding the target

y_valid = label_encoder.transform(y_valid) 
# Defining a function to fix column difference between train and test sets



# This will go inside the main function

def add_missing_dummy_columns(test_set, train_columns):

    # d = test set, columns = train set columns 

    missing_cols = set(train_columns) - set(test_set.columns)

    for c in missing_cols:

        test_set[c] = 0 # add missing columns to test set with empty column values

        

# This is the main function     

def fix_columns(test_set, train_columns):  



    add_missing_dummy_columns(test_set, train_columns)



    # make sure we have all the columns we need

    assert(set(train_columns) - set(test_set.columns) == set())



    extra_cols = set(test_set.columns) - set(train_columns) # these are the extra cols in the test set

    if extra_cols:

        print ("extra columns:", extra_cols)



    test_set = test_set[train_columns] # keep only columns that are in the train set 

    return test_set



# Fixing the test set

X_valid_fixed = fix_columns(X_valid, X_train.columns)
# Best XGBoost classifier

xgb = XGBClassifier(n_estimators = 50, max_depth = 5, gamma = 100, random_state = 0)

xgb.fit(X_train, y_train)

pred_train = xgb.predict(X_train)

print("Classification Report: XGBoost (training data)")

print(classification_report(y_train, pred_train))
# Testing the XGBoost on validation data

pred_valid = xgb.predict(X_valid_fixed)

print("Classification Report: XGBoost (validation data)")

print(classification_report(y_valid, pred_valid))
# Checking XGBoost's predictions on the validation set

print("Confusion Matrix: XGBoost predictions (validation data)")

pd.DataFrame(confusion_matrix(y_valid, pred_valid), columns = ["Predicted 0", "Predicted 1"])
# Best Gradient boosting classifier

gbm = GradientBoostingClassifier(n_estimators = 50, max_depth = 2, random_state = 0)

gbm.fit(X_train, y_train)

pred_train = gbm.predict(X_train)

print("Classification Report: Gradient Boosting (training data)")

print(classification_report(y_train, pred_train))
# Testing the GBM on validation data

pred_valid = gbm.predict(X_valid_fixed)

print("Classification Report: Gradient Boosting (validation data)")

print(classification_report(y_valid, pred_valid))
# Visualizing one tree in the XGBoost ensemble

print("Sample tree visualization from the XGBoost model")

to_graphviz(xgb, num_trees = 0) 
# Visualizing another tree in the XGBoost ensemble

print("Sample tree visualization from the XGBoost model")

to_graphviz(xgb, num_trees = 1) 
# Feature importances plot - XGBoost



plt.figure(figsize = (15, 15))

feat_importances = pd.Series(xgb.feature_importances_, index=X_train.columns)

feat_importances.nlargest(20).plot(kind='barh')

plt.title("Top 20 patterns for predicting readmission within 30 days - XGBoost (64% validation Recall, outliers untreated)", fontsize = 16)

plt.xticks(fontsize = 12)

plt.yticks(fontsize = 14)

plt.show()
# Feature importances plot - GBM



plt.figure(figsize = (15, 15))

feat_importances = pd.Series(gbm.feature_importances_, index=X_train.columns)

feat_importances.nlargest(20).plot(kind='barh')

plt.title("Top 20 patterns for predicting readmission within 30 days - Gradient boosting (60% validation Recall, outliers untreated)", fontsize = 16)

plt.xticks(fontsize = 12)

plt.yticks(fontsize = 14)

plt.show()
# Grouping data by important features and calculating readmission prevalence for each



twenty_two = data[data["discharge_disposition_id"] == "22"]

print("{}% of patients discharged to a rehab facility are readmitted within 30 days".format(np.around((twenty_two.groupby("readmitted")["readmitted"].agg("count")[1]) / 

                                                                                             (twenty_two.groupby("readmitted")["readmitted"].agg("count")[0] + twenty_two.groupby("readmitted")["readmitted"].agg("count")[1]) * 100), 2))

three = data[data["discharge_disposition_id"] == "3"]

print("{}% of patients discharged to a Skilled Nursing Facility are readmitted within 30 days".format(np.around((three.groupby("readmitted")["readmitted"].agg("count")[1]) / 

                                                                                             (three.groupby("readmitted")["readmitted"].agg("count")[0] + three.groupby("readmitted")["readmitted"].agg("count")[1]) * 100), 2))

diabetes = data[data["diabetesMed"] == "Yes"]

print("{}% of patients on diabetes medication are readmitted within 30 days".format(np.around((diabetes.groupby("readmitted")["readmitted"].agg("count")[1]) / 

                                                                                             (diabetes.groupby("readmitted")["readmitted"].agg("count")[0] + diabetes.groupby("readmitted")["readmitted"].agg("count")[1]) * 100), 2))

metformin = data[data["metformin"] == "No"]

print("{}% of patients not prescribed metformin are readmitted within 30 days".format(np.around((metformin.groupby("readmitted")["readmitted"].agg("count")[1]) / 

                                                                                             (metformin.groupby("readmitted")["readmitted"].agg("count")[0] + metformin.groupby("readmitted")["readmitted"].agg("count")[1]) * 100), 2))

age = data[data["age"] == "[50-60)"]

print("{}% of 50-60 year old patients are readmitted within 30 days".format(np.around((age.groupby("readmitted")["readmitted"].agg("count")[1]) / 

                                                                                             (age.groupby("readmitted")["readmitted"].agg("count")[0] + age.groupby("readmitted")["readmitted"].agg("count")[1]) * 100), 2))

two = data[data["discharge_disposition_id"] == "2"]

print("{}% of patients discharged to a short term hospital are readmitted within 30 days".format(np.around((two.groupby("readmitted")["readmitted"].agg("count")[1]) / 

                                                                                             (two.groupby("readmitted")["readmitted"].agg("count")[0] + two.groupby("readmitted")["readmitted"].agg("count")[1]) * 100), 2))

five = data[data["discharge_disposition_id"] == "5"]

print("{}% of patients discharged to another inpatient care institution are readmitted within 30 days".format(np.around((five.groupby("readmitted")["readmitted"].agg("count")[1]) / 

                                                                                             (five.groupby("readmitted")["readmitted"].agg("count")[0] + five.groupby("readmitted")["readmitted"].agg("count")[1]) * 100), 2))

six = data[data["discharge_disposition_id"] == "6"]

print("{}% of patients discharged to home with health service are readmitted within 30 days".format(np.around((six.groupby("readmitted")["readmitted"].agg("count")[1]) / 

                                                                                             (six.groupby("readmitted")["readmitted"].agg("count")[0] + six.groupby("readmitted")["readmitted"].agg("count")[1]) * 100), 2))

fifteen = data[data["discharge_disposition_id"] == "15"]

print("{}% of patients discharged to a swing bed are readmitted within 30 days".format(np.around((fifteen.groupby("readmitted")["readmitted"].agg("count")[1]) / 

                                                                                             (fifteen.groupby("readmitted")["readmitted"].agg("count")[0] + fifteen.groupby("readmitted")["readmitted"].agg("count")[1]) * 100), 2))

twenty_eight = data[data["discharge_disposition_id"] == "28"]

print("{}% of patients discharged to a psychiatric hospital are readmitted within 30 days".format(np.around((twenty_eight.groupby("readmitted")["readmitted"].agg("count")[1]) / 

                                                                                             (twenty_eight.groupby("readmitted")["readmitted"].agg("count")[0] + twenty_eight.groupby("readmitted")["readmitted"].agg("count")[1]) * 100), 2))

glu_serum = data[data["max_glu_serum"] == ">300"]

print("{}% of patients with glu serum > 300 are readmitted within 30 days".format(np.around((glu_serum.groupby("readmitted")["readmitted"].agg("count")[1]) / 

                                                                                             (glu_serum.groupby("readmitted")["readmitted"].agg("count")[0] + glu_serum.groupby("readmitted")["readmitted"].agg("count")[1]) * 100), 2))

pioglitazone = data[data["pioglitazone"] == "Up"]

print("{}% of patients whose pioglitazone level == Up are readmitted within 30 days".format(np.around((pioglitazone.groupby("readmitted")["readmitted"].agg("count")[1]) / 

                                                                                             (pioglitazone.groupby("readmitted")["readmitted"].agg("count")[0] + pioglitazone.groupby("readmitted")["readmitted"].agg("count")[1]) * 100), 2))

# Calculating the prevalence of each of the above categories in the patient population

print("{}% of all patients are discharged to an SNF".format(np.around((len(data[data["discharge_disposition_id"] == "3"]) / len(data)) * 100, 2)))

print("{}% of all patients are discharged to rehab".format(np.around((len(data[data["discharge_disposition_id"] == "22"]) / len(data)) * 100, 2))) 

print("{}% of all patients are discharged to a short term hospital".format(np.around((len(data[data["discharge_disposition_id"] == "2"]) / len(data)) * 100, 2)))

print("{}% of all patients are discharged to another inpatient care institution".format(np.around((len(data[data["discharge_disposition_id"] == "5"]) / len(data)) * 100, 2)))

print("{}% of all patients are discharged to home with health care service".format(np.around((len(data[data["discharge_disposition_id"] == "6"]) / len(data)) * 100, 2))) 

print("{}% of all patients are discharged to a swing bed".format(np.around((len(data[data["discharge_disposition_id"] == "15"]) / len(data)) * 100, 2))) 

print("{}% of all patients are discharged to a psychiatric hospital".format(np.around((len(data[data["discharge_disposition_id"] == "28"]) / len(data)) * 100, 2))) 

print("{}% of all patients have >300 max glu serum".format(np.around((len(data[data["max_glu_serum"] == ">300"]) / len(data)) * 100, 2))) 

print("{}% of all patients have pioglitazone == Up".format(np.around((len(data[data["pioglitazone"] == "Up"]) / len(data)) * 100, 2))) 

# Printing readmission trends for numeric features 

print(data.groupby(["num_diagnoses", "readmitted"])["readmitted"].agg("count"))

print()

print(data.groupby(["days_hospitalized", "readmitted"])["readmitted"].agg("count"))
# Combining features and calculating readmission prevalence for the combined groups

diabetes_snf = (data[(data["discharge_disposition_id"] == "3") & (data["diabetesMed"] == "Yes")])

print("{}% of patients on diabetes medication who are discharged to a Skilled Nursing Facility are readmitted within 30 days".format(np.around((diabetes_snf.groupby("readmitted")["readmitted"].agg("count")[1]) / 

                                                                                             (diabetes_snf.groupby("readmitted")["readmitted"].agg("count")[0] + diabetes_snf.groupby("readmitted")["readmitted"].agg("count")[1]) * 100), 2))



high_counts = (data[(data["days_hospitalized"] > 4) & (data["num_diagnoses"] > 7) & (data["num_medications"] > 12)])

print("{}% of patients with more than 7 diagnoses and more than 4 days spent in the hospital are readmitted within 30 days".format(np.around((high_counts.groupby("readmitted")["readmitted"].agg("count")[1]) / 

                                                                                             (high_counts.groupby("readmitted")["readmitted"].agg("count")[0] + high_counts.groupby("readmitted")["readmitted"].agg("count")[1]) * 100), 2))

diabetes_metformin = (data[(data["diabetesMed"] == "Yes") & (data["metformin"] == "No") & (data["payer_code"] == "unknown")])

print("{}% of patients on diabetes medication including metformin are readmitted within 30 days".format(np.around((diabetes_metformin.groupby("readmitted")["readmitted"].agg("count")[1]) / 

                                                                                             (diabetes_metformin.groupby("readmitted")["readmitted"].agg("count")[0] + diabetes_metformin.groupby("readmitted")["readmitted"].agg("count")[1]) * 100), 2))

rehab_hm = (data[(data["discharge_disposition_id"] == "22") & (data["payer_code"] == "HM")])

print("Zero rehab patients with payer code HM are readmitted within 30 days")

# Preprocessing data with scaling 

# Train validation split

X_train, X_valid, y_train, y_valid = train_test_split(predictors, target, test_size=0.3, random_state=0)



# Separating the numeric columns

num = ["num_medications", "num_procedures", "num_lab_procedures", "days_hospitalized", "num_diagnoses"]

num_cols = X_train[num]



# Scaling the numeric columns to have a mean of 0 and standard deviation of 1



scaler = StandardScaler()

num_scaled = pd.DataFrame(scaler.fit_transform(num_cols), columns = num)



# Separating categorical columns except ID mapping cols and target 

cat = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',

       'admission_source_id', 'payer_code', 'medical_specialty',

       'max_glu_serum', 'A1Cresult', 'metformin',

       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',

       'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',

       'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',

       'insulin', 'glyburide.metformin', 'glipizide.metformin',

       'metformin.rosiglitazone', 'metformin.pioglitazone', 'change',

       'diabetesMed', 'diag_1_grouped',

       'diag_2_grouped', 'diag_3_grouped']



# Dummifying the categorical columns

cat_cols = X_train[cat]

cat_dummies = pd.DataFrame(pd.get_dummies(cat_cols, drop_first = True))



# Aligning indexes so the dfs merge properly

num_scaled.set_index(cat_dummies.index, inplace = True)



# Merging the processed columns

X_train = pd.concat([cat_dummies, num_scaled], axis = 1)



# Reformatting column names to avoid some model fitting errors

import re

regex = re.compile(r"\[|\]|<", re.IGNORECASE)

X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]



# Label encoding the target variable

label_encoder = LabelEncoder() 

y_train = label_encoder.fit_transform(y_train) 



# Oversampling the positive class to balance the dataset

ros = RandomOverSampler(random_state=0)

X, y_train = ros.fit_resample(X_train, y_train)

X_train = pd.DataFrame(X, columns = X_train.columns)
# Preprocessing validation data 



num_cols = X_valid[num]

num_scaled = pd.DataFrame(scaler.transform(num_cols), columns = num)

cat_cols = X_valid[cat]

cat_dummies = pd.DataFrame(pd.get_dummies(cat_cols, cat_cols.columns, drop_first = True))

num_scaled.set_index(cat_dummies.index, inplace = True)

X_valid = pd.concat([cat_dummies, num_scaled], axis = 1)



# Reformatting column names 



X_valid.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_valid.columns.values]



# Label encoding the target

y_valid = label_encoder.transform(y_valid) 



# Fixing the validation set columns

X_valid_fixed = fix_columns(X_valid, X_train.columns)
# Building a KNN model and predicting on train data

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

pred_train = knn.predict(X_train)

print("Classification Report: KNN (training data)")

print(classification_report(y_train, pred_train))
# Applying KN model to validation data 

pred_valid = knn.predict(X_valid_fixed)

print("Classification Report: KNN (validation data)")

print(classification_report(y_valid, pred_valid))
# Building an SVM model and predicting on train data

svm = SVC(kernel = "rbf", C = 0.05)

svm.fit(X_train, y_train)

pred_train = svm.predict(X_train)

print("Classification Report: SVM (training data)")

print(classification_report(y_train, pred_train))
# Applying SVM model to validation data 

pred_valid = svm.predict(X_valid_fixed)

print("Classification Report: SVM (validation data)")

print(classification_report(y_valid, pred_valid))
# Neural network with one hidden layer



seed(3)

set_random_seed(3)



model = Sequential()

model.add(Dense(units=10, input_dim=173, activation='sigmoid', kernel_initializer='normal')) 

model.add(Dense(units=10, input_dim=173, activation='sigmoid', kernel_initializer='normal')) 

model.add(Dense(units=1, activation='sigmoid', kernel_initializer='normal'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ["accuracy"])

%time model.fit(X_train, y_train, epochs=1, batch_size=64)



pred_train = model.predict_classes(X_train)

print("Classification Report: MLP - one hidden layer (training data)")

print(classification_report(y_train, pred_train))
pred_valid = model.predict_classes(X_valid_fixed)

print("Classification Report: MLP - one hidden layer (validation data)")

print(classification_report(y_valid, pred_valid))
# MLP with two hidden layers



seed(4)

set_random_seed(1)



model2 = Sequential()

model2.add(Dense(units=10, input_dim=173, activation='sigmoid', kernel_initializer='normal')) 

model2.add(Dense(units=10, activation='sigmoid', kernel_initializer='normal')) 

model2.add(Dense(units=10, activation='sigmoid', kernel_initializer='normal')) 

model2.add(Dense(units=1, activation='sigmoid', kernel_initializer='normal'))

model2.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ["accuracy"])

%time model2.fit(X_train, y_train, epochs=1, batch_size=64)



pred_train = model2.predict_classes(X_train)

print("Classification Report: MLP - two hidden layers (training data)")

print(classification_report(y_train, pred_train))
pred_valid = model2.predict_classes(X_valid_fixed)

print("Classification Report: MLP - two hidden layers (validation data)")

print(classification_report(y_valid, pred_valid))
# MLP with three hidden layers



seed(4)

set_random_seed(1)



model3 = Sequential()

model3.add(Dense(units=10, input_dim=173, activation='sigmoid', kernel_initializer='normal')) 

model3.add(Dense(units=10, activation='sigmoid', kernel_initializer='normal')) 

model3.add(Dense(units=10, activation='sigmoid', kernel_initializer='normal')) 

model3.add(Dense(units=10, activation='sigmoid', kernel_initializer='normal')) 

model3.add(Dense(units=1, activation='sigmoid', kernel_initializer='normal'))

model3.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ["accuracy"])

%time model3.fit(X_train, y_train, epochs=1, batch_size=64)



pred_train = model3.predict_classes(X_train)

print("Classification Report: MLP - three hidden layers (training data)")

print(classification_report(y_train, pred_train))
pred_valid = model3.predict_classes(X_valid_fixed)

print("Classification Report: MLP - three hidden layers (validation data)")

print(classification_report(y_valid, pred_valid))
# Reading in the test data



data5 = pd.read_csv("../input/Test.csv", na_values = ["NA", " ", "?"])

data6 = pd.read_csv("../input/Test_HospitalizationData.csv", na_values = ["NA", " ", "?"])

data7 = pd.read_csv("../input/Test_Diagnosis_TreatmentData.csv", na_values = ["NA", " ", "?"])



print("The number of rows and columns in the patient demographics data is", data5.shape)

print("The number of rows and columns in the patient hospitalization data is", data6.shape)

print("The number of rows and columns in the patient diagnosis and treatment data is", data7.shape)
# Merging the first three datasets

# The sort argument will align data2's patient IDs with data1's

test_data = data5.merge(data6, on = "patientID", sort = True).merge(data7, on = "patientID", sort = True)



# Mapping the id cols

test_data[id_cols] = test_data[id_cols].astype(str)



test_data["admission_type"]= test_data["admission_type_id"].map(admission_type_id_dict)

test_data["discharge_disposition"]= test_data["discharge_disposition_id"].map(discharge_disposition_id_dict)

test_data["admission_source"]= test_data["admission_source_id"].map(admission_source_id_dict)



# Counting expired patients 

print("The number of expired patients in the test dataset is {}".format(len(test_data[(test_data["discharge_disposition_id"] == "11") | (test_data["discharge_disposition_id"] == "19") | (test_data["discharge_disposition_id"] == "20")])))
# Separating expired patients 

expired = test_data[(test_data["discharge_disposition_id"] == "11") | (test_data["discharge_disposition_id"] == "19") | (test_data["discharge_disposition_id"] == "20")]
# Dropping AdmissionID, acetohexamide, weight and setting PatientID as the index

test_data.drop(["AdmissionID", "weight", "acetohexamide"], axis = 1, inplace = True)



# Filling NA values with "unknown" since they only occur in categorical columns

test_data.fillna("unknown", inplace = True)

test_data.set_index("patientID", inplace = True)



# Feature engineering

# Converting the date columns to datetime type

test_data["Admission_date"] = pd.to_datetime(test_data["Admission_date"], format = "%Y-%m-%d")

test_data["Discharge_date"] = pd.to_datetime(test_data["Discharge_date"], format = "%Y-%m-%d")



# Subtracting the admission_date and discharged_date columns to get a new column called days_hospitalized

test_data["days_hospitalized"] = ((test_data["Discharge_date"] - test_data["Admission_date"]).dt.days)



# Grouping the columns with hundreds of levels



test_data["diag_1_grouped"] = test_data["diagnosis_1"].apply(grouper1)

test_data["diag_2_grouped"] = test_data["diagnosis_2"].apply(grouper1)

test_data["diag_3_grouped"] = test_data["diagnosis_3"].apply(grouper1)



# Dropping the original diagnosis columns

test_data.drop(["diagnosis_1", "diagnosis_2", "diagnosis_3"], axis = 1, inplace = True)



test_data["medical_specialty"] = test_data["medical_specialty"].apply(grouper2)

test_data["payer_code"] = test_data["payer_code"].apply(grouper3)
# Preprocessing the test data



# Scaling the numeric columns to have a mean of 0 and standard deviation of 1

num_cols = test_data[num]

num_scaled = pd.DataFrame(scaler.fit_transform(num_cols), columns = num)



# Separating categorical columns except ID mapping cols 

cat = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id',

       'admission_source_id', 'payer_code', 'medical_specialty',

       'max_glu_serum', 'A1Cresult', 'metformin',

       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',

       'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',

       'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',

       'insulin', 'glyburide.metformin', 'glipizide.metformin',

       'metformin.rosiglitazone', 'metformin.pioglitazone', 'change',

       'diabetesMed', 'diag_1_grouped',

       'diag_2_grouped', 'diag_3_grouped']



# Dummifying the categorical columns

cat_cols = test_data[cat]

cat_dummies = pd.DataFrame(pd.get_dummies(cat_cols, drop_first = True))



# Aligning indexes so the dfs merge properly

num_scaled.set_index(cat_dummies.index, inplace = True)



# Merging the processed columns

test = pd.concat([cat_dummies, num_scaled], axis = 1)



# Reformatting column names to avoid some model fitting errors

import re

regex = re.compile(r"\[|\]|<", re.IGNORECASE)

test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in test.columns.values]



# Fixing any column mismatch

test_fixed = fix_columns(test, X_train.columns)

# Neural network predictions



preds = model.predict_classes(test_fixed)

# Creating a df of preds and patientID

patientID = test_fixed.index

preds_with_id = pd.DataFrame(patientID)

preds_with_id["readmitted"] = preds

# Creating a list of expired patient ids

expired_patients = expired["patientID"]

# Setting the predictions for all expired patients to 0

preds_with_id.loc[preds_with_id["patientID"].isin(expired_patients), "readmitted"] = 0



# Writing the predictions to a csv file

preds_with_id.to_csv("preds.csv", index = False)