# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# imports and setting up of options

import numpy as np # array manipulation

import pandas as pd # csv import, data manipulation

import matplotlib.pyplot as plt # graph rendering

import seaborn as sns # statistical data visualization

import csv



# model feature engineering imports

import sklearn.model_selection as model_selection# split training data for training and validation

from sklearn.impute import SimpleImputer # used for handling missing data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data

from sklearn.preprocessing import StandardScaler,RobustScaler # used for numeric feature scaling 

from sklearn.metrics import accuracy_score # check accuracy on validation data

from sklearn.compose import make_column_transformer # use different encoder of different columns



# import models to fit

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier # random forest classifier

from sklearn.svm import SVC



# display imports

from IPython.display import display # to display pandas DataFrames

import warnings

warnings.filterwarnings('ignore')



# setting library options

# Seaborn options

sns.set_style("whitegrid")

# Some matplotlib options

%matplotlib inline

# General pandas options

pd.set_option("display.width", 1000)

pd.set_option('display.max_columns', None) #show all columns of the dataset

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
le = LabelEncoder()

ohe = OneHotEncoder(sparse=False,drop="first")

sc = StandardScaler()

rsc = RobustScaler()
# define functions



def gen_correlation_map(X):

    """

    Finding the correlation plot for numerical features to detect

    features that are necessary and discard the rest.

    mask : removes the upper triangle as it is redundant

    """

    corr = X._get_numeric_data().corr()

    # Generate a mask for the upper triangle

    mask = np.zeros_like(corr, dtype=np.bool)

    mask[np.triu_indices_from(mask)] = True



    # Setting up the figure

    f, ax = plt.subplots(figsize=(7,5))



    # Generate a custom diverging colormap

    cmap = sns.diverging_palette(220, 10, as_cmap=True)



    # Draw the heatmap with the mask and correct aspect ratio

    heatmap = sns.heatmap(corr, mask=mask, cmap=cmap, center=0.0,

                          vmax = 1, square=True, linewidths=.5, ax=ax)

    plt.show()



    

def object_headers(X):

    """

    Divide the feature dataset into categorical and numerical parts.

    Returns the categorical feature headers.

    """

    out = []

    for cols in X.columns.tolist():

        if X[cols].dtype == "object":

            out.append(cols)

    return out





def display_uniques(X):

    """

    Display the unique values of features. 

    If the #unique values > 10 : display the first 10 uniques.

    """

    for features in X:

        if len(X[features].unique())<= 10:

            print(features,":"," Count = ",len(X[features].unique()),"\n",X[features].unique())

        else:

            print(features,":"," Count = ",len(X[features].unique()),"\n",X[features].unique()[:10])



            

def fit_and_pred_classifier(classifier, X_train, X_test, y_train, y_test):

    """

    Fit the model with X_train and Y_train and predict the 

    accuracy using X_test and Y_test. Print accuracy and return 

    predicted values.

    """

    # Fit the classifier to the training data

    classifier.fit(X_train, y_train)



    # Get the prediction array

    y_pred = classifier.predict(X_test)

    

    # Get the accuracy %

    print("Accuracy with selected features: " + str(accuracy_score(y_test, y_pred) * 100) + "%") 

    

    return y_pred





def object_feature_engineering(X):

    """

    Input : Entire dataframe

    Output : Dataframe where all categorical data are label encoded.

    """

    X_ret = X.copy(deep=True)

    """    X_ret["Gender"] = le.fit_transform(X_ret["Gender"].astype(str))

    X_ret["OverTime"] = le.fit_transform(X_ret["OverTime"].astype(str))"""

    X_ret["BusinessTravel"]=X_ret["BusinessTravel"].map({"Non-Travel": 0,"Travel_Rarely": 1,"Travel_Frequently": 2})

    categorical = object_headers(X_ret)

    for cat in categorical:

        X_ret[cat] = le.fit_transform(X_ret[cat].astype(str))

    return X_ret





def numerical_features(X):

    """

    Returns the Numerical Feature columns of the DataFrame

    """

    categorical = object_headers(X)

    numerical = X.columns.difference(categorical)

    X_categorical = X[categorical].astype("category")

    X_numerical = X[numerical]

    return X_numerical
# initial inputs and nullity checks

# Input Data

training_csv = pd.read_csv("../input/ee-769-assignment1/train.csv")



# remove duplicates if any

before_dedup = training_csv.shape[0]

training_csv.drop_duplicates(inplace=True)

print("Number of duplicates: ", before_dedup - training_csv.shape[0])



# Extracting training features and outputs

X = training_csv.drop(["Attrition"],axis=1) #independent variables

Y = training_csv["Attrition"]  #dependent variabe



# check the number of entries in the data

print("Data Vectors : ",len(X.index))



# check for null entries, if present then remove NA values by : X.dropna()

print("Checking for NULL entries : \n",X.isnull().any())



# checks for the number of features

print("Number of features : ",len(X.columns))

print(X.columns)



#check datatypes of features

print("\nData types : \n",X.dtypes)



categorical = object_headers(X)

print("The Categorical features are : \n",categorical)
X_in = X.drop(["EmployeeCount","EmployeeNumber","ID"],axis=1)

X_numerical = numerical_features(X_in)

print("BEFORE")

gen_correlation_map(X_numerical)



print("Visualize the 'Attrition' data distribution ")

f, axes = plt.subplots(2, 2, figsize=(8,6), sharex=False, sharey=False,constrained_layout=True)

sns.countplot(x=Y,data=Y,ax=axes[0,0])

axes[0,0].set( title ="Attrition Data Distribution")

sns.barplot(x="EnvironmentSatisfaction",y=Y,data=X_numerical,ax=axes[0,1])

axes[0,1].set( title = "EnvironmentSatisfaction VS Attrition")

sns.barplot(x="PerformanceRating",y=Y,data=X_numerical,ax=axes[1,0])

axes[1,0].set( title = "PerformanceRating vs Attrition")

sns.countplot(x="PerformanceRating",data=X_numerical,ax=axes[1,1])

axes[1,1].set( title = "PerformanceRating Distribution");
# for more insights lets drop the redundancies and plot again

remove = ["HourlyRate","DailyRate", "MonthlyRate","PerformanceRating"]

X_init = X_in.drop(remove,axis=1)

X_numerical = numerical_features(X_init)

print("AFTER")

gen_correlation_map(X_numerical)



X_numerical.describe()
# Label Encoding the categorical data

X_label_encoded = object_feature_engineering(X_init)

display(X_label_encoded.head(5))



# passing the data through StandardScaler to normalize the DataFrame

sc.fit(X_label_encoded)

X_standardized = sc.transform(X_label_encoded)
# intialize models to be used to fit and predict the TEST data

rf = RandomForestClassifier(class_weight="balanced",n_estimators=500,criterion="entropy",random_state=0)

svc = SVC(kernel = 'rbf', random_state = 0,gamma=0.001,C=100)

clf = LogisticRegression(random_state=0,solver="saga")

gnb = GaussianNB()

neigh = KNeighborsClassifier(n_neighbors=15)

mlp = MLPClassifier(hidden_layer_sizes=(3,3,3),max_iter=500)
# TEST data feature extraction

test_csv = pd.read_csv("../input/ee-769-assignment1/test.csv")

remove = ["EmployeeCount","EmployeeNumber","ID","HourlyRate","DailyRate", "MonthlyRate","PerformanceRating"]

test = test_csv.drop(remove,axis=1)

X_test_le = object_feature_engineering(test)

X_test_std = sc.transform(X_test_le)

classifiers = ["rf","svc","clf","gnb","neigh","mlp"]

id_data = np.asarray(test_csv["ID"])
def output(X_training,Y,X_test,classifier):

    dict_fit = {"rf":rf,"svc":svc,"clf":clf,"gnb":gnb,"neigh":neigh,"mlp":mlp}

    X_train, X_validate, y_train, y_validate = model_selection.train_test_split(X_training, Y, train_size=0.8,test_size=0.2, random_state=1)

    val_pred = fit_and_pred_classifier(dict_fit[classifier],X_train, X_validate, y_train, y_validate)

    y_test = dict_fit[classifier].predict(X_test)

    return y_test



def append_to_begin(file):

    with open(file, "r+") as f:

        old = f.read() # read everything in the file

        f.seek(0) # rewind

        f.write("ID,Attrition\n" + old) # write the new line before

        f.close()



def write_to_csv_le(X_training,Y,X_test,classifier):

    y_test = output(X_training,Y,X_test,classifier)

    out = np.c_[id_data,y_test]

    filename = "submission" + classifier.upper() + ".csv"

    np.savetxt(filename,out,delimiter=",",fmt="%d")

    append_to_begin(filename)

    



for i in classifiers:

    write_to_csv_le(X_standardized,Y,X_test_std,i)
def write_to_csv_ohe(X_training,Y,X_test,classifier):

    y_test = output(X_training,Y,X_test,classifier)

    out = np.c_[id_data,y_test]

    filename = "submission2" + classifier.upper() + ".csv"

    np.savetxt(filename,out,delimiter=",",fmt="%d")

    append_to_begin(filename)



# using ONE-Hot Encoder and doing the same

from sklearn.compose import ColumnTransformer

# training data

X_ohe = X_init.copy(deep=True)

columnTransformer = ColumnTransformer([('encoder', ohe, categorical)], remainder='passthrough')

X_ohe = pd.DataFrame(columnTransformer.fit_transform(X_ohe))

sc.fit(X_ohe)

X_ohe_std = sc.transform(X_ohe)

# test data

X_test_ohe = pd.DataFrame(columnTransformer.fit_transform(test))

X_test_ohe_std = sc.transform(X_test_ohe)



for i in classifiers:

    write_to_csv_ohe(X_ohe_std,Y,X_test_ohe_std,i)
# using RobustScaler in place of StandardScaler

# label encoded

def write_to_csv_le_rsc(X_training,Y,X_test,classifier):

    y_test = output(X_training,Y,X_test,classifier)

    out = np.c_[id_data,y_test]

    filename = "submission_rsc" + classifier.upper() + ".csv"

    np.savetxt(filename,out,delimiter=",",fmt="%d")

    append_to_begin(filename)





rsc.fit(X_label_encoded)

X_rsc_standardized = rsc.transform(X_label_encoded)

X_rsc_test_std = rsc.transform(X_test_le)

for i in classifiers:

    write_to_csv_le_rsc(X_rsc_standardized,Y,X_rsc_test_std,i)



def write_to_csv_ohe_rsc(X_training,Y,X_test,classifier):

    y_test = output(X_training,Y,X_test,classifier)

    out = np.c_[id_data,y_test]

    filename = "submission2_rsc" + classifier.upper() + ".csv"

    np.savetxt(filename,out,delimiter=",",fmt="%d")    

    append_to_begin(filename)

    

# one hot encoded

rsc.fit(X_ohe)

X_rsc_ohe_std = rsc.transform(X_ohe)

X_test_rsc_ohe_std = sc.transform(X_test_ohe)

for i in classifiers:

    write_to_csv_ohe_rsc(X_rsc_ohe_std,Y,X_test_rsc_ohe_std,i)