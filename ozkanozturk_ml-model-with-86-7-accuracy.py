# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read data

data = pd.read_csv("/kaggle/input/deep-learning-az-ann/Churn_Modelling.csv")
# First 10 rows of data

data.head(10)
# Statistical infos on data

data.describe()
# Checking types and nulls

data.info()
data.isnull().any()
data.drop(["RowNumber","CustomerId","Surname"], axis=1, inplace=True)
# Check data

data.head()
def bar_plot(variable):

    """

    Takes a variable as an input: For example "Gender"

    Outputs a bar plot and counts value

    """

    

    # creating a temporary data frame for stroring index and values

    df = pd.DataFrame({

        variable : data[variable].value_counts().sort_values().index, # getting index after counting values and sorting as ascending

        "Count": data[variable].value_counts().sort_values().values # getting values 

                     }) 

    

    # creating bar plot as a subplot defines as "ax"

    ax = df.plot(

            kind = "bar",

            x = variable,

            y = "Count",

            edgecolor="black",

            legend = False,       

            figsize = (3,3)

                )

    

    # showing percentage on bars 

    

    heights = []  # empty list for storing height of bar of each index item

    

    for patch in ax.patches:                  # loop for getting heights of each bar in ax subplot

        heights.append(patch.get_height())    # gets height of each bar or patch and add to list of heights

    

    for patch in ax.patches:                  # loop for showing percentage on bars

        ax.text(

            patch.get_x(),                    # x position of text

            patch.get_height()-350,           # y poisition of text

            "%" + str(round(patch.get_height() / sum(heights)*100)),  # text to be shown

            fontsize = 11,                     # fontsize of text

            color = "yellow"                   # color of text

         

        )

    # some explanation for plot

    

    plt.xticks(rotation = 0, fontsize = 10)

    plt.ylabel("Count", color="r", fontsize="10")

    plt.xlabel(variable, color="r", fontsize="10")

    plt.title(variable, fontsize="14", color = "r")

    plt.grid(axis="both", color="gray", linewidth=0.3)    # shows grid lines on plot

    plt.show()    
# defining a list contains the categorical variables

categoricals = ["Geography","Gender","HasCrCard","IsActiveMember","Exited","NumOfProducts"]



for each in categoricals:

    bar_plot(each)
# defining function to plot histogram

def hist_plot(variable):

    data[variable].plot(

        kind = "hist",

        bins = 50,

        figsize = (4,3),

        edgecolor = "black"

                        )

    

    plt.xticks(rotation = 0, fontsize = 10)

    plt.xlabel(variable, color="r", fontsize="10")

    plt.title("Frequency of {}".format(variable), fontsize="14", color = "r")

    plt.grid(axis="both", color="gray", linewidth=0.3)    # shows grid lines on plot

    plt.show()
numericals = ["CreditScore","Age","Tenure","Balance","EstimatedSalary"]

for each in numericals:

    hist_plot(each)
def count_plot(variable):

    plt.subplots(figsize=(4,3))   # creates subplots

    sns.countplot(

        x = variable,

        hue = "Exited",

        data = data,

            )

    plt.show()  



    # percentage calculation



    counts = data[data.Exited == 1].groupby(variable)["Exited"].count() 

    sums = data.groupby(variable)["Exited"].count()

    print("Percentage of Exited customers:")

    for i in range (counts.shape[0]):

        print(counts.index[i],"%",round(counts[i]/sums[i]*100))

count_plot("Geography")
count_plot('Gender')
count_plot('HasCrCard')
count_plot('IsActiveMember')
data.plot(kind="box", figsize=(14,3))

plt.show()
new_data = data.copy()  # make a copy of data



# definign a function that normalizes selected feature of a selected data



def normalize_feature(variable, data1):

    data1[variable] = (data1[variable] - np.min(data1[variable]))/(np.max(data1[variable] - np.min(data1[variable])))
# Normalizing some features in a for loop:

for each in ['CreditScore','Age', 'Tenure', 'Balance','EstimatedSalary']:

    normalize_feature(each, new_data)
new_data.plot(kind="box",figsize=(14,4))

plt.show()
def detect_outliers(data,feature):



    # first quartile Q1

    Q1 = np.percentile(data[feature], 25)

    # third quartile Q3

    Q3 = np.percentile(data[feature], 75)

    # IQR = Q3 - Q1

    IQR = Q3 - Q1

    # outlier step = IQR x 1.5

    outlier_step = IQR*1.5

    # outliers = Q1 - outlier step or Q3 + outlier_step 

    outliers = (data[feature] < Q1 - outlier_step) |(data[feature]>Q3 + outlier_step) 

    # detect indeces of outliers in features of df

    outlier_indexes= list(data[outliers].index)

    return outlier_indexes
outliers_CreditScore = detect_outliers(data,"CreditScore")

print("There are {} rows which are outliers in CreditScore column".format(len(outliers_CreditScore)))
outliers_Age = detect_outliers(data,"Age")

print("There are {} rows which are outliers in Age column".format(len(outliers_Age)))
import collections

outliers = collections.Counter((outliers_CreditScore + outliers_Age))



multiple_outliers = list()

for i,v in outliers.items():

    if v > 1:

        multiple_outliers.append(i)



if len(multiple_outliers) == 0:

    print("There is no row that has multiple outliers")

else:

    print("There are {} rows that have multiple outliers".format(len(multiple_outliers)))
sns.heatmap(data.corr(), annot = True, fmt = ".2f")

plt.show()
plt.subplots(figsize=(17,4))

ax=sns.countplot(

    x="Age",

    hue = "Exited",

    data = data

    )



ax.set_title("Age vs Exiting Customers", color = "red")

ax.grid(axis = "y", color = "yellow", linewidth = "0.4")

plt.show()
sns.catplot(

    data = data,

    x = "Age",

    y = "Balance",

    hue = "Exited",

    kind = "bar",

    height = 3.5,

    aspect = 5

)



plt.title("Age vs Balance mean vs Exited Customers", color = "red")

plt.grid(axis = "y", color = "yellow", linewidth = "0.4")

plt.show()
sns.pairplot(data[["CreditScore", "Age","Balance","EstimatedSalary","Exited"]])

plt.show()
df = data.copy()
df["Geography"] = df.Geography.astype("category")

df["Gender"] = df.Gender.astype("category")

df = pd.get_dummies(df,columns=["Geography","Gender"])
df.head()
df["Tenure"] = df.Tenure.astype("category")

df["NumOfProducts"] = df.NumOfProducts.astype("category")

df = pd.get_dummies(df, columns=(["Tenure","NumOfProducts"]))
for each in ["HasCrCard","IsActiveMember","Exited"]:

    df[each] = df[each].astype("category")
df.info()
df.head(3)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
X_train = df.drop("Exited", axis = 1)
for each in ["CreditScore", "Age", "Balance", "EstimatedSalary"]:

    normalize_feature(each, X_train)
X_train.head(3)
y_train = df["Exited"]
print("Length of X_train: ",len(X_train))

print("Shape of X_train: ", X_train.shape)

print("Length of Y_tain: ", len(y_train))

print("Shape of Y_train: ", y_train.shape)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X_train, 

    y_train, 

    test_size = 0.33, 

    random_state = 42

)



print("Length of X_train: ",len(X_train))

print("Length of X_test: ",len(X_test))

print("Length of y_train: ",len(y_train))

print("Length of y_test: ",len(y_test))
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression(random_state = 1)

lr.fit(X_train, y_train)

print("Accuracy with train data: ",round(lr.score(X_train,y_train)*100))

print("Accuracy with test data: ",round(lr.score(X_test, y_test)*100))

# Single run for 3-neighbors:

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

print("3-neighbors KNN accuracy with train data: ", round(knn.score(X_train,y_train)*100))

print("3-neighbors KNN accuracy with test data: ", round(knn.score(X_test,y_test)*100))
accuracy_list_train = []

accuracy_list_test = []

for each in range(1,20):

    knn = KNeighborsClassifier(n_neighbors = each)

    knn.fit(X_train, y_train)

    accuracy_list_train.append(round(knn.score(X_train, y_train)*100))

    accuracy_list_test.append(round(knn.score(X_test, y_test)*100))    



print("Max test accuracy is % {} @ neighbor value of {}".format(

    max(accuracy_list_test),accuracy_list_test.index(max(accuracy_list_test))+1)

     )
f,ax = plt.subplots(figsize=(10,6))

ax.plot(range(1,20), accuracy_list_train, label="Train accuracy")

ax.plot(range(1,20), accuracy_list_test, label="Test accuracy")

ax.legend()

plt.xlabel("N-neighbor", size = "12", color = "red")

plt.ylabel("Accuracy %", size = "12", color = "red")

plt.title("KNN Classification accuracy vs n-neighbor", size = 12, color = "red")

plt.grid()

plt.show()
from sklearn.svm import SVC



svm = SVC(random_state = 1)

svm.fit(X_train, y_train)



print("SVM accuracy with train data :", svm.score(X_train,y_train))

print("SVM accuracy with test data :", svm.score(X_test, y_test))
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()

nb.fit(X_train, y_train)



print("NB accuracy with train data :", nb.score(X_train,y_train))

print("NB accuracy with test data :", nb.score(X_test, y_test))
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)



print("DT accuracy with train data :", dt.score(X_train,y_train))

print("DT accuracy with test data :", dt.score(X_test, y_test))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 100, random_state = 42)



rf.fit(X_train, y_train)



print("RF accuracy with train data :", rf.score(X_train,y_train))

print("RF accuracy with test data :", rf.score(X_test, y_test))
rf_train_accuracy = []

rf_test_accuracy = []



for i in range(1,50):

    rf = RandomForestClassifier(n_estimators = i, random_state = 1)

    rf.fit(X_train, y_train)

    rf.fit(X_test, y_test)

    rf_train_accuracy.append(round(rf.score(X_train, y_train)*100))

    rf_test_accuracy.append(round(rf.score(X_test, y_test)*100))



print("Max test accuracy is % {} @ n_estimator value of {}".format(

    max(rf_test_accuracy),rf_test_accuracy.index(max(rf_test_accuracy))+1)

     )
f,ax = plt.subplots(figsize=(10,6))

ax.plot(range(1,50), rf_train_accuracy, label="Train accuracy")

ax.plot(range(1,50), rf_test_accuracy, label="Test accuracy")

ax.legend()

plt.xlabel("N-estimator", size = "12", color = "red")

plt.ylabel("Accuracy %", size = "12", color = "red")

plt.title("RF Classification accuracy vs n-estimator", size = 12, color = "red")

plt.grid()

plt.show()
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = lr, X = X_train, y = y_train, cv = 10)

print("accuracies :",accuracies)

print("mean accuracy :", accuracies.mean())
knn = KNeighborsClassifier()  # machine learning model

grid = {

    "n_neighbors" : [18,19,30],

    "leaf_size" : [1,2,3]

}



knn_gsCV = GridSearchCV(knn, grid, cv = 10, n_jobs = -1, verbose=1)  



knn_gsCV.fit(X_train, y_train)



print("Best parameter(n_neighbor): ", knn_gsCV.best_params_)

print("Best accuracy according to best parameter: ", knn_gsCV.best_score_)
knn_gsCV.best_estimator_
votingC = VotingClassifier(

    estimators = [("KNN",knn_gsCV.best_estimator_)],

    voting = "soft",

    n_jobs = -1

)



votingC.fit(X_train, y_train)



print("Accuracy score:",votingC.score(X_test, y_test))
# Creating list of classifiers that i want to compare

random_state = 42

classifier_list = [

    DecisionTreeClassifier(random_state = random_state),

    SVC(random_state = random_state),

    RandomForestClassifier(random_state = random_state),

    LogisticRegression(random_state = random_state),

    KNeighborsClassifier()

]



# Creating grids for tuneable parameters

dt_param_grid = {

    "min_samples_split":range(10,500,20),

    "max_depth": range(1,20,2)

} 



svc_param_grid = {

    "kernel" : ["rbf"],

    "gamma": [0.001, 0.01, 0.1, 1],

    "C": [1,10,50,100,200,300,1000]

}



rf_param_grid = {"max_features": [1,3,10],

                "min_samples_split":[2,3,10],

                "min_samples_leaf":[1,3,10],

                "bootstrap":[False],

                "n_estimators":[100,300],

                "criterion":["gini"]}



logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty": ["l1","l2"]}



knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}



# creating list for grids 

classifier_param = [dt_param_grid,

                   svc_param_grid,

                   rf_param_grid,

                   logreg_param_grid,

                   knn_param_grid]
cv_result = []         # cross validation results to be stacked

best_estimators = []   # best estimators to be stacked



for i in range(5):

    clf = GridSearchCV(

        classifier_list[i],

        param_grid = classifier_param[i],

        cv = StratifiedKFold(n_splits = 10),

        scoring = "accuracy",

        n_jobs = -1,

        verbose = 1

    )

    

    clf.fit(X_train, y_train)

    

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])

grid_search_results = pd.DataFrame(

    {

        "machine_learning_models": ["Decision Tree","SVC","Random Forest", "Logistic Regression","KNeighbors"],

        "best_accuricies" : cv_result

    }

)



plt.subplots(figsize=(10,8))

plt.bar(

    grid_search_results.machine_learning_models,

    grid_search_results.best_accuricies    

)



plt.xlabel("Machine Learning Models", color = "red", size = 10)

plt.ylabel("Best Accuricies", color = "red", size = 10)

plt.grid(axis = "y",color = "yellow")

plt.title("Grid Search Cross Validation Results")



plt.show()

best_estimators
votingCls = VotingClassifier(

    estimators = [

        ("dt", best_estimators[0]),

        ("rf",best_estimators[2])    

    ],

    voting = "soft",

    n_jobs = -1

)



votingCls.fit(X_train, y_train)
print("accuracy score of voting classifier: ", votingCls.score(X_test, y_test))
votingCls.predict(X_test)