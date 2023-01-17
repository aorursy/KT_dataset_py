! pip install -q dabl
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from colorama import Fore, Style



import dabl



from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler



plt.style.use("classic")

warnings.filterwarnings('ignore')
def cout(string: str, color=Fore.RED):

    """

    Saves some work 😅

    """

    print(color+string+Style.RESET_ALL)
def statistics(dataframe, column):

    cout(f"The Average value in {column} is: {dataframe[column].mean():.2f}", Fore.RED)

    cout(f"The Maximum value in {column} is: {dataframe[column].max()}", Fore.BLUE)

    cout(f"The Minimum value in {column} is: {dataframe[column].min()}", Fore.YELLOW)

    cout(f"The 25th Quantile of {column} is: {dataframe[column].quantile(0.25)}", Fore.GREEN)

    cout(f"The 50th Quantile of {column} is: {dataframe[column].quantile(0.50)}", Fore.CYAN)

    cout(f"The 75th Quantile of {column} is: {dataframe[column].quantile(0.75)}", Fore.MAGENTA)
data = pd.read_csv("../input/health-care-data-set-on-heart-attack-possibility/heart.csv")

data.head()
data.isna().sum()
# Print Age Column Statistics

statistics(data, 'age')
# Let's plot the age column too

plt.style.use("classic")

sns.distplot(data['age'], color='blue')

plt.title(f"Age Distribution [\u03BC : {data['age'].mean():.2f} years | \u03C3 : {data['age'].std():.2f} years]")

plt.xlabel("Age")

plt.ylabel("Count")

plt.show()
statistics(data, 'sex')
# Pie Chart

plt.style.use("classic")

plt.figure(figsize=(10, 8))

target = [len(data[data['sex'] == 0]), len(data[data['sex'] == 1])]

labels = ["Female", "Male"]

plt.pie(x=target, labels=labels, autopct='%1.2f%%')

plt.title("Gender Values Distribution")

plt.show()
statistics(data, "cp")
# Count Plot

plt.style.use("classic")

plt.figure(figsize=(10, 8))

sns.countplot(data['cp'])

plt.xlabel("Chest Pain Severity Type")

plt.ylabel("Count")

plt.title("Chest Pain Severity Count Plot")

plt.show()
statistics(data, 'trestbps')
# Dist plot of Resting Blood Pressure

plt.style.use("classic")

sns.distplot(data['trestbps'], color='green')

plt.title(f"Resting BP Count [\u03BC : {data['trestbps'].mean():.2f} mmHg | \u03C3 : {data['trestbps'].std():.2f} mmHg]")

plt.xlabel("Resting Blood Pressure (in mmHg)")

plt.ylabel("Count")

plt.show()
statistics(data, "chol")
# Dist plot of Resting Blood Pressure

plt.style.use("classic")

sns.distplot(data['chol'], color='red')

plt.title(f"Serum Cholestrol Count [\u03BC : {data['chol'].mean():.2f} mg/dl | \u03C3 : {data['chol'].std():.2f} mg/dl]")

plt.xlabel("Serum Cholestrol (in mg/dl)")

plt.ylabel("Count")

plt.show()
statistics(data, "fbs")
# Pie Chart

plt.style.use("fivethirtyeight")

plt.figure(figsize=(10, 8))

target = [len(data[data['fbs'] == 0]), len(data[data['fbs'] == 1])]

labels = ["Less than 120 mg/dl", "Greater than 120 mg/dl"]

plt.pie(x=target, labels=labels, autopct='%1.2f%%', explode=[0, 0.1])

plt.title("FBS Values Distribution")

plt.show()
statistics(data, "restecg")
# Count Plot

plt.style.use("ggplot")

plt.figure(figsize=(10, 8))

sns.countplot(data['restecg'])

plt.xlabel("Resting ECG Type")

plt.ylabel("Count")

plt.title("Resting ECG Results")

plt.show()
statistics(data, "thalach")
# Dist plot of Resting Blood Pressure

plt.style.use("classic")

sns.distplot(data['thalach'], color='black')

plt.title(f"Maximum Heart Rate [\u03BC : {data['thalach'].mean():.2f} bpm | \u03C3 : {data['thalach'].std():.2f} bpm]")

plt.xlabel("Maximum Heart Rate Achieved (in bpm)")

plt.ylabel("Count")

plt.show()
statistics(data, "exang")
# Pie Chart

plt.style.use("ggplot")

plt.figure(figsize=(10, 8))

target = [len(data[data['exang'] == 0]), len(data[data['exang'] == 1])]

labels = ["0", "1"]

plt.pie(x=target, labels=labels, autopct='%1.2f%%', explode=[0, 0.05])

plt.title("Exercise Induced Angina Values Distribution")

plt.show()
statistics(data, "oldpeak")
# Dist plot of Resting Blood Pressure

plt.style.use("seaborn-dark")

sns.distplot(data['oldpeak'], color='orange')

plt.title(f"ST Depression [ \u03BC : {data['oldpeak'].mean():.2f} | \u03C3 : {data['oldpeak'].std():.2f} ]")

plt.xlabel("ST Depression Induced by Exercise")

plt.ylabel("Count")

plt.show()
statistics(data, "slope")
# Count Plot

plt.style.use("classic")

plt.figure(figsize=(10, 8))

sns.countplot(data['slope'])

plt.xlabel("Slope")

plt.ylabel("Count")

plt.title("Slope of Peak Exercise")

plt.show()
statistics(data, "ca")
# Count Plot

plt.style.use("fivethirtyeight")

plt.figure(figsize=(10, 8))

sns.countplot(data['ca'])

plt.xlabel("Major Vessels")

plt.ylabel("Count")

plt.title("Number of Major Vessels colored by flouropsy")

plt.show()
statistics(data, "thal")
# Count Plot

plt.style.use("ggplot")

plt.figure(figsize=(10, 8))

sns.countplot(data['thal'])

plt.xlabel("Type of Defect")

plt.ylabel("Count")

plt.title("Type of Defect count")

plt.show()
statistics(data, "target")
# Pie Chart

plt.style.use("fivethirtyeight")

plt.figure(figsize=(10, 8))

target = [len(data[data['target'] == 0]), len(data[data['target'] == 1])]

labels = ["Less Chances", "High Chances"]

plt.pie(x=target, labels=labels, autopct='%1.2f%%', explode=[0, 0])

plt.title("Chances of Heart Attack - Target")

plt.show()
plt.style.use("classic")

plt.figure(figsize=(18, 9))

sns.violinplot(data['sex'], data['age'], palette='Dark2')

plt.title("Age v/s Sex")

plt.xlabel("Gender (0-Female, 1-Male)")

plt.ylabel("Age")

plt.show()
plt.style.use("ggplot")

plt.figure(figsize=(18, 9))

sns.boxplot(data['cp'], data['age'])

plt.title("Age v/s Chest Pain Type")

plt.xlabel("Chest Pain type")

plt.ylabel("Age")

plt.show()
plt.style.use("fivethirtyeight")

plt.figure(figsize=(18, 9))

sns.swarmplot(data['cp'], data['chol'])

plt.title("Serum Cholestrol v/s Chest Pain Type")

plt.xlabel("Chest Pain type")

plt.ylabel("Serum Cholestrol (in mg/dl)")

plt.show()
plt.figure(figsize=(10, 6))

sns.set(style='ticks')

scatter_data = data[["trestbps", "thalach", "target"]]

sns.pairplot(scatter_data)

plt.show()
plt.figure(figsize=(16, 9))

sns.pairplot(data.corr())

plt.show()
plt.figure(figsize=(12, 9))

sns.heatmap(data.corr(), annot=True)

plt.title("Correlation Heatmap", fontsize=25)

plt.show()
# Split the data into training and testing sets

split_pcent = 0.15

split = int(len(data) * split_pcent)



data = data.sample(frac=1)



train = data[split:]

test = data[:split]



trainX = train.drop(['target'], axis=1)

trainY = train['target']



testX = test.drop(['target'], axis=1)

testY = test['target']



cout(f"Training Data Shape is: {train.shape}", Fore.RED)

cout(f"Testing Data Shape is: {test.shape}", Fore.BLUE)
# Standard Scaling the Data

sc = StandardScaler()

trainX = sc.fit_transform(trainX)

testX = sc.transform(testX)
# Define all the classifiers we will be using

names = ["Logistic Regression", "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",

         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",

         "Naive Bayes", "QDA"]



classifiers = [

    LogisticRegression(),

    KNeighborsClassifier(3),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    MLPClassifier(alpha=1, max_iter=1000),

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis()

]
clf_results_roc = {}

clf_results_acc = {}



for name, clf in zip(names, classifiers):    

    # Fit on the traning data

    clf.fit(trainX, trainY)

    

    # Get the test time prediction

    preds = clf.predict(testX)

    

    # Calculate Test ROC_AUC

    score = roc_auc_score(testY, preds)

    

    # Calculate the val accuracy

    val_acc = clf.score(testX, testY)

    

    # Store the results in a dictionary

    clf_results_roc[name] = score

    clf_results_acc[name] = val_acc

    

    cout(f"Classifier: {name}", Fore.YELLOW)

    cout(f"\nval_acc: {val_acc:.2f} | roc_auc: {score:.2f}\n", Fore.BLUE)
# Sort the Model Accuracies based on the test score

sort_clf = dict(sorted(clf_results_acc.items(), key=lambda x: x[1], reverse=True))



# Get the names and the corresponding scores

clf_names = list(sort_clf.keys())[::-1]

clf_scores = list(sort_clf.values())[::-1]



# Plot the results

plt.style.use("fivethirtyeight")

plt.figure(figsize=(14, 8))

sns.barplot(x=clf_names, y=clf_scores)

plt.xlabel("Models")

plt.ylabel("Validation Accuracy")

plt.xticks(rotation=45)

plt.title("Model Comparison - Validation Accuracy")

plt.show()
# Sort the Model Accuracies based on the roc-auc score

sort_clf = dict(sorted(clf_results_roc.items(), key=lambda x: x[1], reverse=True))



# Get the names and the corresponding scores

clf_names = list(sort_clf.keys())[::-1]

clf_scores = list(sort_clf.values())[::-1]



# Plot the results

plt.style.use("fivethirtyeight")

plt.figure(figsize=(14, 8))

sns.barplot(x=clf_names, y=clf_scores)

plt.xlabel("Models")

plt.ylabel("ROC-AUC Score")

plt.xticks(rotation=45)

plt.title("Model Comparison - ROC-AUC Scores")

plt.show()