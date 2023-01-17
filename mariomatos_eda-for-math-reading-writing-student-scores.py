# Import all libraries to be used
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, cross_validate, validation_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve, plot_roc_curve

#data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
data = pd.read_csv('/kaggle/input/exams6k/exams.csv')
data
#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

data.describe()
# NOTE: # dev_null is used to produce no unnecessary output. Also it will not be used anywhere because it is just a variable to put anything meaningless. like /dev/null in linux
dev_null = sns.distplot(data["math score"])
dev_null.set(xlabel="Math Score", ylabel="Frequency")
dev_null = dev_null.set_title("Math Scores Distributions")
dev_null = sns.distplot(data["reading score"])
dev_null.set(xlabel="Reading Score", ylabel="Frequency")
dev_null = dev_null.set_title("Reading Scores Distributions")
dev_null = sns.distplot(data["writing score"])
dev_null.set(xlabel="Writing Score", ylabel="Frequency")
dev_null = dev_null.set_title("Writing Scores Distributions")
data.select_dtypes('object').nunique()
data.isnull().sum()
dev_null = sns.barplot(x="race/ethnicity", y="math score", hue="gender", data=data)
dev_null.set(xlabel="Group", ylabel="Math Score")
dev_null = dev_null.set_title("Math Scores By Group")
dev_null = sns.barplot(x="race/ethnicity", y="reading score", hue="gender", data=data)
dev_null.set(xlabel="Reading Score", ylabel="Frequency")
dev_null = dev_null.set_title("Reading Scores Distributions")
%matplotlib inline

plt.figure(figsize=(25,6))
plt.subplot(1, 3, 1)
sns.distplot(data['math score'])

plt.subplot(1, 3, 2)
sns.distplot(data['reading score'])

plt.subplot(1, 3, 3)
sns.distplot(data['writing score'])

plt.suptitle('Checking for Skewness', fontsize = 15)
plt.show()
dev_null = sns.heatmap(data.corr(), annot=True, fmt=".2f")
dev_null = dev_null.set_title("Frequency Distributions comparing scores")
data.info()
dev_null = sns.countplot(x="race/ethnicity", data=data)
dev_null.set(xlabel="Group", ylabel="Count")
dev_null = dev_null.set_title("Count of Group members")
dev_null = sns.barplot(x="race/ethnicity", y="writing score", hue="gender", data=data)
dev_null.set(xlabel="Group", ylabel="Writing Score")
dev_null = dev_null.set_title("Writing Scores by Group")
countplot = sns.countplot(x="parental level of education", data=data)
countplot.set_xticklabels(countplot.get_xticklabels(), rotation=40, ha="right")
countplot.set(xlabel="Parental Education Lvl", ylabel="Count")
dev_null = countplot.set_title("Count of Students by Education attained by their parents")
dev_null = sns.boxplot(x="race/ethnicity", y="math score", hue="gender", data=data)
dev_null.set(xlabel="Group", ylabel="Math Score")
dev_null = dev_null.set_title("Math Scores by Group")
# check counts for relationship between race/ethnicity and parental education level. 
parent_edu_vs_eth_race = pd.crosstab(index=data["race/ethnicity"], columns=data["parental level of education"])
dev_null = sns.heatmap(parent_edu_vs_eth_race)
dev_null.set(xlabel="Parent Education Lvl", ylabel="Group")
dev_null = dev_null.set_title("Correlation between Groups and Parent Educ. Lvl")
parent_edu_vs_eth_race
bar = sns.barplot(x="parental level of education", y="math score", hue="race/ethnicity", data=data)
bar.set_xticklabels(bar.get_xticklabels(), rotation=40, ha="right")
bar.set(xlabel="Parental Educ. Lvl", ylabel="Math Score")
dev_null = bar.set_title("Relating Parental Educ. Lvl to Math Score")
bar = sns.boxplot(x="parental level of education", y="math score", hue="race/ethnicity", data=data)
bar.set_xticklabels(bar.get_xticklabels(), rotation=40, ha="right")
bar.set(xlabel="Parent Educ. Lvl", ylabel="Math Score")
dev_null = bar.set_title("Math Score and Parent Educ. Lvl")
data["Pass"] = data.apply(lambda x: 1 if x["math score"] >= 65 and x["reading score"] >= 65 and x["writing score"] >= 65 else 0, axis=1)
data = data.drop(["math score", "reading score", "writing score"], axis=1)
data.select_dtypes(include="object")
data

X = data.drop(["Pass"], axis=1)
y = data["Pass"]
X,y
# Add using different parameters.
# Encoding categorical inputs
encoder = OneHotEncoder(handle_unknown="ignore")
encoder.fit(X)
X = encoder.transform(X)

# 80/20 train split ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

mlp = MLPClassifier(
    max_iter=3000,
    hidden_layer_sizes=[17, 13, 7], 
    solver="sgd", 
    random_state=1,
    verbose=False
).fit(X_train, y_train)

y_predicted = mlp.predict(X_test)

y_predicted, y_test.to_numpy() # Todo compare 

def format_scores_as_dataframe(labels, train_scores, test_scores):
    learning_data = {"labels": [], "type": [], "score": []}

    for i in range(len(train_sizes)):
        for j in range(len(train_scores)):
            learning_data["labels"].append(labels[i])
            learning_data["type"].append("train")
            learning_data["score"].append(train_scores[i][j])
            learning_data["labels"].append(labels[i])
            learning_data["type"].append("test")
            learning_data["score"].append(test_scores[i][j])
            
    return pd.DataFrame.from_dict(learning_data)
# Learning Curve | Complexity Curve
train_sizes, train_scores, test_scores = learning_curve(mlp, X, y)

learning_curve_df = format_scores_as_dataframe(train_sizes, train_scores, test_scores)

# train and test learning scores results
ax = sns.lineplot(x="labels", y="score", hue="type", data=learning_curve_df, marker="o", ci=None)
ax.set_title("Learning Curve for MLP Algorithm")
dev_null = ax.set(xlabel="Samples", ylabel="Error")
scores = cross_val_score(mlp, X, y)

scores, scores.mean(), scores.std()

dev_null = sns.lineplot(x=[1,2,3,4,5], y=scores)
dev_null.set_title("Cross Score Distribution")
dev_null = dev_null.set(xlabel="# of runs", ylabel="Accuracy")
cross_val_result = cross_validate(mlp, X, y, return_train_score=True)

#validation_curve(mlp, X, y, param_name="alpha", param_range=[0.0001, 0.001, 0.05])
train_scores, test_scores = validation_curve(mlp, X, y, param_name="hidden_layer_sizes", param_range=([5], [10], [10,5], [15, 10], [25,10,5]))

val_curve_data = {"labels": [], "type": [], "scores": []}
param_ranges = ["[5]", "[10]", "[10,5]", "[15,10]", "[25,10,5]"]

for i in range(len(train_scores)):
    for j in range(len(train_scores[i])):
        val_curve_data["labels"].append(param_ranges[i])
        val_curve_data["type"].append("train")
        val_curve_data["scores"].append(train_scores[i][j])
        val_curve_data["labels"].append(param_ranges[i])
        val_curve_data["type"].append("test")
        val_curve_data["scores"].append(test_scores[i][j])
        
val_curve_df = pd.DataFrame.from_dict(val_curve_data)

ax = sns.lineplot(x="labels", y="scores", hue="type", data = val_curve_df, marker="o", ci=None)
ax.set_title("Validation Curve for our MLP model")
dev_null = ax.set(xlabel="Layers/Neurons", ylabel="Accuracy Score")



confusion_mtrx = confusion_matrix(y_test, y_predicted)
classification_rprt = classification_report(y_test, y_predicted)
accuracy_scr = accuracy_score(y_test, y_predicted)
# TN FP
# FN TP
print("Confusion Matrix")
print(confusion_mtrx)
print("Classification Report")
print(classification_rprt)
print("Accuracy")
print(accuracy_scr)
# Computing AUC score
roc = roc_auc_score(y_test, y_predicted)
dev_null = plot_roc_curve(mlp, X_test, y_test, name="AUC/ROC Curve for MLP")
parameters = { # parameters commented to make running time shorter
    "hidden_layer_sizes": [[8], [5], [2]],#, [8,8], [8,5], [5,8], [5,2], [2,2], [8,5,2], [8,5,5], [13,8,4], [17,13,7]],
    "activation": ["identity", "logistic"],#, "tanh", "relu"], 
    "solver": ["lbfgs", "sgd"],#, "adam"], 
    "max_iter": [200, 500],#, 1000, 2000, 3000, 5000]
}

# Brace yourself, this will take a while
mlp = MLPClassifier()
gs = GridSearchCV(mlp, parameters)
gs.fit(X_train, y_train)
gs.predict(X_test)
gs.best_estimator_
X = data.drop(["Pass"], axis=1)
y = data["Pass"]
X,y
# Encoding categorical inputs
encoder = OneHotEncoder(handle_unknown="ignore")
encoder.fit(X)
X = encoder.transform(X)

# 80/20 train split ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

mlp = MLPClassifier(
    max_iter=10000,
    hidden_layer_sizes=[100], 
    activation="logistic",
    random_state=1,
    verbose=False
).fit(X_train, y_train)

y_predicted = mlp.predict(X_test)

y_predicted, y_test.to_numpy() # Todo compare 
# Learning Curve | Complexity Curve

train_sizes, train_scores, test_scores = learning_curve(mlp, X, y)

learning_curve_df = format_scores_as_dataframe(train_sizes, train_scores, test_scores)

# train and test learning scores results
ax = sns.lineplot(x="labels", y="score", hue="type", data=learning_curve_df, marker="o", ci=None)
ax.set_title("Learning Curve for MLP Algorithm")
dev_null = ax.set(xlabel="Samples", ylabel="Error")
scores = cross_val_score(mlp, X, y)

scores, scores.mean(), scores.std()

dev_null = sns.lineplot(x=[1,2,3,4,5], y=scores)
dev_null.set_title("Cross Score Distribution")
dev_null = dev_null.set(xlabel="# of runs", ylabel="Accuracy")
cross_val_result = cross_validate(mlp, X, y, return_train_score=True)

#validation_curve(mlp, X, y, param_name="alpha", param_range=[0.0001, 0.001, 0.05])
train_scores, test_scores = validation_curve(mlp, X, y, param_name="hidden_layer_sizes", param_range=([5], [10], [10,5], [15, 10], [25,10,5]))

val_curve_data = {"labels": [], "type": [], "scores": []}
param_ranges = ["[5]", "[10]", "[10,5]", "[15,10]", "[25,10,5]"]

for i in range(len(train_scores)):
    for j in range(len(train_scores[i])):
        val_curve_data["labels"].append(param_ranges[i])
        val_curve_data["type"].append("train")
        val_curve_data["scores"].append(train_scores[i][j])
        val_curve_data["labels"].append(param_ranges[i])
        val_curve_data["type"].append("test")
        val_curve_data["scores"].append(test_scores[i][j])
        
val_curve_df = pd.DataFrame.from_dict(val_curve_data)

ax = sns.lineplot(x="labels", y="scores", hue="type", data = val_curve_df, marker="o", ci=None)
ax.set_title("Validation Curve for our MLP model")
dev_null = ax.set(xlabel="Layers/Neurons", ylabel="Accuracy Score")
confusion_mtrx = confusion_matrix(y_test, y_predicted)
classification_rprt = classification_report(y_test, y_predicted)
accuracy_scr = accuracy_score(y_test, y_predicted)
# TN FP
# FN TP
print("Confusion Matrix")
print(confusion_mtrx)
print("Classification Report")
print(classification_rprt)
print("Accuracy")
print(accuracy_scr)
# Computing AUC score
roc = roc_auc_score(y_test, y_predicted)
dev_null = plot_roc_curve(mlp, X_test, y_test, name="AUC/ROC Curve for MLP")
X = data.drop(["Pass"], axis=1)
y = data["Pass"]
X,y
# Encoding categorical inputs
encoder = OneHotEncoder(handle_unknown="ignore")
encoder.fit(X)
X = encoder.transform(X)

# 80/20 train split ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=1)

mlp = MLPClassifier(
    max_iter=3000,
    hidden_layer_sizes=[2], 
    solver="sgd",
    activation="identity",
    random_state=1,
    verbose=False
).fit(X_train, y_train)

y_predicted = mlp.predict(X_test)

y_predicted, y_test.to_numpy() # Todo compare 
# Learning Curve | Complexity Curve

train_sizes, train_scores, test_scores = learning_curve(mlp, X, y)

learning_curve_df = format_scores_as_dataframe(train_sizes, train_scores, test_scores)

# train and test learning scores results
ax = sns.lineplot(x="labels", y="score", hue="type", data=learning_curve_df, marker="o", ci=None)
ax.set_title("Learning Curve for MLP Algorithm")
dev_null = ax.set(xlabel="Samples", ylabel="Error")
scores = cross_val_score(mlp, X, y)

scores, scores.mean(), scores.std()

dev_null = sns.lineplot(x=[1,2,3,4,5], y=scores)
dev_null.set_title("Cross Score Distribution")
dev_null = dev_null.set(xlabel="# of runs", ylabel="Accuracy")
cross_val_result = cross_validate(mlp, X, y, return_train_score=True)

train_scores, test_scores = validation_curve(mlp, X, y, param_name="alpha", param_range=[0.1, 5, 10])
#train_scores, test_scores = validation_curve(mlp, X, y, param_name="hidden_layer_sizes", param_range=([2], [7], [2,2], [7, 2], [10,7,2]))

val_curve_data = {"labels": [], "type": [], "scores": []}
param_ranges = ["[2]", "[7]", "[2,2]", "[7,2]", "[10,7,2]"]

for i in range(len(train_scores)):
    for j in range(len(train_scores[i])):
        val_curve_data["labels"].append(param_ranges[i])
        val_curve_data["type"].append("train")
        val_curve_data["scores"].append(train_scores[i][j])
        val_curve_data["labels"].append(param_ranges[i])
        val_curve_data["type"].append("test")
        val_curve_data["scores"].append(test_scores[i][j])
        
val_curve_df = pd.DataFrame.from_dict(val_curve_data)

ax = sns.lineplot(x="labels", y="scores", hue="type", data = val_curve_df, marker="o", ci=None)
ax.set_title("Validation Curve for our MLP model")
dev_null = ax.set(xlabel="Layers/Neurons", ylabel="Accuracy Score")
confusion_mtrx = confusion_matrix(y_test, y_predicted)
classification_rprt = classification_report(y_test, y_predicted)
accuracy_scr = accuracy_score(y_test, y_predicted)
# TN FP
# FN TP
print("Confusion Matrix")
print(confusion_mtrx)
print("Classification Report")
print(classification_rprt)
print("Accuracy")
print(accuracy_scr)
# Computing AUC score
roc = roc_auc_score(y_test, y_predicted)
dev_null = plot_roc_curve(mlp, X_test, y_test, name="AUC/ROC Curve for MLP")