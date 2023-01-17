# Loading required libraries

import eli5

import lime

import numpy as np

import pandas as pd

import seaborn as sns

import lime.lime_tabular

from sklearn.svm import SVC

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from eli5.sklearn import PermutationImportance

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.inspection import plot_partial_dependence

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score







# Loading the dataset

df = pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")



print(df.shape)

df.head()
df.isna().sum()
# Create just a simple lambda to calculate the proportion of each group 

prop_func = lambda x: round(len(x) / df.shape[0], 5) * 100



df.groupby("Attrition")["Attrition"].agg(["count", prop_func])
f,ax = plt.subplots(figsize=(15,8))

sns.kdeplot(df.loc[df.Attrition == "Yes", "Age"], shade = True, label = "Left")

sns.kdeplot(df.loc[df.Attrition == "No", "Age"], shade = True, label = "Stayed")

ax.set(xlabel = "Age", ylabel = "Density",title = "Age density colored by wether or not the employee left")

ax.set_xticks(range(10, 70, 2))

plt.show()
f,ax = plt.subplots(figsize=(15,8))



# Get the proportion of the genders grouped by the attrition status

grouped_data = df["Gender"].groupby(df["Attrition"]).value_counts(normalize = True).rename("Percentage of group").reset_index()



# Plot the result

sns.barplot(x = "Attrition", y = "Percentage of group", hue = "Gender", data = grouped_data)



# Convert y axis to percentage format

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])



ax.set(title = "Distribution of gender by each of the groups")

plt.show()
f,ax = plt.subplots(figsize=(15,8))

grouped_data = df["JobSatisfaction"].groupby(df["Attrition"]).value_counts(normalize = True).rename("Percentage of group").reset_index()

sns.barplot(x = "JobSatisfaction", y = "Percentage of group", hue = "Attrition", data = grouped_data)



# Convert y axis to percentage format

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])



ax.set(title = "Distribution of job satisfaction by each of the groups")

plt.show()
f,ax = plt.subplots(figsize=(15,8))

grouped_data = df["OverTime"].groupby(df["Attrition"]).value_counts(normalize = True).rename("Percentage of group").reset_index()

sns.barplot(x = "OverTime", y = "Percentage of group", hue = "Attrition", data = grouped_data)



# Convert y axis to percentage format

vals = ax.get_yticks()

ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])



ax.set(title = "Distribution of overtime by each of the groups")

plt.show()
df["Salary hike above average"] = df.PercentSalaryHike > df.PercentSalaryHike.mean()



df_reordered = df.sort_values(by=['Attrition'])



f,ax = plt.subplots(figsize=(20,10))

ax = sns.scatterplot(x="Age", y="MonthlyIncome", hue="Attrition", size = "Salary hike above average", data=df_reordered, alpha = 0.8, sizes = (80,20))

ax.set(ylabel = "Monthly Income", title = "Distribution of overtime by each of the groups")

ax.set_xticks(range(10, 70, 2))

plt.show()
# I'm also removing these columns because they're either an ID or constant and have no variation

correlation = df.drop(["EmployeeCount", "EmployeeNumber", "StandardHours", "Salary hike above average"], axis = 1).corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(correlation, dtype = np.bool)

mask[np.triu_indices_from(mask)] = True



f,ax = plt.subplots(figsize=(20,10))

sns.heatmap(correlation, annot = True, cmap = sns.color_palette("Reds", 11), mask = mask)

plt.show()
#Lets deal with the categorical cloumns now

# simply change yes/no to 1/0 for Attrition, Gender, OverTime

df['Attrition'].replace({'No': 0, 'Yes': 1},inplace = True)

df['Gender'].replace({'Male': 0, 'Female': 1},inplace = True)

df['OverTime'].replace({'No': 0, 'Yes': 1},inplace = True)



# specifying names of categorical columns to be one-hot encoded

categorical_columns = ['BusinessTravel', 'Department', 'EducationField', "JobRole", "MaritalStatus"]



# transform the categorical columns

df = pd.get_dummies(df, columns=categorical_columns)



# Removing unecessary features

df.drop(["EmployeeNumber", "EmployeeCount", "Over18", "Salary hike above average", "StandardHours"], axis = 1, inplace = True)



df.head()
scaler = MinMaxScaler()

df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

df.head()
# specify our x and y variables

x, y = df.drop(["Attrition"], axis = 1), df.Attrition



# instantiate our selector to select the top 10 features

selector = SelectKBest(f_classif, k = 10)



# fit our selector to the data

x = pd.DataFrame(selector.fit_transform(x, y), index = x.index, columns = x.columns[selector.get_support(indices = True)])



# see what are the top selected features from our univariate filter

x.columns
# Splitting our total dataset so that we have 20% of it in the test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 888)



# Set the parameters by cross-validation

tuned_parameters = {

    "gamma" : [0.0001, 0.001, 0.1, 0.5, 1, 1.5, 2, 5, 7.5, 8, 10],

    "C": [0.0005, 0.001, 0.01, 0.1, 1, 8, 10, 100, 250, 500, 1000, 1500]

}



# Creating our support vector machine with class weights for maximizing balance between classes

estimator = SVC(class_weight = "balanced", kernel = "rbf")



# specifying the 5-fold cross validation

# With 5-fold cross validation we'll have 5 iterations where we train our model in 4 parts of the trainig set and validate it on the fifth-part

grid_searcher = GridSearchCV(estimator, tuned_parameters, cv = 5, scoring='roc_auc', verbose = 1)

# Trainig our model

grid_searcher.fit(x_train, y_train)

print(f"Best parameters set found on development set: \n{grid_searcher.best_params_}")
# Using grid_searcher best parameters

svm_classifier = SVC(class_weight = "balanced", C = 0.1, gamma = 1, kernel = "rbf", probability = True)

svm_classifier.fit(x_train, y_train)



# Predict our model on the test set

y_predicted = svm_classifier.predict(x_test)



fig = plt.figure(figsize=(15,8))

heatmap = sns.heatmap(data = pd.DataFrame(confusion_matrix(y_test, y_predicted)), annot = True, fmt = "d", cmap=sns.color_palette("Reds", 50))

heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)

plt.ylabel('Ground Truth')

plt.xlabel('Prediction')

plt.show()
print(f"""Accuray: {round(accuracy_score(y_test, y_predicted), 5) * 100}%

ROC-AUC: {round(roc_auc_score(y_test, y_predicted), 5) * 100}%""")

print(classification_report(y_test, y_predicted))
cost_percentage = 0.214



# I need to read the data again because we've already done standardization an we need to original values back to calculate the cost

df_unstanderdized = pd.read_csv("../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")



# I'm selecting only the employees that were present in the test set

df_unstanderdized = df_unstanderdized.iloc[x_test.index, :]



df_unstanderdized["predicted_values"] = y_predicted
predicted_leavers = df_unstanderdized.loc[(df_unstanderdized.predicted_values == 1) & (df_unstanderdized.Attrition == "Yes"), :]



cost_reduction = predicted_leavers.apply(lambda row: (row.MonthlyIncome * 14) * cost_percentage, axis = 1)
print(f"The total ammount we could have saved this company is ${np.sum(cost_reduction)}!")
perm = PermutationImportance(svm_classifier, random_state=1).fit(x_test, y_test)

eli5.show_weights(perm, feature_names = x_test.columns.tolist())
plot_partial_dependence(svm_classifier, x, [0, 2], feature_names = x.columns)
plot_partial_dependence(svm_classifier, x, [5, 6, 7], feature_names = x.columns)
# Create an explainer object

explainer = lime.lime_tabular.LimeTabularExplainer(training_data = x_train.values, feature_names = x_train.columns.values, discretize_continuous = True, mode = "classification", verbose = True, random_state = 888)



# Explain the first employee in the test set

exp = explainer.explain_instance(x_test.values[0,:], svm_classifier.predict_proba, num_features = x_test.shape[1])



#Plot local explanation

plt = exp.as_pyplot_figure()

plt.tight_layout()

exp.show_in_notebook(show_table = True)