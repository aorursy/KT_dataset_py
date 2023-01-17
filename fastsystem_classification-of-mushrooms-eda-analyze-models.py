import numpy as np  # Importing NumPy library
import pandas as pd  # Importing Pandas library
import matplotlib.pyplot as plt  # Importing Matplotlib library's "pyplot" module
import seaborn as sns  # Imorting Seaborn library

# Ignore all warnings:
import warnings
warnings.filterwarnings("ignore")

# This lines for Kaggle:
import os
print(os.listdir("../input"))
data = pd.read_csv("../input/mushrooms.csv")  # Read CSV file and load into "data" variable
data.info()  # Show detailed information for dataset columns(attributes)
data.head()  # Prints first 5 entries of the dataset
data.tail()  # Prints last 5 entries of the dataset
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder class

label_encoder = LabelEncoder()  # Create a instance for the label encoder
encoded_data = pd.DataFrame()  # Create empty DataFrame

for column in data.columns:
    encoded_data[column] = label_encoder.fit_transform(data[column])  # Iterate all columns and transform its values
encoded_data.head()  # Print first 5 record for the encoded data
encoded_data.describe()  # Print some statistics for data
encoded_data.corr()  # Prints correlation matrix
fig, axes = plt.subplots(figsize=(18, 18))  # This method creates a figure and a set of subplots
sns.heatmap(data=encoded_data.corr(), annot=True, linewidths=.5, cmap="coolwarm", ax=axes)  # Figure out heatmap
plt.show()  # Shows only plot and remove other informations
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 13))  # Adjust subplots

# Draw frequency of the "bruises" values according to "class":
bruises_bar = sns.countplot(x="bruises", hue="class", data=data, ax=axes[0][0]);
bruises_bar.set_xticklabels(["True", "False"])

# Draw frequency of the "gill-spacing" values according to "class":
gill_spacing_bar = sns.countplot(x="gill-spacing", hue="class", data=data, ax=axes[0][1]);
gill_spacing_bar.set_xticklabels(["Close", "Crowded", "Distant"])

# Draw frequency of the "gill-size" values according to "class":
gill_size_bar = sns.countplot(x="gill-size", hue="class", data=data, ax=axes[0][2]);
gill_size_bar.set_xticklabels(["Narrow", "Broad"])

# Draw frequency of the "gill-color" values according to "class":
gill_color_bar = sns.countplot(x="gill-color", hue="class", data=data, ax=axes[1][0]);
gill_color_bar.set_xticklabels(["Black", "Brown", "Gray", "Pink", "White", "Choco", "Purple", "Red", "Buff", "Green", "Yellow", "Orange"], rotation=60)

# Draw frequency of the "stalk-root" values according to "class":
stalk_root_bar = sns.countplot(x="stalk-root", hue="class", data=data, ax=axes[1][1]);
stalk_root_bar.set_xticklabels(["Equal", "Club", "Bulbous", "Rooted", "None"], rotation=60)

# Draw frequency of the "stalk-surface-above-ring" values according to "class":
stalk_sar_bar = sns.countplot(x="stalk-surface-above-ring", hue="class", data=data, ax=axes[1][2]);
stalk_sar_bar.set_xticklabels(["Smooth", "Fibrous", "Silky", "Scaly"], rotation=60)

# Draw frequency of the "stalk-surface-below-ring" values according to "class":
stalk_sbr_bar = sns.countplot(x="stalk-surface-below-ring", hue="class", data=data, ax=axes[2][0]);
stalk_sbr_bar.set_xticklabels(["Smooth", "Fibrous", "Silky", "Scaly"], rotation=60)

# Draw frequency of the "ring-type" values according to "class":
ring_type_bar = sns.countplot(x="ring-type", hue="class", data=data, ax=axes[2][1]);
ring_type_bar.set_xticklabels(["Pendant", "Evanescent", "Large", "Flaring", "None"], rotation=60)

# Draw frequency of the "population" values according to "class":
population_bar = sns.countplot(x="population", hue="class", data=data, ax=axes[2][2]);
population_bar.set_xticklabels(["Scattered", "Numerous", "Abundant", "Several", "Solitary", "Clustered"], rotation=60)

fig.tight_layout()  # Slightly spacing between axis labels and values
plt.show()
poisonous_count = len(data[data["class"] == "p"].index)  # Get poisonous count
edible_count = len(data[data["class"] == "e"].index)  # Get edible count

# Draw Pie Chart:
plt.pie([poisonous_count, edible_count], labels=["Poisonous", "Edible"], autopct='%1.1f%%', radius=2.0, shadow=True, colors=["r", "g"])
plt.show()
fig, ax = plt.subplots(figsize=(12,8))  # For specify figure size
data.groupby(['habitat', 'class']).size().unstack().plot.bar(stacked=True, ax=ax)  # Draw Stacked Bar Chart
plt.show()
encoded_data.drop(["cap-shape", "veil-type"], axis=1, inplace=True)  # Drop "cap-shape" and "veil-type" columns from dataset
encoded_data.columns
X = encoded_data.drop(["class"], axis=1)  # Put all data (except "class" column) to the X variable
y = encoded_data["class"] # Put only "class" column to the Y variable
X.head()
y.head()
from sklearn.preprocessing import StandardScaler  # Import StandartScaler class
std_scaler = StandardScaler()  # Create instance for scaler

X = std_scaler.fit_transform(X)  # Fit and transform data with scaler
from sklearn.decomposition import PCA  # Import class for PCA

for i in range(2, 20):
    pca = PCA(n_components=i)  # n_components = Specify the number of features you want to reduce.
    pca.fit_transform(X)
    print("Sum of Variance Ratio in " + str(i) + " Components: ", sum(pca.explained_variance_ratio_))
pca = PCA(n_components=13)  # We will reduce the feature count to the 13
X = pca.fit_transform(X)  # Fit and transform with data

print(sum(pca.explained_variance_ratio_))  # Print variance ratio
from sklearn.model_selection import train_test_split  # Import "train_test_split" method

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Parameters:
# test_size : It decides how many test data in percentage.
# random_state : This parameter can take any value. This value decides randomness seed.
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
logistic_regression_cls = LogisticRegression(random_state=13)  # Create instance for model
logistic_regression_cls.fit(x_train, y_train)  # Fit data with model

print("Train Score for Logistic Regression: ", logistic_regression_cls.score(x_train, y_train))  # Print Train Score
print("Test Score for Logistic Regression: ", logistic_regression_cls.score(x_test, y_test))  # Print Test Score
logistic_regression_cls.get_params()  # Print hyperparameters and their values for the model
naive_bayes_cls = GaussianNB()  # Create instance for model
naive_bayes_cls.fit(x_train, y_train)  # Fit data with model

print("Train Score for Gaussian Naive Bayes: ", naive_bayes_cls.score(x_train, y_train))  # Print Train Score
print("Test Score for Gaussian Naive Bayes: ", naive_bayes_cls.score(x_test, y_test))  # Print Test Score
naive_bayes_cls.get_params()  # Print hyperparameters and their values for the model
decision_tree_cls = DecisionTreeClassifier(random_state=13)  # Create instance for model
decision_tree_cls.fit(x_train, y_train)  # Fit data with model

print("Train Score for Decision Tree: ", decision_tree_cls.score(x_train, y_train))  # Print Train Score
print("Test Score for Decision Tree: ", decision_tree_cls.score(x_test, y_test))  # Print Test Score
decision_tree_cls.get_params()  # Print hyperparameters and their values for the model
random_forest_cls = RandomForestClassifier(random_state=13)  # Create instance for model
random_forest_cls.fit(x_train, y_train)  # Fit data with model

print("Train Score for Random Forest: ", random_forest_cls.score(x_train, y_train))  # Print Train Score
print("Test Score for Random Forest: ", random_forest_cls.score(x_test, y_test))  # Print Test Score
random_forest_cls.get_params()  # Print hyperparameters and their values for the model
support_vector_cls = SVC(random_state=13)  # Create instance for model
support_vector_cls.fit(x_train, y_train)  # Fit data with model

print("Train Score for SVC: ", support_vector_cls.score(x_train, y_train))  # Print Train Score
print("Test Score for SVC: ", support_vector_cls.score(x_test, y_test))  # Print Test Score
support_vector_cls.get_params()  # Print hyperparameters and their values for the model
knn_cls = KNeighborsClassifier()  # Create instance for model
knn_cls.fit(x_train, y_train)  # Fit data with model

print("Train Score for K-NN: ", knn_cls.score(x_train, y_train))  # Print Train Score
print("Test Score for K-NN: ", knn_cls.score(x_test, y_test))  # Print Test Score
knn_cls.get_params()  # Print hyperparameters and their values for the model
# Specifying hyperparameters' range for the model:
parameters_LR = {"C" : [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                 "penalty" : ["l1", "l2"]}

# Create a Grid Search Cross Validation instance ("n_jobs=-1" means use all cores of the processor):
grid_search_LR = GridSearchCV(logistic_regression_cls, parameters_LR, cv=6, n_jobs=-1, return_train_score=True)

# Fit Grid Search model with data:
grid_search_LR.fit(x_train, y_train)

# Show results for all combinations:
pd.DataFrame(grid_search_LR.cv_results_)[["params", "mean_fit_time", "mean_train_score", "mean_test_score"]]
grid_search_LR.best_params_  # Print best hyperparameters for the model
# Specifying hyperparameters' range for the model:
parameters_DT = {"max_depth" : np.arange(3, 20)}

# Create a Grid Search Cross Validation instance ("n_jobs=-1" means use all cores of the processor):
grid_search_DT = GridSearchCV(decision_tree_cls, parameters_DT, cv=6, n_jobs=-1, return_train_score=True)

# Fit Grid Search model with data:
grid_search_DT.fit(x_train, y_train)

# Show results for all combinations:
pd.DataFrame(grid_search_DT.cv_results_)[["params", "mean_fit_time", "mean_train_score", "mean_test_score"]]
grid_search_DT.best_params_  # Print best hyperparameters for the model
# Specifying hyperparameters' range for the model:
parameters_RF = {"max_depth" : np.arange(5, 15),
                 "n_estimators" : [100, 200, 300]}

# Create a Grid Search Cross Validation instance ("n_jobs=-1" means use all cores of the processor):
grid_search_RF = GridSearchCV(random_forest_cls, parameters_RF, cv=6, n_jobs=-1, return_train_score=True)

# Fit Grid Search model with data:
grid_search_RF.fit(x_train, y_train)

# Show results for all combinations:
pd.DataFrame(grid_search_RF.cv_results_)[["params", "mean_fit_time", "mean_train_score", "mean_test_score"]]
grid_search_RF.best_params_  # Print best hyperparameters for the model
# Specifying hyperparameters' range for the model:
parameters_SVC = {"C" : [0.1, 1, 10, 100],
                 "gamma" : [0.001, 0.01, 0.1, 1]}

# Create a Grid Search Cross Validation instance ("n_jobs=-1" means use all cores of the processor):
grid_search_SVC = GridSearchCV(support_vector_cls, parameters_SVC, cv=6, n_jobs=-1, return_train_score=True)

# Fit Grid Search model with data:
grid_search_SVC.fit(x_train, y_train)

# Show results for all combinations:
pd.DataFrame(grid_search_SVC.cv_results_)[["params", "mean_fit_time", "mean_train_score", "mean_test_score"]]
grid_search_SVC.best_params_  # Print best hyperparameters for the model
# Specifying hyperparameters' range for the model:
parameters_KNN = {"n_neighbors" : np.arange(2, 30)}

# Create a Grid Search Cross Validation instance ("n_jobs=-1" means use all cores of the processor):
grid_search_KNN = GridSearchCV(knn_cls, parameters_KNN, cv=6, n_jobs=-1, return_train_score=True)

# Fit Grid Search model with data:
grid_search_KNN.fit(x_train, y_train)

# Show results for all combinations:
pd.DataFrame(grid_search_KNN.cv_results_)[["params", "mean_fit_time", "mean_train_score", "mean_test_score"]]
grid_search_KNN.best_params_  # Print best hyperparameters for the model
# Logistic Regression Classifier:
logistic_regression_cls_tuned = LogisticRegression(C=0.1, penalty="l2", random_state=13)
logistic_regression_cls_tuned.fit(x_train, y_train)

# Gaussian Naive Bayes Classifier (Not Changed):
naive_bayes_cls_tuned = GaussianNB()
naive_bayes_cls_tuned.fit(x_train, y_train)

# Decision Tree Classifier:
decision_tree_cls_tuned = DecisionTreeClassifier(max_depth=14, random_state=13)
decision_tree_cls_tuned.fit(x_train, y_train)

# Random Forest Classifier:
random_forest_cls_tuned = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=13)
random_forest_cls_tuned.fit(x_train, y_train)

# Support Vector Classifier:
support_vector_cls_tuned = SVC(kernel="rbf", C=1, gamma=0.1, random_state=13)
support_vector_cls_tuned.fit(x_train, y_train)

# K-Nearest Neighbors Classifier:
knn_cls_tuned = KNeighborsClassifier(n_neighbors=3)
knn_cls_tuned.fit(x_train, y_train)

# Find test accuracy for all models:
print("Test Score for Logistic Regression: ", logistic_regression_cls_tuned.score(x_test, y_test))
print("Test Score for Gaussian Naive Bayes: ", naive_bayes_cls_tuned.score(x_test, y_test))
print("Test Score for Decision Tree: ", decision_tree_cls_tuned.score(x_test, y_test))
print("Test Score for Random Forest: ", random_forest_cls_tuned.score(x_test, y_test))
print("Test Score for SVC: ", support_vector_cls_tuned.score(x_test, y_test))
print("Test Score for K-NN: ", knn_cls_tuned.score(x_test, y_test))
from sklearn.metrics import confusion_matrix  # For find confusion matrix
from sklearn.metrics import classification_report  # For print evaluation report
from sklearn.metrics import roc_curve  # For drawing ROC curve
from sklearn.metrics import auc  # For find AUC
# Prediction of test dataset:
y_pred_LR = logistic_regression_cls_tuned.predict(x_test)

# Find confusion matrix for this model:
confusion_matrix_LR = confusion_matrix(y_test, y_pred_LR)

# Plot confusion matrix with Heatmap:
cm_dataframe_LR = pd.DataFrame(confusion_matrix_LR, index=["Edible", "Poisonous"], columns=["Edible", "Poisonous"])
sns.heatmap(cm_dataframe_LR, annot=True, annot_kws={"size": 18}, fmt="d")
plt.title("Logistic Regression")
plt.ylabel('Actual Classes')
plt.xlabel('Predicted Classes')
plt.show()
report_LR = pd.DataFrame(classification_report(y_test, y_pred_LR, 
                                               output_dict=True, 
                                               target_names=["Edible", "Poisonous"]))
report_LR
# Find parameters for drawing ROC curve:
false_positive_rate_LR, true_positive_rate_LR, thresholds_LR = roc_curve(y_test, y_pred_LR)

# Find AUC value:
auc_LR = auc(false_positive_rate_LR, true_positive_rate_LR)

# Draw ROC curve:
plt.figure(figsize=(12, 12))
plt.plot(false_positive_rate_LR, true_positive_rate_LR, label="AUC = %0.2f"%auc_LR)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend(loc='lower right')
plt.title("Receiver Operating Characteristic(ROC) for Logistic Regression")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
# Prediction of test dataset:
y_pred_GNB = naive_bayes_cls_tuned.predict(x_test)

# Find confusion matrix for this model:
confusion_matrix_GNB = confusion_matrix(y_test, y_pred_GNB)

# Plot confusion matrix with Heatmap:
cm_dataframe_GNB = pd.DataFrame(confusion_matrix_GNB, index=["Edible", "Poisonous"], columns=["Edible", "Poisonous"])
sns.heatmap(cm_dataframe_GNB, annot=True, annot_kws={"size": 18}, fmt="d")
plt.title("Gaussian Naive Bayes")
plt.ylabel('Actual Classes')
plt.xlabel('Predicted Classes')
plt.show()
report_GNB = pd.DataFrame(classification_report(y_test, y_pred_GNB, 
                                                output_dict=True, 
                                                target_names=["Edible", "Poisonous"]))
report_GNB
# Find parameters for drawing ROC curve:
false_positive_rate_GNB, true_positive_rate_GNB, thresholds_GNB = roc_curve(y_test, y_pred_GNB)

# Find AUC value:
auc_GNB = auc(false_positive_rate_GNB, true_positive_rate_GNB)

# Draw ROC curve:
plt.figure(figsize=(12, 12))
plt.plot(false_positive_rate_GNB, true_positive_rate_GNB, label="AUC = %0.2f"%auc_GNB)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend(loc='lower right')
plt.title("Receiver Operating Characteristic(ROC) for Gaussian Naive Bayes")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
# Prediction of test dataset:
y_pred_DT = decision_tree_cls_tuned.predict(x_test)

# Find confusion matrix for this model:
confusion_matrix_DT = confusion_matrix(y_test, y_pred_DT)

# Plot confusion matrix with Heatmap:
cm_dataframe_DT = pd.DataFrame(confusion_matrix_DT, index=["Edible", "Poisonous"], columns=["Edible", "Poisonous"])
sns.heatmap(cm_dataframe_DT, annot=True, annot_kws={"size": 18}, fmt="d")
plt.title("Decision Tree Classifier")
plt.ylabel('Actual Classes')
plt.xlabel('Predicted Classes')
plt.show()
report_DT = pd.DataFrame(classification_report(y_test, y_pred_DT, 
                                               output_dict=True, 
                                               target_names=["Edible", "Poisonous"]))
report_DT
# Find parameters for drawing ROC curve:
false_positive_rate_DT, true_positive_rate_DT, thresholds_DT = roc_curve(y_test, y_pred_DT)

# Find AUC value:
auc_DT = auc(false_positive_rate_DT, true_positive_rate_DT)

# Draw ROC curve:
plt.figure(figsize=(12, 12))
plt.plot(false_positive_rate_DT, true_positive_rate_DT, label="AUC = %0.2f"%auc_DT)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend(loc='lower right')
plt.title("Receiver Operating Characteristic(ROC) for Decision Tree Classifier")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
# Prediction of test dataset:
y_pred_RF = random_forest_cls_tuned.predict(x_test)

# Find confusion matrix for this model:
confusion_matrix_RF = confusion_matrix(y_test, y_pred_RF)

# Plot confusion matrix with Heatmap:
cm_dataframe_RF = pd.DataFrame(confusion_matrix_RF, index=["Edible", "Poisonous"], columns=["Edible", "Poisonous"])
sns.heatmap(cm_dataframe_RF, annot=True, annot_kws={"size": 18}, fmt="d")
plt.title("Random Forest Classifier")
plt.ylabel('Actual Classes')
plt.xlabel('Predicted Classes')
plt.show()
report_RF = pd.DataFrame(classification_report(y_test, y_pred_RF, 
                                               output_dict=True, 
                                               target_names=["Edible", "Poisonous"]))
report_RF
# Find parameters for drawing ROC curve:
false_positive_rate_RF, true_positive_rate_RF, thresholds_RF = roc_curve(y_test, y_pred_RF)

# Find AUC value:
auc_RF = auc(false_positive_rate_RF, true_positive_rate_RF)

# Draw ROC curve:
plt.figure(figsize=(12, 12))
plt.plot(false_positive_rate_RF, true_positive_rate_RF, label="AUC = %0.2f"%auc_RF)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend(loc='lower right')
plt.title("Receiver Operating Characteristic(ROC) for Random Forest Classifier")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
# Prediction of test dataset:
y_pred_SVC = support_vector_cls_tuned.predict(x_test)

# Find confusion matrix for this model:
confusion_matrix_SVC = confusion_matrix(y_test, y_pred_SVC)

# Plot confusion matrix with Heatmap:
cm_dataframe_SVC = pd.DataFrame(confusion_matrix_SVC, index=["Edible", "Poisonous"], columns=["Edible", "Poisonous"])
sns.heatmap(cm_dataframe_SVC, annot=True, annot_kws={"size": 18}, fmt="d")
plt.title("SVC")
plt.ylabel('Actual Classes')
plt.xlabel('Predicted Classes')
plt.show()
report_SVC = pd.DataFrame(classification_report(y_test, y_pred_SVC, 
                                                output_dict=True, 
                                                target_names=["Edible", "Poisonous"]))
report_SVC
# Find parameters for drawing ROC curve:
false_positive_rate_SVC, true_positive_rate_SVC, thresholds_SVC = roc_curve(y_test, y_pred_SVC)

# Find AUC value:
auc_SVC = auc(false_positive_rate_SVC, true_positive_rate_SVC)

# Draw ROC curve:
plt.figure(figsize=(12, 12))
plt.plot(false_positive_rate_SVC, true_positive_rate_SVC, label="AUC = %0.2f"%auc_SVC)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend(loc='lower right')
plt.title("Receiver Operating Characteristic(ROC) for SVC")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
# Prediction of test dataset:
y_pred_KNN = knn_cls_tuned.predict(x_test)

# Find confusion matrix for this model:
confusion_matrix_KNN = confusion_matrix(y_test, y_pred_KNN)

# Plot confusion matrix with Heatmap:
cm_dataframe_KNN = pd.DataFrame(confusion_matrix_KNN, index=["Edible", "Poisonous"], columns=["Edible", "Poisonous"])
sns.heatmap(cm_dataframe_KNN, annot=True, annot_kws={"size": 18}, fmt="d")
plt.title("K-NN Classifier")
plt.ylabel('Actual Classes')
plt.xlabel('Predicted Classes')
plt.show()
report_KNN = pd.DataFrame(classification_report(y_test, y_pred_KNN, 
                                                output_dict=True, 
                                                target_names=["Edible", "Poisonous"]))
report_KNN
# Find parameters for drawing ROC curve:
false_positive_rate_KNN, true_positive_rate_KNN, thresholds_KNN = roc_curve(y_test, y_pred_KNN)

# Find AUC value:
auc_KNN = auc(false_positive_rate_KNN, true_positive_rate_KNN)

# Draw ROC curve:
plt.figure(figsize=(12, 12))
plt.plot(false_positive_rate_KNN, true_positive_rate_KNN, label="AUC = %0.2f"%auc_KNN)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.legend(loc='lower right')
plt.title("Receiver Operating Characteristic(ROC) for K-NN Classifier")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
# Create list that keep model names:
model_names = ["Logistic Regression", 
               "Gaussian Naive Bayes",
               "Decision Tree Classification",
               "Random Forest Classification",
               "Support Vector Classification",
               "K-NN Classifiaction"]

# Shortening models' variables names for quick usage:
LR_model = logistic_regression_cls_tuned
GNB_model = naive_bayes_cls_tuned
DT_model = decision_tree_cls_tuned
RF_model = random_forest_cls_tuned
SVC_model = support_vector_cls_tuned
KNN_model = knn_cls_tuned

# Get model hyperparameters into variables:
LR_model_C = str(LR_model.get_params()["C"])
LR_model_penalty = str(LR_model.get_params()["penalty"])
DT_mode_max_depth = str(DT_model.get_params()["max_depth"])
RF_model_n_estimators = str(RF_model.get_params()["n_estimators"])
RF_model_max_depth = str(RF_model.get_params()["max_depth"])
SVC_model_kernel = str(SVC_model.get_params()["kernel"])
SVC_model_C = str(SVC_model.get_params()["C"])
SVC_model_gamma = str(SVC_model.get_params()["gamma"])
KNN_model_n_neighbors = str(KNN_model.get_params()["n_neighbors"])

# Create list that keep model hyperparameters:
model_params = ["C=" + LR_model_C + ", penalty=" + LR_model_penalty,
                "None",
                "max_depth=" + DT_mode_max_depth,
                "n_estimators=" + RF_model_n_estimators + ", max_depth=" + RF_model_max_depth,
                "kernel=" + SVC_model_kernel + ", C=" + SVC_model_C + ", gamma=" + SVC_model_gamma,
                "n_neighbors=" + KNN_model_n_neighbors]

# Create list that keep models' training accuracies:
model_training_accuracies = [LR_model.score(x_train, y_train),
                             GNB_model.score(x_train, y_train),
                             DT_model.score(x_train, y_train),
                             RF_model.score(x_train, y_train),
                             SVC_model.score(x_train, y_train),
                             KNN_model.score(x_train, y_train)]

# Create list that keep models' testing accuracies:
model_testing_accuracies = [LR_model.score(x_test, y_test),
                             GNB_model.score(x_test, y_test),
                             DT_model.score(x_test, y_test),
                             RF_model.score(x_test, y_test),
                             SVC_model.score(x_test, y_test),
                             KNN_model.score(x_test, y_test)]

# Create list that keep models' F1 scores:
model_f1_scores = [report_LR.iloc[0]["weighted avg"],
                   report_GNB.iloc[0]["weighted avg"],
                   report_DT.iloc[0]["weighted avg"],
                   report_RF.iloc[0]["weighted avg"],
                   report_SVC.iloc[0]["weighted avg"],
                   report_KNN.iloc[0]["weighted avg"]]

# Create list that keep models' precisions:
model_precisions = [report_LR.iloc[1]["weighted avg"],
                   report_GNB.iloc[1]["weighted avg"],
                   report_DT.iloc[1]["weighted avg"],
                   report_RF.iloc[1]["weighted avg"],
                   report_SVC.iloc[1]["weighted avg"],
                   report_KNN.iloc[1]["weighted avg"]]

# Create list that keep models' recalls:
model_recalls = [report_LR.iloc[2]["weighted avg"],
                report_GNB.iloc[2]["weighted avg"],
                report_DT.iloc[2]["weighted avg"],
                report_RF.iloc[2]["weighted avg"],
                report_SVC.iloc[2]["weighted avg"],
                report_KNN.iloc[2]["weighted avg"]]

# Create list that keep models' AUC values:
model_AUC_values = [auc_LR, auc_GNB, auc_DT, auc_RF, auc_SVC, auc_KNN]

# Generate table data with column names:
table_data = {"Parameters" : model_params,
              "Training Accuracy" : model_training_accuracies,
              "Testing Accuracy" : model_testing_accuracies,
              "F1 Score" : model_f1_scores,
              "Precision" : model_precisions,
              "Recall" : model_recalls,
              "AUC" : model_AUC_values}

# Create and print result table:
table_dataframe = pd.DataFrame(data=table_data, index=model_names)
table_dataframe
table_dataframe.iloc[:, 1:].plot(kind="bar", ylim=[0.8, 1.0], figsize=(14, 9)) # Y Limit: 0.8 - 1.0
plt.legend(loc='lower right')
plt.show()