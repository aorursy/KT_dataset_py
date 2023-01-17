#Importing the necessary libraries

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc

warnings.filterwarnings("ignore")
#Read the dataset from the path and see a preview of it

df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

df.head()
#Creating bins for age

lstBins = [20,40,50,60,70,90]

df["ageGrp"] = pd.cut(df["age"], bins = lstBins, labels = ["Young", "Young2Old", "Old", "Senior", "Fragile"])



#Binning the rest systolic BP

bpCatLst = [70,100,120,140,160,220]

df["bpGrp"] = pd.cut(df["trestbps"], bins = bpCatLst, labels = ["very low", "low", "normal","high","very high"])



#Binning the cholestrol levels

cholCatLst = [100,200,239,300,350,700]

df["cholGrp"] = pd.cut(df["chol"], bins = cholCatLst, labels = ["normal", "borderline high", "high","very high","risky high"])
#Creating a list of columns for which dummy variables will be created

colLst = ["ageGrp", "bpGrp", "cholGrp"]



#Iterating in the column list to create dummy variables

for col in colLst:

    

    dfTemp = pd.get_dummies(df[col], prefix = col, drop_first=True)

    

    #Merging the dummy variables with existing variables

    df = pd.concat([df,dfTemp] , axis = 1, join = "inner")

#Dropping the columns

df.drop(columns = ["age", "trestbps", "chol", "ageGrp", "bpGrp", "cholGrp"], inplace = True)
y = df["target"].copy()

X = df.drop(columns = ["target"]).copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
model_RF = RandomForestClassifier(n_estimators=200, random_state=42, max_depth = 2, n_jobs = -1)

model_RF.fit(X_train, y_train)

model_RF.score(X_test, y_test)
#Plotting the ROC curve for our model

y_pred = model_RF.predict(X_test)

y_pred_prob = model_RF.predict_proba(X_test)[:, 1]



#Printing the AUC value for the model

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

print("AUC = " + str(auc(fpr, tpr)))



#ROC curve

fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="-.", c=".1")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 8

plt.title('ROC curve')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
#Getting the order of importance of the features in the model

import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(model_RF, random_state=42).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
#Importing packages for constructing the Partial Dependence Plot

from pdpbox import pdp, get_dataset, info_plots



# Create the data that we will plot

pdp_thal = pdp.pdp_isolate(model=model_RF, dataset=X_test, model_features=X_test.columns, feature='cp')



# plot it

pdp.pdp_plot(pdp_thal, 'cp')

plt.show()

# 2 variable PDP of Chest Pain and Coloured Arteries

features_to_plot = ['cp', 'ca']

inter1  =  pdp.pdp_interact(model=model_RF, dataset=X_test, model_features=X_test.columns, features=features_to_plot)



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')

plt.show()