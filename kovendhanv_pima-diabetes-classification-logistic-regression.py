#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, classification_report, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
file_path = "/kaggle/input/pima-indians-diabetes-database/diabetes.csv"
diabetes = pd.read_csv(file_path)
diabetes.shape
diabetes.head()
diabetes.info()
print("No. of people without diabetes: ",diabetes["Outcome"].value_counts()[0])
print("No. of people with diabetes : ",diabetes["Outcome"].value_counts()[1])
print("Percent of people with diabetes : ",round(diabetes["Outcome"].value_counts()[1]/len(diabetes.index)*100,2), "%")
sns.countplot("Outcome", data=diabetes);
#Descriptive Statistics - Five Point Summary
diabetes.describe().T
#Distribution of each feature

sns.set_style("darkgrid")

fig, ax2 = plt.subplots(4, 2, figsize=(16, 16))

sns.distplot(diabetes['Pregnancies'],ax=ax2[0][0], fit=norm)
sns.distplot(diabetes['Glucose'],ax=ax2[0][1], fit=norm)
sns.distplot(diabetes['BloodPressure'],ax=ax2[1][0], fit=norm)
sns.distplot(diabetes['SkinThickness'],ax=ax2[1][1], fit=norm)
sns.distplot(diabetes['Insulin'],ax=ax2[2][0], fit=norm)
sns.distplot(diabetes['BMI'],ax=ax2[2][1], fit=norm)
sns.distplot(diabetes['DiabetesPedigreeFunction'],ax=ax2[3][0], fit=norm)
sns.distplot(diabetes['Age'],ax=ax2[3][1], fit=norm)
#Outliers analysis of each feature

sns.set_style("darkgrid")

fig, ax2 = plt.subplots(4, 2, figsize=(16, 16))

sns.boxplot(diabetes['Pregnancies'],ax=ax2[0][0])
sns.boxplot(diabetes['Glucose'],ax=ax2[0][1])
sns.boxplot(diabetes['BloodPressure'],ax=ax2[1][0])
sns.boxplot(diabetes['SkinThickness'],ax=ax2[1][1])
sns.boxplot(diabetes['Insulin'],ax=ax2[2][0])
sns.boxplot(diabetes['BMI'],ax=ax2[2][1])
sns.boxplot(diabetes['DiabetesPedigreeFunction'],ax=ax2[3][0])
sns.boxplot(diabetes['Age'],ax=ax2[3][1])
sns.countplot("Pregnancies", data=diabetes);
pd.crosstab(diabetes["Pregnancies"], diabetes["Outcome"]).plot()
plt.figure(figsize=(10,10))
sns.pairplot(diabetes, diag_kind='kde', hue="Outcome")
plt.show()
plt.figure(figsize=(10,10))
sns.pairplot(diabetes, kind='reg', hue='Outcome')
plt.show()
plt.figure(figsize=(10,10))
sns.heatmap(diabetes.corr(), cmap='magma', vmin = -1, vmax = 1, annot=True, fmt="0.2f", square=True, linewidths=0.2)
plt.show()
sns.lmplot("Age", "Glucose", data=diabetes, hue='Outcome');
sns.lmplot("BloodPressure", "Glucose", data=diabetes, hue='Outcome');
sns.countplot("Pregnancies", hue="Outcome", data=diabetes)
sns.lmplot(y="Insulin",x="Glucose", hue="Outcome", data=diabetes);
diabetes.columns
#All features except Pregnancies - replacing 0s with NaNs
diabetes[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = diabetes[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(to_replace=0, value=np.nan)
diabetes.head()
diabetes.info()
print("Number of missing values in dataframe : \n", diabetes.isnull().sum())
print("-------------------------")
print("Percentage of columnwise missing values in dataframe : \n", round(diabetes.isnull().mean() * 100, 2))
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')

diabetes_cols = diabetes.columns

diabetes = imputer.fit_transform(diabetes)

diabetes = pd.DataFrame(diabetes, columns = diabetes_cols)

diabetes.head()
X = diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = diabetes['Outcome']
print("Shape of X :", X.shape)
print("Shape of y :",y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=diabetes["Outcome"], random_state=24)
X_train.shape
y_train.shape
X_test.shape
y_test.shape
X_train.sample(3)
X_test.sample(3)
from sklearn.preprocessing import MinMaxScaler
mmScaler = MinMaxScaler()
X_train_scaled = mmScaler.fit_transform(X_train.values)
X_test_scaled = mmScaler.fit_transform(X_test.values)

X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
X_train.sample(3)
X_test.sample(3)
#Logistic Regression with liblinear solver
log_reg_ml = LogisticRegression(solver = "liblinear")
log_reg_ml.fit(X_train, y_train)
#Accuracy score for train set
log_reg_ml.score(X_train, y_train)
#Accuracy score for test set
log_reg_ml.score(X_test, y_test)
#Let us predict the Test set
y_test_pred = log_reg_ml.predict(X_test)
#Let us predict the corresponding Probabilities for Test set
y_test_pred_prob = log_reg_ml.predict_proba(X_test)
#Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred, labels=[1, 0])
plt.figure(figsize = (7,5))
sns.heatmap(cm, annot=True, square=True, fmt = ".2f")
plt.show()
TP = cm[1,1] # True Positive
FN = cm[0,0] # False Negative
FP = cm[0,1] # False Positive
TN = cm[1,0] # True Negative
precision = TP/(TP+FP)
precision
recall = TP/(TP+FN)
recall
sensitivity = TP/(TP+FN)
sensitivity
specificity = TN / (TN + FP)
specificity
#Receiver Operator Characteristic Curve 
roc_auc_score(y_test, y_test_pred)
# Defining the function to plot the ROC curve
def draw_roc(y_test, y_test_pred ):
    fpr, tpr, thresholds = roc_curve( y_test, y_test_pred,
                                              drop_intermediate = False )
    auc_score = roc_auc_score(y_test, y_test_pred)
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()

    return None

# Calling the function
draw_roc(y_test, y_test_pred)
#Precision-Recall curve
p, r, thresholds = precision_recall_curve(y_test, y_test_pred)
plt.plot(thresholds, p[:-1], "b-")
plt.plot(thresholds, r[:-1], "g-")
plt.show()
print(classification_report(y_test, y_test_pred, labels=[1, 0]))