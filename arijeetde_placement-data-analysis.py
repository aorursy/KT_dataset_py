import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, plot_confusion_matrix

from sklearn.preprocessing import LabelEncoder, StandardScaler



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier



le = LabelEncoder()

ss = StandardScaler()
# I knew that there was a Sl_No. Column so i made it the index column

placement = pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv", index_col = "sl_no")
placement.head()
print(placement.info())

print('\nShape of Placement DataFrame: ', placement.shape)
print('Gender Types:                   ', placement['gender'].unique())

print('Senior_Secondary_Board Types:   ', placement['ssc_b'].unique())

print('Higher_Secondary_Board Types:   ', placement['hsc_b'].unique())

print('Higher_Secondary_Subject Types: ', placement['hsc_s'].unique())

print('Degree Types:                   ', placement['degree_t'].unique())

print('Work_Experience Types:          ', placement['workex'].unique())

print('Specialisation Types:           ', placement['specialisation'].unique())

print('Status Types:                   ', placement['status'].unique())
# Countplots for different Categorical variables

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows = 4, ncols = 2, figsize = (12, 22))



sns.countplot(x = placement['gender'], ax = ax1)

ax1.set_title('Gender Count Distribution', fontsize = 14)

ax1.set_xlabel('Gender', fontsize = 12)

ax1.set_ylabel('Count', fontsize = 12)



sns.countplot(x = placement['ssc_b'], ax = ax2)

ax2.set_title('SSC_Board Count Distribution', fontsize = 14)

ax2.set_xlabel('SSC_Board', fontsize = 12)

ax2.set_ylabel('Count', fontsize = 12)



sns.countplot(x = placement['hsc_b'], ax = ax3)

ax3.set_title('HSC_Board Count Distribution', fontsize = 14)

ax3.set_xlabel('HSC_Board', fontsize = 12)

ax3.set_ylabel('Count', fontsize = 12)



sns.countplot(x = placement['hsc_s'], ax = ax4)

ax4.set_title('HSC_Subjects Count Distribution', fontsize = 14)

ax4.set_xlabel('HSC_Subjects', fontsize = 12)

ax4.set_ylabel('Count', fontsize = 12)



sns.countplot(x = placement['degree_t'], ax = ax5)

ax5.set_title('Degree Count Distribution', fontsize = 14)

ax5.set_xlabel('Degree', fontsize = 12)

ax5.set_ylabel('Count', fontsize = 12)



sns.countplot(x = placement['workex'], ax = ax6)

ax6.set_title('Work_Experience Count Distribution', fontsize = 14)

ax6.set_xlabel('Work_Experience', fontsize = 12)

ax6.set_ylabel('Count', fontsize = 12)



sns.countplot(x = placement['specialisation'], ax = ax7)

ax7.set_title('Specialisation Count Distribution', fontsize = 14)

ax7.set_xlabel('Specialisation', fontsize = 12)

ax7.set_ylabel('Count', fontsize = 12)



sns.countplot(x = placement['status'], ax = ax8)

ax8.set_title('Status Count Distribution', fontsize = 14)

ax8.set_xlabel('Status', fontsize = 12)

ax8.set_ylabel('Count', fontsize = 12)
# Count plot for Gender with other Categorical variables

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows = 3, ncols = 2, figsize = (13, 17))



sns.countplot(x = placement['gender'], hue = placement['ssc_b'], ax = ax1)

ax1.set_title('Gender Count Distribution acc SSC_Board', fontsize = 14)

ax1.set_xlabel('Gender', fontsize = 12)

ax1.set_ylabel('Count', fontsize = 12)



sns.countplot(x = placement['gender'], hue = placement['hsc_b'], ax = ax2)

ax2.set_title('Gender Count Distribution acc HSC_Board', fontsize = 14)

ax2.set_xlabel('Gender', fontsize = 12)

ax2.set_ylabel('Count', fontsize = 12)



sns.countplot(x = placement['gender'], hue = placement['hsc_s'], ax = ax3)

ax3.set_title('Gender Count Distribution acc HSC_Subjects', fontsize = 14)

ax3.set_xlabel('Gender', fontsize = 12)

ax3.set_ylabel('Count', fontsize = 12)



sns.countplot(x = placement['gender'], hue = placement['degree_t'], ax = ax4)

ax4.set_title('Gender Count Distribution acc Degree_Type', fontsize = 14)

ax4.set_xlabel('Gender', fontsize = 12)

ax4.set_ylabel('Count', fontsize = 12)



sns.countplot(x = placement['gender'], hue = placement['workex'], ax = ax5)

ax5.set_title('Gender Count Distribution acc Work_Exp', fontsize = 14)

ax5.set_xlabel('Gender', fontsize = 12)

ax5.set_ylabel('Count', fontsize = 12)



sns.countplot(x = placement['gender'], hue = placement['specialisation'], ax = ax6)

ax6.set_title('Gender Count Distribution acc specialisation', fontsize = 14)

ax6.set_xlabel('Gender', fontsize = 12)

ax6.set_ylabel('Count', fontsize = 12)
# Analysis of diff percentages acc to Gender and Status

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(nrows = 5, ncols = 2, figsize = (14, 26))



sns.barplot(x = placement['gender'], y = placement['ssc_p'], color = (0.9,0.6,0.4,0.9), ax=ax1)

ax1.set_title('Average SSC Percentage acc to Gender', fontsize = 14)

ax1.set_xlabel('Gender', fontsize = 12)

ax1.set_ylabel('SSC Percentage', fontsize = 12)



sns.barplot(x = placement['gender'], y = placement['ssc_p'], hue = placement['status'], color = (0.9,0.6,0.4,0.9), ax=ax2)

ax2.set_title('Average SSC Percentage acc to Gender and Status', fontsize = 14)

ax2.set_xlabel('Gender', fontsize = 12)

ax2.set_ylabel('SSC Percentage', fontsize = 12)

ax2.legend(title = "status", loc = "center left", bbox_to_anchor = (1, 0.5))



sns.barplot(x = placement['gender'], y = placement['hsc_p'], color = (0.6,0.8,0.3,0.9), ax=ax3)

ax3.set_title('Average HSC Percentage acc to Gender', fontsize = 14)

ax3.set_xlabel('Gender', fontsize = 12)

ax3.set_ylabel('HSC Percentage', fontsize = 12)



sns.barplot(x = placement['gender'], y = placement['hsc_p'], hue = placement['status'], color = (0.6,0.8,0.3,0.9), ax=ax4)

ax4.set_title('Average HSC Percentage acc to Gender and Status', fontsize = 14)

ax4.set_xlabel('Gender', fontsize = 12)

ax4.set_ylabel('HSC Percentage', fontsize = 12)

ax4.legend(title = "status", loc = "center left", bbox_to_anchor = (1, 0.5))



sns.barplot(x = placement['gender'], y = placement['degree_p'], color = (0.2,0.5,0.9,0.9), ax=ax5)

ax5.set_title('Average Degree Percentage acc to Gender', fontsize = 14)

ax5.set_xlabel('Gender', fontsize = 12)

ax5.set_ylabel('Degree Percentage', fontsize = 12)



sns.barplot(x = placement['gender'], y = placement['degree_p'], hue = placement['status'], color = (0.2,0.5,0.9,0.9), ax=ax6)

ax6.set_title('Average Degree Percentage acc to Gender and Status', fontsize = 14)

ax6.set_xlabel('Gender', fontsize = 12)

ax6.set_ylabel('Degree Percentage', fontsize = 12)

ax6.legend(title = "status", loc = "center left", bbox_to_anchor = (1, 0.5))



sns.barplot(x = placement['gender'], y = placement['etest_p'], color = (0.4,0.9,0.6,0.9), ax=ax7)

ax7.set_title('Average Entrance Test Percentage acc to Gender', fontsize = 14)

ax7.set_xlabel('Gender', fontsize = 12)

ax7.set_ylabel('Entrance Test Percentage', fontsize = 12)



sns.barplot(x = placement['gender'], y = placement['etest_p'], hue = placement['status'], color = (0.4,0.9,0.6,0.9), ax=ax8)

ax8.set_title('Average Entrance Test Percentage acc to Gender and Status', fontsize = 14)

ax8.set_xlabel('Gender', fontsize = 12)

ax8.set_ylabel('Entrance Test Percentage', fontsize = 12)

ax8.legend(title = 'status', loc = 'center left', bbox_to_anchor = (1, 0.5))



sns.barplot(x = placement['gender'], y = placement['mba_p'], color = (0.8,0.4,0.8,0.9), ax=ax9)

ax9.set_title('Average MBA Percentage acc to Gender', fontsize = 14)

ax9.set_xlabel('Gender', fontsize = 12)

ax9.set_ylabel('MBA Percentage', fontsize = 12)



sns.barplot(x = placement['gender'], y = placement['mba_p'], hue = placement['status'], color = (0.8,0.4,0.8,0.9), ax=ax10)

ax10.set_title('Average MBA Percentage acc to Gender and Status', fontsize = 14)

ax10.set_xlabel('Gender', fontsize = 12)

ax10.set_ylabel('MBA Percentage', fontsize = 12)

ax10.legend(title = 'status', loc = 'center left', bbox_to_anchor = (1, 0.5))
# Analysis of SSC_Percentage acc to SSC_Board and Status

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 5))



sns.barplot(x = placement['ssc_b'], y = placement['ssc_p'], color = (0.9,0.6,0.4,0.9), ax=ax1)

ax1.set_title('Average SSC Percentage acc to SSC_Board', fontsize = 14)

ax1.set_xlabel('SSC_Board', fontsize = 12)

ax1.set_ylabel('SSC Percentage', fontsize = 12)



sns.barplot(x = placement['ssc_b'], y = placement['ssc_p'], hue = placement['status'], color = (0.9,0.6,0.4,0.9), ax=ax2)

ax2.set_title('Average SSC Percentage acc to SSC_Board and Status', fontsize = 14)

ax2.set_xlabel('SSC_Board', fontsize = 12)

ax2.set_ylabel('SSC Percentage', fontsize = 12)

ax2.legend(title = "status", loc = "center left", bbox_to_anchor = (1, 0.5))
# Analysis of HSC_Percentages acc to HSC_Board, HSC_Subjects and Status

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (14, 11))



sns.barplot(x = placement['hsc_b'], y = placement['hsc_p'], color = (0.6,0.8,0.3,0.9), ax=ax1)

ax1.set_title('Average HSC Percentage acc to HSC_Board', fontsize = 14)

ax1.set_xlabel('HSC_Board', fontsize = 12)

ax1.set_ylabel('HSC Percentage', fontsize = 12)



sns.barplot(x = placement['hsc_b'], y = placement['hsc_p'], hue = placement['status'], color = (0.6,0.8,0.3,0.9), ax=ax2)

ax2.set_title('Average HSC Percentage acc to HSC_Board and Status', fontsize = 14)

ax2.set_xlabel('HSC_Board', fontsize = 12)

ax2.set_ylabel('HSC Percentage', fontsize = 12)

ax2.legend(title = "status", loc = "center left", bbox_to_anchor = (1, 0.5))



sns.barplot(x = placement['hsc_s'], y = placement['hsc_p'], color = "orange", ax=ax3)

ax3.set_title('Average HSC Percentage acc to HSC_Subjects', fontsize = 14)

ax3.set_xlabel('HSC_Subjects', fontsize = 12)

ax3.set_ylabel('HSC Percentage', fontsize = 12)



sns.barplot(x = placement['hsc_s'], y = placement['hsc_p'], hue = placement['status'], color = "orange", ax=ax4)

ax4.set_title('Average HSC Percentage acc to HSC_Subjects and Status', fontsize = 14)

ax4.set_xlabel('HSC_Subjects', fontsize = 12)

ax4.set_ylabel('HSC Percentage', fontsize = 12)

ax4.legend(title = "status", loc = "center left", bbox_to_anchor = (1, 0.5))
# Analysis of Degree_Percentage acc to Degree_Type and Status

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 5))



sns.barplot(x = placement['degree_t'], y = placement['degree_p'], color = (0.2,0.5,0.9,0.9), ax=ax1)

ax1.set_title('Average Degree Percentage acc to Degree_Type', fontsize = 14)

ax1.set_xlabel('Degree_Type', fontsize = 12)

ax1.set_ylabel('Degree Percentage', fontsize = 12)



sns.barplot(x = placement['degree_t'], y = placement['degree_p'], hue = placement['status'], color = (0.2,0.5,0.9,0.9), ax=ax2)

ax2.set_title('Average Degree Percentage acc to Degree_Type and Status', fontsize = 14)

ax2.set_xlabel('Degree_Type', fontsize = 12)

ax2.set_ylabel('Degree Percentage', fontsize = 12)

ax2.legend(title = "status", loc = "center left", bbox_to_anchor = (1, 0.5))
# Analysis of Entrace Test Percentage acc to work_exp, degree_type and Status

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (14, 11))



sns.barplot(x = placement['degree_t'], y = placement['etest_p'], color = (0.4,0.9,0.6,0.9), ax=ax1)

ax1.set_title('Average Entrace Test Percentage acc to Degree_Type', fontsize = 14)

ax1.set_xlabel('Degree_Type', fontsize = 12)

ax1.set_ylabel('Entrance Test Percentage', fontsize = 12)



sns.barplot(x = placement['degree_t'], y = placement['etest_p'], hue = placement['status'], color = (0.4,0.9,0.6,0.9), ax=ax2)

ax2.set_title('Average Entrance Test Percentage acc to Degree_Type and Status', fontsize = 14)

ax2.set_xlabel('Degree_Type', fontsize = 12)

ax2.set_ylabel('Entrance Test Percentage', fontsize = 12)

ax2.legend(title = "status", loc = "center left", bbox_to_anchor = (1, 0.5))



sns.barplot(x = placement['workex'], y = placement['etest_p'], color = (0.8, 0.8, 0.4, 0.9), ax=ax3)

ax3.set_title('Average Entrace Test Percentage acc to Work_exp', fontsize = 14)

ax3.set_xlabel('Work_Experience', fontsize = 12)

ax3.set_ylabel('Entrance Test Percentage', fontsize = 12)



sns.barplot(x = placement['workex'], y = placement['etest_p'], hue = placement['status'], color = (0.8, 0.8, 0.4, 0.9), ax=ax4)

ax4.set_title('Average Entrace Test Percentage acc to Work_exp and Status', fontsize = 14)

ax4.set_xlabel('Work_Experience', fontsize = 12)

ax4.set_ylabel('Entrance Test Percentage', fontsize = 12)

ax4.legend(title = "status", loc = "center left", bbox_to_anchor = (1, 0.5))
# Analysis of MBA_Percentage acc to Specialisation and Status

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 5))



sns.barplot(x = placement['specialisation'], y = placement['mba_p'], color = (0.8,0.4,0.8,0.9), ax=ax1)

ax1.set_title('Average Degree Percentage acc to Degree_Type', fontsize = 14)

ax1.set_xlabel('Specialisation', fontsize = 12)

ax1.set_ylabel('MBA Percentage', fontsize = 12)



sns.barplot(x = placement['specialisation'], y = placement['mba_p'], hue = placement['status'], color = (0.8,0.4,0.8,0.9), ax=ax2)

ax2.set_title('Average MBA Percentage acc to Degree_Type and Status', fontsize = 14)

ax2.set_xlabel('Specialisation', fontsize = 12)

ax2.set_ylabel('MBA Percentage', fontsize = 12)

ax2.legend(title = "status", loc = "center left", bbox_to_anchor = (1, 0.5))
#scatter plots for numeric values

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2, figsize = (14, 10))



sns.scatterplot(x = placement['ssc_p'], y = placement['hsc_p'], hue = placement['status'], ax = ax1)

ax1.set_title('SSC %age vs HSC %age', fontsize = 14)

ax1.set_xlabel('SSC Percentage', fontsize = 12)

ax1.set_ylabel('HSC Percentage', fontsize = 12)

ax1.legend(loc = "center left", bbox_to_anchor = (1, 0.5))



sns.scatterplot(x = placement['mba_p'], y = placement['degree_p'], hue = placement['status'], ax = ax2)

ax2.set_title('MBA %age vs Degree %age', fontsize = 14)

ax2.set_xlabel('MBA Percentage', fontsize = 12)

ax2.set_ylabel('Degree Percentage', fontsize = 12)

ax2.legend(loc = "center left", bbox_to_anchor = (1, 0.5))



sns.scatterplot(x = placement['etest_p'], y = placement['degree_p'], hue = placement['status'], ax = ax3)

ax3.set_title('Entrance %age vs Degree %age', fontsize = 14)

ax3.set_xlabel('Entrance Percentage', fontsize = 12)

ax3.set_ylabel('Degree Percentage', fontsize = 12)

ax3.legend(loc = "center left", bbox_to_anchor = (1, 0.5))



sns.scatterplot(x = placement['etest_p'], y = placement['mba_p'], hue = placement['status'], ax = ax4)

ax4.set_title('MBA %age vs Entrance %age', fontsize = 14)

ax4.set_xlabel('Entrance Percentage', fontsize = 12)

ax4.set_ylabel('MBA Percentage', fontsize = 12)

ax4.legend(loc = "center left", bbox_to_anchor = (1, 0.5))



fig.tight_layout(pad = 2.0)
Numeric_cols = placement[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']]

Numeric_cols.head()
Corr_num_cols = Numeric_cols.corr()

Corr_num_cols
plt.figure(figsize = (8, 6))

plt.title('Heatmap representing Corr b/w diff Numerical Values', fontsize = 14)

labels = ["SSC_Percentage", "HSC_Percentage", "Degree_Percentage", "Entrance_Percentage", "MBA_Percentage"]

ax = sns.heatmap(Corr_num_cols, cmap = "Blues")

ax.set_xticklabels(labels, rotation = 45)

ax.set_yticklabels(labels, rotation = 0)
plt.figure(figsize = (6,5))

sns.barplot(x = placement['gender'], y = placement['salary'])

plt.title('Average Salary for each Gender', fontsize = 14)

plt.xlabel('Gender', fontsize = 12)

plt.ylabel('Salary', fontsize = 12)
# Analysis of Salary with other Categorical Variables

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(nrows = 3, ncols = 2, figsize = (16,18))



sns.barplot(x = placement['hsc_s'], y = placement['salary'], hue = placement['gender'], color = (0.4, 0.6, 0.9, 0.9), ax = ax1)

ax1.set_title('Average Salary acc to HSC_Subject and Gender', fontsize = 14)

ax1.set_xlabel('HSC_Subject', fontsize = 12)

ax1.set_ylabel('Salary', fontsize = 12)

ax1.legend(title = 'Gender', loc = 'upper right')



sns.barplot(x = placement['degree_t'], y = placement['salary'], hue = placement['gender'], color = (0.6, 0.7, 0.3, 0.9),  ax = ax2)

ax2.set_title('Average Salary acc to Degree_Type and Gender', fontsize = 14)

ax2.set_xlabel('Degree_Type', fontsize = 12)

ax2.set_ylabel('Salary', fontsize = 12)

ax2.legend(title = 'Gender', loc = 'upper center')



sns.barplot(x = placement['workex'], y = placement['salary'], hue = placement['gender'], color = (0.2, 0.6, 0.6, 0.9), ax = ax3)

ax3.set_title('Average Salary acc to Work_Exp and Gender', fontsize = 14)

ax3.set_xlabel('Work Experience', fontsize = 12)

ax3.set_ylabel('Salary', fontsize = 12)

ax3.legend(title = 'Gender', loc = 'upper center')



sns.barplot(x = placement['ssc_b'], y = placement['salary'], hue = placement['gender'], color = (0.8, 0.8, 0.4, 0.9), ax = ax4)

ax4.set_title('Average Salary acc to SSC_Board and Gender', fontsize = 14)

ax4.set_xlabel('SSC_Board', fontsize = 12)

ax4.set_ylabel('Salary', fontsize = 12)

ax4.legend(title = 'Gender', loc = 'upper center')



sns.barplot(x = placement['hsc_b'], y = placement['salary'], hue = placement['gender'], color = (0.6, 0.2, 0.8, 0.9), ax = ax5)

ax5.set_title('Average Salary acc to HSC_Board and Gender', fontsize = 14)

ax5.set_xlabel('HSC_Board', fontsize = 12)

ax5.set_ylabel('Salary', fontsize = 12)

ax5.legend(title = 'Gender')



sns.barplot(x = placement['specialisation'], y = placement['salary'], hue = placement['gender'], color = (0.6, 0.2, 0.5, 0.9), ax = ax6)

ax6.set_title('Average Salary acc to Specialisation and Gender', fontsize = 14)

ax6.set_xlabel('Specialisation', fontsize = 12)

ax6.set_ylabel('Salary', fontsize = 12)

ax6.legend(title = 'Gender', loc = 'upper center')
placement.head()
# One-Hot-Encode the HSC_Subject and Degree_Type

hsc_s_ohc = pd.get_dummies(placement['hsc_s'], drop_first=True)

degree_t_ohc = pd.get_dummies(placement['degree_t'], drop_first=True)
# Label Encode the other Categorical Columns

placement['gender_le'] = le.fit_transform(placement['gender'])

placement['ssc_b_le'] = le.fit_transform(placement['ssc_b'])

placement['hsc_b_le'] = le.fit_transform(placement['hsc_b'])

placement['workex_le'] = le.fit_transform(placement['workex'])

placement['specialisation_le'] = le.fit_transform(placement['specialisation'])



placement['status'] = le.fit_transform(placement['status'])     # Our Target Variable
placement = pd.concat([placement, hsc_s_ohc, degree_t_ohc], axis = 1)
placement.head()
feature_names = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p", "gender_le", "ssc_b_le", "hsc_b_le", "workex_le", "specialisation_le", "Commerce", "Science", "Others", "Sci&Tech"]

X = placement[feature_names]

Y = placement['status']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 25)
# I will use the following Classification Models



# 1. Logistic Regression

# 2. KNeighbourClassifier

# 3. SupportVectorMachine

# 4. DecisionTreeClassifier

# 5. RandomForestClassifier

# 6. XGBClassifier

# 7. GradientBoostingClassifier
# Logistic Regression

x_train = ss.fit_transform(x_train)            # using Standard Scaler to Scale the values

x_test = ss.transform(x_test)



log_reg = LogisticRegression()

log_reg.fit(x_train,y_train)            

log_pred = log_reg.predict(x_test)
plot_confusion_matrix(log_reg, x_test, y_test, cmap = 'Blues')

plt.title('Confusion Matrix for Logistic Regression', fontsize = 14)
print("Accuracy for Logistic Regression: ", accuracy_score(log_pred, y_test))
# KneighbourClassifier

KNC = KNeighborsClassifier(n_neighbors=5)

KNC.fit(x_train, y_train)

KNC_pred = KNC.predict(x_test)
plot_confusion_matrix(KNC, x_test, y_test, cmap = 'Blues')

plt.title('Confusion Matrix for KNeighboursClassifier', fontsize = 14)
print("Accuracy for KNeighbourClassifier: ", accuracy_score(KNC_pred, y_test))
# Support Vector Machines

clf_svm = svm.SVC(kernel="linear", C=2)

clf_svm.fit(x_train, y_train)

svm_pred = clf_svm.predict(x_test)
plot_confusion_matrix(clf_svm, x_test, y_test, cmap = 'Blues')

plt.title('Confusion Matrix for Support Vector Machine', fontsize = 14)
print("Accuracy for Support Vector Machine: ", accuracy_score(svm_pred, y_test))
# Support Vector Machines

DTC = DecisionTreeClassifier(min_samples_leaf = 15)

DTC.fit(x_train, y_train)

DTC_pred = DTC.predict(x_test)
plot_confusion_matrix(DTC, x_test, y_test, cmap = 'Greens')

plt.title('Confusion Matrix for DecisionTreeClassifier', fontsize = 14)
print("Accuracy for DecisionTreeClassifier: ", accuracy_score(DTC_pred, y_test))
# RandomForestClassifier

RF = RandomForestClassifier(n_estimators=100, min_samples_leaf = 10)

RF.fit(x_train, y_train)

RF_pred = RF.predict(x_test)
plot_confusion_matrix(RF, x_test, y_test, cmap = 'Greens')

plt.title('Confusion Matrix for RandomForestClassifier', fontsize = 14)
print("Accuracy for RandomForestClassifier: ", accuracy_score(RF_pred, y_test))
# XGBClassifier

xgb = XGBClassifier()

xgb.fit(x_train, y_train)

xgb_pred = xgb.predict(x_test)
plot_confusion_matrix(xgb, x_test, y_test, cmap = 'Reds')

plt.title('Confusion Matrix for XGBClassifier', fontsize = 14)
print("Accuracy for XGBClassifier: ", accuracy_score(xgb_pred, y_test))
# GradientBoostingClassifier

gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

gbk_pred = gbk.predict(x_test)
plot_confusion_matrix(gbk, x_test, y_test, cmap = 'Reds')

plt.title('Confusion Matrix for GradientBoostingClassifier', fontsize = 14)
print("Accuracy for GradientBoostingClassifier: ", accuracy_score(gbk_pred, y_test))
# Now lets see every models accuracy in a single place

print("Accuracy for diff Models:")

print("1. Logistic Regression:        ", accuracy_score(log_pred, y_test))

print("2. KNeighbourClassifier:       ", accuracy_score(KNC_pred, y_test))

print("3. SupportVectorMachine:       ", accuracy_score(svm_pred, y_test))

print("4. DecisionTreeClassifier:     ", accuracy_score(DTC_pred, y_test))

print("5. RandomForestClassifier:     ", accuracy_score(RF_pred, y_test))

print("6. XGBClassifier:              ", accuracy_score(xgb_pred, y_test))

print("7. GradientBoostingClassifier: ", accuracy_score(gbk_pred, y_test))