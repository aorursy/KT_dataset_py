import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

import warnings

%matplotlib inline 

%config InlineBackend.figure_format = 'retina' # Set to retina version

pd.set_option('display.max_columns', None) # Set max columns output

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

print(df.shape)

display(df.head())
df = df.drop(columns=['Over18', 'EmployeeCount', 'StandardHours', 'EmployeeNumber'])
education_map = {1: 'Below College', 2: 'College', 3: 'Bachelor', 4: 'Master', 5: 'Doctor'}

education_satisfaction_map = {1: 'Low', 2:'Medium', 3:'High', 4:'Very High'}

job_involvement_map = {1: 'Low', 2:'Medium', 3:'High', 4:'Very High'}

job_satisfaction_map = {1: 'Low', 2:'Medium', 3:'High', 4:'Very High'}

performance_rating_map = {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'}

relationship_satisfaction_map = {1: 'Low', 2:'Medium', 3:'High', 4:'Very High'}

work_life_balance_map = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}

# Use the pandas apply method to numerically encode our attrition target variable

df['Education'] = df["Education"].apply(lambda x: education_map[x])

df['EnvironmentSatisfaction'] = df["EnvironmentSatisfaction"].apply(lambda x: education_satisfaction_map[x])

df['JobInvolvement'] = df["JobInvolvement"].apply(lambda x: job_involvement_map[x])

df['JobSatisfaction'] = df["JobSatisfaction"].apply(lambda x: job_satisfaction_map[x])

df['PerformanceRating'] = df["PerformanceRating"].apply(lambda x: performance_rating_map[x])

df['RelationshipSatisfaction'] = df["RelationshipSatisfaction"].apply(lambda x: relationship_satisfaction_map[x])

df['WorkLifeBalance'] = df["WorkLifeBalance"].apply(lambda x: work_life_balance_map[x])
display(df.head())
print("Missing Value:", df.isnull().any().any())
colors = ['#66b3ff', '#ff9999']

explode = (0.05,0.05)

plt.figure(figsize=(5, 5))

plt.pie(df['Attrition'].value_counts(), colors = colors, labels=['No', 'Yes'], 

        autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)

plt.legend()

plt.title("Attrition (Target) Distribution")

plt.show()
numerical_list = ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate',

                  'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',

                  'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']



plt.figure(figsize=(10, 10))

for i, column in enumerate(numerical_list, 1):

    plt.subplot(5, 3, i)

    sns.distplot(df[column], bins=20)

plt.tight_layout()

plt.show()
cate_list = ['Attrition', 'BusinessTravel', 'Department', 'Education', 'EducationField', 

             'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 'JobRole',

             'JobSatisfaction', 'MaritalStatus', 'OverTime', 'PerformanceRating', 'RelationshipSatisfaction',

             'StockOptionLevel', 'WorkLifeBalance']



for i in cate_list:

    df[i] = df[i].astype(object)

    

plt.figure(figsize=(20, 20))

gridspec.GridSpec(7, 3)

locator1, locator2 = [0, 0]



for column in cate_list:

    if column == 'JobRole':

        plt.subplot2grid((7, 3), (locator1, locator2), colspan=3, rowspan=1)

        sns.countplot(df[column], palette='Set2')

        locator1 += 1

        locator2 = 0

        continue

    plt.subplot2grid((7, 3), (locator1, locator2))

    sns.countplot(df[column], palette='Set2')

    locator2 += 1

    if locator2 == 3:

        locator1 += 1

        locator2 = 0

        continue

    if locator1 == 7:

        break

        

plt.tight_layout()

plt.show()
plt.figure(figsize=(20, 20))

sns.heatmap(df.corr(), annot=True, cmap="Greys", annot_kws={"size":15})

plt.show()
plt.figure(figsize=(10, 10))

for i, column in enumerate(numerical_list, 1):

    plt.subplot(5, 3, i)

    sns.violinplot(data=df, x=column, y='Attrition')

plt.tight_layout()

plt.show()
plt.figure(figsize=(20, 20))

gridspec.GridSpec(7, 3)

locator1, locator2 = [0, 0]

for column in cate_list:

    if column == 'JobRole':

        plt.subplot2grid((7, 3), (locator1, locator2), colspan=3, rowspan=1)

        sns.countplot(x=column, hue='Attrition', data=df, palette='BrBG')

        locator1 += 1

        locator2 = 0

        continue

    plt.subplot2grid((7, 3), (locator1, locator2))

    sns.countplot(x=column, hue='Attrition', data=df, palette='BrBG')

    locator2 += 1

    if locator2 == 3:

        locator1 += 1

        locator2 = 0

        continue

    if locator1 == 7:

        break

plt.tight_layout()

plt.show()
from sklearn import preprocessing

from IPython.display import Image

# Reload the data

df = pd.read_csv('../input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')

df = df.drop(columns=['Over18', 'EmployeeCount', 'StandardHours', 'EmployeeNumber'])
for cate_features in df.select_dtypes(include='object').columns:

    le = preprocessing.LabelEncoder()

    df[cate_features] = le.fit_transform(df[cate_features])

    print("Origin Classes:", list(le.classes_))
dummies = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']

df = pd.get_dummies(data=df, columns=dummies)

display(df.head())
std = preprocessing.StandardScaler()

scaled = std.fit_transform(df[numerical_list])

scaled = pd.DataFrame(scaled, columns=numerical_list)

for i in numerical_list:

    df[i] = scaled[i]

display(df.head())
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score

from subprocess import call

from IPython.display import Image 

from imblearn.over_sampling import SMOTE
def my_confusion_matrix(test, test_pred):

    cf = pd.DataFrame(confusion_matrix(test, test_pred), 

                      columns=['Predicted NO', 'Predicted Yes'], 

                      index=['True No', 'True Yes'])

    report = pd.DataFrame(classification_report(test, test_pred, target_names=['No', 'Yes'], 

                                                        output_dict=True)).round(2).transpose()

    display(cf)

    display(report)
def plot_roc_curve(model, y, x):

    tree_auc = roc_auc_score(y, model.predict(x))

    fpr, tpr, thresholds = roc_curve(y, model.predict_proba(x)[:,1])

    plt.figure(figsize=(15, 10))

    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Decision Tree ROC curve (area = %0.2f)' % tree_auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.fill_between(fpr, tpr, color='orange', alpha=0.2)

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic')

    plt.legend(loc="lower right")
X = df.drop(columns=['Attrition'])

y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

oversampler = SMOTE(random_state=0)

smote_X_train, smote_y_train = oversampler.fit_sample(X_train, y_train)
colors = ['#66b3ff', '#ff9999']

explode = (0.05,0.05)

plt.figure(figsize=(5, 5))

plt.pie(pd.Series(smote_y_train).value_counts(), colors = colors, labels=['No', 'Yes'], 

        autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)

plt.legend()

plt.title("Oversampled Targets in Training Set")

plt.show()
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier
params = {"criterion": ("gini", "entropy"), 

          "splitter": ("best", "random"), 

          "max_depth": np.arange(1, 20), 

          "min_samples_split": [2, 3, 4], 

          "min_samples_leaf": np.arange(1, 20)}

tree1_grid = GridSearchCV(DecisionTreeClassifier(random_state=0), params, scoring="roc_auc", n_jobs=-1, cv=5)

tree1_grid.fit(X_train, y_train)
print(tree1_grid.best_score_)

print(tree1_grid.best_params_)

print(tree1_grid.best_estimator_)
tree1_clf = DecisionTreeClassifier(random_state=0, **tree1_grid.best_params_)

tree1_clf.fit(X_train, y_train)

tree.export_graphviz(tree1_clf, out_file='/kaggle/working/tree1.dot', special_characters=True, rounded = True, filled= True,

                     feature_names=X.columns, class_names=['Yes', 'No'])

call(['dot', '-T', 'png', '/kaggle/working/tree1.dot', '-o', '/kaggle/working/tree1.png'])

display(Image("/kaggle/working/tree1.png", height=2000, width=1900))
y_test_pred_tree1 = tree1_clf.predict(X_test)

my_confusion_matrix(y_test, y_test_pred_tree1) # Defined before

tree1_auc = roc_auc_score(y_test, y_test_pred_tree1)

print("AUC:", tree1_auc)
IP = pd.DataFrame({"Features": np.array(X.columns), "Importance": tree1_clf.feature_importances_})

IP = IP.sort_values(by=['Importance'], ascending=False)

plt.figure(figsize=(15, 10))

sns.barplot(x='Importance', y='Features', data=IP[:10])

plt.show()
plot_roc_curve(tree1_clf, y_test, X_test)

plt.show()
tree2_grid = GridSearchCV(DecisionTreeClassifier(random_state=0), params, scoring="roc_auc", n_jobs=-1, cv=5)

tree2_grid.fit(smote_X_train, smote_y_train)
print(tree2_grid.best_score_)

print(tree2_grid.best_params_)

print(tree2_grid.best_estimator_)
tree2_clf = DecisionTreeClassifier(random_state=65, **tree2_grid.best_params_)

tree2_clf.fit(smote_X_train, smote_y_train)

tree.export_graphviz(tree2_clf, out_file='/kaggle/working/tree2.dot', special_characters=True, rounded = True, filled= True,

                     feature_names=X.columns, class_names=['Yes', 'No'])

call(['dot', '-T', 'png', 'tree.dot', '-o', '/kaggle/working/tree2.png'])

display(Image("/kaggle/working/tree2.png", height=2000, width=1900))
y_test_pred_tree2 = tree2_clf.predict(X_test)

my_confusion_matrix(y_test, y_test_pred_tree2)

tree2_auc = roc_auc_score(y_test, y_test_pred_tree2)

print("AUC:", tree2_auc)
plot_roc_curve(tree2_clf, y_test, X_test)

plt.show()
IP = pd.DataFrame({"Features": np.array(X.columns), "Importance": tree2_clf.feature_importances_})

IP = IP.sort_values(by=['Importance'], ascending=False)

plt.figure(figsize=(15, 10))

sns.barplot(x='Importance', y='Features', data=IP[:10])

plt.show()
tree1_fpr, tree1_tpr, tree1_thresholds = roc_curve(y_test, tree1_clf.predict_proba(X_test)[:,1])

tree2_fpr, tree2_tpr, tree2_thresholds = roc_curve(y_test, tree2_clf.predict_proba(X_test)[:,1])

plt.figure(figsize=(15, 10))

plt.plot(tree1_fpr, tree1_tpr, color='skyblue', lw=2, label='Decision Tree ROC curve (area = %0.2f)' % tree1_auc)

plt.plot(tree2_fpr, tree2_tpr, color='darkorange', lw=2, label='Decision Tree (with oversampling) ROC curve (area = %0.2f)' % tree2_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.fill_between(tree2_fpr, tree2_tpr, color='darkorange', alpha=0.2)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic (Without Oversampling)')

plt.legend(loc="lower right")

plt.show()
report1 = pd.DataFrame(classification_report(y_test, y_test_pred_tree1, target_names=['No', 'Yes'], output_dict=True)).round(2).transpose()

report2 = pd.DataFrame(classification_report(y_test, y_test_pred_tree2, target_names=['No', 'Yes'], output_dict=True)).round(2).transpose()

evaluation = pd.DataFrame([{'Method': 'Decision Tree (without oversample)', 'F1': report1['f1-score'][1], 'Precision': report1['precision'][1], 'Recall': report1['recall'][1], 'AUC': tree1_auc}, 

                           {'Method': 'Decision Tree (with oversample)', 'F1': report2['f1-score'][1], 'Precision': report2['precision'][1], 'Recall': report2['recall'][1], 'AUC': tree2_auc}])

display(evaluation)