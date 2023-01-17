import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
diabetes_df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

print(diabetes_df.shape)

diabetes_df.head()
import matplotlib.pyplot as plt

import seaborn as sns



sns.heatmap(diabetes_df.corr(), annot=True, linewidths=1, cmap='YlGnBu')

plt.show()
sns.distplot(diabetes_df['Pregnancies'], kde=False, bins=range(0,17))

plt.show()



pregnancies_group = diabetes_df.groupby(['Pregnancies'], as_index=False)

pregnancies_group_count = pregnancies_group.count()['Outcome']

pregnancies_group_sum = pregnancies_group.sum()['Outcome']

pregnancies_group_percentage = pregnancies_group_sum / pregnancies_group_count * 100



plt.bar(x=range(0,17), height=pregnancies_group_percentage, yerr=pregnancies_group_percentage.std(), tick_label=range(0,17))

plt.title("Number of Pregnancies vs Diabetic Outcome")

plt.xlabel("Number of Pregnancies")

plt.ylabel("% Diabetic Outcome")

plt.show()
def get_bmi_groups(bmi):

    if bmi >= 16 and bmi <18.5:

        return "Underweight"

    elif bmi >= 18.5 and bmi < 25 :

        return "Normal weight"

    elif bmi >= 25 and bmi < 30:

        return "Overweight"

    elif bmi >= 30 and bmi < 35:

        return "Obese Class I (Moderately obese)"

    elif bmi >= 35 and bmi < 40:

        return "Obese Class II (Severely obese)"

    elif bmi >= 40 and bmi < 45:

        return "Obese Class III (Very severely obese)"

    elif bmi >= 45 and bmi < 50:

        return "Obese Class IV (Morbidly Obese)"

    elif bmi >= 50 and bmi < 60:

        return "Obese Class V (Super Obese)"

    elif bmi >= 60:

        return "Obese Class VI (Hyper Obese)"





diabetes_df['bmi_groups'] = diabetes_df['BMI'].apply(get_bmi_groups)



bmi_groups_groupby = diabetes_df.groupby(['bmi_groups'])

bmi_groups_groupby_count = bmi_groups_groupby.count()['Outcome']

bmi_groups_groupby_sum = bmi_groups_groupby.sum()['Outcome']

bmi_groups_groupby_percentage = bmi_groups_groupby_sum / bmi_groups_groupby_count * 100

plt.figure(figsize=(16,4))

plt.bar(x=range(0,9), height=bmi_groups_groupby_percentage, yerr=bmi_groups_groupby_percentage.std(), tick_label=["Normal Weight", "Class 1", "Class 2", 

                                                                        "Class 3", "Class 4", "Class 5", "Class 6", 

                                                                        "Overweight", "Underweight"])

plt.title("BMI class vs Diabetic Outcome")

plt.xlabel("BMI classes")

plt.ylabel("Diabetic Outcome")

plt.show()
def predicted_women_waist(bmi, age):

    c0 = 28.81919

    c1BMI = 2.218007*(bmi)

    age_35 = 0

    if age > 35:

        age_35 = 1

    

    c2IAGE35 = -3.688953 * age_35

    IAGE35 = -0.6570163 * age_35

            

    c3AGEi = 0.125975*(age)

    

    return (c0 + c1BMI + c2IAGE35 + IAGE35 + c3AGEi)



diabetes_df['waist circumference'] = diabetes_df.apply(lambda row: predicted_women_waist(row['BMI'], row['Age']), axis=1)



# Lets apply the same cut off from the previous paper and visualize the results



diabetes_df['waist_cut_off'] = diabetes_df['waist circumference'].apply(lambda size: 1 if size > 88 else 0)



waist_group = diabetes_df.groupby(['waist_cut_off'])

waist_group_count = waist_group.count()['Outcome']

waist_group_sum = waist_group.sum()['Outcome']

waist_group_percentage = waist_group_sum / waist_group_count * 100



plt.bar(x=[0,1], height=waist_group_percentage, yerr=waist_group_percentage.std(), tick_label=['Under Cut Off', 'Over Cut Off'])

plt.title("Waist Cut off vs Diabetic Outcome")

plt.xlabel("Waist Cut off")

plt.ylabel("% Diabetic Outcome")

plt.show()
import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score



X = diabetes_df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 

                 'DiabetesPedigreeFunction', 'Age', 'waist circumference']]

y = diabetes_df['Outcome']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



data_dmatrix = xgb.DMatrix(data=X, label=y)

xgb_clf = xgb.XGBClassifier()

xgb_clf.fit(X_train, y_train)



predictions = xgb_clf.predict(X_test)



f1 = f1_score(y_test, predictions)

print("The F1 score is {}: ".format(f1))
import sklearn.metrics as metrics



fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)

roc_auc = metrics.auc(fpr, tpr)



plt.title('ROC-AUC Curve')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.1f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
feature_importance = pd.DataFrame()

feature_importance['columns'] = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 

                                 'DiabetesPedigreeFunction', 'Age', 'waist circumference']

feature_importance['importances'] = xgb_clf.feature_importances_

feature_importance
X = diabetes_df[['Glucose', 'BMI', 'Age']]

y = diabetes_df['Outcome']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



data_dmatrix = xgb.DMatrix(data=X, label=y)

xgb_clf = xgb.XGBClassifier()

xgb_clf.fit(X_train, y_train)



predictions = xgb_clf.predict(X_test)



f1 = f1_score(y_test, predictions)

print("The F1 score is {}: ".format(f1))



fpr, tpr, threshold = metrics.roc_curve(y_test, predictions)

roc_auc = metrics.auc(fpr, tpr)



plt.title('ROC-AUC Curve')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.1f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()