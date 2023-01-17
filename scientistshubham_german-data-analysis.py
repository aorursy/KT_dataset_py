import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
link='../input/german/german_credit_data new.csv'
german=pd.read_csv(link)
german.head(20)

#Changing the name of the dependent variable as class is already defined in python and thereafter dropping it
german['risk']=german['class']
german=german.drop('class', axis=1)
#Describing the dataset
german.describe()
#Changing the dependent categorical variavble to numeric
german['risk']=german['risk'].replace('good', 1)
german['risk']=german['risk'].replace('bad', 0)
#Selecting the categorical columns

cat_data=german.select_dtypes(exclude='int64')
cat_columns=cat_data.columns

#Showing the category wise distribution
for cat in cat_columns:
    print(german[cat].value_counts())
    print()
cat=cat_data['savings_status'].value_counts()
sns.barplot(cat.index, cat.values)
plt.xticks(rotation=90)
pd.crosstab(german.checking_status, german['risk']).plot(kind='bar')
pd.crosstab(german.credit_history, german['risk']).plot(kind='bar')
pd.crosstab(german.purpose, german['risk']).plot(kind='bar')
pd.crosstab(german.savings_status, german['risk']).plot(kind='bar')
pd.crosstab(german.employment, german['risk']).plot(kind='bar')
lb_make=LabelEncoder()
for col in cat_columns:
    if col!='purpose':
        german[col]=lb_make.fit_transform(german[col])

german.head()
german=pd.get_dummies(german, columns=['purpose'])
german.columns
import statsmodels.api as sm

columns=german.columns.drop('risk')

model=sm.GLM.from_formula("risk ~ checking_status+duration+credit_history+credit_amount+savings_status+employment+installment_commitment+personal_status+other_parties+residence_since+property_magnitude+age+other_payment_plans+housing+ existing_credits+job+num_dependents+own_telephone+foreign_worker+purpose_business+purpose_domestic_appliance+purpose_education+purpose_furniture_equipment+purpose_new_car+purpose_other+purpose_radio_tv+purpose_repairs+purpose_retraining+purpose_used_car", family=sm.families.Binomial(), data=german)

result=model.fit()
print(result.summary())
#drop=german.select_dtypes(include='int64')
#trial=drop.copy()

reduced_columns=['checking_status', 'duration', 'credit_amount','savings_status','installment_commitment', 'personal_status', 'purpose_new_car','credit_history','foreign_worker','purpose_radio_tv','purpose_used_car']
y=german['risk']
reduced=german[reduced_columns]

#x=trial.loc[:, trial.columns!='risk']
#x_train, x_test, y_train, y_test=train_test_split(reduced, y,test_size=0.25, random_state=42)
#x_train.head()
reduced
def find_best_parameter(clf,parameter,X_train,y_train):
    best_model = GridSearchCV(clf,parameter).fit(X_train,y_train)
    return best_model.best_estimator_
#Finding the best parameter using grid search CV

from sklearn.model_selection import GridSearchCV

# Logistic Regression 
log_reg_params = {'C': [0.01, 0.1, 1]}
logreg = find_best_parameter(LogisticRegression(max_iter = 10000), log_reg_params,x_train,y_train)

logreg.fit(x_train, y_train)
predict=logreg.predict(x_test)

print(metrics.accuracy_score(predict, y_test))
print(metrics.confusion_matrix(y_test, predict))

print(classification_report(predict, y_test))
fpr, tpr, _=roc_curve(y_test, predict)

roc_auc=auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', label="ROC Curve (area=%0.2f)" %roc_auc)
plt.plot([0,1], [0,1], color='blue', linestyle='--')
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.xlabel('False Poistive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC AUC Curve')
plt.legend(loc='lower right')
from sklearn.tree import DecisionTreeClassifier

tree=DecisionTreeClassifier(criterion="entropy", max_depth=5)

tree.fit(x_train, y_train)

pred=tree.predict(x_test)

print(metrics.accuracy_score(y_test, pred))

print(metrics.confusion_matrix(y_test, pred))

print(classification_report(predict, y_test))
fpr, tpr, _=roc_curve(y_test, pred)

roc_auc=auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', label="ROC Curve (area=%0.2f)" %roc_auc)
plt.plot([0,1], [0,1], color='blue', linestyle='--')
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.xlabel('False Poistive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC AUC Curve')
plt.legend(loc='lower right')