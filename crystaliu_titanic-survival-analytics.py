import pandas as pd

import numpy as np

import seaborn as sns
td = pd.read_csv('../input/train.csv')
td.head(5)
td.info()
nuvar = ['Age','SibSp','Parch','Fare']
td['Age'].value_counts()
age = td['Age'].dropna() # do it since seaborn can't create plot with NaN in the variable
sns.distplot(age)
age_1 = td['Age'].interpolate(method='linear')

age_2 = td['Age'].interpolate(method='pchip')

age_3 = td['Age'].interpolate(method='cubic')
#sns.distplot(age_1)

sns.distplot(age_2)

#sns.distplot(age_2)
td['Age_all'] = td['Age']. interpolate(method='pchip')
td['SibSp'].value_counts()
sns.distplot(td['SibSp'])
td['SibSp_regrouped'] =  td['SibSp'].apply(lambda x: '0' if x == 0 \

                                                          else '1' if x == 1 \

                                                          else '2 or above')
td['SibSp_regrouped'].value_counts()
td['SibSp_regrouped'] = td['SibSp_regrouped'].astype('category')
td['Parch'].value_counts()
sns.distplot(td['Parch'])
td['Parch_regrouped'] =  td['Parch'].apply(lambda x: '0' if x == 0 \

                                                          else '1' if x == 1 \

                                                          else '2' if x == 2 \

                                                          else '3 or above')
td['Parch_regrouped'].value_counts()
td['Parch_regrouped'] = td['Parch_regrouped'].astype('category')
td['Fare'].value_counts()
sns.distplot(td['Fare'])
nuvar
nuvar[0] = 'Age_all'   # replace age with age_all because seaborn.pairplot() doesn't work well with NaN
nuvar
nuvar_df = td[nuvar]

sns.pairplot(nuvar_df)
corr = td.corr()

sns.heatmap(corr)
list(td)
catvar = ['Survived','Pclass','Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
td['Survived'].value_counts()
td['Pclass'].value_counts()
td['Pclass'] = td['Pclass'].astype('category') # Convert Pclass from int type to category
td['Pclass'].dtype
sns.barplot(x="Pclass", y="Survived", data=td);
td['Sex'].value_counts()
sns.barplot(x='Sex', y='Survived', data=td)
td['Ticket'].value_counts()
td['Cabin'].value_counts()
cabinLetter = ['A','B','C','D','E','F','G']

num = []

for i in cabinLetter:

    num.append(len(td[td['Cabin'].str.contains(i, na = False)]))

num
td['Embarked'].value_counts()
td['Embarked'] = td['Embarked'].fillna('S')  # Use the mode to fill the missing values
sns.barplot(x='Embarked', y='Survived', data=td)
Y = td['Survived'].values
list(td)
feature_list = [

# 'PassengerId',

# 'Survived',

 'Pclass',

# 'Name',

 'Sex',

# 'Age',

 'SibSp',

 'Parch',

# 'Ticket',

 'Fare',

# 'Cabin',

 'Embarked',

 'Age_all',

#'SibSp_regrouped',

#'Parch_regrouped'

]
td[feature_list].info()   # no missing values
X = pd.get_dummies(td[feature_list])
X
feature_list_dummy = list(X)
feature_list_dummy
X = X[[

 'SibSp',

 'Parch',

 'Fare',

 'Age_all',

 'Pclass_1',

 'Pclass_2',

# 'Pclass_3',

 'Sex_female',

# 'Sex_male',

 'Embarked_C',

 'Embarked_Q',

 #'Embarked_S'

      ]]
feature_list_dummy = list(X)
feature_list_dummy
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  
X_scaled.shape
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train, Y_train)
train_predict = lr.predict(X_train)
test_predict = lr.predict(X_test)
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
def print_results(y_true, y_pred):

    print("Accuracy of the Logistic Regression is: {}".format(accuracy_score(y_true, y_pred)))

    print("Precision of the Logistic Regression is: {}".format(precision_score(y_true, y_pred)))

    print("Recall of the Logistic Regression is: {}".format(recall_score(y_true, y_pred)))

    print("f1-score of the Logistic Regression is: {}".format(f1_score(y_true, y_pred)))

    print("Area Under Curve (AUC) of the Logistic Regression is: {}".format(roc_auc_score(y_true, y_pred)))
print("Training set scores:")

print_results(Y_train, train_predict)
print("Testing set scores:")

print_results(Y_test, test_predict)
from sklearn.metrics import roc_curve

#reference: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve
Y_test_pred_proba = lr.predict_proba(X_test)[:,1]
FalsePositiveRate, TruePositiveRate, thresholds = roc_curve(Y_test, Y_test_pred_proba)
import matplotlib.pyplot as plt

% matplotlib inline

plt.style.use("ggplot")
# plot TPR against FPR

plt.plot(FalsePositiveRate, TruePositiveRate, color='red')



# plot 45 degree line

xx = np.linspace(0, 1.0, 20)

plt.plot(xx, xx, color='blue')



plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC")



plt.show()
#lr.coef_.flatten()  

#reference: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.flatten.html
#list(zip(feature_list_dummy, lr.coef_))
df_coeffs = pd.DataFrame(list(zip(feature_list_dummy, lr.coef_.flatten()))).sort_values(by=[1], ascending=False)

df_coeffs.columns = ['feature', 'coeff']

df_coeffs
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000,      #Number of trees

                            max_features=5)        #Number of features considered
rf.fit(X_train, Y_train)
rf_train_predict = rf.predict(X_train)
rf_test_predict = rf.predict(X_test)
def print_results(y_true, y_pred):

    print("Accuracy of the Logistic Regression is: {}".format(accuracy_score(y_true, y_pred)))

    print("Precision of the Logistic Regression is: {}".format(precision_score(y_true, y_pred)))

    print("Recall of the Logistic Regression is: {}".format(recall_score(y_true, y_pred)))

    print("f1-score of the Logistic Regression is: {}".format(f1_score(y_true, y_pred)))

    print("Area Under Curve (AUC) of the Logistic Regression is: {}".format(roc_auc_score(y_true, y_pred)))
print("Training set scores:")

print_results(Y_train, rf_train_predict)
print("Testing set scores:")

print_results(Y_test, rf_test_predict)
from sklearn.metrics import confusion_matrix
Y_test_pred_proba = rf.predict_proba(X_test)[:,1]
confusion_matrix(Y_test, rf_test_predict)
import matplotlib.pyplot as plt



# import some data to play with

#iris = datasets.load_iris()

#X = iris.data

#y = iris.target



# Split the data into a training set and a test set

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



# Run classifier

#classifier = svm.SVC(kernel='linear')

#y_pred = classifier.fit(X_train, y_train).predict(X_test)



# Compute confusion matrix

cm = confusion_matrix(Y_test, rf_test_predict)



print(cm)



# Show confusion matrix in a separate window

plt.matshow(cm)

plt.title('Confusion matrix')

plt.colorbar()

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()



#reference: http://scikit-learn.org/0.15/auto_examples/plot_confusion_matrix.html
df_coeffs_rf = pd.DataFrame(list(zip(feature_list_dummy, rf.feature_importances_.flatten()))).sort_values(by=[1], ascending=False)

df_coeffs_rf.columns = ['feature', 'coeff']

df_coeffs_rf
testd = pd.read_csv('../input/test.csv')
testd.info()
list(testd)
testd = testd[[

 'PassengerId',

 'Pclass',

# 'Name',

 'Sex',

 'Age',

 'SibSp',

 'Parch',

# 'Ticket',

 'Fare',

# 'Cabin',

 'Embarked'

]]
testd['Age'] = testd['Age'].interpolate(method='pchip')
testd['Fare'] = testd['Fare'].fillna(testd['Fare'].median())
testd['Pclass'] = testd['Pclass'].astype('category')
testd.info()
testd = pd.get_dummies(testd)
list(testd)
testd_final = testd[[

# 'PassengerId',

 'Age',

 'SibSp',

 'Parch',

 'Fare',

 'Pclass_1',

 'Pclass_2',

# 'Pclass_3',

 'Sex_female',

# 'Sex_male',

 'Embarked_C',

 'Embarked_Q',

# 'Embarked_S'

]]
rf_test_predict_submit = rf.predict(testd_final)
#rf_test_predict_submit
my_submission = pd.DataFrame({'PassengerId': testd.PassengerId, 'Survived': rf_test_predict_submit})
#my_submission
my_submission.to_csv('submission.csv', index=False)