import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import RepeatedKFold

from sklearn.linear_model import LogisticRegression
for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

data_train = pd.read_csv("/kaggle/input/titanic/train.csv")

data_test = pd.read_csv("/kaggle/input/titanic/test.csv")



# data_train.head()


def quant_sex(value):

    ''' Change variable Sex from qualitative to quantitative values'''

    if value == 'female':

        return 1

    else:

        return 0

    

variables = ['Sex_bin', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked_S', 'Embarked_C', 'Cabine_nula', 'Nome_contem_Miss', 'Nome_contem_Mrs', 'Nome_contem_Master', 'Nome_contem_Col', 'Nome_contem_Major', 'Nome_contem_Mr']



#

# Treino

#

data_train['Embarked_S'] = (data_train['Embarked'] == 'S').astype(int)

data_train['Embarked_C'] = (data_train['Embarked'] == 'C').astype(int)

# data_train['Embarked_Q'] = (train['Embarked'] == 'Q').astype(int)



data_train['Cabine_nula'] = data_train['Cabin'].isnull().astype(int)



data_train['Nome_contem_Miss'] = data_train['Name'].str.contains("Miss").astype(int)

data_train['Nome_contem_Mrs'] = data_train['Name'].str.contains("Mrs").astype(int)



data_train['Nome_contem_Master'] = data_train['Name'].str.contains("Master").astype(int)

data_train['Nome_contem_Col'] = data_train['Name'].str.contains("Col").astype(int)

data_train['Nome_contem_Major'] = data_train['Name'].str.contains("Major").astype(int)

data_train['Nome_contem_Mr'] = data_train['Name'].str.contains("Mr").astype(int)



data_train['Sex_bin'] = data_train['Sex'].map(quant_sex)

sub_data_train = data_train[variables].fillna(-1)

survived_train = data_train['Survived']



#

# Teste

#

data_test['Embarked_S'] = (data_test['Embarked'] == 'S').astype(int)

data_test['Embarked_C'] = (data_test['Embarked'] == 'C').astype(int)

# data_train['Embarked_Q'] = (train['Embarked'] == 'Q').astype(int)



data_test['Cabine_nula'] = data_test['Cabin'].isnull().astype(int)



data_test['Nome_contem_Miss'] = data_test['Name'].str.contains("Miss").astype(int)

data_test['Nome_contem_Mrs'] = data_test['Name'].str.contains("Mrs").astype(int)



data_test['Nome_contem_Master'] = data_test['Name'].str.contains("Master").astype(int)

data_test['Nome_contem_Col'] = data_test['Name'].str.contains("Col").astype(int)

data_test['Nome_contem_Major'] = data_test['Name'].str.contains("Major").astype(int)

data_test['Nome_contem_Mr'] = data_test['Name'].str.contains("Mr").astype(int)



data_test['Sex_bin'] = data_test['Sex'].map(quant_sex)

sub_data_test = data_test[variables].fillna(-1)
baseline = (data_test['Sex'] == 'female').astype(int)

baseline.index = data_test['PassengerId']

baseline.name = 'Survived'

baseline.to_csv('gender_submission_baseline.csv', header = True)



# Imprime os 10 primeiros registros do arquivo

# !head -n10 gender_submission.csv
model = RandomForestClassifier(n_estimators = 100, n_jobs = -1, random_state = 0)



model.fit(sub_data_train, survived_train) 



rf_survived = model.predict(sub_data_test)



# Imprime variavel

# rf_survived
output = pd.Series(rf_survived, index = data_test['PassengerId'], name = 'Survived')

output.to_csv("random_forest.csv", header = True)



# !head -n10 random_forest.csv
model = LogisticRegression()



model.fit(sub_data_train, survived_train) 



lr_survived = model.predict(sub_data_test)



# Imprime variavel

# lr_survived
output = pd.Series(lr_survived, index = data_test['PassengerId'], name = 'Survived')

output.to_csv("logistic_regression.csv", header = True)



# !head -n10 logistic_regression.csv
sub_train, sub_valid, sub_surv_train, sub_surv_valid  = train_test_split(sub_data_train, survived_train, test_size = 0.5)



# Imprime dimensões das tuplas

# sub_train.shape, sub_valid.shape, sub_surv_train.shape, sub_surv_valid.shape



# Baseline

baseline = (sub_valid['Sex_bin'] == 1).astype(np.int64)

acc_baseline = np.mean(sub_surv_valid == baseline)



# Random Forest

model_RF = RandomForestClassifier(n_estimators = 100, n_jobs = -1, random_state = 0)

model_RF.fit(sub_train, sub_surv_train) 

survived_predicted_valid_RF = model_RF.predict(sub_valid)

acc_rf = np.mean(sub_surv_valid == survived_predicted_valid_RF)



# Logistic Regression

model_LR = LogisticRegression()

model_LR.fit(sub_train, sub_surv_train) 

survived_predicted_valid_LR = model_LR.predict(sub_valid)

acc_lr = np.mean(sub_surv_valid == survived_predicted_valid_LR)
def cross_validation(kfold, data, survived, model):

    results = []

    valid_lines = []

    survived_predicted = []

    for train_lines, valid_lines in kfold.split(data):

        # print("Train:", train_lines.shape[0])

        # print("Valid:", valid_lines.shape[0])



        sub_train, sub_valid = data.iloc[train_lines], data.iloc[valid_lines] 

        sub_surv_train, sub_surv_valid =  survived.iloc[train_lines], survived.iloc[valid_lines]



        model.fit(sub_train, sub_surv_train) 

        survived_predicted = model.predict(sub_valid)



        acc = np.mean(sub_surv_valid == survived_predicted)

        results.append(acc)

        # print("Acc:", acc)

        # print()

    return results, valid_lines, survived_predicted
kf = RepeatedKFold(n_splits = 2, n_repeats = 10, random_state = 10)



# Random Forest

model_RF = RandomForestClassifier(n_estimators = 100, n_jobs = -1, random_state = 0)

results_kf_rf, valid_lines_rf, survived_predicted_rf = cross_validation(kf, sub_data_train, survived_train, model_RF)

acc_kf_rf = np.mean(results_kf_rf)



# Logistic Regression

model_LR = LogisticRegression()

results_kf_lr, valid_lines_lr, survived_predicted_lr = cross_validation(kf, sub_data_train, survived_train, model_LR)

acc_kf_lr = np.mean(results_kf_lr)
print("Baseline\t\t", acc_baseline, "\nRandom Forest\t\t", acc_rf, "\nLogistic Regression\t", acc_lr, "\nKFold RF\t\t", acc_kf_rf, "\nKFold LR\t\t", acc_kf_lr)
%matplotlib inline

%pylab inline

pylab.hist(results_kf_rf), pylab.hist(results_kf_lr, alpha=0.8)
data_check = data_train.iloc[valid_lines_rf].copy()

data_check["Predicted"] = survived_predicted_rf

# data_check.head()



errors = data_check[data_check["Survived"] != data_check["Predicted"]]

errors = errors[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Sex_bin', 'Predicted', 'Survived']]

# errors.head()



women = errors[errors["Sex"] == "female"]

men = errors[errors["Sex"] == "male"]



women.sort_values("Survived")
men.sort_values("Survived")
predicted = lr_survived

file_submit = pd.Series(predicted, index = data_test["PassengerId"], name = "Survived" )

file_submit.to_csv("titanic_prediction.csv", header = True)



# head -n10 titanic_prediction.csv