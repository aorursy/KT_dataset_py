

import pandas as pd

import numpy as np

import pprint as pp

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=FutureWarning)
loan = pd.read_csv("../input/loan_predict_train.csv")
loan.shape
loan.info()
loan.describe()
loan.isnull().sum()
loan.Loan_Status.value_counts()
fig, ax = plt.subplots(figsize=(8, 8))

plt.style.use('seaborn-whitegrid')

sns.set(palette="muted")

ax.set_ylim(0,500)

ax = sns.boxplot(x="Loan_Status", y="LoanAmount", data=loan)
fig, ax = plt.subplots(figsize=(8, 10))

ax.set_ylim(0,20000)

ax = sns.boxplot(x="Loan_Status", y="ApplicantIncome", data=loan)
fig, ax = plt.subplots(figsize=(8, 10))

ax.set_ylim(0,20000)

ax = sns.boxplot(x="Loan_Status", y="CoapplicantIncome", data=loan)
plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(15,20))

fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.3)

rows = 3

cols = 3

categorical_col = ['Gender', 'Married', 'Education', 'Property_Area', 'Dependents', 'Self_Employed']

for i, column in enumerate(categorical_col):

    ax = fig.add_subplot(rows, cols, i + 1)

    ax.set(xticks=[])

    ax = pd.crosstab(loan[categorical_col[i]], loan.Loan_Status, normalize='index').plot.bar(ax=ax)
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111)

pd.crosstab([loan.Married, loan.Gender, loan.Dependents], loan.Loan_Status, normalize='index').plot.barh(ax=ax);
loan.loc[(loan.Gender == 'Female') & (loan.Married == 'No') & (loan.Dependents == '3+')]
def do_preprocess(data):

    table = data.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)

    def fage(x):

        return table.loc[x['Self_Employed'],x['Education']]

    # Replace missing values

    data['Self_Employed'].fillna('No', inplace=True)

    data['LoanAmount'].fillna(data[data['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)

    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)

    data['Married'].fillna(data['Married'].mode()[0], inplace=True)

    data['Dependents'].fillna(data['Dependents'].mode()[0], inplace=True)

    data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0], inplace=True)

    data['Credit_History'].fillna(data['Credit_History'].mode()[0], inplace=True)





do_preprocess(loan)
def add_new_features(data):

    data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']

    data['IncomeByLoanAmount'] = data['TotalIncome'] / data['LoanAmount']

    data['AplIncomeByLoanAmount'] = data['ApplicantIncome'] / data['LoanAmount']

   

add_new_features(loan)
def doOneHotEncoding(data, cols):

    for var in cols:

        one_hot = pd.get_dummies(data[var], prefix = var)

        # Drop column B as it is now encoded

        data = data.drop(var,axis = 1)

        # Join the encoded data

        data = data.join(one_hot)

    return data



loan = doOneHotEncoding(loan, ['Gender', 'Married','Dependents','Education','Self_Employed','Property_Area'])

loan.Loan_Status = loan.Loan_Status.map(dict(Y=1,N=0))
loan.Loan_Status.value_counts()
outcome_var = "Loan_Status"

#exclude baseline categorical variables

predictor_var = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',

       'Loan_Amount_Term', 'Credit_History', #'TotalIncome',

       'IncomeByLoanAmount', 'AplIncomeByLoanAmount', #'EMI',

       'Gender_Male', 'Married_No', 'Dependents_0', 'Dependents_1', 'Dependents_3+',

       'Education_Not Graduate', 'Self_Employed_Yes',

       'Property_Area_Semiurban', 'Property_Area_Urban']



logit = sm.Logit(loan[outcome_var], loan[predictor_var])

result = logit.fit_regularized()
result.summary2()
pt = result.pred_table()

pt
print ("Accuracy : %s" % "{0:.3%}".format((pt[0,0]+pt[1,1])/pt.sum()))
predictor_var = ['Loan_Amount_Term', 'Credit_History', 'Property_Area_Semiurban',

                 'Married_No', 'Dependents_1', 'Education_Not Graduate']
logit = sm.Logit(loan[outcome_var], loan[predictor_var])

result = logit.fit_regularized()
result.summary2()
pt = result.pred_table()

pt
print ("Accuracy : %s" % "{0:.3%}".format((pt[0,0]+pt[1,1])/pt.sum()))
from sklearn.linear_model import LogisticRegression

model_logistic = LogisticRegression(random_state=42)
def classify_and_report_metrics(model, data, predictors, outcome):

    X_train, X_test, y_train, y_test = train_test_split(data[predictors], data[outcome], random_state=42, stratify=loan[outcome_var])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print('Accuracy on test set: %s' % '{0:.3%}'.format(model.score(X_test, y_test)))

    cm = confusion_matrix(y_test, y_pred)

    print(cm)

    print(classification_report(y_test, y_pred))
##adding all non-base levels of categorical variables that were found significant

predictor_var += ['Dependents_0', 'Dependents_3+', 'Property_Area_Urban']

predictor_var_logistic = predictor_var
classify_and_report_metrics(model_logistic, loan, predictor_var, outcome_var)
def fit_and_validate(model, data, predictors, outcome):

    #Fit the model:

    model.fit(data[predictors],data[outcome])

    #Make predictions on training set:

    predictions = model.predict(data[predictors])

    #Print accuracy

    accuracy = metrics.accuracy_score(predictions,data[outcome])

    print ("Accuracy : %s" % "{0:.3%}".format(accuracy))



    cm = confusion_matrix(data[outcome], predictions)

    print(cm)

    print(classification_report(data[outcome], predictions))    

    

    #Perform k-fold cross-validation with 5 folds

    kf = KFold(n_splits=5)

    error = []

    for train, test in kf.split(data):

        # Filter training data

        train_predictors = (data[predictors].iloc[train,:])



        # The target we're using to train the algorithm.

        train_target = data[outcome].iloc[train]



        # Training the algorithm using the predictors and target.

        model.fit(train_predictors, train_target)



        #Record error from each cross-validation run

        error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))

    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

    if (isinstance(model, (RandomForestClassifier))):

            #Create a series with feature importances:

            featimp = pd.Series(model.feature_importances_, index=predictors).sort_values(ascending=False)

            print (featimp)

            if (isinstance(model, RandomForestClassifier)):

                if model.get_params()['oob_score'] == True:

                    print('OOB Score %f' % (1 - model.oob_score_))

                else:

                    print('OOB Score False')
fit_and_validate(model_logistic, loan, predictor_var, outcome_var)
predictor_var = [

 'Credit_History', 'IncomeByLoanAmount', 'AplIncomeByLoanAmount',

    'LoanAmount', 'Loan_Amount_Term',

    'Property_Area_Semiurban', 'ApplicantIncome','Married_No','Dependents_1',

    'Education_Not Graduate'

]
model_rf = RandomForestClassifier(random_state=42, n_estimators=200, bootstrap= True, oob_score=True)
classify_and_report_metrics(model_rf, loan, predictor_var, outcome_var)
fit_and_validate(model_rf, loan, predictor_var, outcome_var)
predictor_var = [

 'Credit_History', 'IncomeByLoanAmount', 'AplIncomeByLoanAmount',

    'LoanAmount','Property_Area_Semiurban', 'ApplicantIncome'

]

predictor_var_rf = predictor_var
import pprint as pp

from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 400, num = 2)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

pp.pprint(random_grid)

# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation,

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(loan[predictor_var], loan[outcome_var])

rf_random.best_params_
model_rf = RandomForestClassifier(random_state=42, n_estimators = rf_random.best_params_['n_estimators'], 

                                  min_samples_split = rf_random.best_params_['min_samples_split'], 

                                  min_samples_leaf = rf_random.best_params_['min_samples_leaf'], 

                                  max_features = rf_random.best_params_['max_features'], 

                                  max_depth = rf_random.best_params_['max_depth'],

                                  bootstrap = rf_random.best_params_['bootstrap'],

                                  oob_score = rf_random.best_params_['bootstrap'])
fit_and_validate(model_rf, loan, predictor_var, outcome_var)
print(rf_random.best_params_['n_estimators']), print(rf_random.best_params_['max_features']), 

print(rf_random.best_params_['max_depth']), print(rf_random.best_params_['min_samples_split']),

print(rf_random.best_params_['min_samples_leaf']), print(rf_random.best_params_['bootstrap'])
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

# logit_roc_auc = roc_auc_score(y_test, logistic_model.predict(X_test))

# fpr, tpr, thresholds = roc_curve(y_test, logistic_model.predict_proba(X_test)[:,1])

X_train, X_test, y_train, y_test = train_test_split(loan[predictor_var_logistic], loan[outcome_var], random_state=42, stratify=loan[outcome_var])

model_logistic.fit(X_train, y_train)

logit_roc_auc = roc_auc_score(y_test, model_logistic.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, model_logistic.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')



X_train, X_test, y_train, y_test = train_test_split(loan[predictor_var_rf], loan[outcome_var], random_state=42, stratify=loan[outcome_var])

model_rf.fit(X_train, y_train)

logit_roc_auc = roc_auc_score(y_test, model_rf.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, model_rf.predict_proba(X_test)[:,1])

# plt.figure()

plt.plot(fpr, tpr, label='Random Forest (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')



plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()