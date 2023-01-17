import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import roc_auc_score
DATA_DIR = '/kaggle/input/killer-shrimp-invasion/'

RANDOM_STATE = 0



train = pd.read_csv(DATA_DIR + 'train.csv')

test = pd.read_csv(DATA_DIR + 'test.csv')
X_train = train[['Salinity_today', 'Temperature_today', 'Substrate', 'Depth', 'Exposure']]

X_test = test[['Salinity_today', 'Temperature_today', 'Substrate', 'Depth', 'Exposure']]
# Iterative imputer

imputer = IterativeImputer(max_iter = 10, random_state = RANDOM_STATE)

imputer.fit(X_train)

X_train = pd.DataFrame(imputer.transform(X_train), columns = X_train.columns)

X_test = pd.DataFrame(imputer.transform(X_test), columns = X_test.columns)



# Scaling

scaler = StandardScaler()

scaler.fit(X_train)

X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)

X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
def five_fold_cv(model, X_train, Y_train, verbose = True):

    skf = StratifiedKFold(n_splits = 5)

    fold = 1

    scores = []

    

    for train_index, test_index in skf.split(X_train, Y_train):

        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]

        Y_train_fold, Y_test_fold = Y_train.iloc[train_index], Y_train.iloc[test_index]



        model.fit(X_train_fold, Y_train_fold)



        preds = model.predict_proba(X_test_fold)

        preds = [x[1] for x in preds]



        score = roc_auc_score(Y_test_fold, preds)

        scores.append(score)

        if verbose:

            print('Fold', fold, '     ', score)

        fold += 1

    

    avg = np.mean(scores)

    if verbose:

        print()

        print('Average:', avg)

    return avg
avg = five_fold_cv(LogisticRegression(random_state = RANDOM_STATE), X_train, train['Presence'])
temperatures = X_train[['Temperature_today']]

avg = five_fold_cv(LogisticRegression(random_state = RANDOM_STATE), temperatures, train['Presence'])
#temperatures['Temperature 0.5'] = temperatures['Temperature_today'].apply(lambda x: x ** 0.5)

temperatures['Temperature 1'] = temperatures['Temperature_today']

#temperatures['Temperature 1.5'] = temperatures['Temperature_today'].apply(lambda x: x ** 1.5)

temperatures['Temperature 2'] = temperatures['Temperature_today'] ** 2

temperatures['Temperature 3'] = temperatures['Temperature_today'] ** 3

temperatures['Temperature 4'] = temperatures['Temperature_today'] ** 4

temperatures['Temperature 5'] = temperatures['Temperature_today'] ** 5

temperatures['Temperature 6'] = temperatures['Temperature_today'] ** 6

temperatures['Temperature 7'] = temperatures['Temperature_today'] ** 7

temperatures['Temperature 8'] = temperatures['Temperature_today'] ** 8

temperatures['Temperature 9'] = temperatures['Temperature_today'] ** 9

temperatures['Temperature 10'] = temperatures['Temperature_today'] ** 10
dof_temp_scores = []

dof_temp = ['Temperature 1', 'Temperature 2', 'Temperature 3', 'Temperature 4', 'Temperature 5', 'Temperature 6', 'Temperature 7', 'Temperature 8', 'Temperature 9', 'Temperature 10']

for i in range(10):

    dof_temp_scores.append(five_fold_cv(LogisticRegression(random_state = RANDOM_STATE), temperatures[dof_temp[:i + 1]], train['Presence'], verbose = False))

    

plt.plot(range(1, 11), dof_temp_scores)

plt.xlabel('Degrees of Freedom')

plt.ylabel('Cross Validation Score')

plt.title('Cross Validation Scores For Polynomial Temperature Models')

plt.show()
# Get salinity polynomial features

salinities = X_train[['Salinity_today']]

for i in range(1, 11):

    salinities['Salinity ' + str(i)] = salinities['Salinity_today'] ** i



# Perform cross validation on models

dof_sal_scores = []

dof_sal = ['Salinity 1', 'Salinity 2', 'Salinity 3', 'Salinity 4', 'Salinity 5', 'Salinity 6', 'Salinity 7', 'Salinity 8', 'Salinity 9', 'Salinity 10']

for i in range(10):

    dof_sal_scores.append(five_fold_cv(LogisticRegression(random_state = RANDOM_STATE), salinities[dof_sal[:i + 1]], train['Presence'], verbose = False))



# Create graph of cross validation scores

plt.plot(range(1, 11), dof_sal_scores, label = 'Salinity Model')

plt.plot(range(1, 11), dof_temp_scores, label = 'Temperature Model')

plt.xlabel('Degrees of Freedom')

plt.ylabel('Cross Validation Score')

plt.title('Cross Validation Scores For Polynomial Salinity and Temperature Models')

plt.show()
degrees = 6



# Cross validation

five_fold_cv(LogisticRegression(random_state = RANDOM_STATE), temperatures[dof_temp[:degrees]], train['Presence'])



# Feature extraction from the test data (test data has already been scaled)

test_temperatures = pd.DataFrame()

for i in range(1, degrees + 1):

    test_temperatures['Temperature ' + str(i)] = X_test['Temperature_today'] ** i



# Building the actual model

model = LogisticRegression(random_state = RANDOM_STATE)

model.fit(temperatures[dof_temp[:degrees]], train['Presence'])



# View coefficients

print()

print('Coefficients:', model.coef_)

print('Intercept:   ', model.intercept_)
# Make predictions

preds = model.predict_proba(test_temperatures)

preds = [x[1] for x in preds]



# Save preds to file

res = pd.DataFrame()

res['pointid'] = test['pointid']

res['Presence'] = preds

res.to_csv('predictions.csv', index = False)