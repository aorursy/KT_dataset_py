#import every library needed for completing task



import operator



import pandas as pd

import numpy as np

from scipy import stats



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_predict

from sklearn import metrics
#function needed to make for completing the task easily



def regression_stats(x, y):

    mask = ~np.isnan(x) & ~np.isnan(y)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x[mask], y[mask])

    pearsonr = stats.pearsonr(x[mask], y[mask])

    result = {

        'slope': slope, 'intercept': intercept,

        'r_value': r_value, 'p_value': p_value,

        'std_err': std_err, 'r_squared': r_value ** 2,

        'pearsonr': pearsonr[0]

    }

    return result





def convert_to_ordianl_value(df, col):

    if not isinstance(col, str):

        for i in col:

            df[i] = df[i].astype('category').cat.codes

        else:

            return "task is completed"

    df[col] = df[col].astype('category').cat.codes

    return 'task is comleted"'
#acquire train and test dataset from folder
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

#combine train and test dataset to make it easy to convert train and test dataset together
combine = [train_df, test_df]

train_df.shape, test_df.shape
#look over what kind of columns there are in the dataset 

#or you can check out the data description text files for more detail information



train_df.columns
#find out what columns' data type is number



regression_target = train_df.select_dtypes(include=[np.number])

regression_target = regression_target.drop(['Id', 'SalePrice'], axis=1)

regression_target_columns = regression_target.columns

regression_target_columns
# Find out the most influential features based on regression graph and p-value of two features.



p_value = {}

for col in regression_target_columns:

    stats_value = regression_stats(train_df[col], train_df['SalePrice'])

    p_value.update({col: stats_value.get('p_value')})



p_value = sorted(p_value.items(), key=operator.itemgetter(1))
# Check out the result I found based on p-value one more by looking at the graph



for col, _ in p_value[:5]:

    sns.jointplot(train_df[col], train_df['SalePrice'], kind='reg')
deleted_col = [key for key, value in p_value if value > 0.05]

deleted_col
#find out what numeric values' columns has Nan values significantly



pd.set_option('display.max_columns', 81)

train_df.describe()
#find out what string values' columns has Nan values significantly



train_df.describe(include=['O'])
#These columns has so many Nan values. I decided to delete these columns cause I thought that these are not effective on prediction.

not_enough_columns = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']



for dataset in combine:

    #delete columns which are not related to the result

    dataset.drop(deleted_col, axis=1, inplace=True)

    dataset.drop(not_enough_columns, axis=1, inplace=True)
#Before modeling, need to fill out the Nan values



freq_MasVnrType = train_df['MasVnrType'].dropna().mode()[0]

freq_BsmtQual = train_df['BsmtQual'].dropna().mode()[0]

freq_GarageYrBlt = train_df['GarageYrBlt'].dropna().median()

freq_MasVnrArea = train_df.MasVnrArea.dropna().median()

freq_BsmtFullBath = train_df.BsmtFullBath.mode()[0]

freq_GarageArea = train_df.GarageArea.median()

freq_GarageCars = train_df.GarageCars.mode()[0]

freq_BsmtFinSF1 = train_df.BsmtFinSF1.median()

freq_BsmtUnfSF = train_df.BsmtUnfSF.median()

freq_TotalBsmtSF = train_df.TotalBsmtSF.median()

freq_BsmtExposure = train_df.BsmtExposure.mode()[0]

freq_BsmtFinType1 = train_df.BsmtFinType1.mode()[0]

freq_Electrical = train_df.Electrical.mode()[0]

freq_GarageType = train_df.GarageType.mode()[0]

freq_GarageFinish = train_df.GarageFinish.mode()[0]

freq_GarageQual = train_df.GarageQual.mode()[0]

freq_GarageCond = train_df.GarageCond.mode()[0]

freq_BsmtCond = train_df.BsmtCond.mode()[0]

freq_BsmtFinType2 = train_df.BsmtFinType2.mode()[0]

freq_MSZoning = train_df.MSZoning.mode()[0]

freq_Utilities = train_df.Utilities.mode()[0]

freq_Exterior1st = train_df.Exterior1st.mode()[0]

freq_Exterior2nd = train_df.Exterior2nd.mode()[0]

freq_KitchenQual = train_df.KitchenQual.mode()[0]

freq_Functional = train_df.Functional.mode()[0]

freq_SaleType = train_df.Functional.mode()[0]

freq_LotFrontage = train_df.LotFrontage.median()



for dataset in combine:

    dataset['MasVnrType'] = dataset['MasVnrType'].fillna(freq_MasVnrType)

    dataset['BsmtQual'] = dataset['BsmtQual'].fillna(freq_BsmtQual)

    dataset['BsmtExposure'] = dataset['BsmtExposure'].fillna(freq_BsmtExposure)

    dataset['BsmtFinType1'] = dataset['BsmtFinType1'].fillna(freq_BsmtFinType1)

    dataset['Electrical'] = dataset['Electrical'].fillna(freq_Electrical)

    dataset['GarageType'] = dataset['GarageType'].fillna(freq_GarageType)

    dataset['GarageFinish'] = dataset['GarageFinish'].fillna(freq_GarageFinish)

    dataset['GarageYrBlt'] = dataset['GarageYrBlt'].fillna(freq_GarageYrBlt)

    dataset['GarageQual'] = dataset['GarageQual'].fillna(freq_GarageQual)

    dataset['GarageCond'] = dataset['GarageCond'].fillna(freq_GarageCond)

    dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(freq_MasVnrArea)

    dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(freq_BsmtFullBath)

    dataset['GarageArea'] = dataset['GarageArea'].fillna(freq_GarageArea)

    dataset['GarageCars'] = dataset['GarageCars'].fillna(freq_GarageCars)

    dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(freq_BsmtFinSF1)

    dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(freq_BsmtUnfSF)

    dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(freq_TotalBsmtSF)

    dataset['LotFrontage'] = dataset['LotFrontage'].fillna(freq_LotFrontage)

    dataset['BsmtCond'] = dataset['BsmtCond'].fillna(freq_BsmtCond)

    dataset['BsmtFinType2'] = dataset['BsmtFinType2'].fillna(freq_BsmtFinType2)

    dataset['MSZoning'] = dataset['MSZoning'].fillna(freq_MSZoning)

    dataset['Utilities'] = dataset['Utilities'].fillna(freq_Utilities)

    dataset['Exterior1st'] = dataset['Exterior1st'].fillna(freq_Exterior1st)

    dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(freq_Exterior2nd)

    dataset['KitchenQual'] = dataset['KitchenQual'].fillna(freq_KitchenQual)

    dataset['Functional'] = dataset['Functional'].fillna(freq_Functional)

    dataset['SaleType'] = dataset['SaleType'].fillna(freq_SaleType)
train_df.select_dtypes(exclude=[np.number]).columns.values
#convert the categorical values to orinal values for training the model later



convert_target_columns = train_df.select_dtypes(exclude=[np.number]).columns



for dataset in combine:

    convert_to_ordianl_value(dataset, convert_target_columns)
train_df.head()
#split the data into train and test dataset to train model and predict



X_train = train_df.drop(['Id', 'SalePrice'], axis=1)

Y_train = train_df["SalePrice"]

X_test  = test_df.drop("Id", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression accuracy



tuned_parameters = [{'C': [1, 10, 100, 1000]}]



logreg = LogisticRegression()



clf = GridSearchCV(logreg, tuned_parameters, cv=2)

clf.fit(X_train, Y_train)



print("Best parameters set found on development set:")

print()

print(clf.best_params_)

logreg_pred = clf.predict(X_test)

acc_logreg = round(clf.score(X_train, Y_train) * 100, 2)

print()

print(acc_logreg)
# Support Vector Machines accuracy



tuned_parameters = [

    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},

    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}

]



svc = SVC()



clf = GridSearchCV(svc, tuned_parameters, cv=2)

clf.fit(X_train, Y_train)



print("Best parameters set found on development set:")

print()

print(clf.best_params_)

svc_pred = clf.predict(X_test)

acc_svc = round(clf.score(X_train, Y_train) * 100, 2)

print()

print(acc_svc)
# KNeighborsClassifier accuracy



tuned_parameters = [

    {'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

]



knn = KNeighborsClassifier(n_neighbors = 3)



clf = GridSearchCV(knn, tuned_parameters, cv=2)

clf.fit(X_train, Y_train)



print("Best parameters set found on development set:")

print()

print(clf.best_params_)

knn_pred = clf.predict(X_test)

acc_knn = round(clf.score(X_train, Y_train) * 100, 2)

print()

print(acc_knn)
# Perceptron accuracy



tuned_parameters = [

    {'penalty': ['l2', 'l1', 'elasticnet'], 'shuffle': [True, False]}

]



perceptron = Perceptron()



clf = GridSearchCV(perceptron, tuned_parameters, cv=2)

clf.fit(X_train, Y_train)



print("Best parameters set found on development set:")

print()

print(clf.best_params_)

perceptron_pred = clf.predict(X_test)

acc_perceptron = round(clf.score(X_train, Y_train) * 100, 2)

print()

print(acc_perceptron)
# LinearSVC accuracy



tuned_parameters = [

    {'C': [1, 10, 100, 1000], 'loss': ['hinge', 'squared_hinge']}

]





linear_svc = LinearSVC()



clf = GridSearchCV(linear_svc, tuned_parameters, cv=2)

clf.fit(X_train, Y_train)



print("Best parameters set found on development set:")

print()

print(clf.best_params_)

linear_svc_pred = clf.predict(X_test)

acc_linear_svc = round(clf.score(X_train, Y_train) * 100, 2)

print()

print(acc_linear_svc)
# Stochastic Gradient Descent accuracy



tuned_parameters = [

    {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'], 'shuffle':[True, False]}

]



sgd = SGDClassifier()



clf = GridSearchCV(sgd, tuned_parameters, cv=2)

clf.fit(X_train, Y_train)



print("Best parameters set found on development set:")

print()

print(clf.best_params_)

sgd_pred = clf.predict(X_test)

acc_sgd = round(clf.score(X_train, Y_train) * 100, 2)

print()

print(acc_sgd)
# DecisionTreeClassifier accuracy 



tuned_parameters = [

    {'splitter': ['best', 'random']}

]



decision_tree = DecisionTreeClassifier()



clf = GridSearchCV(decision_tree, tuned_parameters, cv=2)

clf.fit(X_train, Y_train)



print("Best parameters set found on development set:")

print()

print(clf.best_params_)

decision_tree_pred = clf.predict(X_test)

acc_decision_tree = round(clf.score(X_train, Y_train) * 100, 2)

print()

print(acc_decision_tree)
# Random Forest accuracy



tuned_parameters = [

    {'criterion': ['gini', 'entropy']}

]



random_forest = RandomForestClassifier(n_estimators=100)



clf = GridSearchCV(random_forest, tuned_parameters, cv=2)

clf.fit(X_train, Y_train)



print("Best parameters set found on development set:")

print()

print(clf.best_params_)

random_forest_pred = clf.predict(X_test)

acc_random_forest = round(clf.score(X_train, Y_train) * 100, 2)

print()

print(acc_random_forest)
#check out what model was the most effective on prediction

models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_logreg, 

              acc_random_forest, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
# submit the result of random forest model's result



submission = pd.DataFrame({

        "Id": test_df["Id"],

        "SalePrice": random_forest_pred

    })
submission.set_index('Id', inplace=True)
submission.to_csv('output.csv')