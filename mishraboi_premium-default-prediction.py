# Data Wrangling 

import numpy as np

import pandas as pd 



# Data Visualisation 

%matplotlib inline

import matplotlib.pyplot as plt 

import seaborn as sns



# Machine Learning 

from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron 

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, mean_squared_error
train_data = pd.read_csv('../input/train (1).csv')

test_data = pd.read_csv('../input/test (1).csv')

combine = [train_data, test_data]
train_data.head()
test_data.head()
train_data.describe()
train_data.describe(percentiles = [.08, .07, .06])
plt.figure(figsize = (15, 6))

sns.heatmap(train_data.corr(), annot = True)

plt.show()
train_data.isnull().sum()
test_data.isnull().sum()
for dataset in combine: 

    dataset['age'] = dataset['age_in_days']//365

    dataset.drop(['age_in_days'], axis = 1, inplace = True)

train_data.head()
train_data[['sourcing_channel', 'target']].groupby('sourcing_channel', as_index = False).mean()
train_data['IncomeBands'] = pd.cut(train_data['Income'], 5)

train_data[['IncomeBands', 'target']].groupby('IncomeBands', as_index = False).count()
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler()

scaler = scaler.fit(train_data[['Income']])

x_scaled = scaler.transform(train_data[['Income']])

x_scaled
# print(scaler.mean_)

print(scaler.scale_)
train_data['scaled_income'] = x_scaled

train_data.head()
train_data['IncomeBands'] = pd.cut(train_data['scaled_income'], 5)

train_data[['IncomeBands', 'target']].groupby('IncomeBands', as_index = False).count()
print(train_data['Income'].mean())

print(train_data['Income'].median())
plt.hist(train_data['Income'])

plt.show()
upper_bound = 0.95

lower_bound = 0.1

res = train_data['Income'].quantile([lower_bound, upper_bound])

print(res)
true_index = (train_data['Income'] < res.loc[upper_bound])

true_index
false_index = ~true_index
no_outlier_data = train_data[true_index].copy()

no_outlier_data.head()
no_outlier_data['IncomeBands'] = pd.cut(no_outlier_data['Income'], 5)

no_outlier_data[['IncomeBands', 'target']].groupby('IncomeBands', as_index = False).count()
combine = [train_data, test_data]

for dataset in combine: 

    dataset.loc[ dataset['Income'] <= 23603.99, 'Income'] = 0

    dataset.loc[(dataset['Income'] > 23603.99) & (dataset['Income'] <= 109232.0), 'Income'] = 1

    dataset.loc[(dataset['Income'] > 109232.0) & (dataset['Income'] <= 194434.0), 'Income'] = 2

    dataset.loc[(dataset['Income'] > 194434.0) & (dataset['Income'] <= 279636.0), 'Income'] = 3

    dataset.loc[(dataset['Income'] > 279636.0) & (dataset['Income'] <= 364838.0), 'Income'] = 4

    dataset.loc[(dataset['Income'] > 364838.0) & (dataset['Income'] <= 450040.0), 'Income'] = 5

    dataset.loc[ dataset['Income'] > 450040.0, 'Income'] = 6

    

train_data.head()
train_data.loc[false_index, 'Income'] = 5

train_data.head()
train_data.drop(['IncomeBands', 'scaled_income'], axis = 1, inplace = True)

train_data.head()
train_data['AgeBands'] = pd.cut(train_data['age'], 5)

train_data[['AgeBands', 'target']].groupby('AgeBands', as_index = False).count()
for dataset in combine:    

    dataset.loc[ dataset['age'] <= 37.4, 'age'] = 0

    dataset.loc[(dataset['age'] > 37.4) & (dataset['age'] <= 53.8), 'age'] = 1

    dataset.loc[(dataset['age'] > 53.8) & (dataset['age'] <= 70.2), 'age'] = 2

    dataset.loc[(dataset['age'] > 70.2) & (dataset['age'] <= 86.6), 'age'] = 3

    dataset.loc[ dataset['age'] > 86.6, 'age'] = 4

train_data.drop('AgeBands', axis = 1, inplace = True)

combine = [train_data, test_data]

train_data.head()
train_data[['age', 'application_underwriting_score']].groupby('age').mean()
train_data['PremBand'] = pd.cut(train_data['no_of_premiums_paid'], 5)

train_data[['PremBand', 'application_underwriting_score']].groupby('PremBand').count()
print(train_data['application_underwriting_score'].mean())

print(train_data['application_underwriting_score'].std())
print(train_data[train_data['sourcing_channel'] == 'A']['application_underwriting_score'].std())

train_data[['sourcing_channel', 'target']].groupby('sourcing_channel', as_index = False).mean()
# print(train_data[train_data['sourcing_channel'] == 'C']['application_underwriting_score'].std())

train_data[['sourcing_channel', 'application_underwriting_score']].groupby('sourcing_channel', as_index = False).mean()
train_data[['residence_area_type', 'application_underwriting_score']].groupby('residence_area_type', as_index = False).mean()
train_data.dtypes
combine = [train_data, test_data]

for dataset in combine: 

    mask1 = dataset['application_underwriting_score'].isnull()

    for source in ['A', 'B', 'C', 'D', 'E']:

        mask2 = (dataset['sourcing_channel'] == source)

        source_mean = dataset[dataset['sourcing_channel'] == source]['application_underwriting_score'].mean()

        dataset.loc[mask1 & mask2, 'application_underwriting_score'] = source_mean

train_data.head()
dataset['application_underwriting_score'].isnull()
test_data[test_data['Count_3-6_months_late'].isnull()]
sns.countplot(x = 'Count_3-6_months_late', data = train_data, hue = 'target')
sns.countplot(x = 'Count_6-12_months_late', data = train_data, hue = 'target')
combine = [train_data, test_data]

for dataset in combine: 

    dataset['late_premium'] = 0.0

train_data.head()
combine = [train_data, test_data]

for dataset in combine:

        dataset.loc[(dataset['Count_3-6_months_late'].isnull()),  'late_premium'] = np.NaN

        dataset.loc[(dataset['Count_3-6_months_late'].notnull()), 'late_premium'] = dataset['Count_3-6_months_late'] + dataset['Count_6-12_months_late'] + dataset['Count_more_than_12_months_late']

train_data.head() 
train_data['target'].corr(train_data['late_premium'])
plt.figure(figsize = (15, 6))

sns.heatmap(test_data.corr(), annot = True)
sns.regplot(x = 'perc_premium_paid_by_cash_credit', y = 'late_premium', data = train_data)
sns.countplot(x = 'late_premium', data = train_data, hue = 'target')
train_data[['late_premium', 'target']].groupby('late_premium').mean()
# for dataset in [train_data]:

train_data.loc[(train_data['target'] == 0) & (train_data['late_premium'].isnull()),'late_premium'] = 7

train_data.loc[(train_data['target'] == 1) & (train_data['late_premium'].isnull()),'late_premium'] = 2

train_data.head()
print(train_data.isnull().sum())

print('\n')

print(test_data.isnull().sum())
guess_prem = np.zeros(5)

for dataset in [test_data]:

    for i in range(1, 6):

        guess_df = dataset[(dataset['Income'] == i)]['late_premium'].dropna()



        # age_mean = guess_df.mean()

        # age_std = guess_df.std()

        # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



        premium_guess = guess_df.median()

        guess_prem[i - 1] = int(premium_guess) 



    for j in range(1, 6):

        dataset.loc[(dataset.late_premium.isnull()) & (dataset.Income == j), 'late_premium'] = guess_prem[j - 1] + 1



    dataset['late_premium'] = dataset['late_premium'].astype(int)



test_data.head(10)
train_data.drop(['Count_3-6_months_late', 'Count_6-12_months_late', 'Count_more_than_12_months_late'], axis = 1, inplace = True)

test_data.drop(['Count_3-6_months_late', 'Count_6-12_months_late', 'Count_more_than_12_months_late'], axis = 1, inplace = True)
# Converting Area Type and sourcing channel to Ordinal Variables

combine = [train_data, test_data]

for dataset in combine: 

    dataset['residence_area_type'] = dataset['residence_area_type'].map( {'Urban' : 1, 'Rural' : 0} )

    dataset['sourcing_channel'] = dataset['sourcing_channel'].map( {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4} )

train_data.head()

train_data['application_underwriting_score'] = train_data['application_underwriting_score']/100

train_data.head()
upper_bound = 0.95

res = train_data['no_of_premiums_paid'].quantile([upper_bound])

print(res)

true_index = train_data['no_of_premiums_paid'] < res.loc[upper_bound]

false_index = ~true_index

true_index
train_data['PremBand'] = pd.cut(train_data[true_index]['no_of_premiums_paid'], 4)

train_data[['PremBand', 'application_underwriting_score']].groupby('PremBand').count()
# combine = [train_data, test_data]

# for dataset in combine: 

#     dataset.loc[ dataset['no_of_premiums_paid'] <= 6.25, 'no_of_premiums_paid'] = 0

#     dataset.loc[(dataset['no_of_premiums_paid'] > 6.25) & (dataset['no_of_premiums_paid'] <= 10.5), 'no_of_premiums_paid'] = 1

#     dataset.loc[(dataset['no_of_premiums_paid'] > 10.50) & (dataset['no_of_premiums_paid'] <= 14.75), 'no_of_premiums_paid'] = 2

#     dataset.loc[(dataset['no_of_premiums_paid'] > 14.75) & (dataset['no_of_premiums_paid'] <= 19.0), 'no_of_premiums_paid'] = 3

#     dataset.loc[ dataset['no_of_premiums_paid'] > 19.0, 'no_of_premiums_paid'] = 4

    

# train_data.drop('PremBand', axis = 1, inplace = True)

# train_data.head()
upper_bound = 0.90

res = train_data['premium'].quantile([upper_bound])

print(res)

true_index = train_data['premium'] < res.loc[upper_bound]

false_index = ~true_index

true_index
train_data['PremBand'] = pd.cut(train_data[true_index]['premium'], 4)

train_data[['PremBand', 'target']].groupby('PremBand').count()
test_data.head()
combine = [train_data]

for dataset in combine: 

    dataset.loc[ dataset['premium'] <= 5925.0, 'premium'] = 0

    dataset.loc[(dataset['premium'] > 5925.00) & (dataset['premium'] <= 10650.0), 'premium'] = 1

    dataset.loc[(dataset['premium'] > 10650.0) & (dataset['premium'] <= 15375.0), 'premium'] = 2

    dataset.loc[(dataset['premium'] > 15375.0) & (dataset['premium'] <= 201200.0), 'premium'] = 3

    dataset.loc[ dataset['premium'] > 201200.0, 'premium'] = 4

train_data.drop('PremBand', axis = 1, inplace = True)

train_data.head()

combine = [train_data, test_data]
train_data['PremBand'] = pd.cut(train_data['perc_premium_paid_by_cash_credit'], 4)

train_data[['PremBand', 'target']].groupby('PremBand').mean()
combine = [train_data, test_data]

for dataset in combine: 

    dataset.loc[ dataset['perc_premium_paid_by_cash_credit'] <= 0.25, 'perc_premium_paid_by_cash_credit'] = 0

    dataset.loc[(dataset['perc_premium_paid_by_cash_credit'] > 0.25) & (dataset['perc_premium_paid_by_cash_credit'] <= 0.5), 'perc_premium_paid_by_cash_credit'] = 1

    dataset.loc[(dataset['perc_premium_paid_by_cash_credit'] > 0.5) & (dataset['perc_premium_paid_by_cash_credit'] <= 0.75), 'perc_premium_paid_by_cash_credit'] = 2

    dataset.loc[ dataset['perc_premium_paid_by_cash_credit'] > 0.75, 'perc_premium_paid_by_cash_credit'] = 3

train_data.drop('PremBand', axis = 1, inplace = True)

train_data.head()
test_data.head()
train_data[['perc_premium_paid_by_cash_credit', 'late_premium']] = train_data[['perc_premium_paid_by_cash_credit', 'late_premium']].astype(int)

test_data[['perc_premium_paid_by_cash_credit']] = test_data[['perc_premium_paid_by_cash_credit']].astype(int)

test_data.head()
X_train = train_data.drop(['id', 'target', 'premium', 'perc_premium_paid_by_cash_credit'], axis = 1).copy()

y_train = train_data['target']

X_test = test_data.drop(['id', 'perc_premium_paid_by_cash_credit'], axis = 1).copy()

print(X_train.shape, y_train.shape, X_test.shape)
X_train.head()
X_test.head()
from imblearn.over_sampling import SMOTE

print('Number of positive and negative reviews:\n',y_train.value_counts())

sm = SMOTE(random_state=0,ratio=1.0)

X_train_res,y_train_res = sm.fit_sample(X_train,y_train)

print('Shape after oversampling\n',X_train_res.shape) 

print('Equal 1s and 0s \n', np.bincount(y_train_res))
# Without Oversampling



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)

acc_log



from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



cm = confusion_matrix(logreg.predict(X_train),y_train)

print(cm)



print(classification_report(logreg.predict(X_train),y_train))



tnr = np.round(cm[0][0]/(cm[0][0] + cm[1][0]) * 100,3)

tpr = np.round(cm[1][1]/(cm[1][1] + cm[0][1]) * 100,3)

fpr = np.round(cm[1][0] / (cm[1][0] + cm[0][0]) * 100,3)

print('TPR = ',tpr,'%')

print('TNR = ',tnr,'%')

print('FPR = ',fpr,'%')
# With Oversampling



logreg = LogisticRegression()

logreg.fit(X_train_res, y_train_res)

y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train_res, y_train_res) * 100, 2)

acc_log



from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



cm = confusion_matrix(logreg.predict(X_train_res),y_train_res)

print(cm)



print(classification_report(logreg.predict(X_train_res),y_train_res))



tnr = np.round(cm[0][0]/(cm[0][0] + cm[1][0]) * 100,3)

tpr = np.round(cm[1][1]/(cm[1][1] + cm[0][1]) * 100,3)

fpr = np.round(cm[1][0] / (cm[1][0] + cm[0][0]) * 100,3)

print('TPR = ',tpr,'%')

print('TNR = ',tnr,'%')

print('FPR = ',fpr,'%')

coeff_data = pd.DataFrame(train_data.columns.delete(0))

coeff_data.columns = ['Feature']

coeff_data['Correlation'] = pd.Series(logreg.coef_[0])

coeff_data.sort_values(by = 'Correlation', ascending = False)
# Gaussian Naive Bayes ~ Without oversampling



gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

acc_gaussian
# Gaussian Naive Bayes ~ With oversampling



gaussian = GaussianNB()

gaussian.fit(X_train_res, y_train_res)

y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train_res, y_train_res) * 100, 2)

print(acc_gaussian)



cm = confusion_matrix(gaussian.predict(X_train_res),y_train_res)

print(cm)



print(classification_report(gaussian.predict(X_train_res),y_train_res))



tnr = np.round(cm[0][0]/(cm[0][0] + cm[1][0]) * 100,3)

tpr = np.round(cm[1][1]/(cm[1][1] + cm[0][1]) * 100,3)

fpr = np.round(cm[1][0] / (cm[1][0] + cm[0][0]) * 100,3)

print('TPR = ',tpr,'%')

print('TNR = ',tnr,'%')

print('FPR = ',fpr,'%')
# without oversampling

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)

acc_knn
# with oversampling

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train_res, y_train_res)

y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train_res, y_train_res) * 100, 2)



cm = confusion_matrix(knn.predict(X_train_res),y_train_res)

print(cm)



print(classification_report(knn.predict(X_train_res),y_train_res))



tnr = np.round(cm[0][0]/(cm[0][0] + cm[1][0]) * 100,3)

tpr = np.round(cm[1][1]/(cm[1][1] + cm[0][1]) * 100,3)

fpr = np.round(cm[1][0] / (cm[1][0] + cm[0][0]) * 100,3)

print('TPR = ',tpr,'%')

print('TNR = ',tnr,'%')

print('FPR = ',fpr,'%')

# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)

acc_perceptron
# Perceptron - with oversampling



perceptron = Perceptron()

perceptron.fit(X_train_res, y_train_res)

y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train_res, y_train_res) * 100, 2)

acc_perceptron



cm = confusion_matrix(perceptron.predict(X_train_res),y_train_res)

print(cm)



print(classification_report(perceptron.predict(X_train_res),y_train_res))



tnr = np.round(cm[0][0]/(cm[0][0] + cm[1][0]) * 100,3)

tpr = np.round(cm[1][1]/(cm[1][1] + cm[0][1]) * 100,3)

fpr = np.round(cm[1][0] / (cm[1][0] + cm[0][0]) * 100,3)

print('TPR = ',tpr,'%')

print('TNR = ',tnr,'%')

print('FPR = ',fpr,'%')
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)

acc_sgd
# Stochastic Gradient Descent - with oversampling



sgd = SGDClassifier()

sgd.fit(X_train_res, y_train_res)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train_res, y_train_res) * 100, 2)

print(acc_sgd)



cm = confusion_matrix(sgd.predict(X_train_res),y_train_res)

print(cm)



print(classification_report(sgd.predict(X_train_res),y_train_res))



tnr = np.round(cm[0][0]/(cm[0][0] + cm[1][0]) * 100,3)

tpr = np.round(cm[1][1]/(cm[1][1] + cm[0][1]) * 100,3)

fpr = np.round(cm[1][0] / (cm[1][0] + cm[0][0]) * 100,3)

print('TPR = ',tpr,'%')

print('TNR = ',tnr,'%')

print('FPR = ',fpr,'%')
# Decision Tree



decision_tree = DecisionTreeClassifier(max_depth = 7)

decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

acc_decision_tree
# Decision Tree - oversampling



decision_tree = DecisionTreeClassifier(max_depth = 7)

decision_tree.fit(X_train_res, y_train_res)

y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train_res, y_train_res) * 100, 2)

print(acc_decision_tree)





cm = confusion_matrix(decision_tree.predict(X_train_res),y_train_res)

print(cm)



print(classification_report(decision_tree.predict(X_train_res),y_train_res))



tnr = np.round(cm[0][0]/(cm[0][0] + cm[1][0]) * 100,3)

tpr = np.round(cm[1][1]/(cm[1][1] + cm[0][1]) * 100,3)

fpr = np.round(cm[1][0] / (cm[1][0] + cm[0][0]) * 100,3)

print('TPR = ',tpr,'%')

print('TNR = ',tnr,'%')

print('FPR = ',fpr,'%')

# Random Forest



random_forest = RandomForestClassifier(n_estimators = 10)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
# Random Forest - oversampling



random_forest = RandomForestClassifier(n_estimators = 10)

random_forest.fit(X_train_res, y_train_res)

y_pred = random_forest.predict(X_test)

random_forest.score(X_train_res, y_train_res)

acc_random_forest = round(random_forest.score(X_train_res, y_train_res) * 100, 2)

print(acc_random_forest)



cm = confusion_matrix(random_forest.predict(X_train_res),y_train_res)

print(cm)



print(classification_report(random_forest.predict(X_train_res),y_train_res))



tnr = np.round(cm[0][0]/(cm[0][0] + cm[1][0]) * 100,3)

tpr = np.round(cm[1][1]/(cm[1][1] + cm[0][1]) * 100,3)

fpr = np.round(cm[1][0] / (cm[1][0] + cm[0][0]) * 100,3)

print('TPR = ',tpr,'%')

print('TNR = ',tnr,'%')

print('FPR = ',fpr,'%')
pred_values = pd.DataFrame({

    'Model': ['KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 

              'Decision Tree'],

    'Score': [acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_decision_tree]})

pred_values.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "id": test_data["id"],

        "target": y_pred

    })

submission.to_csv('submission.csv', index=False)
submission.describe()