#### Importing Libraries ####



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sn
dataset = pd.read_csv('../input/churn_data.csv') # Users who were 60 days enrolled, churn in the next 30
#### EDA ####



dataset.head(5) # Viewing the Data
dataset.columns
dataset.describe() # Distribution of Numerical Variables
# Cleaning Data

dataset[dataset.credit_score < 300]

dataset = dataset[dataset.credit_score >= 300]
# Removing NaN

dataset.isna().any()
dataset.isna().sum()
dataset = dataset.drop(columns = ['credit_score', 'rewards_earned'])
## Pie Plots

dataset2 = dataset[['housing', 'is_referred', 'app_downloaded', 'web_user', 'app_web_user', 'ios_user', 'android_user', 'registered_phones', 'payment_type', 'waiting_4_loan', 'cancelled_loan', 'received_loan', 'rejected_loan', 'zodiac_sign', 'left_for_two_month_plus', 'left_for_one_month', 'is_referred']]

fig = plt.figure(figsize=(15, 12))

plt.suptitle('Pie Chart Distributions', fontsize=20)

for i in range(1, dataset2.shape[1] + 1):

    plt.subplot(6, 3, i)

    f = plt.gca()

    f.axes.get_yaxis().set_visible(False)

    f.set_title(dataset2.columns.values[i - 1])

   

    values = dataset2.iloc[:, i - 1].value_counts(normalize = True).values

    index = dataset2.iloc[:, i - 1].value_counts(normalize = True).index

    plt.pie(values, labels = index, autopct='%1.1f%%')

    plt.axis('equal')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
## Exploring Uneven Features

dataset[dataset2.waiting_4_loan == 1].churn.value_counts()

dataset[dataset2.cancelled_loan == 1].churn.value_counts()

dataset[dataset2.received_loan == 1].churn.value_counts()

dataset[dataset2.rejected_loan == 1].churn.value_counts()

dataset[dataset2.left_for_one_month == 1].churn.value_counts()
## Correlation with Response Variable

dataset2.drop(columns = ['housing', 'payment_type', 'registered_phones', 'zodiac_sign']).corrwith(dataset.churn).plot.bar(figsize=(20,10), title = 'Correlation with Response variable', fontsize = 15, rot = 45, grid = True)
## Correlation Matrix

sn.set(style="white")



# Compute the correlation matrix

corr = dataset.drop(columns = ['user', 'churn']).corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(18, 15))



# Generate a custom diverging colormap

cmap = sn.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Removing Correlated Fields

dataset = dataset.drop(columns = ['app_web_user'])
## Note: Although there are somewhat correlated fields, they are not colinear

## These feature are not functions of each other, so they won't break the model

## But these feature won't help much either. Feature Selection should remove them.



dataset.to_csv('new_churn_data.csv', index = False)
import random
dataset = pd.read_csv('new_churn_data.csv')
## Data Preparation

user_identifier = dataset['user']

dataset = dataset.drop(columns = ['user'])
# One-Hot Encoding

dataset.housing.value_counts()

dataset.groupby('housing')['churn'].nunique().reset_index()

dataset = pd.get_dummies(dataset)

dataset.columns

dataset = dataset.drop(columns = ['housing_na', 'zodiac_sign_na', 'payment_type_na'])
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns = 'churn'), dataset['churn'], test_size = 0.2, random_state = 0)
# Balancing the Training Set

y_train.value_counts()



pos_index = y_train[y_train.values == 1].index

neg_index = y_train[y_train.values == 0].index



if len(pos_index) > len(neg_index):

    higher = pos_index

    lower = neg_index

else:

    higher = neg_index

    lower = pos_index

random.seed(0)

higher = np.random.choice(higher, size=len(lower))

lower = np.asarray(lower)

new_indexes = np.concatenate((lower, higher))



X_train = X_train.loc[new_indexes,]

y_train = y_train[new_indexes]
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))

X_test2 = pd.DataFrame(sc_X.transform(X_test))

X_train2.columns = X_train.columns.values

X_test2.columns = X_test.columns.values

X_train2.index = X_train.index.values

X_test2.index = X_test.index.values

X_train = X_train2

X_test = X_test2
# Fitting Model to the Training Set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)
# Predicting Test Set

y_pred = classifier.predict(X_test)
# Evaluating Results

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

cm = confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

precision_score(y_test, y_pred) # tp / (tp + fp)

recall_score(y_test, y_pred) # tp / (tp + fn)

f1_score(y_test, y_pred)
df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))

plt.figure(figsize = (10,7))

sn.set(font_scale=1.4)

sn.heatmap(df_cm, annot=True, fmt='g')

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("SVM Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))
# Analyzing Coefficients

pd.concat([pd.DataFrame(X_train.columns, columns = ["features"]),

           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])

           ],axis = 1)
#### Feature Selection ####





# Recursive Feature Elimination

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression
# Model to Test

classifier = LogisticRegression()

# Select Best X Features

rfe = RFE(classifier, 20)

rfe = rfe.fit(X_train, y_train)

# summarize the selection of the attributes

print(rfe.support_)

print(rfe.ranking_)

X_train.columns[rfe.support_]
# New Correlation Matrix

sn.set(style="white")



# Compute the correlation matrix

corr = X_train[X_train.columns[rfe.support_]].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(18, 15))



# Generate a custom diverging colormap

cmap = sn.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}) 
# Fitting Model to the Training Set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train[X_train.columns[rfe.support_]], y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test[X_train.columns[rfe.support_]])



# Evaluating Results

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

cm = confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)

precision_score(y_test, y_pred) # tp / (tp + fp)

recall_score(y_test, y_pred) # tp / (tp + fn)

f1_score(y_test, y_pred)



df_cm = pd.DataFrame(cm, index = (1, 0), columns = (1, 0))

plt.figure(figsize = (10,7))

sn.set(font_scale=1.4)

sn.heatmap(df_cm, annot=True, fmt='g')

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier,

                             X = X_train[X_train.columns[rfe.support_]],

                             y = y_train, cv = 10)

print("SVM Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))



# Analyzing Coefficients

pd.concat([pd.DataFrame(X_train[X_train.columns[rfe.support_]].columns, columns = ["features"]),

           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])

           ],axis = 1)
# Formatting Final Results

final_results = pd.concat([y_test, user_identifier], axis = 1).dropna()

final_results['predicted_churn'] = y_pred

final_results = final_results[['user', 'churn', 'predicted_churn']].reset_index(drop=True)

final_results