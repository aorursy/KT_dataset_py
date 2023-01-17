import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.dummy import DummyClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
# setting up default plotting parameters

%matplotlib inline



plt.rcParams['figure.figsize'] = [20.0, 7.0]

plt.rcParams.update({'font.size': 22,})



sns.set_palette('viridis')

sns.set_style('white')

sns.set_context('talk', font_scale=0.8)
# read in data

df = pd.read_csv('../input/creditcard.csv')



print(df.shape)

df.head()
print(df.Class.value_counts())
# using seaborns countplot to show distribution of questions in dataset

fig, ax = plt.subplots()

g = sns.countplot(df.Class, palette='viridis')

g.set_xticklabels(['Not Fraud', 'Fraud'])

g.set_yticklabels([])



# function to show values on bars

def show_values_on_bars(axs):

    def _show_on_single_plot(ax):        

        for p in ax.patches:

            _x = p.get_x() + p.get_width() / 2

            _y = p.get_y() + p.get_height()

            value = '{:.0f}'.format(p.get_height())

            ax.text(_x, _y, value, ha="center") 



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)

show_values_on_bars(ax)



sns.despine(left=True, bottom=True)

plt.xlabel('')

plt.ylabel('')

plt.title('Distribution of Transactions', fontsize=30)

plt.tick_params(axis='x', which='major', labelsize=15)

plt.show()
# print percentage of questions where target == 1

(len(df.loc[df.Class==1])) / (len(df.loc[df.Class == 0])) * 100
# Prepare data for modeling

# Separate input features and target

y = df.Class

X = df.drop('Class', axis=1)



# setting up testing and training sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
# DummyClassifier to predict only target 0

dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)

dummy_pred = dummy.predict(X_test)



# checking unique labels

print('Unique predicted labels: ', (np.unique(dummy_pred)))



# checking accuracy

print('Test score: ', accuracy_score(y_test, dummy_pred))
# Modeling the data as is

# Train model

lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)

 

# Predict on training set

lr_pred = lr.predict(X_test)
# Checking accuracy

accuracy_score(y_test, lr_pred)
# Checking unique values

predictions = pd.DataFrame(lr_pred)

predictions[0].value_counts()
# f1 score

f1_score(y_test, lr_pred)
# confusion matrix

pd.DataFrame(confusion_matrix(y_test, lr_pred))
recall_score(y_test, lr_pred)
from sklearn.ensemble import RandomForestClassifier
# train model

rfc = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)



# predict on test set

rfc_pred = rfc.predict(X_test)



accuracy_score(y_test, rfc_pred)
# f1 score

f1_score(y_test, rfc_pred)
# confusion matrix

pd.DataFrame(confusion_matrix(y_test, rfc_pred))
# recall score

recall_score(y_test, rfc_pred)
from sklearn.utils import resample
# Separate input features and target

y = df.Class

X = df.drop('Class', axis=1)



# setting up testing and training sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
# concatenate our training data back together

X = pd.concat([X_train, y_train], axis=1)

X.head()
# separate minority and majority classes

not_fraud = X[X.Class==0]

fraud = X[X.Class==1]



# upsample minority

fraud_upsampled = resample(fraud,

                          replace=True, # sample with replacement

                          n_samples=len(not_fraud), # match number in majority class

                          random_state=27) # reproducible results



# combine majority and upsampled minority

upsampled = pd.concat([not_fraud, fraud_upsampled])



# check new class counts

upsampled.Class.value_counts()
# trying logistic regression again with the balanced dataset

y_train = upsampled.Class

X_train = upsampled.drop('Class', axis=1)



upsampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)



upsampled_pred = upsampled.predict(X_test)
# Checking accuracy

accuracy_score(y_test, upsampled_pred)
# f1 score

f1_score(y_test, upsampled_pred)
# confusion matrix

pd.DataFrame(confusion_matrix(y_test, upsampled_pred))
recall_score(y_test, upsampled_pred)
# still using our separated classes fraud and not_fraud from above



# downsample majority

not_fraud_downsampled = resample(not_fraud,

                                replace = False, # sample without replacement

                                n_samples = len(fraud), # match minority n

                                random_state = 27) # reproducible results



# combine minority and downsampled majority

downsampled = pd.concat([not_fraud_downsampled, fraud])



# checking counts

downsampled.Class.value_counts()
# trying logistic regression again with the undersampled dataset



y_train = downsampled.Class

X_train = downsampled.drop('Class', axis=1)



undersampled = LogisticRegression(solver='liblinear').fit(X_train, y_train)



undersampled_pred = undersampled.predict(X_test)
# Checking accuracy

accuracy_score(y_test, undersampled_pred)
# f1 score

f1_score(y_test, undersampled_pred)
# confusion matrix

pd.DataFrame(confusion_matrix(y_test, undersampled_pred))
recall_score(y_test, undersampled_pred)
from imblearn.over_sampling import SMOTE



# Separate input features and target

y = df.Class

X = df.drop('Class', axis=1)



# setting up testing and training sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)



sm = SMOTE(random_state=27, ratio=1.0)

X_train, y_train = sm.fit_sample(X_train, y_train)
smote = LogisticRegression(solver='liblinear').fit(X_train, y_train)



smote_pred = smote.predict(X_test)



# Checking accuracy

accuracy_score(y_test, smote_pred)
# f1 score

f1_score(y_test, smote_pred)
# confustion matrix

pd.DataFrame(confusion_matrix(y_test, smote_pred))
recall_score(y_test, smote_pred)