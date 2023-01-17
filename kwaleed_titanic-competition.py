# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    #print(dirname)

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
# women = train_data.loc[train_data.Sex == 'female']["Survived"]

# rate_women = sum(women)/len(women)



# print("% of women who survived:", rate_women)
# men = train_data.loc[train_data.Sex == 'male']["Survived"]

# rate_men = sum(men)/len(men)



# print("% of men who survived:", rate_men)
# from sklearn.ensemble import RandomForestClassifier



# y = train_data["Survived"]



# features = ["Pclass", "Sex", "SibSp", "Parch"]

# X = pd.get_dummies(train_data[features])

# X_test = pd.get_dummies(test_data[features])



# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

# model.fit(X, y)

# predictions = model.predict(X_test)



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")
from sklearn.model_selection import train_test_split
train_data.shape
train_data.isnull().sum()
test_data.isnull().sum()
# out of 891 total rows 687 values are NaN for Cabin column



# PassengerId, Name, Ticket, Cabin intutively seem not very much related to the survival



dropFeatures = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']



train_data_drop = train_data.drop(columns = dropFeatures)



# Same for test_data

test_data_drop = test_data.drop(columns = dropFeatures)
# Shape of our training data after dropping columns

# PassengerId, Name, Ticket, Cabin

print(f"train_data_drop shape: {train_data_drop.shape}")

print(f"test_data_drop shape: {test_data_drop.shape}")
# # Set mean where ever the value == NaN in Age Column

# # For now we have NaNs only in Age Column

# train_data_drop['Age'].fillna(train_data['Age'].mean(), inplace=True)



# # Same for test_data

# test_data_drop['Age'].fillna(test_data['Age'].mean(), inplace=True)

# test_data_drop['Fare'].fillna(test_data['Fare'].mean(), inplace=True)
# # Set mode where ever the value == NaN in Age Column

# # For now we have NaNs in Age Column and Embarked Column

# train_data_drop['Age'].fillna(train_data['Age'].mode()[0], inplace=True)

# train_data_drop['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)



# # Same for test_data

# test_data_drop['Age'].fillna(test_data['Age'].mode()[0], inplace=True)

# test_data_drop['Fare'].fillna(test_data['Fare'].mode()[0], inplace=True)
# Mean imputation in the 'Age' field in the train_data_after_drop

train_data_drop['Age'].fillna(train_data['Age'].mean(), inplace=True)



# Mean imputation in the 'Age' field in the test_data_after_drop

test_data_drop['Age'].fillna(test_data['Age'].mean(), inplace=True)



# Mode imputation in the 'Fare' field in the test_data_after_drop

test_data_drop['Fare'].fillna(test_data['Fare'].mode()[0], inplace=True)
# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()



# # For train data

# train_data_drop[['Age','SibSp', 'Parch', 'Fare']] = scaler.fit_transform(train_data_drop[['Age','SibSp', 'Parch', 'Fare']])



# # For test data

# test_data_drop[['Age','SibSp', 'Parch', 'Fare']] = scaler.fit_transform(test_data_drop[['Age','SibSp', 'Parch', 'Fare']])
# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()



# # For train data

# train_data_drop[['Age','SibSp', 'Parch', 'Fare']] = scaler.fit_transform(train_data_drop[['Age','SibSp', 'Parch', 'Fare']])



# # For test data

# test_data_drop[['Age','SibSp', 'Parch', 'Fare']] = scaler.fit_transform(test_data_drop[['Age','SibSp', 'Parch', 'Fare']])
#train_data_drop.describe()
#test_data_drop.describe()
# Code to check if NaNs exist in any Column

train_data_drop.isnull().sum()
# Code to check if NaNs exist in any Column

test_data_drop.isnull().sum()
# Splits the training data into train/validation by 70-30

X_train, X_validation, y_train, y_validation = train_test_split(train_data_drop.drop(columns = ['Survived']), 

                                                    train_data_drop['Survived'], 

                                                    stratify=train_data_drop['Survived'], 

                                                    test_size = 0.3)



# Code below converts the categorical columns 'Sex' and 'Embarked' to one hot encoding

X_train = pd.get_dummies(X_train)

X_validation = pd.get_dummies(X_validation)



# Similar for test data set

X_test = pd.get_dummies(test_data_drop)
# 70% of 891 = 623, of the data to be used for training

print(f"X_train Shape: {X_train.shape}")



# 30% of 891 = 268, of the data to be used for validation

print(f"X_validation Shape: {X_validation.shape}")



print(f"X_test Shape: {X_test.shape}")
import numpy as np

import matplotlib.pyplot as plt



from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels



def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
from sklearn.ensemble import RandomForestClassifier



# instantiate the Random Forest Classifier

rfc = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=1)



# fit(train) the model with the training dataset

rfc.fit(X_train, y_train)



# make predictions on the validation dataset

y_pred = rfc.predict(X_validation)
# Import accuracy_score and classification report

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



# Compute validation dataset accuracy  

acc = accuracy_score(y_pred, y_validation)

print("Test set accuracy: {:.2f} percent".format(acc*100))



# Calculate Precision, Recall and F1 Score

print(classification_report(y_validation, y_pred))
np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plot_confusion_matrix(y_validation, y_pred, classes=np.array(['Died', 'Survived']),

                      title='Confusion matrix, without normalization')

plt.show()
predictions = rfc.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('Submission_File_1.csv', index=False)

print("Your submission was successfully saved!")
# Import DecisionTreeClassifier from sklearn.tree

from sklearn.tree import DecisionTreeClassifier



# Instantiate a DecisionTreeClassifier 'dt' with a maximum depth of 6

dt = DecisionTreeClassifier(max_depth=6)



# Fit dt to the training set

dt.fit(X_train, y_train)



# Predict test set labels

y_pred = dt.predict(X_validation)
# Import accuracy_score and classification report

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



# Predict test set labels

#y_pred = dt.predict(X_validation)



# Compute test set accuracy  

acc = accuracy_score(y_pred, y_validation)

print("Test set accuracy: {:.2f} percent".format(acc*100))



# Calculate Precision, Recall and F1 Score

print(classification_report(y_validation, y_pred))
np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plot_confusion_matrix(y_validation, y_pred, classes=np.array(['Died', 'Survived']),

                      title='Confusion matrix, without normalization')

plt.show()
# Create a pd.Series of features importances

importances = pd.Series(data=dt.feature_importances_,

                        index= X_train.columns)



# Sort importances

importances_sorted = importances.sort_values()



# Draw a horizontal barplot of importances_sorted

importances_sorted.plot(kind='barh', color='lightgreen')

plt.title('Features Importances')

plt.show()
# predictions = dt.predict(X_test)



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")
# Import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier



# Import BaggingClassifier

from sklearn.ensemble import BaggingClassifier



# Instantiate dt

dt = DecisionTreeClassifier(random_state=1)



# Instantiate bc

bc = BaggingClassifier(base_estimator=dt, n_estimators=50, random_state=1)
# Fit bc to the training set

bc.fit(X_train, y_train)



# Predict test set labels

y_pred = bc.predict(X_validation)



# Evaluate acc_test

acc_test = accuracy_score(y_validation, y_pred)

print('Test set accuracy of bc: {:.2f} percent'.format(acc_test*100))



# Evaluate Precision, Recall and F1 Score

print(classification_report(y_validation, y_pred))
np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plot_confusion_matrix(y_validation, y_pred, classes=np.array(['Died', 'Survived']),

                      title='Confusion matrix, without normalization')

plt.show()
# predictions = bc.predict(X_test)



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")
# Import KNN and Logistic Regression

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier



# Set seed for reproducibility

SEED=1



# Instantiate lr

lr = LogisticRegression(random_state=SEED)



# Instantiate knn

knn = KNN(n_neighbors=27)



# Instantiate dt

dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)



# Define the list classifiers

classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]
# Import VotingClassifier from sklearn.ensemble

from sklearn.ensemble import VotingClassifier



# Import accuracy_score and classification report

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



# Instantiate a VotingClassifier vc

vc = VotingClassifier(estimators=classifiers)     



# Fit vc to the training set

vc.fit(X_train, y_train)   



# Evaluate the test set predictions

y_pred = vc.predict(X_validation)



# Calculate accuracy score

accuracy = accuracy_score(y_pred, y_validation)

print('Voting Classifier: {:.3f}'.format(accuracy*100))



# Calculate Precision, Recall and F1 Score

print(classification_report(y_validation, y_pred))
np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plot_confusion_matrix(y_validation, y_pred, classes=np.array(['Died', 'Survived']),

                      title='Confusion matrix, without normalization')

plt.show()
# Import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier



# Import AdaBoostClassifier

from sklearn.ensemble import AdaBoostClassifier



# Instantiate dt

dt = DecisionTreeClassifier(max_depth=8, random_state=1)



# Instantiate ada

ada = AdaBoostClassifier(base_estimator=dt, n_estimators=200, random_state=1)



# Fit ada to the training set

ada.fit(X_train, y_train)



# Compute the probabilities of obtaining the positive class

y_pred = ada.predict(X_validation)
# Import accuracy_score and classification report

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



# Compute test set accuracy  

acc = accuracy_score(y_pred, y_validation)

print("Test set accuracy: {:.2f} percent".format(acc*100))



# Calculate Precision, Recall and F1 Score

print(classification_report(y_validation, y_pred))
np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plot_confusion_matrix(y_validation, y_pred, classes=np.array(['Died', 'Survived']),

                      title='Confusion matrix, without normalization')

plt.show()
from sklearn.svm import LinearSVC



svm = LinearSVC()

svm.fit(X_train, y_train)



# Predict test set labels

y_pred = svm.predict(X_validation)
# Import accuracy_score and classification report

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



# Evaluate acc_test

acc_test = accuracy_score(y_validation, y_pred)

print('Test set accuracy of bc: {:.2f} percent'.format(acc_test*100))



# Evaluate Precision, Recall and F1 Score

print(classification_report(y_validation, y_pred))
np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plot_confusion_matrix(y_validation, y_pred, classes=np.array(['Died', 'Survived']),

                      title='Confusion matrix, without normalization')

plt.show()
# Import GradientBoostingRegressor

from sklearn.ensemble import GradientBoostingClassifier



# Instantiate gb

gb = GradientBoostingClassifier(max_depth=2, 

            n_estimators=200,

            random_state=2)



# Fit gb to the training set

gb.fit(X_train, y_train)



# Predict test set labels

y_pred = gb.predict(X_validation)
# Import accuracy_score and classification report

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report



# Evaluate acc_test

acc_test = accuracy_score(y_validation, y_pred)

print('Test set accuracy of bc: {:.2f} percent'.format(acc_test*100))



# Evaluate Precision, Recall and F1 Score

print(classification_report(y_validation, y_pred))
np.set_printoptions(precision=2)



# Plot non-normalized confusion matrix

plot_confusion_matrix(y_validation, y_pred, classes=np.array(['Died', 'Survived']),

                      title='Confusion matrix, without normalization')

plt.show()
# predictions = bc.predict(X_test)



# output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output.to_csv('my_submission.csv', index=False)

# print("Your submission was successfully saved!")