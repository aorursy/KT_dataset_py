import pandas as pd



# Loading datasets as dataframe

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.shape
import numpy as np
# 'Embarked' column has 3 unique values which denotes the destination the passenger intended to reach.

# So, one hot encoding can be applied to extract info from this column



train_mod = pd.get_dummies(train, prefix_sep='_', columns=['Embarked', 'Sex'])

train_mod.head()
# Other string variables cannot be One-Hot-Encoded, so it is safe to drop them



train_mod.drop(['Ticket', 'Cabin', 'Name'], axis = 1, inplace=True)

train_mod.head()
# Calculating missing values

train_mod.isna().sum()
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy='median')

train_imp = imputer.fit_transform(train_mod) # Imputing the missing values i.e., 'Age' column.
train_imp = pd.DataFrame(train_imp, columns=train_mod.columns) # Imputing returns a numpy array and it can be change to pandas DataFrame with this line of code
train_imp.isna().sum()
train_imp.head()
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
# Segregating the training data as features and targets



y = train_imp['Survived']

X = train_imp.drop(['Survived','PassengerId'], axis = 1)
# Build a SVM classifier and return accuracy score of the model



def svm(X, y):

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  

  svc = SVC(kernel='rbf', gamma=1/X_train.shape[1], random_state=42)

  clf = svc.fit(X_train, y_train)



  y_preds = clf.predict(X_test)

  

  return(accuracy_score(y_preds, y_test))
# Build a Random Forest Classifier and return accuracy score of the model and feature_importances



def rfc(X, y):

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  rfc = RandomForestClassifier(n_estimators = 50, max_depth = 10, random_state = 0)

  rfc.fit(X_train, y_train)



  y_preds_rfc = rfc.predict(X_test)

  features = rfc.feature_importances_

  

  return(accuracy_score(y_preds_rfc, y_test), features)
svm(X,y)
rfc_clf = rfc(X, y)

print(rfc_clf[0])
imp_features = np.where(rfc_clf[1] > 0.1) # Selecting feature importances greater than 0.1. Higher the value, more the importance of that feature

# Here it rendered 5 features

X_mod = X[X.columns[imp_features]] # Creating a new dataframe only with important features
print(X.shape, X_mod.shape)
print(rfc(X_mod, y)[0])
svm(X_mod, y)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)



X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
svm(X_scaled, y) # Accuracy after scaling
print(rfc(X_scaled, y)[0])
X_mod_scaled = scaler.fit_transform(X_mod)



print("Accuracy of SVM after scaling and feature selection: {:.2f} \nAccuracy of RFC after scaling and feature selection: {:.2f}"

      .format(svm(X_mod_scaled, y), rfc(X_mod_scaled, y)[0]))
# I am choosing the best model of all - RFC after scaling - to predict test set



def preprocess(test):

    test_mod = pd.get_dummies(test, prefix_sep='_', columns=['Embarked', 'Sex'])

    test_mod.drop(['PassengerId','Ticket', 'Cabin', 'Name'], axis = 1, inplace=True)

    

    imputer_test = SimpleImputer(strategy='median')

    test_imp = imputer_test.fit_transform(test_mod)

    test_imp = pd.DataFrame(test_imp, columns=test_mod.columns)

    

    scaler_test = StandardScaler()

    test_scaled = scaler_test.fit_transform(test_imp)

    

    return pd.DataFrame(test_scaled, columns=test_imp.columns)
test_final = preprocess(test)

rfc = RandomForestClassifier(n_estimators = 50, max_depth = 10, random_state = 0).fit(X_scaled, y)



predictions = rfc.predict(test_final)
StackingSubmission = pd.DataFrame({ 'PassengerId': test['PassengerId'],

                            'Survived': predictions.astype(int) })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)
def svm_kernels(X, y, kernel):    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    

    svm_clf = SVC(kernel=kernel, gamma=1/X_train.shape[1], C=1.0, random_state=42).fit(X_train, y_train)

    y_preds = svm_clf.predict(X_test)

    

    return accuracy_score(y_preds, y_test)



kernels = ['linear', 'poly', 'rbf', 'sigmoid']



scores = []

for kernel in kernels:

    scores.append(svm_kernels(X_scaled, y, kernel))



print(scores)
def poly_svm(X, y, degree):    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    

    svm_clf = SVC(kernel='poly', degree=degree, gamma=1/X_train.shape[1], C=1.0, random_state=42).fit(X_train, y_train)

    y_preds = svm_clf.predict(X_test)

    

    return accuracy_score(y_preds, y_test)



degrees = np.arange(1,10)



scores_poly = []



for degree in degrees:

    scores_poly.append(poly_svm(X_scaled, y, degree))

    

degree_max = scores_poly.index(max(scores_poly))



svm_best = SVC(kernel='poly', degree=degree_max+1, gamma=1/X.shape[1], C=1.0, random_state=42).fit(X_scaled, y)

    
predictions_svm = svm_best.predict(test_final)
StackingSubmissionSVM = pd.DataFrame({ 'PassengerId': test['PassengerId'],

                            'Survived': predictions_svm.astype(int) })

StackingSubmissionSVM.to_csv("StackingSubmissionSVM.csv", index=False)