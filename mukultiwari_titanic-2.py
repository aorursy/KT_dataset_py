# Imports



import numpy as np

import pandas as pd

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
# Importing Dataset

training_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')



# Saving the passengerId of test data for later use.

passengerId = test_data['PassengerId']

# Since passengerId does not have significant contribution to survival directly therefore we will Drop it.

training_data.drop(labels='PassengerId', axis=1, inplace=True)

test_data.drop(labels='PassengerId', axis=1, inplace=True)
training_data.head()
test_data.head()
# Preprocessing and Feature Egineering



def preprocess_data(df):

    

    # Title Feature

    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])



    # Name Leangth

    df['Name_Len'] = df['Name'].apply(lambda x: len(x))



    # Dropping the name feature 

    df.drop(labels='Name', axis=1, inplace=True)

    

    # Categorizing the name length by simply dividing it with 10.

    df.Name_Len = (df.Name_Len/10).astype(np.int64)+1

    

    # Taking care of null values in Age 

    full_data = pd.concat([training_data, test_data])

    

    df_age_mean = full_data.Age.mean()

    df_age_std = full_data.Age.std()

    df_age_null = df.Age.isnull().sum()

    rand_tr_age = np.random.randint(df_age_mean - df_age_std, df_age_mean + df_age_std, size=df_age_null)

    df['Age'][np.isnan(df['Age'])] = rand_tr_age

    df['Age'] = df['Age'].astype(int) + 1



    df.Age = (df.Age/15).astype(np.int64)

    

    # We will create a new feature of family size = SibSp + Parch + 1

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    

    # wheather or not the passenger was alone ?

    df['isAlone'] = df['FamilySize'].map(lambda x: 1 if x == 1 else 0)

    

    df.drop(labels=['SibSp', 'Parch'], axis=1, inplace=True)

    df.drop(labels='Ticket', axis=1, inplace=True)

    

    # Replacing Null values of Fare with Mean

    df['Fare'][np.isnan(df['Fare'])] = df.Fare.mean()

 

    # Categorizing the fare value by dividing it with 20 simply

    df.Fare = (df.Fare /20).astype(np.int64) + 1 

    

    # Making a new feature hasCabin which is 1 if cabin is available else 0

    df['hasCabin'] = df.Cabin.notnull().astype(int)

    df.drop(labels='Cabin', axis=1, inplace=True)

    

    # Since "S" is the most frequent class constituting 72% of the total therefore we will replace null values with "S"

    df['Embarked'] = df['Embarked'].fillna('S')

    

    return df
# Cleaning Data for Classification



train = preprocess_data(training_data)

test = preprocess_data(test_data)

X = train.iloc[:, 1:12].values

y = train.iloc[:, 0].values
# Resolving the categorical data for training set



label_encoder_sex_tr = LabelEncoder()

label_encoder_title_tr = LabelEncoder()

label_encoder_embarked_tr = LabelEncoder()

X[:, 1] = label_encoder_sex_tr.fit_transform(X[:, 1])

X[:, 5] = label_encoder_title_tr.fit_transform(X[:, 5])

X[:, 4] = label_encoder_embarked_tr.fit_transform(X[:, 4])
# Splitting the dataset into training and test set

# For Submission Use whole Data for training



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
# Feature Scaling

# Non_scaled_x_train = X_train

# Non_scaled_x_test = X_test

# scaler_x = MinMaxScaler((-1,1))

# X_train = scaler_x.fit_transform(X_train)

# X_test = scaler_x.transform(X_test)
# For Submission consider whole data

# Feature Scaling

Non_scaled_x_train = X

scaler_x = MinMaxScaler((-1,1))

X_train = scaler_x.fit_transform(X)

y_train = y
# Hyper Parameter tuned

# XGBoost

xgb = XGBClassifier(colsample_bytree = 0.5,

     learning_rate = 0.05,

     max_depth = 6,

     min_child_weight = 1,

     n_estimators = 1000,

     nthread = 4,

     objective = 'binary:logistic',

     seed = 1337,

     subsample = 0.8,

     tree_method= 'gpu_hist')

xgb.fit(Non_scaled_x_train, y_train)



# Hyper Parameter tuned

# Logistic Regression

lr = LogisticRegression(C = 10.0, penalty = 'l2')

lr.fit(X_train, y_train)



# Hyper Parameter tuned

# Logistic Regression

k_svm = SVC(C = 1000, gamma = 0.01, kernel = 'rbf')

k_svm.fit(X_train, y_train)



# Hyper Parameter tuned

# KNN

knn = KNeighborsClassifier(n_neighbors = 6, weights = 'uniform')

knn.fit(X_train, y_train)



# Hyper Parameter tuned

# Random Forest

rf = RandomForestClassifier(criterion = 'gini', min_samples_leaf = 1, min_samples_split = 12, n_estimators = 50)

rf.fit(X_train, y_train)
# Ensemble of Models Hard Voting



from statistics import mode

def predict_surival(df, non_scaled_df):

    xgb_pred = xgb.predict(non_scaled_df)

    lr_pred = lr.predict(df)

    K_svm_pred = k_svm.predict(df)

    knn_pred = knn.predict(df)

    rf_pred = rf.predict(df)

    

#     yTest = np.array([])

#     for i in range(0,len(df)):

#         yTest = np.append(yTest, mode([xgb_pred[i], lr_pred[i], K_svm_pred[i], knn_pred[i], rf_pred[i]]))

#     return yTest.astype(int)



    yTest = np.array([])

    for i in range(0,len(df)):

        yTest = np.append(yTest, mode([xgb_pred[i], rf_pred[i], knn_pred[i]]))

    return yTest.astype(int)
# from sklearn.metrics import classification_report

# # classification report for precision, recall f1-score and accuracy

# matrix = classification_report(yTest,y_test,labels=[1,0])

# print('Classification report : \n',matrix)
# Making Submission



# Preparing test data 

test['Title'] = test['Title'].replace('Dona.', 'Mrs.')

test.head()
titanic_test = test.iloc[:, 0:11].values



# Taking care of categorical data



titanic_test[:, 1] = label_encoder_sex_tr.transform(titanic_test[:, 1])

titanic_test[:, 5] = label_encoder_title_tr.transform(titanic_test[:, 5])

titanic_test[:, 4] = label_encoder_embarked_tr.transform(titanic_test[:, 4])



# Feature Scaling

ns_titanic_test = titanic_test

titanic_test = scaler_x.transform(titanic_test)

y_pred = predict_surival(titanic_test, ns_titanic_test)
# y_pred = rf.predict(titanic_test)

print(type(y_pred[0]))
titanic_submission = pd.DataFrame({'PassengerId':passengerId, 'Survived':y_pred})

titanic_submission.to_csv('submission-esm2.csv', index=False)
len(titanic_submission)