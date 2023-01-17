# Import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn import svm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from keras.utils import to_categorical
# Read data
train_raw = pd.read_csv('/kaggle/input/titanic/train.csv')
test_raw = pd.read_csv('/kaggle/input/titanic/test.csv')
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
# Check data shape
print('Train shape: ', train_raw.shape)
print('Test shape: ', test_raw.shape)
# Explore data
train_raw.info()
test_raw.info()
# View training data
train_raw.head()
# Separate numeric and string(categorical, ordinal) features from test and train data
def separate_cols(df):
    '''
    Returns the numeric and string feature names present in a dataframe
    
    Parameters:
        df(Dataframe): The Dataframe from which the feature names are to be separated
    
    Returns:
        List: A list which contains two lists: numeric feature name list & string feature name list
    '''
    numeric_cols = []
    string_cols = []
    for col in df.columns:
        if(df[col].dtype in ['int','float']):
            numeric_cols.append(col)
        elif(df[col].dtype in ['object']):
            string_cols.append(col)
    return [numeric_cols,string_cols]

numeric_cols_train, string_cols_train = separate_cols(train_raw)
numeric_cols_test, string_cols_test = separate_cols(test_raw)
# Check the number of missing values for each feature of train and test data
def print_missing_values(df):
    '''
    Prints the feature name along with the number of missing values and missing value percentage in the dataframe
    
    Parameters:
        df(Dataframe): The Dataframe from which the missing values are to be printed
    '''
    for col in df.columns:
        if(df[col].isnull().sum() > 0):
            print(col, df[col].isnull().sum(), "%.2f%%" %((df[col].isnull().sum()/df.shape[0])*100))

print("Missing values in train data")
print_missing_values(train_raw)
print("\nMissing values in test data")
print_missing_values(test_raw)
# Copy raw data for further processing
train = train_raw.copy()
test = test_raw.copy()
# Drop Name and PassengerID from train and test data as they have no contribution in making predictions
train = train.drop(["PassengerId","Name"], axis=1)
test = test.drop(["PassengerId","Name"], axis=1)
# We can observe from the missing value percentage that cabin has 
# more than 70% missing data in both train and test set. 
# We can remove Cabin feature from both train and test set
train = train.drop(['Cabin'], axis=1)
test = test.drop(['Cabin'], axis=1)
# Ticket feature is not much useful in prediction
train = train.drop(['Ticket'], axis=1)
test = test.drop(['Ticket'], axis=1)
# We can fill the missing age values based on the Pclass feature using the mean of age with a Pclass group
# age_mean = train[['Age','Pclass']].groupby(['Pclass'], as_index=False).mean()['Age']
# train.loc[train['Pclass'] == 1, 'Age'].fillna(int(age_mean[0]))
# Replace all missing values with mean or mode in train data
train['Age'] = train['Age'].fillna(train['Age'].mean())
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])
# Replace all missing values with mean or mode in test data
test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
# Intermediate copy
train1 = train.copy()
test1 = test.copy()
# Update columns
numeric_cols_train, string_cols_train = separate_cols(train1)
numeric_cols_test, string_cols_test = separate_cols(test1)
# Add parents and siblings to create a new feature
train1['Family_Size'] = train1['Parch'] + train1['SibSp']
test1['Family_Size'] = test1['Parch'] + test1['SibSp']
# Drop Parch and SibSp because we have a new feature named Family_size
train1 = train1.drop(columns=['SibSp', 'Parch'], axis=1)
test1 = test1.drop(columns=['SibSp', 'Parch'], axis=1)
# Test 1 - Does not perform well
# Use dummies to onehotencode categorical features
# Uncomment when required

# dum_train1 = pd.get_dummies(train1[string_cols_train], columns=string_cols_train)
# dum_test1 = pd.get_dummies(train1[string_cols_test], columns=string_cols_test)
# train1 = train1.join(dum_train1)
# test1 = test1.join(dum_test1)
# Drop Embarked and Sex from train and test data
# train1 = train1.drop(columns=['Sex', 'Embarked'], axis=1)
# test1 = test1.drop(columns=['Sex', 'Embarked'], axis=1)
# Test 2
# Use labelencoder for categorical features
le = LabelEncoder()
for col in string_cols_train:
    train1[col] = le.fit_transform(train1[col])
    test1[col] = le.transform(test1[col])
# Scale Age and Fare for train and test data using MinMaxScaler
scaler = MinMaxScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train1[['Age', 'Fare']]),columns=['Age', 'Fare'])
test_scaled = pd.DataFrame(scaler.transform(test1[['Age', 'Fare']]),columns=['Age', 'Fare'])
# Copy the scaled features to processed dataset
train1[['Age', 'Fare']] = train_scaled[['Age', 'Fare']]
test1[['Age', 'Fare']] = test_scaled[['Age', 'Fare']]
# Prepare final training and test data for modeling
X_train = train1.drop(['Survived'], axis=1).copy()
y_train = train1['Survived']
X_test = test1.copy()
# Using the sample submissions as y_test just to get an idea of how models perform
# A better approach might be to use train test split on train data
y_test = submission['Survived']
X_test.head()
# Check the shape of final data
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
def nn_model(optimizer='adam'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

nn_estimator = KerasClassifier(build_fn=nn_model)
nn_model().summary()
# Create hyperparameter space
epochs = [100, 200, 300]
batches = [16, 32, 64]
optimizers = ['rmsprop', 'adam', 'Nadam']

# Create hyperparameter options
hyperparameters = dict(optimizer=optimizers, epochs=epochs, batch_size=batches)

gscv = GridSearchCV(estimator=nn_estimator, param_grid=hyperparameters, n_jobs=-1,cv=3, verbose=1)
grid_result = gscv.fit(X_train, y_train)
# View hyperparameters of best neural network
print(grid_result.best_params_)
print(grid_result.best_score_)
# Check the train and test score using gridsearchcv score
nn_train_score = gscv.score(X_train, y_train) 
nn_test_score = gscv.score(X_test, y_test)
print("Train accuracy: ", nn_train_score)
print("Test accuracy: ", nn_test_score)
# Apply best_params on nn model and get history object
model = nn_model(grid_result.best_params_['optimizer'])
history = model.fit(X_train, y_train, 
                    epochs=grid_result.best_params_['epochs'],
                    batch_size=grid_result.best_params_['batch_size'],
                    verbose=1)
# Check the score of train and test using sequential model evaluate
nn_train_score = model.evaluate(X_train, y_train)[1]
nn_test_score = model.evaluate(X_test, y_test)[1]
print("Train accuracy: ", nn_train_score)
print("Test accuracy: ", nn_test_score)
# Make predictions using neural network model
nn_train_prediction = model.predict(X_train)
nn_test_prediction = model.predict(X_test)
# Convert probability prediction to binary
for i in range(0, len(nn_train_prediction)):
    if(nn_train_prediction[i] > 0.5):
        nn_train_prediction[i] = 1
    else:
        nn_train_prediction[i] = 0
for i in range(0, len(nn_test_prediction)):
    if(nn_test_prediction[i] > 0.5):
        nn_test_prediction[i] = 1
    else:
        nn_test_prediction[i] = 0
# Roc is a good measure for classification problems
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, nn_test_prediction)
nn_roc_auc = auc(false_positive_rate, true_positive_rate)
# Plot confusion matrix for neural network predictions
cm_train = confusion_matrix(y_train, nn_train_prediction)
cr_train = classification_report(y_train, nn_train_prediction)
print("Training Data:")
print(cm_train)
print(cr_train)
cm_test = confusion_matrix(y_test, nn_test_prediction)
cr_test = classification_report(y_test, nn_test_prediction)
print("Testing Data:")
print(cm_test)
print(cr_test)
# plot history
pyplot.plot(history.history['accuracy'], label='train')
pyplot.xlabel('Epochs')
pyplot.ylabel('Accuracy')
pyplot.legend()
pyplot.show()
pyplot.plot(history.history['loss'], label='train')
pyplot.xlabel('Epochs')
pyplot.ylabel('Loss')
pyplot.legend()
pyplot.show()
lr = LogisticRegression(C=5)
lr.fit(X_train, y_train)

lr_train_prediction = lr.predict(X_train)
lr_test_prediction = lr.predict(X_test)
lr_train_score = lr.score(X_train, y_train)
lr_test_score = lr.score(X_test, y_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, lr_test_prediction)
lr_roc_auc = auc(false_positive_rate, true_positive_rate)
rc = RidgeClassifier(alpha=100)
rc.fit(X_train, y_train)

rc_train_prediction = rc.predict(X_train)
rc_test_prediction = rc.predict(X_test)
rc_train_score = rc.score(X_train, y_train)
rc_test_score = rc.score(X_test, y_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, rc_test_prediction)
rc_roc_auc = auc(false_positive_rate, true_positive_rate)
rfc = RandomForestClassifier(n_estimators=1000, max_depth=8, n_jobs=-1)
rfc.fit(X_train, y_train)

rfc_train_prediction = rfc.predict(X_train)
rfc_test_prediction = rfc.predict(X_test)
rfc_train_score = rfc.score(X_train, y_train)
rfc_test_score = rfc.score(X_test, y_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, rfc_test_prediction)
rfc_roc_auc = auc(false_positive_rate, true_positive_rate)
svc = svm.SVC(C=25, gamma=25)
svc.fit(X_train, y_train)

svc_train_prediction = svc.predict(X_train)
svc_test_prediction = svc.predict(X_test)
svc_train_score = svc.score(X_train, y_train)
svc_test_score = svc.score(X_test, y_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, svc_test_prediction)
svc_roc_auc = auc(false_positive_rate, true_positive_rate)
gbc = GradientBoostingClassifier(n_estimators=125)
gbc.fit(X_train, y_train)

gbc_train_prediction = gbc.predict(X_train)
gbc_test_prediction = gbc.predict(X_test)
gbc_train_score = gbc.score(X_train, y_train)
gbc_test_score = gbc.score(X_test, y_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, gbc_test_prediction)
gbc_roc_auc = auc(false_positive_rate, true_positive_rate)
# Voting Classifier with hard voting 
estimator = [('lr',lr),('rc',rc),('rfr',rfc),('svc',svc),('gbc',gbc)]
vhc = VotingClassifier(estimators=estimator, voting='hard') 
vhc.fit(X_train, y_train)

vhc_train_prediction = vhc.predict(X_train)
vhc_test_prediction = vhc.predict(X_test)
vhc_train_score = vhc.score(X_train, y_train)
vhc_test_score = vhc.score(X_test, y_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, vhc_test_prediction)
vhc_roc_auc = auc(false_positive_rate, true_positive_rate)
# Voting Classifier with soft voting 
estimator = [('lr',lr),('rfr',rfc),('gbc',gbc)]
vsc = VotingClassifier(estimators=estimator, voting='soft') 
vsc.fit(X_train, y_train)

vsc_train_prediction = vsc.predict(X_train)
vsc_test_prediction = vsc.predict(X_test)
vsc_train_score = vsc.score(X_train, y_train)
vsc_test_score = vsc.score(X_test, y_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, vsc_test_prediction)
vsc_roc_auc = auc(false_positive_rate, true_positive_rate)
# Compare classifiers
columns = ['Classifier', 'Train Score', 'Test Score', 'AUC Score']
pd.DataFrame([
    ['Neural Network',nn_train_score, nn_test_score, nn_roc_auc],
    ['Logistic',lr_train_score, lr_test_score, lr_roc_auc],    
    ['Ridge',rc_train_score, rc_test_score, rc_roc_auc],    
    ['Random Forest',rfc_train_score, rfc_test_score, rfc_roc_auc],
    ['SVM',svc_train_score, svc_test_score, svc_roc_auc],
    ['Gradient Boosting Classifier',gbc_train_score, gbc_test_score, gbc_roc_auc],
    ['Hard Voting Classifier',vhc_train_score, vhc_test_score, vhc_roc_auc],
    ['Soft Voting Classifier',vsc_train_score, vsc_test_score, vsc_roc_auc],
], columns=columns
)
# Voting Hard classifier and Gradient Boosting performed better
submission['Survived'] = vhc_test_prediction.astype("int64")
submission
submission.to_csv("../working/TitanicSubmission.csv", index=False)