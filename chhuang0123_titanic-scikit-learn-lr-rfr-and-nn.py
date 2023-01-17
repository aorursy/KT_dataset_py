#%% 1
# import data processing lib
import numpy as np
import pandas as pd

# configure data processing lib
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.3f}'.format

# import data visualization lib
from matplotlib import pyplot as plt

#%% 2
# import data set
dataframe = pd.read_csv("../input/train.csv", sep=",")
dataframe.head(10)

#%% 2 - 1
dataframe.info()

# Data columns (total 12 columns):
# int64 (5)
# PassengerId    891 non-null int64
# Survived       891 non-null int64
# Pclass         891 non-null int64
# SibSp          891 non-null int64
# Parch          891 non-null int64
# float64 (2)
# Age            714 non-null float64
# Fare           891 non-null float64
# object (5)
# Name           891 non-null object
# Sex            891 non-null object
# Ticket         891 non-null object
# Cabin          204 non-null object
# Embarked       889 non-null object

#%% 2 - 2
# display target
target = 'Survived'
print("unique count:\n{}\n".format(dataframe[target].value_counts()))
plt.hist(dataframe[target])
plt.show()

#%% 2 - 3
# display null value summary
print('null value summary:')
dataframe.isnull().sum().sort_values(ascending=False)

# Cabin          687
# Age            177
# Embarked         2

#%% 3
def display_column(columns, chart=''):
    for column_name in columns:
        # summary
        print("cloumn name: {}\n".format(column_name))
        print("has null: {}\n".format(dataframe[column_name].isnull().any()))
        print("null count: {}\n".format(dataframe[column_name].isnull().sum()))
        print("unique:\n{}\n".format(dataframe[column_name].unique()))
        print("unique count:\n{}\n".format(dataframe[column_name].value_counts()))
        print("Description:\n{}\n".format(dataframe[column_name].describe()))

        if chart == 'hist':
            # vs target column visualization
            plt.scatter(dataframe[column_name], dataframe[target])
            plt.xlabel(column_name)
            plt.ylabel(target)
            plt.title("{} VS {}".format(column_name, target))
            plt.show()
        elif chart == 'bar':
            # visulation
            dataframe[column_name].value_counts().plot.bar()
            plt.title(column_name)
            plt.show()

#%% 3 - 1
columns = ["Age", "Pclass", "SibSp", "Parch"]
display_column(columns)

#%% 3 - 2
columns = ["Sex", "Embarked"]
display_column(columns, chart='bar')

#%% 4
dataframe = dataframe.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare'])

#%% 5
def transform_sex(dataframe):
    # transform
    # male: 0
    # female: 1
    column_name = 'Sex'
    genders = {"female": 0, "male": 1}

    return dataframe[column_name].map(genders)

#%% 5 - 1
column_name = 'Sex'
dataframe[column_name] = transform_sex(dataframe)
dataframe.head(10)

#%% 5 - 2
display_column([column_name], chart='bar')
#%% 6
def transform_embarked(dataframe):
    # One Hot Encoder
    # S    644
    # C    168
    # Q     77
    column_name = 'Embarked'
    dataframe[column_name].fillna('S')
    dataframe = pd.concat(
        [
            dataframe,
            pd.get_dummies(dataframe[column_name], prefix=column_name)
        ],
        axis=1)

    dataframe = dataframe.drop(columns=[column_name, 'Embarked_Q'])

    return dataframe

#%% 6 - 1
column_name = 'Embarked'
dataframe = transform_embarked(dataframe)
dataframe.head(10)

#%% 7
def transform_age(dataframe):
    column_name = 'Age'

    # replace na with mean
    column_mean = dataframe[column_name].median()
    dataframe[column_name] = dataframe[column_name].fillna(column_mean)

    # binning
    dataframe.loc[dataframe[column_name] < 11, column_name] = 0
    dataframe.loc[(dataframe[column_name] >= 11) & (dataframe[column_name] < 21), column_name] = 1
    dataframe.loc[(dataframe[column_name] >= 21) & (dataframe[column_name] < 31), column_name] = 2
    dataframe.loc[(dataframe[column_name] >= 31) & (dataframe[column_name] < 41), column_name] = 3
    dataframe.loc[(dataframe[column_name] >= 41) & (dataframe[column_name] < 51), column_name] = 4
    dataframe.loc[(dataframe[column_name] >= 51) & (dataframe[column_name] < 61), column_name] = 5
    dataframe.loc[dataframe[column_name] >= 61, column_name] = 6

    dataframe[column_name] = dataframe[column_name].astype(int)

    return dataframe

#%% 7 - 1
column_name = 'Age'
dataframe = transform_age(dataframe)
dataframe.head(10)

#%% 7 - 2
display_column([column_name], chart='bar')

#%% 8
def transform_pclass(dataframe):
    # One Hot Encoder
    # 3    491
    # 1    216
    # 2    184
    column_name = 'Pclass'
    dataframe = pd.concat(
        [
            dataframe,
            pd.get_dummies(dataframe[column_name], prefix=column_name)
        ],
        axis=1)

    dataframe = dataframe.drop(columns=[column_name, 'Pclass_2'])

    return dataframe

#%% 8 - 1
column_name = 'Pclass'
dataframe = transform_pclass(dataframe)
dataframe.head(10)

#%% 9
def transform_relatives(dataframe):
    # cross feature
    column_name = 'Relatives'
    dataframe[column_name] = dataframe['SibSp'] + dataframe['Parch']
    dataframe = dataframe.drop(columns=['SibSp', 'Parch'])
    dataframe[column_name] = dataframe[column_name].astype(int)

    # remove outlier
    dataframe.loc[dataframe[column_name] >= 6, column_name] = 6

    return dataframe

#%% 9 - 1
column_name = 'Relatives'
dataframe = transform_relatives(dataframe)
dataframe.head(10)

#%% 9 - 2
display_column([column_name], chart='bar')

#%% 10
# display correlation
corr = dataframe.corr()
corr[target].sort_values(ascending=False)

# Pclass_1      0.286
# Embarked_C    0.168
# Relatives     0.040
# Age          -0.047
# Embarked_S   -0.156
# Pclass_3     -0.322
# Sex          -0.543
#%% 10 - 1
corr.style.background_gradient()

#%% 11
# define real dataset
minimal_features = [
    "Sex", "Pclass_3"
]
X = dataframe[minimal_features]
y = dataframe['Survived']

#%% 12
def preprocessing_data(source):

    # processing data
    columns = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare']
    target = source.drop(columns=columns)

    column_name = 'Sex'
    target[column_name] = transform_sex(target)

    column_name = 'Embarked'
    target = transform_embarked(target)

    column_name = 'Age'
    target = transform_age(target)

    column_name = 'Pclass'
    target = transform_pclass(target)

    column_name = 'Relatives'
    target = transform_relatives(target)

    return target

#%% 13
# start to train
from sklearn.linear_model import LinearRegression
print('LinearRegression: ')

# split into training and validation dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# train
linreg = LinearRegression()
linreg.fit(X_train, y_train)

#%% 14 
# start to predict
y_pred = linreg.predict(X_test)
y_pred
#%% 14 - 1
def transform_predicted(dataframe, threshold = 0.5):
    column_name = 'Predicted'

    dataframe.loc[dataframe[column_name] >= threshold, column_name] = 1
    dataframe.loc[dataframe[column_name] < threshold, column_name] = 0

    dataframe[column_name] = dataframe[column_name].astype(int)
    
    return dataframe
#%% 14 - 2
# transform
threshold = 0.5
y_pred2 = pd.DataFrame({'Predicted': y_pred})
y_pred2 = transform_predicted(y_pred2, threshold)
y_pred2.head(10)

#%% 14 - 3
y_test2 = pd.DataFrame({'Survived': y_test})
y_test2.head(10)

#%% 14 - 4
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test2['Survived'], y_pred2['Predicted'])
print(cm)

#%% 14 - 5
def display_cm_report(cm):
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TP = cm[1][1]

    print()
    print("TP: {}".format(TP))
    print("FP: {}".format(FP))
    print("FN: {}".format(FN))
    print("TN: {}\n".format(TN))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("accuracy: {}".format(accuracy))
    print("precision: {}".format(precision))
    print("recall: {}".format(recall))
    print("f1_score: {}\n".format(f1_score))

    # visualization
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    classNames = ['Negative', 'Positive']
    tick_marks = np.arange(len(classNames))

    plt.xticks(tick_marks, classNames)
    plt.yticks(tick_marks, classNames)

    s = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))
    plt.show()

#%% 14 - 6
display_cm_report(cm)

#%% 15
# import data
test_dataframe = pd.read_csv("../input/test.csv", sep=",")

#%% 15 - 1
# submission file
test_csv = pd.DataFrame()
test_csv['PassengerId'] = test_dataframe['PassengerId']

#%% 15 - 2
# processing data
test_dataframe = preprocessing_data(test_dataframe)
test_dataframe = test_dataframe[minimal_features]
test_dataframe.head(10)

#%% 16
# start to predict
y_pred = linreg.predict(test_dataframe)

# transform
threshold = 0.5
y_pred2 = pd.DataFrame({'Predicted': y_pred})
y_pred2 = transform_predicted(y_pred2, threshold)
y_pred2.head(10)

#%% 16 - 1
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
y_test = pd.read_csv("../input/gender_submission.csv", sep=",")
cm = confusion_matrix(y_test['Survived'], y_pred2['Predicted'])
print(cm)

#%% 16 - 3
display_cm_report(cm)

#%% 16 - 4
# submit file
test_csv['Survived'] = y_pred2['Predicted']
test_csv.to_csv('submission_all_lin_0.5.csv', index=False)

#%% 17
# tree base
from sklearn.ensemble import RandomForestClassifier
print('RandomForestClassifier: ')

# split into training and validation dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# train
forest_model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=1)
forest_model.fit(X_train, y_train)

#%% 17 - 1
# start to predict
y_pred = forest_model.predict(X_test)

#%% 17 - 2
# transform
threshold = 0.5
y_pred2 = pd.DataFrame({'Predicted': y_pred})
y_pred2 = transform_predicted(y_pred2)
y_pred2.head(10)

#%% 17 - 3
y_test2 = pd.DataFrame({'Survived': y_test})
y_test2.head(10)

#%% 17 - 4
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test2['Survived'], y_pred2['Predicted'])
print(cm)

#%% 17 - 5
display_cm_report(cm)

#%% 18
# start to predict
y_pred = forest_model.predict(test_dataframe)

# transform
threshold = 0.5
y_pred2 = pd.DataFrame({'Predicted': y_pred})
y_pred2 = transform_predicted(y_pred2)
y_pred2.head(10)

#%% 18 - 1
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
y_test = pd.read_csv("../input/gender_submission.csv", sep=",")
cm = confusion_matrix(y_test['Survived'], y_pred2['Predicted'])
print(cm)

#%% 18 - 2
display_cm_report(cm)

#%% 18 - 3
# submit file
test_csv['Survived'] = y_pred2['Predicted']
test_csv.to_csv('submission_all_tree_0.5.csv', index=False)

#%% 19
# NN
from sklearn.neural_network import MLPRegressor
print('MLPRegressor: ')

# split into training and validation dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# train
mlpr_model = MLPRegressor()
mlpr_model.fit(X_train, y_train)

#%% 19 - 1
# start to predict
y_pred = mlpr_model.predict(X_test)

#%% 19 - 2
# transform
threshold = 0.5
y_pred2 = pd.DataFrame({'Predicted': y_pred})
y_pred2 = transform_predicted(y_pred2)
y_pred2.head(10)

#%% 19 - 3
y_test2 = pd.DataFrame({'Survived': y_test})
y_test2.head(10)

#%% 19 - 4
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test2['Survived'], y_pred2['Predicted'])
print(cm)

#%% 19 - 5
display_cm_report(cm)

#%% 20
# start to predict
y_pred = mlpr_model.predict(test_dataframe)

# transform
threshold = 0.5
tansform_survived = np.vectorize(lambda x: 1 if x >= threshold else 0)
y_pred2 = tansform_survived(y_pred)
y_pred2 = pd.DataFrame({'Predicted': y_pred2})

#%% 20 - 1
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
y_test = pd.read_csv("../input/gender_submission.csv", sep=",")
cm = confusion_matrix(y_test['Survived'], y_pred2['Predicted'])
print(cm)

#%% 20 - 2
display_cm_report(cm)

#%% 20 - 3
# submit file
test_csv['Survived'] = y_pred2['Predicted']
test_csv.to_csv('submission_all_nn_0.5.csv', index=False)

#%% 21
from keras.models import Sequential
from keras.layers import Dense
print('Keras NN: ')

# init nn
nn = Sequential()

# add the first hidden layer
nn.add(
    Dense(
        units=3, kernel_initializer='uniform', activation='relu', input_dim=2))

# add output layer
nn.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# compile nn
nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# split into training and validation dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# start to train
nn.fit(X_train, y_train, batch_size=10, epochs=100)

#%% 21 - 1
# start to predict
y_pred = nn.predict(X_test)

#%% 21 - 2
# transform
threshold = 0.5
tansform_survived = np.vectorize(lambda x: 1 if x >= threshold else 0)
y_pred2 = tansform_survived(y_pred)
y_pred2 = pd.DataFrame({'Predicted': y_pred2[:,0]})
y_pred2.head(10)

#%% 21 - 3
y_test2 = pd.DataFrame({'Survived': y_test})
y_test2.head(10)

#%% 21 - 4
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test2['Survived'], y_pred2['Predicted'])
print(cm)

#%% 21 - 5
display_cm_report(cm)

#%% 22
# start to predict
y_pred = nn.predict(test_dataframe)

# transform
threshold = 0.5
tansform_survived = np.vectorize(lambda x: 1 if x >= threshold else 0)
y_pred2 = tansform_survived(y_pred)
y_pred2 = pd.DataFrame({'Predicted': y_pred2[:,0]})

#%% 22 - 1
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
y_test = pd.read_csv("../input/gender_submission.csv", sep=",")
cm = confusion_matrix(y_test['Survived'], y_pred2['Predicted'])
print(cm)

#%% 22 - 2
display_cm_report(cm)

#%% 22 - 3
# submit file
test_csv['Survived'] = y_pred2['Predicted']
test_csv.to_csv('submission_all_keras_nn_0.5.csv', index=False)
