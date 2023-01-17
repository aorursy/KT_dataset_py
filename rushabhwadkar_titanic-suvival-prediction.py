import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings("ignore",category=DeprecationWarning)



import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

import os

import tensorflow as tf

import re as re



from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB 

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier

from mlxtend.classifier import StackingCVClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier

from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer

print("# Imported the libaries")
dir_contents = os.listdir('../input/')

print("Dataset contents : {}".format(dir_contents))

print("Size of Training dataset : " + str(round(os.path.getsize('../input/train.csv')/(1024*2), 2)) +" KB")

print("Size of Testing dataset : " + str(round(os.path.getsize('../input/test.csv')/(1024*2), 2)) +" KB")
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



passengerIds = test_data['PassengerId']

print("Shape of Train dataset : {}".format(train_data.shape))

print("Shape of Test dataset : {}".format(test_data.shape))
train_data.head(4)
train_data['Survived'].value_counts(normalize=True) * 100
print(train_data['Age'].describe())

print("***************************************")



plt.close()

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)

plt.title("Distribution Plot for Age")

sns.distplot(train_data[train_data['Age'].notnull()]['Age'], bins=50)

plt.grid()  



plt.subplot(1, 3, 2)

plt.title("PDF-CDF Plot for Age")

counts, bin_edges = np.histogram(train_data[train_data['Age'].notnull()]['Age'], bins = 50, density = True)

pdf = counts / sum(counts)

cdf = np.cumsum(pdf)

plt.plot(bin_edges[1:], pdf, label="PDF")

plt.plot(bin_edges[1:], cdf, label="CDF")

plt.legend()

plt.grid()  



plt.subplot(1, 3, 3)

plt.title("Violin Plot for Age")

sns.violinplot(x="Survived", y="Age", data=train_data)



plt.grid()    

plt.show()
train_data[(train_data['Age'] < 10) & (train_data['SibSp'] == 0) & (train_data['Parch'] == 0)]
print("# Gender counts")

print(train_data['Sex'].value_counts())

print("*************************")

print("Percentage of Males who survived   : {}%".format(round(train_data[(train_data['Sex']=='male') & 

                                                               (train_data['Survived']==1)].shape[0] * 100 

                                                           / train_data.shape[0], 2)))

print("Percentage of Females who survived : {}%".format(round(train_data[(train_data['Sex']=='female') & 

                                                               (train_data['Survived']==1)].shape[0] * 100 

                                                           / train_data.shape[0], 2)))
full_data = [train_data , test_data]

len(full_data)
for data in full_data:

    print("**************************")

    print(data.info())

    print("**************************")
train_data['Name']
def getTitleFromName(nameText):

    title = str(nameText.split(', ')[1])

    title_search = re.search('([A-Za-z]+)\.', title).group(1)

    return title_search



for data in full_data:

    imp = SimpleImputer(missing_values=np.nan, strategy='median')

    

    # Introducting a new feature - Family Size

    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

    

    # Introducing a new feature - IsAlone

    data['IsAlone'] = 0

    data.loc[data['FamilySize']==1, 'IsAlone'] = 1

        

    # Introducing a new Feature - CabinAlloted

    data['CabinAllotment'] = 0

    data.loc[data['Cabin'].notnull(), 'CabinAllotment'] = 1

    

    # If data has null values in Embarked, just replace it with 'S' as 'S' is quite frequent

    data['Embarked'] = data['Embarked'].fillna('S')

    

    # Mapping Sex

    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}).astype(int)

    

    # Mapping Embarked

    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    

    # Since Age has many null/NA values, we will process it with Median values

    data['Age'] = imp.fit_transform(data[['Age', 'Sex', 'FamilySize']])[:,0].astype(int)

    

    # Since Fare has also some null/Na Values, we will process it with Median values

    data['Fare'] = imp.fit_transform(data[['Fare', 'FamilySize', 'CabinAllotment']])

    

    # Introducing a new feature - PerTicket

    data['PerTicket'] = data['Fare'] / data['FamilySize']

    data.loc[ data['PerTicket'] <= 7.25, 'PerTicket'] = 0

    data.loc[(data['PerTicket'] > 7.25) & (data['PerTicket'] <= 8.3), 'PerTicket'] = 1

    data.loc[(data['PerTicket'] > 8.3) & (data['PerTicket'] <= 23.667), 'PerTicket'] = 2

    data.loc[ data['PerTicket'] > 23.667, 'PerTicket'] = 3

    data['PerTicket'] = data['PerTicket'].astype(int)

    

    # Mapping the Age Values to Categories (Children, Youth, Adults, Senior)

    data.loc[(data['Age'] >=0) & (data['Age'] <= 14), 'Age'] = 0       #Children

    data.loc[(data['Age'] >=15) & (data['Age'] <= 24), 'Age'] = 1      #Youth

    data.loc[(data['Age'] >=25) & (data['Age'] <= 64), 'Age'] = 2      #Adults

    data.loc[data['Age'] >=65, 'Age'] = 3    #Senior

    data['Age'] = data['Age'].astype(int)

    

    #Name Feature Engineering

    data['Title'] = data['Name'].apply(getTitleFromName)

    data['Title'] = data['Title'].replace('Ms', 'Miss')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')

    data['Title'] = data['Title'].replace('Mme', 'Mrs')

    

    data['Status'] = "General"

    data.loc[data['Title'] == 'Capt','Status'] = 'Military'

    data.loc[data['Title'] == 'Col','Status'] = 'Military'

    data.loc[data['Title'] == 'Countess','Status'] = 'Political'

    data.loc[data['Title'] == 'Don','Status'] = 'Military'

    data.loc[data['Title'] == 'Dr','Status'] = 'General'

    data.loc[data['Title'] == 'Jonkheer','Status'] = 'Political'

    data.loc[data['Title'] == 'Lady','Status'] = 'Political'

    data.loc[data['Title'] == 'Major','Status'] = 'Military'

    data.loc[data['Title'] == 'Master','Status'] = 'General'

    data.loc[data['Title'] == 'Rev','Status'] = 'Political'

    data.loc[data['Title'] == 'Sir','Status'] = 'Military'

    

    data['Rank'] = 0

    data.loc[data['Title'] == 'Capt', 'Rank'] = 1

    data.loc[data['Title'] == 'Col', 'Rank'] = 1

    data.loc[data['Title'] == 'Major', 'Rank'] = 2

    data.loc[data['Title'] == 'Don', 'Rank'] = 2

    data.loc[data['Title'] == 'Sir', 'Rank'] = 0

    data.loc[data['Title'] == 'Dr', 'Rank'] = 1

    data.loc[data['Title'] == 'Master', 'Rank'] = 0

    data.loc[data['Title'] == 'Miss', 'Rank'] = 0

    data.loc[data['Title'] == 'Mr', 'Rank'] = 0

    data.loc[data['Title'] == 'Mrs', 'Rank'] = 0

    data.loc[data['Title'] == 'Countess', 'Rank'] = 2

    data.loc[data['Title'] == 'Jonkheer', 'Rank'] = 0

    data.loc[data['Title'] == 'Lady', 'Rank'] = 1

    data.loc[data['Title'] == 'Rev', 'Rank'] = 1

    

    data['Status'] = data['Status'].map({ 'General': 0, 'Military': 1, 'Political': 2})



# Feature Selection

remove_features = ['PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Name', 'Title', 'Fare']

train_data = train_data.drop(remove_features, axis=1)

test_data = test_data.drop(remove_features, axis=1)



train_data.head(5)
y_train = train_data['Survived'].values

X_train = train_data.drop('Survived', axis=1)

X_test = test_data.values
encoding_clf = OneHotEncoder()

train_data_new = encoding_clf.fit_transform(X_train).astype('int')

test_data_new = encoding_clf.transform(X_test).astype('int')
print(train_data_new.shape)

print(train_data_new.shape)
y_train_new = tf.keras.utils.to_categorical(y_train, num_classes=2)
colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train_data.astype(float).corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(35, input_shape=(35,), activation=tf.nn.relu, kernel_initializer='he_uniform'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu, kernel_initializer='he_uniform'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu,kernel_initializer='he_uniform'))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu,kernel_initializer='he_uniform'))

model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



relu_model = model.fit(train_data_new, y_train_new, epochs=500, batch_size=7, verbose=1, validation_split=0.20)
pred = model.predict_classes(test_data_new)

pred
DL_Submission = pd.DataFrame({ 'PassengerId': passengerIds,

                            'Survived': pred })

DL_Submission.to_csv("DL_Submission.csv", index=False)
# mask = ['Pclass', 'Sex', 'Age', 'Embarked', 'FamilySize', 'IsAlone', 'CabinAllotment', 'PerTicket', 'Status', 'Rank']

# X = train_data[mask]

# y = train_data['Survived']
# from sklearn.model_selection import GridSearchCV, train_test_split

# from sklearn.calibration import CalibratedClassifierCV

# from sklearn.metrics import roc_curve, f1_score, confusion_matrix, auc

# g_xtrain, g_xtest, g_ytrain, g_ytest = train_test_split(X, y, test_size=0.30)
from sklearn.linear_model import SGDClassifier



alpha_range = list([10** i for i in range(-10, 6, 1)] + [2**i for i in range(-5, -1, 1)])

loss_range = list(['hinge', 'log', 'modified_huber', 'squared_hinge'])

penalty_range = list(['l1', 'l2', 'elasticnet'])

parameters = {'loss': loss_range, 'penalty': penalty_range, 'alpha': alpha_range}

model = SGDClassifier()

g_clf = GridSearchCV(model, parameters, cv=10, n_jobs=-1, scoring="accuracy")

g_clf.fit(g_xtrain, g_ytrain)



print("Model fitted perfectly.")

print("Best Score (TRAIN): {}".format(g_clf.best_score_))

print("Best Params       : {}".format(g_clf.best_params_))
# optimal_alpha = g_clf.best_params_['alpha']

# optimal_loss = g_clf.best_params_['loss']

# optimal_penalty = g_clf.best_params_['penalty']



# clf_model = SGDClassifier(alpha=optimal_alpha, loss=optimal_loss, penalty=optimal_penalty)

# ccv_clf = CalibratedClassifierCV(clf_model, cv=10)

# ccv_clf.fit(g_xtrain, g_ytrain)



# # Get predicted values for test data

# pred_train = ccv_clf.predict(g_xtrain)

# pred_test = ccv_clf.predict(g_xtest)

# pred_proba_train = ccv_clf.predict_proba(g_xtrain)[:,1]

# pred_proba_test = ccv_clf.predict_proba(g_xtest)[:,1]



# fpr_train, tpr_train, thresholds_train = roc_curve(g_ytrain, pred_proba_train, pos_label=1)

# fpr_test, tpr_test, thresholds_test = roc_curve(g_ytest, pred_proba_test, pos_label=1)

# conf_mat_train = confusion_matrix(g_ytrain, pred_train, labels=[0, 1])

# conf_mat_test = confusion_matrix(g_ytest, pred_test, labels=[0, 1])

# f1_sc = f1_score(g_ytest, pred_test, average='binary', pos_label=1)

# auc_sc_train = auc(fpr_train, tpr_train)

# auc_sc = auc(fpr_test, tpr_test)



# print("Optimal Alpha: {} with Penalty: {} with AUC: {:.2f}%".format(optimal_alpha, optimal_penalty, float(auc_sc*100)))







# plt.figure(figsize=(13,7))

# # Plot ROC curve for training set

# plt.subplot(2, 2, 1)

# plt.title('Receiver Operating Characteristic - TRAIN SET')

# plt.plot(fpr_train, tpr_train, color='red', label='AUC - Train - {:.2f}'.format(float(auc_sc_train * 100)))

# plt.plot([0, 1], ls="--")

# plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

# plt.ylabel('True Positive Rate')

# plt.xlabel('False Positive Rate')

# plt.grid()

# plt.legend(loc='best')



# # Plot ROC curve for test set

# plt.subplot(2, 2, 2)

# plt.title('Receiver Operating Characteristic - TEST SET')

# plt.plot(fpr_test, tpr_test, color='blue', label='AUC - Test - {:.2f}'.format(float(auc_sc * 100)))

# plt.plot([0, 1], ls="--")

# plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")

# plt.ylabel('True Positive Rate')

# plt.xlabel('False Positive Rate')

# plt.grid()

# plt.legend(loc='best')



# #Plotting the confusion matrix for train

# plt.subplot(2, 2, 3)

# plt.title('Confusion Matrix for Training set')

# df_cm = pd.DataFrame(conf_mat_train, index = ["Negative", "Positive"],

#                   columns = ["Negative", "Positive"])

# sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')



# #Plotting the confusion matrix for test

# plt.subplot(2, 2, 4)

# plt.title('Confusion Matrix for Testing set')

# df_cm = pd.DataFrame(conf_mat_test, index = ["Negative", "Positive"],

#                   columns = ["Negative", "Positive"])

# sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')



# plt.tight_layout()

# plt.show()

# final_clf = SGDClassifier(alpha=optimal_alpha, loss=optimal_loss, penalty=optimal_penalty)

# final_clf.fit(X,y)



# pred = final_clf.predict(test_data)



# SGDSubmission = pd.DataFrame({ 'PassengerId': passengerIds,

#                             'Survived': pred })

# SGDSubmission.to_csv("SGDSubmission.csv", index=False)

# os.listdir()
# # Initializing models

# clf1 = SGDClassifier(n_jobs=-1)

# clf2 = GradientBoostingClassifier(n_estimators=50)

# clf3 = RandomForestClassifier(n_estimators=50, n_jobs=-1)

# clf4 = AdaBoostClassifier(n_estimators=50)

# meta_clf = XGBClassifier(n_estimators=100, n_jobs=-1)



# sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3, clf4], 

#                             meta_classifier=meta_clf)



# params = {

#             'sgdclassifier__alpha': alpha_range,

#             'sgdclassifier__loss': loss_range,

#             'sgdclassifier__penalty': penalty_range,

#             'randomforestclassifier__max_depth': list([5, 10, 15]),

#             'meta-xgbclassifier__max_depth': list([3, 7, 11])

#         }



# grid = GridSearchCV(estimator=sclf, 

#                     param_grid=params, 

#                     cv=10, n_jobs=-1, scoring='accuracy',

#                     refit=True)

# grid.fit(X, y)



# print('Best parameters: %s' % grid.best_params_)

# print('Accuracy: %.2f' % grid.best_score_)
# # Initializing models

# clf1 = KNeighborsClassifier()

# clf2 = RandomForestClassifier()

# clf3 = MultinomialNB()

# clf4 = SVC(kernel='rbf')

# clf5 = LogisticRegression()

# clf6 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME")

# meta_clf = XGBClassifier(n_jobs=-1)



# sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5, clf6], 

#                             meta_classifier=meta_clf)



# params = {

#             'kneighborsclassifier__n_neighbors': range(1, 50, 5),

#             'multinomialnb__alpha': [10**i for i in range(-5, 4, 1)],

#             'svc__C': [10**i for i in range(-5, 4, 1)],

#             'logisticregression__C': [10**i for i in range(-5, 4, 1)],

#         }



# grid = GridSearchCV(estimator=sclf, 

#                     param_grid=params, 

#                     cv=10, n_jobs=-1, scoring='accuracy',

#                     refit=True)

# grid.fit(X, y)



# print('Best parameters: %s' % grid.best_params_)

# print('Accuracy: %.2f' % grid.best_score_)
# #Predict from the Test Data

# pred = grid.predict(test_data)
# StackingSubmission = pd.DataFrame({ 'PassengerId': passengerIds,

#                             'Survived': pred })

# StackingSubmission.to_csv("StackingSubmission.csv", index=False)

# os.listdir()