import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVR, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import RFECV

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, classification_report
# import dataset

df = pd.read_csv("../input/studentPor.csv")
# for extracting the names of features that are selected during the feature selection process

def get_feature_names(feature_selector, dataframe, target):

    feature_names = dataframe.drop(target, axis=1).columns

    selected_features = feature_names[feature_selector.get_support()].tolist()

    return selected_features



# for plotting a confusion matrix as a heatmap

def plot_confusion_matrix_heatmap(true_values, pred_values):

    pred_values = pd.Series(pred_values)

    array = confusion_matrix(true_values, pred_values)

    df_cm = pd.DataFrame(array, index = [i for i in "ABCDEF"], columns = [i for i in "ABCDEF"])

    plt.figure(figsize = (5,4))

    ax = sns.heatmap(df_cm, annot=True, annot_kws={"size": 13}, cmap="Reds", fmt='g', cbar=False)

    ax.xaxis.tick_top()

    ax.xaxis.set_label_position('top')

    ax.tick_params(length=0)

    plt.yticks(rotation=0)

    bottom, top = ax.get_ylim()

    ax.set_ylim(bottom + 0.5, top - 0.5)

    plt.xlabel('Predicted Values')

    plt.ylabel('True Values')
Q1 = df['absences'].quantile(0.25)

Q3 = df['absences'].quantile(0.75)

IQR = Q3 - Q1
df['iqr'] = (df['absences'] < (Q1 - 1.5 * IQR)) | (df['absences'] > (Q3 + 1.5 * IQR))



# a new df with outliers removed

df_data = df[df.iqr != True]

df_data = df_data.drop('iqr', axis = 1)



# reset the dataframe index

df_data = df_data.reset_index(drop=True)
df_dummies = pd.get_dummies(df_data, drop_first=True, columns=['school',

                                                               'sex',

                                                               'address',

                                                               'famsize',

                                                               'Pstatus',

                                                               'Mjob',

                                                               'Fjob',

                                                               'reason',

                                                               'guardian',

                                                               'schoolsup',

                                                               'famsup',

                                                               'paid',

                                                               'activities',

                                                               'nursery',

                                                               'higher',

                                                               'internet',

                                                               'romantic'])
# get the actual values for G3 and save to a numpy array

G3 = df_dummies["G3"].values

grade = []

# iterate through every G3 value and save the corresponding grade category to the grade array

for number in G3:

    if number >= 8:

        grade.append('pass') # pass

    else:

        grade.append('fail') # fail



# convert the grade array to a pandas dataframe

grade = pd.DataFrame(data=grade, columns=["grade"])

grade['grade'].value_counts()
grade = []

# iterate through every G3 value and save the corresponding grade category to the grade array

for number in G3:

    if number >= 17:

        grade.append('0') # grade = A

    elif number >= 15:

        grade.append('1') # grade = B

    elif number >= 13:

        grade.append('2') # grade = C

    elif number >= 11:

        grade.append('3') # grade = D

    elif number >= 8:

        grade.append('4') # grade = E

    else:

        grade.append('5') # grade = F



# convert the grade array to a pandas dataframe

grade = pd.DataFrame(data=grade, columns=["grade"])

# create a new dataframe called data by joining grade with df_dummies

data = pd.concat([df_dummies, grade], axis=1)

# remove the original numeric G3 column because it is not needed

data.drop(columns="G3", axis=1, inplace=True)

grade['grade'].value_counts()
# target attribute

target_attribute_name = "grade"

target = data[target_attribute_name]



# predictor attributes

predictors = data.drop(target_attribute_name, axis = 1).values
# assign a constant value to be used for random_state (otherwise a random number is chosen)

seed = 9
# prepare independent stratified data sets for training and testing of the models

predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, 

                                                                                target, 

                                                                                test_size=0.20, 

                                                                                shuffle=True, 

                                                                                stratify=target,

                                                                                random_state=seed)
# create a base classifier used to evaluate a subset of attributes

estimatorSVR = SVR(kernel="linear")

# create the RFE model that incorporates k-fold cross validation where k = 5

selectorSVR = RFECV(estimatorSVR, cv=5)

# use the model on the training data to select the best features for prediction

selectorSVR = selectorSVR.fit(predictors_train, target_train)

# transform the training data to only include the selected features

predictors_train_SVRselected = selectorSVR.transform(predictors_train)

# transform the test data to only include the selected features

predictors_test_SVRselected = selectorSVR.transform(predictors_test)

# print out the features that were selected

get_feature_names(selectorSVR, data, target_attribute_name)
# create a base classifier used to evaluate a subset of attributes

estimatorLR = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=6500)

# create the RFE model that incorporates k-fold cross validation where k = 5

selectorLR = RFECV(estimatorLR, cv=5)

# use the model on the training data to select the best features for prediction

selectorLR = selectorLR.fit(predictors_train, target_train)

# transform the training data to only include the selected features

predictors_train_LRselected = selectorLR.transform(predictors_train)

# transform the test data to only include the selected features

predictors_test_LRselected = selectorLR.transform(predictors_test)

# print out the features that were selected

get_feature_names(selectorLR, data, target_attribute_name)
# note how we set the random_state to the constant seed value

classifier = LinearSVC(random_state=seed)
pipe = Pipeline([('scaler', MinMaxScaler()), ('classifier', classifier)])
# create a list of differant parameters that will be passed into the classifier in each pipeline

params = {'classifier__C': [1000]}



# train final models using the pipeline and params (grid search with cross-validation is used)

modelSVR = GridSearchCV(pipe, params, cv=5, verbose=1, iid=False).fit(predictors_train_SVRselected, target_train)

modelLR = GridSearchCV(pipe, params, cv=5, verbose=1, iid=False).fit(predictors_train_LRselected, target_train)

modelAll = GridSearchCV(pipe, params, cv=5, verbose=1, iid=False).fit(predictors_train, target_train)
accuracy_SVR = modelSVR.best_score_

accuracy_LR = modelLR.best_score_

accuracy_All = modelAll.best_score_



scores = {'Classifier':['Linear SVC'],

          'SVR Features':[accuracy_SVR],

          'LR Features':[accuracy_LR],

          'All Features':[accuracy_All]}



# create dataframe to store all the accuracies

accuracies_df = pd.DataFrame(scores)

accuracies_df.round(2)
classifier = KNeighborsClassifier()
pipe = Pipeline([('scaler', MinMaxScaler()), ('classifier', classifier)])
params = {'classifier__n_neighbors': [7]}



# train final models using the pipeline and params (grid search with cross-validation is used)

modelSVR = GridSearchCV(pipe, params, cv=5, verbose=1, iid=False).fit(predictors_train_SVRselected, target_train)

modelLR = GridSearchCV(pipe, params, cv=5, verbose=1, iid=False).fit(predictors_train_LRselected, target_train)

modelAll = GridSearchCV(pipe, params, cv=5, verbose=1, iid=False).fit(predictors_train, target_train)
accuracy_SVR = modelSVR.best_score_

accuracy_LR = modelLR.best_score_

accuracy_All = modelAll.best_score_



# add accuracies to dataframe

accuracies_df.loc[1] = ['kNN', accuracy_SVR, accuracy_LR, accuracy_All]

accuracies_df.round(2)
# note how we set the random_state to the constant seed value

classifier = svm.SVC(gamma='scale', random_state=seed)
pipe = Pipeline([('scaler', MinMaxScaler()), ('classifier', classifier)])
params = {'classifier__C': [1000]}



# train final models using the pipeline and params (grid search with cross-validation is used)

modelSVR = GridSearchCV(pipe, params, cv=5, verbose=1, iid=False).fit(predictors_train_SVRselected, target_train)

modelLR = GridSearchCV(pipe, params, cv=5, verbose=1, iid=False).fit(predictors_train_LRselected, target_train)

modelAll = GridSearchCV(pipe, params, cv=5, verbose=1, iid=False).fit(predictors_train, target_train)
accuracy_SVR = modelSVR.best_score_

accuracy_LR = modelLR.best_score_

accuracy_All = modelAll.best_score_



# add accuracies to dataframe

accuracies_df.loc[2] = ['SVC', accuracy_SVR, accuracy_LR, accuracy_All]

accuracies_df.round(2)
# note how we set the random_state to the constant seed value

classifier = RandomForestClassifier(random_state=seed)
pipe = Pipeline([('scaler', MinMaxScaler()), ('classifier', classifier)])
params = {'classifier__n_estimators': [250],

          'classifier__max_depth': [4]}



# train final models using the pipeline and params (grid search with cross-validation is used)

modelSVR = GridSearchCV(pipe, params, cv=5, verbose=1, iid=False).fit(predictors_train_SVRselected, target_train)

modelLR = GridSearchCV(pipe, params, cv=5, verbose=1, iid=False).fit(predictors_train_LRselected, target_train)

modelAll = GridSearchCV(pipe, params, cv=5, verbose=1, iid=False).fit(predictors_train, target_train)
accuracy_SVR = modelSVR.best_score_

accuracy_LR = modelLR.best_score_

accuracy_All = modelAll.best_score_



# add accuracies to dataframe

accuracies_df.loc[3] = ['Ensemble (RF)', accuracy_SVR, accuracy_LR, accuracy_All]

accuracies_df.round(2)
classifier = RandomForestClassifier(random_state=seed)

pipe = Pipeline([('scaler', MinMaxScaler()), ('classifier', classifier)])

params = {'classifier__n_estimators': [250],

          'classifier__max_depth': [4]}

modelFinal = GridSearchCV(pipe, params, cv=5, verbose=1, iid=False).fit(predictors_train_LRselected, target_train)
accuracy_Final = modelFinal.score(predictors_test_LRselected, target_test)

print("The accuracy of our final predictive model is: %2d%%" %(accuracy_Final.round(2)*100))
plot_confusion_matrix_heatmap(target_test, modelFinal.best_estimator_.predict(predictors_test_LRselected))
print(classification_report(target_test, modelFinal.best_estimator_.predict(predictors_test_LRselected)))