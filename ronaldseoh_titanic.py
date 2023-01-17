import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree, linear_model, ensemble, neural_network, svm
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import shuffle
from graphviz import Source
from IPython.display import Image
train_data = pd.read_csv('../input/train.csv')
train_data = shuffle(train_data, random_state=0)
train_data.head()
train_data.info()
# Select the columns I'd like to use
X = train_data[['Pclass', 'Sex', 'Age', 'Name', 'Parch', 'SibSp', 'Embarked']].copy()
# Since Pclass represent boarding classes, it should be treated as a categorical variable
X.Pclass = X.Pclass.astype('category')
# Simply dropping NaN values for Age does not seem like a good idea. Let's try replacing them
# with mean values.
#Y.drop(X[pd.isna(X.Age)].index, inplace=True)
#X.drop(X[pd.isna(X.Age)].index, inplace=True)
X.Age = X.Age.fillna(X.Age.mean())
# Hmm, how about extracting family names from the list and convert them into integers?
X.Name = X.Name.str.lower()

# NOTE: There should parentheses that group the whole regular expression together 
# in order to make pandas recognize it
X.Name = X.Name.str.extract(r'(.+(?=,))')

# Converter function
def convert_word_to_int(word):
    result = '';
    
    for i, character in enumerate(word):
        if i >= 6:
            break
        else:
            result += str(ord(character))
    
    return int(result)

X.Name = X.Name.map(convert_word_to_int)
# Fill NaN values for 'Embarked' with the most frequent value
X.Embarked = X.Embarked.fillna(X.Embarked.mode())
# Convert categorical variables into dummy variables
X = pd.get_dummies(X, dummy_na=False)
# Min-Max Scaling
min_max_scaler = MinMaxScaler()
X[['Age', 'Name', 'Parch', 'SibSp']] = min_max_scaler.fit_transform(X[['Age', 'Name', 'Parch', 'SibSp']])
# Total Number of Data Points
print(len(X))
X.head()
# Get the dependent variable - or just Y
Y = train_data[['Survived']].copy()
Y.head()
Y = Y.values.ravel()
submission_data = pd.read_csv('../input/test.csv')
submission_data.head()
# Need to do the same preprocessing done previously on the training data

X_submission = submission_data[['Pclass', 'Sex', 'Age', 'Name', 'Parch', 'SibSp', 'Embarked']].copy()

X_submission.Pclass = X_submission.Pclass.astype('category')

# Simply dropping NaN values for Age does not seem like a good idea. Let's try replacing them
# with mean values.
X_submission.Age = X_submission.Age.fillna(X_submission.Age.mean())

# Hmm, how about extracting family names from the list and convert them into integers?
X_submission.Name = X_submission.Name.str.lower()
# NOTE: There should parentheses that group the whole regular expression together in order to make pandas recognize it
X_submission.Name = X_submission.Name.str.extract(r'(.+(?=,))')

# Converter function
def convert_word_to_int(word):
    result = '';
    
    for i, character in enumerate(word):
        if i >= 6:
            break
        else:
            result += str(ord(character))
    
    return int(result)

X_submission.Name = X_submission.Name.map(convert_word_to_int)

# Fill NaN values for 'Embarked' with the most frequent value
X_submission.Embarked = X_submission.Embarked.fillna(X_submission.Embarked.mode())

# Dummy Variables
X_submission = pd.get_dummies(X_submission, dummy_na=False)

# Min-Max Scaling
X_submission[['Age', 'Name', 'Parch', 'SibSp']] = min_max_scaler.fit_transform(X_submission[['Age', 'Name', 'Parch', 'SibSp']])
X_submission.head()
# Spit the train data for training and testing (because test.csv does not contain answers)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
# lines adapted from https://stackoverflow.com/questions/31161637/grid-search-cross-validation-in-sklearn#31162095

decision_tree_param_grid = {
    'min_samples_leaf': np.linspace(0.01, 0.5),
    'max_depth': np.arange(2, 11)
}

decision_tree_grid_search = GridSearchCV(tree.DecisionTreeClassifier(), decision_tree_param_grid)

decision_tree_grid_search.fit(X_train, Y_train)
decision_tree_grid_search_preds = decision_tree_grid_search.predict_proba(X_train)[:, 1]
decision_tree_grid_search_performance = roc_auc_score(Y_train, decision_tree_grid_search_preds)

print('DecisionTree: Area under the ROC curve = {}'.format(decision_tree_grid_search_performance))
print(decision_tree_grid_search.best_params_)
decision_tree_classifier = tree.DecisionTreeClassifier(
    min_samples_leaf=decision_tree_grid_search.best_params_['min_samples_leaf'],
    max_depth=decision_tree_grid_search.best_params_['max_depth']
)
decision_tree_classifier = decision_tree_classifier.fit(X_train, Y_train)
graph = Source( tree.export_graphviz(decision_tree_classifier, out_file=None, feature_names=X_train.columns))

png_bytes = graph.pipe(format='png')

with open('decison_tree_pipe.png','wb') as f:
    f.write(png_bytes)

Image(png_bytes)
decison_tree_cross_validation_scores = cross_val_score(decision_tree_classifier, X_train, Y_train, cv=5)
print(decison_tree_cross_validation_scores)
decision_tree_predictions = decision_tree_classifier.predict(X_test)
print(accuracy_score(Y_test, decision_tree_predictions))
decision_tree_submission_prediction = decision_tree_classifier.predict(X_submission)
decision_tree_submission_prediction
with open('submission_decisiontree.csv', 'w') as submission_file:
    submission_file.writelines('PassengerId,Survived\n')
    
    for i in range(len(decision_tree_submission_prediction)):
        newline_string = str(submission_data['PassengerId'].iloc[i]) + ',' + str(decision_tree_submission_prediction[i]) + '\n'
        submission_file.writelines(newline_string)
logistic_regression_classifier = linear_model.LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', fit_intercept=True)
logistic_regression_classifier = logistic_regression_classifier.fit(X_train, Y_train)
logistic_predictions = logistic_regression_classifier.predict(X_test)
print(accuracy_score(Y_test, logistic_predictions))
logistic_cross_validation_scores = cross_val_score(logistic_regression_classifier, X_train, Y_train, cv=5)
print(logistic_cross_validation_scores)
logistic_submission_prediction = logistic_regression_classifier.predict(X_submission)
with open('submission_logistic.csv', 'w') as submission_file:
    submission_file.writelines('PassengerId,Survived\n')
    
    for i in range(len(logistic_submission_prediction)):
        newline_string = str(submission_data['PassengerId'].iloc[i]) + ',' + str(logistic_submission_prediction[i]) + '\n'
        submission_file.writelines(newline_string)
# lines adapted from https://stackoverflow.com/questions/31161637/grid-search-cross-validation-in-sklearn#31162095

gradient_boosting_param_grid = {
    #'min_samples_leaf': np.linspace(0.01, 0.5),
    'max_depth': np.arange(2, 11),
    'n_estimators': np.arange(10, 26)
}

gradient_boosting_grid_search = GridSearchCV(ensemble.GradientBoostingClassifier(), gradient_boosting_param_grid)

gradient_boosting_grid_search.fit(X_train, Y_train)
gradient_boosting_grid_search_preds = gradient_boosting_grid_search.predict_proba(X_train)[:, 1]
gradient_boosting_grid_search_performance = roc_auc_score(Y_train, gradient_boosting_grid_search_preds)

print('Gradient Boosting: Area under the ROC curve = {}'.format(gradient_boosting_grid_search_performance))
print(gradient_boosting_grid_search.best_params_)
gradient_boosting_classifier = ensemble.GradientBoostingClassifier(
    #min_samples_leaf=gradient_boosting_grid_search.best_params_['min_samples_leaf'],
    n_estimators=10,
    max_depth=3
)
gradient_boosting_classifier = gradient_boosting_classifier.fit(X_train, Y_train)

gradient_boosting_predictions = gradient_boosting_classifier.predict(X_test)
print(accuracy_score(Y_test, gradient_boosting_predictions))

gradient_boosting_cross_validation_scores = cross_val_score(gradient_boosting_classifier, X_train, Y_train, cv=5)
print(gradient_boosting_cross_validation_scores)

gradient_boosting_submission_prediction = gradient_boosting_classifier.predict(X_submission)
with open('submission_gradient_boosting.csv', 'w') as submission_file:
    submission_file.writelines('PassengerId,Survived\n')
    
    for i in range(len(gradient_boosting_submission_prediction)):
        newline_string = str(submission_data['PassengerId'].iloc[i]) + ',' + str(gradient_boosting_submission_prediction[i]) + '\n'
        submission_file.writelines(newline_string)
# lines adapted from https://stackoverflow.com/questions/31161637/grid-search-cross-validation-in-sklearn#31162095

adaboost_param_grid = {
    'n_estimators': np.arange(10, 26)
}

adaboost_grid_search = GridSearchCV(ensemble.AdaBoostClassifier(), adaboost_param_grid)

adaboost_grid_search.fit(X_train, Y_train)
adaboost_grid_search_preds = adaboost_grid_search.predict_proba(X_train)[:, 1]
adaboost_grid_search_performance = roc_auc_score(Y_train, adaboost_grid_search_preds)

print('AdaBoost: Area under the ROC curve = {}'.format(adaboost_grid_search_performance))
print(adaboost_grid_search.best_params_)
adaboost_classifier = ensemble.AdaBoostClassifier(n_estimators=12)
adaboost_classifier = adaboost_classifier.fit(X_train, Y_train)
adaboost_predictions = adaboost_classifier.predict(X_test)
print(accuracy_score(Y_test, adaboost_predictions))

adaboost_cross_validation_scores = cross_val_score(adaboost_classifier, X_train, Y_train, cv=5)
print(adaboost_cross_validation_scores)

adaboost_submission_prediction = adaboost_classifier.predict(X_submission)
with open('submission_adaboost.csv', 'w') as submission_file:
    submission_file.writelines('PassengerId,Survived\n')
    
    for i in range(len(adaboost_submission_prediction)):
        newline_string = str(submission_data['PassengerId'].iloc[i]) + ',' + str(adaboost_submission_prediction[i]) + '\n'
        submission_file.writelines(newline_string)
# lines adapted from https://stackoverflow.com/questions/31161637/grid-search-cross-validation-in-sklearn#31162095

random_forest_param_grid = {
    #'min_samples_leaf': np.linspace(0.01, 0.5),
    'max_depth': np.arange(2, 11),
    'n_estimators': np.arange(10, 26)
}
random_forest_grid_search = GridSearchCV(ensemble.RandomForestClassifier(), random_forest_param_grid)

random_forest_grid_search.fit(X_train, Y_train)
random_forest_grid_search_preds = random_forest_grid_search.predict_proba(X_train)[:, 1]
random_forest_grid_search_performance = roc_auc_score(Y_train, random_forest_grid_search_preds)

print('Random Forest: Area under the ROC curve = {}'.format(random_forest_grid_search_performance))
print(random_forest_grid_search.best_params_)
random_forest_classifier = ensemble.RandomForestClassifier(
    #min_samples_leaf=random_forest_grid_search.best_params_['min_samples_leaf'],
    n_estimators=13,
    max_depth=3
)

random_forest_classifier = random_forest_classifier.fit(X_train, Y_train)
random_forest_predictions = random_forest_classifier.predict(X_test)
print(accuracy_score(Y_test, random_forest_predictions))
random_forest_submission_prediction = random_forest_classifier.predict(X_submission)
with open('submission_random_forest.csv', 'w') as submission_file:
    submission_file.writelines('PassengerId,Survived\n')
    
    for i in range(len(random_forest_submission_prediction)):
        newline_string = str(submission_data['PassengerId'].iloc[i]) + ',' + str(random_forest_submission_prediction[i]) + '\n'
        submission_file.writelines(newline_string)
mlp_classifier = neural_network.MLPClassifier(
    learning_rate='adaptive',
    solver='adam',
    #early_stopping=True,
    verbose=True,
    shuffle=True
)
mlp_classifier = mlp_classifier.fit(X_train, Y_train)
mlp_predictions = mlp_classifier.predict(X_test)
print(accuracy_score(Y_test, mlp_predictions))
mlp_submission_prediction = mlp_classifier.predict(X_submission)
with open('submission_mlp.csv', 'w') as submission_file:
    submission_file.writelines('PassengerId,Survived\n')
    
    for i in range(len(mlp_submission_prediction)):
        newline_string = str(submission_data['PassengerId'].iloc[i]) + ',' + str(mlp_submission_prediction[i]) + '\n'
        submission_file.writelines(newline_string)
svm_classifier = svm.SVC(kernel='linear')
svm_classifier = svm_classifier.fit(X_train, Y_train)
svm_predictions = svm_classifier.predict(X_test)
print(accuracy_score(Y_test, svm_predictions))
svm_submission_prediction = svm_classifier.predict(X_submission)