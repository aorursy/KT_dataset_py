# data manipulation

import pandas as pd

import numpy as np



# sklearn helper functions

from sklearn.base import clone

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, cross_validate, GridSearchCV, cross_val_predict, cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline



# sklearn ml algorithms

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.linear_model import SGDClassifier, LogisticRegression

from xgboost import XGBClassifier



# Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt



# Hide warnings

import warnings

warnings.filterwarnings('ignore')
# read both train and test set files

df_train_raw = pd.read_csv("../input/digit-recognizer/train.csv")

df_test_raw = pd.read_csv("../input/digit-recognizer/test.csv")
# check dimensions of train and test set

print("Train Set has", df_train_raw.shape[0], "rows and", df_train_raw.shape[1], "columns.")

print("Train Set has", df_test_raw.shape[0], "rows and", df_test_raw.shape[1], "columns.")
# create a copy of train set for exploration purposes

df1 = df_train_raw.copy()
# create a copy of target labels and remove them from train set

train_labels = df1['label'].copy()

df1.drop(columns=['label'], inplace=True)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in sss.split(df_train_raw, df_train_raw['label']):

    strat_train_set = df_train_raw.loc[train_index]

    df_validation = df_train_raw.loc[test_index]

    

# creating a copy of target labels and removing them from validation set

validation_labels = df_validation['label'].copy()

df_validation.drop(columns=['label'], inplace=True)



df_validation_copy = df_validation.copy()
df1.sample(5)
some_digit = np.array(df1.loc[420])

some_digit_image = some_digit.reshape(28, 28)



plt.imshow(some_digit_image, cmap='binary')

plt.axis('off')
preprocessor = Pipeline([

    ('scaler', StandardScaler()),

    ('pca', PCA(n_components=0.95))    # preserving 95% variance of the original data

])



df1 = preprocessor.fit_transform(df1.astype(np.float64))

df_validation = preprocessor.transform(df_validation.astype(np.float64))
MLA = [

    RandomForestClassifier(n_jobs=-1),

    

    SGDClassifier(early_stopping=True, n_iter_no_change=5, n_jobs=-1),

    

    LogisticRegression(n_jobs=-1),    # One-vs-Rest method

    

    LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10, n_jobs=-1),    # Softmax Regression

    

    XGBClassifier(n_jobs=-1)

]
cv_split = ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)



MLA_columns = ['MLA Name', 'MLA Parameters', 'Train Accuracy Mean', 'Test Accuracy Mean', 'Test Accuracy 3*STD', 'Training Time']

MLA_compare = pd.DataFrame(columns = MLA_columns)



MLA_predict = train_labels.copy()



row_index = 0

for alg in MLA:



    # set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())

    

    # cross validation

    cv_results = cross_validate(alg, df_validation, validation_labels, cv=cv_split, scoring=['accuracy'], return_train_score=True)

    

    MLA_compare.loc[row_index, 'Training Time'] = cv_results['fit_time'].mean()

    MLA_compare.loc[row_index, 'Train Accuracy Mean'] = cv_results['train_accuracy'].mean()

    MLA_compare.loc[row_index, 'Test Accuracy Mean'] = cv_results['test_accuracy'].mean()

    MLA_compare.loc[row_index, 'Test Accuracy 3*STD'] = cv_results['test_accuracy'].std()*3



    # save MLA predictions

    alg.fit(df1, train_labels)

    MLA_predict[MLA_name] = alg.predict(df1)

    

    row_index+=1

    

# print and sort table

MLA_compare.sort_values(by = ['Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare
cv_split = ShuffleSplit(n_splits=10, test_size=0.3, train_size=0.6, random_state=0)



shortlisted_model = RandomForestClassifier()

base_results = cross_validate(shortlisted_model, df_validation, validation_labels, cv=cv_split, scoring=['accuracy'], return_train_score=True)



print('BEFORE GridSearch Parameters: ', shortlisted_model.get_params())

print("BEFORE GridSearch Training Accuracy mean: {:.2f}". format(base_results['train_accuracy'].mean()))

print("BEFORE GridSearch Test Accuracy mean: {:.2f}". format(base_results['test_accuracy'].mean()))

print("BEFORE GridSearch Test Accuracy 3*std: +/- {:.2f}". format(base_results['test_accuracy'].std()*3))

print('-'*10)



param_grid = [

    {

        'n_estimators': [100, 300, 500, 1000],

        'criterion': ['gini', 'entropy'],

        'oob_score': [True],

        'random_state': [0]

    }

]



grid_search = GridSearchCV(shortlisted_model, param_grid=param_grid, scoring='accuracy', cv=cv_split, return_train_score=True)

grid_search.fit(df_validation, validation_labels)



print('AFTER GridSearch Parameters: ', grid_search.best_params_)

print("AFTER GridSearch Training Accuracy mean: {:.2f}". format(grid_search.cv_results_['mean_train_score'][grid_search.best_index_]))

print("AFTER GridSearch Test Accuracy mean: {:.2f}". format(grid_search.cv_results_['mean_test_score'][grid_search.best_index_]))

print("AFTER GridSearch Test Accuracy 3*std: +/- {:.2f}". format(grid_search.cv_results_['std_test_score'][grid_search.best_index_]*3))

print('-'*10)
# predictors for voting classifier

predictor1 = RandomForestClassifier(criterion='gini', max_depth=None, n_estimators=500, oob_score=True, random_state=0, n_jobs=-1)

predictor2 = XGBClassifier(n_jobs=-1)

predictor3 = SGDClassifier(early_stopping=True, n_iter_no_change=10, n_jobs=-1)

predictor4 = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=10, n_jobs=-1)
voting_clf = VotingClassifier(

    estimators=[('pred1', predictor1), ('pred2', predictor2), ('pred3', predictor3), ('pred4', predictor4)],

    voting='hard'

)



score = cross_val_score(voting_clf, df_validation, validation_labels, cv=cv_split, scoring='accuracy').mean()

print("Accuracy Score of Voting Classifier: ", score)
df_validation_copy = StandardScaler().fit_transform(df_validation_copy.astype(np.float64))



rfc = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, min_samples_leaf=4, n_jobs=-1)

rfc.fit(df_validation_copy, validation_labels)
data = rfc.feature_importances_



image = data.reshape(28, 28)

plt.imshow(image, cmap = mpl.cm.hot, interpolation="nearest")

plt.axis("off")



cbar = plt.colorbar(ticks=[rfc.feature_importances_.min(), rfc.feature_importances_.max()])

cbar.ax.set_yticklabels(['Not important', 'Very important'])

plt.show()
eval_model = clone(voting_clf)

eval_pred = cross_val_predict(eval_model, df_validation, validation_labels, cv=3)
# confusion matrix

confusion_mx = confusion_matrix(validation_labels, eval_pred)



# plot better visualization for confusion matrix

plt.matshow(confusion_mx, cmap=plt.cm.gray)

plt.show
# Applying preprocess pipeline to test set

df_test_prepared = preprocessor.transform(df_test_raw)
# FINAL MODEL

final_model = clone(voting_clf)

final_model.fit(df1, train_labels)



# submission predictions

final_pred = final_model.predict(df_test_prepared)
# create submission file

my_submission = pd.DataFrame({'ImageId': np.array(range(1,28001)), 'Label': final_pred})

my_submission.to_csv("submission.csv", index=False)