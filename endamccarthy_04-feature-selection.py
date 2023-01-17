import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import statistics



from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest, f_regression, SelectPercentile, RFE



sns.set(style="darkgrid")

%matplotlib inline
df = pd.read_csv("../input/winequality_red.csv")

df.head()
df.describe()
plt.figure(figsize=(10, 6))

sns.countplot(df["quality"], palette="muted")
df.apply(lambda x: sum(x.isnull()), axis=0)
# target attribute

target_attribute_name = 'quality'

target = df[target_attribute_name]



# predictor attributes

predictors = df.drop(target_attribute_name, axis=1).values



# predictor attributes names

predictors_col_names = list(df.drop(target_attribute_name, axis=1).columns)
labelencoder = LabelEncoder()

target = labelencoder.fit_transform(target)
#        quality = df["quality"].values

#        category = []

#        for number in quality:

#            if number > 6:

#                category.append("Good")

#            elif number > 3:

#                category.append("Okay")

#            else:

#                category.append("Poor")

#        category = pd.DataFrame(data=category, columns=["category"])

#        data = pd.concat([df, category], axis=1)

#        data.drop(columns="quality", axis=1, inplace=True)

#

#        # target attribute

#        target_attribute_name = 'category'

#        target = data[target_attribute_name]

#

#        # predictor attributes

#        predictors = data.drop(target_attribute_name, axis=1).values

#

#        labelencoder = LabelEncoder()

#        target = labelencoder.fit_transform(target)
# prepare independent stratified data sets for training and test of the final model

predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, 

                                                                                target, 

                                                                                test_size=0.20, 

                                                                                shuffle=True, 

                                                                                stratify=target)
min_max_scaler = MinMaxScaler()

predictors_train = min_max_scaler.fit_transform(predictors_train)

predictors_test = min_max_scaler.fit_transform(predictors_test)
# create a base classifier used to evaluate a subset of attributes

estimatorSVM = svm.SVR(kernel="linear")

# create the RFE model and select 3 relevant attributes

selectorSVM = RFE(estimatorSVM, 3)

selectorSVM = selectorSVM.fit(predictors_train, target_train)

# summarize the selection of the attributes

print(selectorSVM.support_)

print(selectorSVM.ranking_)
# create a base classifier used to evaluate a subset of attributes

estimatorLR = LogisticRegression(solver='lbfgs', multi_class='auto')

# create the RFE model and select 3 relevant attributes

selectorLR = RFE(estimatorLR, 3)

selectorLR = selectorLR.fit(predictors_train, target_train)

# summarize the selection of the attributes

print(selectorLR.support_)

print(selectorLR.ranking_)
# select only relevant features from training and test seperately - SVM

predictors_train_SVMselected = selectorSVM.transform(predictors_train)

predictors_test_SVMselected = selectorSVM.transform(predictors_test)



# select only relevant features from training and test seperately - LR

predictors_train_LRselected = selectorLR.transform(predictors_train)

predictors_test_LRselected = selectorLR.transform(predictors_test)
# create SVM classifier

classifier = svm.SVC(gamma='scale')
# run final classifier with only features selected using RFE with SVM

model1 = classifier.fit(predictors_train_SVMselected, target_train)

accuracy1 = model1.score(predictors_test_SVMselected, target_test)
# run final classifier with only features selected using RFE with LR

model2 = classifier.fit(predictors_train_LRselected, target_train)

accuracy2 = model2.score(predictors_test_LRselected, target_test)
# run final classifier with all the features

model3 = classifier.fit(predictors_train, target_train)

accuracy3 = model3.score(predictors_test, target_test)
print("Model 1 Accuracy = %.4f" % (accuracy1))

print("Model 2 Accuracy = %.4f" % (accuracy2))

print("Model 3 Accuracy = %.4f" % (accuracy3))
# create base classifiers

estimatorSVM = svm.SVR(kernel="linear")

estimatorLR = LogisticRegression(solver='lbfgs', multi_class='auto')



# create RFE model for both classifiers to find 3 best features

selectorSVM = RFE(estimatorSVM, 3)

selectorLR = RFE(estimatorLR, 3)



# create SVM classifier for final evaluation

classifier = svm.SVC(gamma='scale')



# store the results from loop in a dataframe 

results_df = pd.DataFrame(columns=('score', 'split', 'model'))



# create list of differant % test splits (15%, 20% and 25%)

test_sizes = [0.15, 0.20, 0.25]



# list of model numbers

model = [1, 2, 3]



# counter

row = 0



for i in range(len(test_sizes)):

    for j in range(20):

        predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, 

                                                                                        target, 

                                                                                        test_size=test_sizes[i], 

                                                                                        shuffle=True, 

                                                                                        stratify=target)



        # scale predictors

        predictors_train = min_max_scaler.fit_transform(predictors_train)

        predictors_test = min_max_scaler.fit_transform(predictors_test)



        # use RFE models on data to identify best features

        selectorSVM = selectorSVM.fit(predictors_train, target_train)

        selectorLR = selectorLR.fit(predictors_train, target_train)



        # select only relevant features from training and test seperately - SVM

        predictors_train_SVMselected = selectorSVM.transform(predictors_train)

        predictors_test_SVMselected = selectorSVM.transform(predictors_test)



        # select only relevant features from training and test seperately - LR

        predictors_train_LRselected = selectorLR.transform(predictors_train)

        predictors_test_LRselected = selectorLR.transform(predictors_test)



        # run final classifier with only with features selected using RFE with SVM

        model1 = classifier.fit(predictors_train_SVMselected, target_train)

        model1_score = model1.score(predictors_test_SVMselected, target_test)

        results_df.loc[row] = [model1_score, (test_sizes[i]*100), model[0]]

        row+=1



        # run final classifier with only with features selected using RFE with LR

        model2 = classifier.fit(predictors_train_LRselected, target_train)

        model2_score = model2.score(predictors_test_LRselected, target_test)

        results_df.loc[row] = [model2_score, (test_sizes[i]*100), model[1]]

        row+=1



        # run final classifier with all features

        model3 = classifier.fit(predictors_train, target_train)

        model3_score = model3.score(predictors_test, target_test)

        results_df.loc[row] = [model3_score, (test_sizes[i]*100), model[2]]

        row+=1

            
plt.figure(figsize=(14, 8))

sns.boxplot(x="split", y="score", hue="model", data=results_df, width = .4, palette="Set3")
variance_df = pd.DataFrame(index=[0], columns=('0.15 split, model 1', '0.15 split, model 2', '0.15 split, model 3',

                                               '0.20 split, model 1', '0.20 split, model 2', '0.20 split, model 3',

                                               '0.25 split, model 1', '0.25 split, model 2', '0.25 split, model 3'))



for i in test_sizes:

    for j in model:

        variance = np.var(list(results_df.score[(results_df['model'] == j) & (results_df['split'] == (i*100))]))

        variance_df.at[0, '%.2f split, model %d' % (i,j)] = variance
plt.figure(figsize=(14, 8))

ax = sns.barplot(data=variance_df, palette="Set3")

for item in ax.get_xticklabels():

    item.set_rotation(60)
accuracy2_20 = statistics.mean(list(results_df.score[(results_df['model'] == 1) & (results_df['split'] == (25))]))

print("Model 2 with 0.20 Split Accuracy = %.4f" % (accuracy2_20))
# target attribute

target_attribute_name = 'quality'

target = df[target_attribute_name]

labelencoder = LabelEncoder()

target = labelencoder.fit_transform(target)



# predictor attributes

predictors = df.drop(target_attribute_name, axis=1).values



# predictor attributes names

predictors_col_names = list(df.drop(target_attribute_name, axis=1).columns)
# prepare independent stratified data sets for training and test of the final model

X_train, X_test, y_train, y_test = train_test_split(predictors, 

                                                    target, 

                                                    test_size=0.25, 

                                                    shuffle=True, 

                                                    stratify=target)
# set up pipeline

    # step 1 - scale the data

    # step 2 - select the best features to use (reduce dimensionality)

    # step 3 - use a learning algorithm on the selected features

# note: we will add more options to each step in the next block)

pipe = Pipeline([('scaler', MinMaxScaler()),

                 ('reduce_dim', SelectPercentile(f_regression)),

                 ('regressor', svm.SVC(gamma='scale'))])
# as stated above we can add more options for each step

# first we add options for scaling

scalers_to_test = [StandardScaler(), RobustScaler(), MinMaxScaler()]

# next we add options for learning algorithms

regressors_to_test = [svm.SVC(gamma='scale'), LogisticRegression(solver='lbfgs', multi_class='auto')]

# then we can vary the number of selected features from 1-11 for each variation

n_features_to_test = np.arange(1, 12)
# params will be passed in alongside our pipeline

# we have two sets of params here, each with a different method of selectinf features:

    # first we use the SelectPercentile method (where percentile is the number of attributes)

    # then we use the SelectKBest method (where k is the number of attributes)

params = [{'scaler': scalers_to_test,

           'reduce_dim': [SelectPercentile(f_regression)],

           'reduce_dim__percentile': n_features_to_test,

           'regressor': regressors_to_test},



          {'scaler': scalers_to_test,

           'reduce_dim': [SelectKBest(f_regression)],

           'reduce_dim__k': n_features_to_test,

           'regressor': regressors_to_test}]
# then we can train our final model using the pipeline and params (cross-validation is used)

gridsearch = GridSearchCV(pipe, params, cv=5, verbose=1, iid=False).fit(X_train, y_train)
# we can see which params were chosen as the best performing

gridsearch.best_params_
# we can see which features were selected from the pipeline

final_pipeline = gridsearch.best_estimator_

final_classifier = final_pipeline.named_steps['regressor']

mask = final_pipeline.named_steps['reduce_dim'].get_support()

feature_names = df.drop(target_attribute_name, axis=1).columns

selected_features = feature_names[final_pipeline.named_steps['reduce_dim'].get_support()].tolist()

selected_features
# and finally we can see the accuracy of the final model

print('Final score is: ', gridsearch.score(X_test, y_test))