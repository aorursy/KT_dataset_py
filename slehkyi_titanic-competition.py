import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))



plt.style.use('ggplot')

%matplotlib inline
df = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

all_features = df.columns.tolist()
df.head()
df.info()
df.describe()
df_X = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
_ = pd.plotting.scatter_matrix(df_X, figsize=[10,10], s=150, marker='D')
plt.figure()

sns.countplot(x='Embarked', hue='Embarked', data=df)

plt.xticks([0,1,2],df['Embarked'].drop_duplicates().tolist())
df['Embarked'].loc[df['Embarked'].isnull()] = 'S'

df['Embarked'].count()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])

df['Embarked'] = le.fit_transform(df['Embarked'])

plt.figure(figsize=(12,12))

sns.heatmap(df.corr(), square=True, annot=True, cmap='PRGn')
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan)



to_impute = df[['Age']]

imputed = imp.fit_transform(to_impute)

df['Age'] = pd.Series(imputed.reshape(891,))

plt.figure(figsize=(12,12))

sns.heatmap(df.corr(), square=True, annot=True, cmap='PRGn')
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



X = df.select_dtypes(['number'])

X = X.drop(columns=['PassengerId','Survived'])

y = df['Survived']



# apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k=5)

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)



# concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  # naming the dataframe columns

print(featureScores.nlargest(5,'Score'))  #print 5 best features
from sklearn.ensemble import ExtraTreesClassifier



model = ExtraTreesClassifier(n_estimators=100)

model.fit(X,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers



#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(5).plot(kind='barh')
# if we want to play with different combinations of features

print(all_features)
list_of_features = ['Sex', 'Fare', 'Pclass', 'Age', 'Embarked']

df = X[list_of_features]

df.head()
df_onehot = df.copy()

df_onehot = pd.get_dummies(df_onehot, columns=['Pclass'], prefix=['class'])

df_onehot = pd.get_dummies(df_onehot, columns=['Embarked'], prefix=['emb'])
from sklearn.preprocessing import scale



df_onehot['Fare'] = scale(df_onehot['Fare'])

df_onehot['Age'] = scale(df_onehot['Age'])

# df_onehot['Parch'] = scale(df_onehot['Parch'])



y = y

X = df_onehot
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42, stratify=y)
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# Hyperparameters and classifiers:

# SVC

svm_c_space = [0.1,0.075, 0.05,0.025]

svm_kernels = ['linear', 'poly', 'rbf', 'sigmoid']

svm_gamma = [1,25, 1, 0.75, 0.5]

param_grid_svc = dict(C=svm_c_space,

                      kernel=svm_kernels,

                      gamma=svm_gamma)

clf_svc = SVC()



# Logistic Regression

c_space = np.logspace(0,2,10)

solvers = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']

param_grid_logreg = dict(C=c_space,

                         solver=solvers)

clf_logreg = LogisticRegression()



# KNearest Neigbors

n_neighbors = range(1,12)

param_grid_knn = dict(n_neighbors=n_neighbors)

clf_knn = KNeighborsClassifier()



# Random Forest

estimators_space = [100,250,500]

min_sample_splits = range(2,8)

min_sample_leaves = range(1,5)

max_features = ['sqrt', 'log2', None]

param_grid_forest = dict(n_estimators=estimators_space, 

                        min_samples_split=min_sample_splits,

                        min_samples_leaf=min_sample_leaves,

                        max_features=max_features)

clf_forest = RandomForestClassifier()



# Decision Tree

min_sample_splits = range(2,12)

min_sample_leaves = range(1,8)

max_features = ['sqrt', 'log2', None]

param_grid_tree = dict(min_samples_split=min_sample_splits,

                       min_samples_leaf=min_sample_leaves,

                       max_features=max_features)

clf_tree = DecisionTreeClassifier()



# AdaBoost

# ada_estimators = [25,50,100]

ada_lr = [1, 0.5, 0.1]

param_grid_ada = dict(n_estimators=estimators_space,

                     learning_rate=ada_lr)

clf_ada = AdaBoostClassifier()



# GradientBoosting

gb_lr = [0.2, 0.1, 0.01]

param_grid_gb = dict(n_estimators=estimators_space,

                     learning_rate=gb_lr)

clf_gb = GradientBoostingClassifier()



# LinearDiscriminantAnalysis

lda_solvers = ['svd', 'lsqr', 'eigen']

param_grid_lda = dict(solver=lda_solvers)

clf_lda = LinearDiscriminantAnalysis()
# creating loop to test all classifiers

models_to_test = ['GradientBoosting',

                  #'LDA',

                  #'LogisticRegression',

                  #'KNN',

                  #'SVC',

                  'RandomForest',

                  #'DecisionTree',

                  'AdaBoost'

                 ]

classifier_dict = dict(LogisticRegression=clf_logreg,

                      KNN=clf_knn,

                      SVC=clf_svc,

                      RandomForest=clf_forest,

                      DecisionTree=clf_tree,

                      AdaBoost=clf_ada,

                      GradientBoosting=clf_gb,

                      LDA=clf_lda

                      )

param_grid_dict = dict(LogisticRegression=param_grid_logreg,

                       KNN=param_grid_knn,

                       SVC=param_grid_svc,

                       RandomForest=param_grid_forest,

                       DecisionTree=param_grid_tree,

                       AdaBoost=param_grid_ada,

                       GradientBoosting=param_grid_gb,

                       LDA=param_grid_lda

                      )

cv = 5

score_dict = {}

params_dict = {}

classification_report_dict = {}

conf_matr_dict = {}

best_est_dict = {}



for model in models_to_test:

    # classifier = RandomizedSearchCV(classifier_dict[model], param_grid_dict[model], cv=cv, n_jobs=-1)

    classifier = GridSearchCV(classifier_dict[model], param_grid_dict[model], cv=cv, n_jobs=-1)

    

    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    

    # Print the tuned parameters and score

    print(" === Start report for classifier {} ===".format(model))

    score_dict[model] = classifier.best_score_

    print("Tuned Parameters: {}".format(classifier.best_params_)) 

    params_dict = classifier.best_params_

    print("Best score is {}".format(classifier.best_score_))



    # Compute metrics

    classification_report_dict[model] = classification_report(y_test, y_pred)

    print("Classification report for {}".format(model))

    print(classification_report(y_test, y_pred))

    conf_matr_dict[model] = confusion_matrix(y_test, y_pred)

    print("Confusion matrix for {}".format(model))

    print(confusion_matrix(y_test, y_pred))

    print(" === End of report for classifier {} === \n".format(model))

    

    # Add best estimator to the dict

    best_est_dict[model] = classifier.best_estimator_

    

# Creating summary report

summary_cols = ['Best Score']

summary = pd.DataFrame.from_dict(score_dict, orient='index')

summary.index.name = 'Classifier'

summary.columns = summary_cols

summary = summary.reset_index()
# Visualizing results

plt.xlabel('Best score')

plt.title('Classifier Comparison')



sns.barplot(x='Best Score', y='Classifier', data=summary)
# Get the name of the best performing model

name_of_best_model = summary['Classifier'][summary['Best Score'] == summary['Best Score'].max()].tolist()[0]

clf_selected = best_est_dict[name_of_best_model]

print("Selected model - {}".format(clf_selected))
df_to_test = pd.read_csv('../input/test.csv')

df_to_test.head()
df_to_test.info()
# Drop useless columns

df_test = df_to_test[list_of_features]



# Encode categorical values

df_test['Sex'] = le.fit_transform(df_test['Sex'])

df_test = pd.get_dummies(df_test, columns=['Embarked'], prefix=['emb'])

df_test = pd.get_dummies(df_test, columns=['Pclass'], prefix=['class'])



# Impute missing values

df_test['Age'] = df_test['Age'].fillna(0)

imp.fit_transform(np.array(df_test['Age']).reshape(1,-1))

df_test['Fare'] = df_test['Fare'].fillna(0)

imp.fit_transform(np.array(df_test['Fare']).reshape(1,-1))



# Scale

df_test['Fare'] = scale(df_test['Fare'])

df_test['Age'] = scale(df_test['Age'])

# df_test['Parch'] = scale(df_test['Parch'])

df_test.head()
# Predict the labels of the test set

y_pred = clf_selected.predict(df_test)
df_to_test['Survived'] = pd.Series(y_pred)

pd.DataFrame(df_to_test[['PassengerId','Survived']]).to_csv('predictions.csv', index=False)

df_to_test.head()