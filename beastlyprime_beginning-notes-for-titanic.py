import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

combine = [train, test]
train.head()
train.info()

print('_'*80)

print("Unique value in 'Sex': ", train["Sex"].unique())

print("Value counts: ", train["Sex"].value_counts())
train.describe()
train.describe(include=['O'])
train["family_size"] = float("NaN")

test["family_size"] = float("NaN")
for dataframe in combine:

    dataframe["Embarked"] = dataframe["Embarked"].fillna("S")

    dataframe["Age"] = dataframe["Age"].fillna(dataframe["Age"].median())

test.Fare = test.Fare.fillna(test["Fare"].median())
for dataframe in combine:

    # Convert the male and female groups to integer form

    dataframe['Sex'] = dataframe['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    # Convert the Embarked classes to integer form

    dataframe['Embarked'] = dataframe['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# Or use commands like this

# train.loc[train["Sex"] == "male", "Sex"] = 0
# Create features

train["family_size"] = train["SibSp"] + train["Parch"] + 1

test["family_size"] = test["SibSp"] + test["Parch"] + 1



train['IsAlone'] = 1 #initialize to yes/1 is alone

train['IsAlone'].loc[train['family_size'] > 1] = 0 # now update to no/0 if family size is greater than 1
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["family_size", "Survived"]].groupby(['family_size'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

from pandas.tools.plotting import scatter_matrix



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 12,8



data1 = train
plt.figure(figsize=[16,12])



plt.subplot(231)

plt.boxplot(x=data1['Fare'], showmeans = True, meanline = True)

plt.title('Fare Boxplot')

plt.ylabel('Fare ($)')



plt.subplot(232)

plt.boxplot(data1['Age'], showmeans = True, meanline = True)

plt.title('Age Boxplot')

plt.ylabel('Age (Years)')



plt.subplot(233)

plt.boxplot(data1['family_size'], showmeans = True, meanline = True)

plt.title('Family Size Boxplot')

plt.ylabel('Family Size (#)')



plt.subplot(234)

plt.hist(x = [data1[data1['Survived']==1]['Fare'], data1[data1['Survived']==0]['Fare']], 

         stacked=True, color = ['b','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(235)

plt.hist(x = [data1[data1['Survived']==1]['Age'], data1[data1['Survived']==0]['Age']], 

         stacked=True, color = ['b','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age (Years)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(236)

plt.hist(x = [data1[data1['Survived']==1]['family_size'], data1[data1['Survived']==0]['family_size']], 

         stacked=True, color = ['b','r'],label = ['Survived','Dead'])

plt.title('Family Size Histogram by Survival')

plt.xlabel('Family Size (#)')

plt.ylabel('# of Passengers')

plt.legend()
#graph individual features by survival

fig, saxis = plt.subplots(2, 3,figsize=(16,12))



sns.barplot(x = 'Embarked', y = 'Survived', data=data1, ax = saxis[0,0])

sns.barplot(x = 'Pclass', y = 'Survived', order=[1,2,3], data=data1, ax = saxis[0,1])

sns.barplot(x = 'IsAlone', y = 'Survived', order=[1,0], data=data1, ax = saxis[0,2])



sns.barplot(x = 'Sex', y = 'Survived', order=[1,0], data=data1, ax = saxis[1,1])

sns.pointplot(x = 'SibSp', y = 'Survived',  data=data1, ax = saxis[1,0])

# sns.pointplot(x = 'AgeBin', y = 'Survived',  data=data1, ax = saxis[1,1])

sns.pointplot(x = 'family_size', y = 'Survived', data=data1, ax = saxis[1,2])
#correlation heatmap of dataset

def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(data1)
g = sns.FacetGrid(train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
# sns.distplot(train["Age"])
#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, ensemble

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics



# Construct features and the target

feature_names = ["Pclass", "Sex", "Age", "Fare", "family_size", "Embarked"]

features = train[["Pclass", "Sex", "Age", "Fare", "family_size", "Embarked"]].values

features_test = test[["Pclass", "Sex", "Age", "Fare", "family_size", "Embarked"]].values

target = train["Survived"]
def train_model(model, features, target, fit=False):

    # Split dataset in cross-validation

    # Run model 10x with 60/30 split intentionally leaving out 10%

    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) 

    cv_results = model_selection.cross_validate(model, features, target, cv  = cv_split)

    

    if(fit):

        # fit model

        model = model.fit(features, target)

        return model, cv_results

    

    return cv_results
# Train on a tree

# decision_tree = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)

# decision_tree = decision_tree.fit(features, target)

# print(decision_tree.score(features, target))

# print(decision_tree.feature_importances_)

decision_tree = tree.DecisionTreeClassifier()

trained_tree, cv_results = train_model(decision_tree, features, target, fit=True)



Tree_Predict = trained_tree.predict(features)



# Report

print('Decision Tree Model Accuracy/Precision Score on training set: {:.2f}%\n'

      .format(metrics.accuracy_score(data1['Survived'], Tree_Predict)*100))

print(metrics.classification_report(data1['Survived'], Tree_Predict))

print(np.mean(cv_results['test_score']))
# Make prediciton

prediction_dt = decision_tree.predict(features_test)

PassengerId =np.array(test["PassengerId"]).astype(int)

solution_dt = pd.DataFrame(prediction_dt, PassengerId, columns = ["Survived"])

solution_dt.to_csv("solution_dt.csv", index_label = ["PassengerId"])
# Tune hyper-parameters

param_grid = {'criterion': ['entropy'],  # scoring methodology

              #'splitter': ['best', 'random'], # splitting methodology

              'max_depth': [4,6,8], # max depth tree can grow

              'min_samples_split': [2, 3,.03], # minimum subset size BEFORE new split (fraction is % of total)

              'min_samples_leaf': [10,.03,.05], # minimum subset size AFTER new split split (fraction is % of total)

              'max_features': [None, 'auto'], # max features to consider when performing split

              'random_state': [0] #seed or control random number generator

             }



# 

len(list(model_selection.ParameterGrid(param_grid)))



cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) 



# Choose best model with grid_search:

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html

tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(), 

                                          param_grid=param_grid, scoring = 'roc_auc', cv = cv_split)

tune_model.fit(features, target)



# Report

print('-'*10)

print('Best Parameters: ', tune_model.best_params_)

print("Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 

print("Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))

print("Test w/bin score 3*std: +/- {:.2f}". format(tune_model.cv_results_['std_test_score'][tune_model.best_index_]*100*3))
# The best tree we have now

best_tree = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 6, 

                                        min_samples_leaf = 0.03, min_samples_split = 2)



# Report the best tree

cv_best = train_model(best_tree, features, target)

print(np.mean(cv_best['test_score']))
import graphviz 



best_tree.fit(features, target)

dot_data = tree.export_graphviz(best_tree, out_file = None, 

                                feature_names = feature_names, class_names = True,

                                filled = True, rounded = True)

graph = graphviz.Source(dot_data) 

graph
from sklearn.ensemble import RandomForestClassifier



# Train on a tree

forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)

random_forest = forest.fit(features, target)

print(random_forest.score(features, target))

print(random_forest.feature_importances_)



# Make prediciton

prediction_rf = random_forest.predict(features_test)

solution_rf = pd.DataFrame(prediction_rf, PassengerId, columns = ["Survived"])

solution_rf.to_csv("solution_rf.csv", index_label = ["PassengerId"])
ID = np.arange(0,10)

age = np.arange(10,20)

people = pd.DataFrame(age, ID, columns = ["Age"])

people.to_csv("people.csv", index_label = ["ID"])
people = pd.DataFrame({"ID":ID, "Age":age})

people.to_csv("People.csv", index=False)