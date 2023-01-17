import pandas as pd
titanic_train = pd.read_csv(r'../input/titanic/train.csv')

titanic_train.head()
titanic_train.info()
titanic_train['Sex'].value_counts()
titanic_train['Ticket'].value_counts()
titanic_train['Cabin'].value_counts()
titanic_train['Embarked'].value_counts()

# sample size is heavily skewed in the favour of Southampton
titanic_train['Pclass'].value_counts()
titanic_train['Survived'].value_counts(normalize=True)
titanic_train['SibSp'].value_counts()
titanic_train['Parch'].value_counts()
titanic_train['Fare'].value_counts()
titanic_train.describe()
%matplotlib inline

import matplotlib.pyplot as plt

titanic_train.hist(bins=50, figsize=(20,15))
# create a copy of the training dataset for performing exploratory data analysis

titanic = titanic_train.copy()

titanic.head()
import seaborn as sns
sns.countplot(x='Sex', hue ='Survived', data = titanic)

cross_tab = pd.crosstab(titanic.Sex, titanic.Survived).apply(lambda r:r*100/r.sum(), axis = 1)

print(cross_tab)



## very strong correlation between sex and survival rate.
sns.countplot(x='SibSp',hue='Survived',data=titanic)

cross_tab = pd.crosstab(titanic['SibSp'],titanic['Survived']).apply(lambda r: r*100/r.sum(), axis=1)

print(cross_tab)
sns.countplot(x='Parch',hue='Survived',data=titanic)

cross_tab = pd.crosstab(titanic['Parch'],titanic['Survived']).apply(lambda r: r*100/r.sum(), axis=1)

print(cross_tab)
# create a new column for passenger's travelling alone

titanic['is_alone'] = titanic['SibSp'] + titanic['Parch']

titanic['is_alone'] = titanic['is_alone'].apply(lambda x: 1 if x>0 else 0)



sns.countplot(x='is_alone',hue='Survived',data=titanic)

cross_tab = pd.crosstab(titanic['is_alone'],titanic['Survived']).apply(lambda r: r*100/r.sum(), axis=1)

print(cross_tab)



# passenger's travelling with family has a slightly better chance of survival
sns.countplot(x='Embarked',hue='Survived',data=titanic)

cross_tab = pd.crosstab(titanic['Embarked'],titanic['Survived']).apply(lambda r: r*100/r.sum(), axis=1)

print(cross_tab)



# people who embarked from Cherbourg has slightly better survival rate
sns.countplot(x='Pclass',hue='Survived',data=titanic)

cross_tab = pd.crosstab(titanic['Pclass'],titanic['Survived']).apply(lambda r: r*100/r.sum(), axis=1)

print(cross_tab)



# survival rate tends to decrease with the class in which a person is travelling 1> 2>3
graph = sns.FacetGrid(titanic, hue="Survived", palette="Set1", )

graph = graph.map(plt.hist,"Age", alpha=0.5)

graph.add_legend()

#passenger's less than 20(children), higher survival, 20-60 : poor survival (adults)
graph = sns.FacetGrid(titanic, hue="Survived", palette="Set1", )

graph = graph.map(plt.hist,"Fare", alpha=0.5)

graph.add_legend()

# higher the fare, better the survival rate
titanic['Cabin'] = titanic['Cabin'].fillna('NA')

titanic['Cabin'] = titanic.Cabin.apply(lambda x : 'No' if x == 'NA' else 'Yes')

titanic['Cabin'].value_counts()
sns.countplot(x='Cabin',hue='Survived',data=titanic)

cross_tab = pd.crosstab(titanic['Cabin'],titanic['Survived']).apply(lambda r: r*100/r.sum(), axis=1)

print(cross_tab)
#seperate predicators and labels

titanic = titanic_train.drop(columns='Survived')

titanic_labels = titanic_train['Survived'].copy()
titanic.head()
titanic['Cabin'] = titanic['Cabin'].fillna('NA')

titanic['Cabin'] = titanic.Cabin.apply(lambda x : 'No' if x == 'NA' else 'Yes')



titanic['is_alone'] = titanic['SibSp'] + titanic['Parch']

titanic['is_alone'] = titanic['is_alone'].apply(lambda x: 1 if x>0 else 0)



titanic.head()
titanic = titanic.drop(columns=['PassengerId','Name','Ticket','SibSp', 'Parch'], axis=1)

titanic
# custom DataFrameSelector class for column transformation 

from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self,X):

        return X[self.attribute_names].values
# Create a pipeline for data cleaning 

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import StandardScaler,OneHotEncoder

from sklearn.impute import SimpleImputer



num_attrib = list(titanic[['Age','Fare','is_alone']])

cat_attrib = list(titanic[['Sex','Pclass','Cabin','Embarked']])



# the selector will select a column attributes from the given list

# imputer will fill missing numerical fields . in this case for age with median value i.e 28

# std_scalar for scaling numerical attributes



num_pipeline = Pipeline([

    ('selector', DataFrameSelector(num_attrib)),

    ('imputer',SimpleImputer(strategy='median')),

    ('std_scaler',StandardScaler())

])



# SimpleImputer fills missing values in the categorical fields with mode/ most frequent value

# Onehotencoding is performed for transforming categorical values as numerical



cat_pipeline = Pipeline([

    ('selector', DataFrameSelector(cat_attrib)),

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('one_hot_encoder', OneHotEncoder())

])



# create a full pipeline by combining results of subpipelines(num_pipeline, cat_pipeline)



full_pipeline = FeatureUnion(transformer_list=[

    ('num_pipepline',num_pipeline),

    ('cat_pipeline',cat_pipeline)

])



# fit and transform the training dataset instance on pipeline



titanic_prepared = full_pipeline.fit_transform(titanic)

titanic_prepared
# LINEAR REGRESSION

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(titanic_prepared,titanic_labels)
some_data = titanic.iloc[:5]

some_labels = titanic_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)

print('Predictions:\t', lin_reg.predict(some_data_prepared))

print('Labels:\t',list(some_labels))
import numpy as np

from sklearn.metrics import mean_squared_error

survival_prediction = lin_reg.predict(titanic_prepared)

lin_mse = mean_squared_error(titanic_labels,survival_prediction)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
#Decision Tree Classifier

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(titanic_prepared,titanic_labels)

tree_predict = tree_reg.predict(titanic_prepared)

tree_mse = mean_squared_error(titanic_labels,tree_predict)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
# k-fold cross validation on the training dataset

from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, titanic_prepared,titanic_labels,

                         scoring = "neg_mean_squared_error",cv=10)

rmse_scores = np.sqrt(-scores)



def display_scores(scores):

    print('Scores :', scores)

    print('Mean :', scores.mean())

    print('Standard deviation :', scores.std())

display_scores(rmse_scores)
lin_scores = cross_val_score(lin_reg,titanic_prepared,titanic_labels,

                            scoring = 'neg_mean_squared_error',cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)
# ElasticNet Linear model

from sklearn.linear_model import ElasticNet

en = ElasticNet()

en.fit(titanic_prepared,titanic_labels)



'''

some_data = titanic.iloc[:5]

some_labels = titanic_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)

print('Predictions:\t', en.predict(some_data_prepared))

print('Labels:\t',list(some_labels))



'''

en_scores = cross_val_score(en,titanic_prepared,titanic_labels,

                            scoring = 'neg_mean_squared_error',cv=10)

en_rmse_scores = np.sqrt(-en_scores)

display_scores(en_rmse_scores)
# Finding best paramters using GridSearchCV



from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import ElasticNet



param_values = {'alpha':[0.001, 0.01, 0.1, 1, 10, 100, 1000],

'l1_ratio':[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}



Grid= GridSearchCV(ElasticNet(),param_values,scoring='neg_mean_squared_error',verbose=1, cv=5 )



Grid.fit(titanic_prepared,titanic_labels)
Grid.best_estimator_
Grid.fit(titanic_prepared,titanic_labels).best_score_
# final_model with best possible hyperparamters

final_model = Grid.best_estimator_



test = pd.read_csv(r'../input/titanic/test.csv')

result = pd.read_csv(r'../input/titanic/gender_submission.csv')



X_test = test



X_test['Cabin'] = X_test.Cabin.fillna('NA')

X_test['Cabin'] = X_test.Cabin.apply(lambda x : 'NA' if x == 'No' else 'Yes')

X_test['is_alone'] = X_test['SibSp'] + X_test['Parch']

X_test['is_alone'] = X_test['is_alone'].apply(lambda x: 1 if x>0 else 0)



X_test = X_test.drop(columns=['PassengerId','Name','Ticket', 'SibSp','Parch'])



y_test = result['Survived'].copy()



X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)



final_mse = mean_squared_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)

final_rmse
predictions = np.abs(np.around(final_predictions))

predictions = predictions.astype(int)
passenger_id = list(test['PassengerId'])

prediction_submission = list(zip(passenger_id,predictions))

prediction_submission = pd.DataFrame(prediction_submission, columns = ('PassengerId','Survived'))

prediction_submission
prediction_submission.to_csv("final_result.csv",index=False)