import pandas as pd

import numpy as np

import sklearn 

import plotly.io as pio

import plotly.express as px

pio.templates.default = "plotly_white"



np.random.seed(42)

import seaborn as sns

sns.set()

%matplotlib inline
diabetes_data = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
diabetes_data.info()
diabetes_data.head(5)
diabetes_data.describe().T
p = diabetes_data.hist(bins=50, figsize=(20,15))
# replace zero values by NaN 

outliers_columns = ['BloodPressure', 'BMI', 'Glucose', 'SkinThickness']

 

diabetes_data[outliers_columns] = diabetes_data[outliers_columns].replace(0, np.NaN)



# replace nan with the median

diabetes_data[outliers_columns] = diabetes_data[outliers_columns].fillna(diabetes_data[outliers_columns].median())
p = diabetes_data.hist(bins=50, figsize=(20,15))
p = diabetes_data['Outcome'].value_counts().plot.barh()
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for X_index, y_index in split.split(diabetes_data, diabetes_data["Outcome"]):

    train_set = diabetes_data.loc[X_index]

    test_set = diabetes_data.loc[y_index]
diabetes_data_copy = train_set.copy()
## Null count analysis 

import missingno as msno

p = msno.bar(diabetes_data_copy)
(diabetes_data_copy == 0).sum()
corr_matrix = diabetes_data_copy.corr()
corr_matrix['Outcome'].sort_values(ascending=False)
import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))

p = sns.heatmap(corr_matrix, annot=True, cmap='Blues')
from pandas.plotting import scatter_matrix

attributes = ['Outcome', 'Glucose', 'BMI', 'Age']

p = scatter_matrix(diabetes_data_copy[attributes], figsize=(12,8))
p = sns.pairplot(diabetes_data_copy, hue="Outcome")
X_train = train_set.drop("Outcome", axis=1)

y_train = train_set["Outcome"]



X_test = test_set.drop("Outcome", axis=1)

y_test = test_set["Outcome"]
from sklearn.linear_model import SGDClassifier

from sklearn.svm import LinearSVC, SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin



class NeverTrueClassifier(BaseEstimator):

    def fit(self, X, y=None):

        pass

    def predict(self,X):

        return np.zeros(len(X))
# Select few models to test

def quick_models(prefix=""):

    models = []

    models.append((f"{prefix}Baseline", NeverTrueClassifier()))

    models.append((f"{prefix}SGD", SGDClassifier(random_state=42)))

#     models.append((f"Linear_SVC{prefix}", LinearSVC(random_state=42))) # max iter pb

    models.append((f"{prefix}SVC", SVC(random_state=42)))

    models.append((f"{prefix}KNN", KNeighborsClassifier()))

    models.append((f"{prefix}RandomForest", RandomForestClassifier(random_state=42)))

    return models
# create a class to keep track of the performance

class CompareModels():

    def __init__(self, models, scoring="accuracy", model_type="vanilla"):

        self.models = models

        self.scoring = scoring

        self.type = model_type

        self.names = []

        self.scores = []

        

    def get_scores(self, X,y):

        for name, model in self.models:

            score = cross_val_score(model, X, y, scoring=self.scoring, cv=10)

            self.names.append(name)

            self.scores.append(np.array(score))

            print(f"{name:-<20}> {score.mean()} ({(score).std()})")

            

    def create_table(self):

        self.table = pd.DataFrame({model:scores for model,scores in zip(self.names, self.scores)})

        self.table.columns.name = "Model"

        self.table = self.table.unstack().reset_index(0)

        self.table = self.table.rename(columns={0:self.scoring})

        self.table["type"] = self.type

            

    def summary_table(self):

        return self.table.groupby("Model").mean().reset_index()

    

    def box_plot(self):

        fig = px.box(self.table, y=self.scoring, x="Model")

        fig.update_layout(title=self.type.upper())

        fig.show()

    

    def recap(self, X, y):

        self.get_scores(X, y)

        self.create_table()

        self.box_plot()

        display(self.summary_table().sort_values(by=self.scoring,ascending=False))  
# Test the performance on models without any preprocessing 

models = CompareModels(quick_models())

models.recap(X_train, y_train)
from sklearn.impute import SimpleImputer

from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.preprocessing import StandardScaler 

pd.options.mode.chained_assignment = None  # default='warn'



pre_process = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
# test with Standardisation 

X_train_scaled = pre_process.fit_transform(X_train)



preprocess_models = CompareModels(quick_models(), model_type="standard")

preprocess_models.recap(X_train_scaled, y_train)
# compare pipeline perf

def compare_pipeline(models1, models2):

    df = pd.concat([models1.table, models2.table])

    display(px.box(df, x="Model", y="accuracy", color="type"))

    return df.pivot_table(values="accuracy", index="Model", columns="type")



compare_pipeline(models, preprocess_models)
# feature importance

forest_clf = RandomForestClassifier(random_state=42)

forest_clf.fit(X_train_scaled, y_train)

feature_importances = forest_clf.feature_importances_

pd.Series(index=X_train.columns, data=feature_importances).sort_values().plot.barh()
class TopFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_importances, k):

        self.feature_importances = feature_importances

        self.k = k

    def fit(self, X, y=None):

        self.feature_indices_ = np.argsort(self.feature_importances)[-self.k:]

        return self

    def transform(self, X):

        return X[:, self.feature_indices_]
pipeline_f_select = Pipeline([

    ("standardisation", pre_process), 

    ("feature_selection", TopFeatureSelector(feature_importances, k=8)),

])



# pipeline_f_select.fit_transform(X.copy())
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
def print_best(estimators):

    print(f"Best: {estimators.best_score_} using {estimators.best_params_}")
svc_model = make_pipeline(pipeline_f_select, SVC(random_state=42))



param = dict( 

    svc__kernel = [ 'linear' , 'poly' , 'rbf' , 'sigmoid' ],

    pipeline__feature_selection__k = [2,3,4,5,6,7,8],

    svc__C = [0.1, 0.5, 1, 1.5, 2]

)



prediction_SVC = GridSearchCV(svc_model, param ,cv=5, scoring="accuracy", return_train_score=True)

prediction_SVC.fit(X_train, y_train)



print_best(prediction_SVC)
rf_model = make_pipeline(pipeline_f_select, RandomForestClassifier(random_state=42))



param = dict( 

    randomforestclassifier__n_estimators = [25, 50, 75, 100],

    pipeline__feature_selection__k = [2,3,4,5,6,7,8],

    randomforestclassifier__max_depth = [2,4,6,8, None]

)



prediction_rf = GridSearchCV(rf_model, param ,cv=5, scoring="accuracy", return_train_score=True)

prediction_rf.fit(X_train, y_train)



print_best(prediction_rf)
knn_model = make_pipeline(pipeline_f_select, KNeighborsClassifier())



param = dict( 

    kneighborsclassifier__n_neighbors = np.arange(5,20),

    pipeline__feature_selection__k = [2,3,4,5,6,7,8])



prediction_knn = GridSearchCV(knn_model, param ,cv=5, scoring="accuracy", return_train_score=True)

prediction_knn.fit(X_train, y_train)



print_best(prediction_knn)
# keep the best model SVC with a rbf kernel 4 features and C = 0.5

model = prediction_SVC.best_estimator_
model.score(X_test, y_test)