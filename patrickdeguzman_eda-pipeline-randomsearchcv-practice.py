import numpy as np

import pandas as pd



from time import time



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder,StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator,TransformerMixin

from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, roc_auc_score,make_scorer

from sklearn.model_selection import RandomizedSearchCV,cross_val_score



from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
import plotly_express as px

import seaborn as sns



import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



%matplotlib inline



from IPython.display import display
import os

os.listdir('../input')
training_data = pd.read_csv('../input/census.csv')

testing_data = pd.read_csv('../input/test_census.csv')
explore_training = training_data.copy()
explore_training.head(1)
explore_training.info()
explore_training.hist(figsize=(13,7),bins=20,layout=(2,3))

plt.tight_layout()

plt.show()
explore_training.describe()
explore_training.corr()
sns.jointplot(x='age',y='hours-per-week',data= explore_training)
explore_training['hours-per-week'].sort_values(ascending=False)
px.histogram(explore_training,x='hours-per-week',nbins=25,width=1000,height=400)
explore_training['workclass'].value_counts()
px.histogram(explore_training,x='workclass',y='hours-per-week',histfunc='avg',width=700,height=400)
px.histogram(explore_training,x='education_level',y='hours-per-week',histfunc='avg',width=700,height=400)
explore_training['education_level'].value_counts()
explore_training['marital-status'].value_counts()
explore_training['occupation'].value_counts()
explore_training['relationship'].value_counts()
p = sum(explore_training['relationship']==' Husband')/len(explore_training['relationship'])

print('Percent of our dataset that is labeled "Husband": %.3f'% p)
explore_training['race'].value_counts()
explore_training['sex'].value_counts()
testing_data['sex'].value_counts()
p = sum(explore_training['native-country']==' United-States')/len(explore_training['native-country'])

print('Percent of our dataset that is labeled "United-States": %.3f'% p)
explore_training['native-country'].value_counts()
label_dict = {'<=50K':0,'>50K':1}

training_labels = training_data['income'].map(label_dict)

training_data.drop('income',axis=1,inplace=True)
num_features = ['age','education-num','capital-gain','capital-loss','hours-per-week']

cat_features = ['workclass','education_level','marital-status','occupation',

                'relationship','race','sex','native-country']



num_training_data=training_data[num_features].copy()

cat_training_data=training_data[cat_features].copy()

full_training_data = training_data.copy()
# Creating a custom imputer because SimpleImputer() returns an array. 

    # I want it to return a DataFrame because I'm not comfortable with Pipelines yet and I want to inspect it easily after.

class CustomImputer(BaseEstimator,TransformerMixin):

    def __init__(self,method):

        self.method = method

    def fit(self,X,y=None):

        return self

    def transform(self,X,y=None):

        if self.method == 'median':

            for column in X.columns:

                X[column] = X[column].fillna(X[column].median())

        elif self.method == 'mode':

            for column in X.columns:

                X[column] = X[column].fillna(X[column].mode()[0])            

        return X
# The ordered category mapping that is passed into the transformer below. 

    # Note this is subjective ordering and is not based on any metadata provided.

ordered_categories = {' Preschool':0,' 1st-4th':1,' 5th-6th':2,' 7th-8th':3,

                      ' 9th':4,' 10th':5,' 11th':6,' 12th':7,' HS-grad':8,

                      ' Assoc-acdm':9,' Assoc-voc':10,' Some-college':11,

                      ' Bachelors':12,' Prof-school':13,' Masters':14,' Doctorate':15}
class EduLevelMapper(BaseEstimator,TransformerMixin):

    def __init__(self,ordered_categories,scale=None):

        self.ordered_categories = ordered_categories

        self.scale=scale

    def fit(self,X,y=None):

        return self

    def transform(self,X,y=None):

        # Mapping ordinal values

        X['education_level'] = X['education_level'].map(self.ordered_categories)

        # Normalizing or standardizing the values

        if self.scale == 'normalization':

            x_min = X['education_level'].min()

            x_max = X['education_level'].max()

            x_range = x_max-x_min

            X['education_level']=(X['education_level']-x_min)/x_range

        elif self.scale == 'standardization':

            x_mean = X['education_level'].mean()

            x_std = X['education_level'].std()

            X['education_level'] = (X['education_level']-x_mean)/x_std     

        return X  
class GetDummies(BaseEstimator,TransformerMixin):

    def __init__(self):

        None

    def fit(self,X,y=None):

        return self

    def transform(self,X,y=None):

        return pd.get_dummies(X)
# Note I selected "standardization" rather than "normalization"

cat_pipeline = Pipeline([

    ('imputer',CustomImputer('mode')),

    ('edu_mapper',EduLevelMapper(ordered_categories,'standardization')),

    ('get_dummies',GetDummies())])
# Fitting our categorical data and then transforming it

cat_pipeline_output = cat_pipeline.fit_transform(cat_training_data)
# Category Pipeline Output - note education_level is standardized 

    # and the rest of our categorical features are one-hot encoded

cat_pipeline_output.head()
num_training_data.hist(figsize=(8,5),layout=(2,3))

plt.tight_layout()

plt.show()
# This is the custom transformer that will apply a logarithmic transformation to the skewed features

skewed = ['capital-gain','capital-loss']

class SkewedDistTransformer(BaseEstimator,TransformerMixin):

    def __init__(self,skewed_features):

        self.skewed_features = skewed_features

    def fit(self,X,y=None):

        return self

    def transform(self,X,y=None):

        X[skewed] = X[skewed].apply(lambda x: np.log(x+1))

        return X
# Again, creating a custom scaler (rather than using StandardScaler or MinMaxScaler) 

    # because I want to return a DataFrame

class CustomScaler(BaseEstimator,TransformerMixin):

    def __init__(self,method):

        self.method = method

        None

    def fit(self,X,y=None):

        return self

    def transform(self,X,y=None):

        if self.method == 'normalization':

            for column in X.columns:

                x_min = X[column].min()

                x_max = X[column].max()

                x_range = x_max-x_min

                X[column]=(X[column]-x_min)/x_range

        else:

            for column in X.columns:

                x_mean = X[column].mean()

                x_std = X[column].std()

                X[column] = (X[column]-x_mean)/x_std

        return X
num_pipeline = Pipeline([

    ('imputer',CustomImputer('median')),

    ('transform_skew',SkewedDistTransformer(skewed)),

    ('scaler',CustomScaler('normalization'))

])
# Fitting our numerical data and then transforming it

num_pipeline_output = num_pipeline.fit_transform(num_training_data)
# Numerical Pipeline Output

    # Note capital-gain and loss features were transformed and all features were scaled down

num_pipeline_output.head()
num_training_data.hist(figsize=(8,5),layout=(2,3))

plt.tight_layout()

plt.show()
full_pipeline = ColumnTransformer([

    ("num",num_pipeline,num_features),

    ("cat",cat_pipeline,cat_features),

])
# Fitting and Transforming our full training set

full_pipeline_output = full_pipeline.fit_transform(full_training_data)
# Taking a look at the output

full_pipeline_output
# Sanity Check - Here's our categorical data pipeline output

cat_pipeline_output.head(1)
# Sanity Check - Here's our numerical data pipeline output

num_pipeline_output.head(1)
# Concatenating our pipelines along the second axis (the columns)

num_cat_pipeline_combined = pd.concat([num_pipeline_output,cat_pipeline_output],axis=1)
full_pipeline_output[:,:]
# Sanity Check - Here's our combined data pipeline output

num_cat_pipeline_combined.head(1)
num_cat_pipeline_combined.shape
full_pipeline_output.shape
# Comparing our combined pipelines with our full pipeline output

num_cat_pipeline_combined==full_pipeline_output[:,:]
final_training_data = full_pipeline_output
# Instantiating the 5 classifiers that I want to test.

    # Rather than using full default values, I've selected a few for the cross validation

rf_clf = RandomForestClassifier(n_estimators=100)

svc_clf = SVC(gamma='auto')

tree_clf = DecisionTreeClassifier(max_depth=15)

boost_clf = AdaBoostClassifier(n_estimators=100)

nb_clf = GaussianNB()
def train_cv(learner, X_train, y_train,scoring_function,cv=5): 

    '''

    This function trains a model on training data, scores it, returns cross validation testing results.

    

    Inputs:

       - learner: the learning algorithm to be trained and predicted on

       - X_train: features training set

       - y_train: income training set

       - cv: number of folds used for cross validation testing

    

    Returns:

       - Dictionary containing cross validation testing results

    '''

    scorer = make_scorer(scoring_function)

    

    results = {'name':learner.__class__.__name__}

    # Timing time to train

    start = time()

    cv_scores = cross_val_score(learner,X_train,y=y_train,scoring=scorer,cv=cv)    

    end = time()

    

    # Storing results in the result dictionary

    results['scores'] = cv_scores

    results['average_score'] = cv_scores.mean()

    results['std_dev_score'] = cv_scores.std()

    results['total_time'] = end-start

    results['average_time'] = (end-start)/cv

    

    return results
classifiers = [rf_clf,svc_clf,tree_clf,boost_clf,nb_clf]



model_results = []

for clf in classifiers:

        model_results.append(train_cv(clf,final_training_data,training_labels,scoring_function=roc_auc_score))
result_df = pd.DataFrame(model_results)

result_df.sort_values('average_score',ascending=False).drop(['scores'],axis=1)
boost_parameters = dict(n_estimators=list(np.arange(50,300,10)),

                   learning_rate=list(np.linspace(.5,2.5,20)),)

rf_parameters = dict(n_estimators=list(np.arange(50,300,10)),

                    criterion=['gini','entropy'],

                    min_samples_split=list(np.arange(2,500,5)),)
scorer = make_scorer(roc_auc_score)
boost_rscv_obj = RandomizedSearchCV(boost_clf,boost_parameters,n_iter=30,scoring=scorer,n_jobs=-1,cv=5)
rf_rscv_obj = RandomizedSearchCV(rf_clf,rf_parameters,n_iter=30,scoring=scorer,n_jobs=-1,cv=5)
boost_rscv_fit = boost_rscv_obj.fit(final_training_data,training_labels)

rf_rscv_fit = rf_rscv_obj.fit(final_training_data,training_labels)
best_boost_clf = boost_rscv_fit.best_estimator_

best_rf_clf = rf_rscv_fit.best_estimator_

best_classifiers = [best_boost_clf,best_rf_clf]



best_model_results = []

for clf in best_classifiers:

        best_model_results.append(train_cv(clf,final_training_data,

                                           training_labels,scoring_function=roc_auc_score))

        

best_result_df = pd.DataFrame(best_model_results)

best_result_df.sort_values('average_score',ascending=False).drop(['scores'],axis=1)
final_model = best_boost_clf

final_model
testing_ids = testing_data['Unnamed: 0']

testing_data.drop('Unnamed: 0',axis=1,inplace=True)
# final_testing_data = full_pipeline.transform(testing_data)

final_testing_data = full_pipeline.transform(testing_data)
final_predictions = final_model.predict(final_testing_data)

# final_predictions = final_model.predict_proba(final_testing_data)
submission = pd.concat([testing_ids,pd.DataFrame(final_predictions)],axis=1)
submission.columns = ['id','income']
submission.head()
submission[['id', 'income']].to_csv("submission.csv", index=False)