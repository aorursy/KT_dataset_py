from sklearn import preprocessing, tree

from sklearn.base import BaseEstimator, TransformerMixin

from collections import Counter

import pandas as pd

import numpy as np
#import data

train_raw_data = pd.read_csv('../input/titanic/train.csv')

test_raw_data = pd.read_csv('../input/titanic/test.csv')
class TukeyOutlierRemover(BaseEstimator, TransformerMixin):



    __outliers = []

    __outliers_per_feature = {}



    def __init__(self, n, features):

        self.n = n

        self.features = features



    def transform(self, X):

        return X.drop(self.__outliers, axis=0)



    def fit(self, X, y=None):

        outliers = []

        for col in self.features:

            feature_outliers = []

            feature_outliers.extend(X[X[col].isna()].index)

            Q1 = np.percentile(X[col].dropna(), 25)

            Q3 = np.percentile(X[col].dropna(), 75)

            IQR = Q3 - Q1

            outlier_step = 1.5 * IQR

            feature_outliers.extend(X[(X[col] < Q1 - outlier_step) | (X[col] > Q3 + outlier_step)].index)

            outliers.extend(feature_outliers)

            self.__outliers_per_feature[col] = feature_outliers

        self.__outliers = list(key for key, value in Counter(outliers).items() if value >= self.n)

        return self



    def get_outliers(self):

        return self.__outliers



    def get_outliers_per_feature(self, feature=None):

        if feature is not None:

            return self.__outliers_per_feature[feature]

        else:

            return self.__outliers_per_feature

outlierRemover = TukeyOutlierRemover(3, ['Age','SibSp','Parch','Fare'])

train_raw_data = outlierRemover.fit_transform(train_raw_data)

print(outlierRemover.get_outliers())
joint_raw_data = pd.concat(objs=[train_raw_data, test_raw_data], axis=0, sort=False).reset_index(drop=True)
#OHE

#input column values must be int

def train_ohe(data): 

    ohe = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')

    ohe.fit(data.drop(['PassengerId', 'Survived'], axis=1))

    return ohe



def transform_ohe(data, ohe): 

    columns_to_remove = ['PassengerId']

    if 'Survived' in data.columns:

        columns_to_remove.append('Survived')

    ohe_labels = ohe.transform(data.drop(columns_to_remove, axis=1, inplace=False))

    new_data = pd.concat([data['PassengerId'].reset_index(drop=True),pd.DataFrame(ohe_labels)],axis=1)

    if 'Survived' in columns_to_remove:

        new_data = pd.concat([new_data, data['Survived'].reset_index(drop=True)],axis=1)

    return new_data
def generate_binary_columns(data):

#    data['is_Alone'] = (data['cat_Family_Members']==0)

#    data['is_big_Family'] = (data['cat_Family_Members']==3)

    data['is_Child'] = (data['Age']<=5)

    data['is_Elder'] = (data['Age']>=60)

    data['is_Officer'] = (data['Title']=='Officer')

    data['is_Royalty'] = (data['Title']=='Royalty')

    data['is_Male'] = (data['Title']=='Mr')

    data['is_Female'] = ((data['Title']=='Miss') | (data['Title']=='Mrs'))

    data['is_First_Cabin'] = (data['Cabin']==0)

    data['is_Last_Cabin'] = (data['Cabin']==3)

    return data

Title_Dictionary = {

    "Capt": "Officer",

    "Col": "Officer",

    "Major": "Officer",

    "Jonkheer": "Royalty",

    "Don": "Royalty",

    "Sir" : "Royalty",

    "Dr": "Officer",

    "Rev": "Officer",

    "the Countess":"Royalty",

    "Dona" : "Mrs",

    "Mme": "Mrs",

    "Mlle": "Miss",

    "Ms": "Mrs",

    "Mr" : "Mr",

    "Mrs" : "Mrs",

    "Miss" : "Miss",

    "Master" : "Master",

    "Lady" : "Royalty"

}



def feature_eng(data):

    def get_ticket_prefix(ticket_data):

        def take_prefix(ticket):

            ticket = ticket.replace(".","").replace("/","").strip().split(' ')

            return ticket[0]

        ticket_data = ticket_data.apply(lambda x: take_prefix(x) if not x.isdigit() else "X")

        return ticket_data



    data['Cabin'] = data['Cabin'].fillna('X').str[0:1]

#    data['Cabin'] = data['Cabin'].replace(['F', 'G', 'T', 'A'], 'Internal')

#    data['Cabin'] = data['Cabin'].map( {'F': 'Internal', 'G': 'Internal', 'T': 'Internal', 'A': 'Internal'})

    data['Title'] = data['Name'].apply(lambda x : x.split(",")[1].split(".")[0].strip())

    data['Title'] = data['Title'].map(Title_Dictionary)

#    data['Title'] = data['Title'].map({'Master': 'Rare', 'Royalty': 'Rare', 'Officer': 'Rare'})

    data['Family_Members'] = data['SibSp'].astype(int) + data['Parch'].astype(int) + 1

#    data['cat_Family_Members'] = pd.cut(x=data['Family_Members'], right=False, bins=(1,2,3,5,99), labels=[0,1,2,3])

    data['cat_Age'] = pd.cut(x=data['Age'].dropna(), right=True, bins=(0,3,12,60,100), labels=[0,1,2,3])

    data['log_Fare'] = data['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

    data['Pclass'] = data['Pclass'].astype("category")                                                  

    data['Sex'] = data['Sex'].map( {'female': 0, 'male': 1})

    #data['Ticket'] = get_ticket_prefix(data['Ticket'])

    return data

def to_integer(data, feature):

    le = preprocessing.LabelEncoder()

    le.fit(data[feature])

    data[feature] = le.transform(data[feature])

    

def to_numeric_feature(data):

    columns_to_dummie = ['Title', 'Embarked'] # ##duas juntas

    dum = pd.get_dummies(data[columns_to_dummie], prefix=columns_to_dummie)

    return pd.concat([data, dum], axis=1) 
joint_data = feature_eng(joint_raw_data)



#to_integer(joint_data, 'Title')

#to_integer(joint_data, 'Cabin')

#to_integer(joint_data, 'Embarked')



#joint_data = generate_binary_columns(joint_data)

joint_data = to_numeric_feature(joint_data)



# drop high std features

#joint_data.drop(labels=['Title_Royalty', 'Cabin_G','Cabin_T', 'Cabin_F', 'Cabin_A'], inplace=True, axis=1)
joint_data.head()
class DropFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, features_to_remove):

        self.features_to_remove = features_to_remove



    def transform(self, X):

        X = X.drop(labels=self.features_to_remove, axis=1).astype(float)

        #print(X.head())

        return X



    def fit(self, X, y=None):

        return self





class DfNanFiller(BaseEstimator, TransformerMixin):

    def __init__(self, features):

        self.features = features



    def transform(self, X):

        transformed_X = X.copy()

        for feature in self.features:

            if X[feature].dtype is np.float64:

                transformed_X[feature] = X[feature].fillna(X[feature].dropna().median())

            else:

                transformed_X[feature] = X[feature].fillna(X[feature].mode(dropna=True)[0])

        return transformed_X



    def fit(self, X, y=None):

        return self





class FeatureNanFiller(BaseEstimator, TransformerMixin):

    def __init__(self, nanFeature, base_features):

        self.nanFeature = nanFeature

        self.base_features = base_features



    def fit(self, X, y=None):

        return self



    def transform(self, X):

        if len(self.nanFeature) == 0:

            return X

        transformed_X = X.copy()

        nan_indexes = X[X[self.nanFeature].isna()].index

        for i in nan_indexes:

            median = X[self.nanFeature][self.__get_df_slicer(X, i)].median()

            if not np.isnan(median):

                transformed_X.loc[i, self.nanFeature] = median

            else:

                transformed_X.loc[i, self.nanFeature] = X[self.nanFeature].median()

        return transformed_X



    def __get_slice_expression(self, X, feature, i):

        return (X[feature] == X.loc[i, feature])



    def __get_df_slicer(self, X, index):

        expression = self.__get_slice_expression(X, self.base_features[0], index)

        for i in range(1, len(self.base_features)):

            expression = expression & self.__get_slice_expression(X, self.base_features[i], index)

        return expression
from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.preprocessing import normalize, quantile_transform, FunctionTransformer, MinMaxScaler, StandardScaler
transformators = []

transformators.append(("dfNaFiller", DfNanFiller(['Title', 'Cabin', 'Embarked'])))

transformators.append(("ageNaFiller", FeatureNanFiller('Age', ['Family_Members', 'Title', 'Pclass'])))

transformators.append(("fareNaFiller", FeatureNanFiller('Fare', ['Cabin', 'Pclass'])))

#transformators.append(("dropFeatures", DropFeatures(['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'cat_Age', 'Embarked'])))

transformators.append(("dropFeatures", DropFeatures(['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'cat_Age', 'Cabin', 'Title', 'Embarked'])))

#transformators.append(("printHead", PrintHead()))



transformators.append(("quantilesTransform", FunctionTransformer(quantile_transform)))

#transformators.append(("unitNorm", FunctionTransformer(normalize)))

#transformators.append(("minMax", MinMaxScaler()))

#transformators.append(("scaler", StandardScaler()))



preprocess_pipeline = Pipeline(transformators)

from sklearn import ensemble, linear_model, model_selection

from sklearn import ensemble, linear_model, model_selection, neighbors, neural_network, svm



def divide_data(data):

    ids = data['PassengerId'].astype({"PassengerId": int})

    X = data.drop(labels=['PassengerId', 'Survived'], axis=1)

    if not data['Survived'].isna().any():

        Y = data['Survived'].astype({"Survived": int})

        return ids, X, Y

    else:

        return ids, X

train_data = joint_data[joint_data['PassengerId'].isin(train_raw_data['PassengerId'])]

test_data = joint_data[joint_data['PassengerId'].isin(test_raw_data['PassengerId'])]



passenger_ids_training, X_training, Y_training = divide_data(train_data)

passenger_ids_test, X_test = divide_data(test_data)

def create_pipeline(preprocess_pipeline, model_name, estimator):

    classification_pipeline = Pipeline([(model_name, estimator)])

    full_pipeline = Pipeline([("preprocess", preprocess_pipeline),

                            ("classfication", classification_pipeline)])

    return full_pipeline
def preprocess_without_pipeline(data):

    naFiller = DfNanFiller(['Title', 'Cabin', 'Embarked'])

    ageFeatureNanFiller = FeatureNanFiller('Age', ['Family_Members', 'Title', 'Pclass'])

    fareNaFiller = FeatureNanFiller('Fare', ['Cabin', 'Pclass'])

    dropFeatures = DropFeatures(['Name', 'SibSp', 'Parch', 'Ticket', 'Fare', 'cat_Age', 'Cabin', 'Title', 'Embarked'])

    minMax = MinMaxScaler()

    quantiles = FunctionTransformer(quantile_transform)

    

    data = naFiller.fit_transform(data)

    data = ageFeatureNanFiller.fit_transform(data)

    data = fareNaFiller.fit_transform(data)

    data = dropFeatures.fit_transform(data)

    data = minMax.fit_transform(data)

    #data = quantiles.fit_transform(data)

    return data

def compare_models(proprocess_pipeline, estimators, X, Y, metrics=['accuracy','precision','recall','f1']):

    scores = {}

    score_means = {}

    score_std = {}

    for estimator in estimators:

        name = type(estimator).__name__

        classifier_pipeline = create_pipeline(preprocess_pipeline, name, estimator)

       # scores[name] = model_selection.cross_validate(estimator, X, n_jobs=-1, y=Y, scoring=metrics, cv=kfold, return_train_score=False)

        scores[name] = model_selection.cross_validate(classifier_pipeline, X, n_jobs=-1, y=Y, scoring=metrics, cv=kfold, return_train_score=False)

        score_means[name] = {key:values.mean() for key,values in scores[name].items()}

        score_std[name] = {key:values.std() for key,values in scores[name].items()}        

    return scores, score_means, score_std



def print_scores_as_table(score_means, score_std, print_performance_metrics=True):

    for classifier_name, score_mean_values in score_means.items():

        score_std_values = score_std[classifier_name]

        print(classifier_name)

        print("   metric \t\tmean \t\tstd")

        if not print_performance_metrics:

            score_mean_values.pop('fit_time', False)

            score_mean_values.pop('score_time', False)

            score_std_values.pop('fit_time', False)

            score_std_values.pop('score_time', False)

            for metric,mean in score_mean_values.items():

                print(metric+" = "+str(mean)+"   "+str(score_std_values[metric]))

        print("--------------------------------")



    
kfold = model_selection.StratifiedKFold(n_splits=10)

random_state = 2

n_estimators=1000



estimators = []

estimators.append(svm.SVC(random_state=random_state, gamma='auto'))

estimators.append(tree.DecisionTreeClassifier(random_state=random_state))

estimators.append(neural_network.MLPClassifier(random_state=random_state))

estimators.append(neighbors.KNeighborsClassifier())

log_reg = linear_model.LogisticRegression(random_state=random_state, max_iter=1000, solver='lbfgs')

estimators.append(log_reg)

estimators.append(ensemble.RandomForestClassifier(random_state=random_state, n_estimators=n_estimators))

estimators.append(ensemble.ExtraTreesClassifier(random_state=random_state, n_estimators=n_estimators))

estimators.append(ensemble.GradientBoostingClassifier(random_state=random_state, n_estimators=n_estimators))

estimators.append(ensemble.AdaBoostClassifier(log_reg ,random_state=random_state,learning_rate=0.1))



scores, score_means, score_std = compare_models(preprocess_pipeline, estimators, X_training, Y_training)

print_scores_as_table(score_means, score_std, False)

def train_and_save_submission(models, training_X, training_Y, test_data, test_passenger_ids, name=None):    

    for model in models:

        algorithm = model.fit(training_X, training_Y)

        predictions = model.predict(test_data)



        submission = pd.DataFrame({ 'PassengerId': test_passenger_ids.astype(int), 'Survived': predictions.astype(int) })

        if name is None:

            name = type(model).__name__

        submission.to_csv(name+".csv", index=False)

        
k = 10

n_jobs = -1
X_training2 = preprocess_without_pipeline(X_training)

X_test2 = preprocess_without_pipeline(X_test)
gridsRFC = ensemble.RandomForestClassifier()



param_grid = {

    'n_estimators': [15, 30, 100, 200],

    'max_depth': [10, 50, 100, 200, 300],

    'min_samples_split': [2, 10, 20, 30],

    'min_samples_leaf': [2, 3, 4, 6],

    'max_features': ['sqrt', 'log2'],

    'criterion' : ["gini", "entropy"],

    'max_leaf_nodes': [30, 50, 100, 200, 300],

    'oob_score' : [True, False]

#    'classfication__min_impurity_decrease': [0], #[0, 0.1, 0.5, 1],

#    'classfication__min_weight_fraction_leaf': [0, 0.001] #[0, 0.1, 0.25, 0.5]

}



randomSearch = model_selection.RandomizedSearchCV(estimator = gridsRFC, param_distributions = param_grid, n_iter = 1000, cv = 3, verbose=2, random_state=42, n_jobs = -1)

randomSearch.fit(X_training2, Y_training)

print(randomSearch.best_params_)



print(randomSearch.best_score_)

print(randomSearch.best_estimator_)



train_and_save_submission([randomSearch.best_estimator_], X_training2, Y_training, X_test2, passenger_ids_test, "gridsRFC")

import matplotlib.pyplot as plt

#import warnings

#warnings.simplefilter(action='ignore', category=FutureWarning)



def plot_learning_curve(estimator, X, y, train_sizes=np.linspace(.1, 1.0, 5), cv=None, metric='accuracy'):

    plt.figure()

    plt.title("Learning Curves")

    plt.xlabel("Training samples")

    plt.ylabel(metric)

    train_sizes, train_scores, test_scores = model_selection.learning_curve(estimator, X, y, cv=cv, train_sizes=train_sizes, scoring=metric)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training accuracy")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross validation accuracy")

    plt.legend(loc="best")

    return plt
g = plot_learning_curve(randomSearch.best_estimator_, X_training2, Y_training, cv=kfold)
g = plot_learning_curve(randomSearch.best_estimator_, X_training2, Y_training, cv=kfold, metric='brier_score_loss')