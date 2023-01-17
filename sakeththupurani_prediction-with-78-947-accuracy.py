import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.metrics import *
%matplotlib inline
# A custom built class to ease the process of defining, training and evaluating different models to compare and get the 
# best model that scores high in the leaderboard.
class StackModelClassifier:
    def __init__(self, models=None):
        self.models = {}
        if len(models)>0:
            for name, model, grid in models:
                if grid is not None:
                    self.models[name]=[GridSearchCV(model, param_grid=grid['param_grid'], cv=grid['cv'], scoring=grid['scoring'], 
                                                  refit=True), 0]
                else:
                    self.models[name]=[model, 0]
    
    def add(self, models):
        '''
            Adds a new model passed as argument into the existing stack of models
            Inputs: 
            models = list(tuple(str, sklearn_model_class, dict(tuning parameters)))
                str = custom name for the model
                sklearn_model_class = Any of the sklearn models
                tuning parameters = Arguments to be passed to GridSearchCV to tune the model given
            Example,
                models = [("log", LogisticRegression(), dict(param_grid={"Penalty": ['l1', 'l2'], 
                                                                          "C": [0.001, 0.01, 0.1, 1]}, 
                                                              cv=3, 
                                                              scoring="f1_macro")), 
                          ("svm", SVC(), dict(param_grid={"Kernel": ['linear', 'rbf'], 
                                                            "C": [0.001, 0.01, 0.1, 1]}, 
                                                cv=3, 
                                                scoring="f1_macro")), 
                          ("tree", DecisionTreeClassifier(), None)]
            see the Attribute models to view the added model and use it.
        '''
        tmp=0
        for name, model, grid in models:
                if grid is not None:
                    self.models[name]=[GridSearchCV(model, param_grid=grid['param_grid'], cv=grid['cv'], scoring=grid['scoring'], 
                                                  refit=True), 0]
                else:
                    self.models[name]=[model, 0]
                tmp+=1
        print(f"Successfully added {tmp} models to the stack")
        print(f"Total models in the stack are : {len(self.models)}")
    
    def train(self, X_train, y_train):
        '''
            Trains the models in the stact one after the other.
            Trains each model only once no mattar how many times this method is called by the object.
        '''
        tmp=0
        for model in self.models:
            if self.models[model][1]==0:
                self.models[model][0].fit(X_train, y_train)
                print(f"{model} is trained successfully")
                self.models[model][1] = -1
                tmp+=1
            elif self.models[model][1]==1:
                self.models[model][0].fit(self.train_predictions[X_train_nl], y_train)
                print(f"{model} is trained successfully")
                self.models[model][1] = -2
                tmp+=1
            else:
                print(f"{model} is already in the trained models list. So ignoring this model in this iteration of training")
        print(f"Trained {tmp} models in this iteration of training")
        
    def evaluate(self, X_train, y_train, X_test=None, y_test=None, scoring=None):
        '''
            Evaluates the model on the training set with the specified scoring
            If test sets are provided, the evaluation is performed both on train and test set
            Default scoring is "f1_macro" for Classification and "rmse" for Regression problems
        '''
        self.scores_df = pd.DataFrame(data=None, columns=["Model", "Train_Test", "Scoring", "Score"])
        self.train_predictions = pd.DataFrame()
        self.test_predictions = pd.DataFrame()
        columns = []
        for model in self.models:
            columns.append(model)
            if self.models[model][1]==-1:
                self.train_predictions=pd.concat([self.train_predictions, 
                                                 pd.DataFrame(self.models[model][0].predict(X_train).reshape(-1,1))], axis=1)
            else:
                self.train_predictions=pd.concat([self.train_predictions, 
                                                 pd.DataFrame(self.models[model][0].predict(self.train_predictions[self.X_train_nl]).reshape(-1,1))], 
                                                axis=1)
            self.train_predictions.columns = columns
            if scoring is not None:
                self.scores_df = self.scores_df.append(dict(Model=model, 
                                                               Train_Test='Train', 
                                                               Scoring=scoring.__name__, 
                                                               Score=scoring(y_train, 
                                                                             self.train_predictions[model].values.reshape(-1,1))), 
                                                          ignore_index=True)
            else:
                self.scores_df = self.scores_df.append(dict(Model=model, 
                                                               Train_Test='Train', 
                                                               Scoring='f1_score', 
                                                               Score=f1_score(y_train, 
                                                                              self.train_predictions[model].values.reshape(-1,1))), 
                                                          ignore_index=True)
        columns.clear()
        if X_test is not None and y_test is not None:
            for model in self.models:
                columns.append(model)
                if self.models[model][1] == -1:
                    self.test_predictions=pd.concat([self.test_predictions, 
                                                     pd.DataFrame(self.models[model][0].predict(X_test).reshape(-1,1))], axis=1)
                else:
                    self.test_predictions=pd.concat([self.test_predictions, 
                                                     pd.DataFrame(self.models[model][0].predict(X_test_predictions[self.X_test_nl]).reshape(-1,1))], 
                                                    axis=1)
                self.test_predictions.columns = columns
                if scoring is not None:
                    self.scores_df = self.scores_df.append(dict(Model=model, 
                                                               Train_Test='Test', 
                                                               Scoring=scoring.__name__, 
                                                               Score=scoring(y_test, 
                                                                             self.test_predictions[model].values.reshape(-1,1))), 
                                                          ignore_index=True)
                else:
                    self.scores_df = self.scores_df.append(dict(Model=model, 
                                                                   Train_Test='Test', 
                                                                   Scoring='f1_score', 
                                                                   Score=f1_score(y_test, 
                                                                                  self.test_predictions[model].values.reshape(-1,1))), 
                                                          ignore_index=True)                                                        
                            
        columns.clear()
        plt.figure(figsize=(10,8))
        sns.set_style('darkgrid')
        sns.barplot(data=self.scores_df, y='Model', x='Score', orient='h', hue='Train_Test')
        plt.legend(loc=7, bbox_to_anchor=(1.12,0.96))        
        plt.title("Models By Train and Test socres\nScoringUsed: {}".format(scoring.__name__ if scoring is not None else 'f1_score'))
        plt.show()
    
    def evaluation_metrics(self, model, metrics):
        '''
            Get the detailed evaluation metrics for the selected model to view more insights
            Inputs:
            model = Custom name defined for your model while passing into the stack (To view, use self.models.keys())
            metrics = sklearn.metrics methods you want to view.
        '''
        if model not in self.train_predictions.columns and model not in self.test_predictions.columns:
            print("Please spell the model name correctly. Current models in your object are")
            print(list(self.models.keys()))
        else:
            print("_"*20+f"{metrics.__name__} on Train data"+"_"*20)
            print(metrics(y_train, self.train_predictions[model].values.reshape(-1,1)))
            if model in self.test_predictions.columns:
                print("_"*20+f"{metrics.__name__} on Test data"+"_"*20)
                print(metrics(y_test, self.test_predictions[model].values.reshape(-1,1)))
    
    def ensemble_models(self, models, ensemble, **kwargs):
        '''
            Create selected ensemble of the models passed
        '''
        model_list=[]
        for model in models:
            model_list.append((model, self.models[model][0]))
        print(model_list)
        self.models[str(ensemble.__name__)+"_"+str(kwargs['voting'])]=[ensemble(model_list, **kwargs), 0]
        
    '''def retrain(self, X_train, y_train):
        for model in self.models:
            self.models[model][0].fit(X_train, y_train)
            print(f"{model} is trained successfully")'''
    
    def add_stack_models(self, aggregators, models=None):
        if models is None:
            self.X_train_nl=self.train_predictions.columns
            self.X_test_nl=self.test_predictions.columns if len(self.test_predictions.columns)!=0 else None
        else:
            self.X_train_nl=models
            self.X_test_nl=models
        self.add(aggregators)
        
    def predictions(self, X_test, models=None):
        self.final_predictions = pd.DataFrame()
        self.out_predictions = pd.DataFrame()
        cols=[]
        if models is None:
            models=list(self.models.keys())
        for model in self.models:
            cols.append(model)
            if self.models[model][1] == -1:
                self.final_predictions=pd.concat([self.final_predictions, 
                                                 pd.DataFrame(self.models[model][0].predict(X_test).reshape(-1,1))], axis=1)
                self.final_predictions.columns = cols
        cols.clear()
        for model in models:
            if self.models[model][1] == -1:
                cols.append(model)
                self.out_predictions=pd.concat([self.out_predictions, 
                                                 pd.DataFrame(self.models[model][0].predict(X_test).reshape(-1,1))], axis=1)
            elif self.models[model][1] == -2:
                cols.append(model)
                self.out_predictions=pd.concat([self.out_predictions, 
                                               pd.DataFrame(self.models[model][0].predict(self.final_predictions[models]).reshape(-1,1))], 
                                              axis=1)
            else:
                print(f"The {model} is not trained. So ignoring this model")
            self.out_predictions.columns = cols
    

titanic_train = pd.read_csv("/kaggle/input/titanic/train.csv", index_col="PassengerId")
titanic_test = pd.read_csv("/kaggle/input/titanic/test.csv", index_col="PassengerId")
titanic_set = titanic_train.copy()
titanic_set.head(10)
titanic_set.info()
titanic_set.describe()
titanic_set.describe(include=[np.object])
plt.figure(figsize = (10,8))
sns.FacetGrid(data=titanic_set, row='Sex', col='Pclass').map(sns.countplot, "Survived")
plt.show()
sns.FacetGrid(data=titanic_set, hue="Survived", height=5, aspect=2).map(sns.kdeplot, "Age")
plt.legend()
plt.title("Distribution of Age by Survived")
plt.show()
sns.countplot(data=titanic_set, x="SibSp", hue="Survived")
plt.legend(loc='upper right')
plt.title("Survived By Siblings & Spouse")
plt.show()
sns.countplot(data=titanic_set, x="Parch", hue="Survived")
plt.legend(loc='upper right')
plt.title("Survived By Parent & Child")
plt.show()
sns.FacetGrid(data=titanic_set, col="Pclass").map(
sns.pointplot, "Embarked", "Survived","Sex")
plt.legend()
plt.show()
sns.FacetGrid(data=titanic_set, hue="Survived", height=5, aspect=2).map(sns.kdeplot, "Fare")
plt.legend()
plt.title("Fare distribution by Survived")
plt.show()
# A function to add a new calculated column that tells the information whether a particular passenger on board is alone(1) or is
# with his family members(0)
def IsAlone(cols):
    sibsp=cols[0]
    parch=cols[1]
    return int(sibsp+parch == 0)
titanic_set['IsAlone'] = titanic_set[['SibSp', 'Parch']].apply(IsAlone, axis=1)
sns.countplot(data=titanic_set, x="IsAlone", hue="Survived")
plt.legend(loc='upper right')
plt.title("Survived By IsAlone feature")
plt.show()
# A function to add a new engineered feature which tells the title of a passneger basesd on the feature Name
# Titles that are incorrectly spelled and those that occur in very less frequency are handled
def ExtractTitle(Name):
    title = Name.split(', ')[1].split('.')[0]
    if title in ["Miss", "Ms", "Mme", "Mlle"]:
        return "Miss"
    elif title in ["Mr", "Mrs", "Master"]:
        return title
    else:
        return "Misc"
titanic_set['Title'] = titanic_set['Name'].apply(ExtractTitle)
100*titanic_set['Title'].value_counts()/len(titanic_set['Title'])
titanic_set['Cabin'].unique()
# A function to extract the cabin first letter as a new feature overriding the cabin itself
# A value T is also put as unknown (U) as it is rare and is not present in the test data set 
def cabin(cabin):
    if pd.isnull(cabin) or cabin[0]=="T":
        return "U"
    else:
        return cabin[0]
titanic_set['Cabin'] = titanic_set['Cabin'].apply(cabin)
titanic_set['Cabin'].value_counts()
sns.pointplot(data=titanic_set, x="Cabin", y="Survived", hue="Sex")
plt.title("Survived By Cabin")
plt.legend(loc=7, bbox_to_anchor=[1.24,0.93])
plt.show()
titanic_set['Embarked'].value_counts()
# A function to fill the missing embarked values - equivalent to imputing the most frequent value which is S
def fill_embark(embarked):
        if embarked == "S" or pd.isnull(embarked):
            return "S"
        else:
            return embarked
titanic_set['Embarked'] = titanic_set['Embarked'].apply(fill_embark)
titanic_set['Age'] = titanic_set.groupby(['Sex', 'Pclass', 'Title']).Age.apply(lambda x:x.fillna(x.median()))
# A function for binning the age into different groups based on the Survival rate as per the distribution plot
def age_bins(age):
        if age in range(0,19):
            return 1
        elif age in range(19,36):
            return 2
        elif age in range(36, 81):
            return 3
        elif age>=81:
            return 4
        else:
            return 0
titanic_set['Age'] = titanic_set['Age'].apply(age_bins)
titanic_set['Age'].value_counts()
sns.countplot(data=titanic_set, x='Age', hue="Survived")
plt.title("Survived by Age Groups")
plt.show()
sns.FacetGrid(data=titanic_set, col="Sex",row="IsAlone", hue="Survived").map(sns.countplot, "Age", alpha=0.5)
plt.legend()
plt.show()
# A function for binning the fare based on the distribution plot into different groups
def fare_bins(fare):
        if fare<0:
            return 0
        elif fare in range(0,21):
            return 1
        elif fare in range(21, 201):
            return 2
        else:
            return 3
titanic_set['Fare'] = titanic_set['Fare'].apply(fare_bins)
titanic_set.info()
# A function to encode sex feature
def encode_sex(sex):
    return int(sex=="male")
titanic_set['Sex'] = titanic_set['Sex'].apply(encode_sex)
titanic_set.drop(['Name', 'Ticket', 'SibSp', 'Parch'], axis=1, inplace=True)
titanic_set=pd.concat([titanic_set, pd.get_dummies(titanic_set['Title'], prefix='Title', drop_first=True)], axis=1)
titanic_set=pd.concat([titanic_set, pd.get_dummies(titanic_set['Cabin'], prefix="Cabin", drop_first=True)], axis=1)
titanic_set=pd.concat([titanic_set, pd.get_dummies(titanic_set['Embarked'], prefix="EmbarkedAt", drop_first=True)], axis=1)
titanic_set.drop(['Title', 'Cabin', 'Embarked'], axis=1, inplace=True)
titanic_set.info()
titanic_set.head(15)
# Building a custom processor to take care of all the above data preprocessing steps to any new incoming sample
class CustomProcesser(BaseEstimator, TransformerMixin):
    def __init__(self, pseudo=None):
        self.pseudo = pseudo
    def fit(self, X, y=None):
        return self
    def IsAlone(self, cols):
        sibsp=cols[0]
        parch=cols[1]
        return int(sibsp+parch == 0)
    def ExtractTitle(self, Name):
        title = Name.split(', ')[1].split('.')[0]
        if title in ["Miss", "Ms", "Mme", "Mlle"]:
            return "Miss"
        elif title in ["Mr", "Mrs", "Master"]:
            return title
        else:
            return "Misc"
    def cabin(self, cabin):
        if pd.isnull(cabin) or cabin[0]=="T":
            return "U"
        else:
            return cabin[0]
    def fill_embark(self, embarked):
        if embarked == "S" or pd.isnull(embarked):
            return "S"
        else:
            return embarked
    def age_bins(self, age):
        if age in range(0,19):
            return 1
        elif age in range(19,36):
            return 2
        elif age in range(36, 81):
            return 3
        elif age>=81:
            return 4
        else:
            return 0
    def encode_sex(self, sex):
        return int(sex=="male")
    def fare_bins(self, fare):
        if fare<0:
            return 0
        elif fare in range(0,21):
            return 1
        elif fare in range(21, 201):
            return 2
        else:
            return 3
    def transform(self, X, y=None):
        X['IsAlone'] = X[['SibSp', 'Parch']].apply(self.IsAlone, axis=1)
        X['Title'] = X['Name'].apply(self.ExtractTitle)
        X['Cabin'] = X['Cabin'].apply(self.cabin)
        X['Embarked'] = X['Embarked'].apply(self.fill_embark)
        X['Age'] = X.groupby(['Sex', 'Pclass', 'Title']).Age.apply(lambda x:x.fillna(x.median()))
        X['Age'] = X['Age'].apply(self.age_bins)
        X['Sex'] = X['Sex'].apply(self.encode_sex)
        X['Fare'] = X['Fare'].fillna(method='ffill')
        X['Fare'] = X['Fare'].apply(self.fare_bins)
        X.drop(['Name', 'Ticket', 'SibSp', 'Parch'], axis=1, inplace=True)
        X=pd.concat([X, pd.get_dummies(X['Title'], prefix='Title', drop_first=True)], axis=1)
        X=pd.concat([X, pd.get_dummies(X['Cabin'], prefix="Cabin", drop_first=True)], axis=1)
        X=pd.concat([X, pd.get_dummies(X['Embarked'], prefix="EmbarkedAt", drop_first=True)], axis=1)
        X.drop(['Title', 'Cabin', 'Embarked'], axis=1, inplace=True)
        return X
# building a column transformer pipeline
Features = list(titanic_train.drop('Survived', axis=1).columns)
pipeline = ColumnTransformer([
    ("transformer", CustomProcesser(), Features)
])
titanic_features, titanic_target = pipeline.fit_transform(titanic_train[Features]), titanic_train['Survived']
titanic_features
titanic_features.shape
titanic_features_test = pipeline.fit_transform(titanic_test)
titanic_features_test.shape
#Defining the stack of models to be trained and evaluated upon
stack_clf2=StackModelClassifier([
    ("logistic_regression", LogisticRegression(), None),
    ("svm", SVC(), None), 
    ("svm_tuned", SVC(), dict(param_grid={'kernel': ['linear', 'rbf']}, 
                             cv=5,
                             scoring='f1_macro')), 
    ('knn', KNeighborsClassifier(5), None), 
    ('forest_tuned', RandomForestClassifier(), dict(param_grid=dict(     
                                max_depth = [n for n in range(9, 14)],     
                                min_samples_split = [n for n in range(4, 11)], 
                                min_samples_leaf = [n for n in range(2, 5)],     
                                n_estimators = [n for n in range(10, 60, 10)],
                            ), 
                                                   cv=5, 
                                                   scoring=None))
])
stack_clf2.train(titanic_features, titanic_target)
stack_clf2.evaluate(titanic_features, titanic_target)
#Adding a hard voter on top of the above trained models
stack_clf2.ensemble_models(['logistic_regression', 'svm', 'svm_tuned', 'knn', 'forest_tuned'], 
                         VotingClassifier, voting='hard')
stack_clf2.train(titanic_features, titanic_target)
stack_clf2.evaluate(titanic_features, titanic_target)
#Stacking up the models predictions as a feed to the new model which is a tuned random forest model in this case
stack_clf2.add_stack_models([
    ('forest_tuned_stack', RandomForestClassifier(), dict(param_grid=dict(     
                                max_depth = [n for n in range(9, 14)],     
                                min_samples_split = [n for n in range(4, 11)], 
                                min_samples_leaf = [n for n in range(2, 5)],     
                                n_estimators = [n for n in range(10, 60, 10)],
                            ), 
                                                   cv=5, 
                                                   scoring=None))
], ['logistic_regression', 'svm', 'svm_tuned', 'knn', 'forest_tuned'])
stack_clf2.train(titanic_features, titanic_target)
stack_clf2.evaluate(titanic_features, titanic_target)
#Lets predict the values for test set
stack_clf2.predictions(titanic_features_test)
#Lets checkout the predictions
my_predictions = stack_clf2.out_predictions
my_predictions.head(15)
pd.DataFrame({
    'PassengerId': titanic_test.index, 
    'Survived': my_predictions['VotingClassifier_hard']
}).to_csv('/kaggle/working/submission.csv')
#This subimssion scored me highest of all the models I tried out which is around 78% which landed me in Top22% when I submitted
#the predictions