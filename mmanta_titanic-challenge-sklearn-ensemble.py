# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning models

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier



#Pipeline

from sklearn.pipeline import Pipeline, FeatureUnion



#Cross validation

from sklearn.model_selection import cross_val_score
#load data
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
#Minimum exploratory data analysis (mostly is already explored on excellent blog entries about the competition)
print(train_df.columns.values)
# preview the data

train_df.head()
train_df.describe()
corr_matrix = train_df.corr()

corr_matrix["Survived"].sort_values(ascending=False)
%matplotlib inline

import matplotlib.pyplot as plt

train_df.hist(bins=50, figsize=(20,15))

plt.show()
#Feature Torturing

#

#

#
train_df.info()
# fill missing embarkation port

from sklearn.base import TransformerMixin



class FillPort(TransformerMixin):

    def fit(self, X, y=None):

        self.most_common_port=X.Embarked.dropna().mode()[0]

        return self

    def transform(self, X):

        X["Embarked"].fillna(self.most_common_port,inplace=True)

        return X



# fill missing fare - only one in the test set... really not well thought function

from sklearn.base import TransformerMixin



class FillFare(TransformerMixin):

    def fit(self, X, y=None):

        self.fare=11.00205

        return self

    def transform(self, X):

        X["Fare"].fillna(self.fare,inplace=True)

        return X



#print(test_df.loc[test_df["Fare"].isnull()])



#Calculate fare for the missing fare in the test set



#temp1=test_df.loc[test_df["Pclass"]==3]

#temp2=temp1.loc[temp1["Embarked"]=="S"]

#temp3=temp2.loc[temp2["Age"]>40]

#print(temp3.median())



# change Pclass

from sklearn.base import TransformerMixin



class ChangePclass(TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        tag=pd.get_dummies(X["Pclass"],prefix="Pclass")

        X=pd.concat([X,tag], axis=1)

        return X.drop("Pclass", axis=1)

# change embarked

from sklearn.base import TransformerMixin



class ChangeEmbarked(TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        tag=pd.get_dummies(X["Embarked"],prefix="Embarked")

        X=pd.concat([X,tag], axis=1)

        return X.drop("Embarked", axis=1)

        



# change sex from string to numeric

from sklearn.base import TransformerMixin



class ChangeSex(TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X.iloc[:]["Sex"]=X.iloc[:]["Sex"].map({"male":1,"female":0})

        return X
# change name

from sklearn.base import TransformerMixin



class ChangeName(TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X["Title"]=X["Name"].map(lambda name:name.split(',')[1].split('.')[0].strip())

        Title_Dictionary = {

            "Capt":       "Crew",

            "Col":        "Crew",

            "Major":      "Crew",

            "Jonkheer":   "Rare",

            "Don":        "Rare",

            "Sir" :       "Rare",

            "Dr":         "Crew",

            "Rev":        "Crew",

            "the Countess":"Rare",

            "Dona":       "Rare",

            "Mme":        "Mrs",

            "Mlle":       "Miss",

            "Ms":         "Mrs",

            "Mr" :        "Mr",

            "Mrs" :       "Mrs",

            "Miss" :      "Miss",

            "Master" :    "Master",

            "Lady" :      "Rare"

        }

        X["Title"]=X.Title.map(Title_Dictionary)

        tag=pd.get_dummies(X["Title"],prefix="Title")

        X=pd.concat([X,tag], axis=1)

        return X.drop("Name", axis=1).drop("Title", axis=1)

        

    

# change ticket

from sklearn.base import TransformerMixin



class ChangeTicket(TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):



        # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)

        def cleanTicket(ticket):

            ticket = ticket.replace('.','')

            ticket = ticket.replace('/','')

            ticket = ticket.split()

            ticket = list(ticket)

            ticket = map(lambda t : t.strip() , ticket)

            ticket = list(ticket)

            ticket = filter(lambda t : not t.isdigit(), ticket)

            ticket = list(ticket)

            if len(ticket) > 0:

                return ticket[0]

            else: 

                return 'XXX'

        

        X["Ticket"]=X["Ticket"].map(cleanTicket)

        tag=pd.get_dummies(X["Ticket"],prefix="Ticket")

        X=pd.concat([X,tag], axis=1)

        return X.drop("Ticket",axis=1)

        



    
# change cabin

from sklearn.base import TransformerMixin



class ChangeCabin(TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X["Cabin"].fillna("U", inplace=True)

        X["Cabin"]=X["Cabin"].map(lambda c : c[0])

        tag=pd.get_dummies(X["Cabin"], prefix="Cabin")

        X=pd.concat([X,tag], axis=1)

        return X.drop("Cabin", axis=1)

        





# create family parameters

from sklearn.base import TransformerMixin



class CreateFamily(TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        X["FamilySize"]=X["SibSp"]+X["Parch"]+1

        X["Alone"]=X["FamilySize"].map(lambda s : 1 if s == 1 else 0)

        X["NormalFamily"]=X["FamilySize"].map(lambda s : 1 if 2<=s<=4 else 0)

        X["LargeFamily"]=X["FamilySize"].map(lambda s : 1 if 5<=s else 0)

        return X

 
# fill missing Age with Regressor (must be last - regressor needs numeric attribs)

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor



class ChangeAge(BaseEstimator, TransformerMixin):

    def __init__(self):

        #self.model = LinearRegression()

        self.model = RandomForestRegressor() 

        #self.model = KNeighborsRegressor(n_neighbors=3) 

    def fit(self, X, y=None):

        self.X_withage=X.loc[X["Age"].notnull()]

        y=self.X_withage["Age"]

        X_clear=self.X_withage.drop("Age", axis=1)

        if "Survived" in X_clear.columns:

            X_clear=X_clear.drop("Survived", axis=1)

        self.model.fit(X_clear, y)

        #acc=(cross_val_score(self.model, X_clear, y, cv=5, scoring="accuracy").mean()) * 100

        #print(acc)

        return self

    def transform(self, X):

        X_to_replace=X.drop("Age", axis=1)

        X_to_replace_clear=X_to_replace

        if "Survived" in X_to_replace_clear.columns:

            X_to_replace_clear=X_to_replace_clear.drop("Survived", axis=1)

        y_to_replace=pd.DataFrame(self.model.predict(X_to_replace_clear))

        y_to_replace.columns=["Age"]

        X_to_replace["Age"]=X[:]["Age"].fillna(y_to_replace[:]["Age"]) #is there a way to write this clearly?

        return X_to_replace

        
train_df.info()
feature_engineering_pipeline = Pipeline([

    ('fill_port', FillPort()),

    ('fill_fare', FillFare()),

    ('change_embarked', ChangeEmbarked()),

    ('change_pclass', ChangePclass()),

    ('change_sex', ChangeSex()),

    ('change_name', ChangeName()),

    ('change_ticket', ChangeTicket()),

    ('change_cabin', ChangeCabin()),

    ('create_family', CreateFamily()),

    ('change_age', ChangeAge()),

])



train_df_tr=pd.DataFrame(feature_engineering_pipeline.fit_transform(train_df))

test_df_tr=pd.DataFrame(feature_engineering_pipeline.fit_transform(test_df))



test_df_tr.hist(bins=50, figsize=(20,15))

plt.show()

#train_df_tr.describe()

# Feature selection





# Create a combined dataset (train + test) to check features a do age regression

combined=pd.concat([train_df.drop("Survived",axis=1), test_df], axis=0)

combined_tr=pd.DataFrame(feature_engineering_pipeline.fit_transform(combined))

#combined_tr.info()
X_train = combined_tr[0:891]

X_test = combined_tr[891:]

Y_train = train_df_tr["Survived"]

X_train.shape, Y_train.shape, X_test.shape
# Use Extra Trees to find feature importance

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import GridSearchCV



parameter_grid_selection = {

    'max_depth' : [4, 6, 8],

    'n_estimators': [10, 50,100,200],

    'min_samples_split': [2, 3, 10],

    'min_samples_leaf': [1, 3, 10],

    'bootstrap': [True, False],

}



clf = ExtraTreesClassifier(n_estimators=200)

#grid_selection=GridSearchCV(estimator=clf, param_grid=parameter_grid_selection, cv=5, scoring="accuracy")

#%time grid_selection = grid_selection.fit(X_train, Y_train)

# Selected parameters are in the commented line in the next cell

#print('Best score: {}'.format(grid_selection.best_score_))

#print('Best parameters: {}'.format(grid_selection.best_params_))

#clf = ExtraTreesClassifier(bootstrap=False, max_depth=8, min_samples_leaf=3, min_samples_split=2, n_estimators=50)



clf.fit(X_train, Y_train)



features = pd.DataFrame()

features['feature'] = X_train.columns

features['importance'] = clf.feature_importances_

features.sort_values(['importance'],ascending=False)

model = SelectFromModel(clf, prefit=True)

X_train_selected = model.transform(X_train)

X_train_selected.shape
X_test_selected = model.transform(X_test)

X_test_selected.shape
# Fit different models and record accuracy in cross validation


X_train_selected.shape, Y_train.shape, X_test_selected.shape
#feature scaling



from sklearn.preprocessing import StandardScaler, RobustScaler



scaler=StandardScaler()



X_train = scaler.fit_transform(X_train_selected)

X_test = scaler.fit_transform(X_test_selected)

#Models review

# Logistic Regression



parameter_grid = [

  {'C': [0.1, 0.3, 1, 3,10], 'multi_class': ['multinomial'], 'solver': ['lbfgs']},

  {'C': [0.1, 0.3, 1, 3,10], 'multi_class': ['ovr'], 'solver': ['liblinear']},

 ]



logreg = LogisticRegression()

grid=GridSearchCV(estimator=logreg, param_grid=parameter_grid, cv=5, scoring="accuracy")

%time grid.fit(X_train, Y_train)

Y_pred=grid.predict(X_test)

#%time logreg.fit(X_train, Y_train)

#Y_pred = logreg.predict(X_test)

acc_log = (cross_val_score(logreg, X_train, Y_train, cv=5, scoring="accuracy").mean()) * 100



print (acc_log)

print('Best score: {}'.format(grid.best_score_))

print('Best parameters: {}'.format(grid.best_params_))



# Support Vector Machines



parameter_grid = [

  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},

  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},

]





svc = SVC(probability=True, kernel='rbf', C=1000, gamma=0.001)

#grid=GridSearchCV(estimator=svc, param_grid=parameter_grid, cv=5, scoring="accuracy")

#%time grid.fit(X_train, Y_train)

#Y_pred=grid.predict(X_test)

%time svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = (cross_val_score(svc, X_train, Y_train, cv=5, scoring="accuracy").mean()) * 100



print(acc_svc)

print('Best score: {}'.format(grid.best_score_))

print('Best parameters: {}'.format(grid.best_params_))

# K-Nearest Neighbors



parameter_grid = {

    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8],

}

  



knn = KNeighborsClassifier(n_neighbors = 3)

grid=GridSearchCV(estimator=knn, param_grid=parameter_grid, cv=5, scoring="accuracy")

%time grid.fit(X_train, Y_train)

Y_pred=grid.predict(X_test)

#%time knn.fit(X_train, Y_train)

#Y_pred = knn.predict(X_test)

acc_knn = (cross_val_score(knn, X_train, Y_train, cv=5, scoring="accuracy").mean()) * 100



print(acc_knn)

print('Best score: {}'.format(grid.best_score_))

print('Best parameters: {}'.format(grid.best_params_))

# Gaussian Naive Bayes



gaussian = GaussianNB()

%time gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = (cross_val_score(gaussian, X_train, Y_train, cv=5, scoring="accuracy").mean()) * 100

acc_gaussian
# Linear SVC



linear_svc = LinearSVC()

%time linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = (cross_val_score(linear_svc, X_train, Y_train, cv=5, scoring="accuracy").mean()) * 100

acc_linear_svc
# Stochastic Gradient Descent



parameter_grid={

'loss': ["hinge", "log", "modified_huber","perceptron","squared_hinge"],

'learning_rate': ["constant","optimal","invscaling"],

'alpha': [0.0001, 0.0003, 0.001, 0.003, 0.01,0.03,0.1,0.3,1 ],

'penalty': ["None", "l2", "l1","elasticnet"]

}





sgd = SGDClassifier(eta0=0.001)

grid=GridSearchCV(estimator=sgd, param_grid=parameter_grid, cv=5, scoring="accuracy")

%time grid.fit(X_train, Y_train)

Y_pred=grid.predict(X_test)

#%time sgd.fit(X_train, Y_train)

#Y_pred = sgd.predict(X_test)

acc_sgd = (cross_val_score(sgd, X_train, Y_train, cv=5, scoring="accuracy").mean()) * 100



print(acc_sgd)

print('Best score: {}'.format(grid.best_score_))

print('Best parameters: {}'.format(grid.best_params_))
# Decision Tree



decision_tree = DecisionTreeClassifier()

%time decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = (cross_val_score(decision_tree, X_train, Y_train, cv=5, scoring="accuracy").mean()) * 100

acc_decision_tree
#MLPClassifier



parameter_grid={

'learning_rate': ["constant", "invscaling", "adaptive"],

'hidden_layer_sizes': [(14,4,1), (14,14,1)],

'alpha': [0.001, 0.03, 0.01 ],

'solver' : ["adam", "sgd", "lbfgs"],

'activation': ["logistic", "relu", "tanh"]

}





MLP = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(14,4,1), activation='tanh',alpha=0.03,learning_rate='invscaling',batch_size=1400, max_iter=4000, early_stopping=True)

#grid=GridSearchCV(estimator=MLP, param_grid=parameter_grid, cv=5, scoring="accuracy")

#%time grid.fit(X_train, Y_train)

#Y_pred=grid.predict(X_test)

%time MLP.fit(X_train, Y_train)

Y_pred = MLP.predict(X_test)

acc_mlp = (cross_val_score(MLP, X_train, Y_train, cv=5, scoring="accuracy").mean()) * 100



print(acc_mlp)

print('Best score: {}'.format(grid.best_score_))

print('Best parameters: {}'.format(grid.best_params_))
# Random Forest



parameter_grid = {

    'max_depth': [4, 5, 6, 7, 8],

    'n_estimators': [200,210,240,250],

    'criterion': ['gini', 'entropy'],

}



random_forest = RandomForestClassifier(n_estimators=100)

grid=GridSearchCV(estimator=random_forest, param_grid=parameter_grid, cv=5, scoring="accuracy")

%time grid.fit(X_train, Y_train)

Y_pred=grid.predict(X_test)

#%time random_forest.fit(X_train, Y_train)

#Y_pred = random_forest.predict(X_test)

acc_random_forest = (cross_val_score(random_forest, X_train, Y_train, cv=5, scoring="accuracy").mean()) * 100



print(acc_random_forest)

print('Best score: {}'.format(grid.best_score_))

print('Best parameters: {}'.format(grid.best_params_))

# Extra Trees



parameter_grid = {

    'max_depth': [4, 5, 6, 7, 8],

    'n_estimators': [200,210,240,250],

    'criterion': ['gini', 'entropy'],

}



extra_trees = ExtraTreesClassifier(n_estimators=100)

grid=GridSearchCV(estimator=extra_trees, param_grid=parameter_grid, cv=5, scoring="accuracy")

%time grid.fit(X_train, Y_train)

Y_pred=grid.predict(X_test)

#%time extra_trees.fit(X_train, Y_train)

#Y_pred = extra_trees.predict(X_test)

acc_extra_trees = (cross_val_score(extra_trees, X_train, Y_train, cv=5, scoring="accuracy").mean()) * 100



print(acc_extra_trees)

print('Best score: {}'.format(grid.best_score_))

print('Best parameters: {}'.format(grid.best_params_))

# Gradient Boosting



parameter_grid = {

    'max_depth': [4, 5, 6, 7, 8],

    'n_estimators': [200,210,240,250],

    'criterion': ['friedman_mse', 'mse','mae'],

}



gradient_boosting = GradientBoostingClassifier(n_estimators=100)

grid=GridSearchCV(estimator=gradient_boosting, param_grid=parameter_grid, cv=5, scoring="accuracy")

%time grid.fit(X_train, Y_train)

Y_pred=grid.predict(X_test)

#%time gradient_boosting.fit(X_train, Y_train)

#Y_pred = gradient_boosting.predict(X_test)

acc_gradient_boosting = (cross_val_score(gradient_boosting, X_train, Y_train, cv=5, scoring="accuracy").mean()) * 100



print(acc_gradient_boosting)

print('Best score: {}'.format(grid.best_score_))

print('Best parameters: {}'.format(grid.best_params_))
# Due to random initializations, your scores may vary

models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Gaussian Naive Bayes', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree', 'MLP', 'Gradient Boosting','Extra Trees'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, 

              acc_sgd, acc_linear_svc, acc_decision_tree, acc_mlp, acc_gradient_boosting,acc_extra_trees]})

models.sort_values(by='Score', ascending=False)
# Ensemble of the "tuned" top models

# I include only one out of gb, rf and ef since all of them are "optimizations" on top of Decision Trees



classifiers=[

    ('svc', SVC(C=1000, gamma=0.001, probability=True, kernel='rbf')),

    ('linear_svc', SVC(kernel='linear', probability=True)),

    #('sgd', SGDClassifier(alpha=0.1,learning_rate='optimal',loss='modified_huber', penalty='elasticnet')),

    #('gb', GradientBoostingClassifier(criterion='mae',max_depth=8,n_estimators=200)),

    ('lr', LogisticRegression(C=0.3,multi_class='ovr', solver='liblinear')),

    ('knn', KNeighborsClassifier(n_neighbors = 6)),

    ('mlp', MLPClassifier(solver='lbfgs', hidden_layer_sizes=(14,4,1), alpha=0.03, activation='tanh',learning_rate='constant')),

    #('ef', ExtraTreesClassifier(criterion='gini',max_depth=6,n_estimators=240)),

    ('rf', RandomForestClassifier(criterion='gini',max_depth=4,n_estimators=200)),

]



    

voting=VotingClassifier(classifiers,voting="hard")

%time voting.fit(X_train, Y_train)

Y_pred=voting.predict(X_test)

%time acc_grid_search = (cross_val_score(voting, X_train, Y_train, cv=5, scoring="accuracy").mean()) * 100



print(acc_grid_search)

# Submission



# He didn't confess yet, but he will...



StackingSubmission = pd.DataFrame({ 'PassengerId': test_df_tr["PassengerId"],

                            'Survived': Y_pred })

StackingSubmission.to_csv("StackingSubmission.csv", index=False)