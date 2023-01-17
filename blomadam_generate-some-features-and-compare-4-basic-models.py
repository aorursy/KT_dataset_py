import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/train.csv")
df.head()
df.info()
df[df.Age.isnull()].head()
for i in df.columns:

    print (df[i].value_counts().head(10))
df[df["Fare"] < 0.001].sort_values("Ticket")

# reasearching several of these individuals shows they were employees

# and much more likely to have died

# add a feature that is 1 for free_fare
print( df[df["Embarked"].isnull()])

# https://www.encyclopedia-titanica.org/titanic-survivor/amelia-icard.html

# research shows these should both be set to S for Southampton
master = df[["Master." in x for x in df["Name"]]]["Age"].dropna()

rev = df[["Rev." in x for x in df["Name"]]]["Age"].dropna()

mr = df[["Mr." in x for x in df["Name"]]]["Age"].dropna()

miss = df[["Miss." in x for x in df["Name"]]]["Age"].dropna()

mrs = df[["Mrs." in x for x in df["Name"]]]["Age"].dropna()

quot = df[['"' in x for x in df["Name"]]]["Age"].dropna()

paren = df[["(" in x for x in df["Name"]]]["Age"].dropna()

both = df[["(" in x and '"' in x for x in df["Name"]]]["Age"].dropna()

plt.boxplot([master.values,mr.values,miss.values, \

             mrs.values,quot.values,paren.values, \

             both.values])

plt.title("Ages vs name keywords")

plt.show()
master = df[["Master." in x for x in df["Name"]]]["Fare"].dropna()

rev = df[["Rev." in x for x in df["Name"]]]["Fare"].dropna()

mr = df[["Mr." in x for x in df["Name"]]]["Fare"].dropna()

miss = df[["Miss." in x for x in df["Name"]]]["Fare"].dropna()

mrs = df[["Mrs." in x for x in df["Name"]]]["Fare"].dropna()

quot = df[['"' in x for x in df["Name"]]]["Fare"].dropna()

paren = df[["(" in x for x in df["Name"]]]["Fare"].dropna()

both = df[["(" in x and '"' in x for x in df["Name"]]]["Fare"].dropna()

plt.boxplot([master.values,mr.values,miss.values, \

             mrs.values,quot.values,paren.values,both.values])

plt.title("Fares vs name keywords")

plt.ylim(0,150)

plt.show()

# master
X = df.iloc[:,2:]

y = df.iloc[:,0]
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, Imputer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer

from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion

from sklearn.linear_model import LogisticRegressionCV, ElasticNetCV

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import tree
X.head()
cols = [x for x in X.columns if x !='last_name' and x != 'Age']
# u = StandardScaler()

# v = ElasticNetCV()

# g = GridSearchCV(Pipeline([('scale',u),('fit',v)]),{'fit__l1_ratio':(0.00001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)}, n_jobs=-1)

# g.fit(X[X.Age.notnull()].ix[:,cols],X.Age[X.Age.notnull()])
# g.best_estimator_
# pd.DataFrame(g.cv_results_).sort_values("rank_test_score")
# g.predict(X[X.Age.notnull()].ix[:,cols]).shape
# X.Age[X.Age.notnull()].shape
# g.score(X[X.Age.notnull()].ix[:,cols],X.Age[X.Age.notnull()])
# plt.scatter(X.Age[X.Age.notnull()],g.predict(X[X.Age.notnull()].ix[:,cols]))

# plt.plot(X.Age[X.Age.notnull()],X.Age[X.Age.notnull()])
# g.predict(X[X.Age.isnull()].ix[:,cols])
median_fare = df.Fare.median()
# generate some columns

def feature_cleaning(df:pd.DataFrame)->pd.DataFrame:

    df.drop(["Cabin", "Ticket", "PassengerId"], axis=1, inplace=True)   # I will ignore these columns

    df.Embarked.fillna("S",inplace=True)  # fill using the info above

    df = pd.get_dummies(df,columns=["Pclass","Sex","Embarked"],drop_first=True)  # tokenize these columns



    # feature engineering

    

    df['age_estimated'] = df.Age.map(lambda x: 1 if x%1 > 0.2 else 0)

    df['chaperoned_child'] = ((df.Age < 18) & (df.Parch == 0)).astype(int)

    df['family_size'] = df.SibSp + df.Parch

    

    df['free_fare'] = df.Fare.map(lambda x: 1 if x < 0.001 else 0)

    df["has_master"] = df.Name.map(lambda x: 1 if 'Master.' in x else 0)

    df["has_rev"] = df.Name.map(lambda x: 1 if 'Rev.' in x else 0)

    df["has_mr"] = df.Name.map(lambda x: 1 if 'Mr.' in x else 0)

    df["has_miss"] = df.Name.map(lambda x: 1 if 'Miss.' in x else 0)

    df["has_mrs"] = df.Name.map(lambda x: 1 if 'Mrs.' in x else 0)

    df["has_quote"] = df.Name.map(lambda x: 1 if '"' in x else 0)

    df["has_parens"] = df.Name.map(lambda x: 1 if '(' in x else 0)

    df["last_name"] = df.Name.map(lambda x: x.replace(",","").split()[0] )



    df.drop(["Age","Name"], axis=1, inplace=True)

    df.Fare.fillna(median_fare, inplace=True)

    return df



df = feature_cleaning(df)
df.head()
X = df.iloc[:,1:]

y = df.iloc[:,0]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, random_state=42)
class ModelTransformer(BaseEstimator,TransformerMixin):



    def __init__(self, model=None):

        self.model = model



    def fit(self, *args, **kwargs):

        self.model.fit(*args, **kwargs)

        return self



    def transform(self, X, **transform_params):

        return self.model.transform(X)

    

class SampleExtractor(BaseEstimator, TransformerMixin):

    """Takes in varaible names as a **list**"""



    def __init__(self, vars):

        self.vars = vars  # e.g. pass in a column names to extract



    def transform(self, X, y=None):

        if len(self.vars) > 1:

            return pd.DataFrame(X[self.vars]) # where the actual feature extraction happens

        else:

            return pd.Series(X[self.vars[0]])



    def fit(self, X, y=None):

        return self  # generally does nothing

    

    

class DenseTransformer(BaseEstimator,TransformerMixin):



    def transform(self, X, y=None, **fit_params):

#         print (X.todense())

        return X.todense()



    def fit_transform(self, X, y=None, **fit_params):

        self.fit(X, y, **fit_params)

        return self.transform(X)



    def fit(self, X, y=None, **fit_params):

        return self
kf_shuffle = StratifiedKFold(n_splits=5,shuffle=True,random_state=777)



cols = [x for x in X.columns if x !='last_name']



pipeline = Pipeline([

    ('features', FeatureUnion([

        ('names', Pipeline([

                      ('text',SampleExtractor(['last_name'])),

                      ('dummify', CountVectorizer(binary=True)),

                      ('densify', DenseTransformer()),

                     ])),

        ('cont_features', Pipeline([

                      ('continuous', SampleExtractor(cols)),

                      ])),

        ])),

        ('scale', ModelTransformer()),

        ('fit', LogisticRegressionCV(solver='liblinear')),

])





parameters = {

    'scale__model': (StandardScaler(),MinMaxScaler()),

    'fit__penalty': ('l1','l2'),

    'fit__class_weight':('balanced',None),

    'fit__Cs': (20,),

}



logreg_gs = GridSearchCV(pipeline, parameters, verbose=False, cv=kf_shuffle, n_jobs=-1)

%%time

print("Performing grid search...")

print("pipeline:", [name for name, _ in pipeline.steps])

print("parameters:")

print(parameters)





logreg_gs.fit(X_train, y_train)



print("Best score: %0.3f" % logreg_gs.best_score_)

print("Best parameters set:")

best_parameters = logreg_gs.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))
cv_pred = pd.Series(logreg_gs.predict(X_test))
pd.DataFrame(list(zip(logreg_gs.cv_results_['mean_test_score'],\

                 logreg_gs.cv_results_['std_test_score'])\

                )).sort_values(0,ascending=False).head(10)

# logreg_gs.best_estimator_
confusion_matrix(y_test,cv_pred)
print (classification_report(y_test,cv_pred))
from sklearn.metrics import roc_curve, auc, precision_recall_curve

plt.style.use('seaborn-white')



# Y_score = logreg_gs.best_estimator_.decision_function(X_test)

Y_score = logreg_gs.best_estimator_.predict_proba(X_test)[:,1]



# For class 1, find the area under the curve

FPR, TPR, _ = roc_curve(y_test, Y_score)

ROC_AUC = auc(FPR, TPR)

PREC, REC, _ = precision_recall_curve(y_test, Y_score)

PR_AUC = auc(REC, PREC)



# Plot of a ROC curve for class 1 (has_cancer)

plt.figure(figsize=[11,9])

plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % ROC_AUC, linewidth=4)

plt.plot(REC, PREC, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', linewidth=4)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate', fontsize=18)

plt.ylabel('True Positive Rate', fontsize=18)

plt.title('Logistic Regression for Titanic Survivors', fontsize=18)

plt.legend(loc="lower right")

plt.show()
plt.scatter(y_test,cv_pred,color='r')

plt.plot(y_test,y_test,color='k')

plt.xlabel("True value")

plt.ylabel("Predicted Value")

plt.show()
kf_shuffle = StratifiedKFold(n_splits=5,shuffle=True,random_state=777)



cols = [x for x in X.columns if x !='last_name']



pipeline = Pipeline([

    ('features', FeatureUnion([

        ('names', Pipeline([

                      ('text',SampleExtractor(['last_name'])),

                      ('dummify', CountVectorizer(binary=True)),

                      ('densify', DenseTransformer()),

                     ])),

        ('cont_features', Pipeline([

                      ('continuous', SampleExtractor(cols)),

                      ])),

        ])),

        ('scale', ModelTransformer()),

        ('fit', KNeighborsClassifier()),

])





parameters = {

    'scale__model': (StandardScaler(),MinMaxScaler()),

    'fit__n_neighbors': (2,3,5,7,9,11,16,20),

    'fit__weights': ('uniform','distance'),

}



knn_gs = GridSearchCV(pipeline, parameters, verbose=False, cv=kf_shuffle, n_jobs=-1)

%%time

print("Performing grid search...")

print("pipeline:", [name for name, _ in pipeline.steps])

print("parameters:")

print(parameters)





knn_gs.fit(X_train, y_train)



print("Best score: %0.3f" % knn_gs.best_score_)

print("Best parameters set:")

best_parameters = knn_gs.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))
cv_pred = pd.Series(knn_gs.predict(X_test))
pd.DataFrame(list(zip(knn_gs.cv_results_['mean_test_score'],\

                 knn_gs.cv_results_['std_test_score'])\

                )).sort_values(0,ascending=False).head(10)

# knn_gs.best_estimator_
confusion_matrix(y_test,cv_pred)
print( classification_report(y_test,cv_pred))
from sklearn.metrics import roc_curve, auc

plt.style.use('seaborn-white')



# Y_score = knn_gs.best_estimator_.decision_function(X_test)

Y_score = knn_gs.best_estimator_.predict_proba(X_test)[:,1]





# For class 1, find the area under the curve

FPR, TPR, _ = roc_curve(y_test, Y_score)

ROC_AUC = auc(FPR, TPR)



PREC, REC, _ = precision_recall_curve(y_test, Y_score)

PR_AUC = auc(REC, PREC)



# Plot of a ROC curve for class 1 (has_cancer)

plt.figure(figsize=[11,9])

plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % ROC_AUC, linewidth=4)

plt.plot(REC, PREC, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', linewidth=4)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate', fontsize=18)

plt.ylabel('True Positive Rate', fontsize=18)

plt.title('kNN for Titanic Survivors', fontsize=18)

plt.legend(loc="lower right")

plt.show()
kf_shuffle = StratifiedKFold(n_splits=5,shuffle=True,random_state=777)



cols = [x for x in X.columns if x !='last_name']



pipeline = Pipeline([

    ('features', FeatureUnion([

        ('names', Pipeline([

                      ('text',SampleExtractor(['last_name'])),

                      ('dummify', CountVectorizer(binary=True)),

                      ('densify', DenseTransformer()),

                     ])),

        ('cont_features', Pipeline([

                      ('continuous', SampleExtractor(cols)),

                      ])),

        ])),

#         ('scale', ModelTransformer()),

        ('fit', tree.DecisionTreeClassifier()),

])





parameters = {

#     'scale__model': (StandardScaler(),MinMaxScaler()),

    'fit__max_depth': (2,3,4,None),

    'fit__min_samples_split': (2,3,4,5),

    'fit__class_weight':('balanced',None),

}



dt_gs = GridSearchCV(pipeline, parameters, verbose=False, cv=kf_shuffle, n_jobs=-1)

%%time

print("Performing grid search...")

print("pipeline:", [name for name, _ in pipeline.steps])

print("parameters:")

print(parameters)





dt_gs.fit(X_train, y_train)



print("Best score: %0.3f" % dt_gs.best_score_)

print("Best parameters set:")

best_parameters = dt_gs.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))
cv_pred = pd.Series(dt_gs.predict(X_test))
pd.DataFrame(list(zip(dt_gs.cv_results_['mean_test_score'],\

                 dt_gs.cv_results_['std_test_score'])\

                )).sort_values(0,ascending=False).head(10)

# dt_gs.best_estimator_
confusion_matrix(y_test,cv_pred)
print (classification_report(y_test,cv_pred))
from sklearn.metrics import roc_curve, auc

plt.style.use('seaborn-white')



# Y_score = dt_gs.best_estimator_.decision_function(X_test)

Y_score = dt_gs.best_estimator_.predict_proba(X_test)[:,1]





# For class 1, find the area under the curve

FPR, TPR, _ = roc_curve(y_test, Y_score)

ROC_AUC = auc(FPR, TPR)



PREC, REC, _ = precision_recall_curve(y_test, Y_score)

PR_AUC = auc(REC, PREC)



# Plot of a ROC curve for class 1 (has_cancer)

plt.figure(figsize=[11,9])

plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % ROC_AUC, linewidth=4)

plt.plot(REC, PREC, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', linewidth=4)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate', fontsize=18)

plt.ylabel('True Positive Rate', fontsize=18)

plt.title('Decision Tree for Titanic Survivors', fontsize=18)

plt.legend(loc="lower right")

plt.show()
from sklearn.ensemble import RandomForestClassifier
kf_shuffle = StratifiedKFold(n_splits=5,shuffle=True,random_state=777)



cols = [x for x in X.columns if x !='last_name']



pipeline = Pipeline([

    ('features', FeatureUnion([

        ('names', Pipeline([

                      ('text',SampleExtractor(['last_name'])),

                      ('dummify', CountVectorizer(binary=True)),

                      ('densify', DenseTransformer()),

                     ])),

        ('cont_features', Pipeline([

                      ('continuous', SampleExtractor(cols)),

                      ])),

        ])),

#         ('scale', ModelTransformer()),

        ('fit', RandomForestClassifier()),

])





parameters = {

#     'scale__model': (StandardScaler(),MinMaxScaler()),

    'fit__max_depth': (4,7,10),

    'fit__n_estimators': (25,100,200,300),

    'fit__class_weight':('balanced',None),

    'fit__max_features': ('auto',0.3,0.5),

}



rf_gs = GridSearchCV(pipeline, parameters, verbose=False, cv=kf_shuffle, n_jobs=-1)

%%time

print("Performing grid search...")

print("pipeline:", [name for name, _ in pipeline.steps])

print("parameters:")

print(parameters)





rf_gs.fit(X_train, y_train)



print("Best score: %0.3f" % rf_gs.best_score_)

print("Best parameters set:")

best_parameters = rf_gs.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))
cv_pred = pd.Series(rf_gs.predict(X_test))
pd.DataFrame(list(zip(rf_gs.cv_results_['mean_test_score'],\

                 rf_gs.cv_results_['std_test_score'])\

                )).sort_values(0,ascending=False).head(10)

# rf_gs.best_estimator_
confusion_matrix(y_test,cv_pred)
print (classification_report(y_test,cv_pred))
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

plt.style.use('seaborn-white')



Y_score = rf_gs.best_estimator_.predict_proba(X_test)[:,1]





# For class 1, find the area under the curve

FPR, TPR, _ = roc_curve(y_test, Y_score)

ROC_AUC = auc(FPR, TPR)



PREC, REC, _ = precision_recall_curve(y_test, Y_score)

PR_AUC = auc(REC, PREC)



# Plot of a ROC curve for class 1 (has_cancer)

plt.figure(figsize=[11,9])

plt.plot(FPR, TPR, label='ROC curve (area = %0.2f)' % ROC_AUC, linewidth=4)

plt.plot(REC, PREC, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)

plt.plot([0, 1], [0, 1], 'k--', linewidth=4)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate or Recall', fontsize=18)

plt.ylabel('True Positive Rate or Precision', fontsize=18)

plt.title('Random Forest for Titanic Survivors', fontsize=18)

plt.legend(loc="lower right")

plt.show()
rf_gs.best_estimator_.steps[1][1].feature_importances_[:15]
X_pred = pd.read_csv("../input/test.csv")

pred_ids = X_pred.PassengerId

X_pred = feature_cleaning(X_pred)
X_pred.head()
X_pred.shape
X_test.shape
X_pred.isnull().sum().sum()  # check there are no nulls
predictions = rf_gs.predict(X_pred)
predictions[:5]
pd.DataFrame(list(zip(pred_ids, predictions)), columns=["PassengerId","Survived"]).to_csv("RF_pred.csv", index=None)