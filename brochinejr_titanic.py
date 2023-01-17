#initial set up
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
import time
print(os.listdir("../input"))
import sklearn
from sklearn import metrics
print('The scikit-learn version is {}.'.format(sklearn.__version__))
#Read Train and Test Sets
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#X=train.drop(['PassengerId', 'Survived'], axis=1)
X=train.drop(['Survived'], axis=1)
y=train['Survived']
# SEX ANALISYS
print(X.columns)
# configure graph
fig = plt.figure(figsize=(35,15))
alpha = alpha_scatterplot = 0.2
alpha_bar_chart = 0.55
# survived vs deceased
fig.add_subplot(3,4,1)
train.Survived.value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Survival")

# male survived vs male deceased
fig.add_subplot(3,4,2)
train.Survived[train.Sex == "male"].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Male Survival")

# female survived vs female deceased
fig.add_subplot(3,4,3)
train.Survived[train.Sex == "female"].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Female Survival")

plt.show()
# AGE ANALISYS
train["Age"] = train["Age"].fillna(train["Age"].dropna().median())
train['Age'].unique()
fig = plt.figure(figsize=(35,15))
fig.add_subplot(3,4,1)
plt.hist(train['Age'], color = 'blue', edgecolor = 'black')
# child survived vs child deceased
fig.add_subplot(3,4,2)
train.Survived[train.Age < 18].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("child Survival")

# adult survived vs adult deceased
fig.add_subplot(3,4,3)
train.Survived[train.Age > 18].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Adult Survival")

plt.show()
# CLASS ANALYSIS
train['Pclass'].unique()
fig = plt.figure(figsize=(35,15))
fig.add_subplot(3,4,1)
plt.hist(train['Pclass'], color = 'blue', edgecolor = 'black')
# First Class
fig.add_subplot(3,4,2)
train.Survived[train.Pclass ==1].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("1st Class Survival")

# Second Class
fig.add_subplot(3,4,3)
train.Survived[train.Pclass ==2].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("2nd Class Survival")

# Third Class
fig.add_subplot(3,4,4)
train.Survived[train.Pclass ==3].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("3rd Class Survival")
plt.show()

#Combine Class, Age and Sex
fig = plt.figure(figsize=(35,15))
fig.add_subplot(4,3,1)
#Men Child
train["Survived"][train["Sex"]=='male'][train["Age"]<18][train["Pclass"]==1].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Men-Child-1st")
fig.add_subplot(4,3,2)
train["Survived"][train["Sex"]=='male'][train["Age"]<18][train["Pclass"]==2].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Men-Child-2nd")
fig.add_subplot(4,3,3)
train["Survived"][train["Sex"]=='male'][train["Age"]<18][train["Pclass"]==3].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Men-Child-3rd")
#Men Adult
fig.add_subplot(4,3,4)
train["Survived"][train["Sex"]=='male'][train["Age"]>=18][train["Pclass"]==1].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Men-Adult-1st")
fig.add_subplot(4,3,5)
train["Survived"][train["Sex"]=='male'][train["Age"]>=18][train["Pclass"]==2].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Men-Adult-2nd")
fig.add_subplot(4,3,6)
train["Survived"][train["Sex"]=='male'][train["Age"]>=18][train["Pclass"]==3].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Men-Adult-3rd")

#Women Child
fig.add_subplot(4,3,7)
train["Survived"][train["Sex"]=='female'][train["Age"]<18][train["Pclass"]==1].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Women-Child-1st")
fig.add_subplot(4,3,8)
train["Survived"][train["Sex"]=='female'][train["Age"]<18][train["Pclass"]==2].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Women-Child-2nd")
fig.add_subplot(4,3,9)
train["Survived"][train["Sex"]=='female'][train["Age"]<18][train["Pclass"]==3].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Women-Child-3rd")
#Men Adult
fig.add_subplot(4,3,10)
train["Survived"][train["Sex"]=='female'][train["Age"]>=18][train["Pclass"]==1].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Women-Adult-1st")
fig.add_subplot(4,3,11)
train["Survived"][train["Sex"]=='female'][train["Age"]>=18][train["Pclass"]==2].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Women-Adult-2nd")
fig.add_subplot(4,3,12)
train["Survived"][train["Sex"]=='female'][train["Age"]>=18][train["Pclass"]==3].value_counts(normalize = True).plot(kind='bar', alpha=alpha_bar_chart)
plt.title("Women-Adult-3rd")

plt.show()
#Data Treatment
def data_treat(data):
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1
    data["Embarked"] = data["Embarked"].fillna('S')
    data.loc[data["Embarked"] == "S", "Embarked"] = 1
    data.loc[data["Embarked"] == "C", "Embarked"] = 2
    data.loc[data["Embarked"] == "Q", "Embarked"] = 3
    print(' ============= data is treated =============')

# convert dataset to Train and Validation Sets
from sklearn.model_selection import train_test_split
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2)
data_treat(X_train)
data_treat(X_validation)
features_list=["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]
features_train = X_train[features_list].values
features_validation=X_validation[features_list].values
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

models=[];accs=[];times=[]
def running(mod):
    mod_name=mod
    print(mod_name)
    if mod_name not in models:
        models.append(mod_name)
        start=time.time()
        #print ('================ Creating Models ================')
        model=eval(mod)
        #print('================ Fitting Models ================')
        model.fit(features_train,y_train)
        #print('================ Predictions ================')
        y_model=model.predict(features_validation)
        end=time.time()
        #print("The Model achieve Accuracy:",metrics.accuracy_score(y_test, y_model), 'in', end-start,'sec')
        accs.append(metrics.accuracy_score(y_validation, y_model))
        times.append(end-start)
lista=['BernoulliNB()',
       'DecisionTreeClassifier()',
       'DecisionTreeClassifier(max_depth = 7,min_samples_split = 2, random_state = 1)',
       'ExtraTreeClassifier()',
       'ExtraTreesClassifier()',
       'LogisticRegression(C=10)',
       'LinearSVC()',
       'RidgeClassifier()',
       'RandomForestClassifier()',
       'RandomForestClassifier(n_estimators=50)',
       'RandomForestClassifier(n_estimators=100)',
       'RandomForestClassifier(n_estimators=200)',
       'RandomForestClassifier(max_depth = 7, min_samples_split = 4,n_estimators = 1000,random_state = 1, n_jobs = -1)',
       'NearestCentroid()',
       'MLPClassifier(max_iter=1000)',
       'XGBClassifier()']
rrun=True
if rrun==True:
    for el in lista:
        running(el)
    for_df = [('Model', models),
             ('Accuracy', accs),
             ('Time', times)]
    df_models = pd.DataFrame.from_items(for_df)
    print (df_models)
best_model=df_models[df_models['Accuracy']==df_models['Accuracy'].max()]
print(best_model)
print(str(best_model))
final_model_name=best_model['Model'].values[0]
final_model=eval(final_model_name)
import eli5
from eli5.sklearn import PermutationImportance
final_model.fit(features_train,y_train)
perm = PermutationImportance(final_model, random_state=1).fit(features_validation, y_validation)

final_val=final_model.predict(features_validation)
print("The Model achieve Accuracy:",metrics.accuracy_score(y_validation, final_val))

eli5.show_weights(perm, feature_names = features_list)

from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(final_model, features_train, y_train, cv=5)
print(metrics.accuracy_score(y_train, predicted) )
data_treat(test)
features_test = test[features_list].values

data_treat(X)
features_X = X[features_list].values
to_sub=eval(final_model_name)
to_sub.fit(features_X,y)
final_sub= to_sub.predict(features_test)
ids=test['PassengerId'] 
my_submission = pd.DataFrame({'PassengerId': ids, 'Survived': final_sub})
my_submission.to_csv('submission.csv', index=False)
print (" ======================= End ======================= ")