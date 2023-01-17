%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

from sklearn import tree

from sklearn.model_selection import cross_validate

from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/titanic/train.csv')

df = df.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis=1).copy()

#I dropped these because of NaN's , aren't decision trees supposed to handle those?

df.dropna(axis=0, how='any', inplace = True)

df.reset_index(drop=True, inplace = True)

df = pd.get_dummies(df, columns=['Sex', 'Embarked']).copy()

x = df.iloc[:, df.columns != 'Survived']

y = df['Survived']
print (df.info())
trees = []

for i in range(20):

    out = cross_validate(tree.DecisionTreeClassifier(max_depth = i+1), x, y, return_train_score = True, cv=10)

    trees.append((i+1, np.mean(out['train_score']), np.mean(out['test_score'])))
trees = pd.DataFrame(trees)

trees.columns = ['Depth', 'Train', 'Test']

trees.set_index('Depth')

plt.plot(trees['Depth'], trees['Train'], label = "Train")

plt.plot(trees['Depth'], trees['Test'], label = "Test")

plt.title('train/test as function of tree depth')

plt.xlabel('Tree Depth')

plt.ylabel('Accuracy')

plt.legend()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

forest =RandomForestClassifier(n_estimators=100) #default changed from 10 to 100 in version 0.22

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0) #train/test split

forest.fit(x_train, y_train)

y_pred=forest.predict(x_test)



#Features used at the top of the tree contribute to the final prediction decision of a larger fraction of the input samples. 

#The expected fraction of the samples they contribute to can thus be used as an estimate of the relative importance of the features. 

#In scikit-learn, the fraction of samples a feature contributes to is combined with the decrease in impurity from splitting them to create a normalized estimate of the predictive power of that feature. 

#By averaging the estimates of predictive ability over several randomized trees one can reduce the variance of such an estimate and use it for feature selection. 

#This is known as the mean decrease in impurity, or MDI.

#In practice those estimates are stored as an attribute named feature_importances_ on the fitted model.

#1.11. Ensemble methods Documentation
#This was surprising because I expected Sex to be the biggest indicator of survival



from sklearn import metrics

print ('Training Accuracy:', forest.score(x_train, y_train))

print("Test Accuracy:",metrics.accuracy_score(y_test, y_pred))

feature_imp = pd.Series(forest.feature_importances_,index=list(x_train.columns)).sort_values(ascending=False)

feature_imp 
#I am currently over fitting

#we're still overfitting

#google how to regularize random forest classifier



out = cross_validate(forest, x, y, return_train_score = True, cv=10)

print('Train Score:', np.mean(out['train_score']), 'Test Score:', np.mean(out['test_score']))
forest.fit(x_train, y_train)

feature_imp = pd.Series(forest.feature_importances_,index=list(x_train.columns)).sort_values(ascending=False)

feature_imp 

sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Feature Improtance")

plt.legend()

plt.show()
# Gets score up to 81%

from xgboost import XGBClassifier

boost = XGBClassifier()

out = cross_validate(boost, x, y, return_train_score = True, cv=10)

print('Train Score:', np.mean(out['train_score']), 'Test Score:', np.mean(out['test_score']))
forests = []

for i in range(10): #We only have 10 features len(list(x))

    out = cross_validate(tree.DecisionTreeClassifier(max_features = i+1), x, y, return_train_score = True, cv=10)

    forests.append((i+1, np.mean(out['train_score']), np.mean(out['test_score'])))
forests = pd.DataFrame(forests)

forests.columns = ['# of Features', 'Train', 'Test']

forests.set_index('# of Features')

plt.plot(forests['# of Features'], forests['Train'], label = "Train")

plt.plot(forests['# of Features'], forests['Test'], label = "Test")

plt.title('train/test as function of max_features')

plt.xlabel('Max Features')

plt.ylabel('Accuracy')

plt.legend()
plt.plot(forests['# of Features'], forests['Test'], label = "Test")

plt.xlabel('Max Features')

plt.ylabel('Accuracy')

plt.legend()
#repeated random cross validation

e = []

for i in range(20): 

    forest = RandomForestClassifier(n_estimators= i+1)

    out = cross_validate(forest, x, y, return_train_score = True, cv=100)

    e.append((i+1, np.mean(out['train_score']), np.mean(out['test_score'])))
e = pd.DataFrame(e)

e.columns = ['# of Estimators', 'Train', 'Test']

e.set_index('# of Estimators')

plt.plot(e['# of Estimators'], e['Train'], label = "Train")

plt.plot(e['# of Estimators'], e['Test'], label = "Test")

plt.title('train/test as function of n_estimators')

plt.xlabel('Number of Estimators')

plt.ylabel('Accuracy')

plt.legend()
plt.plot(e['# of Estimators'], e['Test'], label = "Test")

f = []

for i in range(100): 

    boost = XGBClassifier(max_features = i+1)

    out = cross_validate(boost, x, y, return_train_score = True, cv=10)

    f.append((i+1, np.mean(out['train_score']), np.mean(out['test_score'])))
l = []

for i in range(100): 

    forest = RandomForestClassifier(n_estimators = 100, min_samples_leaf = i+1)

    out = cross_validate(forest, x, y, return_train_score = True, cv=10)

    l.append((i+1, np.mean(out['train_score']), np.mean(out['test_score'])))
l = pd.DataFrame(l)

l.columns = ['min_sample_leaf', 'Train', 'Test']

l.set_index('min_sample_leaf')

plt.plot(l['min_sample_leaf'], l['Train'], label = "Train")

plt.plot(l['min_sample_leaf'], l['Test'], label = "Test")

plt.title('train/test as function of min_sample_leaf')

plt.xlabel('Minimum number of samples required to be at a leaf node')

plt.ylabel('Accuracy')

plt.legend()
l = []

for i in range(100): 

    forest = RandomForestClassifier(n_estimators = 100, min_samples_split = i+2)

    out = cross_validate(forest, x, y, return_train_score = True, cv=10)

    l.append((i+1, np.mean(out['train_score']), np.mean(out['test_score'])))
l = pd.DataFrame(l)

l.columns = ['min_samples_split', 'Train', 'Test']

l.set_index('min_samples_split')

plt.plot(l['min_samples_split'], l['Train'], label = "Train")

plt.plot(l['min_samples_split'], l['Test'], label = "Test")

plt.title('train/test as function of min_samples_split')

plt.xlabel('Minimum number of samples required to split a leaf node')

plt.ylabel('Accuracy')

plt.legend()
eta = []

for i in range(100): 

    boost = XGBClassifier(eta = (i+1)/100)

    out = cross_validate(boost, x, y, return_train_score = True, cv=10)

    eta.append((i+1, np.mean(out['train_score']), np.mean(out['test_score'])))
eta = pd.DataFrame(eta)

eta.columns = ['eta', 'Train', 'Test']

eta.set_index('eta')

plt.plot(eta['eta'], eta['Train'], label = "Train")

plt.plot(eta['eta'], eta['Test'], label = "Test")

plt.title('train/test as function of learning rate')

plt.xlabel('Learning Rate')

plt.ylabel('Accuracy')

plt.legend()
from sklearn.model_selection import GridSearchCV
boost = XGBClassifier()

#lambda and alpha are regularization terms that determine the importance of the regularization term

#Regularization penalizes the complexity of our hypothesis function

parameters = {'scale_pos_weight':[0,1,2,3],'learning_rate':[0.1,0.2,0.3,0.4,0.5], 'gamma':list(range(1,10)), 'reg_lambda':list(range(1,10)), 'reg_alpha': list(range(0,10))}

grid = GridSearchCV(boost, parameters)
grid.fit(x,y)
grid.best_score_
grid.best_params_
from xgboost import XGBClassifier

boost = XGBClassifier(gamma = 1, learning_rate= 0.5, reg_alpha= 1, reg_lambda= 7, scale_pos_weight= 2)

out = cross_validate(boost, x, y, return_train_score = True, cv=10)

print('Train Score:', np.mean(out['train_score']), 'Test Score:', np.mean(out['test_score']))
# Gets score up to 81%

from xgboost import XGBClassifier

boost = XGBClassifier(gamma = 1, learning_rate= 0.4, reg_alpha= 0, reg_lambda= 1)

out = cross_validate(boost, x, y, return_train_score = True, cv=10)

print('Train Score:', np.mean(out['train_score']), 'Test Score:', np.mean(out['test_score']))