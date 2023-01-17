import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



pd.set_option('display.max_columns', 100)
df = pd.read_csv('/kaggle/input/eval-lab-2-f464/train.csv')
df.shape
df.head()
df.isnull().sum()
df.corr()
X = df[["chem_0","chem_1", "chem_2", "chem_3", "chem_4", "chem_5", "chem_6", "chem_7", "attribute"]].copy()

y = df["class"].copy()
X_encoded =  pd.get_dummies(X)

X_encoded.head()
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

X_scaled = scaler.fit_transform(X_encoded) 

X_scaled
font = {'fontname':'Arial', 'size':'14'}

title_font = { 'weight' : 'bold','size':'16'}

plt.hist(df['class'], bins=20)

plt.title("ratings")

plt.show()
corr = df.corr()



mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



f, ax = plt.subplots(figsize=(12, 9))



cmap = sns.diverging_palette(220, 10, as_cmap=True)



sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.5, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
sns.regplot(x='chem_1', y='class', data=df)
sns.regplot(x='chem_4', y='class', data=df)
from numpy import loadtxt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30)
from xgboost import XGBClassifier

# fit model no training data

model = XGBClassifier(max_depth=6, min_child_weight = 2, gamma = 0, subsample=0.9, colsample_bytree = 0.9)

model.fit(X_train, y_train)



# make predictions for test data

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]



# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
from sklearn.ensemble import RandomForestClassifier as RFC



rfc_b = RFC(n_estimators=350, min_samples_split=4)

rfc_b.fit(X_train,y_train)

y_pred = rfc_b.predict(X_train)



print('Train accuracy score:',accuracy_score(y_train,y_pred))

print('Test accuracy score:', accuracy_score(y_test,rfc_b.predict(X_test)))
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier().fit(X_train,y_train)

from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)



print(accuracy)
from sklearn.ensemble import BaggingClassifier



bag_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier()).fit(X_train,y_train)

y_pred_bag = bag_clf.predict(X_test)

bag_acc = accuracy_score(y_test,y_pred_bag)



print(bag_acc)
from sklearn.ensemble import RandomForestClassifier



rf_clf = RandomForestClassifier(bootstrap= True,

 max_depth= 50,

 max_features= 'sqrt',

 min_samples_leaf= 1,

 min_samples_split= 2,

 n_estimators= 200).fit(X_train,y_train)

y_pred_rf = rf_clf.predict(X_test)

rf_acc = accuracy_score(y_test,y_pred_rf)



print(rf_acc)
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)

{'bootstrap': [True, False],

 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

 'max_features': ['auto', 'sqrt'],

 'min_samples_leaf': [1, 2, 4],

 'min_samples_split': [2, 5, 10],

 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}



# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(X_train, y_train)



rf_random.best_params_
from sklearn.ensemble import AdaBoostClassifier



ab_clf = AdaBoostClassifier().fit(X_train,y_train)

y_pred_ab = ab_clf.predict(X_test)

ab_acc = accuracy_score(y_test,y_pred_ab)



print(ab_acc)
from sklearn.ensemble import GradientBoostingClassifier



gb_clf = GradientBoostingClassifier().fit(X_train,y_train)

y_pred_gb = gb_clf.predict(X_test)

gb_acc = accuracy_score(y_test,y_pred_gb)



print(gb_acc)


n = len(X_train)

X_A = X_train[:n//2]

y_A = y_train[:n//2]

X_B = X_train[n//2:]

y_B = y_train[n//2:]



clf_1 = DecisionTreeClassifier().fit(X_A, y_A)

y_pred_1 = clf_1.predict(X_B)

clf_2 = RandomForestClassifier(n_estimators=100).fit(X_A, y_A)

y_pred_2 = clf_2.predict(X_B)

clf_3 = GradientBoostingClassifier().fit(X_A, y_A)

y_pred_3 = clf_3.predict(X_B)



X_C = pd.DataFrame({'RandomForest': y_pred_2, 'DeccisionTrees': y_pred_1, 'GradientBoost': y_pred_3})

y_C = y_B

X_C.head()



X_D = pd.DataFrame({'RandomForest': clf_2.predict(X_test), 'DeccisionTrees': clf_1.predict(X_test), 'GradientBoost': clf_3.predict(X_test)})

y_D = y_test



from xgboost import XGBClassifier



xgb_clf = XGBClassifier().fit(X_C,y_C)

y_pred_xgb = xgb_clf.predict(X_D)

xgb_acc = accuracy_score(y_D,y_pred_xgb)



print(xgb_acc)
from sklearn.ensemble import VotingClassifier



estimators = [('rf', RandomForestClassifier()), ('bag', BaggingClassifier()) ,('gb', GradientBoostingClassifier())]



soft_voter = VotingClassifier(estimators=estimators, voting='soft').fit(X_train,y_train)

hard_voter = VotingClassifier(estimators=estimators, voting='hard').fit(X_train,y_train)



soft_acc = accuracy_score(y_test,soft_voter.predict(X_test))

hard_acc = accuracy_score(y_test,hard_voter.predict(X_test))



print("Acc of soft voting classifier:{}".format(soft_acc))

print("Acc of hard voting classifier:{}".format(hard_acc))
df1 = pd.read_csv('/kaggle/input/eval-lab-2-f464/test.csv')
df1.shape
df1.isnull().sum()
X_pred = df1[["chem_0","chem_1", "chem_2", "chem_3", "chem_4", "chem_5", "chem_6", "chem_7", "attribute"]].copy()
X_pred_encoded =  pd.get_dummies(X_pred)

X_pred_encoded.head()
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler



scaler = RobustScaler()

X_pred_scaled = scaler.fit_transform(X_pred_encoded) 



X_pred_scaled
X_pred_encoded.head()
y_pred = rf_random.predict(X_pred_encoded)

predictions = [round(value) for value in y_pred]
submission = pd.DataFrame({'id':df1['id'],'class':predictions})

submission.shape
filename = 'predictions_sub.csv'



submission.to_csv(filename,index=False)