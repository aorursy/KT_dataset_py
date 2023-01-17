import pandas as pd
import seaborn as sns
import numpy as np

from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
data = pd.read_csv('../input/data.csv')
data.head()
data.isnull().sum()
df = data.drop(['Unnamed: 32','id'], axis=1)

print("Final Columns in Dataset")
print('='*50)
print(df.isnull().sum())
print('='*50)
X = df.iloc[:,1:]
y = np.where(df['diagnosis']=='M', 1,0).astype(int)

X_train, X_test, y_Train, y_Test = train_test_split(X, y, test_size =0.2, random_state =5)
model = RandomForestClassifier()

print("Default Parameters ")
print('='*50)

pprint(model.get_params())

print('='*50)
bootstrap_v = [True, False]
n_estimators_v = list(range(100,2000,200))
criterion = ['gini', 'entropy']
min_sample_leaf_v = list(range(1,5,2))
max_features_v = ['sqrt', 'log2']

grid_params  = {
    'bootstrap' : bootstrap_v,
    'n_estimators' : n_estimators_v,
    'criterion' : criterion,
    'min_samples_leaf' : min_sample_leaf_v,
    'max_features' : max_features_v
}

print("Tuning Parameters")
print('='*50)

pprint(grid_params)
print('='*50)
grid_search = GridSearchCV(estimator=model, param_grid=grid_params, cv=3, verbose=1)
grid_search.fit(X_train, y_Train)

print('Best Parameters for our classsifier')
print('='*50)
print(grid_search.best_params_)
print('='*50)
def evaluate(model, X, y):
    
    pprint(model.get_params())
    print('=='*50)
    predictions = model.predict(X)
    report = classification_report(y, predictions)
    
    score = accuracy_score(y_true= y, y_pred= predictions)
    
    print(report)
    print('=='*50)
    print("{} {:0.2f}%".format("Accuracy Score :: ", score*100))
    
    
evaluate(grid_search.best_estimator_, X_test, y_Test)
model.fit(X_train, y_Train)
evaluate(model, X_test, y_Test)