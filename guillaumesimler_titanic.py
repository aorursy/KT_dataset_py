# Import essentials 
## Data science
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## Classic vision
from pathlib import Path
MAIN_PATH = Path('../input/titanic/')

dataset = pd.read_csv(MAIN_PATH / 'train.csv')
dataset.head(10)
def handle_age(df: pd.DataFrame) -> pd.DataFrame:
    """ Handles the age by replacing it by the median value"""
    
    # Age
    median = df['Age'].median()
    
    df['Age'] = df['Age'].fillna(median)
    
    #Fare
    mean = df['Fare'].mean()
    df['Fare'] = df['Fare'].fillna(mean)
    
    return df
dataset = handle_age(dataset)
dataset.describe()
def split_the_name(name:str) -> str:
    
    title = name.split(', ')[1].split('.')[0]
    
    if title not in ['Mrs', "Mr", "Master"]:
        title = 'others'
    
    return title
split_the_name('Braund, Mr. Owen Harris')
dataset['Title'] = dataset['Name'].apply(split_the_name)
plt.hist(dataset['Title'])
def handle_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """ Transform the categorical data 
            - Sex -> female counts als first vakue
            - Pclass -> 1 counts als first value
        in dummy columns
        
        return:
            - the df 
    """
    df = pd.get_dummies(df, columns=['Sex', 'Pclass', 'Embarked', 'Title'], drop_first= True)
    
    return df
dataset = handle_categorical(dataset)
dataset.describe()
corrmap = dataset.corr()
heatmeap = sns.heatmap(corrmap, cbar=True, annot=True,fmt='.2f')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
def prepare_X_Y(df: pd.DataFrame, X_val: list, y_val: list = []):
    """ transform the dataframe into the desired series"""
    
    X = df[X_val].values
    
    if y_val:
        y = df[y_val].values.ravel()
        
        return X, y
    
    return X
    
X_val = [ 'Age', 'Parch','SibSp','Fare', 'Sex_male' ,'Pclass_2', 'Pclass_3', 'Embarked_Q', 'Embarked_S', 'Title_Mr', 'Title_Mrs', 'Title_others']
y_val = ['Survived']

X, y = prepare_X_Y(dataset, X_val, y_val)
# Split training and test_set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.10, random_state=666)
# Scale the value
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
train_values = []
test_values = []
delta = []
captions = []
class_logic_regr = LogisticRegression()
y_pred = class_logic_regr.fit(X_train, y_train)
y_pred = class_logic_regr.predict(X_train)
def analyse_results(y_train, y_pred, X_train):
    cm = confusion_matrix(y_train, y_pred)
    accur = (cm[0][0] + cm[1][1])/len(X_train)

    print(f'The base line would be  {round(accur, 2)} of true prediction (True survival, true demise)')
    print(f'----\n Confusion Matrix : \n\n {cm}\n----')
    
    return accur
accur_train = analyse_results(y_train, y_pred, X_train)
y_pred2 = class_logic_regr.predict(X_test)
accur_test = analyse_results(y_test, y_pred2, X_test)
print(f'Delta between training and test set is: \n\n\t\t {round(accur_test - accur_train, 2)}')
train_values.append(accur_train)
test_values.append(accur_test)
delta.append(accur_test - accur_train)
captions.append('Logistic Regr.')
class_svm = SVC(C = 1, gamma= 0.25, kernel= 'rbf')
class_svm.fit(X_train, y_train)
y_pred = class_svm.predict(X_train)
accur_train = analyse_results(y_train, y_pred, X_train)
y_pred2 = class_svm.predict(X_test)
accur_test = analyse_results(y_test, y_pred2, X_test)
print(f'Delta between training and test set is: \n\n\t\t {round(accur_test - accur_train, 2)}')
OPTIMIZE = False
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf', 'adam'], 'gamma': [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]}]


grid_search = GridSearchCV(estimator = class_svm,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)

if OPTIMIZE:
    grid_search = grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
else:
    best_accuracy = best_parameters = 'No optimization run'

best_parameters
best_accuracy
train_values.append(accur_train)
test_values.append(accur_test)
delta.append(accur_test - accur_train)
captions.append('SVM')
class_NB = GaussianNB()
class_NB.fit(X_train, y_train)
y_pred = class_NB.predict(X_train)
accur_train = analyse_results(y_train, y_pred, X_train)
y_pred2 = class_NB.predict(X_test)
accur_test = analyse_results(y_test, y_pred2, X_test)
print(f'Delta between training and test set is: \n\n\t\t {round(accur_test - accur_train, 2)}')
train_values.append(accur_train)
test_values.append(accur_test)
delta.append(accur_test - accur_train)
captions.append('Naive Bayse')
class_RForest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
class_RForest.fit(X_train, y_train)
y_pred = class_RForest.predict(X_train)
accur_train = analyse_results(y_train, y_pred, X_train)
y_pred2 = class_RForest.predict(X_test)
accur_test = analyse_results(y_test, y_pred2, X_test)
print(f'Delta between training and test set is: \n\n\t\t {round(accur_test - accur_train, 2)}')
OPTIMIZE = False
parameters = [{'n_estimators':[3,5,7,10]}]


grid_search = GridSearchCV(estimator = class_RForest,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)

if OPTIMIZE:
    grid_search = grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_
else:
    best_accuracy = best_parameters = 'No optimization run'

best_parameters
best_accuracy
train_values.append(accur_train)
test_values.append(accur_test)
delta.append(accur_test - accur_train)
captions.append('Random Forest')
class_XGB = XGBClassifier()
class_XGB.fit(X_train, y_train)
y_pred = class_XGB.predict(X_train)
accur_train = analyse_results(y_train, y_pred, X_train)
y_pred2 = class_XGB.predict(X_test)
accur_test = analyse_results(y_test, y_pred2, X_test)
print(f'Delta between training and test set is: \n\n\t\t {round(accur_test - accur_train, 2)}')
train_values.append(accur_train)
test_values.append(accur_test)
delta.append(accur_test - accur_train)
captions.append('XGBoost')
methods = pd.DataFrame([train_values, test_values, delta], columns= captions, index=['training', 'test', 'delta'])
methods
classifiers = [('Logist', class_logic_regr),
               ('SVM', class_svm),
               ('RandomForest', class_RForest),
               ('XGB', class_XGB)]
class_vote = VotingClassifier(classifiers, voting='hard', weights=[4, 4, 2 , 2])
class_vote.fit(X_train, y_train)
y_pred = class_vote.predict(X_train)
accur_train = analyse_results(y_train, y_pred, X_train)
y_pred2 = class_vote.predict(X_test)
accur_test = analyse_results(y_test, y_pred2, X_test)
testdata = pd.read_csv(MAIN_PATH / 'test.csv')
testdata ['Title'] = testdata['Name'].apply(split_the_name)
testdata = handle_categorical(testdata)
testdata = handle_age(testdata)
testdata.describe()
# simplify the data set

X_testdata = prepare_X_Y(testdata, X_val)
# Standardize data
X_testdata = sc.transform(X_testdata)
X_testdata
Y_testdata = class_vote.predict(X_testdata)
Y_testdata
testdata['Survived'] = Y_testdata
export_data = testdata[['PassengerId', 'Survived']]
export_data.to_csv('submission.csv', index=False)