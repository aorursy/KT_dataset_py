import os
import pandas as pd
import numpy as np # fundamental package for scientific computing with Python
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
print("Train dimensions: ", train.shape)
print("Test dimensions: ", test.shape)
targetDistribution = pd.value_counts(train['Target'].values, sort=True)
targetDistribution.plot.barh()
complete_df = pd.concat([train, test], ignore_index=True)
complete_df.tail()
print(complete_df.columns[complete_df.isnull().any()].tolist())
missing_set = complete_df[['v2a1', 'v18q1', 'rez_esc', 'meaneduc', 'SQBmeaned']]
columns = missing_set.columns
percent_missing = missing_set.isnull().sum() * 100 / len(missing_set)
missing_value_df = pd.DataFrame({'column_name': columns,
                                 'percent_missing': percent_missing})
print(missing_value_df)
columns_to_drop = ['v2a1', 'v18q1', 'rez_esc']
train.drop(columns_to_drop, inplace=True, axis=1)
test.drop(columns_to_drop, inplace=True, axis=1)
train["SQBmeaned"].fillna(train["SQBmeaned"].mean(), inplace=True)
train["meaneduc"].fillna(train["meaneduc"].mean(), inplace=True)
test["SQBmeaned"].fillna(test["SQBmeaned"].mean(), inplace=True)
test["meaneduc"].fillna(test["meaneduc"].mean(), inplace=True)
print(train.isnull().values.any())
print(test.isnull().values.any())
pd.set_option('display.max_columns', 500)
train.head()
train.tail()
columns_to_drop2 = ['idhogar', 'Id']
train.drop(columns_to_drop2, inplace=True, axis=1)

id_test = test['Id']
idHogar_test =  test['idhogar']
test.drop(columns_to_drop2, inplace=True, axis=1)
print(train.dependency.unique())
print(train.edjefe.unique())
print(train.edjefa.unique())
warnings.filterwarnings(action='once')

train.dependency[train.dependency == 'yes'] = 1 
train.dependency[train.dependency == 'no']   = 0

train.edjefe[train.edjefe == 'yes'] = 1 
train.edjefe[train.edjefe == 'no']   = 0

train.edjefa[train.edjefa == 'yes'] = 1 
train.edjefa[train.edjefa == 'no']   = 0

test.dependency[test.dependency == 'yes'] = 1 
test.dependency[test.dependency == 'no']   = 0

test.edjefe[test.edjefe == 'yes'] = 1 
test.edjefe[test.edjefe == 'no']   = 0

test.edjefa[test.edjefa == 'yes'] = 1 
test.edjefa[test.edjefa == 'no']   = 0

print("Unique values for dependency: ")
print(train.dependency.unique())
print("Unique values for edjefe: ")
print(train.edjefe.unique())
print("Unique values for edjefa: ")
print(train.edjefa.unique())
y = train['Target'].values
train_Feature = train.copy()
train_Feature.drop(['Target'],inplace=True, axis=1 )
X = train_Feature.values
# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0,
                             n_jobs =-1)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %s (%f)" % (f + 1, train.columns[indices[f]], importances[indices[f]]))
columns_to_drop3 = ['elimbasu5']
train.drop(columns_to_drop3, inplace=True, axis=1)
test.drop(columns_to_drop3, inplace=True, axis=1)
f, ax = plt.subplots(figsize = (138,138))
sns.heatmap(train.corr(),annot= True)
plt.show()
def trimm_correlated(df_in, threshold):
    df_corr = df_in.corr(method='pearson', min_periods=1)
    df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)]*2, dtype=bool))).abs() > threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = df_in[un_corr_idx]
    return df_out
y = train['Target'].values
train.drop(['Target'],inplace=True, axis=1 )
train = trimm_correlated(train, 0.95)
test = test[train.columns]
print(train.shape)
print(test.shape)
X = train.values
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
def evaluate(predictions, test_features, test_labels):
    accuracy = accuracy_score(test_labels, predictions)
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(test_labels, predictions)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['1','2','3','4'],
                          title='Confusion matrix, without normalization')
    
    plt.show()
def runTestingModel(model, X, y, n_folds ,param_grid):
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                              cv = n_folds, n_jobs = -1, verbose = 2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)
    print(grid_search.best_params_)
    best_grid = grid_search.best_estimator_
    predictions = best_grid.predict(X_test)
    grid_accuracy = evaluate(predictions, X_test, y_test)
    return best_grid, predictions
X
y
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'n_estimators': [100, 200, 300, 1000]
}
rf = RandomForestClassifier()
model, predictions  = runTestingModel(rf, X, y, 3, param_grid)
print(predictions)
catColumns =[]
quanColumns =[]
for col in train.columns:
    if len(train[col].unique()) < 15:
        catColumns.append(col)
    else:
        quanColumns.append(col)
print(quanColumns)            
catColumns
print(train['escolari'].unique())
print(train['overcrowding'].unique())
print(train['SQBhogar_nin'].unique())
catColumns.remove('SQBhogar_nin')
quanColumns.append('SQBhogar_nin')
def age_buckets(x): 
    if x < 15: return 1
    elif x < 30: return 2
    elif x < 40: return 3
    elif x < 50: return 4
    else : return 5

train['age'] = train.age.apply(age_buckets)
test['age'] = test.age.apply(age_buckets)
train['age'].head()
print(catColumns)
complete_df = pd.concat([train, test], keys=[0,1])
print(test.shape)
print(train.shape)
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit(complete_df[catColumns])

onehotlabels = enc.transform(complete_df[catColumns]).toarray()

onehotlabels_Train = onehotlabels[:9557]
onehotlabels_Test = onehotlabels[9557:]
onehotlabels_Train.shape
listType = list(train[quanColumns].dtypes)
listType
test.columns
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX_Test = scaler.fit_transform(test[quanColumns])
rescaledX_Train = scaler.fit_transform(train[quanColumns])
print(onehotlabels_Train.shape)
print(rescaledX_Train.shape)
X = np.concatenate((onehotlabels_Train ,rescaledX_Train),axis=1)
print(X.shape)
X_test = np.concatenate((onehotlabels_Test ,rescaledX_Test),axis=1)
import pickle
f = open('Variables.pckl', 'wb')
pickle.dump([X, y, X_test,id_test], f)
f.close()
import pickle

f = open('Variables.pckl', 'rb')
X, y, X_test,id_test = pickle.load(f)
f.close()
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,220,500],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}
rf = RandomForestClassifier()
model, predictions = runTestingModel(rf, X, y, 2, param_grid)
print(predictions)
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
knn = KNeighborsClassifier()
model, predictions = runTestingModel(knn, X, y, 5, param_grid)
from sklearn import svm

Cs = [1, 10,100, 1000]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
sv = svm.SVC(kernel='rbf')    
model, predictions = runTestingModel(sv, X, y, 3, param_grid)

def submit(selected_model, X_train, y, test, id_test):
    """Train and test a model on the dataset"""
    model.fit(X_train, y)
    predictions = model.predict(test)    
    file = pd.DataFrame()
    file['Id'] = id_test
    file['Target'] = predictions
    file.to_csv('submission.csv', index=False)
submit(model, X, y, X_test,id_test )