# Import required libraries into current IDE.

from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold, StratifiedKFold



from imblearn.pipeline import make_pipeline 

from sklearn.metrics import precision_recall_fscore_support as score



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import  RandomForestClassifier

from sklearn.linear_model import LogisticRegressionCV

from sklearn.ensemble import  GradientBoostingClassifier, VotingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



from sklearn.metrics import  accuracy_score, confusion_matrix,roc_auc_score, roc_curve, auc 



from sklearn.preprocessing import StandardScaler, binarize

from sklearn import model_selection



import matplotlib.pyplot as plt

from scipy.stats import skew

import pandas as pd

import numpy as np

import seaborn as sns



%matplotlib inline

import warnings

warnings.filterwarnings(action="ignore", category=DeprecationWarning)

warnings.filterwarnings(action="ignore", category=FutureWarning)
# calculate the fpr and tpr for all thresholds of the classification

def ROC_Curve (train_X, test_X, train_Y, test_Y, rf1):

    from sklearn.metrics import  roc_curve

    pred = rf1.predict_proba(test_X)

    preds = pred[:,1]

    fpr, tpr, threshold = roc_curve(test_Y, preds)

    roc_auc = auc(fpr, tpr)



    pred_train =  rf1.predict_proba(train_X)

    pred_train = pred_train[:,1]

    fpr_train, tpr_train, threshold_train = roc_curve(train_Y, pred_train)

    roc_auc_train = auc(fpr_train, tpr_train)



# method I: plt

    import matplotlib.pyplot as plt

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr_train, tpr_train, 'b', label = 'Train AUC = %0.2f' % roc_auc_train, color='green')

    plt.plot(fpr, tpr, 'b', label = 'Test AUC = %0.2f' % roc_auc, color = 'blue')



    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()
data = pd.read_csv('../input/train.csv')
data.shape
numerical_features = data.select_dtypes(include=[np.number])

numerical_features.columns
data.describe()
data.drop('PassengerId', axis = 1).hist(figsize=(30,20), layout=(4,3))

plt.plot()
skew_values = skew(data[numerical_features.columns], nan_policy = 'omit')

pd.concat([pd.DataFrame(list(numerical_features.columns), columns=['Features']), 

           pd.DataFrame(list(skew_values), columns=['Skewness degree'])], axis = 1,)       
missing_values = data.isnull().sum().sort_values(ascending = False)

percentage_missing_values = missing_values/len(data)

combine_data = pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values count', 'Percentage'])

pd.pivot_table(combine_data, index=combine_data.index,margins=True )
plt.figure(figsize=(20,10))

sns.heatmap(data.drop('PassengerId', axis = 1).corr(), square=True, annot=True, vmax= 1,robust=True, yticklabels=1)

plt.show()
plt.figure(figsize=(20,5))

sns.boxplot(data = data.drop('PassengerId', axis = 1))

plt.show()



# Fare has lot of outliers followed by Age, Sibsp and Parch
# Lat's see survival and pclass relation

var = 'Fare'

plt.scatter(x = data[var], y = data['Survived'])

plt.xlabel('Fare')

plt.ylabel('Survived')

plt.show()
plt.figure(figsize=(20,5))

sns.boxplot(x =data[var])

plt.show()
data.drop(data[data['Fare']> 100].index,  inplace= True)
data.drop(['Age', 'Cabin'], axis = 1, inplace=True)
data.dropna(inplace=True)
categorical_features = data.select_dtypes(include=[np.object])

categorical_features.columns
print('Sex has {} unique values: {}'.format(len(data.Sex.unique()),data.Sex.unique()))

print('Embarked has {} unique values: {}'.format(len(data.Embarked.unique()),data.Embarked.unique()))
data.drop(['Name', 'Ticket'], axis = 1, inplace=True)
data  = pd.get_dummies(data)
data.head()
X = data.drop(['Survived', 'PassengerId'], axis = 1)

y = data['Survived']



from sklearn.feature_selection import  SelectKBest

select = SelectKBest(k = 7)

X_new = select.fit_transform(X, y)



scale = StandardScaler().fit(X_new)

X_new = scale.transform(X_new)



trainX, testX, trainY, testY = train_test_split(X_new, y, stratify = y, test_size = 0.2)



kfold = StratifiedKFold(10, shuffle = True)
from sklearn.linear_model import  RidgeClassifierCV

def rmse_cv(model, X_train, y):

    rmse = (cross_val_score(model, X_train, y, scoring="roc_auc", cv=3))

    return(rmse*100)



models = [LogisticRegressionCV(),

             RidgeClassifierCV(),

             SVC(),

             RandomForestClassifier(n_estimators=50, min_samples_leaf=50, oob_score=True),

             GradientBoostingClassifier(),

             KNeighborsClassifier(n_neighbors = 3),]



names = ['LR','Ridge','svm','RF','GB','KNN']



for model,name in zip(models,names):

    score = rmse_cv(model,trainX,trainY)

    print("{}: {:.6f}, {:4f}".format(name,score.mean(),score.std()))
def train_RF(est, n_depth):

    rf = GradientBoostingClassifier(n_estimators=est, max_depth=n_depth, random_state=10)

    rf_model = rf.fit(trainX, trainY)

    y_pred = rf_model.predict(testX)

    precision, recall, fscore, support = score(testY, y_pred)

    

    print('Est: {} Depth: {} -- precison {} Recall {} Accuracy {} AUC {}'.format(

        est, n_depth, round(precision, 3), round(recall, 3), 

          round(((testY == y_pred).sum()/len(y_pred)), 3), round(roc_auc_score(testY, rf_model.predict(testX))),3))
for n_estimaator in [20, 30,50,70,100]:

    for depth in [4,5,6,7,8]:

        train_RF(n_estimaator, depth)
rf = GradientBoostingClassifier(random_state=10).fit(trainX, trainY)

#print('Train Score', roc_auc_score(trainY, rf.predict(trainX)))

print('Test Score', roc_auc_score(testY, rf.predict(testX)))

#ROC_Curve (trainX, testX, trainY, testY, rf)
print('Test Score', roc_auc_score(testY, rf.predict(testX)))
# Threshold Estimation

def compute_threshold (trainY, pred):

    #pred = estimator.predict_proba(trainX)

    preds = pred[:,1]

    fpr, tpr, threshold = roc_curve(trainY, preds)



    optimal_idx = np.argmax(tpr - fpr)

    optimal_threshold = threshold[optimal_idx]

    return(optimal_threshold)
# Create the parameter grid based on the results of random search 

param_grid = {

    #'max_depth': [3,4],

    #'min_samples_leaf': [3, 4, 5],

   # 'min_samples_split': [8, 10, 12],

    'n_estimators': [20,30,50],

    'learning_rate':[0.01,0.1,10]

}



grid_search = GridSearchCV(estimator = GradientBoostingClassifier(max_features='sqrt', 

                                                              random_state = 10), 

                           param_grid = param_grid, 

                           cv = 3, n_jobs = -1, verbose = 2, scoring = 'roc_auc')

grid_search.fit(trainX, trainY)



#print('Train Score', roc_auc_score(trainY, grid_search.best_estimator_.predict(trainX)))

print('Test Score', roc_auc_score(testY, grid_search.best_estimator_.predict(testX)))

pd.DataFrame(data = grid_search.cv_results_).sort_values('mean_test_score', ascending = False)
proba = grid_search.best_estimator_.predict_proba(trainX)
compute_threshold (trainY, proba)
proba_test = grid_search.best_estimator_.predict_proba(testX)
predicted_val = (proba_test[:,1:] > 0.2676794)
predicted_val

accuracy_score(testY, predicted_val)
from sklearn.preprocessing import  Binarizer

for threshold in [0.4,0.41,0.4,0.45, 0.5,0.6,0.7,0.8,0.9]:

    y_proba = grid_search.best_estimator_.predict_proba(trainX)

    y_proba_new = Binarizer(threshold= threshold).fit_transform(y_proba) 

    print(roc_auc_score(trainY, y_proba_new[:,1:]))
for threshold in [0.4,0.5,0.6, 0.7, 0.8,0.9]:

    y_proba = grid_search.best_estimator_.predict_proba(testX)

    y_proba_new = Binarizer(threshold= threshold).fit_transform(y_proba) 

    print(roc_auc_score(testY, y_proba_new[:,1:]))
test_data = pd.read_csv('../input/test.csv')



test_data.isnull().sum().sort_values(ascending = False)

test_data.drop(['Cabin','Age', 'Ticket','Name'], axis = 1, inplace=True)

test_data.fillna(test_data['Fare'].mean(), axis = 1, inplace=True)

test_data_dummy = pd.get_dummies(test_data)



X_new = select.transform(test_data_dummy.drop('PassengerId', axis = 1))

test_data_X = scale.transform(X_new)

prediction = grid_search.best_estimator_.predict_proba(test_data_X)

predicted_test_val = (prediction[:,1:] > 0.2676794)

final = pd.DataFrame(test_data['PassengerId'])

final['Survived'] = predicted_test_val
# final.to_csv('../input/21Oct2018_2.csv')
bn = Binarizer(threshold= 0.6) 

y_predictions_new = bn.fit_transform( prediction)