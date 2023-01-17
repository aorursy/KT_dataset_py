
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data = train.append(test)

print('Train has {} files and {} columns'.format(train.shape[0], train.shape[1]))

train.info()
train.head()
print('Train has {} files and {} columns'.format(test.shape[0], test.shape[1]))

test.head()

test.info()

train.select_dtypes(int).nunique()
test.select_dtypes(int).nunique() 

print('Empty values by column in Train ', (train.isnull().sum()))

print('Empty values by column in Test ',(test.isnull().sum()))

data['Title'] = data['Name']

for name_string in data['Name']:
    data['Title'] = data['Name'].str.extract('([A-Za-z]+)\.', expand=True)


mapping = {'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr', 'Don': 'Mr', 'Mme': 'Miss',
          'Jonkheer': 'Mr', 'Lady': 'Mrs', 'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'}
data.replace({'Title': mapping}, inplace=True)
titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
for title in titles:
    age_to_impute = data.groupby('Title')['Age'].median()[titles.index(title)]
    data.loc[(data['Age'].isnull()) & (data['Title'] == title), 'Age'] = age_to_impute
    
# Substituting Age values in TRAIN and TEST:
train['Age'] = data['Age'][:891]
test['Age'] = data['Age'][891:]
data['Family_Size'] = data['Parch'] + data['SibSp']

train['Family_Size'] = data['Family_Size'][:891]
test['Family_Size'] = data['Family_Size'][891:]
drop_column = [ 'Cabin', 'Ticket',  'Parch', 'SibSp', 'Name', 'Embarked']
train.drop(drop_column, axis = 1, inplace = True)
test.drop(drop_column, axis = 1, inplace = True)

mapping = {'male':1, 'female':0}
train['Sex'] = train['Sex'].replace(mapping).astype(np.float64)
test['Sex'] = test['Sex'].replace(mapping).astype(np.float64)

test['Fare'].fillna(value = test['Fare'].mode()[0], inplace = True)


from collections import Counter
def detect_outliers(df,n,features):

    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers
Outliers_to_drop = detect_outliers(train,2,[["Pclass", "Sex", "Age", "Fare","Family_Size","Survived"]])
train.loc[Outliers_to_drop]

train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
plt.figure(figsize=[16,18])
plt.subplot(234)
plt.hist(x = [train[train['Survived']==1]['Fare'], train[train['Survived']==0]['Fare']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.title('Fare Histogram by Survival')
plt.xlabel('Fare ($)')
plt.ylabel('# of Passengers')
plt.legend()
sns.barplot(x="Pclass", y="Survived", data=train)
plt.ylabel("Survival Rate")
plt.title("Distribution of Survival by class")
plt.show()

def correlation_heat(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heat(train[["Pclass", "Sex", "Age",  "Fare","Survived"]])

features = ["Pclass", "Sex", "Age", "Fare", "Family_Size"]
#Columns to wrok with in Train
X_train = train[features] #define training features set
y_train = train["Survived"] 
#Columns to work with in Test
X_test = test[features] #define testing features set

from sklearn.model_selection import train_test_split 
X_training, X_valid, y_training, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0) 


from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import warnings 
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, make_scorer

scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')

warnings.filterwarnings('ignore', category = ConvergenceWarning)
warnings.filterwarnings('ignore', category = DeprecationWarning)
warnings.filterwarnings('ignore', category = UserWarning)
model_results = pd.DataFrame(columns = ['model', 'cv_mean', 'cv_std'])

def cv_model(X_train, y_train, model, name, model_results = None, sort = True):
    #10 fold cross validation model##
    cv_scores = cross_val_score(model, X_train, y_train, cv = 10,
                                scoring = scorer, n_jobs = -1)
    print('Mean ', round(cv_scores.mean(), 5),'STd', round(cv_scores.std(), 5))
    if model_results is not None:
        model_results = model_results.append(pd.DataFrame({'model':name,
                                                'cv_mean': cv_scores.mean(),
                                                'cv_std' : cv_scores.std()},
                                                index = [0]), ignore_index = True)
    return model_results

model_results = cv_model(X_train, y_train, LinearSVC(), 'LSVC', 
                         model_results)

model_results = cv_model(X_train, y_train, 
                         GaussianNB(), 'GNB', model_results)

model_results = cv_model(X_train, y_train, 
                         MLPClassifier(hidden_layer_sizes=(16, 32, 64, 64, 32)),
                         'MLP', model_results)

model_results = cv_model(X_train, y_train, 
                          LinearDiscriminantAnalysis(), 
                          'LDA', model_results)

model_results = cv_model(X_train, y_train, 
                         RidgeClassifierCV(), 'RIDGE', model_results)

for n in [5, 10, 20]:
    print('\nKNN with {n} neighbors\n', n)
    model_results = cv_model(X_train, y_train, 
                             KNeighborsClassifier(n_neighbors = n),
                             n, model_results)
    
from sklearn.ensemble import ExtraTreesClassifier

model_results = cv_model(X_train, y_train, 
                         ExtraTreesClassifier(n_estimators = 100, random_state = 10),
                         'EXT', model_results)

model_results = cv_model(X_train, y_train,
                          RandomForestClassifier(100, random_state=10),
                              'RF', model_results)
from xgboost import XGBClassifier
model_results = cv_model(X_train, y_train,
                          XGBClassifier(n_estimators= 100, random_state=10),
                              'XGB', model_results)

model_results.set_index('model', inplace = True)
print(model_results)
model_results['cv_mean'].plot.bar(color = 'orange', figsize = (8, 6),
                                  yerr = list(model_results['cv_std']))
plt.title('Model F1 Score Results');
plt.ylabel('Mean F1 Score (with error bar)');
model_results.reset_index(inplace = True)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score 

xg_clf = XGBClassifier()
parameters_rf = {'n_estimators' : [200],'learning_rate': [0.1],
              'max_depth': [4], "min_child_weight":[6], "gamma":[0], "subsample":[0.80] 
                              }


grid_rf = GridSearchCV(xg_clf, parameters_rf, scoring=make_scorer(accuracy_score))
#make_scorer(accuracy_score)
grid_rf.fit(X_training, y_training)
xg_clf = grid_rf.best_estimator_

pred_xg = xg_clf.predict(X_valid)
acc_xg = accuracy_score(y_valid, pred_xg)
print("The Score for XGBoost is: " + str(acc_xg))
print("The best Score for XGBoost is: " + str(grid_rf.best_score_))






rf_clf = RandomForestClassifier()

parameters_rf = {"n_estimators": [4, 5, 6, 7, 8, 9, 10, 15], "criterion": ["gini", "entropy"], "max_features": ["auto", "sqrt", "log2"], 
                 "max_depth": [2, 3, 5, 10], "min_samples_split": [2, 3, 5, 10]}

grid_rf = GridSearchCV(rf_clf, parameters_rf, scoring=make_scorer(accuracy_score))
grid_rf.fit(X_training, y_training)

rf_clf = grid_rf.best_estimator_

rf_clf.fit(X_training, y_training)
pred_rf = rf_clf.predict(X_valid)
acc_rf = accuracy_score(y_valid, pred_rf)

print("The Score for Random Forest is: " + str(acc_rf))

print("The best Score for Random Forest is: " + str(grid_rf.best_score_))
submission_predictions = rf_clf.predict(X_test)
submission = pd.DataFrame({"PassengerId": test["PassengerId"],
         "Survived": submission_predictions})
 
submission.to_csv("titanicprediction.csv", index=False)
print(submission.shape)