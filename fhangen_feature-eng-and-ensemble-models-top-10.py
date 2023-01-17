# DATA MANAGMENT AND VISUALIZATION
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# MODELS
from sklearn import mixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost.sklearn import XGBClassifier

# Peace and Quiet (Ignorance is Bliss!)
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler

def my_pipeline(train, test):
    
    train['Title'] = train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    
    title_class = {'Capt.':0, 'Don.':0, 'Jonkheer.':0, 'Rev':0,
                      'Mr.':1, 'Dr.':2 , 'Col.':2, 'Major.':2,
                      'Mrs.':3,'Miss.':3,'Master':3,
                      'the':4,'Sir.':4, 'Ms.':4,'Mme.':4,'Mlle.':4, 'Lady.':4}
        
    train['title_class'] = train['Title'].apply(lambda x: title_class.get(x))
        
    mean_title_survived = train[['title_class', 'Survived']].groupby(['title_class'], as_index=False).mean()

    for df in [train, test]:        
        # Process Name into Title, len_name, Class_from_name, len_lastname, num_names,men_surv_title
        df['Title'] = df['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        df['len_name'] = df['Name'].apply(lambda x: len(x))
        df['len_lastname'] = df['Name'].apply(lambda x: len(x.split(',')[0]))
        df['num_names'] = df['Name'].apply(lambda x: len(x.split()))

        
        title_class = {'Capt.':0, 'Don.':0, 'Jonkheer.':0, 'Rev.':0,
                      'Mr.':1, 'Dr.':2 , 'Col.':2, 'Major.':2,
                      'Mrs.':3,'Miss.':3,'Master.':3,
                      'the':4,'Sir.':4, 'Ms.':4,'Mme.':4,'Mlle.':4, 'Lady.':4}
        
        df['title_class'] = df['Title'].apply(lambda x: title_class.get(x))
        
        for i in range(5):
            df.loc[df['title_class'] == i, 'mean_surv_title'] = \
            mean_title_survived[mean_title_survived['title_class'] == i]['Survived'].values[0]
        
        # Process Ticket into Tick_letter, Ticket_len
        df['Tick_letter'] = df['Ticket'].apply(lambda x: str(str(x[0])))
        df['Ticket_len'] = df['Ticket'].apply(lambda x: len(x))
        
        # Age into Age_missing
        df['Age_missing'] = df['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
        ages = df.groupby(['Title', 'Pclass'])['Age']
        df['Age'] = ages.transform(lambda x: x.fillna(x.mean()))
        
        # Embarked fill na
        df['Embarked'] = df['Embarked'].fillna('S')
        
        # SibSp + Parch = fam_size
        df['fam_size'] = df['SibSp'] + df['Parch']
        
        # Cabin into Cabin_missing, Cabin_letter, Cabin_num
        
        df['Cabin_Letter'] = df['Cabin'].apply(lambda x: str(x)[0])
        
        df['Cabin_missing'] = df['Cabin'].apply(lambda x: 1 if pd.isnull(x) else 0)
        
        df['Cabin_num_temp'] = df['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:] if not pd.isnull(x) else np.NaN)
        df['Cabin_num_temp'].replace('', np.NaN, inplace = True)
        df['Cabin_num_temp'].replace(np.NaN, 1000, inplace = True)
        df.loc[df['Cabin_num_temp'].apply(lambda x: len(str(x))) == 4, 'Cabin_num'] = np.NaN
        df.loc[df['Cabin_num_temp'].apply(lambda x: len(str(x))) == 3, 'Cabin_num'] = df['Cabin_num_temp'].apply(lambda x: int(str(x)[:2]))
        df.loc[df['Cabin_num_temp'].apply(lambda x: len(str(x))) == 2, 'Cabin_num'] = df['Cabin_num_temp'].apply(lambda x: int(str(x)[:1]))
        df.loc[df['Cabin_num_temp'].apply(lambda x: len(str(x))) == 1, 'Cabin_num'] = 0
        
        # Interaction Terms Age*Male, Age*Female, Age*Fare, Age*len_name, Age*len_lastname, Age*num_names,
        # Age*fam_size, Age*mean_surv_title, Fare*len_name, Fare*len_lastname, Fare*num_names, Fare*fam_size, Fare*Male, Fare*Female,
        # fam_size*len_name, fam_size*len_lastname, fam_size*num_names
        # len_name*len_lastname, len_name*num_names, len_name*mean_surv_title
        
        df.loc[df['Sex'] == 'male', 'Age*Male'] = df['Age']
        df.loc[df['Sex'] == 'female', 'Age*Male'] = 0
        df.loc[df['Sex'] == 'male', 'Age*female'] = 0
        df.loc[df['Sex'] == 'female', 'Age*female'] = df['Age']
        df['Age*Fare'] = df['Age'] * df['Fare']
        df['Age*len_name'] = df['Age'] * df['len_name']
        df['Age*len_lastname'] = df['Age'] * df['len_lastname']
        df['Age*num_names'] = df['Age'] * df['num_names']
        df['Age*fam_size'] = df['Age'] * df['fam_size']
        df['Age*mean_surv_title'] = df['Age'] * df['mean_surv_title']
        df['Fare*len_name'] = df['Fare'] * df['len_name']
        df['Fare*len_lastname'] = df['Fare'] * df['len_lastname']
        df['Fare*num_names'] = df['Fare'] * df['num_names']
        df['Fare*fam_size'] = df['Fare'] * df['fam_size']
        df.loc[df['Sex'] == 'male', 'Fare*Male'] = df['Fare']
        df.loc[df['Sex'] == 'female', 'Fare*Male'] = 0
        df.loc[df['Sex'] == 'male', 'Fare*female'] = 0
        df.loc[df['Sex'] == 'female', 'Fare*female'] = df['Fare']
        df['fam_size*len_name'] = df['fam_size'] * df['len_name']
        df['fam_size*len_lastname'] = df['fam_size'] * df['len_lastname']
        df['fam_size*num_names'] = df['fam_size'] * df['num_names']
        df['len_name*len_lastname'] = df['len_name'] * df['len_lastname']
        df['len_name*num_names'] = df['len_name'] * df['num_names']
        df['len_name*mean_surv_title'] = df['len_name'] * df['mean_surv_title']
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

my_pipeline(train, test)
cat_attributes = ['Pclass','Sex','Embarked','Cabin_num','Tick_letter','title_class']

num_attributes = ['Age','len_name','len_lastname','Fare',
                  'num_names','fam_size','Ticket_len','mean_surv_title',
                  'Age*Male','Age*female', 'Age*Fare', 'Age*len_name', 'Age*len_lastname',
                 'Age*num_names','Age*fam_size', 'Age*mean_surv_title', 'Fare*len_name',
                 'Fare*len_lastname','Fare*num_names', 'Fare*fam_size', 'Fare*Male',
                 'Fare*female', 'fam_size*len_name','fam_size*len_lastname', 'fam_size*num_names',
                 'len_name*len_lastname', 'len_name*num_names', 'len_name*mean_surv_title']

new_num_attributes = ['Age','len_name','len_lastname','Fare',
                  'num_names','fam_size','Ticket_len','mean_surv_title',
                  'Age*Male','Age*female', 'Age*Fare', 'Age*len_name', 'Age*len_lastname',
                 'Age*num_names','Age*fam_size', 'Age*mean_surv_title', 'Fare*len_name',
                 'Fare*len_lastname','Fare*num_names', 'Fare*fam_size', 'Fare*Male',
                 'Fare*female', 'fam_size*len_name','fam_size*len_lastname', 'fam_size*num_names',
                 'len_name*len_lastname', 'len_name*num_names', 'len_name*mean_surv_title']
# Dummy code categorical attributes.
for column in cat_attributes:

    train = pd.concat((train, pd.get_dummies(train[column], prefix = column)), axis = 1)
    test = pd.concat((test, pd.get_dummies(test[column], prefix = column)), axis = 1)
    
# Create Polynomial features
for column in num_attributes:

    train[column +'* 2'] = train[column] * 2
    train[column +'^ 2'] = train[column] ** 2
    train[column +'^ 3'] = train[column] ** 3
    
    test[column +'* 2'] = test[column] * 2
    test[column +'^ 2'] = test[column] ** 2
    test[column +'^ 3'] = test[column] ** 3
    
    new_num_attributes.append(column +'* 2')
    new_num_attributes.append(column +'^ 2')
    new_num_attributes.append(column +'^ 3')

# Scale all numerical features
for column in new_num_attributes:
    
    test[column] = test[column].fillna(test[column].median())
    
    scaler = StandardScaler()
    scaler = scaler.fit(train[column].values.reshape(-1,1))
    train[column + '_scaled'] = scaler.transform(train[column].values.reshape(-1,1))
    test[column + '_scaled'] = scaler.transform(test[column].values.reshape(-1,1))
    
# Create train_feat and train_target and test_feat, dropping all the categorical variables and non-scaled numerical variables
train_feat = train.drop(columns = ['PassengerId','Survived','Pclass','Name','Sex','SibSp',
                                  'Parch','Ticket','Cabin','Embarked','Title','Cabin_num',
                                   'Cabin_Letter', 'Cabin_num_temp', 'Sex_male',
                                   'Tick_letter','title_class'] + new_num_attributes)
                                   

train_target = train['Survived']

test_feat = test.drop(columns = ['PassengerId','Pclass','Name','Sex','SibSp',
                                  'Parch','Ticket','Cabin','Embarked','Title','Cabin_num',
                                   'Cabin_Letter', 'Cabin_num_temp', 'Sex_male',
                                   'Tick_letter','title_class'] + new_num_attributes)
# When we dummy coded the train and test sets, there are some differences in the binary variables. This rectifies it.
test_feat['Tick_letter_5'] = 0
test_feat['Tick_letter_8'] = 0
test_feat['Cabin_num_12.0'] = 0
test_feat['Cabin_num_14.0'] = 0

# There were also some naming differences between the test and train set, which this fixes
test_feat = test_feat.rename(columns={'title_class_1.0':'title_class_1','title_class_2.0':'title_class_2',
                          'title_class_3.0':'title_class_3','title_class_4.0':'title_class_4',
                                     'title_class_0.0':'title_class_0'})

# XGBoost needs the train and test data to be in the same order:
test_feat = test_feat[train_feat.columns]

print(len(train_feat.columns), len(test_feat.columns))
for df in [train_feat, test_feat]:
    
    #Hand Coded Meaningful Differences Features (All Binary)
    
    df.loc[train['len_name'] < 100, 'len_name_20_24'] = 0
    df.loc[train['len_name'] > 19, 'len_name_20_24'] = 1
    df.loc[train['len_name'] > 24, 'len_name_20_24'] = 0

    df.loc[train['len_name'] < 100, 'len_name_17_19'] = 0
    df.loc[train['len_name'] > 16, 'len_name_17_19'] = 1
    df.loc[train['len_name'] > 19, 'len_name_17_19'] = 0

    df.loc[train['num_names'] == 3, 'num_names_3'] = 1
    df.loc[train['num_names'] != 3, 'num_names_3'] = 0

    df.loc[train['num_names'] == 4, 'num_names_4'] = 1
    df.loc[train['num_names'] != 4, 'num_names_4'] = 0

    df.loc[train['mean_surv_title'] < 2, 'mean_surv_title_1'] = 0
    df.loc[train['mean_surv_title'] > .1, 'mean_surv_title_1'] = 1
    df.loc[train['mean_surv_title'] > .2, 'mean_surv_title_1'] = 0

    df.loc[train['mean_surv_title'] < 2, 'mean_surv_title_2'] = 0
    df.loc[train['mean_surv_title'] > .5, 'mean_surv_title_2'] = 1
    df.loc[train['mean_surv_title'] > .8, 'mean_surv_title_2'] = 0

    df.loc[train['Ticket_len'] == 6, 'Tick_len_6'] = 1
    df.loc[train['Ticket_len'] != 6, 'Tick_len_6'] = 0

    df.loc[train['Cabin_Letter'] == 'n', 'Cabin_n'] = 1
    df.loc[train['Cabin_Letter'] != 'n', 'Cabin_n'] = 0    

print(len(train_feat.columns), len(test_feat.columns))
explore_data = train_feat.copy()
explore_data['Survived'] = train_target
list(explore_data.columns)
explore_data['Survived'].hist()
fig, ax = plt.subplots(figsize=(10, 5)) 
ax = sns.barplot(x="len_name_20_24", y="Survived", hue='Survived', data=explore_data, estimator=lambda x: len(x) / len(train) * 100)
ax.set(ylabel="Percent")
facet = sns.FacetGrid(data = explore_data, hue = "Survived", legend_out=True, size = 5)
facet = facet.map(sns.kdeplot, "Age*Male_scaled")
facet.add_legend()
from sklearn.model_selection import cross_val_score

def model_scores(model, train_features, train_targets):
    model_ = model()
    model_.fit(train_features, train_targets)
    scores = cross_val_score(model_, train_features, train_targets, scoring = 'accuracy', cv=10)
    print(np.mean(scores))
    return model_
from sklearn.tree import DecisionTreeClassifier

dt = model_scores(DecisionTreeClassifier,train_feat, train_target)
from sklearn.ensemble import RandomForestClassifier

rf = model_scores(RandomForestClassifier,train_feat, train_target)
from sklearn.svm import SVC

svc = model_scores(SVC,train_feat, train_target)
from sklearn.neighbors import KNeighborsClassifier

knn = model_scores(KNeighborsClassifier, train_feat, train_target)
from xgboost.sklearn import XGBClassifier

xgb = model_scores(XGBClassifier, train_feat, train_target)
from sklearn.model_selection import GridSearchCV
import scikitplot as skplt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, learning_curve
kfold = StratifiedKFold(n_splits=10)
from sklearn.model_selection import StratifiedKFold, learning_curve

def test_model(model, param_grid, train_feat, train_target):
    grid_search = GridSearchCV(model(), param_grid = param_grid, cv=5, 
                                   scoring='accuracy', n_jobs = -1)
    grid_search.fit(train_feat, train_target)
    best_model = grid_search.best_estimator_

    y_true = train_target
    y_probas = best_model.predict_proba(train_feat)
    skplt.metrics.plot_roc_curve(y_true, y_probas)
    plt.show()

    prob = pd.DataFrame(y_probas)

    predicts = pd.DataFrame(best_model.predict(train_feat))
    predicts['Actual'] = train_target
    predicts.loc[(predicts['Actual'] == 1) & (predicts[0]==1), 'True_Positives'] = 1
    predicts.loc[(predicts['Actual'] == 0) & (predicts[0]==1), 'False_Positives'] = 1
    predicts.loc[(predicts['Actual'] == 0) & (predicts[0]==0), 'True_Negatives'] = 1
    predicts.loc[(predicts['Actual'] == 1) & (predicts[0]==0), 'False_Negatives'] = 1
    
    try:
        true_pos = predicts['True_Positives'].value_counts()[1]
        false_pos = predicts['False_Positives'].value_counts()[1]
        true_neg = predicts['True_Negatives'].value_counts()[1]
        false_neg = predicts['False_Negatives'].value_counts()[1]
        
        print('True Positive Rate (Sensitivity) = ',str(true_pos/(true_pos + false_neg)))
        print('False Positive Rate =',str(false_pos/(false_pos + true_neg)))
        print('True Negative Rate (Specificity) = ',str(true_neg/(false_pos + true_neg)))
        print('Best AUC = ',roc_auc_score(y_true, prob[1]))
        print('Best Accuracy = ', grid_search.best_score_)
        print('Parameters = ', best_model)
    except KeyError:
        print('True Positive Rate (Sensitivity) = 100')
        print('False Positive Rate = 0')
        print('True Negative Rate (Specificity) = 100')
        print('Best AUC = ',roc_auc_score(y_true, prob[1]))
        print('Best Accuracy = ', grid_search.best_score_)
        print('Parameters = ', best_model)

    
    return grid_search.best_estimator_



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
# Big thanks to Yassine Ghouzman for this learning plot code @https://www.kaggle.com/yassineghouzam
# Code source: https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
param_grid_xgb = {'max_depth':[3],'reg_alpha':[.07], 'gamma':[10],
                    'scale_pos_weight':[1], 'n_estimators':[100],
                 'learning_rate':[.1],'subsample':[.8,1]}

xgb_best = test_model(XGBClassifier, param_grid_xgb, train_feat, train_target)
g = plot_learning_curve(xgb_best,"XGB Learning curves",train_feat,train_target,cv=kfold)
param_grid_rf = [{'n_estimators':[100], 'max_features':[4], 'min_samples_leaf':[1],
              'max_depth':[3], 'min_samples_split':[2,10]}]

rf_best = test_model(RandomForestClassifier, param_grid_rf, train_feat, train_target)
param_grid_rf_2 = [{'n_estimators':[100], 'max_features':[2,3,4,20], 'min_samples_leaf':[1,2],
              'max_depth':[8,3], 'min_samples_split':[2,10]}]

rf_best_2 = test_model(RandomForestClassifier, param_grid_rf_2, train_feat, train_target)
g = plot_learning_curve(rf_best,"RF Learning curves",train_feat,train_target,cv=kfold)
g = plot_learning_curve(rf_best_2,"RF Learning curves",train_feat,train_target,cv=kfold)
param_grid_knn = [{'n_neighbors':[20,21,22,23,24,25,26,27,28,29],'weights':['uniform','distance'],
               'algorithm':['auto'],'leaf_size':[10]}]

knn_best = test_model(KNeighborsClassifier, param_grid_knn, train_feat, train_target)
g = plot_learning_curve(knn_best,"RF Learning curves",train_feat,train_target,cv=kfold)
param_grid_ada = [{'n_estimators':[50,100],'learning_rate':[.01],
                 'base_estimator':[DecisionTreeClassifier()]}]

ada_best = test_model(AdaBoostClassifier, param_grid_ada, train_feat, train_target)
g = plot_learning_curve(ada_best,"Adaboost Learning curves",train_feat,train_target,cv=kfold)
param_grid_svc = [{'kernel':['rbf','poly',],'probability':[True], 'gamma':['auto'],'C':[1]}]

svc_best = test_model(SVC, param_grid_svc, train_feat, train_target)
g = plot_learning_curve(svc_best,"SVC learning curve",train_feat,train_target,cv=kfold)
features = train_feat.columns
importances = xgb_best.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(20,100))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()
features = train_feat.columns
importances = rf_best_2.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(20,100))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')
plt.show()
pred = xgb_best.predict(test_feat)
predictions_voting = pd.DataFrame(pred, columns=['Survived'])
test_file = pd.read_csv('../input/test.csv')
predictions_voting = pd.concat((test_file.iloc[:, 0], predictions_voting), axis = 1)
predictions_voting['Survived'] = predictions_voting['Survived'].astype('int32')
predictions_voting.to_csv('TitanicSubmission6-15_v1_XGB.csv', sep=",", index = False)