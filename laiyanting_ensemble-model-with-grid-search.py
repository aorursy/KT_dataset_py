import os

import re

from collections import Counter

from lightgbm.sklearn import LGBMClassifier

from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier

from sklearn.metrics import confusion_matrix, make_scorer, classification_report, fbeta_score, accuracy_score

from sklearn.preprocessing import MinMaxScaler

from catboost import CatBoostClassifier, Pool, cv

from numpy.random import RandomState

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import warnings

import hyperopt

from hyperopt import tpe, hp

import xgboost as xgb

import datetime

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

warnings.filterwarnings(action='ignore', category=FutureWarning)
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



def grid_search(model, parameters, x_, y_):

    print()

    print("Grid Searching...")

    acc_scorer = make_scorer(fbeta_score, beta=1)

    grid_obj = GridSearchCV(model, parameters, scoring=acc_scorer, n_jobs=-1, cv=5)

    grid_obj = grid_obj.fit(x_, y_.values.ravel())

    return grid_obj.best_estimator_





def classifier(model, grid_search_flag, x_train, y_train, x_valid, y_valid, x_test, cols):

    clf_name = model.__class__.__name__

    parameters = parameter_set(clf_name)

    st_time = datetime.datetime.now()

    print("=" * 30)



    if grid_search_flag:

        clf_name += '-Grid'

        print(clf_name)

        print()

        print("Grid Search Parameters:")

        print(parameters)

        model_ = grid_search(model, parameters, x_train, y_train)

    else:

        model_ = model

        print(clf_name)



    print()

    print("Model Parameters:")

    for k, v in model_.get_params().items():

        print('%30s: %10s' % (k, str(v)))

    print()



    model_.fit(x_train, y_train.values.ravel())

    predict_ = model_.predict(x_valid)

    if clf_name == 'XGBClassifier' or clf_name == 'XGBClassifierGrid':

        predict_ = [value for value in predict_]



    importances_ = model_.feature_importances_[:10]

    indices_ = np.argsort(importances_)[::-1]

    print("Feature ranking:")



    for f_ in range(len(importances_)):

        print("%3d. %25s  (%f)" % (f_ + 1, cols[indices_[f_]], importances_[indices_[f_]]))



    f_score_ = fbeta_score(y_valid, predict_, beta=1, average='binary')

    acc_score = accuracy_score(y_valid, predict_)

    plot_result(y_valid, predict_)



    global ensemble_valid, ensemble_test

    ensemble_valid[clf_name] = predict_

    submit_output = model_.predict(x_test)

    ensemble_test[clf_name] = [value for value in submit_output] if 'XGBClassifier' in clf_name else submit_output

    

    print()

    print('Time Cost:', datetime.datetime.now() - st_time)

    print()

    return [clf_name, f_score_*100, acc_score*100]





def parameter_set(clf_name):

    if clf_name == 'RandomForestClassifier':

        parameters = {

            'n_estimators': [50, 100, 150, 200],

            'criterion': ['entropy', 'gini'],

            'max_depth': [4, 6, 7],

            'max_features': [2, 3, 4],

            "min_samples_split": [2, 3, 10],

            "min_samples_leaf": [1, 3, 10],

             }

    if clf_name == 'DecisionTreeClassifier':

        parameters = {

            'criterion': ['entropy', 'gini'],

            'splitter': ['best', 'random'],

            'max_depth': [4, 5, 6, 7],

             }

    if clf_name == 'GradientBoostingClassifier':

        parameters = {

            "loss": ["deviance", 'exponential'],

            "learning_rate": [0.1, 0.05, 0.01],

            'max_depth': [4, 7],

            'min_samples_leaf': [1,5, 10],

            "criterion": ["friedman_mse",  "mae", 'mse'],

            'n_estimators': [50, 100, 150, 200],

            'max_features': [0.3, 0.1] ,

             }

    if clf_name == 'XGBClassifier':

        parameters = {

            # General parameters

            'booster': ['gbtree'],

            # Parameters of Tree booster

            'max_depth': [4, 5, 6, 7],

            'learning_rate': [.005, 0.1, 0.2],

            'min_child_weight': [4, 5, 6],

            'num_parallel_tree': [1, 2, 3],

            'colsample_bytree': [0.7],

            'n_estimators': [50, 100, 150, 200],

            'objective': ['binary:logistic'],

            'eval_metric': ['auc', 'aucpr', 'map']

        }

    if clf_name == 'AdaBoostClassifier':

        parameters = {

            'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5],

            'n_estimators': [1, 2, 50, 100, 150, 200],

        }        

    if clf_name == 'ExtraTreesClassifier':

        parameters = {

            'criterion': ['entropy', 'gini'],

            "max_features": [1, 3, 10],

            "bootstrap": [False],

            'max_depth': [4, 5, 6, 7],

            'n_estimators': [50, 100, 150, 200],

        }

    if clf_name == 'LGBMClassifier':

        parameters = {

            'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5],

            'max_depth': [4, 5, 6, 7],

            'n_estimators': [50, 100, 150, 200],

        }        

        

    return parameters





def get_data():

    # Load in the train and test datasets

    train = pd.read_csv('../input/titanic/train.csv')

    test = pd.read_csv('../input/titanic/test.csv')

    return train, test





def clean_data(dt):

    # Create categorical values for Ticket

    Ticket = []

    for i in list(dt.Ticket):

        if not i.isdigit() :

            Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix

        else:

            Ticket.append("X")

    dt["Ticket"] = Ticket

    dt = pd.get_dummies(dt, columns = ["Ticket"], prefix="Tik")

    

    # Generate sex-pclass cross variable

    dt[ 'Sex_Pclass' ] = np.nan

    dt.loc[ (dt.Sex=='female') & (dt.Pclass==1), 'Sex_Pclass' ] = 2

    dt.loc[ (dt.Sex=='female') & (dt.Pclass==2), 'Sex_Pclass' ] = 3

    dt.loc[ (dt.Sex=='female') & (dt.Pclass==3), 'Sex_Pclass' ] = 3

    dt.loc[ (dt.Sex=='male') & (dt.Pclass==1), 'Sex_Pclass' ] = 1

    dt.loc[ (dt.Sex=='male') & (dt.Pclass==2), 'Sex_Pclass' ] = 1

    dt.loc[ (dt.Sex=='male') & (dt.Pclass==3), 'Sex_Pclass' ] = 2   

    

    # Create categorical values for Pclass

    dt["Pclass"] = dt["Pclass"].astype("category")

    dt = pd.get_dummies(dt, columns = ["Pclass"], prefix="Pcl")



    # Create categorical values for Cabin

    dt["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in dt['Cabin'] ])

    datdtaset = pd.get_dummies(dt, columns = ["Cabin"], prefix="Cab")





    # Create a family size descriptor from SibSp and Parch

    dt["Fsize"] = dt["SibSp"] + dt["Parch"] + 1

    # Create new feature of family size

    dt['Single'] = dt['Fsize'].map(lambda s: 1 if s == 1 else 0)

    dt['SmallF'] = dt['Fsize'].map(lambda s: 1 if  s == 2  else 0)

    dt['MedF'] = dt['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

    dt['LargeF'] = dt['Fsize'].map(lambda s: 1 if s >= 5 else 0)

    

 

    

    # Fill Embarked nan values of dataset set with 'S' most frequent value

    dt['Embarked'] = dt['Embarked'].fillna('S')

    dt['Embarked'] = dt['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    # Create categorical values for Embarked

    dt = pd.get_dummies(dt, columns = ["Embarked"], prefix="Emb")



    dt['Name_length'] = dt['Name'].apply(len)       

    dt_title = [i.split(",")[1].split(".")[0].strip() for i in dt["Name"]]

    dt["Title"] = pd.Series(dt_title)

    # Convert to categorical values Title 

    dt["Title"] = dt["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dt["Title"] = dt["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})

    dt["Title"] = dt["Title"].astype(int)



    # convert Sex into categorical value 0 for male and 1 for female

    dt['Sex'] = dt['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



    #Fill Fare missing values with the median value

    dt["Fare"] = dt["Fare"].fillna(dt["Fare"].median())

    # Apply log to Fare to reduce skewness distribution

    dt["Fare"] = dt["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

#         dt.loc[ dt['Fare'] <= 7.91, 'Fare'] = 0

#         dt.loc[(dt['Fare'] > 7.91) & (dt['Fare'] <= 14.454), 'Fare'] = 1

#         dt.loc[(dt['Fare'] > 14.454) & (dt['Fare'] <= 31), 'Fare']   = 2

#         dt.loc[ dt['Fare'] > 31, 'Fare']  = 3

#         dt['Fare'] = dt['Fare'].astype(int)



    age_avg = dt['Age'].mean()

    age_std = dt['Age'].std()

    age_null_count = dt['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dt['Age'][np.isnan(dt['Age'])] = age_null_random_list

    dt['Age'] = dt['Age'].astype(int)

    dt.loc[ dt['Age'] <= 16, 'Age'] = 0

    dt.loc[(dt['Age'] > 16) & (dt['Age'] <= 32), 'Age'] = 1

    dt.loc[(dt['Age'] > 32) & (dt['Age'] <= 48), 'Age'] = 2

    dt.loc[(dt['Age'] > 48) & (dt['Age'] <= 64), 'Age'] = 3

    dt.loc[ dt['Age'] > 64, 'Age'] = 4 

    return dt





def plot_result(y_t, y_p):

    print()

    print("Confusion Matrix:")

    print(confusion_matrix(y_t, y_p))

    print()

    print("Classification Report:")

    print(classification_report(y_t, y_p))

    pass





def catboost_classifier(x_train, y_train, x_valid, y_valid, x_test, cols, hyper_tune):

    hyper_algo = tpe.suggest

    d_train = Pool(x_train, y_train)

    d_val = Pool(x_valid, y_valid)



    def get_catboost_params(space_):

        params = dict()

        params['learning_rate'] = space_['learning_rate']

        params['depth'] = int(space_['depth'])

        params['l2_leaf_reg'] = space_['l2_leaf_reg']

        params['rsm'] = space_['rsm']

        return params



    def hyperopt_objective(space_):

        params = get_catboost_params(space_)

        sorted_params = sorted(space.items(), key=lambda z: z[0])

        params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])

        print('Params: {}'.format(params_str))



        model_ = CatBoostClassifier(iterations=100,

                                    learning_rate=params['learning_rate'],

                                    depth=int(params['depth']),

                                    loss_function='Logloss',

                                    use_best_model=True,

                                    eval_metric='AUC',

                                    l2_leaf_reg=params['l2_leaf_reg'],

                                    random_seed=5566,

                                    verbose=False,

                                    )

        cv_ = cv(d_train, model_.get_params())

        best_accuracy = np.max(cv_['test-AUC-mean'])

        return 1 - best_accuracy



    if hyper_tune:

        space = {

            'depth': hp.quniform("depth", 4, 7, 1),

            'rsm': hp.uniform('rsm', 0.75, 1.0),

            'learning_rate': hp.loguniform('learning_rate', -3.0, -0.7),

            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 1, 10),

        }

        trials = hyperopt.Trials()

        best = hyperopt.fmin(

            hyperopt_objective,

            space=space,

            algo=hyper_algo,

            max_evals=50,

            trials=trials,

            rstate=RandomState(5566),

            verbose=False,

            show_progressbar=False

        )

        print('-' * 50)

        print('The best params:')

        print(best)

        print('\n\n')



        model = CatBoostClassifier(

            l2_leaf_reg=int(best['l2_leaf_reg']),

            learning_rate=best['learning_rate'],

            depth=best['depth'],

            iterations=100,

            eval_metric='AUC',

            random_seed=42,

            loss_function='Logloss',

            verbose=0

        )

    else:

        model = CatBoostClassifier(

            l2_leaf_reg=6,

            learning_rate=0.24,

            depth=8,

            iterations=100,

            eval_metric='AUC',

            random_seed=42,

            loss_function='Logloss',

            verbose=0

        )



    cv_data = cv(pool=d_train,

                 params=model.get_params(),

                 nfold=5,

                 verbose=False

                 )



    model.fit(x_train, y_train)



    print('Best validation AUC score: {:.2f}±{:.2f} on step {}'.format(

        np.max(cv_data['test-AUC-mean']),

        cv_data['test-AUC-std'][np.argmax(cv_data['test-AUC-mean'])],

        np.argmax(cv_data['test-AUC-mean'])

    ))



    predict_ = model.predict(x_valid)



    f_score_ = fbeta_score(y_valid, predict_, beta=1, average='binary')

    

    acc_score = accuracy_score(y_valid, predict_)

    

    grid_flag = '-Grid' if hyper_tune else ''

 

    global log

    log = log.append(pd.DataFrame([['Catboost' + grid_flag, f_score_ * 100, acc_score*100]], columns=log_cols))



    plot_result(y_valid, predict_)

    global ensemble_test, ensemble_valid

    ensemble_valid['Catboost' + grid_flag] = predict_

    ensemble_test['Catboost' + grid_flag] = model.predict(x_test)

    feature_importances = model.get_feature_importance(Pool(x_train, y_train))

    feature_names = cols

    for score, name in sorted(zip(feature_importances, feature_names), reverse=True):

        if score > 0.1:

            print("%25s  (%f)" % (name, score))



            

def detect_outliers(df, n, outlier_step, features):

    outlier_indices = []

    for col in features:

        print()

        print(col)

        print('-' * 30)

        # 1st quartile (25%)

        Q1 = np.nanpercentile(df[col], 25)

        print('Q1:', Q1)

        # 3rd quartile (75%)

        Q3 = np.nanpercentile(df[col],75)

        print('Q3:', Q3)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        print('IQR:', IQR)

        outlier_step = outlier_step * IQR

        print('outlier_step:', outlier_step)

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        print(outlier_list_col)

        outlier_indices.extend(outlier_list_col)

        print('-' * 30)

    

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    return multiple_outliers
# 讀檔

train_df, test_df = get_data()



# Store our passenger ID for easy access

PassengerId = test_df['PassengerId']
def Missing_Counts( Data ) : 

    missing = Data.isnull().sum()  # 計算欄位中缺漏值的數量 

    missing = missing[ missing>0 ]

    missing.sort_values( inplace=True ) 

    

    Missing_Count = pd.DataFrame( { 'ColumnName':missing.index, 'MissingCount':missing.values } )  # Convert Series to DataFrame

    Missing_Count[ 'Percentage(%)' ] = Missing_Count['MissingCount'].apply( lambda x:round(x/Data.shape[0]*100,2) )

    return  Missing_Count
print( 'train :' )

display( Missing_Counts(train_df) )



print( 'test :' )

display( Missing_Counts(test_df) )
# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(train_df, 2, 1.3,["Age","SibSp","Parch","Fare"])
# Drop outliers

train_df = train_df.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)



train_len = len(train_df)

dataset =  pd.concat(objs=[train_df, test_df], axis=0).reset_index(drop=True)

dataset = dataset.fillna(np.nan)

Survived_Counts = dataset['Survived'].value_counts().reset_index()

Survived_Counts.columns = ['Survived','Counts']

Survived_Counts
plt.figure( figsize=(10,5) )

dataset['Survived'].value_counts().plot( kind='pie', colors=['lightcoral','skyblue'], autopct='%1.2f%%' )

plt.title( 'Survival' )  # 圖標題

plt.ylabel( '' )

plt.show()
# Survied 與其他欄位間的相關係數

Corr_Matrix = dataset.corr()  # 計算相關係數

Corr = Corr_Matrix.loc['Survived',:].sort_values()[:-1]

Corr = pd.DataFrame({ 'Survived':Corr })

Corr
selected_cols = ['Sex','Pclass','Embarked','SibSp','Parch']



plt.figure( figsize=(10,len(selected_cols)*5) )

gs = gridspec.GridSpec(len(selected_cols),1)    

for i, col in enumerate( dataset[selected_cols] ) :        

    ax = plt.subplot( gs[i] )

    sns.countplot( dataset[col], hue=dataset.Survived, palette=['lightcoral','skyblue'] )

    ax.set_yticklabels([])

    ax.set_ylabel( 'Counts' )

    ax.legend( loc=1 )   # upper right:1 ; upper left:2

    for p in ax.patches:

        ax.annotate( '{:,}'.format(p.get_height()), (p.get_x(), p.get_height()+1.5) )

plt.show()
for col in selected_cols:

    l = ['Survived']

    l.append(col) 

    Survival_Rate = dataset[l].groupby(by=col).mean().round(4).reset_index()

    Survival_Rate.columns = [col,'Survival Rate(%)']

    Survival_Rate['Survival Rate(%)'] = Survival_Rate['Survival Rate(%)'].map( lambda x:x*100 )

    display( Survival_Rate )
# Check for Null values

dataset.isnull().sum()
# 清理資料

dataset = clean_data(dataset)
train = dataset[:train_len].copy()

test = dataset[train_len:].copy()

test.drop(labels=["Survived"],axis = 1,inplace=True)
drop_elements = ['PassengerId', 'Name', 'Cabin']

train = train.drop(drop_elements, axis = 1)

test  = test.drop(drop_elements, axis = 1)

## Separate train features and label 



train["Survived"] = train["Survived"].astype(int)

y_train = train["Survived"]

x_train = train.drop(labels = ["Survived"],axis = 1)

x_test = test.copy()
cols = train.columns
cols
# Cross validate model with Kfold stratified cross val

kfold = StratifiedKFold(n_splits=10)
# Split data

x_train, x_valid, y_train, y_valid = train_test_split(

    x_train,

    y_train,

    test_size=0.10,

    random_state=5566

)



# 建立ensemble空間

ensemble_test = pd.DataFrame()

ensemble_valid = pd.DataFrame()



# 預設分類器種類

classifiers = [

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    GradientBoostingClassifier(),

    xgb.XGBClassifier(),    

    AdaBoostClassifier(),

    ExtraTreesClassifier(),

    LGBMClassifier()

]



# 建立分類器效度表

log_cols = ["Classifier", "F1", 'Accuracy']

log = pd.DataFrame([], columns=log_cols)



# flags

gs = 1

# Catboost

catboost_classifier(x_train, y_train, x_valid, y_valid, x_test, cols, hyper_tune=0)

catboost_classifier(x_train, y_train, x_valid, y_valid, x_test, cols, hyper_tune=1)

for clf in classifiers:

    # 先跑一次預設參數分類器

    log_entry = classifier(clf, 0, x_train, y_train, x_valid, y_valid, x_test, cols)

    log = log.append(pd.DataFrame([log_entry], columns=log_cols))

    if gs:

#         gs設1則開始調參

        log_entry = classifier(clf, gs, x_train, y_train, x_valid, y_valid, x_test, cols)

        log = log.append(pd.DataFrame([log_entry], columns=log_cols))

    print()





ensemble_model = RandomForestClassifier()

ensemble_model.fit(ensemble_valid, y_valid.values.ravel())

predictions = ensemble_model.predict(ensemble_valid)



f_score = fbeta_score(y_valid, predictions, beta=1, average='binary')

acc_score = accuracy_score(y_valid, predictions)



log = log.append(pd.DataFrame([['Ensemble', f_score*100, acc_score*100]], columns=log_cols))



importances = ensemble_model.feature_importances_

indices = np.argsort(importances)[::-1]

cols = ensemble_valid.columns

print("Feature ranking:")

for f in range(len(importances)):

    print("%3d. %25s  (%f)" % (f + 1, cols[indices[f]], importances[indices[f]]))



plot_result(y_valid, predictions)



# submission



submit_predictor = ensemble_model.predict(ensemble_test)

submit = pd.DataFrame({'PassengerId': PassengerId, 'Survived': submit_predictor})

submit.to_csv('Submission.csv', index=False, header=True)

# Visualize

sns.set_color_codes("muted")

g = sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.xlabel('Accuracy Score')

plt.title('Classifier\'s Accuracy Score')

for p in g.patches:

    x = p.get_x() + p.get_width() + .3

    y = p.get_y() + p.get_height()/2 + .1

    g.annotate("%.2f" % (p.get_width()), (x, y))



plt.savefig("output.png")

plt.show()
# # submission

# submit_predictor = ensemble_model.predict(ensemble_test)

# submit = pd.DataFrame({'PassengerId': PassengerId, 'Survived': submit_predictor})

# submit.to_csv('Submission.csv', index=False, header=True)