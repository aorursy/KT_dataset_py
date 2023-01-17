import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

sns.set(style="whitegrid")



from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



import optuna

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

import lightgbm as lgbm

from xgboost import XGBClassifier
# check data files



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path    = '/kaggle/input/titanic/'



f_train = pd.read_csv(path + 'train.csv')

f_test  = pd.read_csv(path + 'test.csv')
f_train
f_train.info()
f_train.describe()
def missing_value(df):

    value = (df.isnull().mean())

    return value
# check missing value from train data

missing_value(f_train)
# check missing value from test data

missing_value(f_test)
mask = np.zeros_like(f_train.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



plt.figure(figsize=(15,12))

plt.title("Correlations Among Features",fontsize = 20)

sns.heatmap(f_train.corr(),annot=True, cmap = 'RdBu', mask = mask)
plt.figure(figsize=(10,6))

sns.countplot(data = f_train,

              x = 'Survived',

              palette='RdBu')
temp = f_train.groupby('Survived')['PassengerId'].count().reset_index()

temp.rename(columns={'PassengerId': 'count'}, inplace = True)

temp['Survived'].replace({ 0 : 'Not Survived', 1 : 'Survived'}, inplace = True)



fig = px.pie(temp, values='count', names='Survived',color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
plt.figure(figsize=(10,8))

sns.countplot(data = f_train,

              x = 'Sex',

              hue = 'Survived',

              palette='RdBu')
temp = f_train[f_train.Sex == 'male'].groupby('Survived')['PassengerId'].count().reset_index()

temp.rename(columns={'PassengerId': 'count'}, inplace = True)

temp['Survived'].replace({ 0 : 'Not Survived', 1 : 'Survived'}, inplace = True)



temp1 = f_train[f_train.Sex == 'female'].groupby('Survived')['PassengerId'].count().reset_index()

temp1.rename(columns={'PassengerId': 'count'}, inplace = True)

temp1['Survived'].replace({ 0 : 'Not Survived', 1 : 'Survived'}, inplace = True)



fig = px.pie(temp, values='count', names='Survived',title = 'Survival rate of male',color_discrete_sequence=px.colors.sequential.deep)

fig.show()



fig = px.pie(temp1, values='count', names='Survived',title = 'Survival rate of female',color_discrete_sequence=px.colors.sequential.deep)

fig.show()
plt.figure(figsize=(10,8))

sns.barplot(data = f_train,

              x = 'Pclass',

              y = 'Survived',

              palette='RdBu')
# passengers distribution



temp = f_train.groupby('Pclass')['PassengerId'].count().reset_index()

temp.rename(columns={'PassengerId': 'count'}, inplace = True)



fig = px.pie(temp, values='count', names='Pclass',title = 'Class',color_discrete_sequence=px.colors.sequential.deep)

fig.show()
f_train.Name
# create new feature ( title name )

f_train['Title'] = f_train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
fig = px.histogram(f_train, x="Age", color= 'Survived', marginal="rug", barmode='overlay')

fig.show()
fig = px.box(f_train, x="Title", y="Age", color='Survived',color_discrete_sequence=px.colors.sequential.RdBu)

fig.show()
# Title with missing Age

f_train[f_train.Age.isnull()].Title.value_counts()
f_train.groupby('Title')['Age'].median()
# Fill Missing Age with median Age in Title Group



age_missing = list(f_train[f_train.Age.isnull()].Title.unique())



for i in age_missing:

    median_age = f_train.groupby('Title')['Age'].median()[i]

    f_train.loc[f_train['Age'].isnull() & (f_train['Title'] == i), 'Age'] = median_age
# check missing value

missing_value(f_train)
# merge some title



mapping = {'Mlle': 'Miss', 'Major': 'Rare', 'Col': 'Rare', 'Sir': 'Rare', 'Don': 'Rare', 'Mme': 'Mrs',

          'Jonkheer': 'Rare', 'Lady': 'Rare', 'Capt': 'Rare', 'Countess': 'Rare', 'Ms': 'Miss', 'Dona': 'Rare',

           'Dr': 'Rare', 'Rev': 'Rare'}



f_train.replace({'Title': mapping}, inplace=True)



# group age



bins = [ 0, 12, 30, 60, np.inf]

labels = ['Children', 'Teenager', 'Adult', 'Senior']

f_train['AgeGroup'] = pd.cut(f_train["Age"], bins, labels = labels)

plt.figure(figsize=(20,8))

sns.barplot(data = f_train,

              x = 'Title',

              y = 'Survived',

              palette='RdBu')
plt.figure(figsize=(20,8))

sns.barplot(data = f_train,

              x = 'AgeGroup',

              y = 'Survived',

             hue = 'Sex',

              palette='RdBu')
plt.figure(figsize=(20,8))

sns.barplot(data = f_train,

              x = 'SibSp',

              y = 'Survived',

              palette='RdBu')
plt.figure(figsize=(20,8))

sns.barplot(data = f_train,

              x = 'Parch',

              y = 'Survived',

              palette='RdBu')
# create new features



f_train['Family'] = f_train['SibSp'] + f_train['Parch'] + 1

f_train['TravelAlone']=np.where(f_train['Family']>1, 0, 1)
plt.figure(figsize=(20,8))

sns.barplot(data = f_train,

              x = 'Family',

              y = 'Survived',

              palette='RdBu')
plt.figure(figsize=(20,8))

sns.barplot(data = f_train,

              x = 'TravelAlone',

              y = 'Survived',

              palette='RdBu')
fig = px.scatter(f_train, y = "Age", x= "Fare", color= "Survived")

fig.show()
plt.figure(figsize=(20,8))

sns.barplot(data = f_train,

              x = 'Embarked',

              y = 'Survived',

              hue = 'Pclass',

              palette='RdBu')


# Group Fare



f_train['Fare_Bin'] = pd.qcut(f_train['Fare'], 5)



# Label Encoding



label = LabelEncoder()



f_train['AgeGroup'] = label.fit_transform(f_train['AgeGroup'])

f_train['Fare_Bin'] = label.fit_transform(f_train['Fare_Bin'])

f_train['Title'] = label.fit_transform(f_train['Title'])

f_train['Sex'] = label.fit_transform(f_train['Sex'])



# Drop column



drop_list = ['Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Age','PassengerId']



f_train.drop(drop_list, axis = 1, inplace =True)
mask = np.zeros_like(f_train.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



plt.figure(figsize=(15,12))

plt.title("Correlations Among Features",fontsize = 20)

sns.heatmap(f_train.corr(),annot=True, cmap = 'RdBu', mask = mask)
# do the same thing with test data



# Fill Missing value age and fare



f_test['Title'] = f_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

age_missing = list(f_test[f_test.Age.isnull()].Title.unique())



for i in age_missing:

    median_age = f_test.groupby('Title')['Age'].median()[i]

    f_test.loc[f_test['Age'].isnull() & (f_test['Title'] == i), 'Age'] = median_age

    

f_test.Age.fillna(28, inplace=True)  # because in the test data there is only one person whose title with Ms, so I took it from the train data





missing_value = f_test[(f_test.Pclass == 3) & 

                       (f_test.Embarked == "S") & 

                       (f_test.Sex == "male")].Fare.mean()



f_test.Fare.fillna(missing_value, inplace=True)





# merge some title



mapping = {'Mlle': 'Miss', 'Major': 'Rare', 'Col': 'Rare', 'Sir': 'Rare', 'Don': 'Rare', 'Mme': 'Mrs',

          'Jonkheer': 'Rare', 'Lady': 'Rare', 'Capt': 'Rare', 'Countess': 'Rare', 'Ms': 'Miss', 'Dona': 'Rare',

           'Dr': 'Rare', 'Rev': 'Rare'}



f_test.replace({'Title': mapping}, inplace=True)





# group Age



bins = [ 0, 12, 30, 60, np.inf]

labels = ['Children', 'Teenager', 'Adult', 'Senior']

f_test['AgeGroup'] = pd.cut(f_test["Age"], bins, labels = labels)





# create new feature



f_test['Family'] = f_test['SibSp'] + f_test['Parch']

f_test['TravelAlone']=np.where(f_test['Family']>0, 0, 1)

f_test['Fare_Bin'] = pd.qcut(f_test['Fare'], 5)





# Label Encoding



label = LabelEncoder()



f_test['AgeGroup'] = label.fit_transform(f_test['AgeGroup'])

f_test['Fare_Bin'] = label.fit_transform(f_test['Fare_Bin'])

f_test['Title'] = label.fit_transform(f_test['Title'])

f_test['Sex'] = label.fit_transform(f_test['Sex'])



# Drop column



drop_list = ['Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Age']



f_test.drop(drop_list, axis = 1, inplace =True)
# prepare data for modeling



X = f_train.drop('Survived', axis = 1)

Y = f_train.Survived



X_test = f_test

X_test = X_test.drop('PassengerId',axis = 1)



x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.22, random_state = 0)
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
def basic_model(x_train,y_train,x_val,y_val):



    # Gradient Boosting Classifier

    model = GradientBoostingClassifier()

    model.fit(x_train, y_train)

    y_pred  = model.predict(x_val)

    acc_gbc = round(accuracy_score(y_pred, y_val) * 100, 2)

    print(f'Gradient Boosting Classifier : Score {acc_gbc}')

    

    # Random Forest Classifier

    model = RandomForestClassifier()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_val)

    acc_rfc = round(accuracy_score(y_pred, y_val) * 100, 2)

    print(f'Random Forest Classifier : Score {acc_rfc}')



    # Support Vector Machines

    model = SVC()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_val)

    acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

    print(f'Support Vector Machines : Score {acc_svc}')



    # LightGBM Classifier

    model  = lgbm.LGBMClassifier()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_val)

    acc_lgbm = round(accuracy_score(y_pred, y_val) * 100, 2)

    print(f'LightGBM Classifier : Score {acc_lgbm}')



   # XGB Classifier

    model  = XGBClassifier()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_val)

    acc_xgb = round(accuracy_score(y_pred, y_val) * 100, 2)

    print(f'XGB Classifier : Score {acc_xgb}') 





    return acc_gbc, acc_rfc, acc_svc, acc_lgbm, acc_xgb
# basic model

acc_gbc, acc_rfc, acc_svc, acc_lgbm, acc_xgb = basic_model(x_train, y_train, x_val, y_val)
models_basic = pd.DataFrame({

    'Model': ['Gradient Boosting Classifier','Random Forest Classifier',

              'Support Vector Machines', 'LightGBM Classifier',

              'XGB Classifier'],

              



    'Score Basic Model': [acc_gbc, acc_rfc, acc_svc, acc_lgbm, acc_xgb]

              })



models_basic.sort_values(by='Score Basic Model', ascending=False)
# helper function



class model_objectif(object):

    def __init__(self, models, x, y):

        self.models = models

        self.x = x

        self.y = y



    def __call__(self, trial):

        models, x, y = self.models, self.x, self.y



        classifier_name = models



        if classifier_name == "RFC": 

            

            model = RandomForestClassifier( 

                n_estimators = trial.suggest_int('n_estimators', 10, 100),

                criterion = trial.suggest_categorical('criterion', ['gini', 'entropy']),

                max_depth = trial.suggest_int('max_depth', 1, 6),

                min_samples_split = trial.suggest_int('min_samples_split', 2, 16),

                max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]))

            

            model.fit(x_train, y_train)



        elif classifier_name == "SVM": 

            

            model = SVC( 

                C = trial.suggest_loguniform('C', 0.1, 10),

                kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid']),

                gamma  = trial.suggest_categorical('gamma', ["scale", "auto"]))



            model.fit(x_train, y_train)



        elif classifier_name == "GBC":

            

            model = GradientBoostingClassifier( 

                n_estimators = trial.suggest_int('n_estimators', 10, 100),

                min_samples_leaf =  trial.suggest_int('min_samples_leaf', 1, 10),

                max_depth = trial.suggest_int('max_depth', 1, 6),

                min_samples_split = trial.suggest_int('min_samples_split', 2, 16),

                max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]))



            model.fit(x_train, y_train)



            

        elif classifier_name == "LGBM": 

            

            model = lgbm.LGBMClassifier(

                reg_alpha=trial.suggest_loguniform('reg_alpha', 1e-4, 100.0),

                reg_lambda=trial.suggest_loguniform('reg_lambda', 1e-4, 100.0),

                num_leaves=trial.suggest_int('num_leaves', 10, 40))



            model.fit(x_train, y_train, eval_set = [(x_val, y_val)],

                        early_stopping_rounds=20, verbose=-1)

            

        elif classifier_name == "XGBC":



            model = XGBClassifier(

                xgb_max_depth = trial.suggest_int('max_depth', 1, 6),

                xgb_n_estimators = trial.suggest_int('n_estimators', 10, 100),

                min_child_weight = trial.suggest_int('min_samples_split', 1, 20),

                colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.3, 1.0),

                subsample = trial.suggest_uniform('colsample_bytree', 0.3, 1.0),)



            model.fit(x_train, y_train, eval_set = [(x_val, y_val)],

                        early_stopping_rounds=20, verbose=-1)

              



        y_pred   = model.predict(x_val)

        acc_score = round(accuracy_score(y_pred, y_val) * 100, 2)

        return acc_score



def parameters(study_model):



    print("Number of finished trials: {}".format(len(study_model.trials)))

    print("Best trial:")

    trial = study_model.best_trial

    best_params = study_model.best_params

    print("  Score_value: {}".format(trial.value))

    print("  Params: ")

    for key, value in trial.params.items():

        print("    {}: {}".format(key, value))



    return best_params, trial.value





def visual_study(study_model):



    fig = optuna.visualization.plot_optimization_history(study_model)

    fig.show()

    fig = optuna.visualization.plot_parallel_coordinate(study_model)

    fig.show()

    fig = optuna.visualization.plot_slice(study_model)

    fig.show()

models = 'RFC'

objective = model_objectif(models, X, Y)

study_RFC = optuna.create_study(direction='maximize')

study_RFC.optimize(objective, n_trials=100) # n_trail = 100
best_params_RFC, best_score_RFC = parameters(study_RFC)
visual_study(study_RFC)
models = 'SVM'

objective = model_objectif(models, X, Y)

study_SVM = optuna.create_study(direction='maximize')

study_SVM.optimize(objective, n_trials=100) # n_trail = 100
best_params_SVM, best_score_SVM = parameters(study_SVM)
visual_study(study_SVM)
models = 'GBC'

objective = model_objectif(models, X, Y)

study_GBC = optuna.create_study(direction='maximize')

study_GBC.optimize(objective, n_trials=100) # n_trail = 100
best_params_GBC, best_score_GBC = parameters(study_GBC)
visual_study(study_GBC)
models = 'XGBC'

objective = model_objectif(models, X, Y)

study_XGBC = optuna.create_study(direction='maximize')

study_XGBC.optimize(objective, n_trials=100) # n_trail = 100
best_params_XGBC, best_score_XGBC = parameters(study_XGBC)
visual_study(study_XGBC)
models = 'LGBM'

objective = model_objectif(models, X, Y)

study_LGBM = optuna.create_study(direction='maximize')

study_LGBM.optimize(objective, n_trials=100) # n_trail = 100
best_params_LGBM, best_score_LGBM = parameters(study_LGBM)
visual_study(study_LGBM)
# create dataframe score tuning

models_tuning = pd.DataFrame({

    

    'Model': ['Gradient Boosting Classifier',

              'Random Forest Classifier',

              'Support Vector Machines', 

              'LightGBM Classifier',

              'XGB Classifier'],

              



    'Score Tuning Model': [best_score_GBC, 

                           best_score_RFC, 

                           best_score_SVM, 

                           best_score_LGBM, 

                           best_score_XGBC]

                          })



# merge with score before tuning

model_all = pd.merge(models_basic, models_tuning, on = 'Model')

model_all.sort_values(by='Score Tuning Model', ascending=False, inplace = True)

model_all
def submit_pred(df, test_data):

    

    model_name = df.Model.values[0]

    

    if model_name == 'Random Forest Classifier':

        model  = RandomForestClassifier(**best_params_RFC)

        model.fit(x_train, y_train)

        y_pred   = model.predict(test_data)

        

    if model_name == 'Gradient Boosting Classifier':

        model  = GradientBoostingClassifier(**best_params_GBC)

        model.fit(x_train, y_train)

        y_pred   = model.predict(test_data)

        

    if model_name == 'Support Vector Machines':

        model  = SVC(**best_params_SVM)

        model.fit(x_train, y_train)

        y_pred   = model.predict(test_data)

        

    if model_name == 'LightGBM Classifier':

        model  = lgbm.LGBMClassifier(**best_params_LGBM)

        model.fit(x_train, y_train, eval_set = [(x_val, y_val)],

                  early_stopping_rounds=20, verbose=-1)

        y_pred   = model.predict(test_data)

    

    if model_name == 'XGB Classifier':

        model  = XGBClassifier(**best_params_XGBC)

        model.fit(x_train, y_train, eval_set = [(x_val, y_val)],

                  early_stopping_rounds=20, verbose=-1)

        y_pred   = model.predict(test_data)

        

    print(f'model use {model_name}')

    

    return y_pred
# Sumbit with best Classifier and best Parameters



y_pred = submit_pred(model_all, X_test)



submission = pd.DataFrame({

    "PassengerId": f_test['PassengerId'], 

    "Survived": y_pred

})
submission.head(10)
submission.to_csv('submission.csv', index=False)