import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import randint as randint

from scipy.stats import uniform 

import lightgbm as lgbm



from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from sklearn.metrics import accuracy_score
def load_data(data_path, **kwargs):

    return pd.read_csv(data_path, header=0, delimiter=kwargs['delimiter'])

 

def clean_titanic_dataset(base_data_org, new_data_org, create_dummy=False):

    """

    Cleaning Titanic Dataset.

    

    base_data: is the training data

    new_data: is any validation or test data

    For cleaning the base_data just use base_data,base_data

    P.S. I think using pipeline should be a better way to implement chain of cleaning/transforming data

    I will consider it in the next versions. 

    There is no guarantee that new data set has all values, as a result we need to go through all

    columns and have a strategy for missing values.

    """

    

    base_data = base_data_org.copy(deep=True)

    new_data = new_data_org.copy(deep=True)

    

    # Input values should be pandas dataframe:

    if (type(base_data) != type(pd.DataFrame())) or (type(new_data) != type(pd.DataFrame())):

        print("Input data should be pandas dataframe.")

        return None

        

    # PassengerId is the key component. If data does not have PassengerId attribute

    # or it has missing value, the function should terminate gracefully and give

    # let user know the problem.

    if ('PassengerId' not in base_data) or ('PassengerId' not in new_data):

        print("PassengerId attribute is missing.")

        return None

    

    if (base_data.PassengerId.isnull().sum() != 0) or (new_data.PassengerId.isnull().sum() != 0):

        print("PassengerId cannot have missing value.")

        return None



    # Survived attribute. (It should be Ok for now)

    

    # Pclass 

    # There is not any missing value for Pclass, however, there is no guarantee 

    # for upcoming datasets.

    

    for item in new_data.loc[new_data.Pclass.isnull(),:].iterrows():

            fare = item[1]["Fare"]

            PassengerId = item[1]["PassengerId"]

            if (fare != fare):

                new_data.loc[new_data["PassengerId"]==PassengerId,"Pclass"] = base_data.Pclass.mode()

                new_data.loc[new_data["PassengerId"]==PassengerId,"Fare"] = base_data.Fare.median()

            else:    

                pclass_impute = (base_data.groupby('Pclass')['Fare'].median() - fare).abs().idxmin()

                new_data.loc[new_data["PassengerId"]==PassengerId,"Pclass"] = pclass_impute          

  

    new_data["Pclass"] = new_data["Pclass"].astype(int)   

   

    # I assume name and sex will be provided.

        

    # SibSp

    new_data.loc[new_data.SibSp != new_data.SibSp, "SibSp"] = base_data.SibSp.mode()[0]



    # Parch

    new_data.loc[new_data.Parch != new_data.Parch, "Parch"] = base_data.Parch.mode()[0]

    

    # Adding Family Size and Family Type

    new_data["Family_size"] = new_data["SibSp"] + new_data["Parch"] +1

    new_data["Family_type"] = "Alone"

    new_data.loc[new_data["Family_size"] > 1, "Family_type"] = "Small"

    new_data.loc[new_data["Family_size"] > 4, "Family_type"] = "Big"

        

    # Fare

    # TODO: impute it based on Pclass.

    new_data.loc[new_data.Fare != new_data.Fare, "Fare"] = base_data.Fare.median()

    

    # New feature based on: https://www.kaggle.com/cdeotte/titanic-wcg-xgboost-0-84688

    new_data['Adj_fare'] = new_data['Fare'] / new_data['Family_size']

    

    # Creating new attribute to address Cabin Assignment.

    new_data["Cabin_assigned"] = 1

    new_data.loc[new_data.Cabin != new_data.Cabin,"Cabin_assigned"] = 0

    

    # Impute missing values of Embarked with mode.

    # TODO: use other factors, mode is not accurate.

    new_data.loc[new_data.Embarked != new_data.Embarked,"Embarked"] = base_data.Embarked.mode()[0]



    # Create a title column    

    for my_data in [base_data,new_data]:

        my_data["Title"] = my_data.Name.str.extract('\w+,\s+([a-zA-z\s]+)\s*\..*')

        my_data["Last_Name"] = my_data.Name.str.extract('(\w+),\s+[a-zA-z\s]+\s*\..*')



    title_set = set(base_data.Title.unique())

    title_set.update(['Other','Special'])

    

    # Replace Misc. titles

    mrs = ["Mrs","Mme"]

    miss = ["Mlle","Miss","Ns"]

    other = ["Capt","Col","Major","Dr","Rev","Mlle","Mme","Ms","Ms",]

    special = ["Lady", "Don", "Dona", "Sir", "the Countess", "Jonkheer"]     



    for my_data in [base_data,new_data]:

        my_data["Title"].replace(mrs,"Mrs", inplace=True)

        my_data["Title"].replace(miss,"Miss", inplace=True)

        my_data["Title"].replace(other,"Other", inplace=True)

        my_data["Title"].replace(special,"Special", inplace=True)

   

    Titles = ["Mr","Mrs","Miss","Master","Other","Special"]



    # Any other title other than what we have seen in training data,

    # will be other. 

    new_data.loc[new_data["Title"].apply(lambda x: x not in title_set),"Title"] = "Other"



    # Impute missing values of Age with median of training data

    # See your previous R kernel for other methods of imputing Age.

    # new_data.loc[new_data.Age != new_data.Age, "Age"] = base_data.Age.median()

    

    new_data["Sex_numerical"] = new_data["Sex"].map({"male": 1, "female": 0}).astype(int)   

    new_data["Age_missing"] = 0

    new_data.loc[new_data.Age.isnull(),"Age_missing"] = 1

            

    # estimating missing age based on Title and Pclass:

    for title in Titles:

        for pclass in range(1,4):

            try:

                agg_df = base_data.groupby(['Title','Pclass'])['Age'].median()

                age_to_impute = np.asscalar(agg_df.loc[[(title, pclass)]].values)

                new_data.loc[(new_data['Age'].isnull()) & (new_data['Title'] == title) & (new_data['Pclass'] == pclass), 'Age'] = age_to_impute

            except:

                agg_df = base_data.groupby(['Title'])['Age'].median()

                age_to_impute = np.asscalar(agg_df.loc[[(title)]].values)

                new_data.loc[(new_data['Age'].isnull()) & (new_data['Title'] == title), 'Age'] = age_to_impute



    new_data["Child"] = 0

    new_data.loc[new_data["Age"]<18,"Child"] = 1



        

    # Adding two extra features:

    # Reference: https://www.kaggle.com/cdeotte/titanic-wcg-xgboost-0-84688

    new_data['Fsize_age'] = new_data['Family_size'] + new_data['Age']/70

    new_data['Adj_fare'] = new_data['Fare'] / new_data['Family_size']



    if create_dummy:

        ## Converting categorical values to dummies

        new_data = pd.concat([new_data, pd.get_dummies(new_data["Embarked"], prefix='Embarked')], axis=1)

        new_data = pd.concat([new_data, pd.get_dummies(new_data["Title"], prefix='Title')], axis=1)

        new_data = pd.concat([new_data, pd.get_dummies(new_data["Pclass"], prefix='Pclass')], axis=1)

        new_data = pd.concat([new_data, pd.get_dummies(new_data["Family_type"], prefix='Ftype')], axis=1)



    return new_data



def plot_feature_importance(model,X_train,model_name,num_features):

    fig, axes = plt.subplots(nrows = 1, ncols = 1, sharex="all", figsize=(10,3))

    indices = np.argsort(model.feature_importances_)[::-1][:num_features]

    g = sns.barplot(y = X_train.columns[indices][:num_features],

                    x = model.feature_importances_[indices][:num_features] , orient='h')

    g.set_xlabel("Relative importance",fontsize=12)

    g.set_ylabel("Features",fontsize=12)

    g.tick_params(labelsize=9)

    g.set_title(model_name + " feature importance");

    

    

def evaluate_model_with_test_data(test_set,features,model):

    Y_test = test_set["Survived"]

    X_test = test_set[features]

    return accuracy_score(Y_test, model.predict(X_test)) 



def retrain_with_whole_data(data,model,features):

    Y_train = data["Survived"]

    X_train = data[features]

    return model.fit(X_train,Y_train)



def predict_test_data(model, data, features):

    X_test_prediction = data[features]

    return pd.Series(model.predict(X_test_prediction).astype(int), name="Survived")



## Loading data

train_data = load_data('../input/titanic/train.csv', delimiter = ',')

test_data = load_data('../input/titanic/test.csv', delimiter = ',')
# Cleaning data

tr_data_c = clean_titanic_dataset(train_data,train_data, create_dummy=True)

ts_data_c = clean_titanic_dataset(train_data,test_data, create_dummy=True)

IDtest = ts_data_c["PassengerId"]
# Setting Index Column to PassengerId

tr_data_c.set_index("PassengerId",inplace=True)

ts_data_c.set_index("PassengerId",inplace=True)
# Adding one more features: Family Survival

# The idea is mentioned:

# https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83

# https://www.kaggle.com/vincentlugat/200-lines-randomized-search-lgbm-82-3

# https://www.kaggle.com/shunjiangxu/blood-is-thicker-than-water-friendship-forever

# https://www.kaggle.com/c/titanic/discussion/57447#latest-592673

# https://www.kaggle.com/cdeotte/titanic-using-name-only-0-81818

# Following implementation is according to: https://www.kaggle.com/mauricef/titanic



df = pd.concat([tr_data_c, ts_data_c], axis=0, sort=False)



# Choose Woman or child

df['Is_woman_child'] = ((df.Title == 'Master') | (df.Sex == 'female'))



# Computing family survival rate other than the individual

family = df.groupby(df.Last_Name).Survived

df['Family_total_count'] = family.transform(lambda s: s[df.Is_woman_child].fillna(0).count())

df['Family_total_count'] = df.mask(df.Is_woman_child, df.Family_total_count - 1, axis=0)

df['Family_survived_count'] = family.transform(lambda s: s[df.Is_woman_child].fillna(0).sum())

df['Family_survived_count'] = df.mask(df.Is_woman_child, df.Family_survived_count - df.Survived.fillna(0), axis=0)

df['Family_survival_rate'] = (df.Family_survived_count / df.Family_total_count.replace(0, np.nan))

df['Is_single_traveler'] = df.Family_total_count == 0

df.Family_survival_rate.fillna(0, inplace=True);
train_data_cleaned, test_data_cleaned = df.loc[tr_data_c.index], df.loc[ts_data_c.index]   

# train_data_cleaned: total train data

# test_data_cleaned: competition test data

# train_set: subset of train_data to train prediction models

# test_set: subset of train_data to test the models. 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=914)

for train_index, test_index in split.split(train_data_cleaned, train_data_cleaned[["Sex"]]):

    train_set = train_data_cleaned.iloc[train_index]

    test_set = train_data_cleaned.iloc[test_index]
# Inspired by: https://www.kaggle.com/vincentlugat/200-lines-randomized-search-lgbm-82-3



lgbm_grid_search = False

lgbm_feature_names = [                   

                   'Sex_numerical'

                 , 'Family_survival_rate'

                 , 'Is_single_traveler'

                 #, 'Adj_fare'

                 #, 'Pclass_1'

                 #, 'Pclass_2'

                 #, 'Pclass_3'

                 #, 'Title_Master'

                 #, 'Title_Miss'

                 #, 'Title_Mr'

                 #, 'Ftype_Small'

                 #, 'Ftype_Big'        

                     ]



Y_train = train_set["Survived"]

X_train = train_set[lgbm_feature_names]





if lgbm_grid_search:

    fit_params = {"early_stopping_rounds" : 100, 

                 "eval_metric" : 'auc', 

                 "eval_set" : [(X_train,Y_train)],

                 'eval_names': ['valid'],

                 'verbose': 0,

                 'categorical_feature': 'auto'}



    param_test = {'learning_rate' : [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4],

                  'n_estimators' : [100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000],

                  'num_leaves': randint(6, 200), 

                  'min_child_samples': randint(100, 500), 

                  'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

                  'subsample': uniform(loc = 0.2, scale = 0.8), 

                  'max_depth': [-1, 1, 2, 3, 4, 5, 6, 7,8,9,10],

                  'colsample_bytree': uniform(loc = 0.4, scale = 0.6),

                  'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],

                  'reg_lambda': [0, 0.01, 0.1, 0.3, 1, 5, 10, 20, 50, 100]}



    lgbm_clf = lgbm.LGBMClassifier(random_state = 914, silent = True, metric = 'None', n_jobs = 16)

    grid_search = RandomizedSearchCV(

        estimator = lgbm_clf, param_distributions = param_test, 

        n_iter = 2000,

        scoring = 'accuracy',

        cv = 5,

        refit = True,

        random_state = 914,

        verbose = True)



    grid_search.fit(X_train, Y_train, **fit_params)

    opt_parameters = grid_search.best_params_

    print(grid_search.best_params_)

else:

    opt_parameters =     {

        'colsample_bytree': 0.951896848025216,

        'learning_rate': 0.2,

        'max_depth': 10,

        'min_child_samples': 102,

        'min_child_weight': 0.01,

        'n_estimators': 400,

        'num_leaves': 102,

        'reg_alpha': 2,

        'reg_lambda': 0.5,

        'subsample': 0.4194694182848429}

      

lgbm_best = lgbm.LGBMClassifier(**opt_parameters)

lgbm_model_best_ = lgbm_best.fit(X_train,Y_train)    
# plot feature importance

plot_feature_importance(lgbm_model_best_,X_train,'model_name',3)
evaluate_model_with_test_data(test_set,lgbm_feature_names,lgbm_model_best_)
# Retrain with all data and predict the test data

lgbm_on_training = retrain_with_whole_data(train_data_cleaned,lgbm_model_best_,lgbm_feature_names)

test_Survived_lgbm = predict_test_data(lgbm_on_training, test_data_cleaned,lgbm_feature_names)
results = pd.concat([IDtest,test_Survived_lgbm],axis=1)

results.to_csv("submission.csv",index=False)