import pandas as pd

#gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")



train_y = train['Survived']

train_x = train.drop('Survived', axis=1)
import numpy as np



def fillna_age(dataframe):    

    for index, df in dataframe.iterrows():

        if pd.isna( df.Age ):

            if 'Miss' in df["Name"]:

                dataframe.at[index,"Age"] = 17

            elif 'Mrs' in  df["Name"]:

                dataframe.at[index, "Age"]= 25

            elif 'Master' in  df["Name"]:

                dataframe.at[index, "Age"]= 4

            elif 'Mr' in  df["Name"]:

                dataframe.at[index, "Age"]= 41

            else :

                dataframe.at[index, "Age"]=  dataframe["Age"].median()



    return dataframe



def fillna_fare(dataframe):    

    for index, df in dataframe.iterrows():

        if pd.isna( df.Fare ):

            if df.Pclass == 3:

                dataframe.at[index,"Fare"] = df.FamilyNum * 7

            if df.Pclass == 2:

                dataframe.at[index,"Fare"] = df.FamilyNum * 14

            else :

                dataframe.at[index,"Fare"] = df.FamilyNum * 30



    return dataframe







def add_title(dataframe):



    dataframe["Title"] = float(0)



    for index, df in dataframe.iterrows():



        tmp = df['Name']

        tmp = tmp.replace('Mlle', 'Miss')

        tmp = tmp.replace('Ms', 'Miss')

        tmp = tmp.replace('Mme', 'Mrs')



        tmp_val= 1 #0.25

        if ("Mr." in tmp):

            tmp_val = 0 # 0.156673

        elif ("Miss." in tmp):

            tmp_val = 4 # 0.697802

        elif ("Mrs." in tmp):

            tmp_val = 5 # 0.792000

        elif ("Master." in tmp ):

            tmp_val = 3 # 0.575000

        elif ("Misc." in tmp):

            tmp_val = 2 # 0.444444



        dataframe.at[index, "Title"] = tmp_val



    return dataframe



def add_isalone(dataframe):



    dataframe["IsAlone"] = int(0)



    for index, df in dataframe.iterrows():



        if df.FamilyNum == 1:

            dataframe.at[index, "IsAlone"] = 1



    return dataframe





# pos = left or right

def add_cabinpos(dataframe):    



    dataframe["CabinPos"] = 0



    for index, df in dataframe.iterrows():



        if not(pd.isna( df.Cabin )): 

            tmp = df.Cabin[-1:]

            if (tmp=='1' or tmp=='3' or tmp=='5' or tmp=='7' or tmp=='9'):

                dataframe.at[index, "CabinPos"]= 1

            elif (tmp=='2' or tmp=='4' or tmp=='6' or tmp=='8' or tmp=='0'):

                dataframe.at[index, "CabinPos"]= -1



    return dataframe



def add_familynum(dataframe):

    dataframe["FamilyNum"] = 0



    for index, df in dataframe.iterrows():

        dataframe.at[index, "FamilyNum"] = df.SibSp	+ df.Parch + 1



    return dataframe





def add_isbigfamily(dataframe):

    dataframe["IsBigFamily"] = int(0)



    for index, df in dataframe.iterrows():



        if df.FamilyNum >= 5:

            dataframe.at[index, "IsBigFamily"] = 1



    return dataframe







def create_familysurviverate( dataframe ):



    name_array = []

    family_num = []

    family_survive_count = []

    for _, df in dataframe.iterrows():

        if (df.SibSp + df.Parch) > 0:



            name = df.Name

            name_split = name.split(" ")

            name_split_first = name_split[0].replace(',','')



            if name_split_first in name_array:

                family_num          [name_array.index(name_split_first)] = family_num[name_array.index(name_split_first)] + 1

                family_survive_count[name_array.index(name_split_first)] = family_survive_count[name_array.index(name_split_first)] + df.Survived

            else:

                name_array.append(name_split_first)

                family_num.append(1)

                family_survive_count.append(df.Survived)



    name_survive_ratio = []            

    for i, _ in enumerate(family_num):

        name_survive_ratio.append( family_survive_count[i] / float(family_num[i]))



    fname_svv_dict = {}  

    for index, name_split_first in enumerate(name_array):

        fname_svv_dict[name_split_first] = name_survive_ratio[index]

    

    # print(fname_svv_dict )



    return fname_svv_dict





def add_familysurviverate( dataframe, fname_svv_dict ):

    dataframe["FamilySurviveRate"] = 0.50



    # refs. https://ja.wikipedia.org/wiki/%E3%82%BF%E3%82%A4%E3%82%BF%E3%83%8B%E3%83%83%E3%82%AF%E5%8F%B7%E6%B2%88%E6%B2%A1%E4%BA%8B%E6%95%85

    default_table = [[0.97, 0.86, 0.46], [0.33, 0.08, 0.16]]

    

    for index, row in dataframe.iterrows():

        

        if not row.IsAlone: 

            name = row.Name

            name_split = name.split(" ")

            name_split_first = name_split[0].replace(',','')



            if name_split_first in fname_svv_dict:

                dataframe.at[index, "FamilySurviveRate"] = fname_svv_dict[name_split_first]

        else :

            s_ix = 0 if(row.Sex == 'female')else 1

            c_ix = int(row.Pclass) - 1

            dataframe.at[index, "FamilySurviveRate"] = default_table[s_ix][c_ix]



    return dataframe



def conv_cabin_ch2i( dataframe ):

    dataframe["Cabin"] = dataframe["Cabin"].fillna('D') # D=center = 0

    dataframe.loc[:, "Cabin"][ dataframe["Cabin"].str.contains('A') ] = 'A'

    dataframe.loc[:, "Cabin"][ dataframe["Cabin"].str.contains('B') ] = 'B'

    dataframe.loc[:, "Cabin"][ dataframe["Cabin"].str.contains('C') ] = 'C'

    dataframe.loc[:, "Cabin"][ dataframe["Cabin"].str.contains('D') ] = 'D'

    dataframe.loc[:, "Cabin"][ dataframe["Cabin"].str.contains('E') ] = 'E'

    dataframe.loc[:, "Cabin"][ dataframe["Cabin"].str.contains('F') ] = 'F'

    dataframe.loc[:, "Cabin"][ dataframe["Cabin"].str.contains('G') ] = 'G'

    dataframe.loc[:, "Cabin"][ dataframe["Cabin"].str.contains('T') ] = 'T'



    # adjust ticket class fillna

    dataframe.loc[:, "Cabin"][ dataframe["Cabin"] == 'A' ] = int(0) 

    dataframe.loc[:, "Cabin"][ dataframe["Cabin"] == 'B' ] = int(1)

    dataframe.loc[:, "Cabin"][ dataframe["Cabin"] == 'C' ] = int(2)

    dataframe.loc[:, "Cabin"][ dataframe["Cabin"] == 'D' ] = int(3)

    dataframe.loc[:, "Cabin"][ dataframe["Cabin"] == 'E' ] = int(4)

    dataframe.loc[:, "Cabin"][ dataframe["Cabin"] == 'F' ] = int(5)

    dataframe.loc[:, "Cabin"][ dataframe["Cabin"] == 'G' ] = int(6)

    dataframe.loc[:, "Cabin"][ dataframe["Cabin"] == 'T' ] = int(2)

    dataframe = dataframe.astype({"Cabin": "float"})

    

    return dataframe





def process_df( dataframe, fname_svv_dict ):

    

    dataframe = dataframe.drop('Ticket', axis = 1)



    dataframe = add_title( dataframe )

    dataframe = fillna_age( dataframe )

    dataframe = add_cabinpos( dataframe )

    dataframe = add_familynum( dataframe )

    dataframe = add_isalone( dataframe )

    dataframe = add_isbigfamily( dataframe )

    dataframe = fillna_fare( dataframe )

    dataframe.loc[:, "Fare"] = dataframe["Fare"] / dataframe["FamilyNum"]

    dataframe = add_familysurviverate( dataframe, fname_svv_dict )



 

    dataframe.loc[:, "Sex"][dataframe["Sex"] == "male"]   =  -1

    dataframe.loc[:, "Sex"][dataframe["Sex"] == "female"] =  1

    dataframe.loc[:, "Embarked"][dataframe["Embarked"] == "S" ] = 1

    dataframe.loc[:, "Embarked"][dataframe["Embarked"] == "C" ] = 2

    dataframe.loc[:, "Embarked"][dataframe["Embarked"] == "Q"] =  3

    dataframe["Embarked"] = dataframe["Embarked"].fillna(0)

    dataframe = dataframe.astype({"Sex": "float"})



    dataframe = conv_cabin_ch2i( dataframe )

    dataframe = dataframe.drop('Name', axis=1)

    

    return dataframe



fname_svv_dict = create_familysurviverate( train )

train_x = process_df( train_x, fname_svv_dict )



train_x
from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

import lightgbm as lgb



import eli5

from eli5.sklearn import PermutationImportance
def do_PermutationImportance( target_model ):

    Xtrn, Xval, ytrn, yval = train_test_split( train_x.values, train_y.values, shuffle=True )



    perm = PermutationImportance( target_model.fit( Xtrn, ytrn ), n_iter=10 ).fit(Xval, yval)



    return perm



perm = do_PermutationImportance( svm.SVC(kernel='rbf', C=3000, gamma=0.00007) )



perm_importances = np.array( [perm.feature_importances_] )   

perm_std         = np.array( [perm.feature_importances_std_])

eli5.show_weights(perm, feature_names = train_x.columns.tolist())
perm = do_PermutationImportance( svm.SVC(kernel='rbf', C=3000, gamma=0.00007) )

perm_importances = np.append( perm_importances, [perm.feature_importances_],     axis=0 )

perm_std         = np.append( perm_std        , [perm.feature_importances_std_], axis=0 )

eli5.show_weights(perm, feature_names = train_x.columns.tolist())
for _ in range(48):

    perm = do_PermutationImportance( svm.SVC(kernel='rbf', C=3000, gamma=0.00007) )

    perm_importances = np.append( perm_importances, [perm.feature_importances_],     axis=0 )

    perm_std         = np.append( perm_std        , [perm.feature_importances_std_], axis=0 )



eli5.show_weights(perm, feature_names = train_x.columns.tolist())
for _ in range(50):

    perm = do_PermutationImportance( RandomForestClassifier(n_estimators=300, max_depth=7) )

    perm_importances = np.append( perm_importances, [perm.feature_importances_],     axis=0 )

    perm_std         = np.append( perm_std        , [perm.feature_importances_std_], axis=0 )

eli5.show_weights(perm, feature_names = train_x.columns.tolist())
for _ in range(50):

    perm = do_PermutationImportance( xgb.XGBClassifier(n_estimators=300, max_depth=7) )

    perm_importances = np.append( perm_importances, [perm.feature_importances_],     axis=0 )

    perm_std         = np.append( perm_std        , [perm.feature_importances_std_], axis=0 )

eli5.show_weights(perm, feature_names = train_x.columns.tolist())
for _ in range(50):

    perm = do_PermutationImportance( lgb.LGBMClassifier(n_estimators=300, max_depth=7) )

    perm_importances = np.append( perm_importances, [perm.feature_importances_],     axis=0 )

    perm_std         = np.append( perm_std        , [perm.feature_importances_std_], axis=0 )

eli5.show_weights(perm, feature_names = train_x.columns.tolist())
%matplotlib inline

 

import numpy as np

import matplotlib.pyplot as plt



label = train_x.columns.tolist()

label[-1] = 'SurviveRate' # 文字数長いため

fig = plt.figure(figsize=(20, 5))

ax = fig.add_subplot(1,1,1)

ax.set_yscale('log')

x_list = np.array([ i for i in range(len(perm_importances[0]))])

plt.bar(x_list-0.2, perm_importances.mean(axis=0), width=0.2, label='importance_mean', tick_label=label, align="center")

plt.bar(x_list+0.0, perm_importances.max(axis=0),  width=0.2, label='importance_max' )

plt.bar(x_list+0.2, perm_std.mean(axis=0),         width=0.2, label='importance_std')

plt.legend()
fig = plt.figure(figsize=(20, 5))

ax = fig.add_subplot(1,1,1)

ax.set_xlim([-0.05, 0.05])

plt.hist(perm_importances, label=label, bins=30)

plt.legend()
drop_list = [ "IsBigFamily", "PassengerId"]

train_x = train_x.drop(drop_list, axis=1)
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from sklearn.metrics import log_loss

from sklearn.model_selection import StratifiedKFold, cross_validate



k_num = 5



history=[] 

def search_hp( max_evals, score, space ):

    trials = Trials()

    fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=max_evals)

    

    print("\n"*3)

    print("--searched--")

    print("\n"*3)    



    global history

    history = sorted(history, key=lambda tpl: tpl[1])

    for _hist in history[:10]:

        print( "score:"+ str(_hist[1]) + " ... params:" + str(_hist[0]) )

def svm_score(params):

    model = svm.SVC(**params)

    

    kf = StratifiedKFold(n_splits=k_num, shuffle=True, random_state=0)

    _scores = cross_validate(model, X=train_x, y=train_y.values, cv=kf)

    # 最小化なので符号を反転する

    _score = -1 * _scores['test_score'].mean()      

    print( "param:" + str(params),  "score:" + str(_score) )

    history.append((params, _score))



    return {'loss':_score, 'status':STATUS_OK}



svm_space = {

    'C':          hp.choice('C',         [1000, 2000, 3000, 4000, 5000] ),

    'degree':     hp.choice('degree',    [3,4,5,6,7,8,9,10] ),    

    'gamma':      hp.loguniform('gamma', np.log(1e-8), np.log(1.0)),    

    'tol':        hp.loguniform('tol',   np.log(1e-8), np.log(1.0)),

}



svm_best_score, svm_best_prm = search_hp(50, svm_score, svm_space)
def rf_score(params):

    model = RandomForestClassifier(**params)

    

    kf = StratifiedKFold(n_splits=k_num, shuffle=True, random_state=0)

    _scores = cross_validate(model, X=train_x, y=train_y.values, cv=kf)

    # 最小化なので符号を反転する

    _score = -1 * _scores['test_score'].mean()    

    print( "param:" + str(params),  "score:" + str(_score) )

    history.append((params, _score))



    return {'loss':_score, 'status':STATUS_OK}

    

rf_space = {

    'n_estimators' :     hp.choice('n_estimators',     [10, 30, 100, 300, 1000, 1500, 3000]),

    'max_depth':         hp.choice('max_depth',        [ 3, 5, 7, 9, 15, 20, None ]),

    'min_samples_split': hp.choice('min_samples_split',[ 0.1, 0.2, 0.4, 0.8, 1.0]),

    'min_samples_leaf':  hp.choice('min_samples_leaf', [ 1, 2, 3 ]),

    'max_features':      hp.choice('max_features',     ['auto', 'sqrt', 'log2', None]),

    'n_jobs': hp.choice('n_jobs', [-1] )

}
history=[] 

search_hp(50, rf_score, rf_space)
import numpy as np 



def xgb_score(params):

    model = xgb.XGBClassifier(**params)

    

    kf = StratifiedKFold(n_splits=k_num, shuffle=True, random_state=0)

    _scores = cross_validate(model, X=train_x, y=train_y.values, cv=kf)

    # 最小化なので符号を反転する

    _score = -1 * _scores['test_score'].mean()    

    print( "param:" + str(params),  "score:" + str(_score) )

    history.append((params, _score))

    return {'loss':_score, 'status':STATUS_OK}

    

xgb_space = {

    'n_estimators' :     hp.choice('n_estimators',      [10, 30, 100, 300, 1000, 3000]),

    'max_depth':         hp.choice('max_depth',         [3,4,5,6,7,8,9,10] ),

    'subsample':         hp.quniform('subsample',        0.1, 0.95, 0.05 ),

    'colsample_bytree':  hp.quniform('colsample_bytree', 0.5,  1.0, 0.05 ),

    'learning_rate':     hp.loguniform('learning_rate',  np.log(1e-4), np.log(1e-1)),

    'gamma':             hp.loguniform('gamma',          np.log(1e-8), np.log(1.0)),

    'alpha':             hp.loguniform('alpha',          np.log(1e-8), np.log(1.0)),

    'lambda':            hp.loguniform('lambda',         np.log(1e-8), np.log(1.0)),    

    'n_jobs': hp.choice('n_jobs', [-1] )

}



history=[]

search_hp(50, xgb_score, xgb_space)
def lgbm_score(params):

    model = lgb.LGBMClassifier(**params)

    

    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    _scores = cross_validate(model, X=train_x, y=train_y.values, cv=kf)

    # 最小化なので符号を反転する

    _score = -1 * _scores['test_score'].mean()    

    print( "param:" + str(params),  "score:" + str(_score) )

    history.append((params, _score))

    return {'loss':_score, 'status':STATUS_OK}

    

lgbm_space = {

    'num_leaves':        hp.choice('num_leaves',        [3,7,15,31,63,127] ),    

    'n_estimators' :     hp.choice('n_estimators',      [10, 30, 100, 300, 1000, 3000]),

    'max_depth':         hp.choice('max_depth',         [3,4,5,6,7,8,9,-1] ),

    'subsample_for_bin': hp.choice('subsample_for_bin', [2000, 20000, 200000, 500000] ),

    'subsample':         hp.quniform('subsample',        0.1, 0.95, 0.05 ),

    'min_split_gain':    hp.quniform('min_split_gain',   0.5,  1.0, 0.05 ),

    'colsample_bytree':  hp.quniform('colsample_bytree', 0.5,  1.0, 0.05 ),    

    'learning_rate':     hp.loguniform('learning_rate',  np.log(1e-4), np.log(1e-1)),

    'reg_alpha':         hp.loguniform('reg_alpha',      np.log(1e-8), np.log(1.0)),

    'reg_lambda':        hp.loguniform('reg_lambda',     np.log(1e-8), np.log(1.0))

}



history=[]

search_hp(100, lgbm_score, lgbm_space)
svm_score = 0.8810669387439546

rf_score  = 0.8889258302134169

xgb_score = 0.8911543318697476

lgb_score = 0.896838610827374

svm_prms = {'C': 5000, 'degree': 7, 'gamma': 5.461685273552484e-05, 'tol': 0.006054430142255137}

rf_prms  = {'max_depth': 7, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 0.1, 'n_estimators': 300, 'n_jobs': -1}

xgb_prms = {'alpha': 1.6802879665271848e-06, 'colsample_bytree': 0.8, 'gamma': 0.0003259964944445497, 'lambda': 0.0023499634809638274, 'learning_rate': 0.09495947657869509, 'max_depth': 3, 'n_estimators': 300, 'n_jobs': -1, 'subsample': 0.7000000000000001}

lgb_prms = {'colsample_bytree': 0.55, 'learning_rate': 0.024285603061513013, 'max_depth': 9, 'min_split_gain': 0.8, 'n_estimators': 1000, 'num_leaves': 7, 'reg_alpha': 0.5850700627702747, 'reg_lambda': 8.262269772946297e-05, 'subsample': 0.45, 'subsample_for_bin': 500000}



svm_prms["probability"] = True
svm_model = svm.SVC               (**svm_prms).fit(train_x, train_y.values)

rf_model  = RandomForestClassifier(**rf_prms) .fit(train_x, train_y.values)

xgb_model = xgb.XGBClassifier     (**xgb_prms).fit(train_x, train_y.values)

lgb_model = lgb.LGBMClassifier    (**lgb_prms).fit(train_x, train_y.values)
def create_submission_csv(my_model, test_features, filename):

    pred = my_model.predict(test_features)

    passenger_id = np.array(test["PassengerId"]).astype(int)

    my_solution = pd.DataFrame(pred, passenger_id, columns = ["Survived"])

    my_solution.to_csv(filename, index_label = ["PassengerId"])
test_x = test

test_x = process_df( test_x, fname_svv_dict )

test_x = test_x.drop(drop_list, axis=1)
create_submission_csv( svm_model, test_x, "svm_result.csv")

create_submission_csv( rf_model,  test_x, "rf_result.csv")

create_submission_csv( xgb_model, test_x, "xgb_result.csv")

create_submission_csv( lgb_model, test_x, "lgb_result.csv")
pd.read_csv( "rf_result.csv")
from sklearn.metrics import accuracy_score



def create_ensemble_feature( w_model_list, w_score_list, x ):

    _proba = np.zeros( (len(x), len(w_model_list)) )

    ws_mean = np.array(w_score_list).mean()

    

    for i in range( len(w_model_list) ):

        _w_proba = w_model_list[i].predict_proba(x) * (w_score_list[i] - ws_mean + 1.0)

        _proba[:,i] = _w_proba[:,1] #

        

    return _proba

skf  = StratifiedKFold(n_splits=k_num, shuffle=False, random_state=54)

skf_split = skf.split(train_x, train_y.values)



w_model_list2d = [None]*k_num

w_score_list   = [svm_score,  rf_score,  xgb_score,  lgb_score]



k = 0

for train_index, valid_index in skf_split:

        

    # create weak model 

    X_train = train_x.iloc[train_index]

    y_train = train_y.values[train_index]



    _svm_model = svm.SVC               (**svm_prms).fit(X_train, y_train)

    _rf_model  = RandomForestClassifier(**rf_prms) .fit(X_train, y_train)

    _xgb_model = xgb.XGBClassifier     (**xgb_prms).fit(X_train, y_train)

    _lgb_model = lgb.LGBMClassifier    (**lgb_prms).fit(X_train, y_train)

        

    w_model_list = [_svm_model, _rf_model, _xgb_model, _lgb_model]

    w_model_list2d[k] = w_model_list

    k+=1



def _ensemble_score(params):

    _log_loss = 0.0

    _acc = 0.0

    k = 0

    

    skf = StratifiedKFold(n_splits=k_num, shuffle=False, random_state=54)    

    skf_split = skf.split(train_x, train_y.values)

    for train_index, valid_index in skf_split:



        # ensemble learning

        X_train, X_valid = train_x.iloc[train_index],   train_x.iloc[valid_index]

        y_train, y_valid = train_y.values[train_index], train_y.values[valid_index]

        

        _train_proba = create_ensemble_feature(w_model_list2d[k], w_score_list, X_train)

        ensemble_model = lgb.LGBMClassifier(**params).fit(_train_proba, y_train)



        # validate ensemble learning

        _valid_proba = create_ensemble_feature(w_model_list2d[k], w_score_list, X_valid)

        ensemble_pred = ensemble_model.predict(_valid_proba)

        

        print(y_valid[:10], ensemble_pred[:10])

        

        _log_loss += log_loss(y_valid, ensemble_pred) 

        _acc      += accuracy_score(y_valid, ensemble_pred)



    print( "param:" + str(params),  "score:" + str(_log_loss/k_num) + ", " + str(_acc/k_num) )

    history.append((params, _log_loss/k_num))

    return {'loss':_log_loss/k_num, 'status':STATUS_OK}    

    

ensemble_space = lgbm_space



history=[]

search_hp(50, _ensemble_score, ensemble_space)
ensemble_valid_score = 0.9394948980188929

ensemble_prms = {'colsample_bytree': 0.75, 'learning_rate': 0.009116960816409152, 'max_depth': -1, 'min_split_gain': 0.8, 'n_estimators': 1000, 'num_leaves': 7, 'reg_alpha': 0.005028522047266823, 'reg_lambda': 3.41676672214522e-07, 'subsample': 0.75, 'subsample_for_bin': 2000}
%matplotlib inline

 

import numpy as np

import matplotlib.pyplot as plt



label = ["svm", "random_forest", "xgboost", "lgbm", "ensemble"]

fig = plt.figure(figsize=(10, 5))

ax = fig.add_subplot(1,1,1)

ax.set_ylim([0.8,1.0])

x_list = [i for i in range(5)] 

y_list = [svm_score,  rf_score,  xgb_score,  lgb_score, ensemble_valid_score]

plt.bar(label, y_list, align="center")
_svm_model = svm.SVC               (**svm_prms).fit(train_x, train_y)

_rf_model  = RandomForestClassifier(**rf_prms) .fit(train_x, train_y)

_xgb_model = xgb.XGBClassifier     (**xgb_prms).fit(train_x, train_y)

_lgb_model = lgb.LGBMClassifier    (**lgb_prms).fit(train_x, train_y)

        

w_model_list = [_svm_model, _rf_model, _xgb_model, _lgb_model]

w_score_list = [ svm_score,  rf_score,  xgb_score,  lgb_score]

_train_proba = create_ensemble_feature(w_model_list, w_score_list, train_x)

ensemble_model = lgb.LGBMClassifier(**ensemble_prms).fit(_train_proba, train_y)



_test_proba = create_ensemble_feature(w_model_list, w_score_list, test_x)

create_submission_csv( ensemble_model, _test_proba, "ensemble_result.csv")
pd.read_csv( "ensemble_result.csv")