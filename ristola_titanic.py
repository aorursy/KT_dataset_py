import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import RFECV, RFE

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.tree import export_graphviz

from IPython import display

from sklearn import tree

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, MultiLabelBinarizer

from sklearn.metrics import accuracy_score, make_scorer

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import RandomizedSearchCV, KFold

from sklearn import preprocessing

from sklearn.preprocessing import KBinsDiscretizer

from sklearn.impute import MissingIndicator, KNNImputer

from sklearn.preprocessing import label_binarize

from sklearn.metrics import plot_confusion_matrix



from sklearn.feature_selection import f_classif, f_regression, mutual_info_regression, mutual_info_classif

from sklearn.pipeline import Pipeline



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RepeatedKFold



from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.compose import ColumnTransformer



from sklearn.impute import SimpleImputer



from sklearn.model_selection import train_test_split



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

import xgboost as xgb



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_predict



from scipy.stats import sem

from sklearn.metrics import mean_squared_error

from IPython.display import display, HTML

from sklearn.metrics import mean_absolute_error

from xgboost import plot_tree

from sklearn.metrics import precision_score, recall_score, accuracy_score

from xgboost import plot_tree

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix





%matplotlib inline
def scale_data(train, test=pd.DataFrame(), numeric_features=[],  categorical_features=[],  scaler=preprocessing.StandardScaler()):

    

    """Scales separately both train and test dataframes

    



    A C   train_: + col first dataframe

        test: second dataframe, if empty, only first one will be used

        columns_to_encode, columns needing encoding 

        columns_to_scale, scalng needed

        columns_to_binary, binary colmuns

        scaler= Scaler function like StandardScaler()):

    

    Returns:

         Scaled test and train dataframes (only first one is needed to be returned)

         Column lists as encoding might have created new columns

    """





    def scale(df, numeric_features, categorical_features, scaler):

    

        #scale and encode dataframe columns inplace

        column_list_encode = []

        if categorical_features != []:

            df_encode=pd.DataFrame()

            for col in categorical_features:

                df[col]=df[col].astype('category')

                df_col=pd.concat([df[col],pd.get_dummies(df[col], prefix='Category__' + col,dummy_na=False)],axis=1).drop(col,axis=1)

                df_encode = pd.concat([df_encode, df_col],axis=1) 

            column_list_encode=df_encode.columns[df_encode.columns.str.startswith("Category__")].tolist()

        if numeric_features != []:

            #note, we must store the original index 

            df_scale=pd.DataFrame(scaler.fit_transform(df[numeric_features]), columns=numeric_features, index=df.index)

        if (categorical_features != []) & (numeric_features != []):

            df=pd.concat([df_scale,df_encode],axis=1)

        elif categorical_features != []:

            df=df_encode

        else:

            df=df_scale

        column_list = numeric_features + column_list_encode

        return(df, column_list)





    train_scaled, column_list_train = scale(train, numeric_features, categorical_features, scaler)

    if not test.empty:

        test_scaled, column_list_test = scale(test, numeric_features, categorical_features, scaler)



        #The creation of dummy values can create different set of columns between train/test

        #therefore we need to take only those columns that are available on both

        column_list_diff_train_test = list(set(column_list_train) - set(column_list_test))

        column_list_diff_test_train = list(set(column_list_test) - set(column_list_train))

        column_list = list(set(column_list_train) - set(column_list_diff_train_test) - set(column_list_diff_test_train))

        return(train_scaled, test_scaled, column_list)

    else:

        return(train_scaled, column_list)

        

def force_show_all(df):

    #if we need to really show all

    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):

            display(HTML(df.to_html()))
def select_kbest(X, y, kbest_score_func, k='all'):

    selector = SelectKBest(kbest_score_func, k=k)

    selector.fit(X, y)

    features_names = X.columns

    features_scores = selector.scores_

    features_selected = selector.get_support()

    

    dict = {'Column': features_names, 'Score': features_scores, 'Selected': features_selected}

    features_df = pd.DataFrame(dict)

    features_df.sort_values('Score', inplace=True, ascending=False)

    features_df.reset_index(drop=True, inplace=True)

 

    return(features_df)

    

def draw_features(y, x, title):

    

    #draw list of features 

    fig, ax = plt.subplots() 

    width = 0.4 # the width of the bars 

    ind = np.arange(len(y)) # the x locations for the groups

    ax.barh(ind, y, width, color='green')

    ax.set_yticks(ind+width/10)

    ax.set_yticklabels(x, minor=False)

    plt.title(title)

    plt.xlabel('Relative importance')

    plt.ylabel('Feature') 

    fig.set_size_inches(11, 9, forward=True)

    plt.plot()



    

def draw_results(X, y, model):



    #plots ROC and confusion matrix from y, X based on model model

    #works only if the results are binary 

    

    y_pred = model.predict(X)    

    fpr, tpr, threshold = metrics.roc_curve(y, y_pred)

    roc_auc = metrics.auc(fpr, tpr)



    plt.figure(figsize=(15,7))

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()



def draw_confusion(X, y, model):



    y_pred = model.predict(X)

    y_pred_bin = np.where(model.predict(dtest)<0.5, 0,1)

    tn, fp, fn, tp = confusion_matrix(y, y_pred_bin).ravel()

    





    n= tn+fp+fn+tp

    n_success = tn + tp

    n_fail = fp + fn

    # Our priors Beta distribution parameters

    prior_alpha = 1

    prior_beta = 1

     

    posterior_distr = stats.beta(a, b)

    p_low, p_high = posterior_distr.interval(0.95)

    

    lower, upper = proportion_confint(tn + tp, tn + fp + fn + tp, 0.05)

    

    print("Accuracy: {}".format(accuracy_score(y, y_pred_bin)))

    print("F1 Score: ", f1_score(y, y_pred_bin, average="macro"))

    print("Precision Score: ", precision_score(y, y_pred_bin, average="macro"))

    print("Recall Score: ", recall_score(y, y_pred_bin, average="macro"))

    print('Frequentist onfidence level lower=%.3f, upper=%.3f' % (lower, upper))

    print('Bayesian credible level lower=%.3f, upper=%.3f' % (p_low, p_high))



    cf_matrix = pd.DataFrame(confusion_matrix(y, y_pred_bin), index = ['survived','lost'], columns=['survived','lost'])

    plt.figure(figsize=(10,7))

    sns.heatmap(cf_matrix, annot=True, fmt='000', annot_kws={"size": 18})

    plt.xlabel('Predicted')

    plt.ylabel('True')

    plt.show()



def create_submission(test, model, dtvalues):

    

    #creates a submission dataframe needed in Kaggle competition 

    y_pred = model.predict(dtvalues)

    y_pred_bin = np.where(model.predict(dtvalues)<0.5, 0,1)

    test.loc[:, target] = y_pred_bin

    submission = test[['PassengerId',target]].copy()

    submission[target] = submission[target].astype(int)

    submission.sort_values('PassengerId').reset_index(drop=True, inplace=True)

    return(submission)



def draw_true_vs_predicted(X, y, model, title, binarize=False):

    

    #this point with a histogram both predicted and true

    y_pred =  model.predict(X)

    if binarize:

        y_pred = np.where(model.predict(dtest)<0.5, 0,1)

    legend = ['True ' + title, 'Predicted ' + title]

    plt.hist([y, y_pred], color=['orange', 'green'])

    plt.ylabel("Frequency")

    plt.legend(legend)

    plt.title('True vs- predicted ' + title)

    plt.show()



def predict_and_store(test, train, model, dtvalues):

    

    #predicts values and stores them as target values to test dataframe 

    y_pred =  model.predict(dtvalues)

    test.loc[:, target] = y_pred

    df = pd.concat([train, test])

    df.sort_values('PassengerId', inplace=True)

    df.reset_index(drop=True, inplace=True)

    return(df)
def one_value(gridsearch_params,

              small_better,

              param_a,

              params,

              dvalue,

              metrics,

              early_stopping_rounds,

              Skfold=True,

              verbose=False):

    

    """This searches the best hyperparameter for selected hyperparameter

    



    Args:

        gridsearch_params: hyperparameter-values to be evaluated 

        small_better:  if small value is searched = true

        param_a: name of the hyperparameter

        params: parameters dictionary

        dtrain: traindata

        metrics: the metric that defines the score

        num_boost_round: max evaluation round 

        Skfold: id used True

        verbose: how much data is showed    

    

    Returns:

        parameter dictionary



    """



    

    

    if small_better == True:

        result_best = float(999999)

    else:

        result_best = float(-999999)

    best_params = None



    for i in gridsearch_params:

        # Update our parameters

        if verbose:

            print("xgb.cv with {}={}".format(param_a, i))

        params[param_a] = i

  

        # Run CV

        cv_results = xgb.cv(

            params,

            dvalue,

            nfold =3,

            stratified=Skfold,

            early_stopping_rounds=early_stopping_rounds)

        

        # Update best result

        result_col = "test-" + metrics + "-mean"





        if small_better == True:

            result = cv_results[result_col].min()

            boost_rounds = cv_results[result_col].argmin()

            if result < result_best:

                result_best = result

                best_params = i

        else:

            result = cv_results[result_col].max()

            boost_rounds = cv_results[result_col].argmin()

            if result > result_best:

                result_best = result

                best_params = i

        if verbose:

            print("xgb.cv {} {} for {} rounds".format(metrics, result,  boost_rounds))

        

    print("Best xgb.cv params: {} {}, {}: {}".format(param_a, best_params, metrics, result_best))

    params[param_a] = best_params

    return(params)

def two_values(gridsearch_params,

               small_better,

               param_a,

               param_b,

               params,

               dvalue,

               metrics,

               early_stopping_rounds,

               Skfold=True,

               verbose=False):



    """This searches the best hyperparameter for the two selected hyperparameters

    



    Args:

        gridsearch_params: hyperparameter-values to be evaluated 

        small_better:  if small value is searched = true

        param_a: name of the first hyperparameter

        param_b: name of the second hyperparameter

        params: parameters dictionary

        dtrain: traindata

        metrics: the metric that defines the score

        num_boost_round: max evaluation round 

        Skfold: id used True

        verbose: how much data is showed    

    

    Returns:

        parameter dictionary



    """

    

    if small_better == True:

        result_best = float(999999)

    else:

        result_best = float(-999999)

    best_params = None



    for i, j in gridsearch_params:

        # Update our parameters

 

        if verbose:

            print("xgb.cv with {}={}, {}={}".format(param_a, i, param_b, j))

        params[param_a] = i

        params[param_b] = j

  

        # Run CV

        cv_results = xgb.cv(

            params,

            dvalue,

            nfold =3,

            stratified=Skfold,

            early_stopping_rounds=early_stopping_rounds)

       

        # Update best result

        result_col = "test-" + metrics + "-mean"

    



        if small_better == True:

            result = cv_results[result_col].min()

            boost_rounds = cv_results[result_col].argmin()

            if result < result_best:

                result_best = result

                best_params = (i,j)

        else:

            result = cv_results[result_col].max()

            boost_rounds = cv_results[result_col].argmax()

            if result > result_best:

                result_best = result

                best_params = (i,j)

        if verbose:

            print("xgb.cv {} {} for {} rounds".format(metrics, result, boost_rounds))

        

    print("Best xgb.cv params: {} {}, {} {}, {}: {}".format(param_a, best_params[0], param_b, best_params[1], metrics, result_best))

    

    params[param_a] = best_params[0]

    params[param_b] = best_params[1]

    return(params)
def hyperparameter_grid(params,

                        dtrain,

                        metrics,

                        watclist,

                        testing=True,

                        Skfold=False,

                        Verbose=False):

    

    """This function finds the optimum hyperparameters with a loop

    



    Args:

        params: parameter dictionary

        dtrain: data

        dtest: data

        metrics: the optimization metrics {'auc'}

        num_boost_round: the num_boost running value

        testing: sets variables lighter when true

        num_boost_round: max evaluation round 

        Skfold: id used True

        verbose: how much data is showed    

    

    Returns:

        trainde and optimized model

    """

    num_boost_round = 2000

    early_stopping_rounds = 10

    

    if Verbose == False:

        verbose_eval = num_boost_round

    else:

        verbose_eval = 10

    

    model = xgb.train(

        params,

        dtrain,

        verbose_eval=verbose_eval,

        num_boost_round = num_boost_round,

        early_stopping_rounds = early_stopping_rounds,

        evals=watchlist,

    )

    #for testing purposes a light set to save some time

    if testing:

        rounds=1

        print('testing')

        gridsearch_params_tree = [

            (i, j)

            for i in range(1,8)

            for j in range(1,5)

            ]

        gridsearch_params_0_1 = [i/5. for i in range(0,6)]

        gridsearch_params_0_1_deep = [i/5. for i in range(0,6)]

        gridsearch_params_gamma = [i/5. for i in range(0,26)]

        

        gridsearch_params_pair_0_1 = [

            (i0, i1)

            for i0 in gridsearch_params_0_1

            for i1 in gridsearch_params_0_1

            ]

    else: #for real

        rounds=3

        print('for real')

        gridsearch_params_tree = [

            (i, j)

            for i in range(1,20)

            for j in range(1,20)

            ]

        gridsearch_params_0_1 = [i/20. for i in range(0,21)]

        gridsearch_params_0_1_deep = [i/50. for i in range(0,51)]

        gridsearch_params_gamma = [i/50. for i in range(0,251)]

        gridsearch_params_pair_0_1 = [

            (i0, i1)

            for i0 in gridsearch_params_0_1_deep

            for i1 in gridsearch_params_0_1_deep

            ]

    

    dvalue = dtrain

    result_col = "test-" + metrics + "-mean"

    cv_results = xgb.cv(

            params,

            dvalue,

            stratified=Skfold,

            metrics= metrics

    )

   

    print("Start with xgb.cv params: {}: {}".format(metrics, cv_results[result_col].min()))





    #Tries to do semi-automatic genetic model for hyperparameter selection

    for round in range(rounds):

    

        #Maximum depth/height of a tree

        #Minimum sum of instance weight (hessian) needed in a child

        param_a = 'max_depth'

        param_b = 'min_child_weight'

        params=two_values(gridsearch_params_tree, True, param_a, param_b, params, dvalue, metrics, early_stopping_rounds,  Skfold, Verbose)





        #Gamma finds minimum loss reduction/min_split_loss required to make a further partition 

        param_a = 'gamma'

        paramns=one_value(gridsearch_params_gamma, True, param_a, params, dvalue, metrics, early_stopping_rounds,  Skfold, Verbose)

    



        #L1 regularization term on weights - alpha  - Lasso Regression 

        #adds “absolute value of magnitude” of coefficient as penalty term to the loss function.

        #L2 regularization term on weights - lambda  - Ridge Regression 

        #adds “squared magnitude” of coefficient as penalty term to the loss function.

        #the sample is so small, so most propably no effect

        param_a = 'lambda'

        param_b = 'alpha'

        params=two_values(gridsearch_params_pair_0_1 , True, param_a, param_b, params, dvalue, metrics, early_stopping_rounds, Skfold, Verbose)





        #Subsamble denotes the fraction of observations to be randomly samples for each tree.

        #Colsample_bytree enotes the fraction of columns to be randomly samples for each tree.

        param_a = 'colsample_bytree'

        param_b = 'subsample'

        params=two_values(gridsearch_params_pair_0_1, True, param_a, param_b, params, dvalue, metrics,early_stopping_rounds, Skfold, Verbose)

    

        #Same as learning_rate - this needs to be in sync with num_boost_round (alias n_tree parameter)

        param_a = 'eta'

        paramns=one_value(gridsearch_params_0_1_deep, True, param_a, params, dvalue, metrics,early_stopping_rounds,  Skfold, Verbose)

        

        #Balance of positive and negative weights.  This is regression and binary classification only parameter.

        if params['objective'].startswith('reg'):

            param_a = 'scale_pos_weight'

            paramns=one_value(gridsearch_params_0_1_deep, True, param_a, params, dvalue, metrics, early_stopping_rounds, Skfold, Verbose)





    print('Found hyperparameters with {} rounds '.format(round+1))

    print(params)

    print()

    

    model = xgb.train(

        params,

        dtrain,

        verbose_eval=verbose_eval,

        evals=watchlist,

        num_boost_round = num_boost_round,

        early_stopping_rounds = early_stopping_rounds,

        )

        

    num_boost_round = model.best_iteration + 1



    best_model = xgb.train(

        params,

        dtrain,

        num_boost_round=num_boost_round,

        verbose_eval=verbose_eval,

        evals=watchlist)

        

    return(best_model)
# The real work begins with reading the data



target = 'Survived'

#train = pd.read_csv('train.csv')

#test = pd.read_csv('test.csv')



train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



# Combine the train and test data into one file for some preprocessing

df = pd.concat([train, test])
#what kind of nulls are there

#cabin and age are worst

train.isnull().sum().sort_values(ascending=False)
test.isnull().sum().sort_values(ascending=False)
#Cabin has lots of null values

#Embarked and fare only a few
#Some adversialvalidation after feature enigeering between train and test



#Define which columns should be encoded vs scaled and what columns are binary

train_no_na = train.dropna().copy()

test_no_na = test.dropna().copy()

numeric_features  = ['Age', 'Pclass', 'Fare', 'Parch', 'SibSp']

train_scaled, test_scaled, features = scale_data(train_no_na, test_no_na, numeric_features  = numeric_features)



fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(17, 5))

ax1.set_ylim([0, 1.3])

ax1.set_title('Scaled train')

sns.kdeplot(train_scaled['Age'], ax=ax1)

sns.kdeplot(train['Pclass'], ax=ax1)

sns.kdeplot(train_scaled['Fare'], ax=ax1)

sns.kdeplot(train_scaled['Parch'], ax=ax1)

sns.kdeplot(test_scaled['SibSp'], ax=ax1)



ax2.set_title('Scaled test')

ax2.set_ylim([0, 1.3])

sns.kdeplot(test_scaled['Age'], ax=ax2)

sns.kdeplot(test['Pclass'], ax=ax2)

sns.kdeplot(test_scaled['Fare'], ax=ax2)

sns.kdeplot(test_scaled['Parch'], ax=ax2)

sns.kdeplot(test_scaled['SibSp'], ax=ax2)



plt.show()
df[df['Embarked'].isnull()]
#Embarked null values all to Southampton i.e. S value, was the largest embarkement harbor anyway 

column = "Embarked"

df[column].fillna(df[column].value_counts().index[0], inplace=True)
df['Ticket'].head(5)
#from ticket the ticket number and type can be separated



title = 'Ticket numbers'

column = 'Ticket'



def cleanTicket(ticket):

        ticket = ticket.upper()

        ticket = ticket.replace('.','')

        ticket = ticket.replace('/','')

        ticket = ticket.split()

        if len(ticket) >= 2:

            return pd.Series([ticket[0][:3], int(ticket[-1])])

        else:

            if ticket[0].isdigit():

                return pd.Series(["000", int(ticket[0])])

            else:

                return pd.Series(["000", 0])

        

category = column + "_type_categories"

df[category]=df[column].apply(cleanTicket)[0]



tlist = df[category].value_counts().head(2).index.tolist()

df.loc[df[category].isin(tlist) == False,category] = '000'



category = column + "_number"

df[category]=df[column].apply(cleanTicket)[1]
#Name to titles and to families

titles = {

        "Mr" :         "Mr",

        "Mme":         "Mrs",

        "Ms":          "Mrs",

        "Mrs" :        "Mrs",

        "Master" :     "Master",

        "Mlle":        "Miss",

        "Miss" :       "Miss",

        "Capt":        "Officer",

        "Col":         "Officer",

        "Major":       "Officer",

        "Dr":          "Officer",

        "Rev":         "Officer",

        "Jonkheer":    "Royalty",

        "Don":         "Royalty",

        "Sir" :        "Royalty",

        "Countess":    "Royalty",

        "Dona":        "Royalty",

        "Lady" :       "Royalty"

        }



column = "Name"

category = column + "_title_categories"

extracted_titles = df[column].str.extract(' ([A-Za-z]+)\.',expand=False)    

df[category] = extracted_titles.map(titles)



#There may be passengers in train and test dataset from the same lastname

#So the formulation of last names needs to be done before train/test splits

category = 'Lastname' 

df[category] = df[column].str.split(',').str[0]
#Separate sets

target = 'Survived'

train=df[(df[target].notnull())].copy()

test=df[(df[target].isnull())].copy()

#now on we will show training dataframe
pd.crosstab(train['Embarked'],train['Survived'], normalize='index').plot.bar(stacked=True, title='Embarkment')
pd.crosstab(train['Sex'],train['Survived'], normalize='index').plot.bar(stacked=True, title='Sex')
fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(17, 12))

Female = train[train['Sex']=='female']

Male = train[train['Sex']=='male']

ax = sns.distplot(Female[Female['Survived']==1].Age, label = "OK", ax = axes[0], kde =False)

ax = sns.distplot(Female[Female['Survived']==0].Age, label = "Dead", ax = axes[0], kde =False)

ax.set_title('F')

ax = sns.distplot(Male[Male['Survived']==1].Age, label = "OK", ax = axes[1], kde = False)

ax = sns.distplot(Male[Male['Survived']==0].Age, label = "Dead", ax = axes[1], kde = False)

ax.set_title('M')

plt.show()
#figure out and plot some info based on Pclass, Age and survival

fig, axes = plt.subplots(nrows=3, ncols=2, sharey=True,sharex=True, figsize=(14, 14))



ax = sns.distplot(Female[Female['Pclass']==1]['Age'], label = "1", ax = axes[0,0], kde =False)

ax = sns.distplot(Female[Female['Pclass']==2]['Age'], label = "2", ax = axes[0,0], kde =False)

ax = sns.distplot(Female[Female['Pclass']==3]['Age'], label = "3", ax = axes[0,0], kde =False)

ax.set_title('Female Pclass')

ax = sns.distplot(Male[Male['Pclass']==1]['Age'], label = "1", ax = axes[0,1], kde = False)

ax = sns.distplot(Male[Male['Pclass']==2]['Age'], label = "2", ax = axes[0,1], kde = False)

ax = sns.distplot(Male[Male['Pclass']==3]['Age'], label = "3", ax = axes[0,1], kde = False)

ax.set_title('Male Pclass')



ax = sns.distplot(Female[(Female['Pclass']==1) & (Female['Survived']==1)]['Age'], label = "1", ax = axes[1,0], kde =False)

ax = sns.distplot(Female[(Female['Pclass']==2) & (Female['Survived']==1)]['Age'], label = "2", ax = axes[1,0], kde =False)

ax = sns.distplot(Female[(Female['Pclass']==3) & (Female['Survived']==1)]['Age'], label = "3", ax = axes[1,0], kde =False)

ax.set_title('Female Pclass Survived')

ax = sns.distplot(Male[(Male['Pclass']==1) & (Male['Survived']==1)]['Age'], label = "1", ax = axes[1,1], kde = False)

ax = sns.distplot(Male[(Male['Pclass']==2) & (Male['Survived']==1)]['Age'], label = "2", ax = axes[1,1], kde = False)

ax = sns.distplot(Male[(Male['Pclass']==3) & (Male['Survived']==1)]['Age'], label = "3", ax = axes[1,1], kde = False)

ax.set_title('Male Pclass Survived')





ax = sns.distplot(Female[(Female['Pclass']==1) & (Female['Survived']==0)]['Age'], label = "1", ax = axes[2,0], kde =False)

ax = sns.distplot(Female[(Female['Pclass']==2) & (Female['Survived']==0)]['Age'], label = "2", ax = axes[2,0], kde =False)

ax = sns.distplot(Female[(Female['Pclass']==3) & (Female['Survived']==0)]['Age'], label = "3", ax = axes[2,0], kde =False)

ax.set_title('Female Pclass Not survived')

ax = sns.distplot(Male[(Male['Pclass']==1) & (Male['Survived']==0)]['Age'], label = "1", ax = axes[2,1], kde = False)

ax = sns.distplot(Male[(Male['Pclass']==2) & (Male['Survived']==0)]['Age'], label = "2", ax = axes[2,1], kde = False)

ax = sns.distplot(Male[(Male['Pclass']==3) & (Male['Survived']==0)]['Age'], label = "3", ax = axes[2,1], kde = False)

ax.set_title('Male Pclass Not survived')





plt.show()
pd.crosstab(df['Ticket_type_categories'],df['Survived'], normalize='index').plot.bar(stacked=True, title='Ticket type')
pd.crosstab(df['Ticket_type_categories'],df['Cabin'].str[0])
#The question is if this all should be done after train test split

#There may be passengers in train and test dataset from the same ticket or lastname

#So, the calculation of family size needs to be done before any train/test splitting

#Therefore, train and test will be joined again to one dataframe

df = pd.concat([train, test])
#count family size and create it as a categoric column from SibSp and Parch columns 



column1 = "SibSp"

column2 = "Parch"

category = "SibSP_Parch_family_categories"

middle_category_max = 4



df['SibSP_Parch_familysize'] = df[column1] + df[column2] + 1

df[category] = np.where(df['SibSP_Parch_familysize']==1,1, 

               np.where(df['SibSP_Parch_familysize'].between(1, middle_category_max, inclusive=True), 2, 3))    



df['Parch_family_or_not'] = np.where(df[column2]==0,0,1)



def calculate_familygroup(df,column):

    

    familysize = column + '_familysize'

    df[familysize] = df.groupby(column)[column].transform('count')

    category = column + "_family_categories"

    df[category] = np.where(df[familysize]==1,1, 

                np.where(df[familysize].between(1, middle_category_max, inclusive=True), 2, 3))  



    return(df)



df=calculate_familygroup(df,'Lastname').copy()

df=calculate_familygroup(df,'Ticket_number').copy()

#There may be passengers in train and test dataset from the same ticket

#Due that the Fare_per_person calculation needs to be done for the whole dataset



#Fare per person for same ticket holders. It is obvious that the price paid for all tickets

#is set to all ticket holders,so it needs to be set toright median/average vaues



df['Fare_per_person'] = df['Fare']/(df['Ticket_number_familysize']+1)

    

#Only one missing fare Storey, Mr. Thomas, 3rd class passanger

# Filling the missing value in Fare with the median Fare of 3rd class alone passenger

med_fare = df.groupby(['Pclass', 'Ticket_number_familysize'])['Fare_per_person'].median()[13]

df.loc[df['Fare'].isna(),'Fare_per_person'] = med_fare



#there are 0 fare passangers. They can be free riders or thr fare can be an error. In this case the best might be to set them look like other in the Pclass

med_fare = df.groupby(['Pclass', 'Ticket_number_familysize'])['Fare_per_person'].median()[0]

df.loc[(df['Fare']==0) & (df['Pclass']==1),'Fare_per_person'] = med_fare



med_fare = df.groupby(['Pclass', 'Ticket_number_familysize'])['Fare_per_person'].median()[7]

df.loc[(df['Fare']==0) & (df['Pclass']==2),'Fare_per_person'] = med_fare



med_fare = df.groupby(['Pclass', 'Ticket_number_familysize'])['Fare_per_person'].median()[13]

df.loc[(df['Fare']==0) & (df['Pclass']==3),'Fare_per_person'] = med_fare
pd.crosstab(train['Name_title_categories'],train['Survived'], normalize='index').plot.bar(stacked=False, title='Titles')
pd.crosstab(train['Name_title_categories'],train['Survived'])
#The question is if this all should be done after train test split

#There may be passengers in train and test dataset from the same ticket or lastname



def find_families(df):

    #this function finds lastnames and if they have the same fare_per_person price

    #they are considered to be one family. Pclass is not enough as a separator

    

    values = []

    for (name, group) in df.groupby('Lastname'):

        #common_pclass finds the Pclass that has most values in this Lastname group

        same_fare = group['Fare_per_person'].value_counts().keys().tolist()[0]



        #if the person is not in this group add it to values list

        values.extend(group[group['Fare_per_person'] != same_fare]['PassengerId'].to_list())

    return(values)



values = find_families(df)

while len(values) > 0:

    #if same lastname add_x to the end

    df.loc[df['PassengerId'].isin(values), 'Lastname'] = df[df['PassengerId'].isin(values)]['Lastname'].apply(lambda x: x + "_X")

    values = find_families(df)
#Separate all again to train and test

target = 'Survived'

train=df[(df[target].notnull())].copy()

test=df[(df[target].isnull())].copy()
pd.crosstab(train[train['Ticket_number_familysize']<=3]['Ticket_number_familysize'],train[train['Ticket_number_familysize']<=3]['SibSP_Parch_familysize'], normalize='index').plot.bar(stacked=False, title='Family size')
pd.crosstab(train[train['Lastname_familysize']<=3]['Lastname_familysize'],train[train['Lastname_familysize']<=3]['SibSP_Parch_familysize'], normalize='index').plot.bar(stacked=False, title='Family size')
pd.crosstab(train[train['Ticket_number_familysize']>3]['Ticket_number_familysize'],train[train['Ticket_number_familysize']>3]['SibSP_Parch_familysize'], normalize='index').plot.bar(stacked=False, title='Family size')
pd.crosstab(train[train['Lastname_familysize']>3]['Lastname_familysize'],train[train['Lastname_familysize']>3]['SibSP_Parch_familysize'], normalize='index').plot.bar(stacked=False, title='Family size')
#https://en.wikipedia.org/wiki/Master_(form_of_address)

#it is possible to set some average value to masters without age



fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(17, 7))

master = train[train['Name_title_categories']=='Master']

ax = sns.distplot(master[master['Survived']==1].Age, label = "OK", kde =False)

ax = sns.distplot(master[master['Survived']==0].Age, label = "Dead", kde =False)

ax.set_title('Master / Survival')

plt.show()
print(train[(train['Name_title_categories']=='Master') & (train['Age']<=18)][['Parch', 'Pclass','Survived']].describe()[0:3])

print(train[(train['Name_title_categories']=='Mr') & (train['Age']<=18)][['Parch', 'Pclass','Survived']].describe()[0:3])
print(train[(train['Name_title_categories']=='Miss') & (train['Age']<=18)][['Parch', 'Pclass','Survived']].describe()[0:3])

print(train[(train['Name_title_categories']=='Mrs') & (train['Age']<=18)][['Parch', 'Pclass','Survived']].describe()[0:3])
fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(17, 12))

Master = train[(train['Name_title_categories']=='Master') & (train['Age']<=18)]

Mr = train[(train['Name_title_categories']=='Mr') & (train['Age']<=18)]

ax = sns.distplot(Master[Master['Survived']==1].Age, label = "OK", ax = axes[0], kde =False)

ax = sns.distplot(Master[Master['Survived']==0].Age, label = "Dead", ax = axes[0], kde =False)

ax.set_title('F')

ax = sns.distplot(Mr[Mr['Survived']==1].Age, label = "OK", ax = axes[1], kde = False)

ax = sns.distplot(Mr[Mr['Survived']==0].Age, label = "Dead", ax = axes[1], kde = False)

ax.set_title('M')

plt.show()
grouped_train = df.groupby(['Parch_family_or_not', 'Name_title_categories']).agg(['min', 'max', 'median', 'std', 'count'])



grouped_median_train = grouped_train

grouped_median_train = grouped_median_train.reset_index()[['Parch_family_or_not', 'Name_title_categories', 'Age']]

print(grouped_median_train)
#Predict age

target = 'Age'

numeric_features = []

categorical_features = ['Pclass', 'Name_title_categories','Parch_family_or_not']



features = numeric_features + categorical_features



#get data from original a bit preprocessed dataframe

y = df[df[target].notnull()][target].copy()

train = df[df[target].notnull()][features].copy()

test  = df[df[target].isnull()][features].copy()





#set values to feature engineering attributes



#Check

#k_selected need to have correct values, if they are too large, evrything needs to be started again

#'all' means all

k_selected = 'all'

test_size = 0.2



#Scalers attributes

scaler = MinMaxScaler()

#set values to model configuration attributes

kbest_score_func = mutual_info_regression 





metric = 'rmse'

Skfold=False

Verbose = False

testing=False



params = {

    #Initial xgboost parameters to be automatically tuned

    'objective':'reg:squarederror',

    'booster' : 'gbtree',

    'eval_metric' : metric

    } 

#prepare data scale, onehot encoding etc.

tr_train, tr_test, features = scale_data(train, test, numeric_features, categorical_features, scaler)



features_df = select_kbest(tr_train[features], y, kbest_score_func,k_selected)

selected_columns =  features_df[features_df['Selected']]['Column'].tolist()

tr_train = tr_train[selected_columns].copy()

tr_test = tr_test[selected_columns].copy()



X_train, X_test, y_train, y_test = train_test_split(tr_train, y, test_size=test_size) 

dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)

dtest = xgb.DMatrix(X_test, label=y_test)

dtvalues = xgb.DMatrix(tr_test)



watchlist = [(dtrain, 'train'), (dtest, 'test')]
#draw feature importance based on kBest

title = 'Features kbest ' + target

draw_features(features_df[features_df['Selected']==True]['Score'], features_df[features_df['Selected']==True]['Column'], title)
#configure and finetune model

model = hyperparameter_grid(params, dtrain, metric, watchlist, testing, Skfold, Verbose)    
#draw feature importance based on xgboost

xgb.plot_importance(model)
draw_true_vs_predicted(dtest, y_test, model, target)
#Impute the new found values of age

train = df[df[target].notnull()].copy()

test  = df[df[target].isnull()].copy()

df=predict_and_store(test, train, model, dtvalues)
child_age = 16

dip1=22

dip2=35

dip3=42





def get_person(person):

    age = person[0]

    sex = person[1]

    name_title = person[2]

    if (age <= child_age) & (name_title != 'Mr'): 

        val = 'Child'

    if (age <= child_age) & (name_title == 'Mr'): 

        val = 'Male_adult_young'

    if (age > child_age) & (age <= dip1) & (sex == 'male'): 

        val = 'Male_adult_young'

    if (age > dip1) & (age <= dip2) & (sex == 'male'): 

        val = 'Male_adult_middle'

    if (age > dip2) & (age <= dip3) & (sex == 'male'): 

        val = 'Male_adult_middle'

    if (age > dip3) & (sex == 'male'): 

        val = 'Male_adult_old'

    if (age > child_age) & (age <= dip1) & (sex == 'female'): 

        val = 'Female_adult_young'

    if (age > dip1) & (age <= dip2) & (sex == 'female'): 

        val = 'Female_adult_middle'

    if (age > dip2) & (age <= dip3) & (sex == 'female'): 

        val = 'Female_adult_middle'

    if (age > dip3) & (sex == 'female'): 

        val = 'Female_adult_old'

    return val

df['Family_title'] = df[['Age', 'Sex', 'Name_title_categories']].apply(get_person, axis=1)



#Separate all again to train and test

target = 'Survived'

train=df[(df[target].notnull())].copy()

test=df[(df[target].isnull())].copy()





pd.crosstab(train[train['Pclass']==1]['Family_title'],train[train['Pclass']==1]['Survived']).plot.bar(stacked=False, title='Titles/Pclass 1')

pd.crosstab(train[train['Pclass']==2]['Family_title'],train[train['Pclass']==2]['Survived']).plot.bar(stacked=False, title='Titles/Pclass 2')

pd.crosstab(train[train['Pclass']==3]['Family_title'],train[train['Pclass']==3]['Survived']).plot.bar(stacked=False, title='Titles/Pclass 3')



def create_groupvalue(df, column, mask, col_name):

    #create new columm col_name based on calculations of a column like lastname

    #based on conditions described in mask



    

    temp_group_df = pd.DataFrame(df[column].value_counts())    

    temp_group_df.reset_index(inplace=True)

    temp_group_df.columns=[column, 'number']

    temp_group_df.drop('number', axis=1, inplace=True)

    temp_group_df.sort_values(column).reset_index(drop=True, inplace=True)



    col_name = column + '_' + col_name



    temp_df = pd.DataFrame(df[column][mask].value_counts()).reset_index()

    temp_df.columns=[column, col_name]

    temp_df.sort_values(column).reset_index(drop=True, inplace=True)

    

    temp_group_df = pd.merge(temp_group_df, temp_df, on=column,how='left')

    temp_group_df[col_name].fillna(0, inplace=True)

    temp_group_df.loc[temp_group_df[col_name] > 0, col_name] = 1

     

    df=pd.merge(df, temp_group_df, on=column,how='left')

    df.reset_index(drop=True, inplace=True)

    return(df)



def create_familygroup(df,column):

    #create calculated colums

    familysize = column + '_familysize'

    familysize = 'SibSP_Parch_familysize'

    



    mask = (df['Family_title'].str.startswith('Female_adult')) & (df['Survived'] == 1) & (df[familysize] > 1)

    df=create_groupvalue(df, column, mask, 'Survived_Female').copy()



    mask = (df['Family_title'].str.startswith('Male_adult')) & (df['Survived'] == 1) & (df[familysize] > 1)

    df=create_groupvalue(df, column, mask, 'Survived_Male').copy()



    mask = (df['Family_title'] == 'Child') & (df['Survived'] == 1) & (df[familysize] > 1)

    df=create_groupvalue(df, column, mask, 'Survived_Child').copy()



    return(df)



df =create_familygroup(df,'Lastname').copy()

df =create_familygroup(df,'Ticket_number').copy()



#Separate all again to train and test

target = 'Survived'

train=df[(df[target].notnull())].copy()

test=df[(df[target].isnull())].copy()



pd.crosstab(train['Ticket_number_Survived_Female'],train['Ticket_number_Survived_Male']).plot.bar(stacked=False, title='Same ticket')

pd.crosstab(train['Ticket_number_Survived_Female'],train['Ticket_number_Survived_Child']).plot.bar(stacked=False, title='Same ticket')

pd.crosstab(train['Ticket_number_Survived_Male'],train['Ticket_number_Survived_Female']).plot.bar(stacked=False, title='Same ticket')

pd.crosstab(train['Ticket_number_Survived_Child'],train['Ticket_number_Survived_Female']).plot.bar(stacked=False, title='Same ticket')
pd.crosstab(train['Cabin'].str[0],train['Survived']).plot.bar(stacked=False, title='Cabin')

#find values for deck for customers having already cabin number and customers without it

column = "Cabin" 

target = column  + "_deck" #target





#deck =  {"A": 1, "B": 2, "C": 2, "D": 3, "E": 3, "F": 4, "G": 4, "Y": 0, "T": 1}



deck =  {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "Y": 0, "T": 1}



#So few in class G, combine it with F



#Figure out Cabin categories, fill Na to "Y", it will always be the min value

#if category not available find it from other members of the family, if possible

df[column] = df[column].fillna("Y")

df[target] = df[column].str[0].map(deck)



numeric_features = ['Fare_per_person']

categorical_features = ['Pclass',  'Ticket_type_categories']

features = numeric_features + categorical_features 



#get data from original a bit preprocessed dataframe

#note now the empty value is 0

y = df[df[target]!=0][target].copy()

train = df[df[target]!=0][features].copy()

test  = df[df[target]==0][features].copy()





#set values to feature engineering attributes



#Check

#k_selected need to have correct values, if they are too large, evrything needs to be started again

k_selected = 'all'



test_size = 0.3



#Scalers attributes

scaler = MinMaxScaler()

kbest_score_func = mutual_info_classif

#set values to model configuration attributes



num_class = len(df['Cabin_deck'].unique()) + 1 #how many classes we are working with



metric = 'mlogloss'

Skfold=False

Verbose = False

testing=False



params = {

    # Parameters that we are going to tune.

    'objective':'multi:softmax',

    'num_class' : num_class,

    'booster' : 'gbtree',

    'eval_metric' : metric

} 

#prepare data scale, onehot encoding etc.

tr_train, tr_test, features = scale_data(train, test, numeric_features, categorical_features, scaler)



features_df = select_kbest(tr_train[features], y, kbest_score_func,k_selected)

selected_columns =  features_df[features_df['Selected']]['Column'].tolist()

tr_train = tr_train[selected_columns].copy()

tr_test = tr_test[selected_columns].copy()



#split the initial train dataframe to test/train dataframes

X_train, X_test, y_train, y_test = train_test_split(tr_train, y, test_size=test_size) 



dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)

dtest = xgb.DMatrix(X_test, label=y_test)

dtvalues = xgb.DMatrix(tr_test)



watchlist = [(dtrain, 'train'), (dtest, 'test')]
#draw feature importance based on kBest

title = 'Features kbest ' + target

draw_features(features_df[features_df['Selected']==True]['Score'], features_df[features_df['Selected']==True]['Column'], title)
#configure and finetune model

model = hyperparameter_grid(params, dtrain,metric, watchlist, testing, Skfold, Verbose)    
#draw feature importance based on xgboost

xgb.plot_importance(model)
draw_true_vs_predicted(dtest, y_test, model, target)
y_pred = model.predict(dtest)

lower, upper = proportion_confint(accuracy_score(y_test, y_pred) * len(y_test), len(y_test), 0.05)

print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))

print("F1 Score: ", f1_score(y_test, y_pred, average="macro"))

print("Precision Score: ", precision_score(y_test, y_pred, average="macro"))

print("Recall Score: ", recall_score(y_test, y_pred, average="macro"))

print('Confidence level lower=%.3f, upper=%.3f' % (lower, upper))
#Impute the new found values of cabin

train = df[df[target]!=0].copy()

test  = df[df[target]==0].copy()

df=predict_and_store(test, train, model, dtvalues)





deck =  {1 : "A", 2 : "B", 3 : "C", 4 : "D", 5 : "E", 6 : "F", 7 : "G"}

df[target] = df[target].map(deck)
#Some adversialvalidation after feature enigeering phase between train and test



#Define which columns should be encoded vs scaled and what columns are binary



#Separate all again to train and test

target = 'Survived'

train=df[(df[target].notnull())].copy()

test=df[(df[target].isnull())].copy()





numeric_features = ['Age', 'Fare_per_person', 'Lastname_familysize']



train_scaled, test_scaled, features = scale_data(train, test, numeric_features=numeric_features)



fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(17, 5))

ax1.set_ylim([0, 2])

ax1.set_title('Scaled train')

sns.kdeplot(train_scaled['Fare_per_person'], ax=ax1)

sns.kdeplot(train['Pclass'], ax=ax1)

sns.kdeplot(train_scaled['Age'], ax=ax1)

sns.kdeplot(train_scaled['Lastname_familysize'], ax=ax1)



ax2.set_title('Scaled test')

ax2.set_ylim([0, 2])

sns.kdeplot(test_scaled['Fare_per_person'], ax=ax2)

sns.kdeplot(test['Pclass'], ax=ax2)

sns.kdeplot(test_scaled['Age'], ax=ax2)

sns.kdeplot(test_scaled['Lastname_familysize'], ax=ax2)



plt.show()
#Predict survival through xgboost machine learning algorithms

target = 'Survived'

numeric_features = []

categorical_features = ['Family_title','Pclass','Ticket_number_family_categories','Embarked','Parch_family_or_not', 'Cabin_deck']

features = numeric_features + categorical_features



#get data from orginal  a bit preprocessed dataframe

y = df[df[target].notnull()][target].copy()

train = df[df[target].notnull()][features].copy()

test  = df[df[target].isnull()][features].copy()



#set values to feature engineering attributes 





#Check

#k_selected need to have correct values, if they are too large, evrything needs to be started again

k_selected = 'all'

test_size = 0.20

scaler = MinMaxScaler()

kbest_score_func = mutual_info_classif



num_boost_round = 2000

metric = 'error'

Skfold=True

Verbose = False

testing=False



#Set values for model configuration

params = {

    # Initial xgboost parameters to be tuned

    'objective':'binary:logistic',

    'booster' : 'gbtree',

    'eval_metric' : metric

} 
#prepare data scale, onehot encoding etc.

tr_train, tr_test, features = scale_data(train, test, numeric_features, categorical_features, scaler)



features_df = select_kbest(tr_train[features], y, kbest_score_func,k_selected)

selected_columns =  features_df[features_df['Selected']]['Column'].tolist()

tr_train = tr_train[selected_columns].copy()

tr_test = tr_test[selected_columns].copy()



X_train, X_test, y_train, y_test = train_test_split(tr_train, y, test_size=test_size) 



dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test, label=y_test)

dtest = xgb.DMatrix(X_test, label=y_test)

dtvalues = xgb.DMatrix(tr_test)



watchlist = [(dtrain, 'train'), (dtest, 'test')]
#draw feature importance based on kBest

title = 'Features kbest ' + target

draw_features(features_df[features_df['Selected']==True]['Score'], features_df[features_df['Selected']==True]['Column'], title)
#configure and finetune model

model = hyperparameter_grid(params, dtrain, metric, watchlist, testing, Skfold, Verbose)    
#draw feature importance based on xgboost

xgb.plot_importance(model)
draw_true_vs_predicted(dtest, y_test, model, target, binarize=True)
draw_results(dtest, y_test, model)
draw_confusion(dtest, y_test, model)
train = df[df[target].notnull()].copy()

test  = df[df[target].isnull()].copy()

submission=create_submission(test,model, dtvalues)
submission.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")