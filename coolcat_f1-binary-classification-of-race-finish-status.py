import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from itertools import groupby
import os

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, make_scorer, accuracy_score, precision_score, average_precision_score, \
classification_report, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import KFold 
from sklearn.base import clone

from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE

pd.options.mode.chained_assignment = None 

%config InlineBackend.figure_format = 'retina'
%matplotlib inline

sns.set(style='white', context='notebook', font_scale=1.5)

import warnings
warnings.filterwarnings("ignore")
index_list = ['year', 'name', 'driverRef']
target_var_list = ['statusId']

# List of Formula 1 races in a season in chronological order
races = ['Australian Grand Prix','Chinese Grand Prix','Bahrain Grand Prix','Russian Grand Prix','Spanish Grand Prix','Monaco Grand Prix',
         'Canadian Grand Prix', 'Azerbaijan Grand Prix', 'Austrian Grand Prix', 'British Grand Prix', 'Hungarian Grand Prix',
         'Belgian Grand Prix', 'Italian Grand Prix', 'Singapore Grand Prix', 'Malaysian Grand Prix', 'Japanese Grand Prix',
         'United States Grand Prix', 'Mexican Grand Prix', 'Brazilian Grand Prix', 'Abu Dhabi Grand Prix']

pirelli = pd.read_csv("../input/formula-1-race-finish-status/Pirelli_Tyre_Categories.csv")
pirelli = pirelli[pirelli['name'] != 'German Grand Prix']

# Important to ensure dataframe is sorted by the F1 race calendar in chronological order to ensure dataset is filtered accurately.
sorterIndex = dict(zip(races,range(len(races))))
pirelli['name_Rank'] = pirelli['name'].map(sorterIndex)
pirelli.sort_values(['year','name_Rank'], ascending = True, inplace = True) 
pirelli.drop('name_Rank', 1, inplace = True)
races_dict = pirelli[['year', 'name']].to_dict('list')
# Ensure all classifiers use the same random state
rs=12

clfs = [
    ['Extra Trees',  ExtraTreesClassifier(random_state=rs)],
    ['Random Forest', RandomForestClassifier(random_state=rs, class_weight=None)],
    ['Gradient Boosting', GradientBoostingClassifier(random_state=rs)],
    ['Logistic Regression', LogisticRegression(random_state = rs)],
    ['KNN', KNeighborsClassifier()],
    ['SVC', SVC(random_state=rs, probability=True)],
    ['LDA', LinearDiscriminantAnalysis()],
    ['MLP', MLPClassifier(random_state=rs)],
]

meta_learner_GB = ['Gradient Boosting', GradientBoostingClassifier(random_state=rs, learning_rate=0.01, n_estimators=1000)]

kfold_3 = KFold(n_splits=3)
class Build_train_test_set():
    
    def __init__(self, actual, df, df_test, name, index_list=index_list, target_var_list=target_var_list, 
                 races_dict=races_dict, scaler='StandardScaler'): 

        """
        Splits dataset to a train and test set  
        A train and test set are input variables, but this function allows both sets to be scoped down further according to criteria.
        
        2 methods of train-test split
           Approach 1: Train-test split by year 
               - When initializing class, set name = None
               - This is not a viable options if model includes features that are only known pre-race. eg drivers' selected tyre sets and qualifying position)
           Approach 2: Train-test split by races
               - When initializing class, specify the name variable with a race

           Notes:
           -If you want to filter only races that belong to same category of pirelli assigned tyre combis, ensure the races_dict variable itself only contains the races that fall within the category
        
        df - dataframe containing train set
        df_test - dataframe containing test set
        train_yr - specify the scope of the training set. Format should be a list eg. [2015, 2016]
                 - For eg, if train_yr=[2016], then only 
        test_yr - specify the scope of the test set. Format should be a list eg. [2017]
        qty_races_tofilter - specify the method to scope down train set further with only selected races.
                             For eg, if input is 4, then only the 4 previous races before the currnt race to be tested on will be included in training set. 
        races_dict - A dictionary of the F1 race calendar for multiple seasons (the year of the season and the corresponding races)
        name - race to be tested on
        
        Example of parameter input:
        train_yr = [2015, 2016], test_yr = [2017],  qty_races_tofilter= 4, name = 'Singapore Grand Prix"
        scope of races included in train set: 
         - 2015 Singapore Grand Prix
         - 2016 Singapore Grand Prix
         - 2017 British, Hungarian, Belgian, Italian Grand Prix (These are the 4 races that took place before the 2017 Singapore GP)
        """    
        self.df = df
        self.df_test = df_test       
        self.index_list = index_list
        self.target_var_list = target_var_list
        self.race_dict = races_dict
        self.name = name
        self.scaler = scaler
        self.actual = actual
        
    def train_test_split(self, train_yr, test_yr, qty_races_tofilter):

        train_set = self.df[self.df['year'].isin(train_yr)].reset_index(drop=True) # Scope down training set by year, if required.

        if self.name != None:

            if (isinstance(qty_races_tofilter, str)) or (isinstance(qty_races_tofilter, float)):
                raise ValueError('qty_races_tofilter variable can only be an integer, None or 0. If None is the input, then all races in train set will be included. If 0/False is the input, only races of the same name are selected.')

            elif (isinstance(qty_races_tofilter, int)):
                r = pd.DataFrame(races_dict['name'])
                
                index = r[r[0] == self.name].index.tolist()[-1] # Only extract the index number of the current race. Since the df is already sorted in chronological order, [-1] picks out the index of the race to be tested on.
                train_set = pd.DataFrame()
                # Extract a pool of past races to include in training set
                for k,v in zip(races_dict['name'][index-qty_races_tofilter:index], races_dict['year'][index-qty_races_tofilter:index]):
                    f = self.df[(self.df['year'] == v) & (self.df['name'] == k)]
                    train_set = pd.concat([train_set, f])

                addto_train_set = self.df[(self.df['year'].isin(train_yr)) & (self.df['name']== self.name)].reset_index(drop=True)
                train_set = train_set.append(addto_train_set)

            elif (qty_races_tofilter==None):
                # Extract all past races to include in training set
                races_list = [x for i, x in enumerate(races_dict['name']) if races_dict['name'].index(x) == i] # List of Formula 1 races in a season in chronological order
                target_ibdex = races_list.index(self.name)
                races_before = races_list[:target_ibdex]
                addto_train_set = self.df[(self.df['year'].isin(test_yr)) & (self.df['name'].isin(races_before))].reset_index(drop=True) 
                train_set = train_set.append(addto_train_set)

            test_set = self.df_test[self.df_test['name'] == self.name].reset_index(drop=True) # Only select the test set of the race to test on
 
        elif self.name == None:
            test_set = self.df_test

        train_set = train_set.reset_index(drop=True)
        test_set = test_set.reset_index(drop=True)
        
        # Separate index, features and target variable
        learning_columns = np.setdiff1d(train_set.columns, self.index_list+self.target_var_list)
        X_train = train_set.loc[:, learning_columns]
        X_test = test_set.loc[:, learning_columns]
        Y_train = np.array(train_set[self.target_var_list[0]]).ravel()

        # Apply a scaler on data
        Xs_train, Xs_test = self.scale_data(X_train, X_test, self.scaler)
        
        if self.actual==False:
            Y_test = np.array(test_set[self.target_var_list[0]]).ravel()
        else:
            Y_test = []
            
        return train_set, test_set, Xs_train, Xs_test, Y_train, Y_test

    def scale_data(self, X_train, X_test, scaler):
    
        if scaler == 'StandardScaler':
            SS = StandardScaler()
            Xs_train = SS.fit_transform(X_train)
            Xs_test = SS.fit_transform(X_test)
        elif scaler == 'MinMaxScaler':
            mm = MinMaxScaler()
            Xs_train = mm.fit_transform(X_train)
            Xs_test = mm.fit_transform(X_test)  
        elif scaler == False:
            Xs_train = X_train
            Xs_test = X_test

        return Xs_train, Xs_test
    
class ClassifyOneClf:
    
    def __init__(self, name, train_set, test_set, Xs_train, Xs_test, Y_train, Y_test, clf, generator=kfold_3,
                 index_list=index_list, target_var_list=target_var_list):
        
        """
        This is a general purpose class containing functions enabling predictions on an array of features
        using a single classifier, with classification results and classifier performance statistics stored in dataframes.

        Parameters:
        classifier - Must be in format[title, estimator()] eg: ['Random Forest', RandomForestClassifier()]
        generator - Cross-validator method to split data in train/test sets.
        train_set - dataframe of training set (including index and target variable columns)
        test_set  - dataframe of test set (including index and target variable columns)
        Xs_train - array of training set features
        Xs_test - array of test set features
        Y_train - array of training set actual target variable values
        Y_test - array of test set actual target variable values
        index_list - row identification (eg. year=2016, driverRef)
        target_var_list - list containing target variables
        """
        self.name = name
        self.train_set = train_set
        self.test_set = test_set
        self.Xs_train = Xs_train 
        self.Xs_test = Xs_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.clf = clf
        self.generator = generator
        self.index_list = index_list
        self.target_var_list = target_var_list
        
    def feature_selection(self, fsel_list):
        """
        Creates a feature-selection-classifier pipeline
        
        classifier - Input variable must be in the format: ['LDA', LinearDiscriminantAnalysis()]
        name - Row identification (eg. Australian Grand Prix)
        fsel_list - pass a list of the number of features to gridsearch (eg. [3,4,5])
        Note: Feature selection can be computationally expensive, be careful when setting the param grid before running the function.
        """
        sfs = SequentialFeatureSelector(self.clf[1],
                                        k_features=3,
                                        forward=False, 
                                        floating=False, 
                                        scoring='roc_auc',
                                        verbose=0,
                                        cv=3)

        pipe = Pipeline([('sfs', sfs),
                         (self.clf[0], self.clf[1])
                        ])

        param_grid = [
          {'sfs__k_features': fsel_list}
        ]

        gs = GridSearchCV(estimator=pipe, 
                          param_grid=param_grid, 
                          scoring='roc_auc', 
                          n_jobs=-1, 
                          cv=3,  
                          refit=True)

        # run gridearch
        gs = gs.fit(self.Xs_train, self.Y_train)
        feature_subset = gs.best_estimator_.steps[0][1].k_feature_idx_
        Xs_train_sfs = self.Xs_train[:, feature_subset]
        Xs_test_sfs = self.Xs_test[:, feature_subset]

        df_fea_sel = pd.DataFrame({"Method":self.clf[0], "Index": self.name, 'Best score:': gs.best_score_,
                                  'Best features:': [feature_subset]})
        
        return df_fea_sel, Xs_train_sfs,  Xs_test_sfs
    

    def VIF_filter(self):
        """
        Check for Variance Inflation Factor of features in train set 
        """
        # Convert array to dataframe so that it can be passed to the vif function
        df = pd.DataFrame(self.Xs_train)
        df_test = pd.DataFrame(self.Xs_test)
        
        # For each X, calculate VIF and save in dataframe
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        vif["features"] = df.columns

         # Final list columns that have passed the check
        features_vif_below10 = vif[vif['VIF Factor'] < 10].features.tolist()

        # Final dataframe of selected columns
        df = df[features_vif_below10]
        df_test = df_test[features_vif_below10]
        
        #print 'Columns which pass VIF test:' + str(features_vif_below10)

        return df, df_test

    
    def cross_validate_one_clf(self):
        """
        Perform cross-validation on train set using a classifier.
        """
        cv_results = [] 
        proba_train = pd.DataFrame()
        
        for i, (train_idx, test_idx) in enumerate(self.generator.split(self.Xs_train, self.Y_train)):

            fold_xtrain, fold_ytrain = self.Xs_train[train_idx, :], self.Y_train[train_idx]
            fold_xtest, fold_ytest = self.Xs_train[test_idx, :], self.Y_train[test_idx]
            classifier = self.fit(fold_xtrain, fold_ytrain) # Train classifier on each fold

            # Generate prediction probabilites of each fold, which is then stacked to form full df of train set predictions 
            fold_Pa = classifier.predict_proba(fold_xtest)
            cv_proba = self.format_pred_df(fold_Pa[:,1], self.train_set.loc[test_idx,:])
            proba_train = pd.concat([proba_train, cv_proba])
        
        # Quick way of aggregating cross validation results for all folds
        cv_results.append(cross_val_score(classifier, self.Xs_train, self.Y_train, scoring = "accuracy", cv = self.generator, n_jobs=-1))
        cv_means = np.mean(cv_results)
        cv_std = np.std(cv_results)
        
        # Store cross validation reuslts in dataframe
        cv_stats = pd.DataFrame({'Index': self.name, 'Target Var': self.target_var_list[0], 'Method': self.clf[0], 
                                 "CrossValMeans":cv_means,"CrossValerrors": cv_std}, index=[0])

        cv_stats = cv_stats[['Index', 'Target Var', 'Method', "CrossValMeans", "CrossValerrors"]]
        
        return cv_stats, proba_train

    
    def fit(self, Xs_train, Y_train):
        """
        Fit the model using X as array of features and y as array of labels.
        """
        # Clone does a deep copy of the model in an estimator without actually copying attached data. 
        # It yields a new estimator with the same parameters that has not been fit on any data.
        classifier = clone(self.clf[1])
        classifier.fit(Xs_train, Y_train) 
        return classifier
   

    def predict_with_one_clf(self, Xs_train, Xs_test, test_set):
        """
        Generate predictions on test set using a classifier.
        """
        classifier = self.fit(Xs_train, self.Y_train)
        y_pred = classifier.predict(Xs_test) 
        y_proba = classifier.predict_proba(Xs_test) 
        
        pred = self.format_pred_df(y_pred, test_set)
        proba = self.format_pred_df(y_proba[:,1], test_set)

        if self.Y_test != []:
            results_stats = calc_classification_stats(self.name, self.target_var_list, self.clf, self.Y_test, y_pred, average='binary')
        else:
            results_stats = pd.DataFrame()
 
        return results_stats, pred, proba

    
    def format_pred_df(self, y, dataset):
        """
        Format array of predictions to dataframe
        """
        dataset = dataset[self.index_list+self.target_var_list].reset_index(drop=True)
        pred = pd.DataFrame(y, columns=[self.clf[0] + "_" + str(self.target_var_list[0])])
        dataset = pd.merge(dataset, pred, left_index=True, right_index=True) # Concat indexes to df of predicted results
        return dataset
  
class ClassifyOneRace:
    
    def __init__(self, name, train_set, test_set, Xs_train, Xs_test, Y_train, Y_test, methods, generator=kfold_3, 
                 index_list=index_list, target_var_list=target_var_list):   
        """
        This is a general purpose class containing functions enabling predictions on an array of features
        using a list of classifiers, with classification results and classifier performance statistics
        of each classifier stored in dataframes.

        methods - A nested list of classifiers. 
        """
        self.name = name
        self.train_set = train_set
        self.test_set = test_set
        self.Xs_train = Xs_train 
        self.Xs_test = Xs_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.methods = methods
        self.generator = generator
        self.index_list = index_list
        self.target_var_list = target_var_list
        
    def cross_validate_one_race(self):
        """
        Generate cross-validated predictions on train set by iterating through multiple classifiers 
        """
        Pa_train_race = pd.DataFrame()
        results_all = pd.DataFrame() 
    
        # Iterate through each classifier
        for m in self.methods: 
            # Initialize class to gain access to singular clf specific functions
            c1 = ClassifyOneClf(self.name, self.train_set, self.test_set, self.Xs_train, self.Xs_test, self.Y_train, self.Y_test, m)
            results, proba_train = c1.cross_validate_one_clf()
            results_all = pd.concat([results_all, results]) # Df containing cross-validation performance results of a list of base learners 
            Pa_train_race = self.format_prediction_matrix(self.train_set, Pa_train_race, proba_train, all_cols=False)
        
        return results_all, Pa_train_race
    
    
    def predict_one_race(self, fsel_list):
        """
        Generate test set predictions for one race by iterating through multiple classifiers 
        """
        Pa_test_race = pd.DataFrame()
        P_test_race = pd.DataFrame()     
        results_all = pd.DataFrame()
        fsel_results_all = pd.DataFrame()
    
        # Iterate through each classifier
        for m in self.methods:
            c1 = ClassifyOneClf(self.name, self.train_set, self.test_set, self.Xs_train, self.Xs_test, self.Y_train, self.Y_test, m)
            # For linear discriminant analysis, there is a need to ensure variables are not collinear
            if (m[0] == "LDA"):
                df_train_lda, df_test_lda = c1.VIF_filter()
                results, pred, proba = c1.predict_with_one_clf(np.array(df_train_lda), np.array(df_test_lda), self.test_set)
            if fsel_list!=False:
                # Perform feature selection for trees-based classifiers only.
                if (m[0] != "MLP") and  (m[0] != "LDA"):
                    fsel_results, Xs_train_sfs, Xs_test_sfs = c1.feature_selection(fsel_list)
                    fsel_results_all = pd.concat([fsel_results_all, fsel_results]) # Df containing feature selection results 
                    results, pred, proba = c1.predict_with_one_clf(Xs_train_sfs, Xs_test_sfs, self.test_set)
                else:
                    results, pred, proba = c1.predict_with_one_clf(self.Xs_train, self.Xs_test, self.test_set)
            else:
                results, pred, proba = c1.predict_with_one_clf(self.Xs_train, self.Xs_test, self.test_set)
            
            results_all = pd.concat([results_all, results]) # Df containing performance statistics of a list of base learners 
            P_test_race = self.format_prediction_matrix(self.test_set, P_test_race, pred, all_cols=False)
            Pa_test_race = self.format_prediction_matrix(self.test_set, Pa_test_race, proba, all_cols=False)

        return fsel_results_all, results_all, P_test_race, Pa_test_race
    
    
    def format_prediction_matrix(self, data, P_matrix, pr, all_cols):
        """
        Function to append to a dataframe (P_matrix) each classifier's predictions with each iteration.
        
        P_matrix: Dataframe containing prediction probabilities
        pr: DataFrame containing prediction probabilities fo current 
        all_cols: True: Merge all original features to matrix of prediction probabilities (May be used for ensembling),
                  False: Only index and target variable columns are merged to prediction probabilities
        """
        if (len(P_matrix.columns) == 0):
            if all_cols==True:
                P_matrix = data.reset_index(drop=True) 
            else:
                P_matrix = data[self.index_list+self.target_var_list].reset_index(drop=True) 

        P_matrix = pd.merge(P_matrix, pr, on=self.index_list+self.target_var_list, how='left')
        return P_matrix
                
class ClassifyRaces():
    
    def __init__(self, dfs, dfs_test, methods, generator=kfold_3, races_list=races,
                 index_list=index_list, target_var_list=target_var_list, directory=""): 
        """
        This is a class containing functions enabling predictions on a list of races by iterating through each race.
        
        races_dict - A dictionary of the F1 race calendar (the year of the season and the corresponding races)
        """
        self.dfs = dfs
        self.dfs_test = dfs_test 
        self.methods = methods
        self.generator = generator
        self.races_dict = races_dict
        self.index_list = index_list
        self.target_var_list = target_var_list
        self.directory = directory
        self.races_list = races_list

    def run_models(self, actual, metric, qtys, train_yr, test_yr, model_names, fsel_list):
        """
        Iterate through a list of models and races to generate test set predictions for each race with each model.
        The last step is to select the best performing model by comparing a fixed metric of test results for each model
        Note: Actual test target values are known in selecting the best model. 
        
        actual - Is the actual target variable values known? If yes -> actual=False. If no -> actual=True
        metric - Chosen metric to measure and compare models by
        qtys - list of values to input for 'qty_races_tofilter' parameter when building train-test set
        model_names - list of strings to index each model
        """
        Pa_train_all = pd.DataFrame()
        Pa_test_all = pd.DataFrame()
        df_list = []
        all_report = pd.DataFrame() 
        fsel_results_all = pd.DataFrame()
        cv_results_all = pd.DataFrame()
        results_all = pd.DataFrame()
        
        for name in self.races_list:

            Pa_train_ensem = pd.DataFrame()  
            Pa_test_ensem = pd.DataFrame()
            
            models = []
            for i,j in itertools.product(zip(self.dfs,self.dfs_test), qtys):
                models.append([i[0], i[1], j])
    
            for idx, i in enumerate(models):
                # Initiate class 
                b = Build_train_test_set(actual=False, df=i[0], df_test=i[1], name=name)

                # Create training and test sets
                train_set, test_set, Xs_train, Xs_test, Y_train, Y_test = b.train_test_split(train_yr, test_yr, i[2])

                # Initiate class
                c = ClassifyOneRace(name, train_set, test_set, Xs_train, Xs_test, Y_train, Y_test, self.methods)

                # Generate cross-validated predictions of train set from base-learners
                cv_results, Pa_train_race = c.cross_validate_one_race()
                cv_results['Model'] = model_names[idx]
                
                # Generate first-level predictions of test set
                fsel_results, results, P_test_race, Pa_test_race = c.predict_one_race(fsel_list)
                fsel_results['Model'] = model_names[idx]
                results['Model'] = model_names[idx]
                
                # Suffix model name to train and test columns with predictions
                rename_some_cols(Pa_train_race, model_names[idx], col_start=len(self.index_list+self.target_var_list))
                rename_some_cols(Pa_test_race, model_names[idx], col_start=len(self.index_list+self.target_var_list))
                
                # Merge the predictions of each model horizontally
                Pa_train_ensem = c.format_prediction_matrix(train_set, Pa_train_ensem, Pa_train_race, all_cols=False)
                Pa_test_ensem = c.format_prediction_matrix(test_set, Pa_test_ensem, Pa_test_race, all_cols=False)                       
                
                # Concatenate the following dfs:
                    #1) train set cross-validation results, if any
                    #2) test results of each MODEL and each RACE
                    #3) feature selection results, if any
                fsel_results_all = pd.concat([fsel_results_all, fsel_results])
                cv_results_all = pd.concat([cv_results_all, cv_results])
                results_all = pd.concat([results_all, results])
                
            # Merge the predictions of each race vertically
            Pa_train_all = pd.concat([Pa_train_all, Pa_train_ensem]) # Prediction probabilities generated from cross-validated train set 
            Pa_test_all = pd.concat([Pa_test_all, Pa_test_ensem])
        
        Pa_train_all.to_csv("Pa_train.csv", index = False)
        Pa_test_all.to_csv("Pa_test.csv", index = False)
        cv_results_all.to_csv("cv_results.csv", index = False)
        
        # Based on function's settings, save and return required dfs accordingly
        if actual==False:

            # Find the best peforming model
            m_report = pd.DataFrame()
            for i,j in results_all.groupby(['Model']):
                r = model_report(i, j)
                m_report = pd.concat([m_report, r])
            m_report.sort_values('Average Dist from baseline', ascending=False)

            # Find the best peforming sub-model (Each race must use the same best performing sub-model)
            report = pd.DataFrame(results_all.groupby(['Model','Method'])[metric].agg("mean")).reset_index()
            report = report.sort_values(metric, ascending=False).reset_index(drop=True)
            
            m_report.to_csv("model_report.csv", index = False)
            report.to_csv("submodel_report.csv", index = False)
            results_all.to_csv("results.csv", index = False)
            return m_report, report, Pa_train_all, Pa_test_all, cv_results_all, results_all
        
        if fsel_list==True:
            fsel_results_all.to_csv("fsel_results.csv", index = False)
            return m_report, report, Pa_train_all, Pa_test_all, cv_results_all, results_all, fsel_results_all

        return Pa_train_all, Pa_test_all, cv_results_all

def rename_some_cols(df, suffix, col_start):
    new_names = [(i,i+"_"+suffix) for i in df.iloc[:, col_start:].columns.values]
    return df.rename(columns = dict(new_names), inplace=True)

def model_report(model_name, df):
    abv_baseline = df[df['Distance from baseline'] > 0]
    report = pd.DataFrame(abv_baseline.Method.value_counts()).T
    report['No. of races: Test acc > Baseline acc'] = len(abv_baseline.Index.unique())
    report['List of races: Test acc > Baseline acc'] = [abv_baseline.Index.unique()]
    report['Average Avg Precision'] = df['Avg Precision'].agg("mean")
    report['Average F1 score'] = df['F1 Score'].agg("mean")
    report['Average AUC score'] = df['AUC Score'].agg("mean")
    report['Average Dist from baseline'] = df['Distance from baseline'].agg("mean")
    report.rename(index={'Method': model_name}, inplace=True)

    return report
def calc_classification_stats(race_name, target_var_list, classifier, y_test, y_pred, average):
    """
    Create dataframe containing metrics of classification results.

    y_test - Actual target variable values
    y_pred - Predicted target variable values
    average - This is a paramaeter for sklearn's f1 score metric
    """
    baseline_accuracy = []
    test_accuracy = []
    f1 = []
    avg_precision = []
    auc = []

    baseline_accuracy.append(float(pd.Series(y_test).value_counts().max()) / pd.Series(y_test).count())
    test_accuracy.append(accuracy_score(y_test, y_pred))
    avg_precision.append(average_precision_score(y_test, y_pred))
    f1.append(f1_score(y_test, y_pred, average=average)) 
    auc.append(roc_auc_score(y_test, y_pred))

    results_stats = pd.DataFrame({'Index': race_name, 'Target Var': target_var_list[0], 'Method': classifier[0],
                                  'Avg Precision': avg_precision, 'F1 Score': f1, "AUC Score": auc,
                                  'Test accuracy': test_accuracy, 'Baseline accuracy': baseline_accuracy})

    results_stats['Distance from baseline'] = results_stats['Test accuracy'] - results_stats['Baseline accuracy']

    results_stats = results_stats[['Index', 'Target Var', 'Method', "Test accuracy", 'Distance from baseline',
                                   'AUC Score', 'F1 Score', 'Avg Precision']]

    return results_stats
def plot_algo_results(df, grp_col, metrics_list, sort_method):   
    """
    Function to plot statistics of prediction results
    """
    def calc(df, grp_col, col, new_col_name):
        if new_col_name == 'Std':
            df = pd.DataFrame(df.groupby([grp_col])[col].apply(lambda x: np.std(x)))
  
        elif new_col_name == 'Mean':
            df = pd.DataFrame(df.groupby([grp_col])[col].apply(lambda x: np.mean(x)))
            
        return df.reset_index().rename(columns={col: col+ " (" + new_col_name + ")"})
    
    # This function only calculates mean and standard deviation
    def create_grp_stats(df, grp_col, col):
        p_std = calc(df, grp_col, col, 'Std')
        p_mean = calc(df, grp_col, col, 'Mean')
        p = pd.merge(p_mean, p_std, on=grp_col, how='left')
        return p
    
    # Set order 
    def sort_order(df, col_to_sortby, grp_col):
        if sort_method == "desc":
            ordering = df.sort_values([col_to_sortby])[grp_col].unique()
        elif sort_method == "race":
            ordering = races
        else:
            raise ValueError("Only desc or race are accepted keywords for sort_method variable")

        ids = reversed(list(ordering))
        ids = [str(item) for item in ids]  
        return ids
        
    # plot results
    def plot_barplot(df, x, y, ids, row, col):
        plt.figure()   
        g = sns.barplot(x, y, data = df, order=ids, palette="Set3", orient = "h", ax=axes[row][col])
        g.set_title(x, fontsize=16)
        for p in g.patches:
            width = p.get_width()
            g.text(width*1.05, p.get_y()+0.55*p.get_height(), '{:1.2f}'.format(width), ha='center', va='center')
        
    df_new = pd.DataFrame()
    for i in metrics_list:
        p = create_grp_stats(df, grp_col, i) 
        df_new = pd.concat([df_new, p], axis=1)
        df_new = df_new.T.drop_duplicates().T
    
    if len(df_new.columns) == 3:
        nrows = 1
        ncols = 2
    else:
        nrows = len(df_new.columns)-len(metrics_list)-1
        ncols = 2

    to_plot = df_new.columns[1:]
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15), squeeze=False)
    fig.subplots_adjust(wspace=0.5)
    counter = 0
    for row in range(nrows):
        for col in range(ncols):
            ids = sort_order(df_new, to_plot[counter], grp_col)
            plot_barplot(df_new, to_plot[counter], grp_col, ids, row, col)
            counter += 1
    plt.tight_layout()
            
    return df_new
def conf_mat(df_prob, idx, col_labels):   
    """
    Function to calculate confusion matrix for a chosen model.
    
    df_prob - dataframe containing a column of prediction probabilites and another column of actual target values
    idx - list of class labels (eg. [DNF, FIN])
    col_labels - list of class labels suffixed with model name (eg. [DNF_A, FIN_A])
    """
    conf_arr = np.zeros(shape=(2,2))
    driver_arr = np.array([[0, 0], [0, 0]], dtype=object)
    drivers_FP = []
    drivers_TN = []
    drivers_TP = []
    drivers_FN = []

    prob_arr = np.array(df_prob.iloc[:,-1])
    input_arr = np.array(df_prob.statusId)

    df_prob = df_prob.reset_index(drop=True)
    
    for i in range(len(prob_arr)):
        if int(input_arr[i]) == 0:
            if float(prob_arr[i]) > 0.5:
                # Predicted No, Actual Yes
                conf_arr[0][1] = conf_arr[0][1] + 1
                drivers_FP.append(df_prob.loc[i, 'driverRef'])
                driver_arr[0][1] = drivers_FP # Store list with driver names that match criteria
            else:
                # Predicted No, Actual No
                conf_arr[0][0] = conf_arr[0][0] + 1
                drivers_TN.append(df_prob.loc[i, 'driverRef'])
                driver_arr[0][0] =  drivers_TN
        elif int(input_arr[i]) == 1:
            if float(prob_arr[i]) <= 0.5:
                # Predicted Yes, Actual No
                conf_arr[1][0] = conf_arr[1][0] +1
                drivers_FN.append(df_prob.loc[i, 'driverRef'])
                driver_arr[1][0] = drivers_FN
            else:
                # Predicted Yes, Actual Yes
                conf_arr[1][1] = conf_arr[1][1] +1
                drivers_TP.append(df_prob.loc[i, 'driverRef'])
                driver_arr[1][1] =  drivers_TP
    
    # Convert confusion matrix to percentages
    #cm_sum = np.sum(conf_arr)
    #cm_perc = cm / cm_sum.astype(float) * 100

    conf_matrix = pd.DataFrame(conf_arr, index=idx, columns=col_labels)
    driver_matrix = pd.DataFrame(driver_arr, index=idx, columns=col_labels)
        
    return conf_matrix, driver_matrix

def conf_mat_each_race(df_prob, labels, model_name):
    """
    Function to calculate confusion matrix for each race and returns the drivers that belong to each quarter of the matrix.
    """
    conf_matrix_all = pd.DataFrame()
    driver_matrix_all = pd.DataFrame()

    for name, group in df_prob.groupby(['year', 'name']):
        df_prob_grp = df_prob[df_prob[['year', 'name']].apply(tuple, 1).isin([name])]
        
        idx = pd.MultiIndex.from_product([[name], labels], names=['race', 'status'])
        col_labels = [i+model_name for i in labels]
        
        conf_matrix, driver_matrix = conf_mat(df_prob_grp, idx, col_labels)
        
        # Plot confusion matrix as a heatmap with drivers' names in the annotation
        #fig, ax = plt.subplots(figsize=(4,4))
        #sns.heatmap(conf_matrix, annot=np.array(driver_matrix), fmt = '')
        
        conf_matrix_all = pd.concat([conf_matrix_all, conf_matrix])
        driver_matrix_all = pd.concat([driver_matrix_all, driver_matrix])
        
    return conf_matrix_all, driver_matrix_all
dataset_1 = pd.read_csv("../input/added-master-dataset/dataset.csv")
test_1 = dataset_1[dataset_1['year'] == 2017]
dfs = [dataset_1]
dfs_test = [test_1]
CR = ClassifyRaces(dfs, dfs_test, clfs)
all_models_report, submodel_report, Pa_train_all, Pa_test_all, cv_results_all, results_all = CR.run_models(actual=False, metric='AUC Score', qtys=[3,4,5], train_yr=[2016], test_yr=[2017],
                                                                                                           model_names=['1_3', '1_4', '1_5'], fsel_list=False)
all_models_report
submodel_report.head(10)
DNF_share = pd.Series(Pa_test_all.statusId.value_counts() / sum(Pa_test_all.statusId.value_counts()))[0]

# Matrix of first level prediction results
Pa_test_all_plot = Pa_test_all.drop(index_list+target_var_list, axis=1)
p= Pa_test_all_plot.apply(lambda x: 1*(x >= 0.5).value_counts(normalize=True)).T.reset_index().rename(columns={True: "FIN", False:"DNF"}).sort_values('DNF', ascending=False)
p['Method'] = p['index'].apply(lambda x: x.split('_', 1)[0])

plt.figure(figsize=(8,12))
sns.set(font_scale=1)
sns.barplot(x="DNF", y="index", data=p, hue='Method', dodge=False)

plt.axvline(DNF_share, color="k", linewidth=0.5)
plt.text(0., DNF_share-0.02, "True share DNF")
plt.show()
best_model = results_all[(results_all['Model'] == '1_4')]
metrics_list = ["Distance from baseline", "AUC Score"]
plot_best_model = plot_algo_results(best_model, grp_col="Method", metrics_list=metrics_list, sort_method='desc')
idx_best_model = plot_algo_results(best_model, grp_col="Index", metrics_list=metrics_list, sort_method='desc')
trees_clfs = [
    
    ['Random Forest', RandomForestClassifier(random_state=rs, class_weight=None)],
    ['Gradient Boosting', GradientBoostingClassifier(random_state=rs)],
    ['Extra Trees',  ExtraTreesClassifier(random_state=rs)],
]

races_to_plot = ['Australian Grand Prix', 'Monaco Grand Prix', 'Hungarian Grand Prix']
        
for idx,race in enumerate(races_to_plot):
    for m in trees_clfs:

        b = Build_train_test_set(True, dataset_1, test_1, race)
        train_set1, test_set1, Xs_train1, Xs_test1, Y_train1, Y_test1 = b.train_test_split(train_yr=[2016], test_yr=[2017], qty_races_tofilter=3)
        m[1].fit(Xs_train1, Y_train1)

    cols_f = np.setdiff1d(train_set1.columns, np.array(index_list + target_var_list))
    train_set1_plot = train_set1.loc[:, cols_f]

    nrows = 1
    ncols = 3
    fig, axes = plt.subplots(nrows = 1, ncols = 3, sharex="all", figsize=(15,5), squeeze=False)
    nclassifier = 0
    for row in range(nrows):
        for col in range(ncols):
            name = trees_clfs[nclassifier][0]
            classifier = trees_clfs[nclassifier][1]
            indices = np.argsort(classifier.feature_importances_)[::-1][:40]
            g = sns.barplot(y=train_set1_plot.columns[indices][:40], x = classifier.feature_importances_[indices][:40] ,
                            orient='h',ax=axes[row][col])
            g.set_xlabel("Relative importance",fontsize=12)
            g.set_ylabel("Features",fontsize=12)
            g.tick_params(labelsize=9)
            g.set_title(name + " feature importance")
            fig.suptitle(race)
            nclassifier += 1
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
best_submodel = 'Gradient Boosting_statusId_1_4'
Pa_A = Pa_test_all[index_list+target_var_list+[best_submodel]]
labels = ['DNF', 'FIN']
conf_matrix_A, driver_matrix_A = conf_mat(Pa_A, idx=labels, col_labels=['DNF_A', 'FIN_A'])
conf_matrix_A_r, driver_matrix_A_r = conf_mat_each_race(Pa_A, labels, '_A')
conf_matrix_A
conf_matrix_A_r
driver_matrix_A_r
