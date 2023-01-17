# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, KFold



from sklearn.preprocessing import MinMaxScaler,StandardScaler

from collections import Counter

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Exploratory Data Analysis : Data Clean





# Outlier detection and drop

def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

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



########### Load Data ############

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(train_data ,2,["Age","SibSp","Parch"])

print("Outliers_to_drop:", Outliers_to_drop)

train_data = train_data.drop(Outliers_to_drop, axis = 0 ).reset_index(drop=True)

whole_data = pd.concat([train_data,test_data],axis =0).reset_index(drop=True)

#print("whole_data.shape:", whole_data.shape)

############## Add columne Family Size Fsize ##########



whole_data["Fsize"] = whole_data["SibSp"] + whole_data["Parch"] + 1

whole_data['Single'] = whole_data['Fsize'].map(lambda s: 1 if s == 1 else 0)

whole_data['SmallF'] = whole_data['Fsize'].map(lambda s: 1 if  s == 2  else 0)

whole_data['MedF'] = whole_data['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)

whole_data['LargeF'] = whole_data['Fsize'].map(lambda s: 1 if s >= 5 else 0)



##### Name  and Title ######

whole_data['Title'] = whole_data['Name'].str.extract(r', (\w+\.)')

print("Different Titles: ", whole_data.Title.unique())

whole_data['NameLength'] = whole_data['Name'].str.len()

##############   Replace ##########################

# ['Mr.' 'Mrs.' 'Miss.' 'Master.' 'Ms.' ]

whole_data["Title"] = whole_data["Title"].replace(['Lady.', 'Countess.', 'Dona.', 'Jonkheer.', 'Mme.'], 'Ms.')

whole_data["Title"] = whole_data["Title"].replace(['Capt.', 'Don.', 'Dr.', 'Major.', 'Rev.','Col.', 'Sir.','Mlle.'], 'Mr.')

whole_data['Title'].fillna(whole_data.Title.mode()[0],inplace=True)

print("Different Titles after replacement : ", whole_data.Title.unique())

whole_data["Title"] = whole_data["Title"].map({"Master.":0, "Miss.":1, "Ms." : 2 , "Mrs.":3, "Mr.":4 })

whole_data["Title"] = whole_data["Title"].astype(int)



##### Cabin  ######

whole_data['Cabin_cat'] = whole_data['Cabin'].str[0] # We use RFC later for fillna.

#whole_data["Cabin_missing"] = pd.Series()

whole_data.loc[whole_data['Cabin'].isnull(), ['Cabin_missing']]= 1

whole_data.loc[whole_data['Cabin'].notnull(), ['Cabin_missing']]= 0

#pd.Series(0 if i is 'nan' else 1 for i in whole_data['Cabin'])

print("Cabin_missing mean:", whole_data.Cabin_missing.mean())





##### Embarked  ######

whole_data['Embarked'].fillna(whole_data.Embarked.mode()[0],inplace=True)

##### Ticket  ######

whole_data['Ticket_Letter'] = whole_data['Ticket'].str.split().str[0]

whole_data['Ticket_Letter'] = whole_data['Ticket_Letter'].apply(lambda x: 'U0' if x.isnumeric() else x)

#whole_data['Ticket_Number'] = whole_data['Ticket'].apply(lambda x: pd.to_numeric(x, errors='coerce'))

#whole_data['Ticket_Number'].fillna(0, inplace=True)                                                                                                   

whole_data['Ticket_Letter'] = pd.factorize(whole_data['Ticket_Letter'])[0]  

whole_data.drop('Cabin',axis = 1,inplace = True)

# First fill the 'Fare' NaN cell, otherwise training the RFR with Fare data will lead error.

ModeFare = whole_data.Fare.mode()

#print("Fare mode value:" ,ModeFare)

#print("Fare before fillna:" ,whole_data.info())

#Impute the Embarked NaN cells with mode value

whole_data['Fare'].fillna(ModeFare[0],inplace=True)

# Explore Fare distribution 

g = sns.distplot(whole_data["Fare"], color="m", label="Skewness : %.2f"%(whole_data["Fare"].skew()))

g = g.legend(loc="best")

# Apply log to Fare to reduce skewness distribution

whole_data["Fare"] = whole_data["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

g = sns.distplot(whole_data["Fare"], color="b", label="Skewness : %.2f"%(whole_data["Fare"].skew()))

g = g.legend(loc="best")

#print("Fare after fillna:" ,whole_data.info())





# We use Random Forest Regressor with numerical columns to fill Age column



# We choose only numerical columns

age_df = whole_data.select_dtypes(exclude=['object']).copy()



age_df.drop('Survived', inplace=True, axis=1)

age_df_notnull = age_df.loc[(whole_data['Age'].notnull())]

age_df_isnull = age_df.loc[(whole_data['Age'].isnull())]



# Training data for Age fillna

X = age_df_notnull.loc[:, age_df_notnull.columns != 'Age']

Y = age_df_notnull['Age']



RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)

RFR.fit(X,Y)

predictAges = RFR.predict(age_df_isnull.loc[:, age_df_isnull.columns != 'Age'])

whole_data.loc[whole_data['Age'].isnull(), ['Age']]= predictAges

#print("predictAges: ", predictAges.shape, predictAges)

#print("whole_data after fillna AGE:" ,whole_data.info())





############### GET dummies ##################

#print("whole_data columns: ", '\n whole_data.shape: ',whole_data.columns)

whole_data = pd.get_dummies(whole_data, columns=['Sex', 'Embarked','Pclass'], drop_first = True)

#whole_data = pd.get_dummies(whole_data, columns=['Sex', 'Embarked','Pclass', 'Title','Parch','SibSp'], drop_first = True)



#print("whole_data after get dummies: ", whole_data.info(), '\n whole_data.shape: ',whole_data.shape)







#############  Fare binning and Factorizing ####################

whole_data['Fare_bin'] = pd.qcut(whole_data['Fare'], 5)

# factorize

whole_data['Fare_bin_id'] = pd.factorize(whole_data['Fare_bin'],sort = True)[0]

whole_data.drop(['Fare','Fare_bin'],axis=1, inplace = True)

# If use get_dummies

#whole_data = pd.concat([whole_data, fare_bin_dummies_df], axis=1)



################################################################

train_data_rows = train_data.shape[0]

train_data = whole_data[:train_data_rows]







# Now we need to fill the NaN Cabin_cat cells, since its object type, 

# convert it to integer encoding 'Cabin_code' and then use RFC.



whole_data['Cabin_code'] = whole_data.Cabin_cat.astype('category').cat.codes 

whole_data['Cabin_code'] = whole_data['Cabin_code'].replace({ -1: np.nan})

whole_data.drop('Cabin_cat', inplace=True, axis=1)

# We use Random Forest Classifier with categorical integer columns to fillna

# We choose only numerical columns for training

cabin_df = whole_data.select_dtypes(exclude=['object']).copy()

cabin_df.drop('Survived', inplace=True, axis=1)  #Exclude the label

cabin_df_notnull = cabin_df.loc[(whole_data['Cabin_code'].notnull())]

cabin_df_isnull = cabin_df.loc[(whole_data['Cabin_code'].isnull())]



# Training data for Cabin_code fillna

X = cabin_df_notnull.loc[:, cabin_df_notnull.columns != 'Cabin_code']

#print("X: \n", X.info())

Y = cabin_df_notnull['Cabin_code']

RSEED = 2

RFC_Cabin= RandomForestClassifier(n_estimators=1000, 

                            random_state=RSEED, 

                            max_features = 'log2',

                            n_jobs=-1, verbose = 1)

RFC_Cabin.fit(X,Y)

predictCabins = RFC_Cabin.predict(cabin_df_isnull.loc[:, cabin_df_isnull.columns != 'Cabin_code'])

whole_data.loc[whole_data['Cabin_code'].isnull(), ['Cabin_code']]= predictCabins

#print("predictCabins: ", predictCabins.shape)

#print("Cabin_code unique values: ", whole_data['Cabin_code'].unique())



##### Scaler #####################

scaler = MinMaxScaler()

whole_data.loc[:train_data_rows,'Age'] = scaler.fit_transform(whole_data.loc[:train_data_rows,'Age'].values.reshape(-1, 1))

whole_data.loc[train_data_rows:,'Age']= scaler.transform(whole_data.loc[train_data_rows:, 'Age'].values.reshape(-1, 1))





#whole_data = pd.get_dummies(whole_data, columns=['Cabin_code'], drop_first = True)

#X_test = pd.get_dummies(X_test, columns=['Cabin_code'], drop_first = True)





train_data = whole_data[:train_data_rows]

test_data = whole_data[train_data_rows:]

print("train_data shape:", train_data.shape)

print("test_data shape:", test_data.shape)

print(test_data.columns)



X = train_data.select_dtypes(exclude=['object']).copy()

#X = train_data[['Age', 'NameLength', 'Ticket_Letter', 'Pclass', 'Parch','Cabin_code','SibSp','Fare_bin_id']]

X.drop(['Survived','PassengerId'],axis = 1,inplace = True)

y = train_data["Survived"].astype(int)

X_test = test_data.select_dtypes(exclude=['object']).copy()

#X_test = test_data[[ 'Age', 'NameLength', 'Ticket_Letter','Pclass', 'Parch','Cabin_code','SibSp','Fare_bin_id']]

X_test.drop(['Survived','PassengerId'],axis=1, inplace = True)

print("X_test head:", X_test.head())

print(X.shape, X.columns)

print(X_test.shape,X.columns)

# Cross validate model with Kfold stratified cross val

kfold =StratifiedKFold(n_splits=10)

print(kfold,type(kfold))

# Modeling step Test differents algorithms 

random_state = 2

classifiers = []

classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(ExtraTreesClassifier(random_state=random_state))

classifiers.append(GradientBoostingClassifier(random_state=random_state))

classifiers.append(MLPClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(LinearDiscriminantAnalysis())



cv_results_auc = []

cv_results_acc = []

for classifier in classifiers :

    cv_results_auc.append(cross_val_score(classifier, X, y, scoring = "roc_auc", cv = kfold, n_jobs=4))

    cv_results_acc.append(cross_val_score(classifier, X, y, scoring = "accuracy", cv = kfold, n_jobs=4))



cv_means_auc = []

cv_std_auc = []

for cv_result in cv_results_auc:

    cv_means_auc.append(cv_result.mean())

    cv_std_auc.append(cv_result.std())



cv_means_acc = []

cv_std_acc = []

for cv_result in cv_results_acc:

    cv_means_acc.append(cv_result.mean())

    cv_std_acc.append(cv_result.std())

    



cv_res_auc = pd.DataFrame({"CrossValMeans":cv_means_auc,"CrossValerrors": cv_std_auc,"Algorithm":["SVC","DecisionTree","AdaBoost",

"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})

cv_res_acc = pd.DataFrame({"CrossValMeans":cv_means_acc,"CrossValerrors": cv_std_acc,"Algorithm":["SVC","DecisionTree","AdaBoost",

"RandomForest","ExtraTrees","GradientBoosting","MultipleLayerPerceptron","KNeighboors","LogisticRegression","LinearDiscriminantAnalysis"]})





g1 = sns.barplot("CrossValMeans","Algorithm",data = cv_res_auc, palette="Set3",orient = "h",**{'xerr':cv_std_auc})

g1.set_xlabel("Mean auc")

g1 = g1.set_title("Cross validation scores")



g2 = sns.barplot("CrossValMeans","Algorithm",data = cv_res_acc, palette="Set3",orient = "h",**{'xerr':cv_std_acc})

g2.set_xlabel("Mean Accuracy")

g2 = g2.set_title("Cross validation scores")



# Hyper Search for GradientBoosting

#Top N Features Best GB Params:{'learning_rate': 0.1, 'max_depth': 12, 'n_estimators': 1000}

#Top N Features Best GB Score:0.7903144812001758

GB = GradientBoostingClassifier(random_state=2)

param_grid = {'n_estimators': [1000], 'learning_rate': [0.01, 0.1], 'max_depth': [12]}

GB_search = GridSearchCV(GB, param_grid, n_jobs=-1, cv=5, verbose=True)

GB_search.fit(X, y)

best_GB = GB_search.best_estimator_

predictions_GB = best_GB.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_GB.astype(int)})

print(output.info())

print(output.head())

output.to_csv('my_submission_best_GB_20201025.csv', index=False)

print("Hyper Param Tuning for GradienBoosting DONE!")



print('Top N Features Best GB Params:' + str(GB_search.best_params_))

print('Top N Features Best GB Score:' + str(GB_search.best_score_))

print('Top N Features GB Train Score:' + str(GB_search.score(X, y)))

feature_imp_sorted_GB = pd.DataFrame({'feature': list(X),

                                       'importance': GB_search.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)

features_top_n_GB = feature_imp_sorted_GB.head(15)['feature']

print('Sample 10 Feature from GB Classifier:')

print(str(features_top_n_GB[:10]))
## Hyper Search for LogisticRegression



LR = LogisticRegression()

#Best Params: 

param_grid = {'C': [29.763514416313132], 'max_iter': [200], 'penalty': ['l2'], 'solver': ['liblinear']}



#param_grid = {'penalty' : ['l1', 'l2'],

#              'C' : np.logspace(-4, 4, 20),

#              'solver' : ['liblinear'],

#              'max_iter' : [200,500,1000]

#             }

LR_search = GridSearchCV(LR, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)



# Fit on data

LR_search.fit(X, y)

print("Best Params: \n", LR_search.best_params_)

best_LR = LR_search.best_estimator_

predictions_LR = best_LR.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_LR.astype(int)})

print(output.info())

print(output.head())

output.to_csv('my_submission_best_LR_20201025.csv', index=False)

print("Hyper Param Tuning for LogisticRegression DONE!")



print("predictions_LR.mean:", predictions_LR.mean())

print('Top N Features LR Train Score:' + str(LR_search.score(X, y)))



coef_index = np.squeeze(np.array(np.std(X, 0)) * best_LR.coef_ )



ind = np.argpartition(coef_index, -15)[-15:]

print("ind:", ind)

sorted_ind = ind[np.argsort(coef_index[ind])][::-1]

print("sorted_ind:", sorted_ind)

features_top_n_LR_columns = X.columns[sorted_ind]

print('Sample 10 Features from LR Classifier: \n')

print(str(features_top_n_LR_columns[:10]))

features_top_n_LR = pd.Series(data = list(features_top_n_LR_columns),index = list(sorted_ind), name = 'feature')

print(features_top_n_LR)

## Hyper Search for LDA (Linear Discriminant Analysis)



LDA = LinearDiscriminantAnalysis(priors= [0.68, 0.32])

param_grid = {'solver': ['svd', 'lsqr']}



LDA_search = RandomizedSearchCV(LDA, param_distributions = param_grid, cv = 5, scoring = 'roc_auc', verbose=True, n_jobs=-1)



# Fit on data

LDA_search.fit(X, y)

print("Best Params: \n", LDA_search.best_params_)

# print("Best accuracy: \n", LDA_search.best_score_)

best_LDA = LDA_search.best_estimator_

predictions_LDA = best_LDA.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_LDA.astype(int)})

print(output.info())

print(output.head())

output.to_csv('my_submission_best_LDA_20201025.csv', index=False)

print("Hyper Param Tuning for LDA DONE!")



print("predictions_LDA.mean:", predictions_LDA.mean())

print('Top N Features LR Train Score:' + str(LDA_search.score(X, y)))

coef_index = np.squeeze(np.array(np.std(X, 0)) * best_LDA.coef_ )



ind = np.argpartition(coef_index, -15)[-15:]

print("ind:", ind)

sorted_ind = ind[np.argsort(coef_index[ind])][::-1]

print("sorted_ind:", sorted_ind)

features_top_n_LDA_columns = X.columns[sorted_ind]

print('Sample 10 Features from LDA Classifier: \n')

print(str(features_top_n_LDA_columns[:10]))

features_top_n_LDA = pd.Series(data = list(features_top_n_LDA_columns),index = list(sorted_ind), name = 'feature')

print(features_top_n_LDA)
## Hyper Search for RandomForest

#Best Params: 

random_grid = {'n_jobs': [-1], 'n_estimators': [200,1000], 'min_samples_split': [5], 'min_samples_leaf': [2], 'max_features': ['sqrt','log2'], 'max_depth': [10], 'bootstrap': [False,True]}

## HyperSearch RandomForestClassifier

#n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 4)]

#max_features = ['sqrt','log2']

#max_depth = [8,10,16 ]  # Modified not tested

#min_samples_split = [2, 5, 10]

#min_samples_leaf = [1, 2, 8]

#bootstrap = [False, True]

# Create the random grid

#random_grid = {'n_estimators': n_estimators,

#               'max_features': max_features,

#               'max_depth': max_depth,

#               'min_samples_split': min_samples_split,

#               'min_samples_leaf': min_samples_leaf,

#               'bootstrap': bootstrap,

#               'n_jobs' : [-1]}

#print(random_grid)

#

rf = RandomForestClassifier()

rf_search = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring = "roc_auc", n_iter = 1000, verbose=True)

# Fit the random search model

rf_search.fit(X, y)

print("Best Params: \n", rf_search.best_params_)

#print(rf_search.cv_results_)

best_RFC = rf_search.best_estimator_



predictions_RFC = rf_search.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_RFC.astype(int)})

print(output.info())

print(output.head())

output.to_csv('my_submission_best_RFC_20201025.csv', index=False)

print("Hyper Param Tuning for RandomForestClassifier DONE!")



print("predictions_RFC.mean:", predictions_RFC.mean())

print('Top N Features RFC Train Score:' + str(rf_search.score(X, y)))

feature_imp_sorted_RFC = pd.DataFrame({'feature': list(X),

                                      'importance': rf_search.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)

features_top_n_RFC = feature_imp_sorted_RFC.head(15)['feature']

print('Sample 10 Features from RF Classifier: \n')

print(str(features_top_n_RFC[:10]))
print(type(features_top_n_RFC),features_top_n_RFC)
## Hyper Search for MLP    

#Best Params: 

parameter_space = {'activation': ['tanh'], 'alpha': [0.1], 'early_stopping': [True], 'hidden_layer_sizes': [(128, 64, 32, 16, 8)], 'learning_rate': ['adaptive'], 'solver': ['adam']}

MLP  = MLPClassifier(max_iter=2000)

#parameter_space = {

#    'hidden_layer_sizes': [(128),(128,64,32,16,8),(128,64,32,16),

#                           (128,64,32,16,32,64,128)],

#    'activation': ['tanh', 'relu'],

#    'solver': ['sgd', 'adam'],

#    'alpha': [0.0001, 0.001,0.01,0.05,0.1],

#    'learning_rate': ['constant','adaptive'],

#    'early_stopping': [True]

#}



MLP_search = GridSearchCV(MLP, parameter_space, scoring= 'roc_auc', n_jobs=-1, cv=5,verbose=True)

MLP_search.fit(X, y) # X is train samples and y is the corresponding labels

print("Best Params: \n", MLP_search.best_params_)

predictions_MLP = MLP_search.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_MLP.astype(int)})

print(output.info())

print(output.head())

output.to_csv('my_submission_best_MLP_20201025.csv', index=False)

print("Hyper Param Tuning for MLP DONE!")





print("predictions_MLP.mean:", predictions_MLP.mean())

print('Top N Features MLP Train Score:' + str(MLP_search.score(X, y)))

A = MLP_search.best_estimator_.coefs_[0]

print (A.shape)

B= np.max(A,axis=1)

print("B:", B)

ind = np.argpartition(B, -15)[-15:]

print("ind:", ind)

features_top_n_MLP_index = ind[np.argsort(B[ind])][::-1]

print("features_top_n_MLP_index:", features_top_n_MLP_index)

features_top_n_MLP_columns = X.columns[features_top_n_MLP_index]

print('Sample 10 Features from MLP Classifier: \n')

print(str(features_top_n_MLP_columns[:10]))

features_top_n_MLP = pd.Series(data = list(features_top_n_MLP_columns),index = list(features_top_n_MLP_index), name = 'feature')

print(features_top_n_MLP)

print(type(features_top_n_MLP),features_top_n_MLP)
## Hyper Search for ExtraTree

#Best: 0.876913 using 

param_grid = {'ccp_alpha': [0.001], 'criterion': ['entropy'], 'max_depth': [16], 'max_features': ['sqrt'], 'min_samples_leaf': [4], 'min_samples_split': [16], 'n_estimators': [200]}



ETC = ExtraTreesClassifier(n_estimators=1000, n_jobs=4, min_samples_split=2,

                            min_samples_leaf=1, max_features='auto')

                            

#param_grid={

#        'n_estimators':[int(x) for x in np.linspace(start = 200, stop = 2000, num = 4)],

#        'max_features': ['sqrt','log2'],

#        'criterion':['gini', 'entropy'],

#        'max_depth' : [8, 12, 16 ],

#        'min_samples_split' :[8,10,12,16],

#        'min_samples_leaf' :[2, 4],

#        'ccp_alpha': [0.001, 0.005]

#    }

ETC_search = GridSearchCV(

    estimator=ETC,

    param_grid = param_grid,

    scoring='roc_auc',

    cv=5,

    verbose = True

)



grid_result = ETC_search.fit(X, y)



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))



#for test_mean, train_mean, param in zip(

#        grid_result.cv_results_['mean_test_score'],

#        grid_result.cv_results_['mean_train_score'],

#        grid_result.cv_results_['params']):

#    print("Train: %f // Test : %f with: %r" % (train_mean, test_mean, param))

    

#model = ExtraTreesClassifier(**grid_result.best_params_)

#model.fit(X, y)

predictions_ETC = ETC_search.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_ETC.astype(int)})

print(output.info())

print(output.head())

output.to_csv('my_submission_best_ETC.csv', index=False)

print("Hyper Param Tuning for ExtraTreesClassifier DONE!")



print("predictions_ETC.mean:", predictions_ETC.mean())

print('Top N Features ETC Train Score:' + str(ETC_search.score(X, y)))

feature_imp_sorted_ETC = pd.DataFrame({'feature': list(X),

                                      'importance': ETC_search.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)

features_top_n_ETC = feature_imp_sorted_ETC.head(15)['feature']

print('Sample 10 Features from ETC Classifier: \n')

print(str(features_top_n_ETC[:10]))
# Merge All the 6 models

features_top_n = pd.concat([features_top_n_RFC, features_top_n_MLP, features_top_n_ETC, features_top_n_GB, features_top_n_LDA,features_top_n_LR], ignore_index=True).drop_duplicates()

# from sklearn.model_selection import KFold

# Some useful parameters which will come in handy later on

train_data_X = pd.DataFrame(X[features_top_n])

test_data_X = pd.DataFrame(X_test[features_top_n])

ntrain = train_data_X.shape[0]

ntest = test_data_X.shape[0]

# SEED = 2020 # for reproducibility  # not effective when shuffle = False

NFOLDS = 7 # set folds for out-of-fold prediction

kf = KFold(n_splits = NFOLDS, shuffle=False)

def get_out_fold(clf, x_train, y_train, x_test):

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))

    oof_test_skf = np.empty((NFOLDS, ntest))



    for i, (train_index, test_index) in enumerate(kf.split(x_train)):

        x_tr = x_train[train_index]

        y_tr = y_train[train_index]

        x_te = x_train[test_index]



        clf.fit(x_tr, y_tr)



        oof_train[test_index] = clf.predict(x_te)

        oof_test_skf[i, :] = clf.predict(x_test)



    oof_test[:] = oof_test_skf.mean(axis=0)

    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)



rf = rf_search.best_estimator_

gb = GB_search.best_estimator_

mlp = MLP_search.best_estimator_

lr = LR_search.best_estimator_

lda = LDA_search.best_estimator_

et = ETC_search.best_estimator_



# Create Numpy arrays of train, test and target (Survived) dataframes to feed into the model

x_train = X.values 

# Creates an array of the train data

x_test = X_test.values 

# Creats an array of the test data

y_train = y.values



# Create our OOF train and test predictions. These base results will be used as new features

rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test) 

# Random Forest

mlp_oof_train, mlp_oof_test = get_out_fold(mlp, x_train, y_train, x_test) 

# Multi Layer Perceptron

et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test) 

# Extra Trees

gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test) 

# Gradient Boost

lda_oof_train, lda_oof_test = get_out_fold(lda, x_train, y_train, x_test) 

# Linear Discriminant Analysis 

lr_oof_train, lr_oof_test = get_out_fold(lr, x_train, y_train, x_test) 

# Logistic Regression

print("Training is complete")



# ######  Level 2 ########################

x_train = np.concatenate((rf_oof_train, mlp_oof_train, et_oof_train, gb_oof_train, lda_oof_train, lr_oof_train), axis=1)

x_test = np.concatenate((rf_oof_test, mlp_oof_test, et_oof_test, gb_oof_test, lda_oof_test, lr_oof_test), axis=1)



from xgboost import XGBClassifier

gbm = XGBClassifier( n_estimators= 2000, max_depth= 8, min_child_weight= 2, gamma=0.9, subsample=0.8, 

                     colsample_bytree=0.8, objective= 'binary:logistic', nthread= -1, scale_pos_weight=1).fit(x_train, y_train)

predictions_gbm = gbm.predict(x_test)

StackingSubmission = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions_gbm.astype(int)})

StackingSubmission.to_csv('EnsembleSubmission.csv',index=False)

print(StackingSubmission.head(4))

#score_rf_acc =  cross_val_score(rf,X,y,scoring='Accuracy',cv=5, njobs=-1,verbose=True)

#score_rf_auc =  cross_val_score(rf,X,y,scoring='roc_auc',cv=5, njobs=-1,verbose=True)
