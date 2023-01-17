# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





#Libraries used for preprocessing data/exploratory data analysis

import numpy as np  

import pandas as pd  

import matplotlib.pyplot as plt

import seaborn as sns



#Importing models used in the notebook

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import statsmodels.api as sm 



#Importing objects to help tune hyperparameters and create train/validation split

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split



#Importing objects used for cross-validation

from sklearn.model_selection import StratifiedKFold



#Importing metrics used to analyze model performance

from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve, plot_roc_curve, roc_auc_score



#Importing libraries used for feature importance 

import eli5

from eli5.sklearn import PermutationImportance

from pdpbox import pdp, get_dataset, info_plots



#Load the data

train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

all_data = pd.concat([train_data, test_data], axis = 0)





sns.set()  #Setting the style of plots to seaborns default style

%matplotlib inline  #Plots will automatically show
print(train_data.info())

train_data.head(5)
test_data.info()
all_data.describe(include = 'all') #include is used to include categorical variables such as Sex
#Table showing correlation values amongst features.

train_data.corr()
#Heatmap of the correlated table - useful for when there are lots of features - easier to read.

plt.figure(figsize=(8,8))

sns.heatmap(train_data.corr(), cmap = 'viridis', annot = True, square = True, lw=1, linecolor='black')
#Percentage of survivors

total_pass = 100 * train_data[train_data['Survived'] == 0]['Survived'].count()/train_data['Survived'].count() #% of Passed 

total_surv = (100 - total_pass) #Percentage of survivors

print(f"% of passengers who passed: {total_pass.round(2)}% \n% of passengers who survived: {total_surv.round(2)}%")



#Plotting Number of survivors

sns.countplot(x = 'Survived', data = train_data)
#Creating the axes for the plots

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(22,10))



#Plotting the target distribution of the categorical features - excluding Name and Ticket 

#Feature engineering for these are covered in section 2

sns.countplot(x='Sex', data=train_data, hue = 'Survived', ax=axes[0])

axes[0].set_xlabel('Sex', fontsize=15)

sns.countplot(x='Embarked', data=train_data, hue='Survived', ax=axes[1])

axes[1].set_xlabel('Embarked', fontsize=15)

sns.countplot(x='Pclass', data=train_data, hue='Survived', ax=axes[2])

axes[2].set_xlabel('Pclass', fontsize=15)



#Storing PassengerID for later use in submission CSV file

test_PassengerID = test_data['PassengerId']



#Dropping PassengerID and Name because they will no longer be used.

#Cabin and Ticket help in understanding missing data in section 3 and will therefore not be dropped yet.

train_data.drop(['PassengerId'], axis=1, inplace=True)

test_data.drop(['PassengerId'], axis=1, inplace=True)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))



#Plotting the feature Age

train_data[train_data['Survived'] == 0]['Age'].plot(kind = 'hist', bins = 20, color = 'blue', label = 'Passed', ax = axes[0][0])

train_data[train_data['Survived'] == 1]['Age'].plot(kind = 'hist', bins = 20, color = 'orange', alpha = 0.7, label = 'Survived', ax = axes[0][0])

axes[0][0].legend()

axes[0][0].set_xlabel('Age Dist.', fontsize=15)



#Defining the 95th quantile of ticket prices & plotting the feature Fare

quant_95 = np.quantile(train_data['Fare'], 0.95)

train_data[(train_data['Survived']==0)&(train_data['Fare']<quant_95)]['Fare'].plot(kind = 'hist', bins = 20, color = 'blue', label = 'Passed', ax=axes[0][1])

train_data[(train_data['Survived']==1)&(train_data['Fare']<quant_95)]['Fare'].plot(kind = 'hist', bins = 20, color = 'orange', alpha = 0.7, label = 'Survived', ax=axes[0][1])

axes[0][1].legend()

axes[0][1].set_xlabel('Fare Dist.', fontsize=15)



#Plotting Sibsp

sns.countplot('SibSp', data = train_data, hue='Survived', ax=axes[1][0])

axes[1][0].legend(loc='upper right')

axes[1][0].set_xlabel('SibSp Dist.', fontsize=15)



#Plotting Parch

sns.countplot('Parch', data = train_data, hue='Survived', ax=axes[1][1])

axes[1][1].legend(loc='upper right')

axes[1][1].set_xlabel('Parch Dist.', fontsize=15)
#Extracting the prefix from the entire name - ex: Mr, Mrs, Capt, etc.

train_data['Unique_Title'] = train_data['Name'].apply(lambda full_name: full_name.split(', ')[1].split('.')[0])

test_data['Unique_Title'] = test_data['Name'].apply(lambda full_name: full_name.split(', ')[1].split('.')[0])



#Plotting survival rate for each title

plt.figure(figsize=(20,7))

sns.countplot('Unique_Title',data=train_data, hue='Survived')

plt.legend(['Passed','Survived'], loc='upper right', prop={'size':15})
#Grouping the 17 unique titles into the 4 categories - Mr, Miss/Mrs, Job, Master

#Titles Mr and Master don't need to be touched because they are their own category

train_data['Title'] = train_data['Unique_Title']

train_data['Title'] = train_data['Title'].replace(['Dr','Col','Major','Jonkheer','Capt','Sir','Don','Rev'], 'Job')

train_data['Title'] = train_data['Title'].replace(['Miss','Mrs','Ms','Mlle','Lady','Mme','the Countess','Dona'], 'Miss/Mrs')

test_data['Title'] = test_data['Unique_Title']

test_data['Title'] = test_data['Title'].replace(['Dr','Col','Major','Jonkheer','Capt','Sir','Don','Rev'], 'Job')

test_data['Title'] = test_data['Title'].replace(['Miss','Mrs','Ms','Mlle','Lady','Mme','the Countess','Dona'], 'Miss/Mrs')



#Plotting 4 categories 

plt.figure(figsize=(16,7))

sns.countplot('Title', data=train_data, hue = 'Survived')
#Creating boolean feature Married

train_data['Married'] = train_data['Unique_Title']=='Mrs'

test_data['Married'] = test_data['Unique_Title']=='Mrs'



#Plotting new feature married

sns.countplot('Married', data=train_data, hue='Survived')



#Dropping columns that are not needed anymore - Name, Unique_Title

train_data.drop(['Name', 'Unique_Title'], axis=1, inplace=True)

test_data.drop(['Name', 'Unique_Title'], axis=1, inplace=True)
#Plotting SibSP and Parch to see distribution of target

fig, axs = plt.subplots(ncols = 3, nrows = 1,figsize=(18,6))

axs[0].set_title('Siblings/spouses', size=15)

axs[1].set_title('Parents/children', size=15)

sns.countplot('SibSp', data = train_data, ax=axs[0], hue='Survived')

sns.countplot('Parch', data = train_data, ax=axs[1], hue='Survived')



#Adding SibSp and Parch to create Fam_Size

train_data['Fam_Size'] = train_data['SibSp'] + train_data['Parch']

test_data['Fam_Size'] = test_data['SibSp'] + test_data['Parch']



#Making function to bin family sizes

def grouping(fam_size):

    if fam_size == 0:

        return 'Individual'

    elif 1 <= fam_size <= 2:

        return 'Small Family'

    elif 3 <= fam_size <= 5:

        return 'Medium Family'

    else:

        return 'Large Family'



#Applying grouping function to Fam_Size

train_data['Fam_Size'] = train_data['Fam_Size'].apply(grouping)

test_data['Fam_Size'] = test_data['Fam_Size'].apply(grouping)



#Plotting new feature Fam_Size

axs[2].set_title('Family Size', size=15)

sns.countplot('Fam_Size', data=train_data, hue='Survived', ax=axs[2])



#Dropping Parch and SibSp

train_data.drop(['SibSp','Parch'], axis=1, inplace=True)

test_data.drop(['SibSp','Parch'], axis=1, inplace=True)
#Displaying the median for each class while considering sex 

all_data.groupby(['Pclass', 'Sex'])['Age'].median()
#Filling the missing age datapoints with the median for each class while considering sex

#Using the population data to fill the missing data because it will give the most accurate representation for age

all_data['Age'] = all_data.groupby(['Pclass', 'Sex'])['Age'].transform(lambda age: age.fillna(age.median()))



#Now move data from combined dataframe to training and test datasets.

train_data['Age'] = all_data.iloc[0:891]['Age']

test_data['Age'] = all_data.iloc[891:]['Age']
#Finding the specific two passengers who are missing from embarked

train_data[train_data['Embarked'].isnull()]
#Finding the most frequent port women in first class embarked from (use all the data to get accurate representation)

embarked_mode = all_data[(all_data['Sex']=='female') & all_data['Pclass']==1]['Embarked'].value_counts().index[0]



#Filling the missing values

train_data['Embarked'] = train_data['Embarked'].fillna(embarked_mode)



#Features Cabin and Ticket have not been dropped yet because they helped find information about the missing embarked data

#They completed their purpose and may now be dropped

train_data.drop(['Cabin', 'Ticket'], axis=1, inplace=True)

test_data.drop(['Cabin', 'Ticket'], axis=1, inplace=True)
#Displaying the missing fare data point

test_data[test_data['Fare'].isnull()]
#Finding the median for fare, only considering passengers in 3rd class

class3_fare_median = all_data[all_data['Pclass']==3]['Fare'].median()



#Filling the missing data point with the median

test_data['Fare'] = test_data['Fare'].fillna(class3_fare_median)
#Define the categories that need to be onehotencoded

one_hot_feat = ['Sex', 'Embarked', 'Title', 'Married', 'Fam_Size']



#Use pandas get_dummies to onehot encode the features for training/test sets

#drop_first is on to eliminate multicollinearity

OH_features_train = pd.get_dummies(train_data[one_hot_feat], drop_first=True)

OH_features_test = pd.get_dummies(test_data[one_hot_feat], drop_first=True)



#Concatenate onehot encoded features to training/test sets

train_data = pd.concat([train_data,OH_features_train],axis=1)

test_data = pd.concat([test_data,OH_features_test],axis=1)



#Drop the categorical features that are not one hot encoded

train_data.drop(one_hot_feat, axis=1, inplace = True)

test_data.drop(one_hot_feat, axis=1, inplace = True)
#Defining X and y

X_train = train_data.drop('Survived',axis=1)

X_test = test_data

y = train_data['Survived']



#Defining the seed used for all random_state arguments

SEED=42



#Making arrays of the X and y dataframes - Used for cross-validation in ensemble models

X_train_np_array = X_train.values

y_np_array = y.values

X_test_np_array = X_test.values



#Creating train/test splits of the X and y dataframes - used for validation/predictions in the logit model

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y, test_size=0.2, random_state=SEED)
#Statsmodels does not automatically add an intercept with their logit model - do this manually

logit_X_train_with_intercept = sm.add_constant(X_train_split)



#Fit the model to the training data

logit = sm.Logit(y_train_split, logit_X_train_with_intercept).fit()



#Statsmodels provides a nice summary of the model, with p-values

logit.summary()
#Must add intercept to validation data before using logit model

logit_X_val_with_intercept = sm.add_constant(X_val_split)



#Calculating probabilities of survival for each passenger

logit_X_val_prob = logit.predict(logit_X_val_with_intercept)



#Make function to round probabilities to finalized answers. Cutoff point is: P(X) = 0.5.

def logistic(Column):

    if Column <= 0.5:

        return 0

    else: 

        return 1



#Apply function to get validation data predictions

logit_X_val_pred = logit_X_val_prob.apply(logistic)
#Creating a function to report metrics for logit model

#Function will return confusion matrix, classification report, ROC curve and ROC-AUC Score

def metrics(model_target_probabilities, model_target_predictions):

    print('Confusion Matrix:\n', confusion_matrix(y_val_split, model_target_predictions))

    print('\nClassification Report:\n', classification_report(y_val_split, model_target_predictions))

        

    #Creating tpr and fpr for ROC curve

    false_pos_rates, true_pos_rates, thresholds = roc_curve(y_val_split, model_target_probabilities)

    

    #Calculating ROC_AUC score

    model_roc_auc_score = auc(false_pos_rates, true_pos_rates).round(4)

    print("\n\nAUC_Score: ", model_roc_auc_score)

    

    

    #Plotting ROC Curve

    plt.figure(figsize=(15,10))

    plt.ylabel('TPR', size=20)

    plt.xlabel('FPR', size=20)

    plt.title('ROC Curve', size = 35)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8, label='Random Guessing')

    sns.lineplot(x=false_pos_rates,y=true_pos_rates, label=f'ROC Curve (AUC = {model_roc_auc_score})')

    plt.legend(loc='lower right')

    

metrics(logit_X_val_prob, logit_X_val_pred)
#Make predictions for submission - add intercept to test data

logit_test_with_intercept = sm.add_constant(X_test)

logit_test_prob = logit.predict(logit_test_with_intercept)

logit_test_pred = pd.DataFrame(logit_test_prob.apply(logistic))



#Save predictions into a CSV that can be uploaded to kaggle

logit_entry = pd.concat([test_PassengerID,logit_test_pred],axis=1)

logit_entry.columns = ['PassengerId', 'Survived']

logit_entry = logit_entry.set_index('PassengerId')

logit_entry.to_csv('Logit Entry')
#Define a range of values for all the hyperparameters being optimized

n_est = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(1, 20, num = 10)]

max_depth.append(None)  #Manually add no limit for tree depth - this is standard input

min_samples_split = [2,4,6,8]

min_samples_leaf = [1,2,3]



#Create the random grid by creating a dictionary of the inputs.

rf_param_grid = {'n_estimators':n_est,

                'max_features':max_features,

                'max_depth':max_depth,

                'min_samples_split':min_samples_split,

                'min_samples_leaf':min_samples_leaf}



#Create the model

rf = RandomForestClassifier()
#Instantiate RandomizedSearchCV object

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_grid,

                               n_iter=200, cv=3, verbose=2, random_state=SEED, n_jobs=-1)



#Fit the data to the object and perform search

rf_random.fit(X_train,y)



#Best parameters

rf_best_parameters = rf_random.best_params_

print('Best parameters:\n', rf_best_parameters)
#Define a range of values for all the hyperparameters being optimized

n_est = [int(x) for x in np.linspace(start = 300, stop = 700, num = 16)]  #Intervals of 25

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(15, 25, num = 10)]  

max_depth.append(None)  #Manually add no limit for tree depth - this is standard input

min_samples_split = [5,6,7]

min_samples_leaf = [2,3,4]



#Create the random grid by creating a dictionary of the inputs.

rf_param_grid = {'n_estimators':n_est,

                'max_features':max_features,

                'max_depth':max_depth,

                'min_samples_split':min_samples_split,

                'min_samples_leaf':min_samples_leaf}





#Create the model

rf = RandomForestClassifier()





#Instantiate GriddSearchCV object

rf_random = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=3, verbose=2, n_jobs=-1)



#Fit the data to the gridsearch object and perform search

rf_random.fit(X_train,y)



#Best parameters

rf_best_parameters = rf_random.best_params_

print('Best parameters:\n', rf_best_parameters)
#Create the model with the optimized variables

rf = RandomForestClassifier(n_estimators=353, max_depth=17, max_features='auto', 

                            min_samples_leaf=3, min_samples_split=7,random_state=SEED)



#Fit the model to the training data and get validation predictions/probability which are used for classification metrics

rf.fit(X_train_split, y_train_split)
#Creating a function that crossvalidates data for a model and reports ROC Curve

#Function takes the following as arguments: number of splits for cv, model optimized w/ h.param., modelname (for title of ROC_curve)

def cv_roc_score(num_of_splits, model, title):



    #First define the cross-validation object and the model

    cv = StratifiedKFold(n_splits=num_of_splits)

    classifier = model



    #Next create place holders for the true positive rates and AUC scores for each experiment/split

    tprs = []

    aucs = []

    mean_fpr = np.linspace(0, 1, 100)





    #Creating plot and iterating to find tpr and AUC score for each split

    fig, ax = plt.subplots(figsize=(15,15))

    for i, (train, test) in enumerate(cv.split(X_train_np_array, y_np_array)):

        classifier.fit(X_train_np_array[train], y_np_array[train])

        viz = plot_roc_curve(classifier, X_train_np_array[test], y_np_array[test],

                             name='ROC fold {}'.format(i),

                             alpha=0.3, lw=1, ax=ax)

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)

        interp_tpr[0] = 0.0

        tprs.append(interp_tpr)

        aucs.append(viz.roc_auc)



    #Plotting line that represents ROC curve if random guessing were implemented 

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',

            label='Chance', alpha=.8)



    #Creating the ROC curves, mean AUC score and standard deviation of the AUC score

    mean_tpr = np.mean(tprs, axis=0)

    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)

    std_auc = np.std(aucs)

    ax.plot(mean_fpr, mean_tpr, color='b',

            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),

            lw=2, alpha=.8)



    #Creating grey space for standard deviation of ROC curve

    std_tpr = np.std(tprs, axis=0)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,

                    label=r'$\pm$ 1 std. dev.')





    #Setting axis limits, title and legend

    ax.set_xlim([-0.05,1.05])

    ax.set_ylim([-0.05,1.05])

    ax.set_title(f"ROC Curve - {title}", fontsize=20)

    ax.legend(loc="lower right")



    

    

#Defining paramters for cv_roc_score function

num_splits = 5

title = 'RandomForestClassifier'

rf_classifier = RandomForestClassifier(n_estimators=353, max_depth=17, max_features='auto', 

                                min_samples_leaf=3, min_samples_split=7,random_state=SEED)



#Implementing cv_roc_score function

cv_roc_score(num_of_splits=num_splits, model=rf_classifier, title=title)
#Make a function that creates cross-validation predictions

#The function will take 2 arguments: the number of splits used in cross-validation, the model

#Function will return predictions as a dataframe 

def cv_pred(num_of_splits, model):

    #Creating a dataframe to store the probabilities from each split

    cv_prob_np_array = np.zeros(shape=(X_test.shape[0],num_of_splits))

    cv_prob = pd.DataFrame(cv_prob_np_array, columns=[f'Fold_{split}' for split in range(num_of_splits)])

    

    #First define the cross-validation object and the model

    cv = StratifiedKFold(n_splits=num_of_splits)

    classifier = model

    

    for split, (train, test) in enumerate(cv.split(X_train_np_array, y_np_array)):

        classifier.fit(X_train_np_array[train], y_np_array[train])

        cv_prob.loc[:,f'Fold_{split}'] = classifier.predict_proba(X_test_np_array)[:,1]

    

    avg_prob = cv_prob.mean(axis=1)

    

    #Create function to round probabilities into predictions

    def predictions(Column):

        if Column <= 0.5:

            return 0

        else: 

            return 1

    

    #Use function to make predictions

    final_predictions = avg_prob.apply(predictions)

    

    return final_predictions







#Defining paramters for cv_pred function

num_splits = 5

title = 'RandomForestClassifier'

rf_classifier = RandomForestClassifier(n_estimators=353, max_depth=17, max_features='auto', 

                                min_samples_leaf=3, min_samples_split=7,random_state=SEED)



#Implementing cv_pred function

rf_predictions = cv_pred(num_of_splits=num_splits, model=rf_classifier)
#Gathering and cleaning data for submission

rf_entry = pd.concat([test_PassengerID, rf_predictions],axis=1)

rf_entry.columns = ['PassengerId', 'Survived']

rf_entry = rf_entry.set_index('PassengerId')

rf_entry.to_csv('Random Forest Entry')
#Define a range of values for all the hyperparameters being optimized

eta = [0.05,0.1,0.15,0.2,0.25]

n_est = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 18)] #Increments of 50

max_depth = [int(x) for x in range(1,11)]

max_depth.append(None)  #Manually add no limit - this is standard input

min_child_weight = [int(x) for x in range(1,6,2)]

gamma = [x/10.0 for x in range(0,5)]



#Create the random grid by creating a dictionary of the inputs.

xgb_param_grid = {'learning_rate':eta,

                'n_estimators':n_est,

                'max_depth':max_depth,

                'min_child_weight':min_child_weight,

                'gamma':gamma}





#Create the model

xgb = XGBClassifier()





#Instantiate RandomizedSearchCV object with model and parameters

xgb_random = RandomizedSearchCV(estimator=xgb, param_distributions=xgb_param_grid,

                               n_iter=200, cv=3, verbose=2, random_state=SEED, n_jobs=-1)



#Fit the data to the object and perform search

xgb_random.fit(X_train,y)



#Best parameters

xgb_best_parameters = xgb_random.best_params_

print('Best parameters:\n', xgb_best_parameters)
#Define a range of values for all the hyperparameters being optimized

eta = [0.1,0.15,0.2,0.25,0.3]

n_est = [int(x) for x in np.linspace(start = 700, stop = 1300, num = 12)] #Increments of 50

max_depth = [int(x) for x in np.linspace(start=3, stop=9, num=6)]

max_depth.append(None)  #Manually add no limit - this is standard input

min_child_weight = [int(x) for x in range(3,7,4)]

gamma = [x/10.0 for x in range(1,5)]



#Create the random grid by creating a dictionary of the inputs.

xgb_param_grid = {'learning_rate':eta,

                'n_estimators':n_est,

                'max_depth':max_depth,

                'min_child_weight':min_child_weight,

                'gamma':gamma}



#Create the model

xgb = XGBClassifier()





#Instantiate GriddSearchCV object

xgb_random = GridSearchCV(estimator=xgb, param_grid=xgb_param_grid, cv=3, verbose=2, n_jobs=-1)



#Fit the data to the object and perform search

xgb_random.fit(X_train,y)



#Best parameters

xgb_best_parameters = xgb_random.best_params_

print('Best parameters:\n', xgb_best_parameters)
#Defining paramters for cv_roc_score function

num_splits = 5

title = 'XGBClassifier'

xgb_classifier = XGBClassifier(n_estimators=700, max_depth=6, gamma=0.4, eta=0.25, min_child_weight=3, random_state=SEED)



#Implementing cv_roc_score function

cv_roc_score(num_of_splits=num_splits, model=xgb_classifier, title=title)
#Defining paramters for cv_pred function

num_splits = 5

xgb_classifier = XGBClassifier(n_estimators=700, max_depth=6, gamma=0.4, eta=0.25, min_child_weight=3, random_state=SEED)



#Implementing cv_pred function

xgb_predictions = cv_pred(num_of_splits=num_splits, model=xgb_classifier)
#Gathering and cleaning data for submission

xgb_entry = pd.concat([test_PassengerID, xgb_predictions],axis=1)

xgb_entry.columns = ['PassengerId', 'Survived']

xgb_entry = xgb_entry.set_index('PassengerId')

xgb_entry.to_csv('Extreme Gradient Boost Entry')