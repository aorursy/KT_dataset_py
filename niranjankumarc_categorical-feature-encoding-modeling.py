import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use("ggplot")
#read the data



train_df = pd.read_csv("../input/cat-in-the-dat/train.csv")

test_df = pd.read_csv("../input/cat-in-the-dat/test.csv")

submission_df = pd.read_csv("../input/cat-in-the-dat/sample_submission.csv")
train_df.head()
print("Number of observations in the train data: ", train_df.shape[0])

print("Number of observations in the train data: ", test_df.shape[0])

print("Number of columns in the train data: ", train_df.shape[1])
#checking for missing values



train_df.isna().sum()
#basic stats of the data

train_df.describe()
#basic stats of the string data



train_df.describe(include = "object")
def NumericalVariables_targetPlots(df,segment_by,target_var = "Attrition"):

    """A function for plotting the distribution of numerical variables and its effect on attrition"""

    

    fig, ax = plt.subplots(ncols= 2, figsize = (14,6))    



    #boxplot for comparison

    sns.countplot(x = segment_by, hue = target_var, data=df, ax=ax[0])

    ax[0].set_title("Comparision of " + segment_by + " vs " + target_var)

    

    #distribution plot

    ax[1].set_title("Distribution of "+segment_by)

    ax[1].set_ylabel("Frequency")

    sns.distplot(a = df[segment_by], ax=ax[1], kde=False)

    

    plt.show()
def CategoricalVariables_targetPlots(df, segment_by,invert_axis = False, target_var = "target"):

    

    """A function for Plotting the effect of variables(categorical data) on attrition """

    

    fig, ax = plt.subplots(ncols= 2, figsize = (14,6))

    

    #countplot for distribution along with target variable

    #invert axis variable helps to inter change the axis so that names of categories doesn't overlap

    if invert_axis == False:

        sns.countplot(x = segment_by, data=df,hue="target",ax=ax[0])

    else:

        sns.countplot(y = segment_by, data=df,hue="target",ax=ax[0])

        

    ax[0].set_title("Comparision of " + segment_by + " vs " + "Target")

    

    #plot the effect of variable on attrition

    if invert_axis == False:

        sns.barplot(x = segment_by, y = target_var ,data=df,ci=None)

    else:

        sns.barplot(y = segment_by, x = target_var ,data=df,ci=None)

        

    ax[1].set_title("Target rate by {}".format(segment_by))

    ax[1].set_ylabel("Relative Target Representation")

    plt.tight_layout()



    plt.show()
#analyzing the variable "bin_0"



NumericalVariables_targetPlots(train_df, "bin_0", "target")
#analyzing the variable "bin_1"



NumericalVariables_targetPlots(train_df, "bin_1", "target")
#analyzing the variable "bin_2"



NumericalVariables_targetPlots(train_df, "bin_2", "target")
#analyzing the variable "bin_3"



CategoricalVariables_targetPlots(train_df, "bin_3")
#analyzing the variable "bin_4"



CategoricalVariables_targetPlots(train_df, "bin_4")
#analyzing the variable "nom_0"



CategoricalVariables_targetPlots(train_df, "nom_0")
#analyzing the variable "nom_1"



CategoricalVariables_targetPlots(train_df, "nom_1")
#analyzing the variable "nom_2"



CategoricalVariables_targetPlots(train_df, "nom_2")
#analyzing the variable "nom_3"



CategoricalVariables_targetPlots(train_df, "nom_3")
#analyzing the variable "nom_4"



CategoricalVariables_targetPlots(train_df, "nom_4")
#changing the color scheme

plt.style.use("fivethirtyeight")
#analyzing the variable "ord_0"

NumericalVariables_targetPlots(train_df, "ord_0", "target")
#analyzing the variable "ord_1"

CategoricalVariables_targetPlots(train_df,"ord_1")
#analyzing the variable "ord_2"

CategoricalVariables_targetPlots(train_df,"ord_2")
#analyzing the variable "day"



NumericalVariables_targetPlots(train_df, "day", "target")
#analyzing the variable "day"



NumericalVariables_targetPlots(train_df, "month", "target")
plt.style.use("seaborn")

train_df.target.value_counts(normalize = True).plot(kind = "barh")

plt.title("Distribution of Target Variable")

plt.show()
# changing the binary values T/F to 1/0 and Y/N to 1/0.



train_df["bin_3"] = train_df["bin_3"].apply(lambda x: 0 if x == 'F' else 1)

train_df["bin_4"] = train_df["bin_4"].apply(lambda x: 0 if x == 'N' else 1)



#test data

test_df["bin_3"] = test_df["bin_3"].apply(lambda x: 0 if x == "F" else 1)

test_df["bin_4"] = test_df["bin_4"].apply(lambda x: 0 if x == "N" else 1)
nominal_cat_var = ["nom_" + str(i) for i in range(0,5)]



train_temp_df = pd.get_dummies(train_df, columns = nominal_cat_var, drop_first = True)

test_temp_df = pd.get_dummies(test_df, columns = nominal_cat_var, drop_first = True)
print("Number of columns in the train data after one hot encoding: ", train_temp_df.shape[1])

print("Number of columns in the test data after one hot encoding: ", test_temp_df.shape[1])
#for ordinal variables - ord_1 and ord_2. we will manually replace the columns with integer values



ord1_mapping = {'Grandmaster': 5, 'Expert': 4 , 'Novice':1 , 'Contributor':2 , 'Master': 3}

ord2_mapping = {'Cold': 2, 'Hot':4, 'Lava Hot': 6, 'Boiling Hot': 5, 'Freezing': 1, 'Warm': 3}
train_temp_df["ord_1"] = train_temp_df["ord_1"].map(ord1_mapping)

train_temp_df["ord_2"] = train_temp_df["ord_2"].map(ord2_mapping)



test_temp_df["ord_1"] = test_temp_df["ord_1"].map(ord1_mapping)

test_temp_df["ord_2"] = test_temp_df["ord_2"].map(ord2_mapping)
#converting "ord_3" and "ord_4" as category type and getting the category codes.



for col in ["ord_3", "ord_4"]:

    train_temp_df[col] = train_temp_df[col].astype('category')

    ord_map = dict( zip(train_temp_df[col], train_temp_df[col].cat.codes))

    train_temp_df[col] = train_temp_df[col].map(ord_map)

    test_temp_df[col] = test_temp_df[col].map(ord_map)

    train_temp_df[col] = train_temp_df[col].astype(int)
import string
# Encode 'ord_5' using ACSII values

# Source:- https://www.kaggle.com/c/cat-in-the-dat/discussion/105702#latest-607652



# # Option 1: Add up the indices of two letters in string.ascii_letters

train_temp_df['ord_5_oe_add'] = train_temp_df['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))

test_temp_df['ord_5_oe_add'] = test_temp_df['ord_5'].apply(lambda x:sum([(string.ascii_letters.find(letter)+1) for letter in x]))



train_temp_df.drop('ord_5', axis=1, inplace=True)

test_temp_df.drop('ord_5', axis=1, inplace=True)
nominal_highcat_var  = ["nom_" + str(i) for i in range(5,10)]

nominal_highcat_var
import category_encoders  as ce

from sklearn.feature_extraction import FeatureHasher

#I got this solution from @kabure and @Giba



for col in nominal_highcat_var:

    train_temp_df[f'hash_{col}'] = train_temp_df[col].apply( lambda x: hash(str(x)) % 5000 )

    test_temp_df[f'hash_{col}'] = test_temp_df[col].apply( lambda x: hash(str(x)) % 5000 )  
#drop the variables after transformation

   

train_temp_df.drop(['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'], axis=1, inplace=True)

test_temp_df.drop(['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'], axis=1, inplace=True)
train_temp_df.head()
test_temp_df.head()
X_train = train_temp_df.drop(['id', 'target'],axis = 1)

y_train = train_temp_df['target']

X_test = test_temp_df.drop(['id'], axis = 1)
print('Input training dimension:', X_train.shape)

print('Test data dimension:', X_test.shape)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from sklearn.pipeline import make_pipeline

from sklearn.metrics import roc_auc_score,roc_curve
#declare hyperparameters dictionary



pipelines = {

    'logistic' : make_pipeline(LogisticRegression(random_state = 123)),

    'decisiontree' : make_pipeline(DecisionTreeClassifier(random_state = 123)),

    'randomforest': make_pipeline(RandomForestClassifier(random_state = 123)),

    'adaboost': make_pipeline(AdaBoostClassifier(random_state = 123))

}
#get the all possible parameters for a model



#pipelines["adaboost"].get_params().keys()
#logistic hyperparameters

logistic_hyperparameters = {

    'logisticregression__C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000],

    'logisticregression__penalty' : ['l1', 'l2']

}



#ada boost hyperparameters

ab_hyperparameters = {

    'adaboostclassifier__n_estimators' : [100, 200, 400, 600, 800],

    'adaboostclassifier__learning_rate' : [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1],

    'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R']

}



decisiontree_hyperparameters = {

    "decisiontreeclassifier__max_depth": np.arange(3,12),

    "decisiontreeclassifier__max_features": np.arange(3,10),

    "decisiontreeclassifier__min_samples_split": [2,3,4,5,6,7,8,9,10,11,12,13,14,15],

    "decisiontreeclassifier__min_samples_leaf" : np.arange(1,3)

}



#random forest hyperparameters



rf_hyperparameters = {

    'randomforestclassifier__n_estimators' : [100, 200, 400, 600, 800],

    'randomforestclassifier__max_features' : ['auto', 'sqrt', 'log2'],

    'randomforestclassifier__max_depth' : [int(x) for x in np.linspace(3, 10, num = 1)],

    'randomforestclassifier__min_samples_split' : np.arange(2, 10)

}
hyperparameters = {

    'adaboost' : ab_hyperparameters,

    'randomforest' : rf_hyperparameters,

    'logistic' : logistic_hyperparameters,

    'decisiontree' : decisiontree_hyperparameters

}
k = StratifiedKFold(n_splits=3, random_state=123)
fitted_models = {}



for name, pipeline in pipelines.items():

    print("------- ", name, ' ---------')

    #create a cross validation object from pipelines and hyperparameters

    model = GridSearchCV(pipeline, hyperparameters[name], cv = k , n_jobs=-1,return_train_score=True, verbose = 2,scoring="roc_auc")

    

    #fit model on X train and y train

    model.fit(X_train, y_train)

    

    fitted_models[name] = model

    

    # Print '{name} has been fitted'

    print(name, 'has been fitted.')
##Here we are evaluating based on model roc score



results, names  = [], [] 

best_estimator_dict = {}



for name,model_built in fitted_models.items():

    names.append(name)

    results.append(np.round(model_built.best_score_,4))

    print("Mean AUC Score of "+ name + " :", np.round(model_built.cv_results_["mean_train_score"].mean(),4))

    print("Best AUC Score of "+ name + " :", np.round(model_built.best_score_,4))

    best_estimator_dict[model_built] = np.round(model_built.best_score_,4)
roc_auc_importance = pd.Series(results, names)

roc_auc_importance.plot(kind='barh', cmap = "viridis")

plt.title("Mean AUC_ROC Score Based Cross Validation for Different Models")

plt.xlabel("AUC ROC Score")

plt.ylabel("Model Name")

plt.show()
#finding the best model based on AUC ROC Score



best_fittedobject = max(best_estimator_dict, key=best_estimator_dict.get)
#get the best model from the best fittedobject



best_model = best_fittedobject.best_estimator_.steps[0][1]

best_model
#best model params



best_fittedobject.best_params_
#plot roc curve



def plot_roc( actual, probs ):

    fpr, tpr, thresholds = roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = roc_auc_score( actual, probs )

    plt.figure(figsize=(7, 7))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic plot')

    plt.legend(loc="lower right")

    plt.show()
#AUC ROC Curve on Training Data based on 



plot_roc(y_train, best_model.predict_proba(X_train)[:,1])
#plot the feature importance



try:

    feat_importances = pd.Series(best_model.feature_importances_, index=X_train.columns)

    feat_importances.nlargest(5).plot(kind='barh')

    plt.title("Feature Importance")

    plt.xlabel("Relative Importance")

    plt.ylabel("Variable Name")

    plt.show()

except:

    print("Best Model doesn't support Feature Importance Plot (Logisitc Regression)")
best_fittedobject.cv_results_["mean_train_score"].mean()
#predictions



y_preds = best_model.predict_proba(X_test)[:,1]
#appending the predictions to submission data



submission_df["target"] = y_preds

submission_df.to_csv('best_submission.csv',header=True, index=False)