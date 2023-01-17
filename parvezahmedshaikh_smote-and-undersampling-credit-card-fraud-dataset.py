import pandas as pd # data processing, uploading csv and working with dataframe



#graph plotting library

import seaborn as sns

import matplotlib.pyplot as plt



# Classifier Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier



#splitting the dataset

from sklearn.model_selection import StratifiedKFold



#Over Sampling and under sampling libraries

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import RandomUnderSampler



#Machine learning Pipeline libraries

from imblearn.pipeline import Pipeline



import warnings

warnings.filterwarnings("ignore");
#This dataset is a pre-processed dataset of the original creditcard fraud detection.

#In this dataset the amount and time column have already been scaled using sklearn.preprocessing.RobustScaler() method.

#The scaled_amount and scaled_time columns are scaled version of Amount and Time columns of the original dataset respectively.



df = pd.read_csv("../input/scaled-credit-card-fraud-detection/scaled_creditcard.csv")

df.head()
#printing the percentage of samples of the majority and minority classes.



print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100 , 2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100 , 2), '% of the dataset')
#Graphical distribution of the classes

df['Class'].value_counts().plot(kind='bar');
#First Method of solving this problem which comes in mind will be by 

#taking same number of records of the majority class as the minority class



# Taking all the records i.e. 492 of fraud classes(minority class).

fraud_df = df.loc[df['Class'] == 1]



# Taking the same number of records of the majority class(No Frauds)

non_fraud_df = df.loc[df['Class'] == 0][:492]



#concating the above dataframes to get a single dataframe

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])



# Shuffle dataframe rows

new_df = normal_distributed_df.sample(frac=1, random_state=42)
#Now we have exactly 50-50 percent rows of both majority and minority classes



print('No Frauds', round(new_df['Class'].value_counts()[0]/len(new_df) * 100 , 2), '% of the dataset')

print('Frauds', round(new_df['Class'].value_counts()[1]/len(new_df) * 100 , 2), '% of the dataset')
# Make sure we use the 50-50 percent sample rows in our correlation, as it is balanced dataset



f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,20))



# correlation using Original imbalanced dataframe

corr = df.corr()

sns.heatmap(corr, cmap='coolwarm_r',annot=True, annot_kws={'size':5}, ax=ax1)

ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)



# correlation using new balanced dataframe

sub_sample_corr = new_df.corr()

sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)

ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)

plt.show()
# Preparing data for ML algo by seperating target and the features.

X = new_df.drop('Class', axis=1)

y = new_df['Class']



# Our data is already scaled and balanced

# Splitting the data into training and test sets



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Let's implement simple classifiers



# creating a dictionary of ML objects

classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}



# appying cross validation

from sklearn.model_selection import cross_val_score



#iterating thru each of the algos in the "classifiers" dictionary

for key, classifier in classifiers.items():

    

    #Splitting the data into 5 parts using cv=5 parameter of cross_val_score method

    training_score = cross_val_score(classifier, X_train, y_train, cv=5)

    

    #The above line of code will return 5 accuracy scores

    #printing the mean of the accuracy metrics

    print("Classifiers: ", classifier.__class__.__name__, 

          "Has a training score of", round(training_score.mean() * 100, 2) , "% accuracy score")
# Use GridSearchCV to find the best parameters.

from sklearn.model_selection import GridSearchCV





# Logistic Regression 

log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)

grid_log_reg.fit(X_train, y_train)



# logistic regression best parameters.

log_reg = grid_log_reg.best_estimator_



#--------------------------



# KNeighborsClassifier

knears_params = {"n_neighbors": list(range(2,5,1))}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)

grid_knears.fit(X_train, y_train)



# KNears best estimator

knears_neighbors = grid_knears.best_estimator_



#--------------------------



# Support Vector Classifier

svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

grid_svc = GridSearchCV(SVC(), svc_params)

grid_svc.fit(X_train, y_train)



# SVC best estimator

svc = grid_svc.best_estimator_



#--------------------------



# DecisionTree Classifier

tree_params = {"max_depth": list(range(2,4,1)), "min_samples_leaf": list(range(5,7,1))}

grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)

grid_tree.fit(X_train, y_train)



# tree best estimator

tree_clf = grid_tree.best_estimator_



print("Classifers with their tuned parameters we got via GridSearchCV")

print(log_reg)

print(knears_neighbors)

print(svc)

print(tree_clf)

#dictionary of classifiers objects which we got in the above cell via GridSearchCV

#with the tuned parameters

tuned_classifiers = {

    "LogisiticRegression": log_reg,

    "Knears Neighbors": knears_neighbors,

    "Support Vector Classifier": svc,

    "DecisionTreeClassifier": tree_clf

}



#Making our Classifiers train with the tuned parameters

for key, classifier in tuned_classifiers.items():

    score = cross_val_score(classifier, X_train, y_train, cv=5)

    print(f'{key} Cross Validation Score: {round(score.mean() * 100, 2)}%')
# here we are using the imbalanced dataset 

X = df.drop('Class', axis=1)

y = df['Class']



# StratifiedKFold is used for cross validation

# This cross-validation object is a variation of KFold that returns stratified folds.

# The folds are made by preserving the percentage of samples for each class.

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)



# creating the object for Over Sampling the minority class

over = SMOTE(sampling_strategy=0.01,k_neighbors=5)



# creating the object for Under Sampling the majority class

under = RandomUnderSampler(sampling_strategy=0.5)



# iterating thru the dictionary of tuned classifiers

for key, classifier in tuned_classifiers.items():

    

    # list of steps to be provided to the ML Pipeline

    steps=[('o',over),('u',under),('model',classifier)]

    

    # Creating a ML Pipeline

    FiPipeline=Pipeline(steps=steps)

    

    # Cross validating the classifiers

    scores=cross_val_score(FiPipeline,X,y,cv=sss)

    

    # Printing the mean accuracy score

    print(f"Classifiers: {key} Has a training score of, {round(scores.mean() * 100, 2)} % accuracy score")

# taking a fold of a data to do GridSearchCV for best parameters for SMOTE

for train_index, test_index in sss.split(X, y):

    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]

    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]



# creating a ML pipeline

model = Pipeline([

        ('sampling', SMOTE(sampling_strategy=0.01)),

        ('Random',RandomUnderSampler(sampling_strategy=0.5)),

        ('classification', log_reg)

    ])



# Running GridSearchCV on our ML pipeline by varying the k_neighbors from 1 to 10

# to find the best k_neighbors parameter value

# NOTE :- in paramgrid use above key + 2 _ followed by parameter

SMOTE_KN = GridSearchCV(model,{'sampling__k_neighbors':list(range(1,10))})

SMOTE_KN.fit(original_Xtrain,original_ytrain)



print(SMOTE_KN.best_estimator_)

print(SMOTE_KN.best_score_)

print(SMOTE_KN.best_params_)



# Training logistic regression classifier using SMOTE best parameter value for k_neighbors parameter

log_reg_score = cross_val_score(SMOTE_KN.best_estimator_, original_Xtrain, original_ytrain, cv=5)



# printing the mean accuracy score

print(f'Logistic Regression Cross Validation Score: {round(log_reg_score.mean() * 100, 2)}%')