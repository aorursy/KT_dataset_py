# Introducing libs to import, read, wrangle and explore the data.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re
# Reading in the data
data = pd.read_csv('../input/data.csv')
# Looking at the names of the features 
data.columns
# Taking a look at how to dataset
data.head()
# Checking data types of columns and number of null values(if any)
data.info()
del data['Unnamed: 32']
# As the 'diagnosis' column is our response variable, we must create a numerical feature for it where 1 stands for 'M' and 0 is
# for 'B'
data['y']= data['diagnosis'].apply(lambda x : 1 if x =='M' else 0)
# Checking if classes are unbalanced or balanced
data['diagnosis'].value_counts()
# Creating a plot to show imbalance in classes
sns.countplot(data.diagnosis)
plt.title('Imbalance in classes')
# Checking correlation of features with response variable
print('Features with the max correlation with y :')
print(data.corr()['y'][:-1].sort_values(ascending = False)[:10]) # top 10 features with highest correlations
print('\n')
# Features with lowest correlation (These are not ordered though as mod of negative values hasn't been taken. We must take the mod to measure strength of the neatively correlated features)
print('Features with the least correlation with y :')
print(data.corr()['y'][:-1].sort_values()[:10]) 
# No insights here. Just shows that there is good differnce between most of the means of different features wrt to the classes.
# Also as the range of different features looks to be really different we will have to normalize/scale them.
data.groupby('diagnosis').mean().iloc[:,1:]
# Lets visualize some of the above differences too 
fig,ax = plt.subplots(5,1,figsize = (7,25))
sns.boxplot('diagnosis','radius_mean',data = data,ax=ax[0])
ax[0].set_title('Diagnosis VS Mean of radius_mean')
sns.boxplot('diagnosis','area_mean',data = data,ax=ax[1])
ax[1].set_title('Diagnosis VS Mean of area_mean')
sns.boxplot('diagnosis','smoothness_mean',data = data,ax=ax[2])
ax[2].set_title('Diagnosis VS Mean of smoothness_mean')
sns.boxplot('diagnosis','fractal_dimension_mean',data = data,ax=ax[3])
ax[3].set_title('Diagnosis VS Mean of fracal_dimension_mean')
sns.boxplot('diagnosis','symmetry_worst',data = data,ax=ax[4])
ax[4].set_title('Diagnosis VS Mean of symmetry_worst')
# Checking distributions of the features. Most of them are highly right/positively skewed.
# We must apply logarithmic transformation to them to make them more normal and reduce the rightwards skewness. 
data.iloc[:,:-1].hist(figsize= (20,20))
# Checking skewness strength and direction
skew_before_log = data.loc[:,'radius_mean':'fractal_dimension_worst'].skew()
print(skew_before_log)
# Make X and y to make it easy to put into train and test categories later
y = data.loc[:,'y']
X = data.iloc[:,2:-1]
# log transform ( We add the '0.1' to x in log to avoid getting : log(0) )
X_log = X.apply(lambda x: np.log(x + 0.1))

# To check how the feature distributions look after log transform
X_log.hist(figsize = (20,20))
skew_after_log = X_log.skew()
print('Amount of skewness reduced :')
print('\n')
print(skew_before_log - skew_after_log)
# Scale the entire X ds
# This step also brings back all the features within the (0,1) range removing any negative values that were created due to taking log
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_log)
# Import Preprocessing, metrics and model optimization algos
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, fbeta_score
# Import ML Classification algos
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
# Dividing X and y into the training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,random_state = 0)
# Initializing Different Classifiers
clf_dt = DecisionTreeClassifier()
clf_svc = SVC(gamma = 'auto')
clf_rfc = RandomForestClassifier(n_estimators = 10)
clf_ada = AdaBoostClassifier()
clf_lr = LogisticRegression()
# Making a scoring function to use for CV and GridSearchCV
f_beta_scorer = make_scorer(fbeta_score,beta = 2)
# Making a pipeline that prints classifier name, Cross validation F2 score and F2 score on test set. 
# This function will also return score to help us plot and visualize the scores.
# Also, we will be using F2 score as this problem has unbalanced classes and we need to focus more on recall than precision. An 
# F2 score places more importance on recall value than precision.

def classifier(initialized_clf, X_train, y_train, X_test, y_test):
    
    initialized_clf.fit(X_train,y_train)
    cv_scores = [] 
    
    for i in range(5): # Performing CV 5 times so that we get the means of scores for every iteration for a great estimate. 
        cv_scores.append(cross_val_score(initialized_clf,X_train,y_train,scoring = f_beta_scorer,cv = 5).mean()) 
        
    print(initialized_clf)
    print('\n')
    print('Cross Validation F2 Score is :' , np.mean(cv_scores))
    
    preds = initialized_clf.predict(X_test)
    
    print('F2 score on test set is :', fbeta_score(y_test,preds,2))
    print('\n')
    
    cv_scores_mean_plot = np.mean(cv_scores)
    test_scores_plot = fbeta_score(y_test,preds,2)
    
    return (cv_scores_mean_plot),(test_scores_plot)

# Making a function to plot 2 types of scores: Mean F2 score on 5-fold Cross-Validation and F2 score on Test sets
def plot_classifier(clf_names,clf_cv_score,test_score):
    
    fig, ax = plt.subplots(1,2,figsize = (20,8))
    
    sns.barplot(clf_names,clf_cv_score,ax= ax[0]) 
    ax[0].set_title('F2 CV Scores')
    ax[0].set_yticks(np.linspace(0,1,11))
    ax[0].set_ylim(0,1)
    
    sns.barplot(clf_names,test_score,ax= ax[1])
    ax[1].set_title('F2 test Score')
    ax[1].set_ylim(0,1)
    ax[1].set_yticks(np.linspace(0,1,11))
# Create a list with all initialized models
clfs = [clf_dt, clf_svc, clf_rfc, clf_ada, clf_lr]

# Write code to call both plotting and model testing functions
names = []                                   # List to store the names of the models for plotting
values_cv_scores = np.zeros(len(clfs))       # array of zeros to store CV scores
values_test_scores = np.zeros(len(clfs))     # array to of zeros to store testing scores

for i in range(len(clfs)):                  # This will iterate over the 'clfs' list and run the classifier function on each of the clfs
    values_cv_scores[i],values_test_scores[i] = classifier(clfs[i],X_train,y_train,X_test,y_test)
    print('--------------------------')
    print('\n')
    names.append(re.match(r'[A-Za-z]+',str(clfs[i]))[0]) # Extracting the name of the model by extracting all letters before any special character
    
plot_classifier(names,values_cv_scores,values_test_scores)
# Model Optimization
# 1) RandomForestClassifier :: This is done just for fun and experimentation
clf_rfc = RandomForestClassifier(random_state = 0)

params_rfc = {'criterion':['gini','entropy'],'n_estimators':[5,6,7,8,9,10,11,12,15,20],'min_samples_split':[2,3,4,5]}
best_rfc = GridSearchCV(clf_rfc,params_rfc,f_beta_scorer,cv = 5,iid = 'True')
print('RandomForest')
best_rfc.fit(X_train,y_train)
print('Its best parameters are :',best_rfc.best_params_)
print('F2 score with best parameters :',best_rfc.best_score_)
best_rfc_preds = best_rfc.predict(X_test)
print('F2 score on test set: ',fbeta_score(y_test,best_rfc_preds,2))
print('\n')

# 2) AdaBoostClassifier

clf_dt = DecisionTreeClassifier()
ABC = AdaBoostClassifier(base_estimator = clf_dt,random_state = 0)

params_ada = {'base_estimator__criterion':['gini','entropy'],'base_estimator__max_depth':[1,2,3,4,5,10,15,20],'n_estimators': [10,20,30,40,100,50]}
best_ada = GridSearchCV(ABC,params_ada,f_beta_scorer,cv= 5,iid = 'True')
best_ada.fit(X_train,y_train)
print('AdaBoost')
print('Its best parameters are :',best_ada.best_params_)
print('F2 score with best parameters :',best_ada.best_score_)
best_ada_preds = best_ada.predict(X_test)
print('F2 score on test set :',fbeta_score(y_test,best_ada_preds,2))
print('\n')

# 3) Logistic Regression

params_lr = {'penalty':['l1','l2'],'C':[0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2]}
best_lr = GridSearchCV(clf_lr,params_lr,f_beta_scorer, cv = 5,iid = True)
best_lr.fit(X_train,y_train)
print('Logistic Regression')
print('Its best parameters are :',best_lr.best_params_)
print('F2 score with best parameters :',best_lr.best_score_)
best_lr_preds = best_lr.predict(X_test)
print('F2 score on test set :',fbeta_score(y_test,best_lr_preds,2))
print('\n')

# We will study feature importance for AdaBoostClassifier
feature_names = X.columns.values
feature_values = clf_ada.feature_importances_
# Sorting the features according to their values from max to min
features_imp = sorted(list(zip(feature_values,feature_names)),reverse = True)
fig,ax = plt.subplots(1,1,figsize = (10,10))
sns.barplot(np.array(features_imp)[:,0].astype('float'), np.array(features_imp)[:,1].astype('str'), orient= 'h')
ax.set_title('Feature Importance according to AdaboostClassifier')
print('We can see the 5 most important features according to the AdaBoostClassifier are : \n')
for i in list(np.array(features_imp[:5])[:,1]):
    print(i)
from sklearn.decomposition import PCA
# We create a loop to print out the cumulative sum of the 'explained variance ratio' on using different number of principal 
# components 

for i in range(1, X_train.shape[1] + 1):
    pca = PCA(i,random_state = 0)
    X_transformed = pca.fit_transform(X_train)
    print('for {}, cumulative sum of explained variance ratio of all components is : {}'.format(i,np.cumsum(pca.explained_variance_ratio_)[-1]))
    print('\n')
    
pca = PCA(14,random_state = 0)
X_train_transformed = pca.fit_transform(X_train)
X_test_transformed = pca.transform(X_test)
# We use GridSearchCV again but now on the tranformed data with the principal components
clf_dt = DecisionTreeClassifier()
ABC = AdaBoostClassifier(base_estimator = clf_dt,random_state = 0)

params_ada = {'base_estimator__criterion':['gini','entropy'],'base_estimator__max_depth':[1,2,3,4,5,10,15,20],'n_estimators': [10,20,30,40,100,50]}
best_ada = GridSearchCV(ABC,params_ada,f_beta_scorer,cv= 5,iid = True)
best_ada.fit(X_train_transformed,y_train)
print('AdaBoost')
print('The best parameters are :',best_ada.best_params_)
print('F2 score on using the best parameters :',best_ada.best_score_)
best_ada_preds = best_ada.predict(X_test_transformed)
print('F2 score on the test set :',fbeta_score(y_test,best_ada_preds,2))
print('\n')