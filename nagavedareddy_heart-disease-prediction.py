# Importing all the tools we need.

# Regular EDA (exploratory Data analysis) and visualisation libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#To make our plots visible on notebook
%matplotlib inline  

# Models from Scikit-Learn.
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluations 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve
df = pd.read_csv('../input/heart-disease.csv')
df
df.shape # (rows, columns)

df.tail()

# How many of the each class
df['target'].value_counts()

# Visualising the targets
ax = sns.countplot(x='target',data=df);
#df['target'].value_counts().plot(kind='bar',color=['green','red']);
dict1={0:'Safe',1:'danger'}
ax.legend(dict1.values())

plt.title('Targets count 1/0');
df.info()

df.describe()

df.isna().sum()

df.sex.value_counts()

# compare target column with sex column
pd.crosstab(df.target,df.sex)
print(f'Chance of a male having a disease: {((93/207)*100):.2f}%')
print(f'Chance of a female having a disease: {((72/96)*100):.2f}%')

#create a plot of crosstab
pd.crosstab(df.target,df.sex).plot(kind='bar',figsize=(10,6),color=['salmon','lightblue']);
plt.title('Heart Disease frequency for sex');
#plt.xlabel('0 = No heart Disease, 1 = Heart Disease');
plt.ylabel('Count');
plt.legend(['Female','Male']);
plt.xticks([0,1],['No heart disease','heart disease'],rotation=0);
plt.figure(figsize=(10,6))

# scatter with positive values
plt.scatter(df.age[df.target==1],df.thalach[df.target==1],c='salmon');

# scatter with Negative values
plt.scatter(df.age[df.target==0],df.thalach[df.target==0],c='lightblue');

# customizing the plot
plt.title('Heart Disease as the function of age and Max Heart rate.');
plt.xlabel('Age');
plt.ylabel('Max Heart rate');
plt.legend(['Heart Disease','No Heart Disease']);
df.age.plot(kind='hist');

pd.crosstab(df.cp,df.target)

pd.crosstab(df.cp,df.target).plot.bar(color=['lightblue','salmon'],figsize=(10,6));
plt.title('Heart Disease frequency per Chest pain types');
plt.xlabel('Chest pain Type (CP)');
plt.ylabel('Frequency');
plt.legend(['No Heart Disease','Heart Disease']);
plt.yticks(rotation=0);
plt.xticks(rotation=0);
df.corr()

corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15,12))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt='.2f',
                 cmap='YlGnBu');
ax.set_title('Correlation Matrix of Independent Features and Dependent Feature',fontsize=20);
df.head()

X = df.drop('target',axis=1)
y = df['target']
X
y

# split into train and rest set

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)
X_train,len(X_train)

y_train,len(y_train)

models = {'Logistic Regression':LogisticRegression(),
          'KNN':KNeighborsClassifier(),
          'Random Forest':RandomForestClassifier()}
def fit_and_score(models, X_train, X_test,y_train,y_test):
    '''
    Fits and evaluates given machine learing models.
    models: A dictionary of dofferent Scikit-Learn machine learning models
    X_train : training data (no labels)
    X_test : testing data (no labels)
    y_train : training labels
    y_test : testing labels
    '''
    # set random seed
    np.random.seed(42)
    # make a dictionary to keep model scores
    model_scores={}
    # Loop through models
    for name, model in models.items():
        #Fit the model to the data
        model.fit(X_train,y_train)
        # Evaluate the model and append its score to the model_scores
        model_scores[name]=model.score(X_test,y_test)
    return model_scores
scores = fit_and_score(models,X_train,X_test,y_train,y_test) 
scores

model_compare = pd.DataFrame(data=scores.values(),index=scores.keys(),columns=['Accuracy'])
model_compare.plot.bar();
plt.xticks(rotation=0);
# Let's tune KNN

train_scores=[]
test_scores=[]

# create a list of different values for n_neighbours
neighbours = range(1,21)

# setup KNN instance

knn = KNeighborsClassifier()

# Loop through different n_neighbours
for i in neighbours:
    knn.set_params(n_neighbors=i)
    
    # fit the model to the data
    knn.fit(X_train,y_train)
    
    #update the training scores list
    train_scores.append(knn.score(X_train,y_train))
    
    #update the testing scores list
    test_scores.append(knn.score(X_test,y_test))
train_scores

test_scores

plt.plot(neighbours,train_scores,label='Train score');
plt.plot(neighbours,test_scores,label='Test score');
plt.xlabel('Number of neighbors');
plt.ylabel('Model score');
plt.xticks(np.arange(0,21,1));
plt.legend();

print(f'Maximum KNN score on the Test data: {max(test_scores)*100:.2f}%')

log_reg_grid =  {'C':np.logspace(-4,4,20),
                 'solver':['liblinear']}

rf_grid = {'n_estimators':np.arange(10,1000,50),
           'max_depth':[None,3,5,10],
           'min_samples_split':np.arange(2,20,2),
           'min_samples_leaf':np.arange(1,20,2)}

# Tune LogisticRegression

np.random.seed(42)

# setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                               param_distributions=log_reg_grid,
                               cv=5,
                               n_iter=20,
                               verbose=True)

#Fit random hyperparameter search model for LogisticRegression

rs_log_reg.fit(X_train,y_train)
rs_log_reg.best_params_

rs_log_reg.score(X_test,y_test)


np.random.seed(42)

# setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                               param_distributions=rf_grid,
                               cv=5,
                               n_iter=20,
                               verbose=True)

#Fit random hyperparameter search model for LogisticRegression

rs_rf.fit(X_train,y_train)
rs_rf.best_params_

rs_rf.score(X_test,y_test)

scores

# Different Hyperparameters for our LogisticRegression model
log_reg_grid = {'C': np.logspace(-4,4,30),
                'solver': ['liblinear']}

# setup grid hyperparameter search for LogisticRegression
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)
# Fit Grid search Logistic Regression model to our training data
gs_log_reg.fit(X_train,y_train)
# check the best hyperparameters
gs_log_reg.best_params_

# Evaluate the grid search LogisticRegression model
gs_log_reg.score(X_test,y_test)

y_preds = gs_log_reg.predict(X_test)

y_preds
y_test
# Plot ROC curve and calculate and calculate AUC metric
plot_roc_curve(gs_log_reg,X_test,y_test);
print(confusion_matrix(y_test,y_preds))

# visualise confusion matrix:
# we create a function for plotting ROC curve given y_test and y_preds
def plot_conf_mat(y_test,y_preds):
    '''
    Plots a nice looking confusion matrix using seaborns heatmap()
    '''
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(confusion_matrix(y_test,y_preds),
                     annot=True,
                     cbar=False)
    plt.xlabel('True label')
    plt.ylabel('Predicted label')

plot_conf_mat(y_test,y_preds)
print(classification_report(y_test,y_preds))

# Check best hyperparameters
gs_log_reg.best_params_

# create a new classifier with best parameters

clf = LogisticRegression(C=0.20433597178569418,solver='liblinear')
def cal_cv_scores(clf,X,y,metric):
    '''
    Calculates the cross-validated evaluation metrics of the given classifier as the metric,X and y are provided.
    returns : cross-validated score  
    '''

    score = cross_val_score(clf,
                            X,
                            y,
                            cv=5,
                            scoring=metric )
    return np.mean(score)*100

# cross-validated accuracy
cv_accuracy = cal_cv_scores(clf,X,y,'accuracy')
print(f'Cross-validated accuracy: {cv_accuracy:.2f}%')
# cross-validated Precision
cv_precision = cal_cv_scores(clf,X,y,'precision')
print(f'Cross-validated Precision: {cv_precision:.2f}%')
# cross-validated Recall
cv_recall = cal_cv_scores(clf,X,y,'recall')
print(f'Cross-validated Recall: {cv_recall:.2f}%')
# cross-validated F1-score
cv_f1 = cal_cv_scores(clf,X,y,'f1')
print(f'Cross-validated F1-score: {cv_f1:.2f}%')
cv_metrics = pd.Series([cv_accuracy,cv_precision,cv_recall,cv_f1])
cv_metrics_acc=pd.DataFrame(cv_metrics)
cv_metrics_acc.rename(columns={0:'Score'},inplace=True)
cv_metrics_acc.rename(index={0:'Accuracy',1:'Precision',2:'Recall',3:'F1'},inplace=True)
cv_metrics_acc

cv_metrics_acc.plot.bar();

df.head()

# Fit an instance of Logistic Regression

clf = LogisticRegression(C=0.20433597178569418,
                         solver='liblinear') 
clf.fit(X_train,y_train)
# check coef_
clf.coef_

# Match coef's of features to columns
features_dict = dict(zip(df.columns,list(clf.coef_[0])))
features_dict

plt.figure(figsize=(10,6));
plt.bar(features_dict.keys(),features_dict.values());
plt.title('Feature Importance',fontsize=20);
