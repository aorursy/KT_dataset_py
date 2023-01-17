import warnings

warnings.filterwarnings("ignore")
#DF

import pandas as pd

import numpy as np

from pandas.plotting import scatter_matrix



#Common Model Algorithms

from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeRegressor



#Common Model Helpers

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import roc_auc_score as auc

from sklearn.metrics import roc_curve

from sklearn.metrics import confusion_matrix



#Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



#Configure Visualization Defaults

#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')
from IPython.display import display_html 



sample_submission = pd.read_csv('../input/titanic/gender_submission.csv')



sample_head = sample_submission.head()

sample_tail = sample_submission.tail()

sample_sample = sample_submission.sample(5)



df1_styler = sample_head.style.set_table_attributes("style='display:inline'").set_caption('Head')

df2_styler = sample_tail.style.set_table_attributes("style='display:inline'").set_caption('Tail')

df3_styler = sample_sample.style.set_table_attributes("style='display:inline'").set_caption('Random Sample')



display_html(df1_styler._repr_html_()+df2_styler._repr_html_()+ df3_styler._repr_html_(), raw=True)
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.shape
train.head()
train.tail()
train.sample(5)
train.info()
train.isna().sum().sort_values(ascending = False)
train.describe()
survived = len(train[train['Survived'] ==1])

deaths = len(train[train['Survived'] ==0])



print(f"The number of people who survived is {survived}.")

print()

print(f"The number of people who died is {deaths}.")
num_cols = train[train.select_dtypes(exclude=['object']).columns]

cat_cols = train[train.select_dtypes(include=['object']).columns]



print(f'The numerical cols are: {list(num_cols)} ({len(list(num_cols))}).\n')

print(f'The categorical cols are: {list(cat_cols)} ({len(list(cat_cols))}).\n')

print(f'Total number of cols: {len(list(train))}')
def tight():

    plt.tight_layout()

    plt.show()
fig, (ax1,ax2) = plt.subplots(ncols = 2, nrows =1,figsize =(12,5))



sns.countplot(train['Sex'], hue = train['Survived'],ax =ax1)

ax2.pie(train['Sex'].value_counts()

        ,shadow = True,wedgeprops = {'edgecolor':'black'},autopct='%1.1f%%',labels =['M','F'])



tight()
fig, ax = plt.subplots(figsize =(12,5))



sns.countplot(train['Embarked'], hue = train['Survived'])



tight()
sns.catplot(data = train, kind = 'point', x = 'Pclass', y ='Survived', col ='Embarked', hue ='Sex')



tight()
corr_matrix = num_cols.corr()



display(corr_matrix['Survived'].sort_values(ascending = False))

attributes = list(num_cols)



scatter_matrix(num_cols[attributes],figsize = (12,8))



tight()
fig, ax = plt.subplots(figsize = (12,7))

colormap = sns.diverging_palette(220, 10, as_cmap = True)



sns.heatmap(corr_matrix,linewidth = 0.01,vmax=1.0,square = True, cmap = colormap,annot = True)



bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5)



tight()
sns.lmplot(data = train, x ='Fare',y = 'Survived',hue = 'Sex',height = 5,aspect =2)



tight()
g = sns.FacetGrid( train, hue = 'Survived', aspect=4 )



g.map(sns.kdeplot, 'Fare', shade= True )

g.add_legend()



tight()
fig = plt.figure(figsize=(12, 18))



for i in range(len(num_cols.columns)):

    fig.add_subplot(10, 4, i+1)

    sns.boxplot(y=num_cols.iloc[:,i])



tight()
fig = plt.figure(figsize=(12,18))



try:

    for i in range(len(num_cols.columns)):

        fig.add_subplot(10,4,i+1)

        sns.distplot(num_cols.iloc[:,i].dropna())

        plt.xlabel(num_cols.columns[i])

except RuntimeError:

    pass



tight()
fig, (ax1,ax2) = plt.subplots(ncols = 2, nrows =1, figsize = (12,5), sharex = True)



sns.countplot(y= train['SibSp'],ax =ax1,palette = 'Reds_d')

sns.countplot(y =train['Parch'],ax =ax2, palette ='Blues_d')



tight()
sns.lmplot(data = train, x = 'Survived', y ='Pclass', hue = 'Sex')
fig, (ax1,ax2,ax3) = plt.subplots(ncols = 3, nrows = 1, figsize = (14,5))



sns.pointplot(x = 'Survived', y= 'Fare', hue = 'Sex', data = train,ax =ax1)

sns.pointplot(x = 'Survived', y= 'Pclass', hue = 'Sex', data = train,ax= ax2)

sns.pointplot(x = 'Survived', y= 'SibSp', hue = 'Sex', data = train,ax= ax3)



tight()
sns.catplot(data = train, kind = 'bar', col ='Pclass', x ='Survived', y = 'Fare', hue = 'Sex')

sns.catplot(data = train, kind = 'bar', col ='Pclass', x ='Survived', y = 'Age', hue = 'Sex')



tight()
X = train.copy()

y = train['Survived']
X = X.drop('Survived',axis=1)
X.isnull().sum().sort_values(ascending = False).head(1) #drop this useless as shit
display(corr_matrix['Survived'].sort_values(ascending = False))
X = X.drop('Cabin',axis =1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
cat = list(X[X.select_dtypes(include=['object']).columns])

num = list(X[X.select_dtypes(exclude=['object']).columns])

my_cols = cat + num
X_train = X_train[my_cols].copy()

X_valid = X_valid[my_cols].copy()

X_test = test[my_cols].copy()
num_transformer = Pipeline(steps=[

    ('num_imputer', SimpleImputer(strategy='median')),

    ('std_scaler', StandardScaler())

    ])



cat_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

    ])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', num_transformer, num),       

        ('cat',cat_transformer,cat),

        ])
print("Data Shape: {}".format(train.shape))

print("X_train Shape: {}".format(X_train.shape))

print("y_train Shape: {}".format(y_train.shape))
X_train_prepared = preprocessor.fit_transform(X_train)

X_valid_prepared = preprocessor.transform(X_valid)



print(X_train_prepared.shape)

print(X_valid_prepared.shape)
def display_scores(scores):

    print("Mean:", scores.mean())

    print("Standard deviation:", scores.std())
tree = DecisionTreeClassifier(max_depth =2)



tree.fit(X_train_prepared, y_train)

tree_predictions = tree.predict(X_train_prepared)



print("\nAccuracy Score for Decision Tree Classifier is: " + str(tree.score(X_train_prepared, y_train)))
tree_scores = cross_val_score(tree, X_train_prepared, y_train,scoring="neg_mean_squared_error", cv=10)

tree_rmse_scores = np.sqrt(-tree_scores)



display_scores(tree_rmse_scores)
forest = RandomForestClassifier(n_estimators = 10)



forest.fit(X_train_prepared, y_train)

forest_predictions = forest.predict(X_train_prepared)



print("\nAccuracy Score for Random Forest Classifier is: " + str(forest.score(X_train_prepared, y_train)))
forest_predictions = forest.predict(X_train_prepared)
forest_scores = cross_val_score(forest, X_train_prepared, y_train,scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)



display_scores(forest_rmse_scores)
param_grid = {'bootstrap': [True, False],

 'max_depth': [1,2,3],

 'max_features': ['auto', 'sqrt'],

 'min_samples_leaf': [1, 2, 4],

 'n_estimators': [10, 100]}



forest_cv = GridSearchCV(forest, param_grid, n_jobs= 1, cv =10,scoring='roc_auc')

                  

forest_cv.fit(X_train_prepared, y_train)

print(forest_cv.best_params_)    

print(forest_cv.best_score_)
forest_cv_predictions = forest_cv.predict(X_train_prepared)
svc = SVC(gamma ='auto')



svc.fit(X_train_prepared, y_train)

svc_predictions = forest.predict(X_train_prepared)



print("\nAccuracy Score for SVC is: " + str(svc.score(X_train_prepared, y_train)))
svc_scores = cross_val_score(svc, X_train_prepared, y_train,scoring="neg_mean_squared_error", cv=10)

svc_rmse_scores = np.sqrt(-svc_scores)



display_scores(svc_rmse_scores)
gb = GradientBoostingClassifier(random_state = 10, learning_rate = 0.01)



gb.fit(X_train_prepared, y_train)

gb_predictions = gb.predict(X_train_prepared)



print("\nAccuracy Score for Gradiant Boosting Classifieris: " + str(gb.score(X_train_prepared, y_train)))
gb_scores = cross_val_score(gb, X_train_prepared, y_train,scoring="neg_mean_squared_error", cv=10)

gb_rmse_scores = np.sqrt(-gb_scores)



display_scores(gb_rmse_scores)
param_grid = { 

    "learning_rate": [0.01, 0.05, 0.1],

    "max_depth":[3,5,8],

    "n_estimators":[250,500]}



gb_cv = GridSearchCV(gb, param_grid, n_jobs= 1, cv =5,scoring ='roc_auc')

                

gb_cv.fit(X_train_prepared, y_train)

print(gb_cv.best_params_)    

print(gb_cv.best_score_)
logreg = LogisticRegression(solver = 'liblinear')



logreg.fit(X_train_prepared, y_train)

logreg_predictions = logreg.predict(X_train_prepared)



print("\nAccuracy Score for Logistic Regression is: " + str(logreg.score(X_train_prepared, y_train)))
logreg_scores = cross_val_score(logreg, X_train_prepared, y_train,scoring="neg_mean_squared_error", cv=10)

logreg_rmse_scores = np.sqrt(-logreg_scores)



display_scores(logreg_rmse_scores)
param_grid = { 

    "C": [0.5,1.0,10.0,25.0,50.0],

    "solver":['liblinear','lbfgs','sag'],

    "max_iter":[1000,2500,5000],

}



logreg_cv = GridSearchCV(logreg, param_grid, n_jobs= 1, cv =5,scoring = 'roc_auc')

                

logreg_cv.fit(X_train_prepared, y_train)

print(logreg_cv.best_params_)    

print(logreg_cv.best_score_)
knn = KNeighborsClassifier(n_neighbors = 5)



knn.fit(X_train_prepared, y_train)

knn_predictions = knn.predict(X_train_prepared)



print("\nAccuracy Score for KNN Classifier is: " + str(knn.score(X_train_prepared, y_train)))
knn_scores = cross_val_score(knn, X_train_prepared, y_train,scoring="neg_mean_squared_error", cv=10)

knn_rmse_scores = np.sqrt(-knn_scores)



display_scores(knn_rmse_scores)
param_grid = { 

    "n_neighbors": [1,5,10,25],

    "weights":['uniform','distance'],

    "leaf_size":[30,50,100],

    'algorithm':['auto','brute'],

}



knn_cv = GridSearchCV(knn, param_grid, n_jobs= 1, cv =10,iid = False,scoring ='roc_auc')

                

knn_cv.fit(X_train_prepared, y_train)

print(knn_cv.best_params_)    

print(knn_cv.best_score_)
linear_svm = LinearSVC()



linear_svm.fit(X_train_prepared, y_train)

linear_svm_predictions = linear_svm.predict(X_train_prepared)



print("\nAccuracy Score for Linear SVM Classifier is: " + str(linear_svm.score(X_train_prepared, y_train)))
linear_svm_scores = cross_val_score(linear_svm, X_train_prepared, y_train,scoring="neg_mean_squared_error", cv=10)

linear_svm_rmse_scores = np.sqrt(-linear_svm_scores)



display_scores(linear_svm_rmse_scores)
param_grid = { 

    "C": [1.0,5.0,10.0,25.0],

    "max_iter":[1000,2500,5000,10000,15000],

    "fit_intercept":[True,False],

}



linear_svm_cv = GridSearchCV(linear_svm, param_grid, n_jobs= 1, cv =10,iid = False,scoring = 'roc_auc')

                

linear_svm_cv.fit(X_train_prepared, y_train)

print(linear_svm_cv.best_params_)    

print(linear_svm_cv.best_score_)
forest_predictions = forest_cv.predict(X_valid_prepared)

knn_predictions = knn_cv.predict(X_valid_prepared)

gb_predictions = gb_cv.predict(X_valid_prepared)

logreg_predictions = logreg_cv.predict(X_valid_prepared)

svc_predictions = svc.predict(X_valid_prepared)



fig, ax = plt.subplots(2,3,figsize = (22,10),sharey = True)



sns.heatmap(confusion_matrix(y_valid,forest_predictions),annot = True, fmt ='d',ax=ax[0,0],cmap = 'Greens',annot_kws={"size": 16},)

sns.heatmap(confusion_matrix(y_valid,knn_predictions),annot = True, fmt = 'd', ax=ax[0,1], cmap = 'Reds',annot_kws={"size": 16},)

sns.heatmap(confusion_matrix(y_valid,gb_predictions),annot = True, fmt = 'd', ax=ax[1,0], cmap = 'Blues',annot_kws={"size": 16},)

sns.heatmap(confusion_matrix(y_valid,logreg_predictions),annot = True, fmt = 'd', ax=ax[1,1], cmap = 'Oranges',annot_kws={"size": 16},)

sns.heatmap(confusion_matrix(y_valid,svc_predictions),annot = True, fmt = 'd', ax=ax[0,2], cmap = 'YlGnBu',annot_kws={"size": 16},)



ax[0,0].set_title('Confusion Matrix - Random Forest Classifier')

ax[0,1].set_title('Confusion Matrix - KNN')

ax[1,0].set_title('Confusion Matrix - Gradiant Boosting Classifier')

ax[1,1].set_title('Confusion Matrix - Logistic Regression')

ax[0,2].set_title('Confusion Matrix - SVC')





tight()
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_valid, forest_cv.predict_proba(X_valid_prepared)[:, 1])

fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_valid, knn_cv.predict_proba(X_valid_prepared)[:, 1])

fpr_gbrt, tpr_gbrt, thresholds_gbrt = roc_curve(y_valid, gb_cv.predict_proba(X_valid_prepared)[:, 1])

fpr_logreg, tpr_logreg, thresholds_logreg = roc_curve(y_valid, logreg_cv.predict_proba(X_valid_prepared)[:, 1])

fpr_svc, tpr_svc, thresholds_svc = roc_curve(y_valid, svc.decision_function(X_valid_prepared))

fpr_linearsvm, tpr_linearsvm, thresholds_linearsvm = roc_curve(y_valid, linear_svm_cv.decision_function(X_valid_prepared))



roc_auc_forest = auc(y_valid,forest_cv.predict_proba(X_valid_prepared)[:, 1])

roc_auc_knn = auc(y_valid,knn_cv.predict_proba(X_valid_prepared)[:, 1])

roc_auc_gb = auc(y_valid,gb_cv.predict_proba(X_valid_prepared)[:, 1])

roc_auc_logreg = auc(y_valid,logreg_cv.predict_proba(X_valid_prepared)[:, 1])

roc_auc_svc = auc(y_valid,svc.decision_function(X_valid_prepared))

roc_auc_linearsvm = auc(y_valid,linear_svm_cv.decision_function(X_valid_prepared))
sns.set(style="darkgrid")



fig, ax = plt.subplots(figsize = (14,8))



plt.plot(fpr_rf,tpr_rf,color='mediumseagreen',label = 'Random Forest Classifier (area = %0.2f)' % roc_auc_forest)

plt.plot(fpr_knn,tpr_knn,color='red',label = 'KNN (area = %0.2f)' % roc_auc_knn)

plt.plot(fpr_gbrt,tpr_gbrt,color='cyan',label = 'Gradient Boosting Classifier (area = %0.2f)' % roc_auc_gb)

plt.plot(fpr_logreg,tpr_logreg,color='Orange',label = 'Logistic Regression (area = %0.2f)' % roc_auc_logreg)

plt.plot(fpr_svc,tpr_svc,color='yellow',label = 'SVC (area = %0.2f)' % roc_auc_svc)

plt.plot(fpr_linearsvm,tpr_linearsvm,color='slategrey',label = 'Linear SVM (area = %0.2f)' % roc_auc_linearsvm)

plt.plot([0, 1], [0, 1],linestyle='--',color ='black')



plt.legend()

plt.title('ROC Curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate (Recall)')



tight()
X_test_prepared = preprocessor.transform(X_test)
final_model = gb_cv.best_estimator_
final_predictions = final_model.predict(X_test_prepared)
output = pd.DataFrame({'PassengerId': X_test.PassengerId,

                       'Survived': final_predictions})



output.to_csv('submission.csv', index=False)