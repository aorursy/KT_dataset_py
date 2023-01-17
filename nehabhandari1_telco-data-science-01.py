import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics#Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import train_test_split,cross_val_score
import seaborn as sns
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image 
from sklearn import tree
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_footer
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_quoting
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import metrics

Telco_data = pd.read_csv("../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")
Telco_data.head()
Telco_data.shape
Telco_data.columns
Telco_data.dtypes
Telco_data.info()
Telco_data.isna().sum()
Telco_data.describe()
Telco_data.max()
Telco_data.min()
Telco_data.nunique()
print(len(Telco_data[(Telco_data['TotalCharges']==' ')]))
Telco_data.shape
Telco_No_Space=Telco_data[(Telco_data['TotalCharges']!=' ')]
Telco_No_Space.shape
Telco_No_Space['TotalCharges'] = pd.to_numeric(Telco_No_Space['TotalCharges'])
Telco_No_Space.dtypes
check_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingMovies']
for c in check_cols:
    print( c + '=', Telco_No_Space[c].unique())
Telco_No_Space.head()
Telco_No_Space["SeniorCitizen"] = Telco_No_Space["SeniorCitizen"].replace(to_replace=[0, 1], value=['No', 'Yes'])

Telco_No_Space.head()
Telco_No_Space.replace(['No internet service','No phone service'],'No', inplace=True)
for i in Telco_No_Space:
    print(i+"=",Telco_No_Space[i].unique())
Telco_dummies=pd.get_dummies(Telco_No_Space)
Telco_dummies.head()
Telco_dummies.dtypes

sns.set(style="ticks", color_codes=True)

fig, axes = plt.subplots(nrows = 3,ncols = 5,figsize = (25,15))
sns.countplot(x = "gender", data = Telco_No_Space, ax=axes[0][0])
sns.countplot(x = "Partner", data = Telco_No_Space, ax=axes[0][1])
sns.countplot(x = "Dependents", data = Telco_No_Space, ax=axes[0][2])
sns.countplot(x = "PhoneService", data = Telco_No_Space, ax=axes[0][3])
sns.countplot(x = "MultipleLines", data = Telco_No_Space, ax=axes[0][4])
sns.countplot(x = "InternetService", data = Telco_No_Space, ax=axes[1][0])
sns.countplot(x = "OnlineSecurity", data = Telco_No_Space, ax=axes[1][1])
sns.countplot(x = "OnlineBackup", data = Telco_No_Space, ax=axes[1][2])
sns.countplot(x = "DeviceProtection", data = Telco_No_Space, ax=axes[1][3])
sns.countplot(x = "TechSupport", data = Telco_No_Space, ax=axes[1][4])
sns.countplot(x = "StreamingTV", data = Telco_No_Space, ax=axes[2][0])
sns.countplot(x = "StreamingMovies", data = Telco_No_Space, ax=axes[2][1])
sns.countplot(x = "Contract", data = Telco_No_Space, ax=axes[2][2])
sns.countplot(x = "PaperlessBilling", data = Telco_No_Space, ax=axes[2][3])
ax = sns.countplot(x = "PaymentMethod", data = Telco_No_Space, ax=axes[2][4])
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show(fig)
Telco_No_Space['Churn'].value_counts()
sns.set(style="ticks", color_codes=True)

sns.countplot(x = "Churn", data = Telco_No_Space)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show(fig)
print('Vizualizing the Numeric variables')
fig, axes = plt.subplots(1, 3, figsize=(20,5))
#Ploting the histogram
Telco_No_Space["tenure"].plot.hist(color='DarkBlue', alpha=0.7, bins=50, title='Tenure',ax=axes[0])
Telco_No_Space["MonthlyCharges"].plot.hist(color='DarkBlue', alpha=0.7, bins=50, title='MonthlyCharges',ax=axes[1])
Telco_No_Space["TotalCharges"].plot.hist(color='DarkBlue', alpha=0.7, bins=50, title='TotalCharges',ax=axes[2])

print('Frequency Distribution of Numeric Variable')
fig, axes = plt.subplots(1, 3, figsize=(20,5))
sns.distplot( Telco_No_Space["tenure"] , kde=True, rug=False, color="skyblue", ax=axes[0])
sns.distplot( Telco_No_Space["MonthlyCharges"] , kde=True, rug=False, color="olive", ax=axes[1])
sns.distplot( Telco_No_Space["TotalCharges"] , kde=True, rug=False, color="gold", ax=axes[2])
fig, axes = plt.subplots(1, 3, figsize=(20,5))
sns.boxplot(x=Telco_No_Space["tenure"], orient="v", color="salmon",ax=axes[0])
sns.boxplot(x=Telco_No_Space["MonthlyCharges"], orient="v", color="skyblue",ax=axes[1])
sns.boxplot(x=Telco_No_Space["TotalCharges"] , orient="v", color="green",ax=axes[2])
plt.figure(figsize =(10,10))
Correlation=Telco_No_Space.select_dtypes(include=[np.number]).corr()
sns.heatmap(Correlation,annot=True)

numeric_data =Telco_No_Space.select_dtypes(include=[np.number])
numeric_data
categorical_data=Telco_No_Space.select_dtypes(exclude=[np.number])
categorical_data

Telco_No_Space[['Churn','gender','customerID']].groupby(['Churn','gender']).agg('count')

Telco_No_Space[['Churn','SeniorCitizen','customerID']].groupby(['Churn','SeniorCitizen']).agg('count')

Telco_No_Space[['Churn','Partner','customerID']].groupby(['Churn','Partner']).agg('count')
Telco_No_Space[['Churn','Dependents','customerID']].groupby(['Churn','Dependents']).agg('count')
Telco_No_Space[['Churn','PhoneService','customerID']].groupby(['Churn','PhoneService']).agg('count')
Telco_No_Space[['Churn','MultipleLines','customerID']].groupby(['Churn','MultipleLines']).agg('count')
Telco_No_Space[['Churn','InternetService','customerID']].groupby(['Churn','InternetService']).agg('count')
Telco_No_Space[['Churn','OnlineSecurity','customerID']].groupby(['Churn','OnlineSecurity']).agg('count')
Telco_No_Space[['Churn','OnlineBackup','customerID']].groupby(['Churn','OnlineBackup']).agg('count')
Telco_No_Space[['Churn','DeviceProtection','customerID']].groupby(['Churn','DeviceProtection']).agg('count')
Telco_No_Space[['Churn','TechSupport','customerID']].groupby(['Churn','TechSupport']).agg('count')
Telco_No_Space[['Churn','StreamingTV','customerID']].groupby(['Churn','StreamingTV']).agg('count')
Telco_No_Space[['Churn','StreamingMovies','customerID']].groupby(['Churn','StreamingMovies']).agg('count')
Telco_No_Space[['Churn','Contract','customerID']].groupby(['Churn','Contract']).agg('count')
Telco_No_Space[['Churn','PaperlessBilling','customerID']].groupby(['Churn','PaperlessBilling']).agg('count')
Telco_No_Space[['Churn','PaymentMethod','customerID']].groupby(['Churn','PaymentMethod']).agg('count')
Telco_No_Space[['Churn','MonthlyCharges']].groupby(['Churn']).agg('mean')
Telco_No_Space[['Churn','MonthlyCharges']].groupby(['Churn']).agg('mean').unstack(1).plot(kind='bar', subplots=True)
g = sns.FacetGrid(Telco_No_Space, col="Churn", col_order=["Yes", "No"])
g = g.map(plt.hist, 'MonthlyCharges', color="m")
Telco_No_Space[['Churn','tenure']].groupby(['Churn']).agg('mean').unstack(1).plot(kind='bar', subplots=True)
g = sns.FacetGrid(Telco_No_Space, col="Churn", col_order=["Yes", "No"])
g = g.map(plt.hist, 'tenure', color="m")
g = sns.FacetGrid(Telco_No_Space, col="Churn", col_order=["Yes", "No"])
g = g.map(plt.hist, 'TotalCharges', color="m")
Telco_No_Space_1 = Telco_No_Space.drop(['customerID'], axis = 1)
Telco_No_Space_1.shape
Telco_No_Space_1.dtypes
cat_data=Telco_No_Space_1.select_dtypes(exclude=[np.number]).astype('category')
Telco_No_Space_1[cat_data.columns]=cat_data
Telco_No_Space_1.dtypes
X_1=Telco_No_Space_1.drop(["Churn"],axis=1)
y_1=Telco_No_Space_1[["Churn"]]
#Telco_dummies=pd.get_dummies(Telco_No_Space_1)
#cols=['tenure','MonthlyCharges','TotalCharges']
#num_data = Telco_dummies.loc[:,cols]
#norm_data = (num_data-num_data.mean())/num_data.std()
#Telco_dummies.drop(cols,axis=1, inplace=True)
#Telco_dummies=pd.concat([Telco_dummies,norm_data],axis=1)
#Telco_dummies.head()

Telco_dummies=pd.get_dummies(Telco_No_Space_1)
X=Telco_dummies.drop(["Churn_Yes","Churn_No"],axis=1)
y=Telco_dummies[["Churn_Yes"]]

# Create decision tree classifer object
clf = RandomForestClassifier(random_state=0, n_jobs=-1)

# Train model
model = clf.fit(X, y)

# Calculate feature importances
importances = model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X.columns[i] for i in indices]

# Create plot
plt.figure(figsize=(20,10))

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(X.shape[1]), importances[indices])

# Add feature names as x-axis labels
plt.xticks(range(X.shape[1]),names, rotation=90)

# Show plot
plt.show()


#Telco_dummies=pd.get_dummies(Telco_No_Space_1)
X_1=Telco_dummies.drop(["Churn_Yes","Churn_No","TotalCharges","MonthlyCharges","tenure"],axis=1)
y=Telco_dummies[["Churn_Yes"]]

# Create decision tree classifer object
clf = RandomForestClassifier(random_state=0, n_jobs=-1)

# Train model
model = clf.fit(X_1, y)

# Calculate feature importances
importances = model.feature_importances_

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [X_1.columns[i] for i in indices]

# Create plot
plt.figure(figsize=(20,10))

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(X_1.shape[1]), importances[indices])

# Add feature names as x-axis labels
plt.xticks(range(X_1.shape[1]), names, rotation=90)

# Show plot
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#print length of X_train, X_test, y_train, y_test
print ("X_train: ", len(X_train))
print("X_test: ", len(X_test))
print("y_train: ", len(y_train))
print("y_test: ", len(y_test))
scaled_logistic_pipe = Pipeline(steps = [('sc', StandardScaler()),('classifier', LogisticRegression())])

#scaled_logistic_param_grid = { "classifier__penalty": ['l2','l1'], "classifier__C": np.logspace(0, 4, 10)}

C = np.logspace(-4, 4, 50)
# Create a list of options for the regularization penalty
penalty = ['l2']
# Create a dictionary of all the parameter options 
# Note has you can access the parameters of steps of a pipeline by using '__’
scaled_logistic_param_grid = dict(classifier__C=C,
                  classifier__penalty=penalty)
np.random.seed(1)

scaled_logistic_grid_search = GridSearchCV(scaled_logistic_pipe, scaled_logistic_param_grid, cv=10)

scaled_logistic_grid_search.fit(X_train, y_train.values.ravel())

scaled_logistic_model = scaled_logistic_grid_search.best_estimator_

 

print('Cross Validation Score:', scaled_logistic_grid_search.best_score_)

print('Best Hyperparameters:  ', scaled_logistic_grid_search.best_params_)

print('Training Accuracy:     ', scaled_logistic_model.score(X_train, y_train))

y_pred_logistic=scaled_logistic_model.predict(X_test)

print("Logistic Accuracy:",metrics.accuracy_score(y_test, y_pred_logistic))
print(classification_report(y_test, y_pred_logistic))
preds_logistic = scaled_logistic_model.predict_proba(X_test)
preds_logistic = preds_logistic[:,1]
fpr_logistic, tpr_logistic, _ = metrics.roc_curve(y_test, preds_logistic)
auc_score_logistic = metrics.auc(fpr_logistic, tpr_logistic)
plt.title('ROC Curve')
plt.plot(fpr_logistic, tpr_logistic, label='AUC = {:.2f}'.format(auc_score_logistic))

unscaled_knn_pipe = Pipeline(steps = [('classifier', KNeighborsClassifier())])

unscaled_knn_param_grid = {'classifier__n_neighbors': range(1,10),'classifier__p': [1,2,3]}

np.random.seed(1)

unscaled_knn_grid_search = GridSearchCV(unscaled_knn_pipe, unscaled_knn_param_grid, cv=10, refit='True')

unscaled_knn_grid_search.fit(X_train, y_train.values.ravel())

unscaled_knn_model = unscaled_knn_grid_search.best_estimator_

 

print('Cross Validation Score:', unscaled_knn_grid_search.best_score_)

print('Best Hyperparameters:  ', unscaled_knn_grid_search.best_params_)

print('Training Accuracy:     ', unscaled_knn_model.score(X_train, y_train))

y_pred_knn=unscaled_knn_model.predict(X_test)

print("KNN Accuracy:",metrics.accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))
preds_knn = unscaled_knn_model.predict_proba(X_test)
preds_knn = preds_knn[:,1]
fpr_knn, tpr_knn, _ = metrics.roc_curve(y_test, preds_knn)
auc_score_knn = metrics.auc(fpr_knn, tpr_knn)
plt.title('ROC Curve')
plt.plot(fpr_knn, tpr_knn, label='AUC = {:.2f}'.format(auc_score_knn))
scaled_knn_pipe = Pipeline(steps = [('sc', StandardScaler()),('classifier', KNeighborsClassifier())])

scaled_knn_param_grid = {'classifier__n_neighbors': range(1,10),'classifier__p': [1,2,3]}

np.random.seed(1)

scaled_knn_grid_search = GridSearchCV(scaled_knn_pipe, scaled_knn_param_grid, cv=10, refit='True')

scaled_knn_grid_search.fit(X_train, y_train.values.ravel())

scaled_knn_model = scaled_knn_grid_search.best_estimator_

 

print('Cross Validation Score:', scaled_knn_grid_search.best_score_)

print('Best Hyperparameters:  ', scaled_knn_grid_search.best_params_)

print('Training Accuracy:     ', scaled_knn_model.score(X_train, y_train))

y_pred_knn=scaled_knn_model.predict(X_test)

print("KNN Accuracy:",metrics.accuracy_score(y_test, y_pred_knn))

print(classification_report(y_test, y_pred_knn))
preds_knn = scaled_knn_model.predict_proba(X_test)
preds_knn = preds_knn[:,1]
fpr_knn, tpr_knn, _ = metrics.roc_curve(y_test, preds_knn)
auc_score_knn = metrics.auc(fpr_knn, tpr_knn)
plt.title('ROC Curve')
plt.plot(fpr_knn, tpr_knn, label='AUC = {:.2f}'.format(auc_score_knn))
unscaled_tree_pipe = Pipeline(steps = [('decisiontree', DecisionTreeClassifier())])

#Create lists of parameter for Decision Tree Classifier
criterion = ['gini', 'entropy']
max_depth = [1,2,3,4,5,6,7,8,9,10,11,12]
    
unscaled_tree_param_grid = dict(decisiontree__criterion=criterion,decisiontree__max_depth=max_depth)

np.random.seed(1)

unscaled_tree_grid_search = GridSearchCV(unscaled_tree_pipe, unscaled_tree_param_grid, cv=10)

unscaled_tree_grid_search.fit(X_train, y_train)

unscaled_tree_model = unscaled_tree_grid_search.best_estimator_

 

print('Cross Validation Score:', unscaled_tree_grid_search.best_score_)

print('Best Hyperparameters:  ', unscaled_tree_grid_search.best_params_)

print('Training Accuracy:     ', unscaled_tree_model.score(X_train, y_train))

y_pred_tree=unscaled_tree_model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))
preds_tree = unscaled_tree_model.predict_proba(X_test)
preds_tree = preds_tree[:,1]
fpr_tree, tpr_tree, _ = metrics.roc_curve(y_test, preds_tree)
auc_score_tree = metrics.auc(fpr_tree, tpr_tree)
plt.title('ROC Curve')
plt.plot(fpr_tree, tpr_tree, label='AUC = {:.2f}'.format(auc_score_tree))
unscaled_rf_pipe = Pipeline([("classifier", RandomForestClassifier())])
unscaled_rf_param_grid = {
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_depth":[5,8,15,25,30,None],
                 "classifier__min_samples_leaf":[1,2,5,10,15,100],
                 "classifier__max_leaf_nodes": [2, 5,10]}
unscaled_rf_grid_search = GridSearchCV(unscaled_rf_pipe, unscaled_rf_param_grid, cv=5, verbose=0,n_jobs=-1)

unscaled_rf_grid_search.fit(X_train, y_train.values.ravel())

unscaled_rf_model = unscaled_tree_grid_search.best_estimator_

 

print('Cross Validation Score:', unscaled_rf_grid_search.best_score_)

print('Best Hyperparameters:  ', unscaled_rf_grid_search.best_params_)

print('Training Accuracy:     ', unscaled_rf_model.score(X_train, y_train))

y_pred_rf=unscaled_rf_model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
preds_rf = unscaled_rf_model.predict_proba(X_test)
preds_rf = preds_rf[:,1]
fpr_rf, tpr_rf, _ = metrics.roc_curve(y_test, preds_rf)
auc_score_rf = metrics.auc(fpr_rf, tpr_rf)
plt.title('ROC Curve')
plt.plot(fpr_rf, tpr_rf, label='AUC = {:.2f}'.format(auc_score_rf))

categorical_features = categorical_data.columns[1:len(categorical_data.columns)-1].tolist()
numerical_features = numeric_data.columns.tolist()
target = "Churn"
df=Telco_No_Space
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)



class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
pipeline_tree = Pipeline([
   

    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling features from the post's subject line
            ("categorical_features", Pipeline([
                ('selector', ItemSelector(key=categorical_features)),
                                ("onehot",OneHotEncoder()),
                
            ])),

            # Pipeline for standard bag-of-words model for body
            ("numerical_features", Pipeline([
                ('selector', ItemSelector(key=numerical_features)),
                                ("scaler",StandardScaler()),
             
            ])),    

        ], 
    )),

    # Use a SVC classifier on the combined features
     ("classifier",tree.DecisionTreeClassifier(max_depth = 5,random_state=42)),
])

pipeline_tree.fit(df_train, df_train[target])
pred_tree = pipeline_tree.predict(df_test)


print(classification_report(df_test[target], pred_tree))

preds_tree = pipeline.predict_proba(df_test)
preds_tree = preds_tree[:,1]
fpr_tree, tpr_tree, _ = metrics.roc_curve(df_test[target].replace(to_replace=['No', 'Yes'], value=[0, 1]), preds_tree)
auc_score_tree = metrics.auc(fpr_tree, tpr_tree)
plt.title('ROC Curve')
plt.plot(fpr_tree, tpr_tree, label='AUC = {:.2f}'.format(auc_score_tree))
pipeline_rf = Pipeline([
   

    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling features from the post's subject line
            ("categorical_features", Pipeline([
                ('selector', ItemSelector(key=categorical_features)),
                                ("onehot",OneHotEncoder()),
                
            ])),

            # Pipeline for standard bag-of-words model for body
            ("numerical_features", Pipeline([
                ('selector', ItemSelector(key=numerical_features)),
                                ("scaler",StandardScaler()),
             
            ])),    

        ], 
    )),

    # Use a SVC classifier on the combined features
     ("classifier",RandomForestClassifier(max_depth = 8, max_leaf_nodes = 10, min_samples_leaf = 10, n_estimators=10)),
])

pipeline_rf.fit(df_train, df_train[target])
pred_rf = pipeline_rf.predict(df_test)


print(classification_report(df_test[target], pred_rf))



preds_rf = pipeline_rf.predict_proba(df_test)
preds_rf = preds_rf[:,1]
fpr_rf, tpr_rf, _ = metrics.roc_curve(df_test[target].replace(to_replace=['No', 'Yes'], value=[0, 1]), preds_rf)
auc_score_rf = metrics.auc(fpr_rf, tpr_rf)
plt.title('ROC Curve')
plt.plot(fpr_rf, tpr_rf, label='AUC = {:.2f}'.format(auc_score_rf))

from sklearn import svm
pipeline_svm = Pipeline([
   

    # Use FeatureUnion to combine the features from subject and body
    ('union', FeatureUnion(
        transformer_list=[

            # Pipeline for pulling features from the post's subject line
            ("categorical_features", Pipeline([
                ('selector', ItemSelector(key=categorical_features)),
                                ("onehot",OneHotEncoder()),
                
            ])),

            # Pipeline for standard bag-of-words model for body
            ("numerical_features", Pipeline([
                ('selector', ItemSelector(key=numerical_features)),
                                ("scaler",StandardScaler()),
             
            ])),    

        ], 
    )),

    # Use a SVC classifier on the combined features
     ("classifier",svm.SVC(probability=True)),
])


param_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
grid_params_svm = [{'classifier__kernel': ['linear', 'rbf'], 'classifier__C': param_range}]
jobs = -1

gs_svm = GridSearchCV(estimator=pipeline_svm,
                      param_grid=grid_params_svm,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=jobs)



#gs_svm.fit(X_train, y_train)

gs_svm.fit(df_train, df_train[target])
gs_svm_model = gs_svm.best_estimator_
pred_svm = gs_svm_model.predict(df_test)

print(classification_report(df_test[target], pred_svm))



preds_svm = gs_svm_model.predict_proba(df_test)
preds_svm = preds_svm[:,1]
fpr_svm, tpr_svm, _ = metrics.roc_curve(df_test[target].replace(to_replace=['No', 'Yes'], value=[0, 1]), preds_svm)
auc_score_svm = metrics.auc(fpr_svm, tpr_svm)
plt.title('ROC Curve')
plt.plot(fpr_svm, tpr_svm, label='AUC = {:.2f}'.format(auc_score_svm))