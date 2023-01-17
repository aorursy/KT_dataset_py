#Regular explanatory data analysis and plotting libraries.
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('fivethirtyeight')

#models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#model evaluators
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,precision_score,recall_score,f1_score,plot_roc_curve
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.head() #first five rows
#lets count the total number of values in the target colums.
df['target'].value_counts()
#Plot our target column
df['target'].value_counts().plot(kind='bar',color=["teal", "indigo"]);
#Finding if there is any missing values in the dataset.
df.isna().sum()
df.info()
df.describe()
df['sex'].value_counts()
# comparing target column with sex column.
pd.crosstab(df['target'],df['sex'])
pd.crosstab(df['target'],df['sex']).plot(kind='bar',color=['darkgreen','cornflowerblue'])
plt.title('Heart disease frequency for sex')
plt.xlabel('0=No Heart Disease, 1=Heart Disease')
plt.ylabel('Number')
plt.legend(['Female','Male']);
fig,ax = plt.subplots(nrows=1,
                     ncols=1,
                     figsize=(12,8))

#positive examples
ax.scatter(df['age'][df['target']==1],
          df['thalach'][df['target']==1],
          c = 'darkgreen')

#negative examples
ax.scatter(df['age'][df['target']==0],
          df['thalach'][df['target']==0],
          c = 'red')

ax.set(title='Heart disease in function of Age and Max Heart Rate',
      xlabel='Age',
      ylabel = 'Max Heart Rate')
ax.legend(['Disease','No disease']);
#age distribution
df['age'].plot(kind='hist')
pd.crosstab(df['cp'],df['target'])
pd.crosstab(df['cp'],df['target']).plot(kind='bar',
                                       figsize=(12,8),
                                       color=['springgreen','purple'])
plt.title('Heart Disease frequency Per chest pain')
plt.xlabel('Chest Pain Type')
plt.ylabel('Frequency')
plt.legend(['No Heart Disease','Heart Disease']);
df_corr = df.corr()
df_corr
corr_matrix = df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix,
           annot=True,
           linewidths=0.5,
           fmt='.2f',
           cmap='YlGnBu');
# droping target variable
X = df.drop('target',axis=1)

# only target column
y = df['target']
X.head()
y.head()
#for reproducible code
np.random.seed(45)

#spliting our data into training and testing set.
X_train,X_test,y_train,y_test = train_test_split(X, #independent variable
                                                 y, #dependent variable
                                                 test_size=0.2) #percentage of data used for testing
#lets look at the shape of our training and testing data
X_train.shape, X_test.shape, y_train.shape,y_test.shape
# Using logistic regression
log = LogisticRegression(max_iter=1000).fit(X_train,y_train)
log_score = log.score(X_test,y_test)
print('The accuracy score of Logistic regression is: {:.2f}%'.format(log_score*100))

#Using KNeighbor
knn = KNeighborsClassifier().fit(X_train,y_train)
knn_score = knn.score(X_test,y_test)
print('The accuracy of K-Nearest Neighbors is: {:.2f}%'.format(knn_score*100))

#Using Random Forest 
clf = RandomForestClassifier().fit(X_train,y_train)
clf_score = clf.score(X_test,y_test)
print('The accuracy of Random Forest is: {:.2f}%'.format(clf_score*100))
model_score={log_score,knn_score,clf_score}
model_comparison = pd.DataFrame(model_score,index=['Logistic Regression','K-Nearest Neighbors','Random Forest'])
model_comparison.plot(kind='barh')
plt.ylabel('Algorithms')
plt.xlabel('Accuracy Score')
plt.title('Accuracy Comparison between different Models')
plt.legend('Accuracy');
# Logistic Regression hyperperimeters
log_grid = {"C": np.logspace(-4, 4, 20),
            "solver": ["liblinear"]}

# Random Forest hyperperimeter
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}
np.random.seed(20)

#setup random hyperparameter search.
log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_grid,
                                cv=5,
                                n_iter=20,#try 20 different combinations of hyperparameters
                                verbose=True)

#fitting the model
log_reg.fit(X_train,y_train)
#checking the best parameters for logistic regression
log_reg.best_params_
print('The accuacy of Logistic regression using RandomizedSearchCV is: {:.2f}%'.format(log_reg.score(X_test,y_test)*100))
np.random.seed(20)

#setup random hyperparameter search.
rand = RandomizedSearchCV(RandomForestClassifier(),
                                param_distributions=rf_grid,
                                cv=5,
                                n_iter=20,#try 20 different combinations of hyperparameters
                                verbose=True)

#fitting the model
rand.fit(X_train,y_train)
#checking the best parameters for random forest
rand.best_params_
print('The accuracy of Random forest using RandomizedSearchCV is: {:.2f}%'.format(rand.score(X_test,y_test)*100))
log_search = GridSearchCV(LogisticRegression(),
                          param_grid=log_grid,
                          cv=5,
                          verbose=True)
#fitting the model
log_search.fit(X_train,y_train)
#checking the best hyperperimeter.
log_search.best_params_
print('The accuacy of Logistic regression using GridSearchCV is: {:.2f}%'.format(log_search.score(X_test,y_test)*100))
#making predictions on test data
y_preds = log_search.predict(X_test)
y_preds
y_test.values
from sklearn.metrics import plot_roc_curve
#plotting the curve
plot_roc_curve(log_search,X_test,y_test);
#displaying confusion matrix.
print(confusion_matrix(y_test,y_preds))
#plotting confusion matrix.
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(log_search,X_test,y_test);
# classification report
print(classification_report(y_test, y_preds))
# our best hyperparameter
log_search.best_params_
from sklearn.model_selection import cross_val_score

#lets use the best model with best hyperpararmeters.
clf = LogisticRegression(C=0.615848211066026,
                        solver='liblinear')
#cross validation accuracy score
cross_val_accuracy = cross_val_score(clf,X,y,cv=5,scoring='accuracy')
cross_val_accuracy
#Let's find the average of the above 5 values.
cross_val_accuracy = np.mean(cross_val_accuracy)
print('Cross validation accuracy score is: {:.2f}'.format(cross_val_accuracy))
#cross validation precision score.
cross_val_precision = cross_val_score(clf,X,y,cv=5,scoring='precision')
cross_val_precision
#Let's find the average of the above 5 values.
cross_val_precision = np.mean(cross_val_precision)
print('Cross validation Precision score is: {:.2f}'.format(cross_val_precision))
#cross validation recall score.
cross_val_recall = cross_val_score(clf,X,y,cv=5,scoring='recall')
cross_val_recall
#Let's find the average of the above 5 values.
cross_val_recall = np.mean(cross_val_recall)
print('Cross validation recall score is: {:.2f}'.format(cross_val_recall))
#cross validation recall score.
cross_val_f1 = cross_val_score(clf,X,y,cv=5,scoring='f1')
cross_val_f1
#Let's find the average of the above 5 values.
cross_val_f1 = np.mean(cross_val_f1)
print('Cross validation f1 score is: {:.2f}'.format(cross_val_f1))
# Visualizing cross-validated metrics
cross_val_metrics = pd.DataFrame({"Accuracy": cross_val_accuracy,
                            "Precision": cross_val_precision,
                            "Recall": cross_val_recall,
                            "F1": cross_val_f1},
                          index=[0])
cross_val_metrics.T.plot.bar(title="Cross-Validated Metrics", legend=False);
clf.fit(X_train,y_train)
features_dict = dict(zip(df.columns, list(clf.coef_[0])))
features_dict
# Visualize feature importance
features_df = pd.DataFrame(features_dict, index=[0])
features_df.T.plot.bar(title="Feature Importance", legend=False);
pd.crosstab(df["sex"], df["target"])
# Contrast slope (positive coefficient) with target
pd.crosstab(df["slope"], df["target"])