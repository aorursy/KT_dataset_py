from IPython.display import Image
Image(filename='../input/heart-image/images.jpg', height='100',width='700')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas_profiling
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, precision_score, recall_score,f1_score, accuracy_score,classification_report 
from sklearn.feature_selection import RFE
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
data.head()
data.shape
data.info()
data.describe()
mn.matrix(data)
100*(data.isnull().sum()/(len(data)))
data.info()
data.head()
data.profile_report()
data.head()
data['age'] = data['age'].astype('int64')
data['platelets'] = data['platelets'].astype('int64')
data.diabetes.value_counts()
sns.heatmap(data.corr(),cmap='Wistia',annot=True)
scaler = StandardScaler()
var1 = ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_sodium','time']
data[var1] = scaler.fit_transform(data[var1])
data.head()
X = data.iloc[:,:-1]
y = data.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.7,random_state = 20)
X_train.head()
X_train.shape
X_test.shape
log_reg1 = LogisticRegression()
log_reg1.fit(X_train,y_train)
prediction1 = log_reg1.predict(X_test)
prediction1
confusion_matrix(y_test,prediction1)
recall_score(y_test,prediction1)
precision_score(y_test,prediction1)
accuracy_score(y_test,prediction1)
X_train_lm = sm.add_constant(X_train)
log_reg2 = sm.GLM(y_train,X_train_lm).fit()
log_reg2.summary()
# Dropping 'smoking' as it has a very high p value
X_train_drop_smoking = X_train.drop(['smoking'], axis = 1)
X_train_lm = sm.add_constant(X_train_drop_smoking)
log_reg3 = sm.GLM(y_train,X_train_lm).fit()
log_reg3.summary()
rfe = RFE(log_reg1, 10)             # running RFE with 13 variables as output
rfe = rfe.fit(X_train, y_train)
rfe.support_
list(zip(X_train.columns, rfe.support_, rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train_2 = X_train.drop(['creatinine_phosphokinase','smoking'],1)
X_train_lm = sm.add_constant(X_train_2)
log_reg4 = sm.GLM(y_train,X_train_2).fit()
log_reg4.summary()
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Model 5
log_reg5 = LogisticRegression()
log_reg5.fit(X_train_2,y_train)
# Dropping 'creatinine_phosphokinase' and 'smoking' from X_test as per RFE
X_test_1 = X_test.drop(['creatinine_phosphokinase','smoking'],1)
prediction2 = log_reg5.predict(X_test_1)
# Precision score 
precision_score(y_test,prediction2)
# Recall score
recall_score(y_test,prediction2)
# F1 score
f1_score(y_test,prediction2)
#------------Importing the Decision Tree Library---------------------
from sklearn.tree import DecisionTreeClassifier
# Lets fit the decision tree
dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)
# Importing required packages for visualization
from IPython.display import Image  
from six import StringIO  
from sklearn.tree import export_graphviz
!pip install pydotplus
import pydotplus, graphviz


# plotting tree with max_depth=3
dot_data = StringIO()  

export_graphviz(dt, out_file=dot_data, filled=True, rounded=True,
                feature_names=X.columns, 
                class_names=['No Death', "Death"])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)
print(accuracy_score(y_train, y_train_pred))
confusion_matrix(y_train, y_train_pred)
print(accuracy_score(y_test, y_test_pred))
confusion_matrix(y_test, y_test_pred)
print(precision_score(y_train, y_train_pred))
print(precision_score(y_test, y_test_pred))
print(f1_score(y_train, y_train_pred))
print(f1_score(y_test, y_test_pred))
print(recall_score(y_train, y_train_pred))
print(recall_score(y_test, y_test_pred))
# Function to plot the decision tree
def get_dt_graph(dt_classifier):
    dot_data = StringIO()
    export_graphviz(dt_classifier, out_file=dot_data, filled=True,rounded=True,
                    feature_names=X.columns, 
                    class_names=['Death', "No Death"])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return graph
# Function to evaluate the model
def evaluate_model(dt_classifier):
    print("Train Accuracy :", accuracy_score(y_train, dt_classifier.predict(X_train)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, dt_classifier.predict(X_train)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, dt_classifier.predict(X_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, dt_classifier.predict(X_test)))
dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)
gph = get_dt_graph(dt_default)
Image(gph.create_png())
evaluate_model(dt_default)
dt = DecisionTreeClassifier(random_state=42)
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")
%%time
grid_search.fit(X_train, y_train)
score_df = pd.DataFrame(grid_search.cv_results_)
score_df.head()
grid_search.best_estimator_
dt_best = grid_search.best_estimator_
evaluate_model(dt_best)
print(classification_report(y_test, dt_best.predict(X_test)))
gph = get_dt_graph(dt_best)
Image(gph.create_png())
