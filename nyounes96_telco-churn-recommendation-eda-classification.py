import pandas as pd 
import os
import numpy as np 
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
import warnings
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
df = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head(5)
print(df.isnull().sum())
df['Churn'].replace('Yes',1,inplace=True)
df['Churn'].replace('No',0,inplace=True)

#There is an issue with the Total Charges colummns (the data is stored as a string)
print(' The data type for the Total Charges Column is:',type(df['TotalCharges'].loc[4]))
#While attempting to convert this to a numeric type, ran into another problem at some positions,empty strings
print(df['TotalCharges'][(df['TotalCharges'] == ' ')])

# Drop rows where there is no value for Total Charges 
index = [488,753,936,1082,1340,3331,3826,4380,5218,6670,6754]
for i in index: 
    df.drop(i,axis=0,inplace=True)
# Convert from str to float
df['TotalCharges'].apply(float)

# Inspecting frequency in the different demographic variables that are not related to the service
dem = ['gender','SeniorCitizen','Partner','Dependents']

for i in dem: 
    sns.barplot(x = df.groupby(str(i))['Churn'].sum().reset_index()[str(i)]\
                , y = df.groupby(str(i))['Churn'].sum().reset_index()['Churn'],)
    plt.show()
    print(df.groupby(str(i))['customerID'].count().reset_index())

cat = ['PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

for i in cat: 
    sns.barplot(x = df.groupby(str(i))['Churn'].sum().reset_index()[str(i)]\
                , y = df.groupby(str(i))['Churn'].sum().reset_index()['Churn'],)
    plt.show()
    print(df.groupby(str(i))['customerID'].count().reset_index())
    
pay = ['Contract','PaperlessBilling','PaymentMethod']

for i in pay: 
    sns.barplot(x = df.groupby(str(i))['Churn'].sum().reset_index()[str(i)]\
                , y = df.groupby(str(i))['Churn'].sum().reset_index()['Churn'])
    plt.show()
    print(df.groupby(str(i))['customerID'].count().reset_index())


# Convert Binary Categories to 0's and 1's
df['Partner'].replace('Yes',1,inplace=True)
df['Partner'].replace('No',0,inplace=True)
df['Dependents'].replace('Yes',1,inplace=True)
df['Dependents'].replace('No',0,inplace=True)
df['gender'].replace('Male',1,inplace=True)
df['gender'].replace('Female',0,inplace=True)
df['PhoneService'].replace('Yes',1,inplace=True)
df['PhoneService'].replace('No',0,inplace=True)
df['PaperlessBilling'].replace('Yes',1,inplace=True)
df['PaperlessBilling'].replace('No',0,inplace=True)

## Prepare Categorical Variables with more than 2 categories
cat_X = df[['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',\
            'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',\
           'Contract','PaymentMethod']]
# Dummy Categorical Variables 
for i in cat_X: 
    cat_X = pd.concat([cat_X,pd.get_dummies(cat_X[str(i)],\
                                            drop_first=True,prefix=str(i))],axis=1)


cat_X = cat_X.drop(columns=['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',\
            'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',\
           'Contract','PaymentMethod'])

features = pd.concat([df[['tenure','Partner','Dependents','gender','PhoneService',\
                          'PaperlessBilling','MonthlyCharges','TotalCharges']],cat_X],axis=1)

# Used stratified split as the classes are imbalanced
X=features
y= df['Churn']
X_train, X_test, y_train, y_test = train_test_split(features, df['Churn'], \
                                                    test_size=0.33, random_state=42,stratify=y)
my_DT = tree.DecisionTreeClassifier(max_depth=3)
my_DT.fit(X_train, y_train)
dot_data = StringIO()
export_graphviz(my_DT, out_file=dot_data,feature_names=features.columns,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

X = features
y = df['Churn']
depth_range = np.arange(1,50,1)
val_scores = []
for d in depth_range:
    my_DT = tree.DecisionTreeClassifier(max_depth=d)
    scores = cross_val_score(my_DT, X, y, cv=10, scoring='accuracy')
    val_scores.append(scores.mean())
print(val_scores)

#Plot results from cross-validation
fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(10,5))
ax1.plot(depth_range, val_scores)
ax1.set_xlabel('Max_Depth Values')
ax1.set_ylabel('Cross-Validated Accuracy Scores')

# A more zoomed in version of the first plot
ax2.plot(depth_range,val_scores)
ax2.set_xlim(1,15)
ax2.set_xlabel('Max_Depth Values')
ax2.set_ylabel('Cross-Validated Accuracy Scores')


my_DT = tree.DecisionTreeClassifier(max_depth=3)
my_DT.fit(X_train,y_train)
print(my_DT.score(X_train,y_train))
print(my_DT.score(X_test,y_test))

# What are the 10 most important features for classification ? 
imp = pd.DataFrame(my_DT.feature_importances_).sort_values(by=0,ascending=False).\
head(10).index.values
imp_vals = pd.DataFrame(my_DT.feature_importances_).sort_values(by=0,ascending=False).\
head(10)

for i,j in zip(imp,imp_vals[0]):
    print(features.columns[i],j)
    
y_pred = my_DT.predict(X_test)
def cm(pred):
    cm = confusion_matrix(y_test, pred)
    fig = plt.plot(figsize=(8,5))
    sns.heatmap(cm,annot=True,cmap='Blues',fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    print(f1_score(y_test,pred))
    return plt.show()
cm(y_pred)

y_proba_DT = my_DT.predict_proba(X_test)
def roc_auc(prediction,model):
    fpr, tpr, thresholds = metrics.roc_curve(y_test,prediction)
    auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic '+str(model))
    plt.plot(fpr, tpr, color='blue', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'--',color='red')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return plt.show()
roc_auc(y_proba_DT[:, 1],'Decision Tree Classifier')
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF.fit(X_train,y_train)
print(RF.score(X_train,y_train))
print(RF.score(X_test,y_test))
warnings.filterwarnings('ignore')
# Number of trees in random forest
n_estimators = np.arange(10,1000,10)
# Number of features to consider at every split
# Maximum number of levels in tree
max_depth = np.arange(1,25,2)
# Minimum number of samples required to split a node
min_samples_split = [2,4,8]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf_random = RandomizedSearchCV(estimator = RF, param_distributions = grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)



print(rf_random.best_score_)
print(rf_random.best_params_)
RFR=RandomForestClassifier(bootstrap= True, max_depth= 11, min_samples_split=\
 2, n_estimators=30,min_samples_leaf= 4)
RFR.fit(X_train,y_train)
print(RFR.score(X_train,y_train))
print(RFR.score(X_test,y_test))
y_pred_rf = RFR.predict(X_test)
cm(y_pred_rf)
y_proba_rf = RFR.predict_proba(X_test)
roc_auc(y_proba_rf[:,1],'Random Forest Classifier')
lr = LogisticRegression()
print(lr.fit(X_train,y_train))
print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))
penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)
hyperparameters = dict(C=C, penalty=penalty)
clf = GridSearchCV(lr, hyperparameters, cv=5, verbose=0)
grid_model = clf.fit(X_train, y_train)
print('Best Penalty:', grid_model.best_estimator_.get_params()['penalty'])
print('Best C:', grid_model.best_estimator_.get_params()['C'])
print(grid_model.score(X_train,y_train))
print(grid_model.score(X_test,y_test))
y_pred_lr = grid_model.predict(X_test)
cm(y_pred_lr)
y_proba_lr = grid_model.predict_proba(X_test)
roc_auc(y_proba_lr[:,1],'Logistic Regression')