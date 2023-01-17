import pandas as pd
import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
# reading .csv file
data=pd.read_csv('../input/predicting-employee-status/employee_data (1).csv')
data.head()#getting top 5 rows of our dataframe
data.dtypes
data.describe(include='all')#getting all summary statistics of our data
data.isnull().sum()#finding number of null values in individual column
data.status.value_counts()#getting count of different classes in a column 
sns.countplot(x='status',data=data)
plt.title('status distribution')
sns.countplot(x='status',data=data,hue='department')
plt.title('status distribution vs department')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
data.dtypes
#creating a function which splits a dataframe into train,validation and test
def split_x(df):
    train,test=train_test_split(df,test_size=0.3,random_state=1)
    val,test=train_test_split(test,test_size=0.5,random_state=1)
    return train,val,test
#splitting data in train,validation and test
train,val,test=split_x(data)
print('train shape:',train.shape,'\n','val shape:',val.shape,'\n','test shape:',test.shape,'\n')
train.describe(include='all')
val.describe(include='all')
#getting percentage of missing values in each column
round(train.isnull().sum()/len(train)*100,2)
print('percent of null values in department:',round(train.department.isnull().sum()/len(train)*100,2),'%')
train.department.isnull().sum()
train.department.value_counts()
sns.countplot(x='department',data=train,order=train.department.value_counts().index)
plt.title('Department wise distribution')
plt.xticks(rotation=45)
#after complete examination of data creating a single function to fill all na's in a dataframe
def fill_all_na(df):
    #creating a new department 'No_dept'(not sure about what to impute it with)...later will create a categorical column
    df.department.fillna(value='No_dept',axis=0,inplace=True)
    df.filed_complaint.fillna(value=0,axis=0,inplace=True)#filling na's with 0...beacuse most probably people will not file a complaint.
    df.last_evaluation.fillna(value=0.716819,axis=0,inplace=True)#filling with the mean
    df.recently_promoted.fillna(value=0,axis=0,inplace=True)#filling na's with 0
    df.satisfaction.fillna(value=0.622162,axis=0,inplace=True)
    df.tenure.fillna(value=3.50218,axis=0,inplace=True)
    return df
train=fill_all_na(train)
val=fill_all_na(val)
test=fill_all_na(test)
train.describe(include='all')
round(train.isnull().sum()/len(train)*100,2)
train.dtypes
#creating a function to convert few variables in categorical type.
def into_category(df):
    cols=['department','filed_complaint','recently_promoted','salary','status']
    df[cols]=df[cols].astype('category')
    return df
train=into_category(train)
val=into_category(val)
test=into_category(test)
train.dtypes
#creating a new columnn 'No_dept' which gives a value 1 if a row in department column has a string 'No_dept'
#else 0....it is to give 
def new_col(df):
    df['no_dept']=[1 if x=='No_dept' else 0 for x in df['department']]
    df['no_dept']=df['no_dept'].astype('category')
    return df
train=new_col(train)
val=new_col(val)
test=new_col(test)

train.dtypes
train.describe(include='all')
sns.boxplot(x='status',data=train,y='satisfaction')
plt.title('status vs satisfaction')
#observations
#people who leave are usually less satisfied of their job(role)
sns.boxplot(x='status',y='avg_monthly_hrs',data=train)  
plt.title('status vs avg_monthly_hrs')
#apart of not being satisfied they work for more hours than other employees.
sns.countplot(x='status',data=train,hue='filed_complaint')
plt.title('status vs count vs file_complaint')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.boxplot(x='status',data=train,y='last_evaluation')
plt.title('status vs last_evaluation')
#people who left the company usually scored more than those in the company.
sns.countplot(x='n_projects',data=train,hue='status')
plt.title('#projects vs count vs status')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.boxplot(x='n_projects',data=train,y='last_evaluation',hue='status')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('#projects vs last_evaluation vs status')
sns.boxplot(x='recently_promoted',data=train,y='last_evaluation',hue='status')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('recently promoted vs last_evaluation vs status')
sns.boxplot(x='salary',data=train,y='last_evaluation',hue='status',order=['low','medium','high'])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('salary vs last_evaluation vs status')
sns.boxplot(x='tenure',data=train,y='satisfaction',hue='status',order=[0,1,2,3,4,5,6,7,8,9,10])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


#function to split data into predictors and targets
def splitting_y(df):
  df_x=df.loc[:,df.columns!='status']
  df_y=df['status']
  return df_x,df_y
train_x,train_y=splitting_y(train)
val_x,val_y=splitting_y(val)
test_x,test_y=splitting_y(test)
train_x.describe(include='all')
train_x.dtypes
#creating list of numerical and categorical columns
num_cols=['avg_monthly_hrs','last_evaluation','n_projects','satisfaction','tenure']
cat_cols=['department','filed_complaint','recently_promoted','salary','no_dept']
#creating numerical and categorical dataframes for train,validation and test
train_num_x=train_x.loc[:,num_cols]
train_cat_x=train_x.loc[:,cat_cols]
val_num_x=val_x.loc[:,num_cols]
val_cat_x=val_x.loc[:,cat_cols]
test_num_x=test_x.loc[:,num_cols]
test_cat_x=test_x.loc[:,cat_cols]
#standardizing the numerical columns using standardscaler()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_num_x[train_num_x.columns])
train_num_x[train_num_x.columns] = scaler.transform(train_num_x[train_num_x.columns])
val_num_x[val_num_x.columns] = scaler.transform(val_num_x[val_num_x.columns])
test_num_x[test_num_x.columns] = scaler.transform(test_num_x[test_num_x.columns])
#creating dummy variables of categorical variables
train_dummy_x=pd.get_dummies(train_cat_x,drop_first=True)
val_dummy_x=pd.get_dummies(val_cat_x,drop_first=True)
test_dummy_x=pd.get_dummies(test_cat_x,drop_first=True)
#combining standardized numerical dataframe and dummyfied categorical dataframe.
full_train_x=pd.concat([train_num_x,train_dummy_x],axis=1)
full_val_x=pd.concat([val_num_x,val_dummy_x],axis=1)
full_test_x=pd.concat([test_num_x,test_dummy_x],axis=1)
full_train_x.describe(include='all')
full_val_x.isnull().sum()
#no null values are present
#importing required packages for model building
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
knn=KNeighborsClassifier()
param_grid_knn_1={
    'n_neighbors':np.arange(3,15,1),
    'weights':['uniform','distance'],
    'algorithm':['auto','brute'],
}
knn_random=RandomizedSearchCV(estimator=knn,param_distributions=param_grid_knn_1,n_iter=1000,n_jobs=-1,cv=5,verbose=1)
knn_random.fit(full_train_x,train_y)#training knn algorithm
knn_random.best_params_#getting best performing hyperparameters
#using grid search to search for best parameters around the output of random search parameters
param_grid_knn2={
    'n_neighbors':np.arange(2,10,1),
    'weights':['uniform','distance'],
    'algorithm':['auto','brute'],
}
knn_grid=GridSearchCV(estimator=knn,param_grid=param_grid_knn2,n_jobs=-1,cv=5,verbose=1)
#training knn model with grid search
knn_grid.fit(full_train_x,train_y)
knn_grid.best_params_#best parameters of grid search are similar to random search
from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt
knn2=KNeighborsClassifier(algorithm= 'auto', n_neighbors= 2, weights= 'uniform')
#learning curves for knn
plot_learning_curve(estimator=knn2,title='knn_learning_curves',X=full_train_x,y=train_y,ylim=(0.75,1.05),cv=5)
#learning seems to be good we will train knn with best parameters
knn2.fit(full_train_x,train_y)
#predicting the validation labels 
knn_prediction_val=knn2.predict(full_val_x)
#classification report to get all important metrics required for classification
from sklearn.metrics import classification_report
print('knn classification report\n')
print(classification_report(val_y,knn_prediction_val))
from sklearn import tree
#creating a random search for some hyper parameters given in param_grid_1
dt=tree.DecisionTreeClassifier()
param_grid_1={
    'criterion':['gini','entropy'],
    'max_depth':np.arange(4,20,1),
    'min_samples_split':np.arange(0.001,0.1,0.01),
    'max_features':['log2','sqrt','auto'],
    'min_weight_fraction_leaf':np.arange(0.001,0.25,0.05)
}
r_search=RandomizedSearchCV(dt,param_distributions=param_grid_1,n_iter=1000,verbose=1)
r_search.fit(full_train_x,train_y)
#getting best performing hyper parameters from random search 
r_search.best_params_
#creating another parameter grid for grid search by taking values around the best performing random search
#hyper parameters
param_grid_2={
    'criterion':['gini','entropy'],
    'max_depth':np.arange(12,18,1),
    'min_samples_split':np.arange(0.001,0.01,0.01),
    'max_features':['log2','sqrt'],
    'min_weight_fraction_leaf':np.arange(0.001,0.05,0.01)
}
dt=tree.DecisionTreeClassifier()
grid_search=GridSearchCV(estimator=dt,param_grid = param_grid_2,cv=5,verbose=1,n_jobs=-1)
grid_search.fit(full_train_x,train_y)
grid_search.best_params_#getting best parameters of grid search
#learning curve for decision tree
dt=tree.DecisionTreeClassifier(criterion= 'gini',max_depth= 16,max_features= 'log2',min_samples_split= 0.001,min_weight_fraction_leaf= 0.001)
plot_learning_curve(estimator=dt,title='dt_learning_curves',X=full_train_x,y=train_y,ylim=(0.75,1.05),cv=5)
#learning curve seems ok....fitting dt with best parameters
dt=tree.DecisionTreeClassifier(criterion= 'gini',max_depth= 16,max_features= 'log2',min_samples_split= 0.001,min_weight_fraction_leaf= 0.001)
dt.fit(full_train_x,train_y)
#predicting validation labels
dt_prediction_val=dt.predict(full_val_x)
#getting classification report for decision tree
print('Decision tree classification report\n\n')
print(classification_report(val_y,dt_prediction_val))
#importing random forest classifier
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.get_params
#creating parameter grid for random search
grid_forest_1={'criterion':['gini','entropy'],
      'n_estimators':np.arange(5,200,1),
      'max_depth':np.arange(5,20,1),
      'min_samples_split':np.arange(0.001,0.1,0.01),
      'max_features':['log2','sqrt','auto'],    
      'min_weight_fraction_leaf':np.arange(0.001,0.25,0.05)
}
#getting best parameters form random search
rf_random=RandomizedSearchCV(estimator=rf,param_distributions=grid_forest_1,n_iter=500,n_jobs=-1,cv=5,verbose=1)
rf_random.fit(full_train_x,train_y)
rf_random.best_params_
grid_forest_2={'criterion':['entropy'],
      'n_estimators':np.arange(115,135,2),
      'max_depth':(17,18,19,20,21),
      'min_samples_split':np.arange(0.001,0.01,0.005),
      'max_features':['log2'],    
      'min_weight_fraction_leaf':np.arange(0.0001,0.1,0.005)
}
rf=RandomForestClassifier()
grid_search_rf=GridSearchCV(estimator=rf,param_grid = grid_forest_2,cv=3,n_jobs=-1,verbose=1)
grid_search_rf.fit(full_train_x,train_y)
grid_search_rf.best_params_
rf=RandomForestClassifier(criterion='entropy',max_depth= 17,max_features='log2',min_samples_split= 0.001,min_weight_fraction_leaf= 0.001,n_estimators= 129)
rf.fit(full_train_x,train_y)
plot_learning_curve(estimator=rf,title='RF_learning_curves',X=full_train_x,y=train_y,ylim=(0.75,1.05),cv=5)
rf_predictions_val_y=rf.predict(full_val_x)
print(classification_report(val_y,rf_predictions_val_y))
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.get_params
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
param_grid_gbc_1={
    'n_estimators':np.arange(50,200,10),
    'learning_rate':np.arange(0.01,0.3,0.01),
    'subsample':np.arange(0.5,1.0,0.1),
    'max_depth':np.arange(2,7,1),
    'max_features':['sqrt','log2'],
    'verbose':[1]
}
random_gbc=RandomizedSearchCV(estimator=gbc,param_distributions=param_grid_gbc_1,n_jobs=-1,n_iter=1000,verbose=1)
random_gbc.fit(full_train_x,train_y)
random_gbc.best_params_
#searching for better parameters in random search with grid search
param_grid_gbc_2={'learning_rate':np.arange(0.1,0.2,0.05),
 'max_depth': (5,6,7,8),
 'max_features': ['log2'],
 'n_estimators': np.arange(158,162,1),
 'subsample': np.arange(0.8,0.9,0.02),
}
grid_search_gbc=GridSearchCV(estimator=gbc,param_grid=param_grid_gbc_2,n_jobs=-1,cv=3,verbose=1)
grid_search_gbc.fit(full_train_x,train_y)
grid_search_gbc.best_params_
gradient_boosting=GradientBoostingClassifier(learning_rate= 0.15,max_depth= 8,max_features= 'log2',n_estimators= 161,subsample= 0.8)

#plotting learning curves
plot_learning_curve(estimator=gradient_boosting,title='GB_learning_curves',X=full_train_x,y=train_y,ylim=(0.75,1.05),cv=5)
gradient_boosting.fit(full_train_x,train_y)
gb_prediction_val=gradient_boosting.predict(full_val_x)
print('gradient boosting classification report\n\n')
print(classification_report(val_y,gb_prediction_val))
score_df=pd.DataFrame({' ':['Employeed','Left'],'KNN-f1_score':[0.97,0.90],'Decision Tree-f1_score':[0.9,0.90],'Random_forest-f1_score':[0.98,0.93],'Gradient_boosting-f1_score':[0.99,0.96]})
score_df
#since we got same f1 scores on validation...choose random forest or gradient boosting for predictions
