
import pandas as pd
import numpy as np
import seaborn as sns
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from imblearn.over_sampling import SMOTE
from sklearn.compose import make_column_transformer
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score, roc_curve,precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, precision_score,recall_score,accuracy_score, plot_roc_curve, cohen_kappa_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import eli5
pd.options.mode.chained_assignment = None
%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Import dataset 

df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
print(f'Dataset contains {df.shape[0]} rows, and {df.shape[1]} columns.')
# Inspect all the features data types and missing values
pd.concat([df.dtypes,df.isna().sum(),df.nunique()],axis=1).rename(columns={0: "Types", 1: "Nulls", 2:'Nunique'})
df.drop('customerID', axis=1, inplace=True)
# Replace spaces with null values in column of total charges
df['TotalCharges'] = df["TotalCharges"].replace(" ",np.nan).astype(float)
print(f'TotalCharges number of Nan values: {df["TotalCharges"].isna().sum()}')
print(f'TotalCharges percentage of Nan values: {np.round(df["TotalCharges"].isna().sum()/len(df)*100,3)}%')

# Fill misiing values with midian
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Describe statistis for numerical features
df.describe().T.round(2)
val = df["Churn"].value_counts().values.tolist()
labels = 'Not Churned', 'Churned'
fig1, ax1 = plt.subplots(figsize=(8,8))
ax1.pie(val, explode=(0,0.1), labels=labels, autopct='%1.1f%%',colors = [ "#56738f" ,'#e74c3c'],
        shadow=True, startangle=90,  textprops={'fontsize': 20, 'weight':'bold'})
ax1.axis('equal') 
plt.show()
# visualizing the Probability Density of each continuous variable, and churn rate by deciles. 
df_num = df.copy()
df_num['Churn'] = df_num['Churn'].map({'No':0, 'Yes':1})

for i in ('tenure','MonthlyCharges', 'TotalCharges'):
    
    plt.figure(figsize=(14,5))
    sns.set(font_scale = 1.1)
    sns.kdeplot(df_num[df_num['Churn']== 1][str(i)], color="#e74c3c", shade=True, label = 'Churn', alpha=0.7)
    sns.kdeplot(df_num[df_num['Churn']== 0][str(i)], color="#34495e", shade=True, label = 'Not Churn', alpha=0.7)
    plt.title(f'Customer Churned - {i}', size=15,fontweight='bold')
    plt.show();
    

    
    plt.figure(figsize=(14,5))
    sns.set(font_scale = 1.1)
    churn_mc = df_num.groupby(pd.qcut(df_num[i],10,precision=0))[['Churn']].mean().round(2)
    sns.set(font_scale = 1.1)
    ax = sns.barplot(data=churn_mc, x=churn_mc.index,y=churn_mc.Churn,hue=churn_mc['Churn'],palette=("OrRd"),dodge=False);
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()]); plt.title(f'By {i} deciles');plt.legend("")
    plt.show()
    
    
# countplot for each category + churn perecetage 
plt.style.use('seaborn-dark')
fig, ax = plt.subplots(figsize = (10, 100))

for n, i in enumerate(df.columns[df.nunique() <5].drop('Churn')):
    
    ax = plt.subplot(20, 1, n+1)
    no = df[i].value_counts().plot(kind='bar', ax=ax,width=.5,  color="#34495e", alpha=0.8)
    plt.title(f'Customer Churned by {i}', fontweight='bold',size=20)
    churn = df[df.Churn=='Yes'][i].value_counts().plot(kind='bar',ax=ax, width=.4, color="#e74c3c", alpha=0.75)
    plt.xticks(rotation=0,fontsize=17);
    plt.legend(['Total', 'Churn rate'],fontsize= 15)
    fig.subplots_adjust(hspace=0.3)
    bars = ax.patches
    half = int(len(bars)/2)
    left_bars = bars[:half]
    right_bars = bars[half:]
    
    for left, right in zip(left_bars, right_bars):
        height_l = left.get_height()
        height_r = right.get_height()
        total = height_l
        ax.text(right.get_x() + right.get_width()/2, 70,
                '{0:.0%}'.format(height_r/(total)), ha="center", fontsize=25, color='w')
    
    plt.xlabel(""); 
    plt.tight_layout();

# In order to get all the correlation between the features, I first factorized the features of the object type,
# and then I joined the numerical data types.
corr = (df.select_dtypes(include='object').apply(lambda x: pd.factorize(x)[0])\
       .join(df.select_dtypes(include=['float','int64']))).corr().round(2)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(14, 10))
    ax = sns.heatmap(corr,mask=mask, cmap='Blues',annot=True,annot_kws={"size":8},linewidths=0.5);
    sns.set(font_scale=1.5);
    plt.title('Cooralation Matrix', size=20);
# Correlation of Churn with other variables
corr['Churn'].drop('Churn').sort_values().plot(kind='bar', figsize=(14,6));
binary_categories=[]
for col in df.columns:
    if df[col].isin(['Yes']).any():
        binary_categories.append(col)
        print(col)
binary_categories_dic = {'Yes': 1,
                         'No': 0,
                         'No internet service': 0,
                         'No phone service': 0}

for col in binary_categories:
    df[col] = df[col].map(binary_categories_dic)
    
# ordering columns by dtypes
df = df[df.dtypes.sort_values().index.tolist()]

# grouping features by type (numerical, categorical)
num_features = ['tenure','MonthlyCharges', 'TotalCharges']
cat_features = df.select_dtypes(include='object').columns.tolist()
name_num_features = df.drop(['Churn'], axis=1).select_dtypes(include=['int64','float']).columns.tolist()
df.info()
# stepes for pipeline transformation
column_trans = make_column_transformer(
                (StandardScaler(), num_features),
                (OneHotEncoder(), cat_features),
                 remainder='passthrough')
# Splitting the data-set into independent (Churn) and dependent features

X = df.drop('Churn', axis=1)
y = df.Churn.values

# Split the dataset into train (60%), validate(20%), test (20%)
# We'll use stratify parameter to ensure the proportion of the class labels in each subset is the same.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=12)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=12)
print(f'training set: {X_train.shape[0]}')
print(f'validation set: {X_val.shape[0]}')
print(f'testing set: {X_test.shape[0]}')
res_features = num_features
pass_features = [e for e in name_num_features if e not in num_features]
pipe = Pipeline([('ct', column_trans), ('classifier', LogisticRegression())]).fit(X_test, y_test)
exp_features = res_features+list(pipe.named_steps['ct'].named_transformers_['onehotencoder']\
                                 .get_feature_names(cat_features)) + pass_features

# Creating a parallel dateset for later exploration and feature engineering & selection 
sm = SMOTE(random_state=4)
X_train_res = pd.DataFrame(column_trans.fit_transform(X_train), columns = exp_features)
X_test_res  = pd.DataFrame(column_trans.transform(X_test), columns = exp_features)
X_train_smote, y_train_smote = sm.fit_sample(X_train_res, y_train)
X_train_smote = pd.DataFrame(X_train_smote, columns = exp_features)
# Creating ML-models dictionary (with default setting)

models={
    'LogReg'            : LogisticRegression(max_iter=2000),
    'DecisionTree'      : DecisionTreeClassifier(),
    'RandomForest'      : RandomForestClassifier(),
    'SVM'               : SVC(probability=True),
    'KNN'               : KNeighborsClassifier(),
    'AdaBoost'          : AdaBoostClassifier(),
    'XGBoost'           : XGBClassifier(),
    'LightGBM'          : LGBMClassifier(),
    
}


# Cross validation on train data

df_metrics=pd.DataFrame([])
predicts={}

for model_name in models.keys():
    
    metrics={}
    pipe = Pipeline([('ct', column_trans), ('sm', SMOTE(random_state=12)), ('classifier', models[model_name])])
    pred = pipe.fit(X_train, y_train)
    predicts[model_name] = pred.predict(X_val)
    metrics['accuracy']= cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()
    metrics['roc_auc'] = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc').mean()
    metrics['precision'] = cross_val_score(pipe, X_train, y_train, cv=5, scoring='precision').mean()
    metrics['recall'] = cross_val_score(pipe, X_train, y_train, cv=5, scoring='recall').mean()
    metrics['f1'] = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1').mean()
    df_metrics=pd.concat([df_metrics,pd.DataFrame(metrics,index=[model_name]).T],axis=1)
    


df_metrics.T.style.highlight_max(color='lightgreen').set_precision(3)
df_metrics.T.plot(kind='bar', figsize=(14,7));
plt.legend(loc=(1.04,0)); plt.xticks(rotation=45);
# Compare how well each algorithm used to identify true positive (sensitivity) vs. false positive (specificity).

model_test = {}
fig, ax = plt.subplots(figsize=(12,8))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--');
plt.title('ROC Curves'); ax.margins(0,0)

for n, model_name in enumerate(models.keys()):
  
    pipe = Pipeline([('ct', column_trans), ('sm', SMOTE(random_state=12)), ('classifier', models[model_name])])
    model_test[model_name] = pipe.fit(X_train, y_train)
    if n == 0:
        disp = plot_roc_curve(model_test[model_name], X_val,  y_val, name=model_name, ax=ax)
    else:
        plot_roc_curve(model_test[model_name], X_val, y_val, name=model_name, ax=disp.ax_, lw=2) 
plt.subplots(figsize = (15, 40))

for n, model in enumerate(models.keys()):
    
    plt.subplot(8, 2, n+1)
    cm = confusion_matrix(y_val, predicts[model])
    sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True,
            cmap = 'Blues_r', cbar=False, annot_kws={"fontsize":20});
    plt.title(model)
    plt.ylabel('Actual Churn');
    plt.xlabel('Predicted Churn');
    plt.tight_layout()   
def cv_new_features(new_X_train_features):
    
    ''' 
    cross validation without and with the new features.
     '''
    
    # cross-validation without new features
    pipe = Pipeline([('sm', SMOTE(random_state=12)), ('classifier', LogisticRegression(max_iter=2000))])
    sf_auc = cross_val_score(pipe, X_train_res, y_train, cv=5, scoring='roc_auc').mean()
    sf_f1  = cross_val_score(pipe, X_train_res, y_train, cv=5, scoring='f1').mean()
    # cross-validation with new features
    pipe = Pipeline([('sm', SMOTE(random_state=12)), ('classifier', LogisticRegression(max_iter=2000))])
    sf_auc2 = cross_val_score(pipe, new_X_train_features, y_train, cv=5, scoring='roc_auc').mean()
    sf_f2  = cross_val_score(pipe, new_X_train_features, y_train, cv=5, scoring='f1').mean()
    print(f'Results before adding new features, Auc: {round(sf_auc,3)}  f1:{round(sf_f1,3)}')
    print(f'Results after adding new features, Auc: {round(sf_auc2,3)}  f1:{round(sf_f2,3)}')
# Add Kmeans clustering. We will create another column (km) which represents the cluster to which the sample belongs,
# using first the Kmeans algorithm which basically refers to the collection of data points that are aggregated together
# due to certain similarities. W'll yuse defualt setting of 8 clusters.

X_train_km, X_test_km =X_train_res, X_test_res
km = KMeans()
km.fit(X_train_km)
X_train_km['km'] = km.labels_
X_test_km['km'] = km.predict(X_test_km)
ohe = OneHotEncoder() # one hot encoding 8 clusters.
X_train_km = X_train_km.join(pd.DataFrame(ohe.fit_transform(X_train_km['km'].values.reshape(-1, 1))\
                                          .toarray())).drop('km', axis=1)
X_test_km = X_test_km.join(pd.DataFrame(ohe.transform(X_test_km['km'].values.reshape(-1, 1))\
                                          .toarray())).drop('km', axis=1)

cv_new_features(X_train_km)
# add polynomial features (created by raising existing features to an exponent) to the numerical features(num_features).
# 
X_train_pf, X_test_pf = X_train_res, X_test_res
data = X_train_pf[num_features]
trans = PolynomialFeatures(degree=2, interaction_only=False)
data = trans.fit_transform(data)
pf = pd.DataFrame(data, columns=trans.get_feature_names())
print(pf.head()) # the numerical features ('tenure',MonthlyCharges',TotalCharges' as x0,x1,x2) and their polynomials generated.
X_train_pf = pd.concat([X_train_res, pf.drop(['1','x0', 'x1', 'x2'], axis=1)], axis=1)
cv_new_features(X_train_pf)
# Build a dictionary of hyperparameters for evrey ML algorithm
search_parms = dict()

search_parms['LogReg'] =            {'classifier__C': [0.1, 0.5, 0.75, 1, 10],
                                    'classifier__solver':['lbfgs','saga']}


search_parms['DecisionTree'] =     {
                                    'classifier__max_depth':[3,6,10,15,25,30,None],
                                    'classifier__min_samples_leaf':[1,2,5,10,15,30],
                                    'classifier__max_leaf_nodes': [2, 5,10]}

search_parms['RandomForest'] =     {
                                    'classifier__n_estimators': [10, 100, 1000],
                                    'classifier__max_depth':[5,8,15,25,30,None],
                                    'classifier__min_samples_leaf':[1,2,5,10,15,30],
                                    'classifier__max_leaf_nodes': [2, 5,10]}

search_parms['SVM'] =              {
                                   'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                                   'classifier__gamma': [1,0.1,0.01,0.001],
                                   'classifier__kernel': ['rbf', 'poly', 'sigmoid']}

search_parms['KNN'] =              {
                                    'classifier__n_neighbors':[4,5,6,7,8,9,10],
                                    'classifier__leaf_size':[1,2,3,5],
                                    'classifier__weights':['uniform', 'distance'],
                                    'classifier__algorithm':['auto', 'ball_tree','kd_tree','brute']}


search_parms['AdaBoost'] =         {
                                    'classifier__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.5],
                                    'classifier__n_estimators': [50, 75, 100, 200, 300, 500]}


search_parms['XGBoost'] =          {
                                   'classifier__min_child_weight': [1, 5, 10],
                                   'classifier__gamma': [0.5, 1, 1.5, 2, 5],
                                   'classifier__subsample': [0.6, 0.8, 1.0],
                                   'classifier__colsample_bytree': [0.6, 0.8, 1.0],
                                   'classifier__max_depth': [3, 4, 5]}

search_parms['LightGBM'] =         {
                                   'classifier__max_depth': [25,50, 75],
                                   'classifier__learning_rate' : [0.01,0.05,0.1],
                                   'classifier__num_leaves': [300,900,1200],
                                   'classifier__n_estimators': [200]}

# Creating scores metrics dictionary

scorers = {
            'accuracy': 'accuracy',
            'AUC':      'roc_auc',
            'Recall':   'recall',
            'precision':'precision',
            'f1':        'f1'}


df_metrics_gs=pd.DataFrame([])
predicts={}
b_parms={}

for model_name in models.keys():

    metrics={}
    pipe =                Pipeline([('ct', column_trans), ('sm', SMOTE(random_state=12)), ('classifier', models[model_name])])
    predicts[model_name] = RandomizedSearchCV(pipe, search_parms[model_name],cv=5,scoring=scorers,
                                              refit='AUC', random_state=42).fit(X_train, y_train)
    metrics['accuracy']  = predicts[model_name].cv_results_['mean_test_accuracy'].mean()
    metrics['precision'] = predicts[model_name].cv_results_['mean_test_precision'].mean()
    metrics['recall']    = predicts[model_name].cv_results_['mean_test_Recall'].mean()
    metrics['roc_auc']   = predicts[model_name].cv_results_['mean_test_AUC'].mean() 
    metrics['f1']        = predicts[model_name].cv_results_['mean_test_f1'].mean()
    b_parms[model_name]  = predicts[model_name].best_params_ 
    df_metrics_gs =        pd.concat([df_metrics_gs,pd.DataFrame(metrics,index=[model_name]).T],axis=1)



df_metrics_gs.T.style.highlight_max(color='lightgreen').set_precision(3)
# Ranked Models after RandomsearchCV based on roc_auc and f1 score
df_metrics_gs.T[['roc_auc', 'f1']].round(3).rank(pct=True).sum(axis=1).sort_values(ascending=False)
# Results of the models chosen from the cross validation process.

top_models = df_metrics_gs.T.iloc[[0,5]].round(3)
top_models
# Let's sort all the features by correlation to the target
corr_features= abs(X_train_smote.corrwith(pd.DataFrame(y_train_smote)[0])).sort_values(ascending=False)
corr_features_to_test_var = corr_features.index.tolist()
corr_features.head()
# loop from 3 to length of features list and add another feature each time
# cross-validate and store in a DataFrame

f1_results = pd.DataFrame(columns=top_models.index, index=np.arange(3,(len(corr_features))), dtype='int')

for n in range(3,len(corr_features_to_test_var)+1):
    X_train_f = pd.DataFrame(X_train_res, columns = exp_features)[corr_features_to_test_var[0:n]]
    
    for model in top_models.index:
        pipe = Pipeline([('sm', SMOTE(random_state=12)), ('classifier', models[model])])
        f1_results.loc[n,str(model)] = cross_val_score(pipe, X_train_f, y_train, cv=5, scoring='f1').mean()
        
print('Model name - best F1 score, Number of features')
print([(x, round(f1_results[x].max(),3), f1_results[x].idxmax()) for x in f1_results.columns])
f1_results.plot();
plt.title('F1 Score');plt.xlabel('Number of features');
plt.plot(f1_results.iloc[:,0].idxmax(), f1_results.iloc[:,0].max(), 'o', markersize=10,c="b", mew=4,)
plt.plot(f1_results.iloc[:,1].idxmax(), f1_results.iloc[:,1].max(), 'o', markersize=10, c="orange", mew=4);
def report(y_true, y_pred):
    print("Accuracy = " , accuracy_score(y_true, y_pred).round(3))
    print("Precision = " ,precision_score(y_true, y_pred).round(3))
    print("Recall = " ,recall_score(y_true, y_pred).round(3))
    print("F1 Score = " ,f1_score(y_true, y_pred).round(3))
    print("Cohen kappa = " ,cohen_kappa_score(y_true, y_pred).round(3))
# best hyperparameters after RandomizedSearchCV
print(b_parms['AdaBoost'])
# changing hyperparamters model, and test on validation and test datasets

pipe_ada = Pipeline([('ct', column_trans), ('sm', SMOTE(random_state=12)),
                     ('classifier', AdaBoostClassifier(n_estimators = 500, learning_rate = 0.1))])
ada_clf = pipe_ada.fit(X_train, y_train)
clf_pred_val = ada_clf.predict(X_val)
clf_pred_test = ada_clf.predict(X_test)
print('Test report on Validation set:')
report(y_val, clf_pred_val)
print('\n')
print('\n\nTest report on Test set:')
report(y_test, clf_pred_test)
cm = confusion_matrix(y_val, clf_pred_val)
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True,
        cmap = 'Blues_r', cbar=False, annot_kws={"fontsize":20});
plt.title('AdaBoost - test set')
plt.ylabel('Actual Churn');
plt.xlabel('Predicted Churn');
plt.tight_layout()   
eli5.explain_weights(pipe_ada.named_steps['classifier'], top=50, feature_names =exp_features)
# best hyperparameters after RandomizedSearchCV
b_parms['LogReg']
# Validation set 
pipe_lr =  Pipeline([('ct', column_trans), ('sm', SMOTE(random_state=12)),
                     ('classifier', LogisticRegression(solver = 'saga', C = 0.5, max_iter=2000))])
lr_clf = pipe_lr.fit(X_train, y_train)
clf_pred_val = lr_clf.predict(X_val)
clf_pred_test = lr_clf.predict(X_test)
print('Test report on Validation set:')
report(y_val, clf_pred_val)
print('\n')
# print('\n\nTest report on Test set:')
# report(y_test, clf_pred_test)
cm = confusion_matrix(y_val, clf_pred_val)
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True,
        cmap = 'Blues_r', cbar=False, annot_kws={"fontsize":20});
plt.title('Logistic Regression - validation set')
plt.ylabel('Actual Churn');
plt.xlabel('Predicted Churn');
plt.tight_layout()   
eli5.explain_weights(pipe_lr.named_steps['classifier'], top=50, feature_names = exp_features)
clf_pred_proba = pipe_lr.predict_proba(X_val)[:,1]
precision, recall, thresholds = precision_recall_curve(y_val, clf_pred_proba)

mark = np.argwhere(thresholds == min(thresholds, key=lambda x:abs(x-0.5)))
plt.figure(figsize=(10,7.5))
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.plot(recall[mark], precision[mark], 'o', markersize=10,label="0.5 threshold", fillstyle="none", c="k", mew=4)
plt.fill_between(recall, precision, alpha=0.2, color='b')
plt.title('Precision-Recall curve. Logistic Regression (val set)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.0])
plt.xlim([0.0, 1.0]); plt.legend(loc=1);
# Logistic Regression Precision, recall and F1 for different threshold values.
precision, recall, thresholds = precision_recall_curve(y_val, clf_pred_proba)
plt.figure(figsize=(11,7.5))
plt.plot(thresholds, recall[1:], label="Recall",linewidth=3)
plt.plot(thresholds, precision[1:], label="Precision",linewidth=3);
plt.plot(thresholds, (2 * (precision[1:] * recall[1:]) / (precision[1:] + recall[1:])), label="f1",linewidth=3);
plt.title('Precision, recall and F1 for different threshold values (val set)');
plt.xlabel('Threshold');plt.ylabel('Proportion')
plt.axvline(x=0.5,color='r', lw=2, linestyle='--', alpha=0.7, label='Classifier threshold 0.5');
plt.legend();
f1 = (2 * (precision[1:] * recall[1:]) / (precision[1:] + recall[1:]))
adjusted_threshold = thresholds[np.argmax(f1)].round(3)
print(f'Currently F1 score: {f1_score(y_val, clf_pred_val).round(3)}')
print(f'Max F1 score: {np.max(f1).round(3)}')
print(f'New adjusted classifier threshold: {adjusted_threshold}')
lr_clf_decision_thr = (pipe_lr.predict_proba(X_test)[:,1] > adjusted_threshold)
cm = confusion_matrix(y_test, lr_clf_decision_thr)
plt.subplots(figsize = (8,6))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True,cmap = 'Blues_r', cbar=False, annot_kws={"fontsize":25});
plt.title(f'Logistic Regression. threshold -> {adjusted_threshold}\n Test Data Results')
plt.ylabel('Actual Churn');plt.xlabel('Predicted Churn');plt.tight_layout();
report(y_test, lr_clf_decision_thr)
# divides a data series into 10 parts, and group based on the predicted churn probability (value between 0.0 and 1.0).
# in ascending order, that first decile contain highest probability score, and then calculate the true churn rate per group.

lift = pd.DataFrame({'Churn':y_test,
                     'Pred':pipe_lr.predict_proba(X_test)[:,1].round(2)})
grouped = lift.groupby(pd.qcut(lift['Pred'],10,labels=False)+1)
lift_df = pd.DataFrame()
lift_df['min_prob'] = grouped.min()['Pred']
lift_df['max_prob'] = grouped.max()['Pred']
lift_df['#customers'] = grouped.size()
lift_df['churn'] = grouped.sum()['Churn']
lift_df['%d_churn_rate'] = round(grouped.sum()['Churn'] / grouped.size(),2)
lift_df['%g_churn_rate'] = round(lift_df['churn']/y_test.sum(),2)#.apply('{:.0%}'.format)
lift_df['%base_rate'] = round(y_test.mean(),3)
lift_df['lift'] = round(lift_df['%d_churn_rate'] / lift_df['%base_rate'],2)
lift_df = lift_df.sort_values(by="min_prob", ascending=False)
lift_df['d_churn_rate'] = lift_df['%d_churn_rate']#.apply('{:.0%}'.format)
lift_df.index = range(1,11)
lift_df.index.rename('Decile', inplace=True)
fig, ax = plt.subplots(figsize=(11,8))
plt.bar(lift_df.index,lift_df['%d_churn_rate']);
plt.xlabel('Customers'); plt.ylabel('Churn Rate');plt.xticks(lift_df.index)
ax.set_ylabel('Churn Rate', fontsize=16,size=20, rotation=360, labelpad=35);
ax.yaxis.set_label_coords(-0.05,1.04)
ax.grid(False)
plt.axhline(lift_df['%base_rate'].mean(), color='r', linestyle='--', lw=2.5);
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()]);
ax2 = ax.twinx()
plt.plot(lift_df.index,lift_df['lift'], lw=3.5, color='navy', marker='.', markersize=20)
plt.margins(0,0);ax2.grid(False); plt.ylim(0,3);plt.ylabel('Lift')
plt.gca().set_xticklabels(['{:.0f}%'.format(x*10) for x in plt.gca().get_xticks()]);
plt.yticks(color='navy')
ax2.set_ylabel('Lift', fontsize=16, color='navy', size=20, rotation=360, labelpad=35);
ax2.yaxis.set_label_coords(+1.05,1.1)
lift_df