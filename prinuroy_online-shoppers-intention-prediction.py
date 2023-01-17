import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency,ttest_ind
from statsmodels.stats.proportion import proportions_ztest

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,cross_val_score
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.metrics import classification_report,plot_roc_curve

from sklearn.ensemble import AdaBoostClassifier
print(os.listdir('../input'))
df = pd.read_csv('/kaggle/input/online-shoppers-intention/online_shoppers_intention.csv')
df.head()
print('Rows:{}'.format(df.shape[0]))
print('Columns:{}'.format(df.shape[1]))
df.info()
df.isna().sum()
df.dropna(inplace=True)
num = ['Administrative', 'Administrative_Duration', 'Informational',
       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
       'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
df[num].describe()
# Distribution of customers on Revenue
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
df['Revenue'].value_counts().plot(kind='bar',color='blue',alpha=0.8)
#plt.title('Revenue Analysis')
plt.xlabel('True/False')
plt.ylabel('Number of counts');

plt.subplot(1,2,2)
size = df['Revenue'].value_counts().values
labels = df['Revenue'].value_counts().index
colors = ['green', 'red']
explode = [0, 0.2]
plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%',)
plt.suptitle('Revenue Analysis');
# Distribution of customers on Weekends
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
df['Weekend'].value_counts().plot(kind='bar',color='blue',alpha=0.8)
#plt.title('Weekend Analysis')
plt.xlabel('True/False')
plt.ylabel('Number of counts');

plt.subplot(1,2,2)
size = df['Weekend'].value_counts().values
labels = df['Weekend'].value_counts().index
colors = ['green', 'red']
explode = [0, 0.2]
plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%',)
plt.suptitle('Weekend Analysis');
# Distribution of customers based on VisitorType
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
df['VisitorType'].value_counts().plot(kind='bar',color='blue',alpha=0.8)
#plt.title('Weekend Analysis')
plt.xlabel('True/False')
plt.ylabel('Number of counts');

plt.subplot(1,2,2)
size = df['VisitorType'].value_counts().values
labels = df['VisitorType'].value_counts().index
colors = ['green','red','blue']
explode = [0,0.2,0]
plt.pie(size, colors = colors, labels = labels, explode = explode, shadow = True, autopct = '%.2f%%',)
plt.suptitle('VisitorType Analysis');
# Distribution of customers based on Traffic Type
sns.countplot(df['TrafficType'],palette='coolwarm')
plt.title('Different Sources')
plt.xlabel('Traffic Type')
plt.ylabel('count');
# Distribution of customers based on Region
sns.countplot(df['Region'],palette='husl')
plt.title('Different Region')
plt.xlabel('Region')
plt.ylabel('count');
#Distribution of customers based on Browser
sns.countplot(df['Browser'],palette='pastel')
plt.title('Browser Used')
plt.xlabel('Browser')
plt.ylabel('count');
#Distribution of customers based on Operating Systems
sns.countplot(df['OperatingSystems'],palette='inferno')
plt.title('Operating Systems Used')
plt.xlabel('Operating Systems')
plt.ylabel('count');
#Distribution of customers on Months
sns.countplot(df['Month'],palette='inferno')
plt.title('Customers on Different months')
plt.xlabel('Months')
plt.ylabel('count')
plt.show()
# Analysis on duration spent in different pages
plt.figure(figsize=(10,5))

plt.subplot(3,1,1)
sns.boxplot(df['Administrative_Duration'],palette='dark')
plt.title('Duration (in secs) in Administrative Pages')
plt.show()

plt.subplot(3,1,2)
sns.boxplot(df['Informational_Duration'],palette='dark')
plt.title('Duration (in secs) in Informational Pages')
plt.show()

plt.subplot(3,1,3)
sns.boxplot(df['ProductRelated_Duration'],palette='dark')
plt.title('Duration (in secs) in Product Pages')
plt.show()
# Analysis on various factors like bounce rates,exit rates etc
plt.figure(figsize=(20,10))

plt.subplot(2,2,1)
sns.distplot(df['BounceRates'],color='blue')
plt.title('Distribution of Bounce Rates');

plt.subplot(2,2,2)
sns.distplot(df['ExitRates'],color='red')
plt.title('Distribution of Exit Rates');

plt.subplot(2,2,3)
sns.distplot(df['PageValues'],color='green',kde=False)
plt.title('Distribution of Page Values');

plt.subplot(2,2,4)
sns.distplot(df['SpecialDay'],color='blue',kde=False)
plt.title('Distribution of Special Day');
ct = pd.crosstab(df['Revenue'],df['Weekend'])
ct.plot(kind='bar',stacked=True,colormap='jet')
plt.title('Revenue vs Weekend');
ct = pd.crosstab(df['Revenue'],df['Weekend'])
ct
print('{:.2f}% making transactions during weekends'.format((499/2868)*100))
print('{:.2f}% making transactions during week days'.format((1409/9462)*100))
ct = pd.crosstab(df['Revenue'],df['VisitorType'])
ct.plot(kind='bar',stacked=True,colormap='jet')
plt.title('Revenue vs VisitorType');
ct = pd.crosstab(df['Revenue'],df['VisitorType'])
ct
print('{:.2f}% of new visitors are making transactions'.format((422/1694)*100))
print('{:.2f}% of returning visitors are making transactions'.format((1470/10551)*100))
ct=pd.crosstab(df['TrafficType'],df['Revenue'])
ct.plot(kind='bar',stacked=True,colormap='jet')
plt.title('Revenue vs TrafficType');
ct=pd.crosstab(df['Region'],df['Revenue'])
ct.plot(kind='bar',stacked=True,colormap='jet')
plt.title('Revenue vs Region');
ct=pd.crosstab(df['Browser'],df['Revenue'])
ct.plot(kind='bar',stacked=True,colormap='jet')
plt.title('Revenue vs Browser');
ct=pd.crosstab(df['OperatingSystems'],df['Revenue'])
ct.plot(kind='bar',stacked=True,colormap='jet')
plt.title('Revenue vs Operating Systems');
ct=pd.crosstab(df['Month'],df['Revenue'])
ct.plot(kind='bar',stacked=True,colormap='jet')
plt.title('Revenue vs Month');
sns.stripplot(df['Revenue'],df['BounceRates'])
plt.show()
sns.stripplot(df['Revenue'],df['ExitRates'])
plt.show()
sns.stripplot(df['Revenue'],df['PageValues'])
plt.show()
sns.violinplot(df['Revenue'],df['SpecialDay'])
plt.show()
sns.boxplot(df['Revenue'],df['Administrative'])
plt.show()
sns.boxplot(df['Revenue'],df['Administrative_Duration'])
plt.show()
sns.boxplot(df['Revenue'],df['Informational'])
plt.show()
sns.boxplot(df['Revenue'],df['Informational_Duration'])
plt.show()
sns.boxplot(df['Revenue'],df['ProductRelated'])
plt.show()
sns.boxplot(df['Revenue'],df['ProductRelated_Duration'])
plt.show()
df1 = df.copy()
df['Revenue'] = df['Revenue'].astype('int64')
df['Weekend'] = df['Weekend'].astype('int64')
df = pd.get_dummies(df,drop_first=True)
df.head()
df.info()
y = df['Revenue']
X = df.drop(['Revenue'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
models = {'Decision Tree':DecisionTreeClassifier(random_state=42),
        'Random Forest':RandomForestClassifier(n_estimators=50,random_state=42),
        'K Nearest Neighbors':KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes':GaussianNB()
        }
for name, algo in models.items():
    model = algo
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(name)
    print('Train Score :',model.score(X_train, y_train)*100)
    print('Test Score :',accuracy_score(y_test, y_pred)*100)
    print('Precision :',precision_score(y_test,y_pred,average='micro')*100)
    print('Recall :',recall_score(y_test,y_pred,average='micro')*100)
    print('F1 Score :',f1_score(y_test,y_pred)*100)
    print('AUC Score :',roc_auc_score(y_test,y_pred)*100)
    print('Confusion matrix: ')
    print(confusion_matrix(y_test,y_pred))
    print('\n')
y=df['Revenue']
X=df.drop(['Revenue'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
sm=SMOTE(random_state=42)
X_train, y_train = sm.fit_sample(X_train, y_train)
for name, algo in models.items():
    model = algo
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(name)
    print('Train Score :',model.score(X_train, y_train)*100)
    print('Test Score :',accuracy_score(y_test, y_pred)*100)
    print('Precision :',precision_score(y_test,y_pred,average='micro')*100)
    print('Recall :',recall_score(y_test,y_pred,average='micro')*100)
    print('F1 Score :',f1_score(y_test,y_pred)*100)
    print('AUC Score :',roc_auc_score(y_test,y_pred)*100)
    print('Confusion matrix: ')
    print(confusion_matrix(y_test,y_pred))
    print('\n')
y=df['Revenue']
X=df.drop(['Revenue'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

sm=SMOTE(random_state=42)
X_train, y_train = sm.fit_sample(X_train, y_train)
for name, algo in models.items():
    model = algo
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(name)
    print('Train Score :',model.score(X_train, y_train)*100)
    print('Test Score :',accuracy_score(y_test, y_pred)*100)
    print('Precision :',precision_score(y_test,y_pred,average='micro')*100)
    print('Recall :',recall_score(y_test,y_pred,average='micro')*100)
    print('F1 Score :',f1_score(y_test,y_pred)*100)
    print('AUC Score :',roc_auc_score(y_test,y_pred)*100)
    print('Confusion matrix: ')
    print(confusion_matrix(y_test,y_pred))
    print('\n')
# Filter based Method : Multi-collinearity
mask = np.zeros_like(df.corr(),dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
fig,ax= plt.subplots()
fig.set_size_inches(20,15)
sns.heatmap(df.corr(),
           annot=True,
           mask = mask,
           cmap = 'RdBu_r',
           linewidths=0.1,
           linecolor='white',
           vmax = .9,
           square=True)
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20)
plt.show()
ct=pd.crosstab(df1['Revenue'],df1['Month'])
chi_value,p_value,dof,ec=chi2_contingency(ct)
print('p-value of Month :',p_value)
ct=pd.crosstab(df1['Revenue'],df1['OperatingSystems'])
chi_value,p_value,dof,ec=chi2_contingency(ct)
print('p-value of Month :',p_value)
ct=pd.crosstab(df1['Revenue'],df1['Browser'])
chi_value,p_value,dof,ec=chi2_contingency(ct)
print('p-value of Month :',p_value)
ct=pd.crosstab(df1['Revenue'],df1['Region'])
chi_value,p_value,dof,ec=chi2_contingency(ct)
print('p-value of Month :',p_value)
ct=pd.crosstab(df1['Revenue'],df1['TrafficType'])
chi_value,p_value,dof,ec=chi2_contingency(ct)
print('p-value of Month :',p_value)
ct=pd.crosstab(df1['Revenue'],df1['Weekend'])
print(ct)
count=np.array([1409,499])
obs=np.array([9462,2868])
zstat,p_value=proportions_ztest(count,obs) 
print('p-value of Weekend :',p_value)
new_data=df1.groupby('Revenue')
new_data_rev=new_data.get_group(1)
new_data_no_rev=new_data.get_group(0)
tstat,p_value=ttest_ind(new_data_rev['Administrative'],new_data_no_rev['Administrative'])
print('p-value of Administartive :',p_value)
tstat,p_value=ttest_ind(new_data_rev['Administrative_Duration'],new_data_no_rev['Administrative_Duration'])
print('p-value of Administartive_Duration :',p_value)
tstat,p_value=ttest_ind(new_data_rev['Informational'],new_data_no_rev['Informational'])
print('p-value of Informational :',p_value)
tstat,p_value=ttest_ind(new_data_rev['Informational_Duration'],new_data_no_rev['Informational_Duration'])
print('p-value of Informational_Duration :',p_value)
tstat,p_value=ttest_ind(new_data_rev['ProductRelated'],new_data_no_rev['ProductRelated'])
print('p-value of ProductRelated :',p_value)
tstat,p_value=ttest_ind(new_data_rev['ProductRelated_Duration'],new_data_no_rev['ProductRelated_Duration'])
print('p-value of Informational_Duration :',p_value)
tstat,p_value=ttest_ind(new_data_rev['BounceRates'],new_data_no_rev['BounceRates'])
print('p-value of BounceRates :',p_value)
tstat,p_value=ttest_ind(new_data_rev['ExitRates'],new_data_no_rev['ExitRates'])
print('p-value of ExitRates :',p_value)
tstat,p_value=ttest_ind(new_data_rev['PageValues'],new_data_no_rev['PageValues'])
print('p-value of PageValues :',p_value)
tstat,p_value=ttest_ind(new_data_rev['SpecialDay'],new_data_no_rev['SpecialDay'])
print('p-value of SpecialDay :',p_value)
y=df['Revenue']
X=df.drop(['Revenue','Region','Administrative','Informational','ProductRelated','BounceRates'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

sm=SMOTE(random_state=42)
X_train, y_train = sm.fit_sample(X_train, y_train)
for name, algo in models.items():
    model = algo
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(name)
    print('Train Score :',model.score(X_train, y_train)*100)
    print('Test Score :',accuracy_score(y_test, y_pred)*100)
    print('Precision :',precision_score(y_test,y_pred,average='micro')*100)
    print('Recall :',recall_score(y_test,y_pred,average='micro')*100)
    print('F1 Score :',f1_score(y_test,y_pred)*100)
    print('AUC Score :',roc_auc_score(y_test,y_pred)*100)
    print('Confusion matrix: ')
    print(confusion_matrix(y_test,y_pred))
    print('\n')
# param-grid

# dt-grid
dt_grid = {'criterion':['gini','entropy'],
           'max_features':['auto','sqrt','log2'],
           'splitter':['best','random'],
           'max_depth':[5,10,15]}

# rf grid
rf_grid = {'criterion':['gini','entropy'],
           'n_estimators':np.arange(10, 500, 50),
           'max_features':['auto','sqrt','log2'],
           'max_depth':[None,5,8,10],
           'min_samples_split': np.arange(2, 20, 2),
           'min_samples_leaf': np.arange(1, 20, 2)}

# KNN Grid
knn_grid = {'n_neighbors':np.arange(1,21)}
# Randomized Search CV-DT
rs_dt = RandomizedSearchCV(DecisionTreeClassifier(random_state=42),
                   param_distributions=dt_grid,
                   cv=5,
                   n_iter=20,
                   verbose=2)

# Fitting model
rs_dt.fit(X_train,y_train);
rs_dt.best_params_
y_pred = rs_dt.predict(X_test)
print('Randomized Search CV Decision Tree')
print(f'Train Score :{rs_dt.score(X_train,y_train)*100:.2f}%')
print(f'Test Score :{rs_dt.score(X_test,y_test)*100:.2f}%')
print(f'AUC Score :{roc_auc_score(y_test,y_pred)*100:.2f}%')
# Randomized Search CV-RF
rs_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                   param_distributions=rf_grid,
                   cv=5,
                   n_iter=20,
                   verbose=2)

# Fitting model
rs_rf.fit(X_train,y_train);
rs_rf.best_params_
y_pred = rs_rf.predict(X_test)
print('Randomized Search CV Random Forest')
print(f'Train Score :{rs_rf.score(X_train,y_train)*100:.2f}%')
print(f'Test Score :{rs_rf.score(X_test,y_test)*100:.2f}%')
print(f'AUC Score :{roc_auc_score(y_test,y_pred)*100:.2f}%')
# Randomized Search CV-KNN
rs_knn = RandomizedSearchCV(KNeighborsClassifier(),
                   param_distributions=knn_grid,
                   cv=5,
                   n_iter=20,
                   verbose=2)

# Fitting model
rs_knn.fit(X_train,y_train);
rs_knn.best_params_
y_pred = rs_knn.predict(X_test)
print('Randomized Search CV KNN')
print(f'Train Score :{rs_knn.score(X_train,y_train)*100:.2f}%')
print(f'Test Score :{rs_knn.score(X_test,y_test)*100:.2f}%')
print(f'AUC Score :{roc_auc_score(y_test,y_pred)*100:.2f}%')
model = AdaBoostClassifier(
    RandomForestClassifier(n_estimators=360,
                           min_samples_split=16,
                           min_samples_leaf=1,
                           max_features='sqrt',
                           max_depth=None,
                           criterion='entropy',
                           random_state=42),
    n_estimators=200
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Train Score :',model.score(X_train, y_train)*100)
print('Test Score :',accuracy_score(y_test, y_pred)*100)
print('Precision :',precision_score(y_test,y_pred,average='micro')*100)
print('Recall :',recall_score(y_test,y_pred,average='micro')*100)
print('F1 Score :',f1_score(y_test,y_pred)*100)
print('AUC Score :',roc_auc_score(y_test,y_pred)*100)
print('Confusion matrix: ',confusion_matrix(y_test,y_pred))
y_pred = rs_rf.predict(X_test)
plot_roc_curve(rs_rf,X_test,y_test);
# Classification report
print(classification_report(y_test,y_pred))
# Confusion Matrix
sns.heatmap(confusion_matrix(y_test,y_pred),
            annot=True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values');
confusion_matrix(y_test,y_pred)
# best params
rs_rf.best_params_
# New model with best params
model = RandomForestClassifier(n_estimators=360,
                               min_samples_split=16,
                               min_samples_leaf=1,
                               max_features='sqrt',
                               max_depth=None,
                               criterion='entropy',
                               random_state=42)
model.fit(X_train,y_train)
feature_names = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_names).sort_values(ascending=False)
feature_imp
sns.barplot(feature_imp,feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('Importance of Features')
plt.show()
