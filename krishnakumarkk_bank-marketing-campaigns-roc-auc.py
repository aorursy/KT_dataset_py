# Data Manipulation Libraries

import os

import pandas as pd

import numpy as np



# Vizualization Libraries

import plotly.express as px

import seaborn as sns

import matplotlib.pyplot as plt



# pre-processing

from sklearn.preprocessing import StandardScaler



# ML model Libraries

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier



import warnings

warnings.filterwarnings('ignore')

bank_data = pd.read_csv('../input/bank-marketing-campaigns-dataset/bank-additional-full.csv',sep=';',engine='python')
# Pandas Profiling - Enable if needed

# from pandas_profiling import ProfileReport

# profile = ProfileReport(df, title="Pandas Profiling Report",explorative=True)

# profile.to_notebook_iframe()

# profile.to_file('initial_report.html')
bank_data_with_unknowns = bank_data.copy()

#Drop Duplicate rows

bank_data_with_unknowns.drop_duplicates(subset=None, keep='first', inplace=True)

bank_data_with_unknowns.head()
bank_data_with_unknowns['pdays'].value_counts()

bank_pdays = bank_data_with_unknowns.loc[bank_data_with_unknowns['pdays'] == 999,'y']

bank_pdays.value_counts()

bank_not_pdays = bank_data_with_unknowns.loc[bank_data_with_unknowns['pdays'] != 999,['y','pdays']]

#sns.violinplot(x='y',y='pdays',data=bank_not_pdays)

sns.swarmplot(x='y',y='pdays',data=bank_not_pdays)
#dropping column pdays

bank_data_with_unknowns.drop(columns=['pdays'],inplace=True)
#Checking if we need the column 'previous'

bank_data_with_unknowns['previous'].value_counts()

#zero contributes to 86.3% of the data.

plt.figure(figsize=(15,5))

plt.subplot(121)

bank_not_previous =  bank_data_with_unknowns.loc[bank_data_with_unknowns['previous'] != 0,['y','previous']]

sns.violinplot(x='y',y='previous',data=bank_not_previous)

plt.subplot(122)

bank_previous =  bank_data_with_unknowns.loc[bank_data_with_unknowns['previous'] == 0,['y','previous']]

sns.countplot(x='y',data=bank_previous)
#Percentage of yes and no

bnp = bank_not_previous['y'].value_counts()

print(bnp['yes']/bnp['no'])

bp = bank_previous['y'].value_counts()

print(bp['yes']/bp['no'])
#Checking if we need the column 'poutcome'

bank_data_with_unknowns['poutcome'].value_counts()

#zero contributes to 86.3% of the data.

plt.figure(figsize=(15,5))

plt.subplot(121)

bank_not_nonexistent =  bank_data_with_unknowns.loc[bank_data_with_unknowns['poutcome'] != 'nonexistent',['y']]

sns.countplot(x='y',data=bank_not_nonexistent)

plt.subplot(122)

bank_nonexistent =  bank_data_with_unknowns.loc[bank_data_with_unknowns['poutcome'] == 'nonexistent',['y']]

sns.countplot(x='y',data=bank_nonexistent)

#Percentage of yes and no

bnne = bank_not_nonexistent['y'].value_counts()

print(bnne['yes']/bnne['no'])

bne = bank_nonexistent['y'].value_counts()

print(bne['yes']/bne['no'])
#handling job unknown values

bank_data_with_unknowns['job'].value_counts()

bank_data_with_unknowns['job'] = bank_data_with_unknowns['job'].str.replace('.','')

bank_data_with_unknowns['job'] = bank_data_with_unknowns['job'].str.replace('-','')

bank_data_with_unknowns.loc[bank_data_with_unknowns['job'] == 'unknown','job'] = 'admin'
#Handling marital unknown values

bank_data_with_unknowns['marital'].value_counts()

bank_data_with_unknowns.loc[bank_data_with_unknowns['marital'] == 'unknown','marital'] = 'married'
#handling education

bank_data_with_unknowns['education'].value_counts()

bank_data_with_unknowns.loc[bank_data_with_unknowns['education'] == 'basic.9y','education'] = 'basic'

bank_data_with_unknowns.loc[bank_data_with_unknowns['education'] == 'basic.6y','education'] = 'basic'

bank_data_with_unknowns.loc[bank_data_with_unknowns['education'] == 'basic.4y','education'] = 'basic'

bank_data_with_unknowns['education'] = bank_data_with_unknowns['education'].str.replace('.','')

bank_data_with_unknowns['education'].value_counts()



#converting the unknown values to basic instead of university degree

bank_data_with_unknowns.loc[bank_data_with_unknowns['education'] == 'unknown','education'] = 'basic'
#Dropping default column as it dosent contribute to any useful info

bank_data_with_unknowns.drop(columns=['default'],inplace=True)
#handling unknows in housing as yes as it has the highest frequency

bank_data_with_unknowns.loc[bank_data_with_unknowns['housing'] == 'unknown','housing'] = 'yes'
#handling unknows in loan as no as it has the highest frequency

bank_data_with_unknowns.loc[bank_data_with_unknowns['loan'] == 'unknown','loan'] = 'no'
#Contribution of contact

plt.figure(figsize=(15,5))

plt.subplot(121)

bank_telephone =  bank_data_with_unknowns.loc[bank_data_with_unknowns['contact'] == 'telephone',['y']]

sns.countplot(x='y',data=bank_telephone)

plt.subplot(122)

bank_cellular =  bank_data_with_unknowns.loc[bank_data_with_unknowns['contact'] == 'cellular',['y']]

sns.countplot(x='y',data=bank_cellular)
bt = bank_telephone['y'].value_counts()

bc = bank_cellular['y'].value_counts()

print(bt['yes']/bt['no'],bc['yes']/bc['no'])
#the responses 'yes' & 'no' are almost similarly distributed so dropping this column

bank_data_with_unknowns.drop(columns=['contact'],inplace=True)
#handling duration -  this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known.

#dropped rows where duration was zero (4 rows)

bank_data_with_unknowns = bank_data_with_unknowns[~(bank_data_with_unknowns['duration']==0)] 
#clearly it is a contributing factor - so lets deal with the outliers

px.box(bank_data_with_unknowns,x='y',y='duration')

#duration can be classified as greater_than_1600, greater_than_800, greater_than_400 and less_than_400 - data needs to be capped, we will do that later
#Handling campaign

px.violin(bank_data_with_unknowns,x='y',y='campaign')

#Distribution is significantly different so we will keep this variable - values greater than 10 could be capped
#we have handled all unknowns so

bank_eda_data = bank_data_with_unknowns.copy() 

job_vs_duration = px.box(bank_eda_data, x="duration", y="job", color="y",notched=True,template='simple_white',color_discrete_sequence=px.colors.qualitative.Pastel)

job_vs_duration.update_traces(quartilemethod="exclusive",orientation='h') #Quantile at 2.5 and 7.5

job_vs_duration.show()
#Campaign vs duration calls

campaign_vs_duration = px.scatter(bank_eda_data, x="campaign", y="duration",color='y',template='simple_white',color_discrete_sequence=px.colors.qualitative.Pastel)

campaign_vs_duration.show()
#campaign vs month

campaign_vs_month = bank_eda_data.copy()

sort_order = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

campaign_vs_month.month = campaign_vs_month.month.astype("category")

campaign_vs_month.month.cat.set_categories(sort_order, inplace=True)

campaign_vs_month = campaign_vs_month.sort_values(['month'])

plt.bar(campaign_vs_month['month'],campaign_vs_month['campaign'])

plt.show()
#Yes and No vs (Job, Month, Marital status, Education, Day of week, Housing, Contact, default)



plt.figure(figsize = (15, 30))

sx = plt.subplot(5,2,1)

sns.countplot(x="job",hue="y", data=bank_eda_data, palette="Set2")

sx.set_xticklabels(sx.get_xticklabels(),rotation=45)

sx = plt.subplot(5,2,2)



bde_copy = bank_eda_data.copy()

bde_copy.month = bde_copy.month.astype("category")

bde_copy.month.cat.set_categories(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], inplace=True)



sns.countplot(x="month",hue="y", data=bde_copy.sort_values(['month']), palette="Set2")

sx = plt.subplot(5,2,3)

sns.countplot(x="marital",hue="y", data=bde_copy, palette="Set2")

sx = plt.subplot(5,2,4)

sns.countplot(x="education",hue="y", data=bde_copy, palette="Set2")

sx.set_xticklabels(sx.get_xticklabels(),rotation=45)

sx = plt.subplot(5,2,5)

sns.countplot(x="housing",hue="y", data=bde_copy, palette="Set2")

sx = plt.subplot(5,2,6)

sns.countplot(x="loan",hue="y", data=bde_copy, palette="Set2")

sx = plt.subplot(5,2,7)

sns.countplot(x="day_of_week",hue="y", data=bde_copy, palette="Set2")
# Only - Yes vs (Job, Month, Marital status, Education, Day of week, Housing)



plt.figure(figsize = (15, 30))

bank_only_yes = bank_eda_data.copy()

bank_only_yes = bank_only_yes[bank_only_yes['y']=='yes']

sx2 = plt.subplot(5,2,1)

sns.countplot(x="job",hue="y", data=bank_only_yes, palette="Set2")

sx2.set_xticklabels(sx2.get_xticklabels(),rotation=45)

sx2 = plt.subplot(5,2,2)



bde_only_yes_copy = bank_only_yes.copy()

bde_only_yes_copy.month = bde_copy.month.astype("category")

bde_only_yes_copy.month.cat.set_categories(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], inplace=True)



sns.countplot(x="month",hue="y", data=bde_only_yes_copy.sort_values(['month']), palette="Set2")

sx2 = plt.subplot(5,2,3)

sns.countplot(x="marital",hue="y", data=bank_only_yes, palette="Set2")

sx2 = plt.subplot(5,2,4)

sns.countplot(x="education",hue="y", data=bank_only_yes, palette="Set2")

sx2.set_xticklabels(sx2.get_xticklabels(),rotation=45)

sx2 = plt.subplot(5,2,5)

sns.countplot(x="housing",hue="y", data=bank_only_yes, palette="Set2")

sx2 = plt.subplot(5,2,6)

sns.countplot(x="loan",hue="y", data=bank_only_yes, palette="Set2")

sx2 = plt.subplot(5,2,7)

sns.countplot(x="day_of_week",hue="y", data=bank_only_yes, palette="Set2")
fig, ax = plt.subplots()

# the size of A4 paper

fig.set_size_inches(14.7, 5.27)

sns.boxplot(x='cons.price.idx',y='month',data=bde_copy,ax=ax)
sns.jointplot(x ='emp.var.rate', y ='cons.price.idx', data = bank_eda_data) 

sns.jointplot(x ='emp.var.rate', y ='euribor3m', data = bank_eda_data) 

sns.jointplot(x ='emp.var.rate', y ='nr.employed', data = bank_eda_data) 

sns.jointplot(x ='euribor3m', y ='nr.employed', data = bank_eda_data, kind ='kde') 

sns.jointplot(x ='cons.price.idx', y ='nr.employed', data = bank_eda_data, kind ='kde') 
#Checking for outliers in data

plt.figure(figsize = (15, 5))

ax=plt.subplot(121)

plt.boxplot(bank_eda_data['duration'])

ax.set_title('duration')

ax=plt.subplot(122)

plt.boxplot(bank_eda_data['campaign'])

ax.set_title('campaign')

# We choose not to perform any outlier filtering as the values observed are not so extreme. Although its optional if you see fit.

# #We can directly apply interquantile range filter for duration and campaign.

# numerical_features=['campaign','duration']

# for cols in numerical_features:

#     Q3 = bank_eda_data[cols].quantile(0.95)

#     Q1 = bank_eda_data[cols].quantile(0.05)

#     IQR = Q3 - Q1

#     filter = (bank_eda_data[cols] <= (Q3 + 1.5 *IQR))

#     bank_eda_data=bank_eda_data.loc[filter]



# #replotting after applying filter

# plt.figure(figsize = (5, 5))



# ax3=plt.subplot(121)

# plt.boxplot(bank_eda_data['duration'])

# ax3.set_title('duration')



# ax3=plt.subplot(122)

# plt.boxplot(bank_eda_data['campaign'])

# ax3.set_title('campaign')
bank_eda_data['previous'].value_counts()
bank_eda_data.dtypes
bank_eda_data.head()
bank_preprocess = bank_eda_data.copy()
bank_preprocess.dtypes
category_features = ['job','marital','education','housing','loan','month','day_of_week','poutcome']

bank_preprocess[category_features].head()
bank_preprocess_one_hot = pd.get_dummies(bank_preprocess, columns = category_features)

bank_preprocess_one_hot = pd.get_dummies(bank_preprocess_one_hot, columns = ['y'],drop_first=True)

bank_preprocess_one_hot.head()
bank_one_hot_data = bank_preprocess_one_hot.copy()

X = bank_one_hot_data.drop(columns=['y_yes'])

y = bank_one_hot_data['y_yes']

ssc = StandardScaler(with_mean=True,with_std=True)

ssc.fit_transform(X)
X.columns
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

log = LogisticRegression()

log.fit(x_train,y_train)

y_pred = log.predict(x_test)

print('Accuracy Score : %f' % (accuracy_score(y_pred, y_test)))

parameters = {'C':[0.001, 0.1, 1, 10, 100]}

log_gsmodel = GridSearchCV(estimator=log, param_grid = parameters)

log_gsmodel_result = log_gsmodel.fit(x_train,y_train)

print("Best: %f using %s" % (log_gsmodel_result.best_score_, log_gsmodel_result.best_params_))

print('mean_test_score : %s' % (log_gsmodel_result.cv_results_['mean_test_score']))

print('std_test_Score : %s' % (log_gsmodel_result.cv_results_['std_test_score']))

print('params: %s' %(log_gsmodel_result.cv_results_['params']))
ds_tree_entropy = DecisionTreeClassifier(criterion='entropy',random_state=42)

ds_tree_entropy.fit(x_train,y_train)

ds_tree_entropy_y_pred = ds_tree_entropy.predict(x_test)

ds_tree_entropy_y_pred
print("Model Entropy - no max depth")

print("Accuracy:", metrics.accuracy_score(y_test,ds_tree_entropy_y_pred))

print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test,ds_tree_entropy_y_pred))

print('Precision score for "Yes"' , metrics.precision_score(y_test,ds_tree_entropy_y_pred, pos_label = 1))

print('Precision score for "No"' , metrics.precision_score(y_test,ds_tree_entropy_y_pred, pos_label = 0))

print('Recall score for "Yes"' , metrics.recall_score(y_test,ds_tree_entropy_y_pred, pos_label = 1))

print('Recall score for "No"' , metrics.recall_score(y_test,ds_tree_entropy_y_pred, pos_label = 0))
entr_parameters = {'max_depth':[2,3,4,5,6,7]}

dt_entr_gsmodel = GridSearchCV(estimator=ds_tree_entropy, param_grid = entr_parameters)

dt_entr_gsmodel_result = dt_entr_gsmodel.fit(x_train,y_train)

print("Best: %f using %s" % (dt_entr_gsmodel_result.best_score_, dt_entr_gsmodel_result.best_params_))

print('mean_test_score : %s' % (dt_entr_gsmodel_result.cv_results_['mean_test_score']))

print('std_test_Score : %s' % (dt_entr_gsmodel_result.cv_results_['std_test_score']))

print('params: %s' %(dt_entr_gsmodel_result.cv_results_['params']))
ds_tree_gini = DecisionTreeClassifier(criterion='gini',random_state=42)

ds_tree_gini.fit(x_train,y_train)

ds_tree_gini_y_pred = ds_tree_gini.predict(x_test)

print("Model Entropy - no max depth")

print("Accuracy:", metrics.accuracy_score(y_test,ds_tree_gini_y_pred))

print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test,ds_tree_gini_y_pred))

print('Precision score for "Yes"' , metrics.precision_score(y_test,ds_tree_gini_y_pred, pos_label = 1))

print('Precision score for "No"' , metrics.precision_score(y_test,ds_tree_gini_y_pred, pos_label = 0))

print('Recall score for "Yes"' , metrics.recall_score(y_test,ds_tree_gini_y_pred, pos_label = 1))

print('Recall score for "No"' , metrics.recall_score(y_test,ds_tree_gini_y_pred, pos_label = 0))
gini_parameters = {'max_depth':[2,3,4,5,6,7]}

dt_gini_gsmodel = GridSearchCV(estimator=ds_tree_gini, param_grid = gini_parameters)

dt_gini_gsmodel_result = dt_gini_gsmodel.fit(x_train,y_train)

print("Best: %f using %s" % (dt_gini_gsmodel_result.best_score_, dt_entr_gsmodel_result.best_params_))

print('mean_test_score : %s' % (dt_gini_gsmodel_result.cv_results_['mean_test_score']))

print('std_test_Score : %s' % (dt_gini_gsmodel_result.cv_results_['std_test_score']))

print('params: %s' %(dt_gini_gsmodel_result.cv_results_['params']))
nb = GaussianNB()

nb_result = nb.fit(x_train,y_train)

nb_y_pred = nb.predict(x_test)

print(accuracy_score(nb_y_pred,y_test))


sgd = SGDClassifier(loss='modified_huber',shuffle=True,random_state=42)

sgd.fit(x_train,y_train)

sgd_y_pred = sgd.predict(x_test)

print(accuracy_score(sgd_y_pred,y_test))
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)

knn_y_pred = knn.predict(x_test)

print(accuracy_score(knn_y_pred,y_test))
knn_parameters = {'n_neighbors': range(10,20)}

knn_gsmodel = GridSearchCV(estimator=KNeighborsClassifier(), param_grid = knn_parameters)

knn_gsmodel_result = knn_gsmodel.fit(x_train,y_train)

print("Best: %f using %s" % (knn_gsmodel_result.best_score_, knn_gsmodel_result.best_params_))

print('mean_test_score : %s' % (knn_gsmodel_result.cv_results_['mean_test_score']))

print('std_test_Score : %s' % (knn_gsmodel_result.cv_results_['std_test_score']))

print('params: %s' %(knn_gsmodel_result.cv_results_['params']))


rfm = RandomForestClassifier(n_estimators=20,oob_score=True,n_jobs=1,random_state=42,max_features=None,min_samples_leaf=10)

rfm.fit(x_train,y_train)

rfm_y_pred = rfm.predict(x_test)

print(accuracy_score(rfm_y_pred,y_test))
rfm_parameters = {'n_estimators': [80,90,100]}

rfm_gsmodel = GridSearchCV(estimator=RandomForestClassifier(oob_score=True,n_jobs=1,random_state=42,max_features=None,min_samples_leaf=10), param_grid = rfm_parameters)

rfm_gsmodel_result = rfm_gsmodel.fit(x_train,y_train)

print("Best: %f using %s" % (rfm_gsmodel_result.best_score_, rfm_gsmodel_result.best_params_))

print('mean_test_score : %s' % (rfm_gsmodel_result.cv_results_['mean_test_score']))

print('std_test_Score : %s' % (rfm_gsmodel_result.cv_results_['std_test_score']))

print('params: %s' %(rfm_gsmodel_result.cv_results_['params']))
rfm_gsmodel_result.best_estimator_
vote_classify = VotingClassifier(estimators=[

    ('log_be', log_gsmodel_result.best_estimator_), # Logistic Regression

    ('dt_entr_be', dt_entr_gsmodel_result.best_estimator_), #Decision tree entropy

    ('dt_gini_be', dt_gini_gsmodel_result.best_estimator_), #Decision tree Gini

    ('nb_be', nb), #Naive bayes

    ('sgd_be', sgd), #Stocastic Gradient Descent

    ('knn_be', knn_gsmodel_result.best_estimator_), #K-nearest Neighbors

    ('rfm_be', rfm_gsmodel_result.best_estimator_)], voting='soft') # Random Forest

vote_classify_model = vote_classify.fit(x_train, y_train)

vote_classify_ypred = vote_classify_model.predict(x_test)

print(accuracy_score(vote_classify_ypred,y_test))
print(confusion_matrix(vote_classify_ypred,y_test))
print(classification_report(vote_classify_ypred,y_test))
fpr, tpr, _ =  roc_curve(y_test,vote_classify_model.predict_proba(x_test)[:,1])

roc_auc = auc(fpr,tpr)
plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange',

         lw=lw, label = 'ROC curve (area = %0.3f)' % (roc_auc))

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
X.columns
plt.figure(figsize=(15,5))

plt.bar(X.columns,rfm_gsmodel_result.best_estimator_.feature_importances_)

plt.xticks(X.columns,rotation='vertical')

plt.show()
# #Bonus SVM Models - COMMENTED BECAUSE OF HIGH EXECUTION TIME

# from sklearn.svm import SVC

# x_discarded, x_chosen, y_discarded, y_chosen = train_test_split(X,y,test_size=0.1,random_state=42)

# x_train, x_test, y_train, y_test = train_test_split(x_chosen,y_chosen,test_size=0.25,random_state=42)

# #Linear svm

# svc_linear = SVC(kernel='linear', gamma='auto')

# svc_linear.fit(x_train,y_train)

# svc_linear_y_pred = svc_linear.predict(x_test)

# print('linear: %s' %(accuracy_score(svc_linear_y_pred,y_test)))

# #quadratic svm 

# svc_quadratic = SVC(kernel='poly',degree=2, gamma='auto')

# svc_quadratic.fit(x_train,y_train)

# svc_quadratic_y_pred = svc_quadratic.predict(x_test)

# print('quadratic: %s' %(accuracy_score(svc_quadratic_y_pred,y_test)))

# #cubic svm

# svc_cubic = SVC(kernel='poly',degree=3, gamma='auto')

# svc_cubic.fit(x_train,y_train)

# svc_cubic_y_pred = svc_cubic.predict(x_test)

# print('cubic: %s' %(accuracy_score(svc_cubic_y_pred,y_test)))