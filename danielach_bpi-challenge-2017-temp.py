import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import  StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn import metrics

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
bpi_challenge_2017_data = pd.read_csv('../input/bpichallenge2017/BPI_Challenge_2017.csv') 
bpi_challenge_2017_data.head()
bpi_challenge_2017_data.info()
bpi_challenge_2017_data.isnull().sum().sort_values(ascending=False)
df_temporal_exp = bpi_challenge_2017_data.copy()
df_temporal_exp.head()
df_temporal_exp['datetime'] = pd.to_datetime(df_temporal_exp['time:timestamp'])
df_temporal_exp.sample(2)
df_temporal_exp['week'] = df_temporal_exp['datetime'].dt.strftime("%G_WK%V")
df_temporal_exp.sample(2)
apps_created = df_temporal_exp[df_temporal_exp['concept:name'] == 'A_Create Application']
apps_created.head()
apps_created_per_week = apps_created.groupby(['week'])['case:concept:name'].count()
apps_created_per_week.head()
plt.figure(figsize=[16,10])
ax = sns.barplot(x=apps_created_per_week.index, y=apps_created_per_week.values)
ax.set(ylabel="Number of applications", xlabel = "Week")
ax.set_title("Number of applications per week")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()
apps_pending = df_temporal_exp[df_temporal_exp['concept:name'] == 'A_Pending']
apps_pending.head()
pendings_per_week = apps_pending.groupby(['week'])['case:concept:name'].count()
pendings_per_week.sample(5)
plt.figure(figsize=[16,10])
ax = sns.barplot(x=pendings_per_week.index, y=pendings_per_week.values)
ax.set(ylabel="Number of applications pendings", xlabel = "Week")
ax.set_title("Number of applications pendings per week")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()
apps_name_pending = df_temporal_exp[df_temporal_exp['concept:name'] == 'A_Pending']['case:concept:name'].drop_duplicates()
events_apps_pending = df_temporal_exp.merge(apps_name_pending, on='case:concept:name')
events_apps_pending.sample(5)
mean_ev_apps_pending_per_week = (events_apps_pending.groupby(['week','case:concept:name'])['EventID'].count()
                                 .groupby(['week']).mean()
                                 .reset_index()
                                 .rename(index = str, columns = {'EventID': "Mean of events"}))
mean_ev_apps_pending_per_week.sample(5)
plt.figure(figsize=[16,10])
ax = sns.barplot(x='week', y='Mean of events', data=mean_ev_apps_pending_per_week)
ax.set(ylabel="Mean of events", xlabel = "Week")
ax.set_title("Mean of events per week of pending applications")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()
apps_start_date = (df_temporal_exp[df_temporal_exp['concept:name'] == 'A_Create Application']
                    .filter(['case:concept:name','datetime'])
                    .rename(index=str, columns={'datetime' : 'start_date'}))

apps_end_date = (df_temporal_exp[df_temporal_exp['concept:name'].isin(['A_Cancelled', 'A_Denied', 'A_Pending'])]
                    .filter(['case:concept:name', 'datetime'])
                    .rename(index=str, columns={'datetime': 'end_date'}))
                    
apps_duration = pd.merge(apps_start_date, apps_end_date, how='inner', on='case:concept:name')
apps_duration['num_weeks'] = (apps_duration['end_date'] - apps_duration['start_date']).dt.days/7

apps_duration.filter(['case:concept:name', 'num_weeks']).sample(5)
apps_duration['num_weeks'].describe().astype(int)
plt.figure(figsize=[16,10])
ax = sns.boxplot(x=apps_duration['num_weeks'])
ax.set_title('Applications duration (weeks)')
ax.set_xlabel('Number of weeks')
plt.show()
plt.figure(figsize=[16,10])
num_weeks = apps_duration['num_weeks'].astype(int)
ax = sns.countplot(num_weeks)
for p in ax.patches:
    (ax.annotate('{:.2f}%'.format(p.get_height()/len(num_weeks)*100), 
                 (p.get_x(), p.get_height()+1), fontsize=14))
ax.set_title("Number of Weeks Distribution")
ax.set(xlabel="Number of Weeks", ylabel="Number of Applications")
plt.show()
df_temporal = df_temporal_exp[['datetime', 'case:concept:name', 'concept:name', 'EventID']].copy(deep=True)
df_temporal.head()
df_temporal['start_week'] = (df_temporal['datetime']
                             .apply(lambda x : x - timedelta(x.dayofweek))
                             .dt.strftime('%d-%m-%Y'))
df_temporal['start_week'] = df_temporal['start_week'].apply(lambda x : datetime.strptime(x, '%d-%m-%Y'))
df_temporal.sample(2)
df_temporal['quarter'] = df_temporal['datetime'].dt.quarter
                            #.apply(lambda x: '1º Quarter' if x == 1 
                               #else ('2º Quarter' if x == 2 
                                     #else ('3º Quarter' if x == 3 
                                           #else '4º Quarter'))))
df_temporal.sample(5)
df_temporal['month'] =  df_temporal['datetime'].dt.month
df_temporal.sample(5)
apps_start_date = (df_temporal[df_temporal['concept:name'] == 'A_Create Application']
                    .filter(['case:concept:name','datetime'])
                    .rename(index=str, columns={'datetime' : 'creation_date'}))

df_temporal = df_temporal.merge(apps_start_date, on='case:concept:name', how='right')
df_temporal.head()
apps_final_date = (df_temporal[df_temporal['concept:name'].isin(['A_Denied', 'A_Cancelled', 'A_Pending'])]
                    .filter(['case:concept:name','datetime'])
                    .rename(index=str, columns={'datetime' : 'completion_date'}))

df_temporal = df_temporal.merge(apps_final_date, on='case:concept:name', how='right')
df_temporal.head()
df_temporal['creation_date'] = df_temporal['creation_date'].apply(lambda x : datetime.strptime(x.strftime('%d-%m-%Y'), '%d-%m-%Y'))
df_temporal['completion_date'] = df_temporal['completion_date'].apply(lambda x : datetime.strptime(x.strftime('%d-%m-%Y'), '%d-%m-%Y'))

df_temporal.head()
df_temporal['end_week_prev'] = df_temporal['start_week'].apply(lambda x : x - timedelta(1))
df_temporal['duration'] = ((df_temporal['end_week_prev'] - df_temporal['creation_date']).dt.days).apply(lambda x : 0 if x < 0 else x)
df_temporal.sample(5)
ev_app_per_week = (df_temporal.groupby(['case:concept:name', 'start_week'])['EventID'].count()
                                .reset_index()
                                .rename(index = str, columns = {'EventID': "ev_by_week"}))

ev_app_per_week["events_accum"] = ev_app_per_week.groupby(['case:concept:name'])['ev_by_week'].cumsum() 
df_temporal = df_temporal.merge(ev_app_per_week, on=['case:concept:name', 'start_week'], how='inner')
df_temporal.head()
df_temporal['events_qty'] = df_temporal['events_accum'] - df_temporal['ev_by_week']
df_temporal.head()
apps_offer_qty = (df_temporal[df_temporal['concept:name'] == 'O_Create Offer']
    .groupby(['case:concept:name', 'start_week'])['EventID'].count()
    .reset_index()  
    .rename(index = str, columns = {'EventID': "offer_by_week"}))

apps_offer_qty["offers_accum"] = apps_offer_qty.groupby(['case:concept:name'])['offer_by_week'].cumsum() 
apps_offer_qty.head()
df_temporal = df_temporal.merge(apps_offer_qty, on=['case:concept:name', 'start_week'], how='outer')
df_temporal[:20]
df_temporal['offers_qty'] = df_temporal['offers_accum'] - df_temporal['offer_by_week']
df_temporal.head()
df_temporal.isnull().sum().sort_values(ascending=False)
df_temporal = df_temporal.fillna(0)
df_temporal.isnull().sum().sort_values(ascending=False)
df_temporal['offers_accum'] = df_temporal['offers_accum'].astype(int)
df_temporal['offers_qty'] = df_temporal['offers_qty'].astype(int)
df_temporal.head()
df_temporal['end_week'] = (df_temporal['start_week']
                           .apply(lambda x : x + timedelta(6)))
df_temporal.sample(2)
df_temporal['label'] = df_temporal['completion_date'] <= df_temporal['end_week']
df_temporal.sample(10)
df_temporal = df_temporal.drop(['concept:name', 'datetime', 'EventID', 'ev_by_week', 'offer_by_week'], axis=1)
df_temporal
df_temporal = df_temporal.drop_duplicates()
df_temporal[:30]
plt.figure(figsize=[16,10])
labels_occ = df_temporal['label']
ax = sns.barplot(x=labels_occ.value_counts().index, y=labels_occ.value_counts(), orient='v')
for p in ax.patches:
    (ax.annotate('{:.2f}%'.format(p.get_height()/len(labels_occ)*100), 
                 (p.get_x(), p.get_height()+1), fontsize=14))
ax.set(xlabel="Label", ylabel = "Quantity")
ax.set_title("Label Distribution")
plt.show()
df_model = df_temporal[['events_accum', 'events_qty', 'offers_accum', 'offers_qty', 'case:concept:name', 'label']].copy()
x_train_test, x_validation = train_test_split(df_model['case:concept:name'].drop_duplicates(), test_size=0.2)
x_train_test = df_model.merge(x_train_test, on="case:concept:name")
x_validation = df_model.merge(x_validation, on="case:concept:name")
y_validation = x_validation['label']
x_validation = x_validation.iloc[:,0:4]
x_train, x_test = train_test_split(x_train_test['case:concept:name'].drop_duplicates(), test_size=0.25)
x_train = x_train_test.merge(x_train, on="case:concept:name")
x_test = x_train_test.merge(x_test, on="case:concept:name")

y_test = x_test['label']
y_train = x_train['label']
x_train = x_train.iloc[:,0:4]
x_test = x_test.iloc[:,0:4]
pipe = Pipeline([
   ('scaler', StandardScaler()),
   ('classifier', LogisticRegression())
])
param_grid = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l2'],
     'classifier__C' : np.logspace(-4, 4, 20)},
    {'classifier' : [KNeighborsClassifier()],
     'classifier__leaf_size' : [1,5,10,30],
     'classifier__n_neighbors' : [2,5,10,20], 
     'classifier__p' : [1,2]},
    {'classifier' : [DecisionTreeClassifier(random_state=199)],
     'classifier__max_depth' : [10, 50, 100],
     'classifier__min_samples_split' : [1, 5, 10]},
    {'classifier' : [RandomForestClassifier(random_state=199)],
     'classifier__n_estimators' : [10,50,250],
     'classifier__max_depth' : [10, 50, 100],
     'classifier__min_samples_split' : [1, 5, 10]}
]
clf_gs = GridSearchCV(estimator=pipe, param_grid=param_grid, n_jobs=-1)
clf_gs.fit(x_train, y_train)
print("Best params (CV score=%0.5f):" % clf_gs.best_score_)
print(clf_gs.best_params_)
y_pred = clf_gs.predict(x_test)
accuracy_inb = metrics.accuracy_score(y_test, y_pred)
recall_inb = metrics.recall_score(y_test, y_pred)
precision_inb = metrics.precision_score(y_test, y_pred)
f1_inb = metrics.f1_score(y_test, y_pred)

print("Accuracy: ", accuracy_inb)
print("Recall: ", recall_inb)
print("Precision: ", recall_inb)
print("F1-Score: ", f1_inb)
plt.figure(figsize=(10,8))
ax = (sns.heatmap(metrics.confusion_matrix(y_test, y_pred), annot=True, annot_kws={"size": 20}, 
                 linewidths=.1, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True)))
plt.yticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
plt.xticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
ax.set_title("Confusion Matrix")
ax.set(xlabel="Predicted", ylabel="Actual");
plt.show()
y_pred_val = clf_gs.predict(x_validation)
accuracy_inb_val = metrics.accuracy_score(y_validation, y_pred_val)
recall_inb_val = metrics.recall_score(y_validation, y_pred_val)
precision_inb_val = metrics.precision_score(y_validation, y_pred_val)
f1_inb_val = metrics.f1_score(y_validation, y_pred_val)

print("Accuracy: ", accuracy_inb_val)
print("Recall: ", recall_inb_val)
print("Precision: ", recall_inb_val)
print("F1-Score: ", f1_inb_val)
plt.figure(figsize=(10,8))
ax = (sns.heatmap(metrics.confusion_matrix(y_validation, y_pred_val), annot=True, annot_kws={"size": 20}, 
                 linewidths=.1, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True)))
plt.yticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
plt.xticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
ax.set_title("Confusion Matrix")
ax.set(xlabel="Predicted", ylabel="Actual");
plt.show()
features = x_train.columns

features_importances = pd.Series(clf_gs.best_estimator_.named_steps["classifier"].feature_importances_, index=features)

plt.figure(figsize=[16,10])
ax = sns.barplot(x=features_importances.values, y=features_importances)
ax.set_title("Feature importances")
plt.xticks(list(range(len(features_importances))), features)
plt.show()
y = x_train_test.label
x = x_train_test
rus = RandomUnderSampler(random_state=0)
x_rus, y_rus = rus.fit_sample(x, y)
y_rus_index, y_rus_counts = np.unique(y_rus, return_counts=True)
plt.figure(figsize=[16,10])
ax = sns.barplot(x=y_rus_index, y=y_rus_counts, orient='v')
for p in ax.patches:
    (ax.annotate('{:.2f}%'.format(p.get_height()/len(y_rus)*100), 
                 (p.get_x(), p.get_height()+1), fontsize=14))
ax.set(xlabel="Label", ylabel = "Quantity")
ax.set_title("Label Distribution")
plt.show()
x_rus_df = (pd.DataFrame(x_rus, columns=df_model.columns))
x_rus_df.head()
x_train_rus, x_test_rus = train_test_split(x_rus_df['case:concept:name'].drop_duplicates(), test_size=0.25)
x_train_rus = x_rus_df.merge(x_train_rus, on="case:concept:name")
x_test_rus = x_rus_df.merge(x_test_rus, on="case:concept:name")

y_test_rus = x_test_rus['label'].astype(bool)
y_train_rus = x_train_rus['label'].astype(bool)
x_train_rus = x_train_rus.iloc[:,0:4]
x_test_rus = x_test_rus.iloc[:,0:4]
clf_gs.fit(x_train_rus, y_train_rus)
print("Best params (CV score=%0.5f):" % clf_gs.best_score_)
print(clf_gs.best_params_)
y_pred_rus = clf_gs.predict(x_test_rus)
accuracy_rus = metrics.accuracy_score(y_test_rus, y_pred_rus)
recall_rus = metrics.recall_score(y_test_rus, y_pred_rus)
precision_rus = metrics.precision_score(y_test_rus, y_pred_rus)
f1_rus = metrics.f1_score(y_test_rus, y_pred_rus)

print("Accuracy: ", accuracy_rus)
print("Recall: ", recall_rus)
print("Precision: ", precision_rus)
print("f1-Score: ", f1_rus)
plt.figure(figsize=(10,8))
ax = (sns.heatmap(metrics.confusion_matrix(y_test_rus, y_pred_rus), annot=True, annot_kws={"size": 20}, 
                 linewidths=.1, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True)))
plt.yticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
plt.xticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
ax.set_title("Confusion Matrix")
ax.set(xlabel="Predicted", ylabel="Actual");
plt.show()
y_pred_rus_val = clf_gs.predict(x_validation)
accuracy_rus_val = metrics.accuracy_score(y_validation, y_pred_rus_val)
recall_rus_val = metrics.recall_score(y_validation, y_pred_rus_val)
precision_rus_val = metrics.precision_score(y_validation, y_pred_rus_val)
f1_rus_val = metrics.f1_score(y_validation, y_pred_rus_val)

print("Accuracy: ", accuracy_rus_val)
print("Recall: ", recall_rus_val)
print("Precision: ", precision_rus_val)
print("F1-Score: ", f1_rus_val)
plt.figure(figsize=(10,8))
ax = (sns.heatmap(metrics.confusion_matrix(y_validation, y_pred_rus_val), annot=True, annot_kws={"size": 20}, 
                 linewidths=.1, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True)))
plt.yticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
plt.xticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
ax.set_title("Confusion Matrix")
ax.set(xlabel="Predicted", ylabel="Actual");
plt.show()
ros = RandomOverSampler(random_state=0)
x_ros, y_ros = ros.fit_sample(x, y)
y_ros_index, y_ros_counts = np.unique(y_ros, return_counts=True)
plt.figure(figsize=[16,10])
ax = sns.barplot(x=y_ros_index, y=y_ros_counts, orient='v')
for p in ax.patches:
    (ax.annotate('{:.2f}%'.format(p.get_height()/len(y_ros)*100), 
                 (p.get_x(), p.get_height()+1), fontsize=14))
ax.set(xlabel="Label", ylabel = "Quantity")
ax.set_title("Label Distribution")
plt.show()
x_ros_df = pd.DataFrame(x_ros, columns=df_model.columns)
x_ros_df.head()
x_train_ros, x_test_ros = train_test_split(x_ros_df['case:concept:name'].drop_duplicates(), test_size=0.25)
x_train_ros = x_ros_df.merge(x_train_ros, on="case:concept:name")
x_test_ros = x_ros_df.merge(x_test_ros, on="case:concept:name")

y_test_ros = x_test_ros['label'].astype(bool)
y_train_ros = x_train_ros['label'].astype(bool)
x_train_ros = x_train_ros.iloc[:,0:4]
x_test_ros = x_test_ros.iloc[:,0:4]
clf_gs.fit(x_train_ros, y_train_ros)
print("Best params (CV score=%0.5f):" % clf_gs.best_score_)
print(clf_gs.best_params_)
y_pred_ros = clf_gs.predict(x_test_ros)
accuracy_ros = metrics.accuracy_score(y_test_ros, y_pred_ros)
recall_ros = metrics.recall_score(y_test_ros, y_pred_ros)
precision_ros = metrics.precision_score(y_test_ros, y_pred_ros)
f1_ros = metrics.f1_score(y_test_ros, y_pred_ros)

print("Accuracy: ", accuracy_ros)
print("Recall: ", recall_ros)
print("Precision: ", precision_ros)
print("F1-Score: ", f1_ros)
plt.figure(figsize=(10,8))
ax = (sns.heatmap(metrics.confusion_matrix(y_test_rus, y_pred_rus), annot=True, annot_kws={"size": 20}, 
                 linewidths=.1, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True)))
plt.yticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
plt.xticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
ax.set_title("Confusion Matrix")
ax.set(xlabel="Predicted", ylabel="Actual");
plt.show()
y_pred_ros_val = clf_gs.predict(x_validation)
accuracy_ros_val = metrics.accuracy_score(y_validation, y_pred_ros_val)
recall_ros_val = metrics.recall_score(y_validation, y_pred_ros_val)
precision_ros_val = metrics.precision_score(y_validation, y_pred_ros_val)
f1_ros_val = metrics.f1_score(y_validation, y_pred_ros_val)

print("Accuracy: ", accuracy_ros_val)
print("Recall: ", recall_ros_val)
print("Precision: ", precision_ros_val)
print("F1-Score: ", f1_ros_val)
plt.figure(figsize=(10,8))
ax = (sns.heatmap(metrics.confusion_matrix(y_validation, y_pred_ros_val), annot=True, annot_kws={"size": 20}, 
                 linewidths=.1, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True)))
plt.yticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
plt.xticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
ax.set_title("Confusion Matrix")
ax.set(xlabel="Predicted", ylabel="Actual");
plt.show()
param_grid_cost = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l2'],
     'classifier__C' : np.logspace(-4, 4, 20),
     'classifier__class_weight': ['balanced']},
    {'classifier' : [KNeighborsClassifier()],
     'classifier__leaf_size' : [1,5,10,30],
     'classifier__n_neighbors' : [2,5,10,20], 
     'classifier__p' : [1,2],
     'classifier__weights': ['distance']},
    {'classifier' : [DecisionTreeClassifier(random_state=199)],
     'classifier__max_depth' : [10, 50, 100],
     'classifier__min_samples_split' : [1, 5, 10],
     'classifier__class_weight': ['balanced']},
    {'classifier' : [RandomForestClassifier(random_state=199)],
     'classifier__n_estimators' : [10,50,250],
     'classifier__max_depth' : [10, 50, 100],
     'classifier__min_samples_split' : [1, 5, 10],
     'classifier__class_weight': ['balanced', 'balanced_subsample']}
]
clf_gs_cost = GridSearchCV(estimator=pipe, param_grid=param_grid_cost, n_jobs=-1)
clf_gs_cost.fit(x_train, y_train)
print("Best params (CV score=%0.5f):" % clf_gs_cost.best_score_)
print(clf_gs_cost.best_params_)
y_pred_cs = clf_gs_cost.predict(x_test)
accuracy_cs = metrics.accuracy_score(y_test, y_pred_cs)
recall_cs = metrics.recall_score(y_test, y_pred_cs)
precision_cs = metrics.precision_score(y_test, y_pred_cs)
f1_cs = metrics.f1_score(y_test, y_pred_cs)

print("Accuracy: ", accuracy_cs)
print("Recall: ", recall_cs)
print("Precision: ", precision_cs)
print("F1-Score: ", f1_cs)
plt.figure(figsize=(10,8))
ax = (sns.heatmap(metrics.confusion_matrix(y_test, y_pred_cs), annot=True, annot_kws={"size": 20}, 
                 linewidths=.1, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True)))
plt.yticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
plt.xticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
ax.set_title("Confusion Matrix")
ax.set(xlabel="Predicted", ylabel="Actual");
plt.show()
y_pred_cs_val = clf_gs_cost.predict(x_validation)
accuracy_cs_val = metrics.accuracy_score(y_validation, y_pred_cs_val)
recall_cs_val = metrics.recall_score(y_validation, y_pred_cs_val)
precision_cs_val = metrics.precision_score(y_validation, y_pred_cs_val)
f1_cs_val = metrics.f1_score(y_validation, y_pred_cs_val)

print("Accuracy: ", accuracy_cs_val)
print("Recall: ", recall_cs_val)
print("Precision: ", precision_cs_val)
print("F1-Score: ", f1_cs_val)
plt.figure(figsize=(10,8))
ax = (sns.heatmap(metrics.confusion_matrix(y_test, y_pred_cs), annot=True, annot_kws={"size": 20}, 
                 linewidths=.1, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True)))
plt.yticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
plt.xticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
ax.set_title("Confusion Matrix")
ax.set(xlabel="Predicted", ylabel="Actual");
plt.show()
x_new = x.copy()
y_new = x_new.label
x_new['case:concept:name'] = x_new['case:concept:name'].str.slice(start=12)
x_new.head()
tkl = TomekLinks()
x_tkl, y_tkl = tkl.fit_sample(x_new, y_new)
y_tkl_index, y_tkl_counts = np.unique(y_tkl, return_counts=True)
plt.figure(figsize=[16,10])
ax = sns.barplot(x=y_tkl_index, y=y_tkl_counts, orient='v')
for p in ax.patches:
    (ax.annotate('{:.2f}%'.format(p.get_height()/len(y_tkl)*100), 
                 (p.get_x(), p.get_height()+1), fontsize=14))
ax.set(xlabel="Label", ylabel = "Quantity")
ax.set_title("Label Distribution")
plt.show()
x_tkl_df = pd.DataFrame(x_tkl, columns=df_model.columns)
x_tkl_df.head()
x_train_tkl, x_test_tkl = train_test_split(x_tkl_df['case:concept:name'].drop_duplicates(), test_size=0.25)
x_train_tkl = x_tkl_df.merge(x_train_tkl, on="case:concept:name")
x_test_tkl = x_tkl_df.merge(x_test_tkl, on="case:concept:name")

y_test_tkl = x_test_tkl['label'].astype(bool)
y_train_tkl = x_train_tkl['label'].astype(bool)
x_train_tkl = x_train_tkl.iloc[:,0:4]
x_test_tkl = x_test_tkl.iloc[:,0:4]
clf_gs.fit(x_train_tkl, y_train_tkl)
print("Best params (CV score=%0.5f):" % clf_gs.best_score_)
print(clf_gs.best_params_)
y_pred_tkl = clf_gs.predict(x_test_tkl)
accuracy_tkl = metrics.accuracy_score(y_test_tkl, y_pred_tkl)
recall_tkl = metrics.recall_score(y_test_tkl, y_pred_tkl)
precision_tkl = metrics.precision_score(y_test_tkl, y_pred_tkl)
f1_tkl = metrics.f1_score(y_test_tkl, y_pred_tkl)

print("Accuracy: ", accuracy_tkl)
print("Recall: ", recall_tkl)
print("Precision: ", precision_tkl)
print("F1-Score: ", f1_tkl)
plt.figure(figsize=(10,8))
ax = (sns.heatmap(metrics.confusion_matrix(y_test_tkl, y_pred_tkl), annot=True, annot_kws={"size": 20}, 
                 linewidths=.1, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True)))
plt.yticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
plt.xticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
ax.set_title("Confusion Matrix")
ax.set(xlabel="Predicted", ylabel="Actual");
plt.show()
y_pred_tkl_val = clf_gs.predict(x_validation)
accuracy_tkl_val = metrics.accuracy_score(y_validation, y_pred_tkl_val)
recall_tkl_val = metrics.recall_score(y_validation, y_pred_tkl_val)
precision_tkl_val = metrics.precision_score(y_validation, y_pred_tkl_val)
f1_tkl_val = metrics.f1_score(y_validation, y_pred_tkl_val)

print("Accuracy: ", accuracy_tkl_val)
print("Recall: ", recall_tkl_val)
print("Precision: ", precision_tkl_val)
print("F1-Score: ", f1_tkl_val)
plt.figure(figsize=(10,8))
ax = (sns.heatmap(metrics.confusion_matrix(y_validation, y_pred_tkl_val), annot=True, annot_kws={"size": 20}, 
                 linewidths=.1, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True)))
plt.yticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
plt.xticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
ax.set_title("Confusion Matrix")
ax.set(xlabel="Predicted", ylabel="Actual");
plt.show()
smt = SMOTE(random_state=0)
x_smt, y_smt = smt.fit_sample(x_new, y_new)
y_smt_index, y_smt_counts = np.unique(y_smt, return_counts=True)
plt.figure(figsize=[16,10])
ax = sns.barplot(x=y_smt_index, y=y_smt_counts, orient='v')
for p in ax.patches:
    (ax.annotate('{:.2f}%'.format(p.get_height()/len(y_smt)*100), 
                 (p.get_x(), p.get_height()+1), fontsize=14))
ax.set(xlabel="Label", ylabel = "Quantity")
ax.set_title("Label Distribution")
plt.show()
x_smt_df = pd.DataFrame(x_smt, columns=df_model.columns)
x_smt_df.head()
x_train_smt, x_test_smt = train_test_split(x_smt_df['case:concept:name'].drop_duplicates(), test_size=0.25)
x_train_smt = x_smt_df.merge(x_train_smt, on="case:concept:name")
x_test_smt = x_smt_df.merge(x_test_smt, on="case:concept:name")

y_test_smt = x_test_smt['label'].astype(bool)
y_train_smt = x_train_smt['label'].astype(bool)
x_train_smt = x_train_smt.iloc[:,0:4]
x_test_smt = x_test_smt.iloc[:,0:4]
clf_gs.fit(x_train_smt, y_train_smt)
print("Best params (CV score=%0.5f):" % clf_gs.best_score_)
print(clf_gs.best_params_)
y_pred_smt = clf_gs.predict(x_test_smt)
accuracy_smt = metrics.accuracy_score(y_test_smt, y_pred_smt)
recall_smt = metrics.recall_score(y_test_smt, y_pred_smt)
precision_smt = metrics.precision_score(y_test_smt, y_pred_smt)
f1_smt = metrics.f1_score(y_test_tkl, y_pred_tkl)

print("Accuracy: ", accuracy_smt)
print("Recall: ", recall_smt)
print("Precision: ", precision_smt)
print("F1-Score: ", f1_smt)
plt.figure(figsize=(10,8))
ax = (sns.heatmap(metrics.confusion_matrix(y_test_tkl, y_pred_tkl), annot=True, annot_kws={"size": 20}, 
                 linewidths=.1, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True)))
plt.yticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
plt.xticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
ax.set_title("Confusion Matrix")
ax.set(xlabel="Predicted", ylabel="Actual");
plt.show()
y_pred_smt_val = clf_gs.predict(x_validation)
accuracy_smt_val = metrics.accuracy_score(y_validation, y_pred_smt_val)
recall_smt_val = metrics.recall_score(y_validation, y_pred_smt_val)
precision_smt_val = metrics.precision_score(y_validation, y_pred_smt_val)
f1_smt_val = metrics.f1_score(y_validation, y_pred_smt_val)

print("Accuracy: ", accuracy_smt_val)
print("Recall: ", recall_smt_val)
print("Precision: ", precision_smt_val)
print("F1-Score: ", f1_smt_val)
plt.figure(figsize=(10,8))
ax = (sns.heatmap(metrics.confusion_matrix(y_validation, y_pred_smt_val), annot=True, annot_kws={"size": 20}, 
                 linewidths=.1, fmt='d', cmap=sns.cubehelix_palette(as_cmap=True)))
plt.yticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
plt.xticks([0.5,1.5,2], ['True', 'False'], va='center', fontsize=14)
ax.set_title("Confusion Matrix")
ax.set(xlabel="Predicted", ylabel="Actual");
plt.show()
results = (pd.DataFrame({
    "Technique": ["Unbalanced (train and test)", "Unbalanced (train and val)",  "RUS (train and test)",  
                  "RUS (train)\ Unbalanced data (val)", "ROS (train and test)", "ROS (train)\ Unbalanced data (val)", 
                  "TomekLink (train and test)",  "TomekLink (train)\ Unbalanced data (val)",
                  "SMOTE (train and test)", "SMOTE (train)\ Unbalanced data (test)", "Cost Sensitive (train and test)", 
                  "Cost-Sensitive (train)\ Unbalanced data (val)"],
    "Accuracy": [accuracy_inb, accuracy_inb_val, accuracy_rus, accuracy_rus_val, accuracy_ros, accuracy_ros_val, 
                 accuracy_tkl, accuracy_tkl_val, accuracy_smt, accuracy_smt_val, accuracy_cs, accuracy_cs_val],
    "Recall":  [recall_inb, recall_inb_val, recall_rus, recall_rus_val, recall_ros, recall_ros_val, 
                recall_tkl, recall_tkl_val, recall_smt, recall_smt_val, recall_cs, recall_cs_val],
    "Precision":  [precision_inb, precision_inb_val, precision_rus, precision_rus_val, precision_ros, precision_ros_val, 
                   precision_tkl, precision_tkl_val, precision_smt, precision_smt_val, precision_cs, precision_cs_val],
    "F1-Score":  [f1_inb, f1_inb_val, f1_rus, f1_rus_val, f1_ros, f1_ros_val, f1_tkl, f1_tkl_val, f1_smt, f1_smt_val, 
                  f1_cs, f1_cs_val],

}))

results.sort_values(by=['F1-Score'], ascending=False)
plt.figure(figsize=(16,10))
ax = results['F1-Score'].plot(kind='bar')
ax.set(title='F1-Score by techniques')
    
for p in ax.patches:
    (ax.annotate('{:.3f}%'.format(p.get_height()*100), 
                 (p.get_x() +  p.get_width()/2, p.get_y() + p.get_height()*1.01), ha='center'))
 
plt.xticks(np.arange(len(results['Technique'])), results['Technique'] , rotation=90)
plt.show()