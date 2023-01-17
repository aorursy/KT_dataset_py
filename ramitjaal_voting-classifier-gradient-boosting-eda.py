import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style = 'whitegrid')
import matplotlib.pyplot as plt
import plotly.offline as ply
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode()
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score,recall_score
df = pd.read_csv("../input/diabetes.csv")
df.head(5)
df.info()
df.describe().T
df.isna().sum()
df.isnull().sum()
sns.countplot(x=df.dtypes ,data=df)
plt.ylabel("number of data type")
plt.xlabel("data types")
corr_df = df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr_df, cmap = 'coolwarm', linecolor = 'white', linewidth =1, annot = True)
corr_df["Outcome"].sort_values(ascending = False)
values = pd.Series(df["Outcome"]).value_counts()
trace = go.Pie(values=values)
ply.iplot([trace])
plt.figure(figsize = (20,10))
sns.scatterplot(x = df['Age'], y = df['BMI'], palette="ch:r=-.2,d=.3_r", hue = df["Outcome"])
fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(df["Age"], ax = ax[0,0])
sns.distplot(df["Pregnancies"], ax = ax[0,1])
sns.distplot(df["Glucose"], ax = ax[1,0])
sns.distplot(df["BMI"], ax = ax[1,1])
sns.distplot(df["BloodPressure"], ax = ax[2,0])
sns.distplot(df["SkinThickness"], ax = ax[2,1])
sns.distplot(df["Insulin"], ax = ax[3,0])
sns.distplot(df["DiabetesPedigreeFunction"], ax = ax[3,1])
sns.pairplot(df, hue='Outcome')

df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

df['Glucose'].fillna(df['Glucose'].mean(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].mean(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].mean(), inplace = True)
df['Insulin'].fillna(df['Insulin'].mean(), inplace = True)
df['BMI'].fillna(df['BMI'].mean(), inplace = True)
fig, ax = plt.subplots(4,2, figsize=(16,16))
sns.distplot(df["Age"], ax = ax[0,0])
sns.distplot(df["Pregnancies"], ax = ax[0,1])
sns.distplot(df["Glucose"], ax = ax[1,0])
sns.distplot(df["BMI"], ax = ax[1,1])
sns.distplot(df["BloodPressure"], ax = ax[2,0])
sns.distplot(df["SkinThickness"], ax = ax[2,1])
sns.distplot(df["Insulin"], ax = ax[3,0])
sns.distplot(df["DiabetesPedigreeFunction"], ax = ax[3,1])

ss = StandardScaler()
X = ss.fit_transform(df)
X =  pd.DataFrame(ss.fit_transform(df.drop(["Outcome"],axis = 1),),
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head(5)
y = df['Outcome']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
logp = log_reg.predict(X_test)
y_train_pred_log = cross_val_predict(log_reg, X_train, y_train, cv=3)
confusion_matrix(y_train,y_train_pred_log)
print('Precision Score {}'.format(round(precision_score(y_test,logp),3)))
print('Recall Score {}'.format(round(recall_score(y_test,logp),3)))
print("ROC AUC {}".format(round(roc_auc_score(y_test,logp),3)))
gbrt = GradientBoostingClassifier(random_state=42)
gbrt.fit(X_train, y_train)
gbrtp = gbrt.predict(X_test)
y_gbrt = cross_val_predict(gbrt, X_train, y_train, cv=3)
confusion_matrix(y_train,y_gbrt)
print('Precision Score {}'.format(round(precision_score(y_test,gbrtp),3)))
print('Recall Score {}'.format(round(recall_score(y_test,gbrtp),3)))
print("ROC AUC {}".format(round(roc_auc_score(y_test,gbrtp),3)))
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train,y_train)
ranp = forest_clf.predict(X_test)
y_train_pred_ran = cross_val_predict(forest_clf, X_train, y_train, cv=3)
confusion_matrix(y_train,y_train_pred_ran)
print('Precision Score {}'.format(round(precision_score(y_test,ranp),3)))
print('Recall Score {}'.format(round(recall_score(y_test,ranp),3)))
print("ROC AUC {}".format(round(roc_auc_score(y_test,ranp),3)))
param_grid = [{'n_estimators':np.arange(1,50)}]
forest_reg = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(forest_reg,param_grid,cv=5)
grid_search.fit(X_train,y_train)
grid_search.best_estimator_
print("Best Score {}".format(str(grid_search.best_score_)))
print("Best Parameters {}".format(str(grid_search.best_params_)))
forest_g = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=46, n_jobs=None,
            oob_score=False, random_state=42, verbose=0, warm_start=False)
forest_g.fit(X_train,y_train)
rang = forest_g.predict(X_test)
y_train_pred_rang = cross_val_predict(forest_g, X_train, y_train, cv=3)
confusion_matrix(y_train,y_train_pred_rang)
print('Precision Score {}'.format(round(precision_score(y_test,rang),3)))
print('Recall Score {}'.format(round(recall_score(y_test,rang),3)))
print("ROC AUC {}".format(round(roc_auc_score(y_test,rang),3)))
voting_clf = VotingClassifier(estimators=[('lr', log_reg),('rf', forest_g)], voting='hard')
voting_clf.fit(X_train,y_train)
votinglr= voting_clf.predict(X_test)
y_train_pred_vt = cross_val_predict(voting_clf, X_train, y_train, cv=3)
confusion_matrix(y_train,y_train_pred_vt)
print('Precision Score {}'.format(round(precision_score(y_test,votinglr),3)))
print('Recall Score {}'.format(round(recall_score(y_test,votinglr),3)))
voting_gb = VotingClassifier(estimators=[('lr', log_reg),('gb', gbrt)], voting='hard')
voting_gb.fit(X_train,y_train)
votinggb= voting_clf.predict(X_test)
y_train_pred_vt = cross_val_predict(voting_gb, X_train, y_train, cv=3)
confusion_matrix(y_train,y_train_pred_vt)
voting_rg = VotingClassifier(estimators=[('rf', forest_g),('gb', gbrt)], voting='hard')
voting_rg.fit(X_train,y_train)
votingrg= voting_rg.predict(X_test)
y_train_pred_vt = cross_val_predict(voting_rg, X_train, y_train, cv=3)
confusion_matrix(y_train,y_train_pred_vt)
voting_lgr = VotingClassifier(estimators=[('lr', log_reg),('gb', gbrt),('rf',forest_g)], voting='hard')
voting_lgr.fit(X_train,y_train)
votinglgr= voting_lgr.predict(X_test)
y_train_pred_vt = cross_val_predict(voting_lgr, X_train, y_train, cv=3)
confusion_matrix(y_train,y_train_pred_vt)
for clf in (voting_clf,voting_gb,voting_rg,voting_lgr):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(round(roc_auc_score(y_test,y_pred),3))