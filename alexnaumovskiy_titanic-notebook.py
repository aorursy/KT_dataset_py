import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print('GPU device not found: {}'.format(device_name))
else: 
  print('Found GPU at: {}'.format(device_name))
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns

import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import ExtraTreeClassifier
import xgboost as xgb
import datetime
# Count nulls in data
def check_cnt_null():
    rows = df_main.shape[0]
    df_null_analyse = pd.DataFrame([(cols,rows,df_main[cols].isnull().sum()) for cols in df_main.columns],columns=['column','cnt_total','cnt_null'])
    df_null_analyse['pct_null'] = np.round(df_null_analyse['cnt_null']/rows*100,2)
    print(df_null_analyse[df_null_analyse.cnt_null>0])
df_test = pd.read_csv('../input/titanic/test.csv',delimiter=',',header=0)
df_train = pd.read_csv('../input/titanic/train.csv',delimiter=',',header=0)
# Mark survived in test_data  == -1
df_test['Survived'] = -1
df_main = pd.concat([df_train,df_test],sort=False)
df_main.sample(3)
# Create Filters for train and test data:
F_train = df_main.Survived > -1
F_test = df_main.Survived == -1
df_main[F_test].sample(3)
df_main[F_train].sample(3)
df_main.drop(columns=['Survived']).describe()
df_main.drop(columns=['Survived']).describe(include='object')
df_main.info()
check_cnt_null()
sns.set()
f, axes = plt.subplots(1,4,figsize=(22,3))
df_main[F_train].Survived.value_counts(sort=False).plot.\
        pie(autopct='%1.1f%%', ax=axes[0], title='ALL',\
            colors = ['red', 'green'],labels=['Die','Survived'])
df_main[(F_train) & (df_main.Sex=='male')].Survived.value_counts(sort=False).plot.\
        pie(autopct='%1.1f%%', ax=axes[1], title ='Male',\
            colors = ['red', 'green'],labels=['Die','Survived'])
df_main[(F_train) & (df_main.Sex=='female')].Survived.value_counts(sort=False).plot.\
        pie(autopct='%1.1f%%', ax=axes[2], title ='Female',\
            colors = ['red', 'green'] ,labels=['Die','Survived'])

sns.barplot(x='Pclass',y='Survived',data=df_main[F_train],ax=axes[3],hue='Sex',ci=None)
axes[3].set_title('Proba survived by Sex&Pclass')
for p in axes[3].patches:
    axes[3].annotate(format(p.get_height(), '.2f'),(p.get_x() + p.get_width()/1.5,p.get_height()),\
                     ha = 'center', va = 'center', xytext = (0, 5), textcoords ='offset points')
plt.show()
check_cnt_null()
df_age_tmp = df_main.Age
max_age =np.int64(np.max(df_age_tmp))

f, ax = plt.subplots(1,1, figsize=(20, 4))
sns.distplot(df_age_tmp,color='c',ax=ax,bins=max_age,hist_kws={'range': (0.0, max_age)},kde_kws={'cut':0})
ax.set_xticks(ticks=np.arange(0,max_age,1))
ax.axvline(df_age_tmp.mean(), color='r', linestyle='--',lw=2)
ax.axvline(df_age_tmp.median(), color='g', linestyle='-',lw=2)
ax.axvline(df_age_tmp.mode()[0], color='b', linestyle='-',lw=2)
ax.set_title('Age distr')
plt.tight_layout()
plt.show()

del df_age_tmp
f,ax = plt.subplots(1,1,figsize=(18,3))
sns.kdeplot(df_main[df_main.Survived == 1].Age,label ='Survived',ax=ax)
sns.kdeplot(df_main[df_main.Survived == 0].Age,label = 'Die',ax=ax)
ax.set_xticks(ticks=np.arange(0,max_age,5))
plt.show()
df_main['Salutation'] = df_main.Name.str.replace('.',',').str.split(r',',expand=True)[1].str.strip()
ax = sns.catplot(x='Salutation',y='Age',data=df_main,kind='bar',col='Sex',estimator=np.median,height=3,aspect=1.8)
ax.set_xticklabels(rotation=90)
plt.show()
df_main['Salutation'].value_counts()
df_main['Salutation'].replace(['Mlle','Mme','Ms','Lady','the Countess','Dona'],\
                              ['Miss','Miss','Miss','Mrs','Mrs','Mrs',],inplace=True)
df_main['Salutation'].replace(['Major','Sir','Don','Jonkheer','Col','Rev','Capt',],\
                              ['Mr','Mr','Mr','Other','Other','Other','Mr'],inplace=True)
df_main.loc[(df_main.Salutation=='Dr'),'Salutation'] = df_main[(df_main.Salutation=='Dr')].\
        apply(lambda x: 'Mr' if (x.Sex=='male') else 'Mrs',axis=1)

df_salutation = pd.DataFrame(df_main[~df_main.Age.isnull()].\
                             groupby(['Salutation','Pclass']).Age.median()).reset_index()
df_salutation.T
F_age_null = df_main.Age.isnull()
df_main.loc[F_age_null,'Age'] = df_main.loc[F_age_null].apply(lambda x:
        df_salutation.loc[(df_salutation.Salutation==x.Salutation)\
                          & (df_salutation.Pclass==x.Pclass),'Age'].median(),axis=1)
f, axes = plt.subplots(1,3,figsize=(18,3))
sns.barplot(x='Salutation',y='Survived',data=df_main[F_train],ci=None,ax = axes[0])
axes[0].set_title('Proba Survived')
sns.barplot(x='Salutation',y='Age',data=df_main[F_train],estimator=len,ax=axes[1])
axes[1].set_title('Count Salutation')
sns.barplot(x='Salutation',y='Age',data=df_main[F_train],estimator=np.median,ax=axes[2],ci=None)
axes[2].set_title('Median age Salutation')
for i,ax_cur in enumerate(axes):
    for p in ax_cur.patches:
        ax_cur.annotate(format(p.get_height(),'.2f'),(p.get_x() + p.get_width()/1.5,p.get_height()),\
                        ha = 'center', va = 'center', xytext = (0, 5),textcoords ='offset points')
g = sns.FacetGrid(df_main[(F_train)], col="Salutation",height=2.5,aspect=1.5,col_wrap=3,sharex=False,hue='Sex')
g.map(sns.distplot, "Age");
f, axa = plt.subplots(1,1,figsize=(22,3))
sns.distplot(df_main.Age,ax=axa,hist=True)
axa.set_xticks(ticks=np.arange(0,80,2))
axa.set_title('Age bin')
age_bins= [0,11,17,25,36,55,80]
for xc in  age_bins:
  axa.axvline(xc, color='g', linestyle='--',lw=2)
plt.show()
age_labels = np.arange(1,len(age_bins),1)
df_main['Age_bin'] = pd.cut(df_main.Age,bins=age_bins,labels=age_labels).astype(int)
f, (axes) = plt.subplots(1,3,figsize=(24,4))
sns.barplot(x='Age_bin',y='Survived',hue='Sex',data=df_main[F_train],ax=axes[0],ci=None)
axes[0].set_title('Proba Survived by Sex')
sns.barplot(x='Age_bin',y='Survived',data=df_main[F_train],ax=axes[1],ci=None)
axes[1].set_title('Proba Survived by Age_bin')
sns.barplot(x='Age_bin',y='Survived',hue='Survived',data=df_main[F_train],ax=axes[2],ci=None,estimator=len)
axes[2].set_title('Count Survived by Age_bin')
axes[2].set_ylabel('Count')
for axc in axes:
    for p in axc.patches:
        axc.annotate(format(p.get_height(), '.2f'),(p.get_x() + p.get_width()/1.5,p.get_height()),\
                     ha = 'center', va = 'center', xytext = (0, 5), textcoords ='offset points')
plt.show()
check_cnt_null()
f,(ax1,ax2) =plt.subplots(1,2,figsize=(15,3))
df_main.Embarked.value_counts().plot.pie(autopct='%1.1f%%',title='Count ALL',ax=ax1)
sns.barplot(x='Embarked', y='Survived',hue='Sex', data=df_main[F_train], ci=None,ax=ax2)
plt.show()
df_main.loc[df_main.Embarked.isnull(),:]
moda_embarked = df_main[(df_main.Sex=='female')&(df_main.Pclass==1)&(df_main.Survived==1)].Embarked.mode()[0]
df_main.loc[df_train.Embarked.isnull(),'Embarked'] = moda_embarked
moda_embarked
check_cnt_null()
F_Fare_zero = df_main.Fare.isnull()
df_fare_null = df_main[F_Fare_zero].iloc[0]
df_main.loc[(F_Fare_zero),:]
df_main.loc[(F_Fare_zero),'Fare'] = df_main[(df_main.Salutation == df_fare_null.Salutation) &\
                                            (df_main.Pclass == df_fare_null.Pclass) &\
                                            (df_main.Age_bin == df_fare_null.Age_bin)].Fare.median()
df_main.loc[(F_Fare_zero),'Fare']
df_main.loc[(df_main.Fare < 1),'Fare'] = df_main.Fare.median()
fare_cats,fbins = pd.qcut(df_main.Fare,5,retbins=True)
fbins[0]  = np.round(fbins[0])
fbins[-1] = np.ceil(fbins[-1])
fbins
f, axf = plt.subplots(1,1,figsize=(22,3))
sns.distplot(np.log10(df_main.Fare),ax=axf,hist=False)
axf.set_xticks(ticks=np.arange(0.1,2.9,0.08))
axf.set_title('Fare bins')
for xc in fbins:
    axf.axvline(np.log10(xc), color='g', linestyle='--',lw=2)
plt.show()
fare_labels = np.arange(1,len(fbins),1)
df_main['Fare_bin'] = pd.cut(df_main.Fare,bins=fbins,labels=fare_labels).astype(int)
f, axes = plt.subplots(1,4,figsize=(22,3))
sns.barplot(x='Fare_bin',y='Survived',data=df_main[F_train],ci=None,ax = axes[0])
axes[0].set_title('Proba Survived')
sns.barplot(x='Fare_bin',y='Survived',hue='Sex',data=df_main[F_train],ci=None,ax = axes[1])
axes[1].set_title('Proba Survived by Sex')
sns.barplot(x='Fare_bin',y='Survived',data=df_main[F_train],estimator=len,ax=axes[2])
axes[2].set_title('Count Fare_bin')
sns.barplot(x='Fare_bin',y='Fare',data=df_main[F_train],estimator=np.median,ax=axes[3],ci=None)
axes[3].set_title('Median fare Fare_bin')
for i,ax_cur in enumerate(axes):
    for p in ax_cur.patches:
        axes[i].annotate(format(p.get_height(),'.2f'),(p.get_x()+ p.get_width()/1.5,p.get_height()),\
                         ha = 'center', va= 'center',xytext =(0, 5),textcoords ='offset points')
check_cnt_null()
df_main.drop(columns='Cabin',inplace=True)
df_main['Family'] = df_main[['SibSp','Parch']].apply(lambda x : 1 if (x.SibSp + x.Parch)> 0 else 0,axis=1)
df_main['Family_size'] = df_main[['SibSp','Parch']].apply(lambda x : (x.SibSp + x.Parch),axis=1)
f, axes = plt.subplots(1,4,figsize=(24,3))
sns.barplot(x='Family',y='Survived',hue='Pclass',data=df_main[F_train],ci=None,ax = axes[0])
axes[0].set_title('Proba Family Survived')
sns.barplot(x='Family',y='Survived',hue='Pclass',data=df_main[F_train],estimator=len,ax=axes[1])
axes[1].set_title('Count Families')
sns.barplot(x='Family_size',y='Family_size',data=df_main[F_train],estimator=len,ax=axes[2],ci=None)
axes[2].set_title('Count Family_size')
sns.barplot(x='Family_size',y='Survived',data=df_main[F_train],ax=axes[3],ci=None)
axes[3].set_title('Proba Survived Family_size')
for i,ax_cur in enumerate(axes):
    for p in ax_cur.patches:
        ax_cur.annotate(format(p.get_height(),'.2f'),(p.get_x() + p.get_width()/1.5,p.get_height()),\
                        ha = 'center', va = 'center', xytext = (0, 5),textcoords ='offset points')
def family_bins(x):
  if (x == 0):
      return 0
  elif (x == 1) | (x== 2):
      return 1
  elif ( x==3 ):
      return 3
  elif (x == 4)|(x==5):
      return 4
  elif ( x==6 ):
      return 5
  else:
      return 6
df_main['Family_bin'] = df_main['Family_size'].apply(lambda x : family_bins(x)).astype(str)
f, axes = plt.subplots(1,3,figsize=(22,3))
sns.barplot(x='Family_bin',y='Survived',data=df_main[F_train],ci=None,ax=axes[0])
axes[0].set_title('Proba Survived')
sns.barplot(x='Family_bin',y='Survived',data=df_main[(F_train)&(df_main.Sex=='male')],ci=None,ax=axes[1])
axes[1].set_title('Proba Male Survived')
sns.barplot(x='Family_bin',y='Survived',data=df_main[(F_train)&(df_main.Sex=='female')],ci=None,ax=axes[2])
axes[2].set_title('Proba Female Survived')
for ax_cur in axes:
    for p in ax_cur.patches:
        ax_cur.annotate(format(p.get_height(),'.2f'),(p.get_x() + p.get_width()/1.5,p.get_height()),\
                        ha = 'center', va = 'center', xytext = (0, 5),textcoords ='offset points')
plt.show()
df_main.drop(columns=['SibSp','Parch','Family_size'],inplace=True)
df_main['Ticket_type'] = df_main.Ticket.str.replace(r'[^a-zA-Z]', '').str.upper().str[0].fillna('N')
df_main.Ticket_type.value_counts()
df_main[F_train][['Ticket_type','Survived']].groupby(['Ticket_type']).mean().reset_index().T
df_main.loc[df_main.Ticket_type=='N','Ticket_type'] = df_main[df_main.Ticket_type=='N'].\
        Ticket.apply(lambda x : 'N' + str(int(np.median([int(d) for d in str(x)]))))
# df_main.loc[df_main.Ticket_type=='N','Ticket_type'] = df_main[df_main.Ticket_type=='N'].Ticket.apply(lambda x : 'N' + str(len(str(x))))
df_main[F_train][['Ticket_type','Survived']].groupby(['Ticket_type']).mean().reset_index().T
f, axes = plt.subplots(1,3,figsize=(23,3))
sns.barplot(x='Ticket_type',y='Survived',data=df_main[F_train],ci=None,ax=axes[0])
axes[0].set_title('Proba Survived Ticket_type')
sns.barplot(x='Ticket_type',y='Survived',estimator=len,data=df_main[F_train],ax=axes[1],ci=None)
axes[1].set_title('Count Ticket_type')
sns.barplot(x='Ticket_type',y='Fare',estimator=np.median,data=df_main,ax=axes[2],ci=None)
axes[2].set_title('Median Fare Ticket_type')
for i,ax_cur in enumerate(axes):
    for p in ax_cur.patches:
        ax_cur.annotate(format(p.get_height(),'.2f'),(p.get_x() + p.get_width()/1.5,p.get_height()),\
                        ha = 'center', va = 'center', xytext = (0, 5),textcoords ='offset points')
plt.show()
df_visual = df_main.drop(columns=['PassengerId','Name','Ticket','Survived'])
def number_encode_features(init_df):
    result = init_df.copy() 
    encoders = {}
    for column in result.columns:
        if  result.dtypes[column] == np.object: 
            encoders[column] = LabelEncoder() 
            result[column] = encoders[column].fit_transform(result[column]) 
    return result, encoders
encoded_data, encoders = number_encode_features(df_visual)
df_visual =encoded_data.copy()

fig = plt.figure(figsize=(19,8))
cols = 5
rows = np.ceil(float(df_visual.shape[1])/cols)
for i, name_col in enumerate(df_visual.columns):
    ax = fig.add_subplot(rows,cols,i+1)
    ax.set_title(name_col)
    df_visual[name_col].hist(axes=ax,color='C'+str(i%cols))
    plt.xticks(rotation="vertical")
plt.subplots_adjust(hspace=0.7, wspace=0.3) 
del df_visual
df_corr = df_main[F_train].drop(columns=['PassengerId','Name','Ticket','Fare','Age'])
def number_encode_features(init_df):
    result = init_df.copy() 
    encoders = {}
    for column in result.columns:
        if  result.dtypes[column] == np.object: 
            encoders[column] = LabelEncoder() 
            result[column] = encoders[column].fit_transform(result[column]) 
    return result, encoders
encoded_data, encoders = number_encode_features(df_corr)
df_corr =encoded_data.copy()
plt.subplots(figsize=(10,10))
sns.heatmap(df_corr.corr(), square=True,linewidths=0.1,linecolor='black',vmax = .9,annot=True)
plt.show()
del df_corr
df_main.sample(3)
y_train = np.array(df_main.loc[F_train,'Survived'])

df_main['Fare_bin'] = df_main['Fare_bin'].astype(str)
df_main['Age_bin'] = df_main['Age_bin'].astype(str)
df_main['Family_bin'] = df_main['Family_bin'].astype(str)

df_model = df_main.drop(columns=['Name','Ticket','PassengerId','Survived','Fare','Age'])
df_model['Pclass'] = df_main['Pclass'].map({1:'Hight',2:'Middle',3:'Low'})

df_model.sample(3)
cat_columns = list(df_model.select_dtypes(include=['object','category']).columns)
digit_columns = set(df_model.columns) - set(cat_columns)

print(f'cat_columns:{cat_columns}')
print(f'digit_columns:{digit_columns}')
print('----------------------------------------------------------------------------')
df_onehot_only = pd.get_dummies(df_model[cat_columns], prefix_sep='_')
print(f'One hot shape {df_onehot_only.shape}')

df_model = pd.concat([df_model[digit_columns],df_onehot_only],axis=1)
print(f'Full shape {df_model.shape}')
print('----------------------------------------------------------------------------')
print(f'df_model.columns: {df_model.columns}')
df_model_train = df_model[F_train]
df_model_test = df_model[F_test]

X_train_model = np.array(df_model_train)
X_test_model = np.array(df_model_test)
print(f'X_train_model.shape={X_train_model.shape}')
print(f'X_test_model.shape={X_test_model.shape}')
# split by train & test
prc_split = 0.25
X_train_, X_test_, y_train_, y_test_ = train_test_split(X_train_model,y_train, test_size=prc_split)

print(f'X_train_.shape = {X_train_.shape}')
print(f'X_test_.shape = {X_test_.shape}')
# графики confusion matrix и ROC AUC
def draw_classification_metrics(y_test,y_predict,title_model):
  plt.rcParams['figure.figsize'] = (8, 3)

  # confusion matrix
  plt.subplot(1, 2, 1,aspect='equal')
  cm = metrics.confusion_matrix(y_test, y_predict.round())
  plt.title(str(title_model),fontsize=10)
  sns.heatmap(cm ,annot = True,fmt='g')

  # ROC AUC
  plt.subplot(1, 2, 2,aspect='equal')
  fpr, tpr, _ = metrics.roc_curve(y_test, y_predict)
  plt.plot([0, 1], [0, 1], 'k--')
  plt.plot(fpr, tpr,)
  plt.ylabel("True Positive Rate", fontsize=10)
  plt.xlabel("False Positive Rate", fontsize=10)
  plt.title('ROC AUC ' + str(title_model)+':' +str(round(metrics.auc(fpr, tpr),3)),fontsize=10)
  plt.show()

# график топ признаков 
def plot_feature_importances(model, columns):
  nr_f = 14
  imp = pd.Series(data = model.best_estimator_.feature_importances_, 
                  index=columns).sort_values(ascending=False)
  plt.figure(figsize=(7,5))
  plt.title("Feature importance")
  ax = sns.barplot(y=imp.index[:nr_f], x=imp.values[:nr_f], orient='h')
df_all_models = pd.DataFrame(columns=['Test_Train','Model','Accuracy'])
xgbm = xgb.XGBClassifier()
log_regr = linear_model.LogisticRegression()
GBC = GradientBoostingClassifier()
forest = RandomForestClassifier()
extra_tree = ExtraTreeClassifier()

estimator_list = [xgbm, log_regr,GBC,forest,extra_tree]
for i, cur_estimator in enumerate(estimator_list):
        clf_cur = cur_estimator.fit(X_train_,y_train_)
        cur_estimator_name = cur_estimator.__class__.__name__
        y_cur = clf_cur.predict(X_test_)
        y_cur_proba = clf_cur.predict_proba(X_test_)
        y_cur_train = clf_cur.predict(X_train_)
        accur_test = metrics.accuracy_score(y_test_,y_cur).round(3)
        accur_train = metrics.accuracy_score(y_train_,y_cur_train).round(3)
        print(f'{cur_estimator_name} ===============================')
        print(f'accuracy test={accur_test}, train={accur_train}')
        print(metrics.classification_report(y_test_,y_cur))
        draw_classification_metrics(y_test_,y_cur_proba[:,1],cur_estimator_name) 
        df_all_models.loc[len(df_all_models)] = ['Train',cur_estimator_name,accur_train]
        df_all_models.loc[len(df_all_models)] = ['Test',cur_estimator_name,accur_test]

f, ax = plt.subplots(1,1,figsize=(18,3))
sns.barplot(x='Model',y='Accuracy',hue='Test_Train',data=df_all_models,ax=ax)
ax.legend(loc=3)
for p in ax.patches:
        ax.annotate(format(p.get_height(),'.3f'),(p.get_x() + p.get_width()/1.5,p.get_height()),\
                    ha = 'center', va = 'center', xytext = (0, 5),textcoords ='offset points')
plt.show()
%%time
forest = RandomForestClassifier()
forest_grid={ 'criterion':['entropy'],  
             'max_depth':[10,11,12],
             'min_samples_split': [11,12,13,14],         
             'min_samples_leaf': [1],
             'max_features': [None]
              }
gs_forest = GridSearchCV(forest, forest_grid, cv=10,n_jobs=-1)
gs_forest.fit(X_train_, y_train_)

print(gs_forest.best_params_, gs_forest.best_score_)
forest_best_params = {'criterion': 'entropy', 'max_depth': 11, 'max_features': None, 'min_samples_leaf': 1, 'min_samples_split': 13}
forest = RandomForestClassifier(**forest_best_params)
clf_forest = forest.fit(X_train_,y_train_)

y_forest = clf_forest.predict(X_test_)
y_forest_proba = clf_forest.predict_proba(X_test_)
y_forest_train = clf_forest.predict(X_train_)

print(metrics.classification_report(y_test_,y_forest))
print(metrics.classification_report(y_train_,y_forest_train))
draw_classification_metrics(y_test_,y_forest_proba[:,1],'Forest (Random Tree)')  
plot_feature_importances(gs_forest, df_model_test.columns)
# %%time
# import warnings
# warnings.filterwarnings("ignore")

# log_reg = linear_model.LogisticRegression()
# log_reg_grid = {'penalty':['l1','l2'],'C':np.logspace(-3,1,5000)}

# gs_log =GridSearchCV(log_reg,log_reg_grid,cv=10,n_jobs=-1)
# gs_log.fit(X_train_,y_train_)

# print(gs_log.best_params_,gs_log.best_score_)
log_regr_best_params = {'C': 1.312875804833967, 'penalty': 'l2'}
log_regr = linear_model.LogisticRegression(**log_regr_best_params)
clf_log_regr = log_regr.fit(X_train_,y_train_)

y_log_regr = clf_log_regr.predict(X_test_)
y_log_regr_proba = clf_log_regr.predict_proba(X_test_)
y_log_regr_train = clf_log_regr.predict(X_train_)

print(metrics.classification_report(y_test_,y_log_regr))
print(metrics.classification_report(y_train_,y_log_regr_train))
draw_classification_metrics(y_test_,y_log_regr_proba[:,1],'Logical Regres.')  
%%time
xgbm = xgb.XGBClassifier()

xgbm_grid = {'max_depth':[4,5,6],
              "min_child_weight" : [3,4,5],
              "gamma" : [ 0.0,0.05, 0.1],
              'learning_rate':[0.01,0.1,0.05,0.06,0.07]
             }

gs_xgbm =GridSearchCV(xgbm,xgbm_grid,cv=10,n_jobs=-1)
gs_xgbm.fit(X_train_,y_train_)

print(gs_xgbm.best_params_,gs_xgbm.best_score_)
xgbm_best_params = {'gamma': 0.05, 'learning_rate': 0.06, 'max_depth': 5, 'min_child_weight': 4} 
xgbm = xgb.XGBClassifier(**xgbm_best_params)
clf_xgbm = xgbm.fit(X_train_,y_train_)

y_xgbm = clf_xgbm.predict(X_test_)
y_xgbm_proba = clf_xgbm.predict_proba(X_test_)
y_xgbm_train = clf_xgbm.predict(X_train_)


print(metrics.classification_report(y_test_,y_xgbm))
print(metrics.classification_report(y_train_,y_xgbm_train))
draw_classification_metrics(y_test_,y_xgbm_proba[:,1],'Xgboost') 
plot_feature_importances(gs_xgbm, df_model_train.columns)
%%time
GBC = GradientBoostingClassifier()
GBC_grid = { 
            'min_samples_leaf':[6],
            'max_depth':[7],
            'max_features':['log2'],
            'warm_start':[False],
            'learning_rate':[0.02],
            'n_estimators' :[100,300,500]
            }
gs_gbc = GridSearchCV(GBC, GBC_grid, cv=10)
gs_gbc.fit(X_train_,y_train_)

print(gs_gbc.best_params_, gs_gbc.best_score_)
GBC_best_params = {'learning_rate': 0.02, 'max_depth': 7, 'max_features': 'log2', 'min_samples_leaf': 6, 'warm_start': False,'n_estimators':1000} 

GBC = GradientBoostingClassifier(**GBC_best_params)
clf_GBC = GBC.fit(X_train_,y_train_)

y_GBC = clf_GBC.predict(X_test_)
y_GBC_proba = clf_GBC.predict_proba(X_test_)
y_GBC_train = clf_GBC.predict(X_train_)

print(metrics.classification_report(y_test_,y_GBC))
print(metrics.classification_report(y_train_,y_GBC_train))

draw_classification_metrics(y_test_,y_GBC_proba[:,1],'GBC')  
plot_feature_importances(gs_gbc, df_model_train.columns)
estimator_list = [GBC]
fig, axes = plt.subplots(1,len(estimator_list),figsize=(20,3))
if not isinstance(axes,np.ndarray):
    axes=[axes]

for i, cur_model in enumerate(estimator_list):
        cur_estimator_name = cur_model.__class__.__name__
        print(cur_estimator_name,':')
        y_predict = cur_model.predict(X_test_model)
        print('check predicts: ',y_predict[:10])

        submission = pd.DataFrame({'PassengerId':df_test['PassengerId'],'Survived':y_predict.astype(int)})
        print('check submissions: ',submission.loc[:9,'Survived'].ravel())
        now = datetime.datetime.now()

        submission['Survived'].value_counts().plot.pie(autopct='%1.2f%%', ax=axes[i],title = cur_estimator_name)
  
        submission.to_csv('submission_titanic_'+cur_estimator_name+'_'+now.strftime("%Y%m%d%H%M%S") +'.csv',index=False)
plt.show()