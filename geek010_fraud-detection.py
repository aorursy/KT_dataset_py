##author : Ahmet Turkmen 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
import plotly 
import xgboost as xgb 
import lightgbm as lgb 
import gc as memory_free
from skopt import BayesSearchCV 
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from matplotlib.colors import ListedColormap
from scipy import interp
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier 
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
import matplotlib.patches as patches
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score,auc

def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
def print_status(optimum_result):
    models = pd.DataFrame(bayes_cv_hyper_tuning.cv_results_)
    best_parameters = pd.Series(bayes_cv_hyper_tuning.best_params_)
    print ('Model {}\n Best ROC-AUC: {}\n best parameters: {}\n'.format(len(models),np.round(bayes_cv_hyper_tuning.best_score_,4),bayes_cv_hyper_tuning.best_params_))
## Preparing data in order to minimize memory-usage
## Datapreprocessing
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'}

df  = pd.read_csv('../input/train_sample.csv',dtype=dtypes)


## train dataset is used because since it is a competition and we want to make class project by evaluating performance 
## of different classification algoritms on data by making some manipulation on data, we used train set by splitting it 
## %30 > test , %70 train. 
## 'is_attributed'  our label. 

## check null values in colunms 
print(df.isnull().sum())
print(df.shape[0])
print('total number of null values in attributed_time feature is {} '.format(df.isnull().sum()['attributed_time']))
## it might be good idea to remove attributed_time coloumn all, it might create noisy on data. 

print('Extracting new features...')
df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')

memory_free.collect()

print('grouping by ip-day-hour combination...')
gp = df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
df = df.merge(gp, on=['ip','day','hour'], how='left')
del gp
memory_free.collect()

print('grouping by ip-app combination...')
gp = df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
df = df.merge(gp, on=['ip','app'], how='left')
del gp
memory_free.collect()


print('grouping by ip-app-os combination...')
gp = df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
df = df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
memory_free.collect()


# Adding features with var and mean hour (inspired from nuhsikander's script)
print('grouping by : ip_day_chl_var_hour')
gp = df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
df = df.merge(gp, on=['ip','day','channel'], how='left')
del gp
memory_free.collect()

print('grouping by : ip_app_os_var_hour')
gp = df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
df = df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
memory_free.collect()

print('grouping by : ip_app_channel_var_day')
gp = df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
df = df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
memory_free.collect()

print('grouping by : ip_app_chl_mean_hour')
gp = df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
print("merging...")
train_df = df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
memory_free.collect()

print("vars and data type: ")

df=df.fillna(0)
df['ip_tcount'] = df['ip_tcount'].astype('uint16')
df['ip_app_count'] = df['ip_app_count'].astype('uint16')
df['ip_app_os_count'] = df['ip_app_os_count'].astype('uint16')
df['ip_tchan_count']=df['ip_tchan_count'].astype('uint32')
df['ip_app_os_var']=df['ip_app_os_var'].astype('uint32')
df['ip_app_channel_var_day']=df['ip_app_channel_var_day'].astype('uint32')
df.info()

## check number of label, is it balanced or unbalanced data. 
label_dist=df.is_attributed.value_counts()
print('Proportion:', round(label_dist[1] / label_dist[0], 5), ': 1')
print(df.is_attributed.value_counts())
label_dist.plot(kind='bar', title='Count (target)');
def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)
bayes_cv_hyper_tuning = BayesSearchCV(
    estimator  = xgb.XGBClassifier(
        n_jobs=1,
        objective='binary:logistic',
        eval_metric='auc',
        silent=1,
        tree_method = 'approx'
    ),
    search_spaces={
        'learning_rate':(0.01,1.0,'log-uniform'),
        'min_child_weight':(0,10),
        'max_depth':(0,50),
        'max_delta_step':(0,20),
        'subsample':(0.01,1.0,'uniform'),
        'n_estimators':(50,100),
        'scale_pos_weight':(1e-6,500,'log-uniform')
    },
    scoring = 'roc_auc',
    cv = StratifiedKFold(n_splits = 5, shuffle=True, random_state=42),
    n_jobs = 3,
    n_iter=10,
    verbose=0,
    refit=True,
    random_state=42
)
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
def plot_roc_graph(df,classifier,class_name):
    from sklearn.metrics import classification_report 
    scaler = StandardScaler()
    cv = StratifiedKFold(n_splits = 10, shuffle=True, random_state=42)
    x = df.loc[:, df.columns != 'is_attributed']
    y = df.loc[:,'is_attributed']
    x = scaler.fit_transform(x)
    dtype = [('ip','uint32'), ('app','uint16'), ('device','uint16'),('os','uint16'),('channel','uint16')]
    index = ['Row'+str(i) for i in range(1, len(x)+1)]
    x = pd.DataFrame(x, index=index)
    # plot arrows
    fig1 = plt.figure(figsize=[12,12])
    ax1 = fig1.add_subplot(111,aspect = 'equal')
    ax1.add_patch(patches.Arrow(0.45,0.5,-0.25,0.25,width=0.3,color='green',alpha = 0.5))
    ax1.add_patch(patches.Arrow(0.5,0.45,0.25,-0.25,width=0.3,color='red',alpha = 0.5))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)
    i = 1
    for train,test in cv.split(x,y):
        prediction = classifier.fit(x.iloc[train],y.iloc[train]).predict_proba(x.iloc[test])
        fpr, tpr, t = roc_curve(y[test], prediction[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i= i+1
     
    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue',
             label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC - '+class_name)
    plt.legend(loc="lower right")
    plt.text(0.32,0.7,'More accurate area',fontsize = 12)
    plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
    plt.show()


df=df.drop(['click_time','attributed_time'],axis=1)
features=df.drop(['is_attributed'],axis=1).columns 
# scaler = StandardScaler()
X_org=df[features]
y_org=df['is_attributed']
# X_org = scaler.fit_transform(X_org)
X_train,X_test, y_train,y_test=train_test_split(X_org,y_org,test_size=0.3,random_state=1)

model = XGBClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
probs=model.predict_proba(X_test)
accu = accuracy_score(y_test,y_pred)
print('Accuracy : %.2f%%' % (accu *100.0))
from sklearn.metrics import confusion_matrix
conf_mat=confusion_matrix(y_test,y_pred)
print(y_test.value_counts())
labels = ['Not Fraudulent', 'Fraudulent']
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_mat,labels, title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_mat, classes=labels, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

print(df.is_attributed.value_counts())
df.is_attributed.value_counts().plot(kind='pie',title='Distribution of data')
features=df.drop(['is_attributed'],axis=1).columns 
X_org=df[features]
y_org=df['is_attributed']
X_train,X_test, y_train,y_test=train_test_split(X_org,y_org,test_size=0.3,random_state=1)
decision_tree_classifer = DecisionTreeClassifier(criterion='gini')

decision_tree_classifer.fit(X_train,y_train)
y_pred=decision_tree_classifer.predict(X_test)
probs=decision_tree_classifer.predict_proba(X_test)
accu = accuracy_score(y_test,y_pred)
print(y_test.value_counts())
labels = ['Not Fraudulent', 'Fraudulent']

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labels,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                      title='Normalized confusion matrix')

plt.show()




plot_roc_graph(df,decision_tree_classifer,'DecisionTreeClassifier')
from sklearn.naive_bayes import GaussianNB
naive_bayes_gaussian =  GaussianNB()
plot_roc_graph(df,naive_bayes_gaussian,'GaussianNB')
random_forest = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
plot_roc_graph(df,random_forest,'RandomForest')
X = df.loc[:, df.columns != 'is_attributed']
y = df.loc[:,'is_attributed']
bayes_cv_hyper_tuning.fit(X.values,y.values, callback=print_status)
not_fraudulent = df[df['is_attributed']==0]
fraudulent  = df[df['is_attributed']==1]
not_fraudulent_under = not_fraudulent.sample(label_dist[1])
df_undered=pd.concat([not_fraudulent_under,fraudulent],axis=0)
print('Randomly under-sampled:\n{}'.format(df_undered.is_attributed.value_counts()))
df_undered.is_attributed.value_counts().plot(kind='pie',title='Dist. of resampled data')

df_undered=df_undered.reset_index()
df_undered=df_undered.drop(['index'],axis=1)
plot_roc_graph(df_undered,random_forest,'RandomForest > UnderSampled')
df_undered.columns

fraudulent_over = fraudulent.sample(label_dist[0],replace=True)
df_over=pd.concat([not_fraudulent,fraudulent_over],axis=0)
print('Randomly under-sampled:\n{}'.format(df_over.is_attributed.value_counts()))
df_over.is_attributed.value_counts().plot(kind='bar',title='Dist. of resampled data')
df_over = df_over.reset_index()
df_over = df_over.drop(['index'],axis=1)
plot_roc_graph(df_over,naive_bayes_gaussian,'Naive Bayes - Oversampling')
X_under=df_undered[features]
y_under=df_undered['is_attributed']
bayes_cv_hyper_tuning.fit(X_under.values,y_under.values, callback=print_status)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_org = pca.fit_transform(X_org)
plot_2d_space(X_org, y_org, 'Imbalanced dataset (2 PCA components)')
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(return_indices=True)
X_rus, y_rus, id_rus = rus.fit_sample(X_org, y_org)

print('Removed indexes:', id_rus)

plot_2d_space(X_rus, y_rus, 'Random under-sampling')
from imblearn.over_sampling import SMOTE
smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(X_org, y_org)
plot_2d_space(X_sm, y_sm, 'SMOTE over-sampling')

df_train_all = pd.read_csv('../input/train.csv',nrows=37000000,dtype=dtypes)
df_train_all.columns
fraudulant_addition=df_train_all[df_train_all['is_attributed']==1]
fraudulant_addition=fraudulant_addition.drop(['click_time','attributed_time'],axis=1)
df_=pd.concat([fraudulant_addition,df])
df_.is_attributed.value_counts().plot(kind='bar',title='regenerated dataset')
df_=df_.reset_index()
df_=df_.drop(['index'],axis=1)

X_regenerated = df_[features]
y_regenerated = df_['is_attributed']
X_train,X_test, y_train,y_test=train_test_split(X_regenerated,y_regenerated,test_size=0.3,random_state=1)
bayes_cv_hyper_tuning.fit(X_train.values,y_train.values, callback=print_status)

