import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
data= pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

print('Dimensions: ',data.shape[0],'rows','x',data.shape[1],'columns')

data.head()
sns.set(font_scale=1.5)

sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white','figure.figsize':(10,5)})

sns.set_style("white")



tg_values=data['SARS-Cov-2 exam result'].value_counts()

tg_values.plot.barh(color=tuple(["r", "black"]))

plt.title('SARS-Cov-2 exam result')

print("Negative exam results: "+"{:.2%}".format(tg_values[0]/tg_values.sum())+' ('+str(tg_values[0])+' records)')

print("Positive exam results: "+"{:.2%}".format(tg_values[1]/tg_values.sum())+'  ('+str(tg_values[1])+' records)')

print('')
#Labeling encode the target variable

def positive_bin(x):

    if x == 'positive':

        return 1

    else:

        return 0

data['SARS-Cov-2 exam result_bin']=data['SARS-Cov-2 exam result'].map(positive_bin)
nulls=(data.isnull().sum()/len(data))*100

print('Percentage (%) of nulls for each feature:')

nulls.sort_values(ascending=False)

sns.set(font_scale=1.5)

sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white','figure.figsize':(8,5)})

sns.set_style("white")



pos=data[data['SARS-Cov-2 exam result_bin']==1]

neg=data[data['SARS-Cov-2 exam result_bin']==0]



nulls_neg=(neg.isnull().sum().sort_values(ascending=False)/len(neg))*100

nulls_pos=(pos.isnull().sum().sort_values(ascending=False)/len(pos))*100



ax=sns.distplot(nulls_pos[nulls_pos>0],color='red',bins=20,kde_kws={"color": "red", "label": "Positive"})

ax=sns.distplot(nulls_neg[nulls_neg>0],color='black',bins=20,kde_kws={"color": "black", "label": "Negative"})

# ax=sns.distplot(nulls_neg[nulls_neg>0],color='black',kde=False, norm_hist=False,bins=20) # histogram

ax.set(xlabel='% of Nulls',title='Features nulls KDE (% Nulls > 0)',label='3')

plt.grid(False)

plt.show()
ax=sns.distplot(nulls[nulls>0],color='blue',bins=20,kde_kws={"color": "blue", "label": "All Exam Results"})

ax.set(xlabel='% of Nulls',title='Variables Nulls KDE (% Nulls > 0)')

plt.grid(False)

plt.show()
variables_corr=nulls.loc[nulls<90].index.tolist()
corr = data[variables_corr].corr()['SARS-Cov-2 exam result_bin'].abs().sort_values(ascending=False)

corr
corr=data[variables_corr].corr().abs()



fig, ax = plt.subplots(figsize=(15, 15))

colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=colormap, annot=True, fmt='.2f')

plt.xticks(range(len(corr.columns)), corr.columns);

plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()
data['SARS-Cov-2 exam result_Baseline']=0

print("Baseline accuracy: "+"{:.2%}".format((data['SARS-Cov-2 exam result_Baseline']==data['SARS-Cov-2 exam result_bin']).sum()/len(data['SARS-Cov-2 exam result_Baseline'])))
nulls.drop(['SARS-Cov-2 exam result','Patient ID','SARS-Cov-2 exam result_bin'],inplace=True)
selecting_features=nulls.loc[nulls<90].index.tolist()

features=selecting_features

features.append('SARS-Cov-2 exam result_bin')
print(features)
df=data[features]



def bins(x):

    if x == 'detected' or x=='positive':

        return 1

    elif x=='not_detected' or x=='negative':

        return 0

    else:

        return x

    

for col in df.columns:

    df[col]=df[col].apply(lambda row: bins(row))
pd.set_option('display.max_columns', None)

df.describe()
from sklearn.impute import KNNImputer



X=df.drop(['SARS-Cov-2 exam result_bin'],axis=1)



temp = X

imputer = KNNImputer(n_neighbors=3)

temp = imputer.fit_transform(X.values)



X = pd.DataFrame(temp, columns=X.columns)

y = df['SARS-Cov-2 exam result_bin']

X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,random_state=5)

print('Train shape:',X_train.shape)

print('Test  shape:',X_test.shape)
from sklearn.ensemble import RandomForestClassifier



from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import log_loss, accuracy_score



resultados1=[]

kf=RepeatedKFold(n_splits=10, n_repeats=1, random_state=5)

for train,valid in kf.split(X_train):

    

    Xtr, Xvld = X_train.iloc[train], X_train.iloc[valid]

    ytr, yvld = y_train.iloc[train], y_train.iloc[valid]

    

    rf= RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)

    rf.fit(Xtr,ytr)

    

    p=rf.predict(Xvld)

    acc=accuracy_score(yvld,p)

    resultados1.append(acc)
from sklearn.linear_model import LogisticRegression



resultados2=[]

kf=RepeatedKFold(n_splits=10, n_repeats=1, random_state=5)

for train,valid in kf.split(X_train):

    

    Xtr, Xvld = X_train.iloc[train], X_train.iloc[valid]

    ytr, yvld = y_train.iloc[train], y_train.iloc[valid]

    

    lr= LogisticRegression(max_iter=50)

    lr.fit(Xtr,ytr)

    

    p=lr.predict(Xvld)

    acc=accuracy_score(yvld,p)

    resultados2.append(acc)
from sklearn.tree import DecisionTreeClassifier



resultados3=[]

kf=RepeatedKFold(n_splits=10, n_repeats=1, random_state=5)

for train,valid in kf.split(X_train):

    

    Xtr, Xvld = X_train.iloc[train], X_train.iloc[valid]

    ytr, yvld = y_train.iloc[train], y_train.iloc[valid]

    

    dt= DecisionTreeClassifier(random_state=3, max_depth=8)

    dt.fit(Xtr,ytr)

    

    p=dt.predict(Xvld)

    acc=accuracy_score(yvld,p)

    resultados3.append(acc)

print('')

p1=rf.predict(X_test)

p1[:]=rf.predict(X_test)

acc=accuracy_score(y_test,p1)

print("Vanilla Random Forest          Test accuracy: "+"{:.2%}".format(acc)+' / Train accuracy: '+"{:.2%}".format(np.mean(resultados1)))

p2=lr.predict(X_test)

p2[:]=lr.predict(X_test)

acc=accuracy_score(y_test,p2)

print("Vanilla Logistic Regression    Test accuracy: "+"{:.2%}".format(acc)+' / Train accuracy: '+"{:.2%}".format(np.mean(resultados2)))

p3=dt.predict(X_test)

p3[:]=dt.predict(X_test)

acc=accuracy_score(y_test,p3)

print("Vanilla Decision Tree          Test accuracy: "+"{:.2%}".format(acc)+' / Train accuracy: '+"{:.2%}".format(np.mean(resultados3)))

print('')
visual=pd.concat([X_test,y_test],axis=1)

visual['predict']=p2

visual2=visual[visual['SARS-Cov-2 exam result_bin']==visual['predict']]



print('Positive results in the test sample: ',visual[visual['SARS-Cov-2 exam result_bin']==1].shape[0])

print('Positive results correctly predicted: ',visual2[visual2['predict']==1].shape[0])

print('Only positives accuracy: ',"{:.2%}".format(visual2[visual2['predict']==1].shape[0]/visual[visual['SARS-Cov-2 exam result_bin']==1].shape[0]))
sns.set(font_scale=1.5)

sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white','figure.figsize':(8,5)})

sns.set_style("white")



pred_prob=lr.predict_proba(X_test)



from sklearn.metrics import precision_recall_curve,roc_curve



scores=pred_prob[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, scores)



plt.rcParams["axes.grid"] = True



plt.plot([1,0],[0,1],linestyle = '--',lw = 2,color = 'grey')

plt.plot(recall[:-1],precision[:-1],label='Logistic Regression', color='red',lw=2, alpha=1)

plt.xlim(0, 1)

plt.ylim(0, 1)

plt.grid(True)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision - Recall curve')

plt.legend(loc="upper right")

plt.show()



plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'grey')

scores=pred_prob[:,1]

fpr, tpr, thresholds = roc_curve(y_test,scores)

plt.plot(fpr,tpr,label='Logistic Regression', color='black',lw=2, alpha=1)

plt.xlim(0, 1)

plt.ylim(0, 1)

plt.grid(True)

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc="lower right")

plt.show()
from sklearn.metrics import plot_confusion_matrix



disp = plot_confusion_matrix(lr, X_test, y_test,display_labels=('Negative','Positive'),cmap=plt.cm.Reds,values_format='.00f')

disp.ax_.set_title('Confusion Matrix - Exam Results')

disp.ax_.grid(False)
from imblearn.over_sampling import SMOTE



sm = SMOTE(random_state=2,n_jobs=-1,sampling_strategy=1)

X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
resultados3=[]

kf=RepeatedKFold(n_splits=10, n_repeats=1, random_state=5)

for train,valid in kf.split(X_train_res):

    

    Xtr, Xvld = X_train_res.iloc[train], X_train_res.iloc[valid]

    ytr, yvld = y_train_res.iloc[train], y_train_res.iloc[valid]

    

    rf_os= RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)

    rf_os.fit(Xtr,ytr)

    

    p=rf_os.predict(Xvld)

    acc=accuracy_score(yvld,p)

    resultados3.append(acc)
from imblearn.under_sampling import RandomUnderSampler



rus = RandomUnderSampler(random_state=27,sampling_strategy=1)

X_train_res2, y_train_res2 = rus.fit_resample(X_train, y_train)
resultados3=[]

kf=RepeatedKFold(n_splits=10, n_repeats=1, random_state=5)

for train,valid in kf.split(X_train_res2):

    

    Xtr, Xvld = X_train_res2.iloc[train], X_train_res2.iloc[valid]

    ytr, yvld = y_train_res2.iloc[train], y_train_res2.iloc[valid]

    

    rf_us= RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)

    rf_us.fit(Xtr,ytr)

    

    p=rf_us.predict(Xvld)

    acc=accuracy_score(yvld,p)

    resultados3.append(acc)

p1=rf_us.predict(X_test)

p1[:]=rf_us.predict(X_test)

acc=accuracy_score(y_test,p1)
disp = plot_confusion_matrix(rf_us, X_test, y_test,display_labels=('Negative','Positive'),cmap=plt.cm.Reds,values_format='.00f')

disp.ax_.set_title('UnderSampling Confusion Matrix - Exam Results')

disp.ax_.grid(False)



disp = plot_confusion_matrix(rf_os, X_test, y_test,display_labels=('Negative','Positive'),cmap=plt.cm.Reds,values_format='.00f')

disp.ax_.set_title('OverSampling Confusion Matrix - Exam Results')

disp.ax_.grid(False)
pred_prob=rf_us.predict_proba(X_test)



scores=pred_prob[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, scores)



plt.rcParams["axes.grid"] = True



plt.plot([1,0],[0,1],linestyle = '--',lw = 2,color = 'grey')

plt.plot(recall[:-1],precision[:-1],label='Random Forest + Random Undersampling', color='red',lw=2, alpha=1)

plt.xlim(0, 1)

plt.ylim(0, 1)

plt.grid(True)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision Recall curve')

plt.legend(loc="upper right")

plt.show()



pred_prob=rf_os.predict_proba(X_test)



scores=pred_prob[:,1]

precision, recall, thresholds = precision_recall_curve(y_test, scores)



plt.plot([1,0],[0,1],linestyle = '--',lw = 2,color = 'grey')

plt.plot(recall[:-1],precision[:-1],label='Random Forest + SMOTE Oversampling', color='black',lw=2, alpha=1)

plt.xlim(0, 1)

plt.ylim(0, 1)

plt.grid(True)

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.title('Precision Recall curve')

plt.legend(loc="upper right")

plt.show()
from mlxtend.feature_selection import SequentialFeatureSelector as sfs



model=sfs(RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0),k_features=8,forward=False,verbose=2,cv=10,n_jobs=-1,scoring='accuracy')

model.fit(X_train_res2,y_train_res2)
var=list(model.k_feature_names_)

var
X_train_res3=X_train_res2[var]

y_train_res3=y_train_res2



resultados3=[]

kf=RepeatedKFold(n_splits=10, n_repeats=1, random_state=5)

for train,valid in kf.split(X_train_res3):

    

    Xtr, Xvld = X_train_res3.iloc[train], X_train_res3.iloc[valid]

    ytr, yvld = y_train_res3.iloc[train], y_train_res3.iloc[valid]

    

    dt= RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)

    dt.fit(Xtr,ytr)

    

    p=dt.predict(Xvld)

    acc=accuracy_score(yvld,p)

    resultados3.append(acc)
disp = plot_confusion_matrix(dt, X_test[var], y_test,display_labels=('Negative','Positive'),cmap=plt.cm.Reds,values_format='.00f')

disp.ax_.set_title('Confusion Matrix - Exam Results')

disp.ax_.grid(False)