from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import binarize

from sklearn import metrics

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier

from sklearn import preprocessing

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn import svm

from sklearn import tree

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report , confusion_matrix

from sklearn.model_selection import cross_val_score

import numpy as np

import seaborn as sns

import matplotlib.pyplot as  plt

from sklearn.model_selection import train_test_split

from scipy import stats

from matplotlib import rc

import collections
df = pd.read_csv('../input/HR_comma_sep.csv')

da =df[df['left']==1]

df.describe()
da = df[df['left']==1]

da.describe()
def Histogram(data,to_plot):

    for i in range(len(to_plot)):

        plt.hist(data[to_plot[i]])

        plt.axvline(data[to_plot[i]].mean(),color='r')

        plt.xlabel(to_plot[i])

        plt.show()
to_plt =['satisfaction_level','last_evaluation','average_montly_hours']

Histogram(df,to_plt)
sns.set(font_scale=2)

fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(20,6))

asx = df[df['left']==1]

sns.kdeplot(data=asx['last_evaluation'],color='b',ax=axs[0],shade=True,label='left')

axs[0].set_xlabel('last_evaluation')

asd = df[df['left']==0]

sns.kdeplot(data=asd['last_evaluation'],color='g',ax=axs[0],shade=True,label='stayed')

sns.countplot(x=asx['sales'],ax=axs[1],label='Department wise attrition')

plt.xticks(rotation=90)

plt.tight_layout()

plt.show()
sns.set(font_scale=1.5)

plt.figure(figsize=(12,10))

sns.heatmap(df.corr(),annot=True)

plt.xticks(rotation=90)

plt.show()
sns.set(color_codes=True)

plot = sns.FacetGrid(df,col='left',hue='left',size=5)

plot.map(sns.kdeplot,'satisfaction_level','last_evaluation',shade=True,cmap='Blues')

plt.show()

# dt is the data of all the employees in the third group

dt=da[[all([a,b]) for a,b in zip(da['last_evaluation'] > df['last_evaluation'].mean(),da['satisfaction_level']>df['satisfaction_level'].mean())]]

# dl is the data of all the employees who had low salaries in the above group

dl = dt[dt['salary']=='low']

sns.countplot(dl['sales'])

plt.xticks(rotation=90)

plt.show()
depts = df['sales'].unique()

avgs =[]

avgl =[]

avgn =[]

avgm =[]

for i in depts:

    mean = df['satisfaction_level'][df['sales']==i].mean()

    avgs.append(mean)       

for i in depts:

    mean = df['last_evaluation'][df['sales']==i].mean()

    avgl.append(mean)       

for i in depts:

    mean = df['number_project'][df['sales']==i].mean()

    avgn.append(mean)

for i in depts:

    mean = df['average_montly_hours'][df['sales']==i].mean()

    avgm.append(mean)

averages=pd.DataFrame({'Depts': depts,'AVGS':avgs,'AVGL':avgl,'AVGN':avgn,'AVGM':avgm},index=None)
sns.set(style="whitegrid",font_scale=2)

q = sns.PairGrid(averages.sort_values('AVGS',ascending=False),y_vars='Depts',x_vars=['AVGS','AVGN','AVGM','AVGL'],size=12,aspect=0.5)

q.map(sns.stripplot,orient='h',palette="Reds_r", edgecolor="gray",size=30,)

titles = ["Last Evaluation", "Satisfaction", "Number of Projects",'Average Monthly Hours']

plt.title('Department wise performance based on our KPI')

for ax, title in zip(q.axes.flat, titles):



    # Set a different title for each axes

    ax.set(title=title)



    # Make the grid horizontal instead of vertical

    ax.xaxis.grid(False)

    ax.yaxis.grid(True)

plt.show()
ds = df[df['left']==1]

x_vars=['satisfaction_level','last_evaluation','number_project']

depts = df['sales'].unique()

lavgs =[]

lavgl =[]

lavgn =[]

lavgm =[]

for i in depts:

    mean = ds['satisfaction_level'][da['sales']==i].mean()

    lavgs.append(mean)       

for i in depts:

    mean = ds['last_evaluation'][da['sales']==i].mean()

    lavgl.append(mean)       

for i in depts:

    mean = ds['number_project'][da['sales']==i].mean()

    lavgn.append(mean)       

for i in depts:

    mean = ds['average_montly_hours'][da['sales']==i].mean()

    lavgm.append(mean)

Laverages=pd.DataFrame({'Depts': depts,'AVGS':lavgs,'AVGL':lavgl,'AVGN':lavgn,'AVGM':lavgm},index=None)
sns.set(style="whitegrid",font_scale=3)

lq = sns.PairGrid(Laverages.sort_values('AVGL',ascending=False),y_vars='Depts',x_vars=['AVGL','AVGS','AVGN','AVGM'],size=12,aspect=0.5)

lq.map(sns.stripplot,orient='h',palette="Reds_r", edgecolor="black",size=30)

titles = ["Last Evaluation", "Satisfaction", "Number of Projects",'Average Monthly Hours']

plt.title('Department wise performance based on our KPI')

for ax, title in zip(lq.axes.flat, titles):



#     Set a different title for each axes

    ax.set(title=title)



    # Make the grid horizontal instead of vertical

    ax.xaxis.grid(False)

    ax.yaxis.grid(True)

plt.show()
X = df.drop(['left'],axis=1)

y = df['left']

le = LabelEncoder()

X['salary']= le.fit_transform(X['salary'])

X['sales']= le.fit_transform(X['sales'])
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=2,stratify=y)

# Creating a seperate scaled set to be used in some models to improve our results

Xscaled_train = pd.DataFrame(preprocessing.scale(X_train.values))

Xscaled_test  =  pd.DataFrame(preprocessing.scale(X_test.values))
gnb = GaussianNB() # Gaussian Naive Bayes

gnb.fit(Xscaled_train,y_train)

gnbpred = gnb.predict_proba(Xscaled_test)

print (roc_auc_score(y_test,gnbpred[:,1]))
def params_tuning(model,X_train,y_train,X_test,y_test,metrics,param_grid,clf=False,conf=False):

    """Tune parameters of the model using a grid search, this function just makes the job easier."""

    gs=GridSearchCV(model,param_grid=param_grid,scoring=metrics,cv=10)

    gs.fit(X_train,y_train)

    predicted = gs.predict(X_test)

    proba = gs.predict_proba(X_test)[:,1]

    if clf == True:

        print (classification_report(y_test,predicted))

    if conf == True:

        print (confusion_matrix(y_test,predicted))

    print (gs.best_params_)

    print (roc_auc_score(y_test,proba),'Optimised Score')
knn = KNeighborsClassifier()

knn.fit(X_train,y_train)

knnp = knn.predict_proba(X_test)

print (roc_auc_score(y_test,knnp[:,1]),"Initial Score")

weight_options = ['uniform','distance']

params_grid_knn = dict(n_neighbors = range(1,18) ,weights=weight_options)

params_tuning(knn,X_train,y_train,X_test,y_test,'roc_auc',params_grid_knn)
svc = svm.SVC(probability=True,random_state=12)

svc.fit(Xscaled_train,y_train)

scaledp = svc.predict_proba(Xscaled_test)

print (roc_auc_score(y_test,scaledp[:,1]),'Initial Score')

svcp = dict(C=np.linspace(0.1,1,5))

params_tuning(svc,Xscaled_train,y_train,Xscaled_test,y_test,'roc_auc',svcp)

logreg = LogisticRegression(random_state=12)

logreg.fit(Xscaled_train,y_train)

logp = logreg.predict_proba(Xscaled_test)

print (roc_auc_score(y_test,logp[:,1]),"Inital Score")

logp = dict(C=np.linspace(0.16,0.2,5),solver=['newton-cg', 'lbfgs','sag'])

params_tuning(logreg,Xscaled_train,y_train,Xscaled_test,y_test,'roc_auc',logp,clf=True)
logreg= LogisticRegression(C=0.17,solver='sag')

logreg.fit(Xscaled_train,y_train)

logregprob = logreg.predict(Xscaled_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, logregprob)

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.show()
def evaluate_threshold(threshold):

    """returns sensitivity and specifity for a given threshold value"""

    print('Sensitivity:', tpr[thresholds > threshold][-1])

    print('Specificity:', 1 - fpr[thresholds > threshold][-1])
tr = tree.DecisionTreeClassifier(class_weight='balanced',random_state=12)

tr.fit(X_train,y_train)

predtree = tr.predict(X_test)

print (roc_auc_score(y_test,predtree))
def importance_plotting(data,x,y,palette,title):

    sns.set(style="whitegrid")

    ft = sns.PairGrid(data,y_vars=y,x_vars=x,size=5,aspect=1)

    ft.map(sns.stripplot,orient='h',palette=palette, edgecolor="black",size=15)

    for ax, title in zip(ft.axes.flat, titles):

    # Set a different title for each axes

        ax.set(title=title)

    # Make the grid horizontal instead of vertical

        ax.xaxis.grid(False)

        ax.yaxis.grid(True)

    plt.show()
fo = {'Features':df.drop('left',axis=1).columns.tolist(),'Importance':tr.feature_importances_}

Importance = pd.DataFrame(fo,index=None).sort_values('Importance',ascending=False)

titles = ["Importance of the various Features in predicting the outcome"]

importance_plotting(Importance,'Importance','Features','Greens_r',titles)
rf = RandomForestClassifier() #Random Forest

rf.fit(X_train,y_train)

rfpred=rf.predict(X_test)

rfp = dict(n_estimators=np.arange(5,25,10))

print (roc_auc_score(y_test,rfpred),'Initial Score')

params_tuning(rf,X_train,y_train,X_test,y_test,'roc_auc',rfp)
ho = {'Features':df.drop('left',axis=1).columns.tolist(),'Importance':rf.feature_importances_}

ImportanceRF = pd.DataFrame(ho,index=None).sort_values('Importance',ascending=False)

importance_plotting(ImportanceRF,'Importance','Features','Greens_r',titles)
ada = AdaBoostClassifier(algorithm='SAMME')

ada.fit(Xscaled_train,y_train)

adaproba= ada.predict_proba(Xscaled_test)

print (roc_auc_score(y_test,adaproba[:,1]),'Initial Score')

adap = dict(n_estimators=[25,50,75],learning_rate =[0.25,0.5,0.75,1])

params_tuning(ada,Xscaled_train,y_train,Xscaled_test,y_test,'roc_auc',adap,clf=True)
fpr, tpr, thresholds = metrics.roc_curve(y_test,adaproba[:,1])

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.show()
evaluate_threshold(0.479)

# We decrease the threshold to increase sensitivity and I reached this value through simple trial and error.

binned = (binarize(adaproba[:,1].reshape(-1,1),0.479))

print (classification_report(y_test,binned))
gbr = GradientBoostingClassifier()

gbr.fit(Xscaled_train,y_train)

gbrp = gbr.predict_proba(Xscaled_test)

print (roc_auc_score(y_test,gbrp[:,1]),'Initial Score')
# gbr = GradientBoostingClassifier(min_samples_leaf=10,max_features='sqrt',n_estimators=82,min_samples_split=400,max_depth=15,subsample=0.85,random_state=12)

# We start by training model specific parameter n_estimators

gbr = GradientBoostingClassifier(min_samples_split=75,max_depth=8,min_samples_leaf=50,max_features='sqrt',subsample=0.8,random_state=12)

gbrd = dict(n_estimators=range(20,81,10))

params_tuning(gbr,Xscaled_train,y_train,Xscaled_test,y_test,'roc_auc',gbrd)

### tree specific parameters

gbr = GradientBoostingClassifier(n_estimators=70,min_samples_split=75,max_depth=8,min_samples_leaf=50,max_features='sqrt',subsample=0.8,random_state=12)

gbrd = dict(max_depth=range(5,16,2),min_samples_split = range(15,80,10))

params_tuning(gbr,Xscaled_train,y_train,Xscaled_test,y_test,'roc_auc',gbrd)

# subsample

gbr = GradientBoostingClassifier(min_samples_split=15,max_depth=15,n_estimators=70,min_samples_leaf=50,max_features='sqrt',random_state=12)

gbrd = dict(subsample=[0.75,0.8,0.85,0.9])

params_tuning(gbr,Xscaled_train,y_train,Xscaled_test,y_test,'roc_auc',gbrd)

scores_list= pd.DataFrame(collections.OrderedDict([('Model',['Gaussian Naive Bayes','K Neighbors','Support Vector Machine','Logistic Regression','Decision Tree','Random Forest','Adaptive Boosting','Gradient Boosting']),('AUC_Score',[0.848839820268,0.977592568086,0.974202826362,0.802274102992,0.972431754854,0.988187674457,0.958517593389,0.992851230606])])) 

print (scores_list.sort_values('AUC_Score',ascending=False))