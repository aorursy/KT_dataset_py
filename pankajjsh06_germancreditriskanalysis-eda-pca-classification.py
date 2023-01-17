import numpy as np

import pandas as pd 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

import seaborn as sns               # Provides a high level interface for drawing attractive and informative statistical graphics

%matplotlib inline

sns.set()

from subprocess import check_output



import warnings                                            # Ignore warning related to pandas_profiling

warnings.filterwarnings('ignore') 



def annot_plot_num(ax,w,h):                                    # function to add data to plot

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    for p in ax.patches:

        ax.annotate('{0:.1f}'.format(p.get_height()), (p.get_x()+w, p.get_height()+h))

        

def annot_plot(ax,w,h):                                    # function to add data to plot

    ax.spines['top'].set_visible(False)

    ax.spines['right'].set_visible(False)

    for p in ax.patches:

         ax.annotate(f"{p.get_height() * 100 / df.shape[0]:.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),

         ha='center', va='center', fontsize=11, color='black', rotation=0, xytext=(0, 10),

         textcoords='offset points')



import os

print(os.listdir("../input"))
df = pd.read_csv('../input/german_credit_data.csv')
print('**********Shape of the Dataset.*******************************************')

print(df.shape)

print('**********Column names******************************************************')

print(df.columns)

print('**********Total number of null values in each column.**********************')

print(df.isnull().sum())

print('**********Total number of unique values in each column*********************')

print(df.nunique())

df.head()
df.rename(columns=lambda x:x.replace('Unnamed: 0','id'), inplace = True )
df.info()
df['Checking account'].fillna('no-info', inplace = True)

df['Saving accounts'].fillna('no-info', inplace = True)
ax = sns.countplot('Risk', data = df)

plt.ylabel('Total number of credit holders.')

annot_plot_num(ax,0.08,1)

plt.show()
ax = sns.countplot('Sex',hue='Risk', data = df)

plt.ylabel('Total number of credit holders.')

annot_plot(ax,0.08,1)

plt.show()
plt.figure(figsize=(12,7))

ax = sns.countplot('Duration',hue='Risk', data = df)

plt.ylabel('Total number of credit holders.')

annot_plot(ax,0.08,1)

plt.show()
gender_df = df.groupby(['Sex','Risk'])['Purpose'].value_counts()

gender_df
plt.figure(figsize=(12,7))

ax = sns.countplot('Sex',hue='Job', data = df)

plt.ylabel('Total number of credit holders.')

annot_plot_num(ax,0.008,1)

plt.show()
plt.show()

plt.figure(figsize=(12,7))

ax = sns.countplot('Risk',hue='Job', data = df)

plt.ylabel('Total number of credit holders.')

annot_plot_num(ax,0.008,1)

plt.show()
plt.figure(figsize=(12,7))

ax = sns.countplot('Risk',hue='Housing', data = df)

plt.ylabel('Total number of credit holders.')

annot_plot_num(ax,0.008,1)

plt.show()
gender_df = df.groupby(['Purpose','Risk'])['Sex'].value_counts()

gender_df
purpose_group = gender_df.groupby('Purpose')

fig = plt.figure()

count =  1



for gender, group in purpose_group:

    ax = fig.add_subplot(2,4,count)

    ax.set_title(gender)

    ax = group[gender].plot.bar(figsize = (10,5), width = 0.8)

    

    count+=1

    

    plt.xlabel('')

    plt.yticks([])

    plt.ylabel('Holders')

    

    total_of_holders = []

    for i in ax.patches:

        total_of_holders.append(i.get_height())

        total = sum(total_of_holders)

    for i in ax.patches:

         ax.text(i.get_x()+0.2, i.get_height()-1.5,s= i.get_height(),color="black",fontweight='bold')

plt.tight_layout()

plt.show()
job_df = df.groupby(['Job','Sex'])['Risk'].value_counts()

job_group = job_df.groupby('Job')



fig = plt.figure()

count =  1



for gender, group in job_group:

    ax = fig.add_subplot(2,4,count)

    ax.set_title(gender)

    ax = group[gender].plot.bar(figsize = (10,5), width = 0.8)

    

    count+=1

    

    plt.xlabel('')

    plt.yticks([])

    plt.ylabel('Holders')

    

    total_of_holders = []

    for i in ax.patches:

        total_of_holders.append(i.get_height())

        total = sum(total_of_holders)

    for i in ax.patches:

         ax.text(i.get_x()+0.2, i.get_height()-1.5,s= i.get_height(),color="black",fontweight='bold')

plt.tight_layout()

plt.show()
sns.pairplot(data = df)

plt.show()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



df['Sex'] = le.fit_transform(df['Sex'])

df['Housing'] = le.fit_transform(df['Housing'])

df['Purpose'] = le.fit_transform(df['Purpose'])

df['Risk'] = le.fit_transform(df['Risk'])

df['Saving accounts'] = le.fit_transform(df['Saving accounts'])

df['Checking account'] = le.fit_transform(df['Checking account'])

df.head(2)
sns.countplot('Risk', data = df)

plt.show()
corr = df.corr()

plt.figure(figsize=(18,10))

sns.heatmap(corr, annot = True)

plt.show()
df.columns
features = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts',

       'Checking account', 'Credit amount', 'Duration', 'Purpose']

x = df.loc[:,features].values

y = df.loc[:,['Risk']].values #target var
#Step 1: Standardize the data

from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(x) 
pd.DataFrame(data = X, columns = features).head()
# Setp 2 : PCA Projection to 2D

from sklearn.decomposition import PCA

pca = PCA(n_components=3)

principalComponents = pca.fit_transform (x)

principalDf = pd.DataFrame(data = principalComponents, columns=['PC1','PC2','PC3']) #PC = Principal component 

principalDf.head()
finalDf = pd.concat([principalDf,df[['Risk']]], axis = 1)

finalDf.head()
# Step 3 - Visualize the Data in 2D.

fig = plt.figure(figsize=(10,5))

ax = fig.add_subplot(1,1,1)

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 Component PCA', fontsize = 20)



targets = [1,0]



colors = ['r','g']



for target, color in zip(targets,colors):

    indicesToKeep = finalDf['Risk'] == target

    ax.scatter(finalDf.loc[indicesToKeep, 'PC1']

               , finalDf.loc[indicesToKeep, 'PC2']

               , c = color

               , s = 50)

ax.legend(targets)

plt.show()
pca.explained_variance_ratio_
y = df['Risk']

X = df.drop(['Risk','id'], axis = 1)
#Split the dataset into train and test dataset.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
from sklearn.preprocessing import StandardScaler

Scaler_X = StandardScaler()

X_train = Scaler_X.fit_transform(X_train)

X_test = Scaler_X.transform(X_test)
y_test.value_counts()
#let's check what params will be best suitable for random forest classification.

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.model_selection import cross_val_score



rfc_clf = RandomForestClassifier()

params = {'n_estimators':[25,50,100,150,200,500],'max_depth':[0.5,1,5,10],'random_state':[1,10,20,42],

          'n_jobs':[1,2]}

grid_search_cv = GridSearchCV(rfc_clf, params, scoring='precision')

grid_search_cv.fit(X_train, y_train)

print(grid_search_cv.best_estimator_)

print(grid_search_cv.best_params_)
rfc_clf = grid_search_cv.best_estimator_

rfc_clf.fit(X_train,y_train)

rfc_clf_pred = rfc_clf.predict(X_test)

print('Accuracy:',accuracy_score(rfc_clf_pred,y_test) )

print('Confusion Matrix:', confusion_matrix(rfc_clf_pred,y_test).ravel()) #tn,fp,fn,tp

print('Classification report:')

print(classification_report(rfc_clf_pred,y_test))



# Let's make sure the data is not overfitting

score_rfc = cross_val_score(rfc_clf,X_train,y_train,cv = 10).mean()

print('cross val score:', score_rfc)
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

# Implement gridsearchcv to see which are our best p



params = {'C': [0.75, 0.85, 0.95, 1], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 

          'degree': [3, 4, 5]}



svc_clf = SVC(random_state=42)

grid_search_cv = GridSearchCV(svc_clf, params)

grid_search_cv.fit(X_train, y_train)
print(grid_search_cv.best_estimator_)

print(grid_search_cv.best_params_)
svc_clf = grid_search_cv.best_estimator_

svc_clf.fit(X_train,y_train)

svc_pred = svc_clf.predict(X_test)



print('Accuracy:',accuracy_score(svc_pred,y_test) )

print('Confusion Matrix:', confusion_matrix(svc_pred,y_test,labels=[0,1])) #tn,fp,fn,tp

print('Classification report:')

print(classification_report(svc_pred,y_test))



# Let's make sure the data is not overfitting

score_svc = cross_val_score(svc_clf,X_train,y_train, cv = 10).mean()

print('cross val score:', score_svc)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,y_train)

lr_pred = lr.predict(X_test)



print('Accuracy:',accuracy_score(lr_pred,y_test) )

print('Confusion Matrix:', confusion_matrix(lr_pred,y_test,labels=[0,1])) #tn,fp,fn,tp

print('Classification report:')

print(classification_report(lr_pred,y_test))



# Let's make sure the data is not overfitting

score_lr = cross_val_score(lr,X_train,y_train,cv=10).mean()

print('cross val score:', score_lr)
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

gbc.fit(X_train,y_train)

gbc_pred = gbc.predict(X_test)



print('Accuracy:',accuracy_score(gbc_pred,y_test) )

print('Confusion Matrix:', confusion_matrix(gbc_pred,y_test,labels=[0,1])) #tn,fp,fn,tp

print('Classification report:')

print(classification_report(gbc_pred,y_test))



# Let's make sure the data is not overfitting

score_gbc = cross_val_score(gbc,X_train,y_train, scoring='accuracy', cv = 10).mean()

print('cross val score:', score_gbc)
models = pd.DataFrame({'Models':['Random Forest Classifier','Logistic Regression', 

                                 'Gradient Boost Classifier', 'Support Vector Classifier'],

                      'Score':[score_rfc,score_lr,score_gbc,score_svc]})

models.sort_values(by='Score', ascending = False)

fig, ax_arr = plt.subplots(nrows = 2, ncols = 2, figsize = (20,15))



from sklearn import metrics



#gbc

gbc_prob = gbc.predict_proba(X_test)[:,1]

fprgbc, tprgbc, thresholdsgbc = metrics.roc_curve(y_test, gbc_prob)

roc_auc_gbc = metrics.auc(fprgbc,tprgbc)

ax_arr[0,0].plot(fprgbc, tprgbc,'b',label = 'AUC = %0.2f' % roc_auc_gbc,color = 'red')

ax_arr[0,0].plot([0, 1], [0, 1], 'k--')

ax_arr[0,0].set_xlabel('False Positive Rate')

ax_arr[0,0].set_ylabel('True Positive Rate')

ax_arr[0,0].set_title('ROC for GBC.', fontsize = 20)

ax_arr[0,0].legend(loc = 'lower right', prop={'size': 16})





#Random forest

rfc_prob = rfc_clf.predict_proba(X_test)[:,1]

fprRfc, tprRfc, thresholdsRfc = metrics.roc_curve(y_test, rfc_prob)

roc_auc_rfc = metrics.auc(fprRfc,tprRfc)

ax_arr[0,1].plot(fprRfc, tprRfc,'b',label = 'AUC = %0.2f' % roc_auc_rfc,color = 'green')

ax_arr[0,1].plot([0, 1], [0, 1], 'k--')

ax_arr[0,1].set_xlabel('False Positive Rate')

ax_arr[0,1].set_ylabel('True Positive Rate')

ax_arr[0,1].set_title('ROC for RFC.', fontsize = 20)

ax_arr[0,1].legend(loc = 'lower right', prop={'size': 16})





#Logistic Regression

lr_prob = lr.predict_proba(X_test)[:,1]

fprLr, tprLr, thresholdsLr = metrics.roc_curve(y_test, lr_prob)

roc_auc_lr = metrics.auc(fprLr,tprLr)

ax_arr[1,0].plot(fprLr, tprLr,'b',label = 'AUC = %0.2f' % roc_auc_lr,color = 'blue')

ax_arr[1,0].plot([0, 1], [0, 1], 'k--')

ax_arr[1,0].set_xlabel('False Positive Rate')

ax_arr[1,0].set_ylabel('True Positive Rate')

ax_arr[1,0].set_title('ROC for Logistic.', fontsize = 20)

ax_arr[1,0].legend(loc = 'lower right', prop={'size': 16})



#For All

ax_arr[1,1].plot(fprgbc,tprgbc, label ='Gradient Boost', color = 'red')

ax_arr[1,1].plot(fprRfc,tprRfc, label ='Random Forest', color = 'green')

ax_arr[1,1].plot(fprLr,tprLr, label ='Logistic Regression', color = 'blue')

ax_arr[1,1].plot([0, 1], [0, 1], 'k--')

ax_arr[1,1].set_title('Receiver Operating Comparison ',fontsize=20)

ax_arr[1,1].set_ylabel('True Positive Rate',fontsize=20)

ax_arr[1,1].set_xlabel('False Positive Rate',fontsize=15)

ax_arr[1,1].legend(loc = 'lower right', prop={'size': 16})



plt.subplots_adjust(wspace=0.2)

plt.tight_layout() 

plt.show()
pca = PCA(n_components=5)

pca.fit(X_train)
pca.n_components_
#Apply the mapping (transform) to both the training set and the test set.

train_X = pca.transform(X_train)

test_X = pca.transform(X_test)
from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults

# default solver is incredibly slow thats why we change it

# solver = 'lbfgs'

logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_X, y_train)
logR_pred = logisticRegr.predict(test_X)
logisticRegr.score(test_X,y_test)
confusion_matrix(logR_pred,y_test)
logR_cross_val_score = cross_val_score(logisticRegr,train_X,y_train, cv = 10).mean()

logR_cross_val_score