import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



import os

print(os.listdir("../input"))
# Load Data

heart_data = pd.read_csv('../input/heart.csv')
heart_data.head(10)
heart_data.describe()
heart_data.isnull().any()
# No Null Values in any column
all_columns = heart_data.columns.values.tolist()

num_columns = ['age','trestbps','chol','thalach','oldpeak']

cat_columns = [clm_name for clm_name in all_columns if clm_name not in num_columns]

print('Columns with continuous data : {} Count = {}\nColumns with catagorical data : {} Count = {}'.format(num_columns,len(num_columns),cat_columns,len(cat_columns)))
heart_data[heart_data.duplicated() == True]
heart_data.drop_duplicates(inplace = True)

heart_data[heart_data.duplicated() == True]
# Removed the duplicate row
# Sex distribution 



male_count = heart_data.sex.value_counts().tolist()[0]

female_count = heart_data.sex.value_counts().tolist()[1]

print('Male :',male_count)

print('Female :',female_count)
fig, (ax1,ax2) = plt.subplots(1,2,figsize = (12,5),constrained_layout=True)

plt.subplots_adjust(wspace = 0.5)



ax1.bar(heart_data.sex.unique(),heart_data.sex.value_counts(),color = ['blue','red'],width = 0.8)

ax1.set_xticks(heart_data.sex.unique())

ax1.set_xticklabels(('Male','Female'))



ax2.pie((male_count,female_count), labels = ('Male','Female'), autopct='%1.1f%%', shadow=True, startangle=90, explode=[0,0.3])



plt.show()
# Population Distribution with age and sex



fig, (ax1,ax2) = plt.subplots(1,2, figsize = (20,5),constrained_layout=True)

bin_x = range(25,80,2)



ax1.hist(heart_data.age.tolist(),bins=bin_x,rwidth=0.9)

ax1.set_xticks(range(25,80,2))

ax1.set_xlabel('Age',fontsize=15)

ax1.set_ylabel('Population Count',fontsize=15)

ax1.set_title('Total population distribution',fontsize=20)



ax2.hist(heart_data[heart_data['sex']==1].age.tolist(),label = 'Male',bins=bin_x,rwidth=0.9)

ax2.hist(heart_data[heart_data['sex']==0].age.tolist(),label = 'Female',bins=bin_x,rwidth=0.5)

ax2.legend()

ax2.set_xticks(range(25,80,2))

ax2.set_xlabel('Age',fontsize=15)

ax2.set_ylabel('Population Count',fontsize=15)

ax2.set_title('Male vs female',fontsize=20)



plt.show()
# Population distribution for heart disease



x = heart_data.groupby(['age','target']).agg({'sex':'count'})

y = heart_data.groupby(['age']).agg({'sex':'count'})

z = (x.div(y, level='age') * 100)

q= 100 - z



fig, axes = plt.subplots(2,2, figsize = (20,12))

plt.subplots_adjust(hspace = 0.5)



axes[0,0].hist(heart_data[heart_data['target']==1].age.tolist(),bins=bin_x,rwidth=0.8)

axes[0,0].set_xticks(range(25,80,2))

axes[0,0].set_xlabel('Age Range',fontsize=15)

axes[0,0].set_ylabel('Population Count',fontsize=15)

axes[0,0].set_title('People suffering from heart disease',fontsize=20)



axes[0,1].hist(heart_data[heart_data['target']==0].age.tolist(),bins=bin_x,rwidth=0.8)

axes[0,1].set_xticks(range(25,80,2))

axes[0,1].set_xlabel('Age Range',fontsize=15)

axes[0,1].set_ylabel('Population Count',fontsize=15)

axes[0,1].set_title('People not suffering from heart disease',fontsize=20)



axes[1,0].scatter(z.xs(1,level=1).reset_index().age,z.xs(1,level=1).reset_index().sex,s=(x.xs(1,level=1).sex)*30,edgecolors = 'r',c = 'yellow')

axes[1,0].plot(z.xs(1,level=1).reset_index().age,z.xs(1,level=1).reset_index().sex)

axes[1,0].set_xticks(range(25,80,2))

axes[1,0].set_yticks(range(0,110,5))

axes[1,0].set_xlabel('Age',fontsize=15)

axes[1,0].set_ylabel('%',fontsize=15)

axes[1,0].set_title('% of people with heart disease by age',fontsize=20)



axes[1,1].scatter(z.xs(1,level=1).reset_index().age,q.xs(1,level=1).reset_index().sex,s=(x.xs(0,level=1).sex)*30,edgecolors = 'r',c = 'yellow')

axes[1,1].plot(z.xs(1,level=1).reset_index().age,q.xs(1,level=1).reset_index().sex)

axes[1,1].set_xticks(range(25,80,2))

axes[1,1].set_yticks(range(0,110,5))

axes[1,1].set_xlabel('Age',fontsize=15)

axes[1,1].set_ylabel('%',fontsize=15)

axes[1,1].set_title('% of people with no heart disease by age',fontsize=20)



plt.show()
# Looking at other features and how they are distributed.

# Scatter plot for continuous data

# Pie plot for catagorical data
fig, axes = plt.subplots(6,2, figsize = (20,40))

plt.subplots_adjust(hspace = 0.5)



axes[0,0].scatter(heart_data[heart_data['target']==0][['age','thalach']].sort_values(by = ['age']).age,heart_data[heart_data['target']==0][['age','thalach']].sort_values(by = ['age']).thalach, c = 'g',label = 'target=0')

axes[0,0].scatter(heart_data[heart_data['target']==1][['age','thalach']].sort_values(by = ['age']).age,heart_data[heart_data['target']==1][['age','thalach']].sort_values(by = ['age']).thalach, c = 'r',label = 'target=1')

axes[0,0].set_title('thalach distribution',fontsize=20)

axes[0,0].set_xticks(range(25,80,2))

axes[0,0].set_xlabel('Age',fontsize=15)

axes[0,0].set_ylabel('thalach',fontsize=15)

axes[0,0].axhline(np.mean(heart_data['thalach']),xmin=0,xmax=1,linewidth=1, color='black',linestyle = '--')

axes[0,0].axvline(np.mean(heart_data['age']),ymin=0,ymax=1,linewidth=1, color='b',linestyle = '--')

axes[0,0].legend()



axes[0,1].scatter(heart_data[heart_data['target']==0][['age','trestbps']].sort_values(by = ['age']).age,heart_data[heart_data['target']==0][['age','trestbps']].sort_values(by = ['age']).trestbps, c = 'g',label = 'target=0')

axes[0,1].scatter(heart_data[heart_data['target']==1][['age','trestbps']].sort_values(by = ['age']).age,heart_data[heart_data['target']==1][['age','trestbps']].sort_values(by = ['age']).trestbps, c = 'r',label = 'target=1')

axes[0,1].set_title('trestbps distribution',fontsize=20)

axes[0,1].set_xticks(range(25,80,2))

axes[0,1].set_xlabel('Age',fontsize=15)

axes[0,1].set_ylabel('trestbps',fontsize=15)

axes[0,1].axhline(np.mean(heart_data['trestbps']),xmin=0,xmax=1,linewidth=1, color='r',linestyle = '--')

axes[0,1].axvline(np.mean(heart_data['age']),ymin=0,ymax=1,linewidth=1, color='b',linestyle = '--')



# heart_data[heart_data['target']==1][['age','chol',]].sort_values(by = ['age'])

axes[1,0].scatter(heart_data[heart_data['target']==0][['age','chol',]].sort_values(by = ['age']).age,heart_data[heart_data['target']==0][['age','chol',]].sort_values(by = ['age']).chol,c = 'g',label = 'target=0')

axes[1,0].scatter(heart_data[heart_data['target']==1][['age','chol',]].sort_values(by = ['age']).age,heart_data[heart_data['target']==1][['age','chol',]].sort_values(by = ['age']).chol,c = 'r',label = 'target=1')

axes[1,0].set_title('chol distribution',fontsize=20)

axes[1,0].set_xticks(range(25,80,2))

axes[1,0].set_xlabel('Age',fontsize=15)

axes[1,0].set_ylabel('chol',fontsize=15)

axes[1,0].axhline(np.mean(heart_data['chol']),xmin=0,xmax=1,linewidth=1, color='r',linestyle = '--')

axes[1,0].axvline(np.mean(heart_data['age']),ymin=0,ymax=1,linewidth=1, color='b',linestyle = '--')



axes[1,1].scatter(heart_data[heart_data['target']==0][['age','oldpeak',]].sort_values(by = ['age']).age,heart_data[heart_data['target']==0][['age','oldpeak',]].sort_values(by = ['age']).oldpeak,c = 'g',label = 'target=0')

axes[1,1].scatter(heart_data[heart_data['target']==1][['age','oldpeak',]].sort_values(by = ['age']).age,heart_data[heart_data['target']==1][['age','oldpeak',]].sort_values(by = ['age']).oldpeak,c = 'r',label = 'target=1')

axes[1,1].set_title('oldpeak distribution',fontsize=20)

axes[1,1].set_xticks(range(25,80,2))

axes[1,1].set_xlabel('Age',fontsize=15)

axes[1,1].set_ylabel('oldpeak',fontsize=15)

axes[1,1].axhline(np.mean(heart_data['oldpeak']),xmin=0,xmax=1,linewidth=1, color='r',linestyle = '--')

axes[1,1].axvline(np.mean(heart_data['age']),ymin=0,ymax=1,linewidth=1, color='b',linestyle = '--')



fbs_count = heart_data['fbs'].value_counts()

labels = [('fbs = '+ str(x)) for x in fbs_count.index]

axes[2,0].pie(fbs_count,labels = labels,autopct='%1.1f%%',shadow=True, startangle=45)

axes[2,0].axis('equal')

axes[2,0].set_title('fbs share',fontsize=15)



restecg_count = heart_data['restecg'].value_counts()

labels = [('restecg = '+ str(x)) for x in restecg_count.index]

axes[2,1].pie(restecg_count,labels = labels,autopct='%1.1f%%',shadow=True, startangle=45,explode = [0,0,0.5])

axes[2,1].axis('equal')

axes[2,1].set_title('restecg share',fontsize=15)



exang_count = heart_data['exang'].value_counts()

labels = [('exang = '+ str(x)) for x in exang_count.index]

axes[3,0].pie(exang_count,labels = labels,autopct='%1.1f%%',shadow=True, startangle=45)

axes[3,0].axis('equal')

axes[3,0].set_title('exang share',fontsize=15)



slope_count = heart_data['slope'].value_counts()

labels = [('slope = '+ str(x)) for x in slope_count.index]

axes[3,1].pie(slope_count,labels = labels,autopct='%1.1f%%',shadow=True, startangle=45)

axes[3,1].axis('equal')

axes[3,1].set_title('slope share',fontsize=15)



ca_count = heart_data['ca'].value_counts()

labels = [('ca = '+ str(x)) for x in ca_count.index]

axes[4,0].pie(ca_count,labels = labels,autopct='%1.1f%%',shadow=True, startangle=45)

axes[4,0].axis('equal')

axes[4,0].set_title('ca share',fontsize=15)



thal_count = heart_data['thal'].value_counts()

labels = [('thal = '+ str(x)) for x in thal_count.index]

axes[4,1].pie(thal_count,labels = labels,autopct='%1.1f%%',shadow=True, startangle=45)

axes[4,1].axis('equal')

axes[4,1].set_title('thal share',fontsize=15)



cp_count = heart_data['cp'].value_counts()

labels = [('cp = '+ str(x)) for x in cp_count.index]

axes[5,0].pie(cp_count,labels = labels,autopct='%1.1f%%',shadow=True, startangle=45)

axes[5,0].axis('equal')

axes[5,0].set_title('CP share',fontsize=15)



target_count = heart_data['target'].value_counts()

labels = [('target = '+ str(x)) for x in target_count.index]

axes[5,1].pie(target_count,labels = labels,autopct='%1.1f%%',shadow=True, startangle=45)

axes[5,1].axis('equal')

axes[5,1].set_title('target share',fontsize=15)



plt.show()
#  Lets look at the correlation matrix and plot it using Pandas Style and Matplotlib

heart_data.corr().round(decimals =2).style.background_gradient(cmap = 'Oranges')
names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']

correlations = heart_data.corr()

# plot correlation matrix

fig, ax = plt.subplots(1,1, figsize = (10,8),constrained_layout=True)



cax = ax.matshow(correlations, vmin=-1, vmax=1,cmap = 'afmhot')

fig.colorbar(cax)

ticks = np.arange(0,14,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names)

ax.set_yticklabels(names)

for i in range(len(names)):

    for j in range(len(names)):

        text = ax.text(j, i, heart_data.corr().as_matrix(columns= None)[i, j].round(decimals =2),

                       ha="center", va="center", color="black")

plt.show()
# Corelation with target



x = heart_data.corr()

pd.DataFrame(x['target']).sort_values(by='target',ascending = False).style.background_gradient(cmap = 'Greens')
# Importing stuff

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MinMaxScaler,RobustScaler, StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
# We have already saved all the continous columns



print('Columns with continous data = ',num_columns)



import scipy.stats as stats



fig, axes = plt.subplots(2,2, figsize = (15,12))

plt.subplots_adjust(hspace = 0.2)



h= np.sort(heart_data.thalach)

fit = stats.norm.pdf(h, np.mean(h), np.std(h)) 

axes[0,0].plot(h,fit,'--')

axes[0,0].hist(h,density=True) 

axes[0,0].set_title("thalach")

axes[0,0].set_ylabel('Density')



h2= np.sort(heart_data.trestbps)

fit2 = stats.norm.pdf(h2, np.mean(h2), np.std(h2)) 

axes[0,1].plot(h2,fit2,'--')

axes[0,1].hist(h2,density=True) 

axes[0,1].set_title("trestbps")

axes[0,1].set_ylabel('Density')



h3= np.sort(heart_data.chol)

fit3 = stats.norm.pdf(h3, np.mean(h3), np.std(h3)) 

axes[1,0].plot(h3,fit3,'--')

axes[1,0].hist(h3,density=True) 

axes[1,0].set_title("chol")

axes[1,0].set_ylabel('Density')



h4= np.sort(heart_data.oldpeak)

fit4 = stats.norm.pdf(h4, np.mean(h4), np.std(h4)) 

axes[1,1].plot(h4,fit4,'--')

axes[1,1].hist(h4,density=True) 

axes[1,1].set_title("oldpeak")

axes[1,1].set_ylabel('Density')



plt.show()



print(r"Scaling them using MinMax Scaler")
mm = MinMaxScaler()



num_data = heart_data[num_columns]

num_data_tf = mm.fit_transform(num_data)

num_data_tf
ohe = OneHotEncoder()



cat_columns.remove('target')

print(cat_columns)

cat_data = heart_data[cat_columns]

cat_data_tf = ohe.fit_transform(cat_data).toarray()



heart_data_tf = np.hstack([num_data_tf,cat_data_tf])

heart_data_tf
# Anomoly detection with DBscan (eps value was set using trial and error)

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=2.05)

pred = dbscan.fit_predict(heart_data_tf)

dbanom = heart_data[pred == -1]



columns_continous_data =  ['trestbps', 'chol', 'thalach', 'oldpeak']



fig, axes = plt.subplots(2,2, figsize = (10,8))



p =0

for i in range(0,2):

    for j in range(0,2):

        axes[i,j].scatter(heart_data.index,heart_data[columns_continous_data[p]])

        axes[i,j].scatter(dbanom.index,dbanom[columns_continous_data[p]],c='r')

        axes[i,j].set_title('Anomalies with DBSCAN'+" | plot = "+columns_continous_data[p])

        p +=1

        

plt.show()
#  Anomoly detection with EE

from sklearn.covariance import EllipticEnvelope

ee = EllipticEnvelope(contamination=.03)

ee.fit_predict(heart_data_tf)

eeanom = heart_data[ee.predict(heart_data_tf) == -1]



fig, axes = plt.subplots(2,2, figsize = (10,8))



p =0

for i in range(0,2):

    for j in range(0,2):

        axes[i,j].scatter(heart_data.index,heart_data[columns_continous_data[p]])

        axes[i,j].scatter(eeanom.index,eeanom[columns_continous_data[p]],c='r')

        axes[i,j].set_title('Anomalies with DBSCAN'+" | plot = "+columns_continous_data[p])

        p +=1



plt.show()
#Checking for similar Anomolies in both above methods and removing them from dataset

df = pd.DataFrame

df= dbanom.index.intersection(eeanom.index)

heart_data_tf_df = pd.DataFrame(heart_data_tf)

heart_data_tf_df.drop(df,inplace = True)

heart_data.drop(df,inplace = True)
heart_data.info()
heart_data_tf_df.head()
heart_data.head()
# Set X as feature data and Y as target data for Unscaled Data. set X_tf as feature data for scaled data.

X = heart_data.drop(['target'],axis =1)

Y = heart_data.target



X_tf = heart_data_tf_df
from sklearn import feature_selection

chi2, pval = feature_selection.chi2(X,Y)

print(chi2)
print(X.columns)
dep = pd.DataFrame(chi2)

dep.columns = ['Dependency']

dep.index = X.columns

print("""Looks like "fbs" has lowest effect on target and "thalach" has highest""")

dep.sort_values('Dependency', ascending = False).style.background_gradient(cmap = 'terrain')
# Since the number of columns are not much and all features looks important, we can go ahead with same number of columns. 
Y.value_counts()
# Target is not much imbalanced and there is no need to balance it, but will use oversampling, SMOTE method to balance data.
from imblearn.over_sampling import SMOTE

smote = SMOTE()

X_reshaped, Y_reshaped = SMOTE().fit_sample(X, Y)
import collections

collections.Counter(Y_reshaped)

# Balanced the classes, but will not use it in predictions
print("Split the data into train and test with unscaled data:")

trainX,testX,trainY,testY = train_test_split(X,Y,test_size = 0.3,random_state = None)

print("trainX,testX,trainY,testY")

print("Split the data into train and test with un scaled data:")

trainX_tf,testX_tf,trainY_tf,testY_tf = train_test_split(X_tf,Y,test_size = 0.3,random_state = None)

print("trainX_tf,testX_tf,trainY_tf,testY_tf")
rf = RandomForestClassifier()
#Using grid search to get best params for Randomforest

params = {

    'n_estimators':[10,50,100,150,200,250],

    'random_state': [10,5,15,20,50]

         }

gs = GridSearchCV(rf, param_grid=params, cv=5, n_jobs=-1)

gs.fit(trainX,trainY)
n_est = []

rnd_sta = []

score = []

rand_state_list =  [5,10,15,20,50]

for x in range(len(gs.cv_results_['params'])):

    n_est.append(gs.cv_results_['params'][x]['n_estimators'])

    rnd_sta.append(gs.cv_results_['params'][x]['random_state'])

    score.append(gs.cv_results_['mean_test_score'][x])



grid_frame = pd.DataFrame()

grid_frame['n_est'] = n_est

grid_frame['rnd_sta'] = rnd_sta

grid_frame['score'] = score



grid_frame[grid_frame['rnd_sta'] == 10]



plt.figure(figsize=(10,6))



for value in rand_state_list:

    plt.plot(grid_frame[grid_frame['rnd_sta'] == value].n_est,grid_frame[grid_frame['rnd_sta'] == value].score,'-o',label = 'random_state = value')



plt.title('Mean Score with different params')

plt.grid(True)

plt.legend()

plt.show()
# Grid Search Score with test Data

print("Grid search score with random forest classifier = ",gs.score(testX,testY)*100)
#Best Params

gs.best_params_
# Creating the Confusion matrix

pred = gs.predict(testX)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_pred=pred, y_true=testY)
# Using grid search to get best params for Randomforest . Now with scaled data (StandardScaler)

params = {

    'n_estimators':[10,50,100,150,200,250,300],

    'random_state': [10,5,15,20,50]

         }

gs = GridSearchCV(rf, param_grid=params, cv=10, n_jobs=-1)

gs.fit(trainX_tf,trainY_tf)
n_est = []

rnd_sta = []

score = []

for x in range(len(gs.cv_results_['params'])):

    n_est.append(gs.cv_results_['params'][x]['n_estimators'])

    rnd_sta.append(gs.cv_results_['params'][x]['random_state'])

    score.append(gs.cv_results_['mean_test_score'][x])



grid_frame = pd.DataFrame()

grid_frame['n_est'] = n_est

grid_frame['rnd_sta'] = rnd_sta

grid_frame['score'] = score



grid_frame[grid_frame['rnd_sta'] == 10]



plt.figure(figsize=(10,6))



for value in rand_state_list:

    plt.plot(grid_frame[grid_frame['rnd_sta'] == value].n_est,grid_frame[grid_frame['rnd_sta'] == value].score,'-o',label = 'random_state = value')



plt.title('Mean Score with different params')

plt.grid(True)

plt.legend()

plt.show()
# Grid Search Score wih scaled test data 

print("Grid search score with random forest classifier (Scaled Data)= ",gs.score(testX_tf,testY_tf)*100)
#Best Params (We save best params for future use)

print(gs.best_params_)

n_est = gs.best_params_['n_estimators']

rnd_st = gs.best_params_['random_state']
# Creating the Confusion matrix

pred = gs.predict(testX_tf)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_pred=pred, y_true=testY_tf)
from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,RandomForestClassifier, VotingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
models = [SVC(kernel='linear',C =100),

          SGDClassifier(max_iter=1000,tol=0.003),

          DecisionTreeClassifier(),

          ExtraTreeClassifier(),

          AdaBoostClassifier(), 

          BaggingClassifier(), 

          GradientBoostingClassifier(),

          RandomForestClassifier(n_estimators=n_est,random_state=rnd_st),

          GaussianNB(),

          KNeighborsClassifier(), 

          LogisticRegression(max_iter=1000,solver='lbfgs')]



modelnames = ['SVC',

              'SGDClassifier',

              'DecisionTreeClassifier',

              'ExtraTreeClassifier',

              'AdaBoostClassifier', 

              'BaggingClassifier', 

              'GradientBoostingClassifier',

              'RandomForestClassifier',

              'GaussianNB',

              'KNeighborsClassifier', 

              'LogisticRegression']
# Train and test with unscaled model

scores_unscaled = []

for index,model in enumerate(models):

    try:

        model.fit(trainX,trainY)

        print(modelnames[index],"Accuracy =",round(model.score(testX,testY)*100,2),"%")

        scores_unscaled.append(round(model.score(testX,testY)*100,2))

    except:

        print("Skipped",modelnames[index])
# plt.bar(range(len(modelnames)),scores_unscaled, color = ['blue','red'])

plt.plot(range(len(modelnames)),scores_unscaled, '-o')



plt.xticks(range(0,11,1),labels = modelnames, rotation = 90)

plt.grid(visible=True)



plt.show()
# Train and test with scaled model

scores_unscaled = []

for index,model in enumerate(models):

    try:

        model.fit(trainX_tf,trainY_tf)

        print(modelnames[index],"Accuracy =",round(model.score(testX_tf,testY_tf)*100,2),"%")

        scores_unscaled.append(round(model.score(testX_tf,testY_tf)*100,2))

    except:

        print("Skipped",modelnames[index])
plt.plot(range(len(modelnames)),scores_unscaled, '-o')



plt.xticks(range(0,11,1),labels = modelnames, rotation = 90)

plt.grid(visible=True)



plt.show()
#  base_estimator=DecisionTreeClassifier

ab = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=1000)

ab.fit(trainX,trainY)

print('AdaBoost Accuracy with Decision Tree = ',(ab.score(testX,testY)*100))



ab = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=1000)

ab.fit(trainX_tf,trainY_tf)

print('AdaBoost Accuracy with Decision Tree (Scaled Data)= ',(ab.score(testX_tf,testY_tf)*100))
#  base_estimator=RandomForest

ab = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=1000,random_state=10),n_estimators=1000)

ab.fit(trainX,trainY)

print('AdaBoost Accuracy with Random Forest = ',(ab.score(testX,testY)*100))



ab = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=1000,random_state=10),n_estimators=1000)

ab.fit(trainX_tf,trainY_tf)

print('AdaBoost Accuracy with Random Forest (Scaled Data)= ',(ab.score(testX_tf,testY_tf)*100))
#  base_estimator=LogisticRegression

ab = AdaBoostClassifier(base_estimator=LogisticRegression(max_iter=1000,solver = 'lbfgs'),n_estimators=1000)

ab.fit(trainX,trainY)

print('AdaBoost Accuracy with Logistic Reg = ',(ab.score(testX,testY)*100))



ab = AdaBoostClassifier(base_estimator=LogisticRegression(max_iter=1000,solver = 'lbfgs'),n_estimators=1000)

ab.fit(trainX_tf,trainY_tf)

print('AdaBoost Accuracy with Logistic Reg (Scaled Data)= ',(ab.score(testX_tf,testY_tf)*100))
#  base_estimator=SVC

ab = AdaBoostClassifier(algorithm='SAMME',base_estimator=SVC(kernel='linear',C = 1000, gamma=1),n_estimators=1000)

ab.fit(trainX,trainY)

print('AdaBoost Accuracy with SVC = ',(ab.score(testX,testY)*100))



ab = AdaBoostClassifier(algorithm='SAMME',base_estimator=SVC(kernel='linear',C = 1000, gamma=1),n_estimators=1000)

ab.fit(trainX_tf,trainY_tf)

print('AdaBoost Accuracy with SVC = (Scaled Data)',(ab.score(testX_tf,testY_tf)*100))
estimators = [ 

    ('RandomForestClassifier',RandomForestClassifier(n_estimators=n_est, random_state=rnd_st)),

    ('SVC(kernel= rfb',SVC(kernel='rbf', probability=True,gamma=1)),

    ('SVC(kernel= linear',SVC(kernel='linear',C = 100, gamma=1, probability=True)),

    ('KNeighborsClassifier',KNeighborsClassifier()),

    ('AdaBoostClassifier LR',AdaBoostClassifier(base_estimator=LogisticRegression(solver = 'lbfgs'),n_estimators=1000)),

    ('LogisticRegression',LogisticRegression(max_iter= 10000,solver = 'lbfgs')),

    ('GaussianNB',GaussianNB())

]
vc = VotingClassifier(estimators=estimators, voting='hard')
vc.fit(trainX,trainY)

print('Voting Classifier accuracy = ',(vc.score(testX,testY)*100))
# Checking who contribute what % in voting
weights = []

for est,name in zip(vc.estimators_,vc.estimators):

    score = est.score(testX,testY)

    print (name[0], score*100)

    weights.append((100/(10-(score*10))))

# Converting n saving Score in weights to be used later

print('Weights = ', weights)
# Adjusting weights and recalculating accuracy

vc = VotingClassifier(estimators=estimators, voting='soft', weights=weights)

vc.fit(trainX,trainY)

print('Voting Classifier accuracy = ',(vc.score(testX,testY)*100))
vc.fit(trainX_tf,trainY_tf)

print('Voting Classifier accuracy = ',(vc.score(testX_tf,testY_tf)*100))
weights = []

for est,name in zip(vc.estimators_,vc.estimators):

    score = est.score(testX_tf,testY_tf)

    print (name[0], score*100)

    weights.append((100/(10-(score*10))))

# Converting n saving Score in weights to be used later

print('Weights = ', weights)