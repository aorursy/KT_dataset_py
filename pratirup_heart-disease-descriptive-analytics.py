import numpy as np

import pandas as pa

import matplotlib.pyplot as plt

import seaborn as sn

from scipy import stats



import warnings

warnings.filterwarnings('ignore')



color = sn.color_palette()

sn.set_style("ticks")

plt.style.use('fivethirtyeight')

plt.style.use('ggplot')
data = pa.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
print(data.shape)

print('Length of the data {}'.format(len(data)))

print(data.info())
data = data.drop([48 ,281,  92, 158, 163, 164, 251],axis=0).reset_index(drop=True)
#Missing Values

data.isnull().sum().sum()
data.describe()
plt.figure(figsize=(20,4))

sn.distplot(data['age'],color='blue',label='Skewness : %.2f'%data['age'].skew())

plt.legend()



features = ['trestbps','chol','oldpeak','thalach']

plt.figure(figsize=(20,10))



for i in range(1, 5):

    ax=plt.subplot(2, 2, i)

    ax=sn.distplot(data[features[i-1]],label='Skewness : %.2f'%data[features[i-1]].skew(),color='blue')

    ax=sn.distplot(data[features[i-1]],label='Kurtosis : %.2f'%data[features[i-1]].kurtosis())



    plt.legend(loc='best')
plt.figure(figsize=(20,6))

stats.probplot(data['trestbps'],dist="norm",plot=plt)

plt.title('trestbps')



features = ['age','chol','oldpeak','thalach']

plt.figure(figsize=(20,10))



for i in range(1, 5):

    ax=plt.subplot(2, 2, i)

    ax=stats.probplot(data[features[i-1]],dist="norm",plot=plt)

    plt.title(features[i-1])
columns = ['age','trestbps','chol','oldpeak','thalach']

for i in columns:

    alpha= 0.001#singificance-level

    k2,p = stats.normaltest(data[i],nan_policy='omit')

    

    if p>alpha:

        print('{} ----- Normally distributed (Retain the null hypothesis)'.format(i))

    else:

        print('{} -----  Not normally distributed (Reject the null hypothesis)'.format(i))
sn.set_style("ticks")



age_heart_disease = data.groupby('target')['age']



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(20,5))

ax = sn.distplot(data['age'],ax=axis1)

ax.set(xlabel='Age')

ax = sn.distplot(age_heart_disease.get_group(0),ax=axis2)

ax.set(xlabel='Age With Heart Disease')

ax = sn.distplot(age_heart_disease.get_group(1),ax=axis3) 

ax.set(xlabel='Age Without Heart Disease')
groups_mean = data.groupby('target')['age'].mean()

groups_std = data.groupby('target')['age'].std()



groups = pa.DataFrame({'Group':[0,1],'Sample_Age_Mean':groups_mean.values,'Sample_Age_Std':groups_std.values,'Sample_Size':

                      [len(data.age[data['target'] == 0]),len(data.age[data['target'] == 1])]})

groups
sn.distplot(age_heart_disease.get_group(0),label='Heart_disease_yes')

sn.distplot(age_heart_disease.get_group(1),label='Heart_disease_no')

plt.legend()
def t_test(mean1,mean2,u1_u2,n1,n2,std1,std2):

    t_stat = ((mean1 - mean2) - (u1_u2)) / np.sqrt((std1**2/n1)+(std2**2/n2))

    #print(t_stat)

    return t_stat
# **calculting degrees of freedom**

def degree_freedom(std1,std2,n1,n2):

    su = ((std1**2/n1)+(std2**2/n2))**2

    de = ((std1**2)/n1)**2/(n1-1) + ((std2**2)/n2)**2/(n2-1)

    df = np.round(su/de)

    return df
# T-statistic

print('The corrosponding t-statistic {}'.format(t_test(56.735294,52.643750,0,136,160,7.923930,9.551151)))

print('The corrosponding degrees of freedom {}'.format(degree_freedom(7.923930,9.551151,136,160)))

pvalues = 2*(1-stats.t.cdf(4.027932863828695,294))

print('P-values -> {} '.format(pvalues))
## U can also use this inbuilt library function for t-test

#from scipy import stats

stats.ttest_ind(data.age[data['target'] == 0],data.age[data['target'] == 1],equal_var=False)
sn.countplot(data.cp[data['target']==0])
obs = pa.crosstab(data['cp'],data.target[data['target']==0])

obs = [102,9,18,7]

exp = [136*0.30,136*0.25,136*0.40,136*0.05]

values = pa.DataFrame({'Observed_Freq':obs,'Expected_Freq':exp})

values
#apply chi square test

stats.chisquare(values.Observed_Freq,values.Expected_Freq)
thal_heart_disease = data.groupby('target')['thalach']



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(25,5))

ax = sn.distplot(data['thalach'],ax=axis1)

ax.set(xlabel='max heart rate')

ax = sn.distplot(age_heart_disease.get_group(0),ax=axis2)

ax.set(xlabel='max heart rate With Heart Disease')

ax = sn.distplot(age_heart_disease.get_group(1),ax=axis3) 

ax.set(xlabel='max heart rate Without Heart Disease')
groups_mean = data.groupby('target')['thalach'].mean()

groups_std = data.groupby('target')['thalach'].std()



groups = pa.DataFrame({'Group':[0,1],'Sample_thalach_Mean':groups_mean.values,'Sample_thalach_Std':groups_std.values,

                       'Sample_Size':[len(data.thalach[data['target'] == 0]),len(data.thalach[data['target'] == 1])]})

groups
sn.distplot(thal_heart_disease.get_group(0),label='Heart_disease_yes')

sn.distplot(thal_heart_disease.get_group(1),label='Heart_disease_no')

plt.legend()
stats.ttest_ind(data.thalach[data['target'] == 0],data.thalach[data['target'] == 1],equal_var=False)
trestbps_heart_disease = data.groupby('target')['trestbps']



fig, (axis1,axis2,axis3,axis4) = plt.subplots(1,4,figsize=(25,5))

ax = sn.distplot(data['trestbps'],ax=axis1)

ax.set(xlabel='Resting blood pressure')

ax = sn.distplot(trestbps_heart_disease.get_group(0),ax=axis2)

ax.set(xlabel='Resting blood pressure With Heart Disease')

ax = sn.distplot(trestbps_heart_disease.get_group(1),ax=axis3) 

ax.set(xlabel='Resting blood pressure Without Heart Disease')

for i in range(0,2):

    ax=sn.distplot(trestbps_heart_disease.get_group(i),ax=axis4)

    ax.set(xlabel='Resting blood pressure With & withiout Heart Disease')
chol_heart_disease = data.groupby('target')['chol']



fig, (axis1,axis2,axis3,axis4) = plt.subplots(1,4,figsize=(25,5))

ax = sn.distplot(data['chol'],ax=axis1)

ax.set(xlabel='Serum cholestoral')

ax = sn.distplot(chol_heart_disease.get_group(0),ax=axis2)

ax.set(xlabel='Serum cholestoral Without Heart Disease')

ax = sn.distplot(chol_heart_disease.get_group(1),ax=axis3) 

ax.set(xlabel='Serum cholestoral With Heart Disease')

for i in range(0,2):

    ax=sn.distplot(chol_heart_disease.get_group(i),ax=axis4)

    ax.set(xlabel='Serum cholestoral With & withiout Heart Disease')
z =pa.crosstab(data['target'],data['exang'],margins=True)

z
marginal_prob = (97/296) # marginal probability

support = (74/296)

confidence = (support/marginal_prob)

print(confidence)
sn.countplot(data['target'],palette='winter_r')
x = []

for i in range(0,len(data)):

    if((data['age'][i] > 0) & (data['age'][i] < 20) ):

        x.append('0-20')

    elif((data['age'][i] > 20) & (data['age'][i] < 40) ):

        x.append('21-50')

    elif((data['age'][i] > 40) & (data['age'][i] < 60) ):

        x.append('51-60')

    else:

        x.append('> 60')

        

data['Group_Age'] = x
plt.figure(figsize=(25,6))

pa.crosstab(data['age'],data['target']).plot(kind="bar",figsize=(20,6))
import squarify

fig,(axis1,axis2,axis3) = plt.subplots(1,3,figsize=(20,5))

data['Group_Age'].value_counts()

labels = data['Group_Age'].value_counts().index

sizes = data['Group_Age'].value_counts().values



perc = [str('{:5.2f}'.format(i/data['Group_Age'].value_counts().sum()*100)) + "%" for i in data['Group_Age'].value_counts()]

lbl = ["Age" + " " + el[0] + " = " + el[1] for el in zip(data['Group_Age'].value_counts().index, perc)]

squarify.plot(sizes=sizes, label=lbl, alpha=.8,ax=axis1)





plt.title('Age Group Of Peoples')

sn.barplot(x='Group_Age',y='target',hue='sex',data=data,palette=sn.cubehelix_palette(),ci=None,ax=axis2)

plt.title('Age Group Vs Sex Vs Target')

sn.barplot(x='sex',y='target',data=data,ci=None,ax=axis3)

plt.title('Sex Vs Target')
plt.style.use('ggplot')

fig,(axis1,axis2) = plt.subplots(1,2,figsize=(20,6))



sn.boxenplot(x='cp',y='trestbps',hue='target',data=data,ax=axis1)

sn.barplot(x='cp',y='target',data=data,ci=None,ax=axis2)
sn.barplot(x='cp',y='target',hue='sex',palette='winter_r',ci=None,data=data)
sn.set_style("ticks")



fig,(axis1,axis2,axis3) = plt.subplots(1,3,figsize=(20,5))

sn.swarmplot(x='fbs',y='age',hue='target',data=data,ax=axis1)

sn.boxenplot(x='fbs',y='age',hue='target' ,data=data,ax=axis2)

sn.barplot(x='fbs',y='target',data=data,palette='summer',ci=None,ax=axis3)
sn.set_style("ticks")



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(20,6))

sn.boxenplot(y='chol',x='target',hue='sex',data=data,ax=axis1)

sn.swarmplot(x='target',y='chol',hue='cp',data=data,ax=axis2)
groups_mean = data.groupby('target')['chol'].mean()

groups_std = data.groupby('target')['chol'].std()



groups = pa.DataFrame({'Group':[0,1],'Sample_chol_Mean':groups_mean.values,'Sample_chol_Std':groups_std.values,

                       'Sample_Size':[len(data.thalach[data['target'] == 0]),len(data.thalach[data['target'] == 1])]})

groups
stats.ttest_ind(data.chol[data['target'] == 0],data.chol[data['target'] == 1],equal_var=False)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(20,5))



sn.barplot(x='exang',y='target',data=data,palette='winter_r',ci=None,ax=axis1)

sn.boxenplot(x='target',y='thalach',hue='exang',data=data,ax=axis2)

plt.figure(figsize=(20,8))



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(25,6))

sn.lineplot(x='age',y='thalach',hue='target',data=data,ax=axis1)

sn.lineplot(x='age',y='thalach',hue='exang',data=data,ax=axis2)

pa.crosstab(data['thal'],data['cp']).plot(kind="bar",figsize=(25,5))



fig, (axis1,axis2) = plt.subplots(1,2,figsize=(25,6))





#sn.barplot(x='thal',y='target',data=data,palette='winter_r',ci=None,ax=axis1)

sn.pointplot(x='thal',y='target',data=data,ax=axis1)



labels = data['thal'].value_counts().index

sizes = data['thal'].value_counts().values



perc = [str('{:5.2f}'.format(i/data['thal'].value_counts().sum()*100)) + "%" for i in data['thal'].value_counts()]

lbl = [str(el[0]) + " = " + el[1] for el in zip(data['thal'].value_counts().index, perc)]

squarify.plot(sizes=sizes, label=lbl, alpha=.8,ax=axis2)



plt.figure(figsize=(25,4))

sn.barplot(x='thal',y='target',hue='cp',palette='summer',ci=None,data=data)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(20,5))



#sn.barplot(x='ca',y='target',hue='sex',data=data,palette='winter_r',ci=None,ax=axis1)

sn.pointplot(x='ca',y='target',hue='sex',data=data,ax=axis1)

sn.barplot(x='ca',y='target',data=data,ci=None,ax=axis2)
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(20,5))



sn.barplot(x='cp',y='target',hue='exang',data=data,palette='spring',ci=None,ax=axis1)

sn.countplot(data.cp[data['exang'] == 1],ax=axis2)
#plt.figure(figsize=(25,5))

#sn.lineplot(x='oldpeak',y='target',hue='slope',data=data)

#sns.set(style="ticks", context="talk")

#plt.style.use("dark_background")

sn.set_style("ticks")





fig, (axis1,axis2) = plt.subplots(1,2,figsize=(20,5))



sn.barplot(x='slope',y='target',data=data,palette='winter_r',ci=None,ax=axis1)

sn.barplot(x='restecg',y='target',data=data,palette='spring',ci=None,ax=axis2)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(20,6))

sn.swarmplot(x='target',y='oldpeak',hue='slope',data=data,ax=axis1)

sn.barplot(x='target',y='oldpeak',data=data,ci=None,ax=axis2)



plt.figure(figsize=(25,5))



sn.pointplot(x='restecg',y='target',hue='sex',data=data)
#data['target'] = data['target'].astype(str)

sn.pairplot(data,vars=['age','trestbps','chol','oldpeak','thalach'],hue='target',palette="husl")
columns = list(data.columns)

columns.remove('Group_Age')



plt.figure(figsize=(15,8))

sn.heatmap(data[columns].corr(),annot=True,cmap='BuPu')
catagorical_features = ['sex','cp','fbs','restecg','exang','slope','ca','thal']



for items in catagorical_features:

    data[items] = data[items].astype('category')#convert the catagorical variable which are actually catagorical into dummy variable



target = data['target']

data = data.drop(['target','Group_Age'],axis=1)



#### Convert into dummy variables

data_encoded = pa.get_dummies(data[data.columns],drop_first=True)
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import RFECV

from sklearn.model_selection import train_test_split



X_train,X_test,Y_train,Y_test = train_test_split(data_encoded,target,test_size=0.3,random_state=42)



random_class = RandomForestClassifier(random_state=42)



rfcev  = RFECV(estimator=random_class,step=1,cv=5,scoring='accuracy')



rfcev_model = rfcev.fit(X_train,Y_train)



print("The optimal number of features is {}".format(rfcev_model.n_features_))

print("Best Features:",X_train.columns[rfcev_model.support_])
optimal_features = list(data_encoded.columns[rfcev_model.support_])

data_encoded = data_encoded[optimal_features]

scores = []
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()

data_encoded = standard_scaler.fit_transform(data_encoded[['age','trestbps','chol','thalach','oldpeak']])
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

import xgboost as xgb

import lightgbm as lgm
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedShuffleSplit

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
class model_creation:

    def __init__(self,model_name,model,estimators,cv=5):

        self.model_name = model_name

        self.model = model

        self.estimators = estimators

        self.cv = cv

        

    def training(self):

        X_train,X_test,Y_train,Y_test = train_test_split(data_encoded,target,test_size = 0.3,random_state=42)

        cross_val = StratifiedShuffleSplit(n_splits=5,test_size=0.2,random_state=42)#Cross validation

        scoring = 'roc_auc'

        best_model = self.hyper_parameter_tuning(X_train,Y_train,cross_val,scoring)

        helper_function = self.helper_function(X_train,Y_train,X_test,Y_test,best_model,scoring)

        

        return best_model

        

    def hyper_parameter_tuning(self,X_train,Y_train,cross_val,scoring):

        

        grid_model_initialize = RandomizedSearchCV(self.model,self.estimators,cv=cross_val,scoring=scoring)

        grid_model = grid_model_initialize.fit(X_train,Y_train)

        return grid_model,grid_model.best_score_

    

    def helper_function(self,X_train,Y_train,X_test,Y_test,best_model,scoring):

        

        final_model = best_model[0].best_estimator_.fit(X_train,Y_train)

        scores  = cross_val_score(final_model,X_train,Y_train,cv=5,scoring=scoring,verbose=0)

        

        cross_mean = scores.mean()

        cross_std = scores.std()

        

        test_score = final_model.score(X_test,Y_test)

        

        ## Draw Confusion Matrix.

        fig, (axis1,axis2) = plt.subplots(1,2,figsize=(14,5))



    

        predicted_value = best_model[0].best_estimator_.predict(X_test)

        cm = metrics.confusion_matrix(Y_test,predicted_value)

        sn.heatmap(cm,annot=True,fmt=".2f",cmap="Greens",ax=axis1).set_title("Confusion Matrix") 

        

        ## Draw Roc Curve

    

        test_results_df = pa.DataFrame({'actual':Y_test})

        test_results_df = test_results_df.reset_index()



        predict_probabilites = pa.DataFrame(best_model[0].best_estimator_.predict_proba(X_test))

        test_results_df['chd_1'] = predict_probabilites.iloc[:,1:2]



        fpr,tpr,thresholds = metrics.roc_curve(test_results_df.actual,test_results_df.chd_1,drop_intermediate=False)



        auc_score = metrics.roc_auc_score(test_results_df.actual,test_results_df.chd_1)



        plt.plot(fpr,tpr,label="ROC Curve (area = %.2f)"% auc_score)

        plt.plot([0,1],[0,1],'k--')

        plt.xlim([0.0,1.0])

        plt.ylim([0.0,1.05])

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.legend(loc='lower right')



        ## print classification rreport



        print(metrics.classification_report(Y_test,predicted_value))

        
random = RandomForestClassifier()



n_estimators = [int(x) for x in np.linspace(start=200,stop=2000,num=10)] #Boosting parameters

max_features = ['auto', 'sqrt']# Boosting Parameters

max_depth = [int(x) for x in np.linspace(10,200,num=20)] #Max depth of the tree

max_depth.append(None)

bootstrap = [True,False] # Bootstrap here means how the samples will be chosen with or without replacement



# Total Combination 10*2*20*2 = 800 !



param_grid = {'n_estimators':[10,20,30],

              'max_features':max_features,

              'bootstrap':bootstrap}



radnomobj = model_creation('RF',random,param_grid)

random_mod = radnomobj.training()

scores.append(random_mod[1])
param_grid = {'n_neighbors':[x for x in range(1,40)],'weights':['uniform','distance']}



knn_model =  KNeighborsClassifier()



knnobj = model_creation('KNN',knn_model,param_grid)

knn_mod = knnobj.training()

scores.append(knn_mod[1])
logit_model = LogisticRegression()



param_grid = {'C':[0.001,0.01,0.05,1,100],'penalty':['l1','l2']}



logitobj = model_creation('Logit',logit_model,param_grid)

logit_mod = logitobj.training()

scores.append(logit_mod[1])
ada_model = AdaBoostClassifier()



param_grid = {'n_estimators':[int(x) for x in np.linspace(start=20,stop=300,num=15)],

              'learning_rate':np.arange(.1,4,.3)}



adaobj = model_creation('ADA',ada_model,param_grid)

ada_mod = adaobj.training()

scores.append(ada_mod[1])
n_estimators = [int(x) for x in np.linspace(start=20,stop=120,num=6)]

learning_rate = [0.1,0.01,0.05,0.001]

max_depth= np.arange(2,5,1)





param_grid = {'n_estimators':n_estimators,'learning_rate':learning_rate,'max_depth':max_depth}



grad_model = GradientBoostingClassifier()



grad_obj = model_creation('GRAD_BOOST',grad_model,param_grid)

grad_mod = grad_obj.training()

scores.append(grad_mod[1])
from xgboost.sklearn import XGBClassifier



param_grid = {'max_depth':range(3,8,2),'min_child_weight':range(1,10,2),'gamma':[0.5,1,1.5,2,5],

              'subsample':[0.6,0.8,1.0],'colsample_bytree':[0.6,0.8,1.0]}



xgboost_model = XGBClassifier(learning_rate=0.025,n_estimators=600,objective='binary:logistic',silent=True,nthread=1)



xgb_obj = model_creation('XGBOOST',xgboost_model,param_grid)

xgb_mod = xgb_obj.training()

scores.append(xgb_mod[1])
model_scores = pa.DataFrame({'Name':['Random_F','KNN','Logit','ADA','GRAD_B','XGB_B'],'Scores':scores})

plt.figure(figsize=(20,5))

sn.barplot(x='Name',y='Scores',data=model_scores,palette='winter_r')