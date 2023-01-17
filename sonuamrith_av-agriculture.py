import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sb
import matplotlib.pyplot as plt
import os
from collections import Counter
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
agri=pd.read_csv('../input/av-janatahack-machine-learning-in-agriculture/train_yaOffsB.csv')
agri.head()
agri_test=pd.read_csv('../input/av-janatahack-machine-learning-in-agriculture/test_pFkWwen.csv')
agri_test.head()
print('No of rows present in the dataset',agri.shape[0])
agri.info()
#Which of these columns are categorical columns
print(np.unique(agri['Crop_Type'].values))
print(np.unique(agri['Soil_Type'].values))
print(np.unique(agri['Pesticide_Use_Category'].values))
print(agri['Number_Doses_Week'].values)
print(np.unique(agri['Season'].values))
#Are there any missing values present in this dataset
agri.isnull().sum()#So the column number-week-used has 9000 missing rows
#Lets check out the distribution of classes to know whether its balanced or imbalanced
count=Counter(agri['Crop_Damage'].values)
sb.barplot(list(count.keys()),list(count.values()))
#Lots of alive crops than non alive crops so clearly its a imbalanced classification problem
ax=sb.scatterplot(x=agri['Estimated_Insects_Count'].values,y=agri['Number_Weeks_Quit'].values,hue=agri['Crop_Damage'].values,palette='viridis_r')
ax.set_xlabel('Insects_count')
ax.set_ylabel('Number_weeks_quit')
#Observe two things here i.e as the number of weeks with no pesticide increases then insect count also increases
#Another point is as the insect_count increases then crop damage due to other reasons are increasing
#Less data i.e imbalnced data isnt giving much insights why not make it a balanced classification
#Handling missing values
from sklearn.linear_model import LinearRegression
log=LinearRegression()
cropts=agri.loc[agri['Number_Weeks_Used'].isnull(),['Crop_Type','Number_Weeks_Used']].values
croptr=agri.loc[agri['Number_Weeks_Used'].notnull(),['Crop_Type','Number_Weeks_Used']].values
print(croptr.shape)
print(cropts.shape)
log.fit(croptr[:,0].reshape(-1,1),croptr[:,1])
crop_pred=log.predict(cropts[:,0].reshape(-1,1))
#Imputing missing values
agri.loc[agri['Number_Weeks_Used'].isnull(),'Number_Weeks_Used']=crop_pred

#Now lets check if there are any missing values after imputing
agri.isnull().sum()
print(count.values())
X=agri.loc[:,agri.columns.values[1:-1]]
Y=agri.loc[:,'Crop_Damage']
#Checking the score of baseline models 
#Here we are not using any tricks of feature engineering,outlier removal or removing those features which are correlated or feature scaling
#Yet i want to know which model performs despite all of this on this data
log=LogisticRegression(class_weight='balanced',max_iter=1000)
kn=KNeighborsClassifier(weights='distance')
decision=DecisionTreeClassifier(class_weight='balanced')
svc=SVC(class_weight='balanced') 
nb=GaussianNB()
models=[]
models.append(('logistic',log))
models.append(('knn',kn))
models.append(('decision',decision))
models.append(('nb',nb))
results=[]
names=[]
for name,model in tqdm(models):
    fold=KFold(n_splits=6)
    cv_results=cross_val_score(model,X,Y,scoring='f1_weighted',cv=fold)
    results.append(cv_results)
    names.append(name)
sb.boxplot(x=names,y=results)
pipelines=[]
pipelines.append(('Scaled_logistic',Pipeline([('scaler',StandardScaler()),('LOG',LogisticRegression(class_weight='balanced',max_iter=1000))])))
pipelines.append(('Scaled_KNN',Pipeline([('scaler',StandardScaler()),('knn',KNeighborsClassifier(weights='distance'))])))
pipelines.append(('Scaled_Decision',Pipeline([('scaler',StandardScaler()),('decision',DecisionTreeClassifier(class_weight='balanced'))])))
pipelines.append(('Scaled_nb',Pipeline([('scaler',StandardScaler()),('nb',GaussianNB())])))
std_results=[]
std_name=[]
for name,model in pipelines:
    print(model)
    fold=KFold(n_splits=6)
    cv_std=cross_val_score(model,X,Y,cv=fold,scoring='f1_weighted')
    std_results.append(cv_std)
    std_name.append(name)
sb.boxplot(x=std_name,y=std_results)
agri.head()
#Let see the effect of removal of outlier and see whether it increases any performance
integer=['Estimated_Insects_Count','Number_Doses_Week','Number_Weeks_Used','Number_Weeks_Quit']
insect=np.percentile(agri['Estimated_Insects_Count'].values,np.arange(0,110,10))
plt.figure(figsize=(6,4))
plt.subplot(121)
plt.plot(np.arange(0,110,10),insect)
zin_insect=np.percentile(agri['Estimated_Insects_Count'].values,np.linspace(98,100,10))
plt.subplot(122)
plt.plot(np.linspace(98,100,10),zin_insect)
plt.subplots_adjust(right=2.5)
insect=np.percentile(agri['Number_Doses_Week'].values,np.arange(0,110,10))
plt.figure(figsize=(6,4))
plt.subplot(121)
plt.plot(np.arange(0,110,10),insect)
zin_insect=np.percentile(agri['Number_Doses_Week'].values,np.linspace(99,100,5))
print(zin_insect)
plt.subplot(122)
plt.plot(np.linspace(99,100,5),zin_insect)
plt.subplots_adjust(right=2.5)
agri2=agri.loc[agri.loc[:,'Number_Doses_Week']!=95,:]
agri2.shape
week=np.percentile(agri['Number_Weeks_Used'].values,np.arange(0,110,10))
plt.figure(figsize=(6,4))
plt.subplot(121)
plt.plot(np.arange(0,110,10),week)
zin_week=np.percentile(agri['Number_Weeks_Used'].values,np.linspace(90,100,10))
print(zin_week)
plt.subplot(122)
plt.plot(np.linspace(90,100,10),zin_week)
plt.subplots_adjust(right=2.5)
week_q=np.percentile(agri['Number_Weeks_Quit'].values,np.arange(0,110,10))
plt.figure(figsize=(6,4))
plt.subplot(121)
plt.plot(np.arange(0,110,10),week_q)
zin_weekq=np.percentile(agri['Number_Weeks_Quit'].values,np.linspace(95,100,10))
print(zin_weekq)
plt.subplot(122)
plt.plot(np.linspace(95,100,10),zin_weekq)
plt.subplots_adjust(right=2.5)
agri2=agri2.loc[agri2.loc[:,'Number_Weeks_Quit']!=50,:]
agri2.shape
X1=agri2.loc[:,agri2.columns.values[1:-1]]
Y1=agri2.loc[:,'Crop_Damage'].values
print(X1.shape)
print(Y1.shape)
stdo_results=[]
stdo_name=[]
for name,model in pipelines:
    print(model)
    fold=KFold(n_splits=6)
    cv_std=cross_val_score(model,X1,Y1,cv=fold,scoring='f1_weighted')
    stdo_results.append(cv_std)
    stdo_name.append(name)
sb.boxplot(x=stdo_name,y=stdo_results)
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
over=SMOTE(sampling_strategy={1:int(0.4*74328),2:int(0.4*74328)})
under=RandomUnderSampler(sampling_strategy={0:int(2*22298)})
X2,Y2=over.fit_resample(agri2.loc[:,agri2.columns.values[1:-1]],agri2['Crop_Damage'].values)
print(X2.shape)
print(Y2.shape)
X2,Y2=under.fit_resample(X2,Y2)
print(X2.shape)
print(Y2.shape)
stdm_results=[]
stdm_name=[]
for name,model in pipelines:
    print(model)
    fold=KFold(n_splits=6)
    cv_std=cross_val_score(model,X2,Y2,cv=fold,scoring='f1_weighted')
    stdm_results.append(cv_std)
    stdm_name.append(name)
sb.boxplot(x=stdm_name,y=stdm_results)
#After doing SMOTE+undersampling lets see the class distribution of new dataset
c=Counter(Y2)
sb.barplot(list(c.keys()),list(c.values()))
from imblearn.under_sampling import TomekLinks
tomek=TomekLinks(sampling_strategy='majority')
X3,Y3=tomek.fit_resample(agri2.loc[:,agri2.columns.values[1:-1]],agri2['Crop_Damage'].values)
print(X3.shape,Y3.shape)
Counter(Y3)
stdu_results=[]
stdu_name=[]
for name,model in pipelines:
    print(model)
    fold=KFold(n_splits=6)
    cv_std=cross_val_score(model,X3,Y3,cv=fold,scoring='f1_weighted')
    stdu_results.append(cv_std)
    stdu_name.append(name)
sb.boxplot(x=stdu_name,y=stdu_results)
X2=pd.DataFrame(X2,columns=agri.columns[1:-1])
X2=X2.assign(Crop_Damage=Y2)
X2.head()
ax1=sb.scatterplot(x=X2['Estimated_Insects_Count'].values,y=X2['Number_Weeks_Quit'].values,hue=X2['Crop_Damage'].values,palette='rainbow')
ax1.set_xlabel('Insects_count')
ax1.set_ylabel('Number_weeks_quit')
#Which crop type had more crop damages and which crop type had more succesful harvest
ax2=sb.countplot(X2['Crop_Type'].values,hue=X2['Crop_Damage'].values)

Counter(X2['Crop_Type'].values)
print('Sucess percentage in crop type 0 is =',(32000/81672)*100)
print('Sucess percentage in crop type 1 is =',(12000/22386)*100)
ax3=sb.countplot(X2['Soil_Type'].values,hue=X2['Crop_Damage'].values,palette='viridis_r')
ax4=sb.countplot(X2['Pesticide_Use_Category'].values,hue=X2['Crop_Damage'].values,palette='Oranges')
ax5=sb.countplot(X2['Season'].values,hue=X2['Crop_Damage'].values,palette='coolwarm')
#Lets try dosage/day as a feature
X2.loc[:,'dosage/day']=X2.loc[:,'Number_Doses_Week'].apply(lambda x:x/7)
sb.boxplot(x=(X2['Crop_Damage'].values),y=X2['dosage/day'].values)
#Trying dosage/month as a feature
X2.loc[:,'dosage/month']=X2.loc[:,'Number_Doses_Week'].apply(lambda x:x*4.34)
sb.boxplot(x=X2['Crop_Damage'].values,y=X2['dosage/month'].values)
#One season =4 months,then dosage for whole season=dosage/month*4
X2.loc[:,'dosage/season']=X2.loc[:,'dosage/month'].apply(lambda x:x*4)
sb.boxplot(x=X2['Crop_Damage'].values,y=X2['dosage/season'].values)
#Lets see how much dosage is used per insect i.e insect/dosage ratio
def ins_dos(x,y):
    return(x/(y))
X2.loc[:,'insect/dosage']=X2.loc[:,['Estimated_Insects_Count','dosage/season']].apply(lambda x:ins_dos(x[0],x[1]),axis=1)
sb.boxplot(x=X2['Crop_Damage'].values,y=X2['insect/dosage'].values)
X2.head(5)
#Hypothesis
#1.Does the insects proportion decides in sucess of crop harvest i.e more the insects more the damage
#2.Does excess amount of pesticide without gap of quiting decides damage of harvest
#3.Does the crop type and pesticide combination decides crop harvest damage
#4.Does the soil type and pesticide combination decides harvest damage
#5.Does excess dossage for a crop decides harvest damage
#plotting distribution of insects count
plt.figure(figsize=(6,4))
plt.subplot(131)
plt.hist(X2.loc[X2.loc[:,'Crop_Damage']==0,'Estimated_Insects_Count'].values,density=True)
plt.xlabel('Crop_damage=0')
plt.subplot(132)
plt.hist(X2.loc[X2.loc[:,'Crop_Damage']==1,'Estimated_Insects_Count'].values,density=True)
plt.xlabel('Crop_damage=1')
plt.subplot(133)
plt.hist(X2.loc[X2.loc[:,'Crop_Damage']==2,'Estimated_Insects_Count'].values,density=True)
plt.xlabel('Crop_damage=2')
plt.subplots_adjust(right=2.5)
#Seeing the above graph in terms of percentiles
plt.figure(figsize=(6,4))
plt.subplot(131)
a=np.percentile(X2.loc[X2.loc[:,'Crop_Damage']==0,'Estimated_Insects_Count'].values,np.arange(0,110,10))
plt.plot(np.arange(0,110,10),a,color='r')
plt.scatter(np.arange(0,110,10),a)
plt.xlabel('Crop_damage=0')
plt.subplot(132)
b=np.percentile(X2.loc[X2.loc[:,'Crop_Damage']==1,'Estimated_Insects_Count'].values,np.arange(0,110,10))
plt.plot(np.arange(0,110,10),b,color='violet')
plt.scatter(np.arange(0,110,10),b,color='black')
plt.xlabel('Crop_damage=1')
plt.subplot(133)
c=np.percentile(X2.loc[X2.loc[:,'Crop_Damage']==2,'Estimated_Insects_Count'].values,np.arange(0,110,10))
plt.plot(np.arange(0,110,10),c)
plt.scatter(np.arange(0,110,10),c,color='red')
plt.xlabel('Crop_damage=2')
plt.subplots_adjust(right=2.5)
#Observations:
#1.When the crop harvest is sucessfull 80 percent of the values are below 2000
#2.When the crop harvest isnt successfull for other reasons 80 percent of the values are below 2500
#3.When the crop harvest isnt successfull because of pesticides 80 percent of the values of insects are below 
X2[(X2.loc[:,'Pesticide_Use_Category']==3) & (X2.loc[:,'Number_Weeks_Quit']==0)]
X2.head()
#What should be the ideal dosage for a particular pesticide category which gives success
sucess_pest=X2.loc[(X2.loc[:,'Pesticide_Use_Category']==2)&(X2.loc[:,'Crop_Damage']==0)]
#Lets plot the dosage ditribution
sb.distplot(np.log(sucess_pest.loc[:,'Number_Doses_Week'].values),kde=True)
#What will be the dosage for particular pesticide which gives harvest damage
fail_pest=X2.loc[(X2.loc[:,'Pesticide_Use_Category']==2)&(X2.loc[:,'Crop_Damage']==1)]
fail_pest2=X2.loc[(X2.loc[:,'Pesticide_Use_Category']==2)&(X2.loc[:,'Crop_Damage']==2)]

plt.figure(figsize=(6,4))
plt.subplot(131)
s=sb.distplot(np.log(np.cumsum(sucess_pest.loc[:,'Number_Doses_Week'].values)),color='red')
s.set_xlabel('Dosage range of pest2 with success')
plt.subplot(132)
s1=sb.distplot(np.log(np.cumsum(fail_pest.loc[:,'Number_Doses_Week'].values)),kde=True,color='brown')
s1.set_xlabel('Dosage range for pest2 with failure1')
plt.subplot(133)
s2=sb.distplot(np.log(np.cumsum(fail_pest2.loc[:,'Number_Doses_Week'].values)),kde=True,color='orange')
s2.set_xlabel('Dosage range for pest2 with failure due to pesticide')
plt.subplots_adjust(right=2.5)
#Lets make a new feature
X2.loc[(X2.loc[:,'Pesticide_Use_Category']==2)&(X2.loc[:,'Crop_Damage']==0),'Dosage_pest2']=np.log(np.cumsum(sucess_pest.loc[:,'Number_Doses_Week'].values))
X2.loc[(X2.loc[:,'Pesticide_Use_Category']==2)&(X2.loc[:,'Crop_Damage']==1),'Dosage_pest2']=np.log(np.cumsum(fail_pest.loc[:,'Number_Doses_Week'].values))
X2.loc[(X2.loc[:,'Pesticide_Use_Category']==2)&(X2.loc[:,'Crop_Damage']==2),'Dosage_pest2']=np.log(np.cumsum(fail_pest2.loc[:,'Number_Doses_Week'].values))
lin=LinearRegression()
lin.fit(X2.loc[X2.loc[:,'Dosage_pest2'].notnull(),'Crop_Type'].values.reshape(-1,1),X2.loc[X2.loc[:,'Dosage_pest2'].notnull(),'Dosage_pest2'].values)
X2.loc[X2.loc[:,'Dosage_pest2'].isnull(),'Dosage_pest2']=lin.predict(X2.loc[X2.loc[:,'Dosage_pest2'].isnull(),'Crop_Type'].values.reshape(-1,1))

sb.boxplot(x=X2.loc[:,'Crop_Damage'].values,y=X2.loc[:,'Dosage_pest2'].values)
pd.set_option('use_inf_as_na',True)
X2.loc[X2.loc[:,'insect/dosage'].isnull(),'insect/dosage']=X2.loc[X2.loc[:,'insect/dosage'].notnull(),'insect/dosage'].values.mean()
Xtr,Xcv,ytr,ycv=train_test_split(X2.loc[:,X2.columns.values!='Crop_Damage'].values,X2.loc[:,'Crop_Damage'].values,test_size=0.2)
print(Xtr.shape,ytr.shape)
print(Xcv.shape,ycv.shape)