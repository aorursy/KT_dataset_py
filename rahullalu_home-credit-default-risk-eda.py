# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import os
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from lightgbm.sklearn import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

#Dataset view
path1="../input/home-credit-default-risk/"
data_files=list(os.listdir(path1))
df_files=pd.DataFrame(data_files,columns=['File_Name'])
df_files['Size_in_MB']=df_files.File_Name.apply(lambda x:round(os.stat(path1+x).st_size/(1024*1024),2))
df_files
desc=pd.read_csv("../input/columns-description/columns_description.csv")
#All functions

#FUNCTION FOR PROVIDING FEATURE SUMMARY
def feature_summary(df_fa):
    print('DataFrame shape')
    print('rows:',df_fa.shape[0])
    print('cols:',df_fa.shape[1])
    col_list=['Null','Unique_Count','Data_type','Max/Min','Mean','Std','Skewness','Sample_values']
    df=pd.DataFrame(index=df_fa.columns,columns=col_list)
    df['Null']=list([len(df_fa[col][df_fa[col].isnull()]) for i,col in enumerate(df_fa.columns)])
    #df['%_Null']=list([len(df_fa[col][df_fa[col].isnull()])/df_fa.shape[0]*100 for i,col in enumerate(df_fa.columns)])
    df['Unique_Count']=list([len(df_fa[col].unique()) for i,col in enumerate(df_fa.columns)])
    df['Data_type']=list([df_fa[col].dtype for i,col in enumerate(df_fa.columns)])
    for i,col in enumerate(df_fa.columns):
        if 'float' in str(df_fa[col].dtype) or 'int' in str(df_fa[col].dtype):
            df.at[col,'Max/Min']=str(round(df_fa[col].max(),2))+'/'+str(round(df_fa[col].min(),2))
            df.at[col,'Mean']=df_fa[col].mean()
            df.at[col,'Std']=df_fa[col].std()
            df.at[col,'Skewness']=df_fa[col].skew()
        df.at[col,'Sample_values']=list(df_fa[col].unique())
           
    return(df.fillna('-'))


def drop_corr_col(df_corr):
    upper = df_corr.where(np.triu(np.ones(df_corr.shape),
                          k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    return(to_drop)

#Reading training data
train=pd.read_csv(path1+'application_train.csv')
print('application_train Feature Summary')
with pd.option_context('display.max_rows',train.shape[1]):
    train_fs=feature_summary(train)   
#Understanding target/dependent variable
pie_labels=['Defaulters-'+str(train['TARGET'][train.TARGET==1].count()),'Non Defaulters-'+str(train['TARGET'][train.TARGET==0].count())]
pie_share=[train['TARGET'][train.TARGET==1].count()/train['TARGET'].count(),
           train['TARGET'][train.TARGET==0].count()/train['TARGET'].count()]
figureObject, axesObject = plt.subplots()
pie_colors=('red','green')
pie_explode=(.3,.0)
axesObject.pie(pie_share,labels=pie_labels,explode=pie_explode,autopct='%.2f%%',colors=pie_colors,startangle=0,shadow=True)
axesObject.axis('equal')
plt.title('Percentage of Defaulters and Non Defaulters',color='blue')
plt.show()
print('FEATURE SUMMARY: Categorical Features')
cat_features=train_fs[train_fs.Data_type=='object'].index
print('Total categorical features:',len(cat_features))
cat_fs=train_fs[train_fs.Data_type=='object']
cat_fs['Desc']=cat_fs.index
for ind in cat_fs['Desc'].values:
    cat_fs.at[ind,'Desc']=desc.Description[(desc.Table=='application') & (desc.Row==ind)].values[0]
display(cat_fs.iloc[:,7:9])
display(cat_fs.iloc[:,:3])
#Replacing space with underscore in all categorical values
for col in cat_features:
    train[col]=train[col].apply(lambda x: str(x).replace(" ","_"))
   
#converting call categorical features into dummies 
cat_train=pd.DataFrame()
for col in cat_features:
    dummy=pd.get_dummies(train[col],prefix=col)
    cat_train=pd.concat([cat_train,dummy],axis=1)
cat_train.head()
del dummy
gc.collect()
print('Newly created dummy columns:',len(cat_train.columns))
%%time
#creating correlation matrix with absolute values
corr=cat_train.corr().abs()
#identifying features with high correlation value
to_drop=drop_corr_col(corr)
cat_train.drop(to_drop,axis=1,inplace=True)
print('Drop following features as they have high correlation other columns:\n',to_drop,'\n')
print('Categorical Features after dropping correlated features:',cat_train.shape)
train_X,test_X,train_y,test_y=train_test_split(cat_train,train['TARGET'],random_state=200)
model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
model.fit(train_X,train_y)
print('Createing a basic LGBM classifier on categorical data. To check can newly created features be consumed by a model')
print('roc auc score:',roc_auc_score(test_y,model.predict_proba(test_X)[:,1]))
indices = np.argsort(model.feature_importances_)[::-1]
names = [cat_train.columns[i] for i in indices[:138]]
fig,ax=plt.subplots(figsize=(20,40))
plt.title("Feature Importance - Categorical Features",fontsize=35)
plt.ylabel("Categorical Features",fontsize=35)
plt.xlabel("Feature Importance",fontsize=35)
df_fi_cat=pd.DataFrame(model.feature_importances_[indices[:138]],columns=['Feature_imp'])
df_fi_cat['names']=names
df_fi_cat.sort_values(by='Feature_imp',inplace=True)
plt.barh(range(138),df_fi_cat['Feature_imp'],align='edge')
plt.yticks(range(138),df_fi_cat['names'],color='g',fontsize=15)
for i in range(0,138,2):
    ax.get_yticklabels()[i].set_color("red")

plt.show()
chi2_selector=SelectKBest(chi2,k=138)
feature_kbest=chi2_selector.fit_transform(cat_train,train['TARGET'])
df_chi=pd.DataFrame(chi2_selector.scores_,columns=['chi_score'])
df_chi['columns']=cat_train.columns
df_chi_s=df_chi.sort_values(by='chi_score')
fig,ax=plt.subplots(figsize=(20,40))
plt.title("Chi-squared statistics for categorical features",fontsize=30)
plt.ylabel("Categorical Features",fontsize=30)
plt.xlabel("Chi-squared statistic",fontsize=30)
plt.barh(range(len(df_chi_s['chi_score'])),df_chi_s['chi_score'],align='edge',color='rgbkymc')
plt.yticks(range(len(df_chi_s['chi_score'])),df_chi_s['columns'],color='g',fontsize=15)
for i in range(0,138,2):
    ax.get_yticklabels()[i].set_color("red")
plt.show()
print('Feature with Chi-square statistic less than 1:',len(df_chi_s[df_chi_s.chi_score<1]['columns']))
cat_train.drop(df_chi_s[df_chi_s.chi_score<1]['columns'],axis=1,inplace=True)
train_X,test_X,train_y,test_y=train_test_split(cat_train,train['TARGET'],random_state=200)
model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
model.fit(train_X,train_y)
print('Score after dropping features with Chi-squared statistic less than 1')
print('roc auc score:',roc_auc_score(test_y,model.predict_proba(test_X)[:,1]))
%%time
shape_1=cat_train.shape[1]
roc_auc=np.zeros([len(range(10,shape_1,5)),2],float)
k=0
df_chi_s.sort_values(by='chi_score',ascending=False,inplace=True)
for i in range(10,shape_1,5):
    train_X,test_X,train_y,test_y=train_test_split(cat_train[df_chi_s['columns'][:i]],train['TARGET'],random_state=200)
    model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
    model.fit(train_X,train_y)
    roc_auc[k][0]=i
    roc_auc[k][1]=roc_auc_score(test_y,model.predict_proba(test_X)[:,1])                                
    k=k+1

df_roc=pd.DataFrame(roc_auc,columns=['Features','roc_auc_score'])
df_roc.sort_values(by='roc_auc_score',inplace=True,ascending=False)
print('Top five roc_auc_scores with Feature count')
df_roc.head()
df_roc.sort_values(by='Features',inplace=True)
plt.figure(figsize=(40,10))
plt.title("Categorical Feature selection and roc_auc_score - highlighting top 5",fontsize=30)
plt.xlabel("Feature count",fontsize=30)
plt.ylabel("roc_auc_score",fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(df_roc['Features'],df_roc['roc_auc_score'],color='b',linewidth=2)
plt.hlines(xmin=0,xmax=np.max(roc_auc[:,0]),y=np.max(roc_auc[:,1]),color='g',linestyle='dashed')
df_roc.sort_values(by='roc_auc_score',inplace=True,ascending=False)
for i in range(len(df_roc)):
    plt.plot(df_roc.iloc[i,0],df_roc.iloc[i,1],'bo')
    if i<=4:
        plt.text(df_roc.iloc[i,0],df_roc.iloc[i,1],(('('+str(np.int(df_roc.iloc[i,0]))+','+str(round(df_roc.iloc[i,1],4))+')')),color='r',fontsize=28,rotation=90)
        plt.vlines(ymin=0.613,ymax=df_roc.iloc[i,1],x=df_roc.iloc[i,0],color='r',linestyle='dotted')
    if i==16:
        plt.text(df_roc.iloc[i,0],df_roc.iloc[i,1],(('('+'features'+','+'Score')+')'),color='r',fontsize=28,rotation=90)
        plt.vlines(ymin=0.613,ymax=df_roc.iloc[i,1],x=df_roc.iloc[i,0],color='r',linestyle='dotted')
plt.show()
print('FEATURE SUMMARY: Binary Features')
bin_features=train_fs[((train_fs.Data_type=='int64') | (train_fs.Data_type=='float64')) & (train_fs.Unique_Count==2)].index
print('Total binary features (including TARGET):',len(bin_features))
bin_fs=train_fs[((train_fs.Data_type=='int64') | (train_fs.Data_type=='float64')) & (train_fs.Unique_Count==2)]
bin_fs['Desc']=bin_fs.index
for ind in bin_fs['Desc'].values:
    bin_fs.at[ind,'Desc']=desc.Description[(desc.Table=='application') & (desc.Row==ind)].values[0]
display(bin_fs.iloc[:,7:9])
display(bin_fs.iloc[:,:7])
corr_bin=train[bin_features[1:]].corr().abs()
print('Number of strongly correlated binary features:',drop_corr_col(corr_bin))
train_X,test_X,train_y,test_y=train_test_split(train[bin_features].drop(['TARGET'],axis=1),train['TARGET'],random_state=200)
model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
model.fit(train_X,train_y)
print('Creating a basic LGBM classifier on binary features')
print('roc auc score:',roc_auc_score(test_y,model.predict_proba(test_X)[:,1]))
indices = np.argsort(model.feature_importances_,)[::-1]
names = [train[bin_features].drop(['TARGET'],axis=1).columns[i] for i in indices]
plt.figure(figsize=(30,10))
plt.title("Feature Importance - Top Binary Features",fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Binary Features",fontsize=30)
plt.ylabel("Feature Importance",fontsize=30)
plt.bar(range(32), model.feature_importances_[indices])
plt.xticks(range(32), names,rotation=90)

plt.show()
chi2_selector=SelectKBest(chi2,k=32)
feature_kbest=chi2_selector.fit_transform(train[bin_features].drop(['TARGET'],axis=1),train['TARGET'])
df_chi=pd.DataFrame(chi2_selector.scores_,columns=['chi_score'])
df_chi['columns']=bin_features[1:]
df_chi_bins=df_chi.sort_values(by='chi_score',ascending=False)
plt.figure(figsize=(40,10))
plt.title("Chi-squared statistics for binary features",fontsize=30)
plt.xlabel("Binary Features",fontsize=30)
plt.ylabel("Chi-squared statistics",fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.bar(range(len(df_chi_bins['chi_score'])),df_chi_bins['chi_score'],align='edge',color='rgbkymc')
plt.xticks(range(len(df_chi_bins['chi_score'])),df_chi_bins['columns'],rotation=90,color='g')
plt.show()
%%time
roc_auc_bin=np.zeros([len(range(5,33,3)),2],float)
k=0

for i in range(5,33,3):
    train_X,test_X,train_y,test_y=train_test_split(train[df_chi_bins['columns'][:i]],train['TARGET'],random_state=200)
    model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
    model.fit(train_X,train_y)
    roc_auc_bin[k][0]=i
    roc_auc_bin[k][1]=roc_auc_score(test_y,model.predict_proba(test_X)[:,1])
    k=k+1
df_roc_bin=pd.DataFrame(roc_auc_bin,columns=['Features','roc_auc_score'])
df_roc_bin.sort_values(by='roc_auc_score',inplace=True,ascending=False)
print('Top five roc_auc_scores with Feature count')
df_roc_bin.head()
df_roc_bin.sort_values(by='Features',inplace=True)
plt.figure(figsize=(40,10))
plt.title("Binary Feature selection and roc_auc_score - highlighting top 5",fontsize=30)
plt.xlabel("Feature count",fontsize=30)
plt.ylabel("roc_auc_score",fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(df_roc_bin['Features'],df_roc_bin['roc_auc_score'],color='b',linewidth=3)
plt.hlines(xmin=0,xmax=np.max(roc_auc_bin[:,0]),y=np.max(roc_auc_bin[:,1]),color='g',linestyle='dashed')
df_roc_bin.sort_values(by='roc_auc_score',inplace=True,ascending=False)
for i in range(len(df_roc_bin)):
    plt.plot(df_roc_bin.iloc[i,0],df_roc_bin.iloc[i,1],'bo')
    if i<=4:
        plt.text(df_roc_bin.iloc[i,0],df_roc_bin.iloc[i,1],
                 (('('+str(np.int(df_roc_bin.iloc[i,0]))+','+str(round(df_roc_bin.iloc[i,1],4))+')')),color='r',fontsize=25,rotation=90)
        plt.vlines(ymin=0.563,ymax=df_roc_bin.iloc[i,1],x=df_roc_bin.iloc[i,0],color='r',linestyle='dotted')
    if i==8:
        plt.text(df_roc_bin.iloc[i,0],df_roc_bin.iloc[i,1],(('('+'features'+','+'Score')+')'),color='r',fontsize=25,rotation=90)
        plt.vlines(ymin=0.563,ymax=df_roc_bin.iloc[i,1],x=df_roc_bin.iloc[i,0],color='r',linestyle='dotted')
plt.show()
print('FEATURE SUMMARY: Continuous Features')
con_features=train_fs[((train_fs.Data_type=='float64') | (train_fs.Data_type=='int64')) & (train_fs.Unique_Count!=2)].index
print('Total continuous features:',len(con_features))
con_fs=train_fs[((train_fs.Data_type=='float64') | (train_fs.Data_type=='int64')) & (train_fs.Unique_Count!=2)]
con_fs['Desc']=con_fs.index
for ind in con_fs['Desc'].values:
    con_fs.at[ind,'Desc']=desc.Description[(desc.Table=='application') & (desc.Row==ind)].values[0]
with pd.option_context('display.max_rows',train.shape[1]):
    display(con_fs.iloc[:,7:9])
    display(con_fs.iloc[:,:7])
train_X,test_X,train_y,test_y=train_test_split(train[con_features].drop(['SK_ID_CURR'],axis=1),train['TARGET'],random_state=200)
model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
model.fit(train_X,train_y)
print('Creating a basic model on continuous features')
print('roc auc score',roc_auc_score(test_y,model.predict_proba(test_X)[:,1]))
indices = np.argsort(model.feature_importances_)[::-1]
names = [train[con_features].drop(['SK_ID_CURR'],axis=1).columns[i] for i in indices[:]]
plt.figure(figsize=(35,10))
plt.title("Feature Importance - Continuous Features",fontsize=30)
plt.xlabel("Continuous Features",fontsize=30)
plt.ylabel("Feature importance",fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.bar(range(72), model.feature_importances_[indices[:]])
plt.xticks(range(72), names,rotation=90)

plt.show()
#Updated null values for continuous features their mean value 
for col in con_features[1:]:
    if train_fs.at[col,'Null']!=0:
        train[col]=train[col].fillna(train_fs.at[col,'Mean'])
Fvalue_selector=SelectKBest(f_classif,k=72)
feature_kbest=Fvalue_selector.fit_transform(train[con_features[1:]],train['TARGET'])
df_Fvalue=pd.DataFrame(Fvalue_selector.scores_,columns=['F-value'])
df_Fvalue['columns']=con_features[1:]
df_Fvalue_s=df_Fvalue.sort_values(by='F-value',ascending=False)
plt.figure(figsize=(40,10))
plt.title("F-value for continuous features",fontsize=30)
plt.xlabel("Continuous Features",fontsize=30)
plt.ylabel("F-value statistics",fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.bar(range(len(df_Fvalue_s)),df_Fvalue_s['F-value'],align='edge',color='rgbkymc')
plt.xticks(range(len(df_Fvalue_s)),df_Fvalue_s['columns'],rotation=90,color='g')
plt.show()
%%time
roc_auc_con=np.zeros([len(range(4,73,4)),2],float)
k=0

for i in range(4,73,4):
    train_X,test_X,train_y,test_y=train_test_split(train[df_Fvalue_s['columns'][:i]],train['TARGET'],random_state=200)
    model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
    model.fit(train_X,train_y)
    roc_auc_con[k][0]=i
    roc_auc_con[k][1]=roc_auc_score(test_y,model.predict_proba(test_X)[:,1])
    k=k+1
df_roc_con=pd.DataFrame(roc_auc_con,columns=['Features','roc_auc_score'])
df_roc_con.sort_values(by='roc_auc_score',inplace=True,ascending=False)
print('Top five roc_auc_scores with Feature count')
df_roc_con.head()
df_roc_con.sort_values(by='Features',inplace=True)
plt.figure(figsize=(40,10))
plt.title("Continuous Feature selection and roc_auc_score - highlighting top 5",fontsize=30)
plt.xlabel("Feature count",fontsize=30)
plt.ylabel("roc_auc_score",fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(df_roc_con['Features'],df_roc_con['roc_auc_score'],color='b',linewidth=3)
plt.hlines(xmin=0,xmax=np.max(roc_auc_con[:,0]),y=np.max(roc_auc_con[:,1]),color='g',linestyle='dashed')
df_roc_con.sort_values(by='roc_auc_score',inplace=True,ascending=False)
for i in range(len(df_roc_con)):
    plt.plot(df_roc_con.iloc[i,0],df_roc_con.iloc[i,1],'bo')
    if i<=4:
        plt.text(df_roc_con.iloc[i,0],df_roc_con.iloc[i,1],
                 (('('+str(np.int(df_roc_con.iloc[i,0]))+','+str(round(df_roc_con.iloc[i,1],4))+')')),color='r',fontsize=25,rotation=90)
        plt.vlines(ymin=0.73,ymax=df_roc_con.iloc[i,1],x=df_roc_con.iloc[i,0],color='r',linestyle='dotted')
    if i==15:
        plt.text(df_roc_con.iloc[i,0],df_roc_con.iloc[i,1],(('('+'features'+','+'Score')+')'),color='r',fontsize=25,rotation=90)
        plt.vlines(ymin=0.73,ymax=df_roc_con.iloc[i,1],x=df_roc_con.iloc[i,0],color='r',linestyle='dotted')
plt.show()
print('Concatenating all categorical, binary and continuous features ')
final_train=pd.concat([cat_train,train[bin_features],train[con_features]],axis=1)
print('Shape of final training data set:',final_train.shape)


del train,cat_train,bin_features,model,con_features
gc.collect()
train_X,test_X,train_y,test_y=train_test_split(final_train.drop(['TARGET','SK_ID_CURR'],axis=1),final_train['TARGET'])
model =LGBMClassifier(learning_rate=0.05,n_estimators=200,n_jobs=-1,reg_alpha=0.1,min_split_gain=.1,verbose=-1)
model.fit(train_X,train_y)
print('Creating a final LGBM classifier on final training dataset')
print('roc auc score:',roc_auc_score(test_y,model.predict_proba(test_X)[:,1]))
indices = np.argsort(model.feature_importances_)[::-1]
names = [final_train.columns[i] for i in indices[:50]]
plt.figure(figsize=(30,10))
plt.title("Feature Importance - Top 50 Features",fontsize=30)
plt.xlabel("Features",fontsize=30)
plt.ylabel("Feature Importance",fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.bar(range(50), model.feature_importances_[indices[:50]])
plt.xticks(range(50), names,rotation=90)

plt.show()