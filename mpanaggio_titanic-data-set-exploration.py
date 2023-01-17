# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import display, HTML
pd.options.display.max_rows = 50
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
display(train_df.head(n=100))
print('Number of passengers:',train_df.shape[0])
print('Features and datatypes:')
print(train_df.dtypes)
from nameparser import HumanName
def separate_names(name):
    name_dict={'first': None, 
               'title': None, 
               'middle': None, 
               'last': None, 
               'suffix': None, 
               'nickname': None}
    try:
        name_object = HumanName(name)   
        for key,val in name_object.as_dict().items():
            if val!='':            
                name_dict[key]=val
    finally:
        return list(name_dict.values())

def parse_name_col(df):
    df['First name'],df['Title'],df['Middle name'],df['Last name'],df['Suffix'],df['Given name']=zip(*df['Name'].map(separate_names))
    return df.copy()


def get_maiden_name(df):
    _,_,_,df['Maiden name'],_,_=zip(*df['Given name'].map(separate_names))
    return df.copy()

def fix_missing_names(df):   
    tmp=df[df['First name'].isnull()].copy()
              #display(tmp)\n    
    first,_,middle,last,_,_=zip(*tmp['Given name'].map(separate_names))
    new_column = pd.DataFrame({'First name': list(first), 'Middle name': list(middle), 'Last name': list(last)},index=tmp.index)    
    tmp.update(new_column) 
    df.update(tmp)
    return df.copy()
 
import re
def separate_cabin(cabin):
    if cabin != cabin:
        let=None
        num=None
    else:
        let=re.findall(r"[A-Za-z]", cabin)
        num=re.findall(r"[\d]+", cabin)
        #print(cabin,let,num)
        #if len(let)>1 or len(num)>1:
        #    raise('Error! Invalid Cabin!')
        if len(set(let))==1:
            let=let[0]
        else:
            let="".join(let)
        
    return let,num
def fix_cabin_col(df):
    df['Cabin letter'],df['Cabin number']=zip(*df['Cabin'].map(separate_cabin))
    return df.copy()

def fix_titles(df):
        
    df.loc[contains(df['Title'],'Lady'),'Title']='Lady.'
    df.loc[contains(df['Title'],'Countess'),'Title']='Countess'
    df['Title']=df.Title.map(lambda x: str(x).replace('.',''))
    
    combined_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
    } # inspired by this kernel: https://www.kaggle.com/vispra/titanic1
    df['Title']=df['Title'].map(combined_titles)
    return df.copy()
def contains(df,name):
    if name==name:
        return df.map(lambda x: name in str(x))
    return None

def clean_data(df):
    if 'Survived' in df.columns:
        df['Survived']=df['Survived'].astype('bool')
    df=parse_name_col(df)
    df=fix_missing_names(df)
    df=get_maiden_name(df)
    df=fix_cabin_col(df)
    df=fix_titles(df)
    return df.copy()
train_df=clean_data(train_df)
test_df=clean_data(test_df)
display(train_df.head(n=10))
#display(fix_titles(train_df))
#display(fix_titles(train_df)['Title'].unique())
#print(train_df['Cabin letter'].unique())

print((train_df.index-train_df['PassengerId']).min())
print((train_df.index-train_df['PassengerId']).max())
plt.figure(figsize=(20,10))

# linear
k=1;
ax=[];
for col in ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
    ax.append(plt.subplot(2,3,k))
    plt.scatter(train_df['PassengerId'],train_df[col],s=1)
    plt.ylabel(col)
    if k<=3:
        plt.setp(ax[k-1].get_xticklabels(), visible=False)
    else:
         plt.xlabel('ID')
    k+=1
plt.show()
display(train_df.describe())
print("%d passengers out of %d survived." % (train_df.Survived.sum(),train_df.shape[0]))

display(train_df.groupby('Survived').sum())
display(train_df.groupby('Survived').mean())
for col in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
    p=train_df.loc[~np.isnan(train_df[col]),col].hist(by=train_df['Survived'],sharey=True,figsize=(10,2))
    p[0].set_title('Died')
    p[0].set_xlabel(col)
    p[1].set_title('Survived')
    p[1].set_xlabel(col)
plt.show()
for col in ['Sex','Embarked','Title','Cabin letter']:
    totals=train_df.groupby([col]).count()['PassengerId']
    #display(totals)
    survived=train_df.groupby(['Survived',col]).count()['PassengerId']
    fracs=survived/totals
    ax=fracs.unstack().plot(kind='bar', stacked=False)
    for k in range(len(ax.patches)):
        p=ax.patches[k]
        ind=int(np.floor(k/2))
        t=totals.iloc[ind]
        #print(ind,t,p)
        fracstr=r'$\frac{'+str(int(np.round(t*p.get_height()))) +'}{'+str(t) +'}$';
        ax.annotate(fracstr, (p.get_x()+p.get_width()/2, p.get_height()+0.02),size=12,va='bottom',ha='center')
    plt.ylim(0,1.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
print("Passengers with missing data by column:")
categories=test_df.columns
missing_df=pd.DataFrame(index=categories)
missing_df['Training data']=train_df.drop(columns="Survived").isnull().sum()
missing_df['Testing data']=test_df.isnull().sum()
display(missing_df)
for cat in ["Pclass","Sex","SibSp","Title"]: #,"Parch","Embarked","Title","Cabin letter"]:
    tmp_df=pd.DataFrame()
    tmp_df["count"]=train_df.groupby(by=cat)["Age"].count()
    tmp_df["mean"]=train_df.groupby(by=cat)["Age"].mean()
    tmp_df["std"]=train_df.groupby(by=cat)["Age"].std()
    display(tmp_df)
from sklearn.model_selection import train_test_split
X=train_df[["SibSp","Title"]]
X["Pclass"]=pd.Series(train_df["Pclass"]).replace([1.0,2.0,3.0],['one','two','three'])
X["Sex"]=pd.Series(train_df["Sex"]).replace(['female','male'],[0,1])
X=pd.get_dummies(X)
y=train_df["Age"]
#display(X)
X_test=X[y.isnull()]
y_test=y[y.isnull()]
X_train=X[~y.isnull()]
y_train=y[~y.isnull()]
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score,cross_val_predict
lr=Ridge(alpha=0.5)
scores = cross_val_score(lr, X_train,y_train,cv=10,scoring='r2')
print("Avg Rsquared from CV:",scores.mean())
predicted = cross_val_predict(lr, X_train,y_train,cv=10)
lr.fit(X_train,y_train)
print("intercept: %.3f" %(lr.intercept_))
for k in range(len(X_train.columns)):
    print("coefficient for %s: %.3f" %(X_train.columns[k],lr.coef_[k]))
fare_df=pd.DataFrame()
fare_df["count"]=train_df.groupby(by=["Pclass","Embarked"])["Fare"].count()
fare_df["mean"]=train_df.groupby(by=["Pclass","Embarked"])["Fare"].mean()
fare_df["std"]=train_df.groupby(by=["Pclass","Embarked"])["Fare"].std()
display(fare_df)
def fill_in_missing_values(df,lr,fare_df):
    # fill in missing cabin letters with unknown
    df['Cabin letter']=df['Cabin letter'].fillna("Unknown")
    
    # fill in missing ages with predictions from a linear model
    tmp_df=df[["SibSp","Title"]]
    tmp_df["Sex"]=pd.Series(df["Sex"]).replace(['female','male'],[0,1])
    tmp_df["Pclass"]=pd.Series(df["Pclass"]).replace([1.0,2.0,3.0],['one','two','three'])
    tmp_df=pd.get_dummies(tmp_df)
    if 'Title_Royalty' not in tmp_df.columns:
        tmp_df['Title_Royalty']=0
    age_pred=lr.predict(tmp_df)
    df.at[df['Age'].isnull(),'Age']=age_pred[df['Age'].isnull()]
    
    # fill in missing fare values using a lookup table
    rows=pd.DataFrame(df[df['Fare'].isnull()])
    for (ind,row) in rows.iterrows():
        Pclass=row["Pclass"]
        Embarked=row["Embarked"]
        prediction=fare_df.loc[Pclass,"mean"][Embarked]
        df.at[ind,"Fare"]=prediction
    #df['Fare']=df['Fare'].fillna(df['Fare'].mean())
    return df.copy()
#fill_in_missing_values(test_df,lr,fare_df)
train_df=fill_in_missing_values(train_df,lr,fare_df)
test_df=fill_in_missing_values(test_df,lr,fare_df)
print(train_df.shape)
print(test_df.shape)
#display(train_df)
tmp=test_df.copy()
tmp["Survived"]=None
tmp.index=tmp.index+(train_df.shape[0])
merged_df=pd.concat([train_df,tmp],sort=True)
lastnames=merged_df["Last name"].unique()
#mydict={}
family_res_df=pd.DataFrame(columns=["Last name","surviving family members","known family members"])
for name in lastnames:
    sub_df=merged_df[((merged_df["Last name"]==name) | (merged_df["Maiden name"]==name)) & ((merged_df["Parch"]>0) | (merged_df['SibSp']>0))]
    #display(sub_df)
    known_survivors=(sub_df.Survived.sum())
    unknown=(sub_df.Survived.isnull().sum())
    for ind in (sub_df.index.values):
        family_res_df.at[ind,"Last name"]=name
        if merged_df.loc[ind,'Survived']==1:
            family_res_df.at[ind,"surviving family members"]=known_survivors-1
            family_res_df.at[ind,"known family members"]=sub_df.shape[0]-unknown-1
        elif merged_df.loc[ind,'Survived']==0:
            family_res_df.at[ind,"surviving family members"]=known_survivors
            family_res_df.at[ind,"known family members"]=sub_df.shape[0]-unknown-1
        else:
            family_res_df.at[ind,"surviving family members"]=known_survivors
            family_res_df.at[ind,"known family members"]=sub_df.shape[0]-unknown
family_res_df['surviving family members'].replace(False,0,inplace=True)
family_res_df['surviving family members'].replace(True,1,inplace=True)

display(family_res_df.head())

def add_family_survival_rate(train_df,test_df,family_res_df):
    train_df['Family survival rate']=None
    test_df['Family survival rate']=None
    test_start_ind=train_df.shape[0]
    for ind in family_res_df.index.values:
        sv=family_res_df.loc[ind,'surviving family members']
        kv=family_res_df.loc[ind,'known family members']
        if kv>0:
            if ind in train_df.index:
                train_df.at[ind,'Surviving family members']=sv
                train_df.at[ind,'Known family members']=kv
                train_df.at[ind,'Family survival rate']=sv/kv
            else:
                test_df.at[ind-test_start_ind,'Surviving family members']=sv
                test_df.at[ind-test_start_ind,'Known family members']=kv
                test_df.at[ind-test_start_ind,'Family survival rate']=sv/kv
            
        else:
            if ind in train_df.index:
                train_df.at[ind,'Surviving family members']=sv
                train_df.at[ind,'Known family members']=kv
                train_df.at[ind,'Family survival rate']=None
            else:
                test_df.at[ind-test_start_ind,'Surviving family members']=sv
                test_df.at[ind-test_start_ind,'Known family members']=kv
                test_df.at[ind-test_start_ind,'Family survival rate']=None
    return train_df,test_df
train_df,test_df=add_family_survival_rate(train_df,test_df,family_res_df)
#display(train_df['Family survival rate'].unique())
#display(test_df['Family survival rate'].unique())
train_df.to_csv('train_processed.csv')
test_df.to_csv('test_processed.csv')

