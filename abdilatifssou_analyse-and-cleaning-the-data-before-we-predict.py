import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.tail(10)
data.shape
b=data.shape[0]#we need this later
data.columns
#data.shape
data.info()
data.drop_duplicates(subset=data.columns.values[:-1], keep='first',inplace=True)
print(b-data.shape[0]," duplicated Rows has been removed")
data.shape
sns.countplot(x='Class',data=data)
data.Class.value_counts()
g=sns.FacetGrid(data,col='Class')
g.map(plt.hist,'Time', bins=20)
g=sns.FacetGrid(data,col='Class')
g.map(plt.hist,'Amount', bins=20)
#sns.pairplot(data)
plt.subplots(figsize=(20,20))
sns.heatmap(data.corr(), vmax=.8 , square=True,annot=True,fmt='.2f')
data.corr().nlargest(31,'Class')['Class']
def feature_dist(df0,df1,label0,label1,features):
    plt.figure()
    fig,ax=plt.subplots(6,5,figsize=(30,45))
    i=0
    for ft in features:
        i+=1
        plt.subplot(6,5,i)
        # plt.figure()
        sns.distplot(df0[ft], hist=False,label=label0)
        sns.distplot(df1[ft], hist=False,label=label1)
        plt.xlabel(ft, fontsize=11)
        #locs, labels = plt.xticks()
        plt.tick_params(axis='x', labelsize=9)
        plt.tick_params(axis='y', labelsize=9)
    plt.show()

t0 = data.loc[data['Class'] == 0]
t1 = data.loc[data['Class'] == 1]
features = data.columns.values[:30]
feature_dist(t0,t1 ,'Normal', 'Busted', features)
def showboxplot(df,features):
    melted=[]
    plt.figure()
    fig,ax=plt.subplots(5,6,figsize=(30,20))
    i=0
    for n in features:
        melted.insert(i,pd.melt(df,id_vars = "Class",value_vars = [n]))
        i+=1
    for s in np.arange(1,len(melted)):
        plt.subplot(5,6,s)
        sns.boxplot(x = "variable", y = "value", hue="Class",data= melted[s-1])
    plt.show()


showboxplot(data,data.columns.values[:-1])


X=data.drop(['Class'],axis=1)
y=data['Class']
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40, shuffle =True)
#we combine the train data here for the function of removing outliers 
X_train['Class']=y_train

def Remove_Outliers(df,features):
    
    
    Positive_df = df[df["Class"] == 1]#1
    Negative_df = df[df["Class"] == 0]#0
    before=df.shape[0]

    for n in features:
        
        desc1 = Positive_df[n].describe()
        lower_bound1 = desc1[4] - 1.5*(desc1[6]-desc1[4])
        upper_bound1 = desc1[6] + 1.5*(desc1[6]-desc1[4])
        
        desc0 = Negative_df[n].describe()
        lower_bound0 = desc0[4] - 1.5*(desc0[6]-desc0[4])
        upper_bound0 = desc0[6] + 1.5*(desc0[6]-desc0[4])

        df=df.drop(df[(((df[n]<lower_bound1) | (df[n]>upper_bound1))
                      &
                      (df['Class']==1))
                      |
                      (((df[n]<lower_bound0) | (df[n]>upper_bound0))
                      &
                      (df['Class']== 0))].index)

    after=df.shape[0]
    print("number of deleted outiers :",before-after)
    return df


a=Remove_Outliers(X_train,X_train.columns.values[:-1])
X_train=a.iloc[:,:-1]
y_train=a.iloc[:,-1]
def showboxplot(df,features):
    melted=[]
    plt.figure()
    fig,ax=plt.subplots(5,6,figsize=(30,20))
    i=0
    for n in features:
        melted.insert(i,pd.melt(df,id_vars = "Class",value_vars = [n]))
        i+=1
    #print(melted[29])
    # print(len(melted))
    #print(np.arange(len(melted)+1))
    for s in np.arange(1,len(melted)):
        plt.subplot(5,6,s)
        sns.boxplot(x = "variable", y = "value", hue="Class",data= melted[s-1])
    plt.show()


showboxplot(a,a.columns.values[:-1])


from sklearn.preprocessing import StandardScaler

X_train=StandardScaler().fit_transform(X_train)
X_test=StandardScaler().fit_transform(X_test)
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix
#logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


Train_acc_log = round(logreg.score(X_train, y_train) * 100, 3)
Test_acc_log = round(logreg.score(X_test, y_test) * 100, 3)
acc_logreg=round(accuracy_score(y_test, y_pred)*100,3)

print("Score : ",Test_acc_log)

sns.heatmap(confusion_matrix(y_test , y_pred), center=True,annot=True,fmt='.1f')