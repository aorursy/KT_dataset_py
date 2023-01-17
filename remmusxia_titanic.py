import numpy as np 

import pandas as pd 

from pandas.core.frame import DataFrame





import seaborn as sns

import matplotlib.pyplot as plt





from sklearn import preprocessing



from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier



from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis





















df_train=pd.read_csv("../input/titanic/train.csv")

df_test=pd.read_csv("../input/titanic/test.csv")



df=pd.concat([df_train, df_test], axis=0,sort=False)

df.reset_index(drop=True, inplace=True)

df.drop(['Survived'],axis=1,inplace=True)
df_train.head()
print("===== Data shape =====\n", DataFrame({"tain" : df_train.shape, "test" : df_test.shape}, index=["rows", "columns"]))
df_train.info()

df_test.info()
def Unique_Values(df_train,df_test):

    train_list = df_train.apply(lambda x: x.unique().size * 100 /x.size)

    test_list = df_test.apply(lambda x: x.unique().size * 100 /x.size)

    Unique_list = pd.concat([train_list,test_list], axis=1,sort=False)

    Unique_list=Unique_list.round(decimals=2)

    Unique_list.columns=['Percentage_train','Percentage_test']

    return Unique_list
Unique_Values(df_train,df_test)
def Missing_values(df):

    total = df.isnull().sum().sort_values(ascending=False)   

    percent =  (df.isnull().sum()*100/df.isnull().count()).sort_values(ascending=False)

    percent=percent.round(decimals=2)

    missing_data = pd.concat([total, percent],axis=1, keys =['Total', 'Percentage'])

    return "No missing data" if (missing_data[percent>0]).empty  else (missing_data[percent>0])
def Missing_list(df_train,df_test):

    train_missing = Missing_values(df_train)

    test_missing = Missing_values(df_test)

    Missing_list = pd.concat([train_missing,test_missing], axis=1,sort=False)

    Missing_list.columns=['Total_train','Percentage_train','Total_test', 'Percentage_test']

    return Missing_list
Missing_list(df_train,df_test)
df_scale=df_train[['Pclass','Age','SibSp','Parch','Fare']].copy()

scale = preprocessing.StandardScaler().fit(df_scale[['Age','Fare']])

df_scale[['Age','Fare']] = scale.transform(df_scale[['Age','Fare']])



df_scale.boxplot();
df_train['Source']= list('train' for i in range(df_train.shape[0]))

df_test['Source']= list('test' for i in range(df_test.shape[0]))



df=pd.concat([df_train, df_test], axis=0,sort=False)

df.reset_index(drop=True, inplace=True)
def plot_catVsTarget(var): 

    plt.figure(figsize=(13,4))

    plt.subplot(1, 2, 1)

    sns.countplot(x=var, data=df, hue = 'Source')

    plt.subplot(1, 2, 2)

    sns.barplot(x = var, y = 'Survived',data=df[0:891])

    plt.show()   
def plot_contVsTarget(var,break_bins=40):

    plt.figure(figsize=(13,4))

    plt.subplot(1, 2, 1)

    bins = np.linspace(min(df[var]), max(df[var]), break_bins)

    plt.hist(df[0:891][var], bins, alpha=0.5, label='train')

    plt.hist(df[891:][var], bins, alpha=0.5, label='test')

    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)

    sns.distplot(df[0:891][var][df[0:891]['Survived']==1])

    sns.distplot(df[0:891][var][df[0:891]['Survived']==0])

    plt.legend(labels=['Survived=1','Survived=0'])

    plt.show()
sns.countplot(x=df_train["Survived"], data=df_train);
df.columns
plot_catVsTarget('Pclass')
df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]



title_mapDict = {

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

                    }



df['Title'] = df['Title'].map(title_mapDict)



plot_catVsTarget('Title')
df['Title'].unique()
df['NLength'] = df['Name'].apply(len)



plot_contVsTarget('NLength')
plot_catVsTarget('Sex')
df['Age'].fillna(100, inplace=True)

plot_contVsTarget('Age')


df['Nfamille'] = df['Parch'] + df['SibSp'] + 1



plot_catVsTarget('Nfamille')

sns.catplot(x="Pclass", hue="Nfamille", col="Survived",data=df[0:891], kind="count");

f_list=[]

for f in df['Nfamille']:

    if f==1:

        f_list.append(0)

    elif (f<=4) & (f>=2):

        f_list.append(1)

    else:

        f_list.append(2)

        

df['Fsize']=f_list



plot_catVsTarget('Fsize')

TickCountDict={}

TickCountDict=df['Ticket'].value_counts()

TickCountDict.head()

df['Tshares']=df['Ticket'].map(TickCountDict)



plot_catVsTarget('Tshares')

sns.catplot(x="Pclass", hue="Tshares", col="Survived",data=df[0:891], kind="count");
t_list=[]

for t in df['Tshares']:

    if t==1:

        t_list.append(0)

    elif (t<=4) & (t>=2):

        t_list.append(1)

    else:

        t_list.append(2)

        

df['Tsize']=t_list



plot_catVsTarget('Tsize')
df[df['Fare'].isnull().values==True]

df['Fare']=df['Fare']/df['Tshares']



Missing_Fare = df.Fare[(df['Pclass']==3)&(df['Title']=='Mr')&(df['Embarked']=='S')& (df['Tshares']==1)].describe()

df['Fare'].fillna(Missing_Fare['mean'], inplace=True)



plot_contVsTarget('Fare')
df['Cabin'].fillna('U', inplace=True)

df['Ctype']=df['Cabin'].map(lambda x: x[0])
# Find out all tickets that the cabin type is unknown and the ticket type is shared 

cabin_df=df[df['Tshares']>1]

cabinDict={}



for t in cabin_df['Ticket'].unique():

    if len(cabin_df['Ctype'][cabin_df['Ticket']==t].unique()) >1:

        cabinDict[t]=list(cabin_df['Ctype'][cabin_df['Ticket']==t].unique())

        

print(cabinDict)



for key in cabinDict.copy():

    if 'U' not in cabinDict[key]:

        cabinDict.pop(key)

        

print(cabinDict)
# Filling some missing data in Cabin with more accurate values.

for key in cabinDict.copy():

    cabinDict[key].remove('U')



print(cabinDict)

for key in cabinDict:

    df.loc[(df['Ticket']==key),'Ctype']=cabinDict[key]
plot_catVsTarget('Ctype')
plot_catVsTarget('Embarked')
sns.catplot(x="Pclass", hue="Embarked", data=df[0:891], kind="count");
Age_df=df.copy()

Age_df.head()
for var in ['Sex','Embarked','Title','Ctype']:

    Age_df[var]=Age_df[var].apply(str)

    var01 = pd.get_dummies(Age_df[var])

    Age_df = pd.merge(Age_df,var01,left_index=True,right_index=True)



Age_df.drop(['PassengerId','Survived','Name','Sex','Ticket','Cabin', 'Embarked','Title','Ctype','Source'],axis=1,inplace=True)
Age_df_train=Age_df[Age_df['Age']!=100].copy()

Age_df_train.drop(['Age'],axis=1,inplace=True)
Age_df_test=Age_df[Age_df['Age']==100].copy()

Age_df_test.drop(['Age'],axis=1,inplace=True)



Age_y_train=Age_df.Age[Age_df['Age']!=100].copy()



Age_df_train.info()
alg = RandomForestClassifier(n_estimators=100)



alg.fit(Age_df_train, Age_y_train.astype('int'))



Age_predictions=alg.predict(Age_df_test)



df.loc[df['Age']==100, ['Age']] = Age_predictions



plot_contVsTarget('Age')
df['Lastname']=df['Name'].map(lambda x:x.split(',')[0].strip())
df[df['Nfamille']==11]
df.loc[df['Nfamille']==11,'Age']= 60

df.loc[df['Nfamille']==11,'Sex']='Male'

df[df['Lastname']=='Sage']
df[df['Nfamille']==8]
df.loc[df['Nfamille']==8,'Age']=60

df.loc[df['Nfamille']==8,'Sex']='Male'

df[df['Lastname']=='Goodwin']
df[(df['Nfamille']==7)].sort_values(by=['Lastname'])
df[(df['Nfamille']==7) & (df['Fare']>7)]
df.loc[(df['Nfamille']==7) & (df['Fare']<7) & (df['Lastname']=='Andersson'),'Age']=60

df.loc[(df['Nfamille']==7) & (df['Fare']<7) & (df['Lastname']=='Andersson'),'Sex']='Male'

df[df['Lastname']=='Andersson']
def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    

    plt.title('Pearson Correlation of Features', y=1.05, size=15)



correlation_heatmap(df[0:891])

df.info()
for var in ['Sex','Embarked','Title','Ctype']:

    df[var]=df[var].apply(str)

    var01 = pd.get_dummies(df[var])

    df = pd.merge(df,var01,left_index=True,right_index=True)
df_train1=df[0:len(df_train)]

df_test1=df[len(df_train):]



Target=df_train['Survived']
df.columns
train_var= ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'NLength', 'Nfamille', 'Fsize', 'Tshares', 'Tsize', 'Male', 'female', 'male', 'C_x', 'Q', 'S', 'nan', 'Master', 'Miss',

       'Mr', 'Mrs', 'Officer', 'Royalty', 'A', 'B', 'C_y', 'D', 'E', 'F', 'G',

       'T', 'U']
kfold=StratifiedKFold(n_splits=10)

classifiers=[]

classifiers.append(SVC())

classifiers.append(DecisionTreeClassifier())

classifiers.append(RandomForestClassifier())

classifiers.append(ExtraTreesClassifier())

classifiers.append(GradientBoostingClassifier())

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression())

classifiers.append(LinearDiscriminantAnalysis())
cv_results=[]

for classifier in classifiers:

    cv_results.append(cross_val_score(classifier,df_train1[train_var],Target,

                                      scoring='accuracy',cv=kfold,n_jobs=-1))

    

cv_means=[]

cv_std=[]

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())

    

cvResDf=pd.DataFrame({'cv_mean':cv_means,

                     'cv_std':cv_std,

                     'algorithm':['SVC','DecisionTreeCla','RandomForestCla','ExtraTreesCla',

                                  'GradientBoostingCla','KNN','LR','LinearDiscrimiAna']})



cvResDf=cvResDf.sort_values('cv_mean', ascending=False)





sns.barplot(data=cvResDf,x='cv_mean',y='algorithm',**{'xerr':cv_std})


alg = GradientBoostingClassifier()

alg.fit(df_train1[train_var], Target)





ids = df_test1['PassengerId']





predictions= alg.predict(df_test1[train_var])



output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions.astype(np.int64)})

output.to_csv('titanic-predictions.csv', index = False)
