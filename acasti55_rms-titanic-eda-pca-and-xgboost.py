import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

import re

print(os.listdir("../input"))



from sklearn.metrics import classification_report,confusion_matrix

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

##data_with_imputed_values = my_imputer.fit_transform(original_data)

# Any results you write to the current directory are saved as output.

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier



trainpath = '../input/train.csv'

testpath = '../input/test.csv'
train = pd.read_csv(trainpath)

test = pd.read_csv(testpath)
train.head()
g= sns.catplot(data=train,hue='Survived',x='Pclass',y='Parch',col='Sex',kind='bar',height=4,aspect=1)
g= sns.catplot(data=train,hue='Survived',x='Pclass',y='SibSp', col='Sex',kind='bar',height=4,aspect=1)
sns.distplot(train[train['Survived']==1]['Age'].dropna(), label='survived')

sns.distplot(train[train['Survived']==0]['Age'].dropna(), label='not survived')

plt.legend()

plt.figure()

sns.distplot(train[(train['Survived']==1)&(train['Sex']=='male')]['Age'].dropna(), label='survived male')

sns.distplot(train[(train['Survived']==1)&(train['Sex']=='female')]['Age'].dropna(), label='survived female')

plt.legend()



plt.figure()

sns.distplot(train['Age'].dropna(), label='train')

sns.distplot(test['Age'].dropna(), label='test')

plt.legend()
train = pd.read_csv(trainpath)

test = pd.read_csv(testpath)

#dictionary with aggregation of tickets name

def femaleSbSp(col1,col2,col3):

    if col1=='female': #sex

        if col2>0: #SbSp

            if col3==3: #Pclass

                return 1

            else:

                return 0

        else:

            return 0

    else:

        return 0

    

def femaleParch(col1,col2,col3):

    if col1=='female': #Sex

        if col2>0: #Parch

            if col3!=2:

                return 1

            else:

                return 0

        else:

            return 0

    else:

        return 0



def df_fixer(df,target,t=False):

    cols_with_nulls = df.drop(target,axis=1).isnull().sum()[df.drop(target,axis=1).isnull().sum()>0].sort_values(ascending=False).index.tolist()

    

    for nacol in cols_with_nulls:

        print(nacol)

        if df[nacol].dtype!= object:

            df[nacol].fillna(df.groupby(['Sex','Pclass'])[nacol].transform('mean') ,inplace=True)

        else:

            df[nacol].fillna('other',inplace=True)  

            

    # there are multiple tickets sold to people with summed Fare.    

    #df.isnull().sum()[df.isnull().sum()>0]

    multiple_tickets = pd.DataFrame(df.Ticket.value_counts()[df.Ticket.value_counts()>0].reset_index())

    multiple_tickets.columns = ['Ticket','N_Ticket']

    df = pd.merge(df,multiple_tickets, on=['Ticket'],how='left')

    df['Fare_adj'] = df['Fare']/df['N_Ticket']

    

    ## clean cabin numbers

    df['Cabin_n'] = df['Cabin'].apply(lambda x: len(x.split(' ')))

    df['Cabin_deck'] = df['Cabin'].apply(lambda x: x[0])

    

    df['AgeBin'] =  pd.cut(df['Age'].astype(int), 5)

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)

    titles = ['Mrs', 'Mr', 'Master', 'Miss', 'Dr', 'Rev']

    #df['Title'] = df['Name'].apply(lambda x: x.split(', ')[1].split('.')[0])

    Titlemap = {'Mlle':'Miss','Mme':'Mrs','Ms':'Mr','Lady':'Mrs'}

    #df['BigTitle'] = df['Title'].apply(lambda x: 1 if x=='Dr' else 0)

    df['Title'] = df['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

    stat_min = 10

    title_names = (df['Title'].value_counts() < stat_min)

    df['Title'] = df['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

    label = LabelEncoder()

    

    df['Sex_Code'] = label.fit_transform(df['Sex'])

    df['Embarked_Code'] = label.fit_transform(df['Embarked'])

    df['Title_Code'] = label.fit_transform(df['Title'])

    df['AgeBin_Code'] = label.fit_transform(df['AgeBin'])



    df['Family'] = df['SibSp']+df['Parch']+1

    df['Alone'] = df['Family'].apply(lambda x: 1 if x==1 else 0)

    #if you're male and have family you most probably are left behind !

    df['Male_Family'] = df.apply(lambda x: 1 if (x.Family+x.Sex_Code)/x.Pclass<1 else 0,axis=1)

    df['Female_SbSp'] = df.apply(lambda x: femaleSbSp(x.Sex,x.SibSp,x.Pclass),axis=1)

    df['Female_Parch'] = df.apply(lambda x: femaleParch(x.Sex,x.Parch,x.Pclass),axis=1)

    #if ratio is low means man paid for lot of people

    df['Male_fare'] = df['Sex_Code']/(df['Fare_adj']+1)



    df.drop(['Age','Fare','Sex','Name','Ticket','Cabin_deck','Cabin','Embarked','Title','AgeBin'],axis=1,inplace=True)



    return df
train = df_fixer(train, 'Survived',t=True)

test = df_fixer(test, [], t=False)
train.isnull().sum().sort_values(ascending=False).head()
test.head(1)
sns.countplot(x='Pclass',hue='Survived', data=train[train.Sex_Code==1])
sns.countplot(x='Pclass',hue='Survived', data=train[train.Sex_Code==0])
# Compute the correlation matrix

corr = train.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15, 12))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
sns.barplot(y='Survived', x='Pclass', data=train, hue='Sex_Code')
sns.barplot(y='Fare_adj', x='Sex_Code', hue='Survived', data=train)
train.groupby(['Survived','Sex_Code','Pclass']).mean().round(2)
train.head(2)
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import PCA

sc = StandardScaler()

PCAdf = train.drop(['PassengerId','Survived'],axis=1).copy()

PCAdf_std = sc.fit_transform(PCAdf)

pca = PCA(n_components=6)

x_pca = pca.fit_transform(PCAdf_std)

plt.figure(figsize=(8,6))

sns.scatterplot(x_pca[:,0],x_pca[:,1],hue=train['Survived'],palette='viridis',s=120)

plt.xlabel('First principal component')

#plt.legend()

plt.ylabel('Second Principal Component')

train['PC1'] = x_pca[:,0]

train['PC2'] = x_pca[:,1]

train['PC3'] = x_pca[:,2]

train['PC4'] = x_pca[:,3]
X =  train.drop(['Survived','PassengerId'],axis=1)

y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25,random_state=101)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold
model = XGBClassifier(n_estimators=2500,learning_rate=0.215, max_depth=7, n_jobs=-1, random_state=42)

model.fit(X_train,y_train)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,model.predict(X_test)))

acc_xgb = round(model.score(X_test, y_test) * 100, 2)

model.fit(X,y)
n_estimators = [165,170,175]

depth = [2,3,4]

lr = [0.21, 0.215, 0.22]

pred_res = []

top_result = 0

top_model = 0

for n in n_estimators:

    for d in depth:

        for l in lr:

            

            model = XGBClassifier(n_estimators=n, learning_rate=l, max_depth=d, random_state=42)

            kfold = KFold(n_splits=5, random_state=7)

            results = cross_val_score(model, X, y, cv=kfold)

            if results.mean() > top_result:

                print(n,d,l, 'new top', results.mean())

                top_result = results.mean()

                top_model = model

            #print(n, " Estimators. Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

            pred_res.append([n,d,l,str(results.mean().round(3))])
pd.DataFrame(pred_res, columns=['estimators','depth','l.rate','result_mean']).sort_values(by='result_mean',ascending=False).head()
top_model.fit(X,y)

feat_imp = pd.DataFrame(top_model.feature_importances_, index=X.columns)

feat_imp.reset_index().sort_values(by=0,ascending=False).head(10)
from matplotlib.colors import ListedColormap

def plot_decision_regions(X,y, classifier, resolution=0.02):

    markers = ('s','x','o','^','v')

    colors = ('red','blue','lightgreen','gray','cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    

    x1_min, x1_max = X[:,0].min()-1, X[:,0].max()+1

    x2_min, x2_max = X[:,1].min()-1, X[:,1].max()+1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())

    plt.ylim(xx2.min(), xx2.max())

        

    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
kfold = KFold(n_splits=6, random_state=7)

results = cross_val_score(random_forest, X, y, cv=kfold)

print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
random_forest.fit(X, y)
models = pd.DataFrame({

    'Model': ['XGB','Random Forest'],

    'Score': [acc_xgb,acc_random_forest]})

models.sort_values(by='Score', ascending=False)
#sc = StandardScaler()

#PCAdf = train.drop(['PassengerId','Survived'],axis=1).copy()

#PCAdf_std = sc.fit_transform(PCAdf)

#pca = PCA(n_components=6)

#x_pca = pca.fit_transform(PCAdf_std)
PCAdf.shape
test.shape
Pid = test['PassengerId']

X_test = sc.fit_transform(test.drop('PassengerId',axis=1))

x_test_pca = pca.transform(X_test)

plt.figure(figsize=(8,6))

sns.scatterplot(x_test_pca[:,0],x_test_pca[:,1],s=120)

plt.xlabel('First principal component')

#plt.legend()

plt.ylabel('Second Principal Component')

test['PC1'] = x_test_pca[:,0]

test['PC2'] = x_test_pca[:,1]

test['PC3'] = x_test_pca[:,2]

test['PC4'] = x_test_pca[:,3]

newPred = top_model.predict(test.drop(['PassengerId'],axis=1))
output = pd.DataFrame({'PassengerId':Pid,'Survived': newPred})

output.to_csv('submission.csv', index=False)
output