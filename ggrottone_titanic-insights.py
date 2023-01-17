# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



models = [
    ('LR', LogisticRegression()),
    ('SVM', SVC()),
    ('KNN', KNeighborsClassifier()),
    ('KNN-2',KNeighborsClassifier(algorithm='auto', leaf_size=26, metric='minkowski', 
                           metric_params=None, n_jobs=1, n_neighbors=6, p=2, 
                           weights='uniform')),
    ('RF', RandomForestClassifier()),
    ('RF2', RandomForestClassifier(max_depth=None,
              max_features= 3,
              min_samples_split=4,
              min_samples_leaf=5,
              bootstrap= False,
              n_estimators=200,
              criterion="gini")),
    ('XGB', XGBClassifier(learning_rate=0.1, 
                  reg_lambda=0.3,
                  gamma= 1,
                  subsample=0.8,
                  max_depth=2,
                  n_estimators=300)),
    ('GBC', GradientBoostingClassifier()),
    ('lGBM', LGBMClassifier())

]




# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')


testid = df_test['PassengerId']

y=df_train['Survived']
train_len = len(df_train)
df_train = pd.concat([df_train, df_test])

df_train.shape
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns',13 )

df_train['Fare'].fillna(df_train['Fare'].median(), inplace = True)

# Making Bins
df_train['FareBin'] = pd.qcut(df_train['Fare'], 5)

label = LabelEncoder()
df_train['FareBin_Code'] = label.fit_transform(df_train['FareBin'])

#df_train.drop(['Fare'], 1, inplace=True)
colunas = df_train.columns
print(colunas)

sns.heatmap(df_train.corr(), annot=True)
df_train['Family_Size'] = df_train['Parch'] + df_train['SibSp']
#df_test['Family_Size'] = df_test['Parch'] + df_test['SibSp']
df_train['Last_Name'] = df_train['Name'].apply(lambda x: str.split(x, ",")[0])
df_train['Fare'].fillna(df_train['Fare'].mean(), inplace=True)

DEFAULT_SURVIVAL_VALUE = 0.5
df_train['Family_Survival'] = DEFAULT_SURVIVAL_VALUE

for grp, grp_df in df_train[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    if (len(grp_df) != 1):
        # A Family group is found.
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                df_train.loc[df_train['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin==0.0):
                df_train.loc[df_train['PassengerId'] == passID, 'Family_Survival'] = 0
                
for _, grp_df in df_train.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    df_train.loc[df_train['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    df_train.loc[df_train['PassengerId'] == passID, 'Family_Survival'] = 0
df_train.shape

df_train.hist(column='Family_Survival')
df_train = df_train.drop(["Name","Ticket","Fare","Cabin","Embarked",'PassengerId','SibSp','Parch','Family_Size','Age',"FareBin"],axis = 1)
df_train['Sex'].replace('male', '0', inplace = True)
df_train['Sex'].replace('female', '1', inplace = True)

df_train = df_train.drop(["Last_Name"],axis=1)

df_train['Sex'] = df_train['Sex'].values.astype(np.float64)
df_train.dtypes
df_train.head()






train_corr2 = df_train.corr()
print(train_corr2)
conta = df_train[df_train["Survived"]==1]
conta.count()
sns.heatmap(train_corr2, annot=True)
plt.show()
plt.rcParams["figure.figsize"] = (20,4)

for i in list(df_train.columns):
    table = pd.crosstab(df_train[i], df_train["Survived"])
    table.div(table.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True)
    plt.title(i + ' X Survived')
    plt.xlabel(i + ' Status')
    plt.ylabel('Proportion')




df_train = df_train.drop(["Survived"],axis=1)
#df_train['Pclass'].replace(3, 4, inplace = True)
X=df_train.copy()
df_train = X[:train_len]
df_test = X[train_len:]
y_train = y



df_train.head()
#from sklearn.decomposition import PCA
#pca = PCA(n_components=3)
#features = df_train.columns
#x = df_train.loc[:, features].values
#principalComponents = pca.fit_transform(x)
#principalDf = pd.DataFrame(data = principalComponents
 #            , columns = ['principal component 1', 'principal component 2','principal component 3'])
#from sklearn.decomposition import PCA
#pca = PCA(n_components=3)
#features = df_test.columns
#x_test= df_test.loc[:, features].values
#principalComponents = pca.fit_transform(x_test)
#principalDf_test = pd.DataFrame(data = principalComponents
#             , columns = ['principal component 1', 'principal component 2','principal component 3'])
#df_train = principalDf.copy()
#df_test = principalDf_test.copy()
scaler = StandardScaler()
scaler2 = StandardScaler()
df_train = scaler.fit_transform(df_train)

df_test = scaler2.fit_transform(df_test)



y_tr = y
X_tr = df_train.copy()

y_tr


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_tr, y_tr, test_size=0.6, random_state=42)

for name, model in models:
        clf = model
        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        print(name, accuracy)



f_model= LGBMClassifier()
f_model.fit(X_train,y_train)
y_test = f_model.predict(df_test)
df_final = pd.read_csv('/kaggle/input/titanic/test.csv')
df_final.shape
df_final['Survived'] = y_test
df_final.shape
df_final=df_final[["PassengerId","Survived"]]
df_final.to_csv (r'/kaggle/working/titanic.csv', index = False, header=True)
df_final.shape
df_train




