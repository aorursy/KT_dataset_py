import pandas as pd , numpy as np , matplotlib.pyplot as plt , seaborn as sns , warnings

%matplotlib inline



warnings.filterwarnings('ignore')



df = pd.read_csv('../input/titanic/train.csv')

df_test1 = pd.read_csv('../input/titanic/test.csv')



df.head()
df.tail()
df.shape
df.columns
df.index
df.info()
df.describe()
sns.countplot(x='Survived',hue='Sex',data=df)
sns.countplot(x='Survived',hue='Pclass',data=df)
sns.countplot(df.Survived)
plt.hist(df.Age)
sns.boxplot(x='Pclass',y='Age',data=df)
df.isnull().sum()
def impute_age(cols):

    Age=cols[0]

    Pclass=cols[1]



    if pd.isnull(Age):



        if Pclass == 1:

            return 37

        elif Pclass ==2:

            return 29

        elif Pclass == 3:

            return 24

    else:

        return Age

    

df.Age=df[['Age','Pclass']].apply(impute_age,axis=1)

df = df.drop(columns = ['Ticket','Name','Embarked','PassengerId','Cabin'])



dfn=df._get_numeric_data()

nc=list(dfn)

dfc=df.drop(columns=nc)

cc=list(dfc)



dfc=pd.get_dummies(dfc,drop_first=True)



data=pd.concat([dfn,dfc],axis=1)

data.shape
df.isnull().sum()
data.isnull().sum()
sns.heatmap(data.corr(),cmap='viridis',annot = True)
sns.pairplot(data)
c= data.corr()    # Finding correlation



i = 0



# replacing diogonal corr() which is 1 to NaN for finding

#i.e manulating and get the informative features by removing high values of correlation.



while True:    

    try:

        c.iloc[i,i] = np.nan

        i += 1

    except:

        break
data.Survived.value_counts()
# Getting high corr. values w.r.t output because it supports the output...



features = c[(c['Survived'] > 0.1) | (c['Survived'] < -0.1)].dropna(how = 'all')['Survived']     

features_col = list(features.index)

print(features.shape, len(features_col) , features_col)     # Exactly what i want....
sns.heatmap(data[features_col].corr(),cmap='viridis',annot = True)
sns.pairplot(data[features_col])
# Preparing  pipeline for all the models

# Here DecisionTreeClassifier wins the race...



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score,roc_auc_score

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix,classification_report





X = data[features_col]

y = data.Survived

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)







models = [LogisticRegression(penalty='l2'),DecisionTreeClassifier()

          ,RandomForestClassifier(),KNeighborsClassifier(),SVC()]



for model in models:

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    print(accuracy_score(y_test,y_pred),type(model).__name__)
X = data[features_col]

y = data.Survived

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1)



model = DecisionTreeClassifier()



model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test,y_pred),type(model).__name__)
print(confusion_matrix(y_test,y_pred)) 
print(classification_report(y_test,y_pred))
X.columns
df_test = df_test1[['Pclass', 'Fare', 'Sex']]

df_test.isnull().sum()
df_test[df_test['Fare'] != df_test['Fare']]
df_test[df_test['Pclass'] == 3]['Fare'].mean()
df_test = df_test.fillna(12.46)
df_test.isnull().sum()
df_test = pd.get_dummies(df_test,drop_first = True)
X_train = data[['Pclass', 'Fare', 'Sex_male']]

X_test = df_test

y_train = data.Survived



model = DecisionTreeClassifier()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
predictions = pd.concat([df_test1[['PassengerId']],pd.DataFrame(y_pred,columns = ['Survived'])],axis=1)
predictions.shape
predictions