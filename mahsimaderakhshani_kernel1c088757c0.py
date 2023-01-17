import pandas as pd

import numpy as np

import matplotlib.pyplot as plt;

import seaborn as sns

%matplotlib inline
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.tail()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.isnull().sum().sort_values(ascending=False)
sns.set_style('whitegrid')

sns.countplot(x='Survived',data=train,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
sns.set_style('whitegrid')

sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
train['Age'].hist(bins=30,color='darkred',alpha=0.7)
sns.countplot(x='SibSp',data=train)
sns.countplot(x='Parch',data=train)
train['Fare'].hist(color='green',bins=40,figsize=(8,4))
plt.figure(figsize=(12, 7))

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 37



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
train['Embarked'] = train['Embarked'].fillna('S')

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)
train.head()
train.dropna(inplace=True)
train.info()
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.head()
column_names = list(train.columns)
x_names = column_names[1:] # Select all columns except medianHouseValue

y_name = column_names[0]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'],axis=1), 

                                                    train['Survived'], test_size=0.10, 

                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)

X_test.head()
predictions
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier()

dt_model.fit(X_train,y_train)
dt_pred = dt_model.predict(X_test)
print(confusion_matrix(y_test,dt_pred))
print(classification_report(y_test,dt_pred))
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=500)

rf.fit(X_train,y_train)
rf_pre=rf.predict(X_test)
print(confusion_matrix(y_test,rf_pre))
print(classification_report(y_test,rf_pre))
sns.heatmap(test.isnull())
test.drop('Cabin',axis=1,inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)
test.info()
test.head()
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
sex_test = pd.get_dummies(test['Sex'],drop_first=True)

embark_test= pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test = pd.concat([test,sex_test,embark_test],axis=1)
test.head()
train.head()
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(train.drop(['Survived'],axis=1),train['Survived'] )
test_prediction = rf.predict(test)
test_prediction.shape
test_pred = pd.DataFrame(test_prediction, columns= ['Survived'])
new_test = pd.concat([test, test_pred], axis=1, join='inner')
new_test.head()
df= new_test[['PassengerId' ,'Survived']]
df.head()
df.to_csv('predictions.csv' , index=False)
corr_matrix = train.corr()

corr_matrix
sns.heatmap(corr_matrix, 

            cmap = 'YlGnBu',

            annot = True,

            linewidths = .3,

            linecolor = 'white'

           )
from sklearn.decomposition import  PCA



n_components = X_train.shape[1]

pca = PCA(n_components=n_components)

X_train_PCA = pd.DataFrame(pca.fit_transform(X_train), index=X_train.index, columns=list(['PC' + str(x + 1) for x in range(n_components)]))

X_train_PCA.head()
sns.heatmap(X_train_PCA.assign(y = y_train).corr(), 

            cmap = 'YlGnBu',

            annot = True,

            linewidths = .3,

            linecolor = 'white',

            fmt='.1g'

           )
for i in range(1, n_components + 1):

    pc_name = 'PC' + str(i)

    print("Variance of " + pc_name + ": " + str(np.var(X_train_PCA[pc_name])))
n_components = 4

pca = PCA(n_components=n_components)

columns = ['PC' + str(x + 1) for x in range(n_components)]

X_train_PCA = pd.DataFrame(pca.fit_transform(X_train), index=X_train.index, columns=columns)

X_train_PCA.head()
n_components = 4

pca = PCA(n_components=n_components)

columns = ['PC' + str(x + 1) for x in range(n_components)]

X_test_PCA = pd.DataFrame(pca.fit_transform(X_test), index=X_test.index, columns=columns)

X_test_PCA.head()
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train_PCA,y_train)

predictions_PCA = logmodel.predict(X_test_PCA)
print(classification_report(y_test,predictions_PCA))
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)




predictions = logmodel.predict(X_test)

X_test.head()







predictions



from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#Grid Search

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score





grid_values = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}

grid_clf_acc = GridSearchCV(logmodel, param_grid = grid_values,scoring = 'recall')

grid_clf_acc.fit(X_train, y_train)



#Predict values based on new parameters

y_pred_acc = grid_clf_acc.predict(X_test)



# New Model Evaluation metrics 

print(classification_report(y_test,y_pred_acc))


