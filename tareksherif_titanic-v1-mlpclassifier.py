# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read data files
X=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
X.head()

test.head()

test['Survived']=np.nan
full=pd.concat([X,test])

def data_inv(df):
    print('Number of Persons: ',df.shape[0])
    print('dataset variables: ',df.shape[1])
    print('-'*20)
    print('dateset columns: \n')
    print(df.columns)
    print('-'*20)
    print('data-type of each column: \n')
    print(df.dtypes)
    print('-'*20)
    print('missing rows in each column: \n')
    c=df.isnull().sum()
    print(c[c>0])
    print('-'*20)
    print('Missing vaules %age vise:\n')
    print((100*(df.isnull().sum()/len(df.index))))
    print('-'*20)
    print('Pictorial Representation:')
    plt.figure(figsize=(8,6))
    sns.heatmap(df.isnull(), yticklabels=False,cbar=False, cmap='viridis')
    plt.show()   
data_inv(full)#function call
sns.heatmap(full.corr(), annot = True)
#fillna
from statistics import mode
full['Embarked']=full['Embarked'].fillna(mode(full['Embarked'])) 
full['Fare'].fillna(full['Fare'].dropna().median(),inplace=True)
full['Age'] = full.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))
full.isnull().sum()
full['Fam']=full['Parch']+full['SibSp']
full=pd.get_dummies(data=full,columns=['Sex','Embarked'],drop_first=True)
full.info()
full=full.drop(['Cabin','Ticket','Name','Parch','SibSp'],axis=1)

#Data Standardization 
preprocessing.StandardScaler().fit(full).transform(full.astype(float))
test = full[full['Survived'].isna()].drop(['Survived'], axis = 1)
train = full[full['Survived'].notna()]
train.info()




X=train[['Age','Fare','Fam','Pclass','Sex_male','Embarked_Q' ,'Embarked_S']]

y=train[['Survived']].astype(np.int8)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)
Scores=[]
hidden_layer_sizes=[]


for cols in range(50,55):
    for rows in range(3,5):
        hidden_layer=(cols,rows)

        from sklearn.neural_network import MLPClassifier
        MLPClassifierModel = MLPClassifier(activation='logistic', # can be also identity , tanh,logistic , relu
                                           solver='lbfgs',  # can be lbfgs also sgd , adam
                                           alpha=0.1 ,hidden_layer_sizes=hidden_layer,random_state=33)
        MLPClassifierModel.fit(X_train, y_train)

        MLPClassifier_y_pred = MLPClassifierModel.predict(X_test)
        Scores.append(MLPClassifierModel.score(X_test, y_test))
        hidden_layer_sizes.append(str(hidden_layer))
        



models = pd.DataFrame({
    'hidden_layer': hidden_layer_sizes,
    'Score': Scores})
models.sort_values(by='Score', ascending=False )



 
plt.plot(hidden_layer_sizes,Scores)
plt.ylabel('Accuracy ')
plt.xlabel('hidden_layer_sizes ')
plt.tight_layout()
plt.show()
from sklearn.neural_network import MLPClassifier
MLPClassifierModel = MLPClassifier(activation='logistic', # can be also identity , tanh,logistic , relu
                                   solver='lbfgs',  # can be lbfgs also sgd , adam
                                   learning_rate='adaptive', # can be constant also invscaling , adaptive
                                   early_stopping= False,
                                   alpha=0.1 ,hidden_layer_sizes=(52, 3),random_state=33)
MLPClassifierModel.fit(X_train, y_train)

MLPClassifier_y_pred = MLPClassifierModel.predict(X_test)
MLPClassifierModel.fit(X, y)
MLPClassifier_y_pred= MLPClassifierModel.predict(test[['Age','Fare','Fam','Pclass','Sex_male','Embarked_Q' ,'Embarked_S']])
Id=test['PassengerId']
sub_df=pd.DataFrame({'PassengerId':Id,'Survived':MLPClassifier_y_pred})
sub_df.to_csv('submission.csv',index=False)
sub_df.head()
