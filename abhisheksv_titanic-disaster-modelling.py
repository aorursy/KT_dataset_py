# importing required packages

import pandas as pd

import numpy as np

import seaborn as sns

import warnings

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

df=pd.read_csv(r'/kaggle/input/titanic/train.csv')
df.shape
df.columns
df.info()
df.head()
#To analyse whether the passenger is survived or not, lets take the columns that are useful, as here we can observe in titanic ship what are the factors that effect for a person to survive or not.
# Lets take Pclass,Sex,Age,SibSp,parch,Fare,Embarked to predict survived or not.
df=df[['Pclass','Sex','Age','SibSp','Parch','Survived','Fare','Embarked']]# 
df.head(2)
#Lets do some analysis
sns.countplot(x=df['Pclass'],data=df) # Pclass3 as more number of passengers fallowed by Pclass 1 and 2.
sns.countplot(x=df['Pclass'],hue=df['Survived'])
sns.countplot(x=df['Sex'],hue=df['Survived']) #Females has survived more
g=sns.FacetGrid(df,col='Survived')

g.map(plt.hist,'Age',bins=20)# Adults have survived more compared to elder people
g1=sns.FacetGrid(df,row='Pclass',col='Survived')

g1.map(plt.hist,'Age') # there are more survived people in Pclass 1 with Age between 0-35 years
# Checking null values.

df.isna().sum()
df['Embarked'].unique()
df['Embarked'].value_counts()
df['Embarked']=df['Embarked'].fillna('S')
df.isna().sum()
Sex_dummy=pd.get_dummies(df['Sex'],drop_first=True)

embarked_dummy=pd.get_dummies(df['Embarked'],drop_first=True)
df.drop(['Sex','Embarked'],axis=1,inplace=True)
df=pd.concat([df,Sex_dummy,embarked_dummy],axis=1)
df.columns
##using IterativeImputer package we treat null valued col 'Age'

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer
impute_it=IterativeImputer()

df=impute_it.fit_transform(df)
df=pd.DataFrame(df)
df.head(3)
df.columns=[['Pclass', 'Age', 'SibSp', 'Parch', 'Survived', 'Fare', 'male', 'Q','S']]
df.head(2)
X=df[['Pclass','Age','male','Parch','SibSp','Fare','S','Q']]
Y=df.iloc[:,4]
# since none of the features fallowing normal distribution, lets use non linear algorithms, such as DTC,RFC,Bagging
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(criterion='gini',random_state=40)

rfc.fit(X,Y)
Y_pred=rfc.predict(X)
# predict probabilities

prob=rfc.predict_proba(X)
prob=pd.DataFrame(prob)
prob.head(3)
df=pd.concat([df,prob],axis=1)
df.head(2)
df.columns=[['Pclass','Age','Sibsp','Parch','Survived','Fare','male','Q','S','prob_0','prob_1']]
df.head()
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

cm=confusion_matrix(Y,Y_pred)

print(cm)
score=accuracy_score(Y,Y_pred)

print(score)
cr=classification_report(Y,Y_pred)

print(cr)
from sklearn.metrics import roc_auc_score

rocaucscore=roc_auc_score(Y,Y_pred)

print(rocaucscore)
#We can then use the roc_auc_score() function to calculate the true-positive rate and false-positive rate for the predictions using a set of thresholds that can then be used to create a ROC Curve plot.

#rfc_probs=roc_auc_score(Y,prob)
from sklearn.model_selection import StratifiedKFold

skf=StratifiedKFold(n_splits=10, shuffle=False, random_state=None)

accuracy=[]

skf.get_n_splits(X,Y)

for train_index,test_index in skf.split(X,Y):

    #print('Train:',train_index,'validation:',test_index)

    x1_train,x1_test=X.iloc[train_index],X.iloc[test_index]

    y1_train,y1_test=Y.iloc[train_index],Y.iloc[test_index]

    

    rfc.fit(x1_train,y1_train)

    prediction=rfc.predict(x1_test)

    score=accuracy_score(y1_test,prediction)

    accuracy.append(score)



print(np.mean(accuracy))
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

fpr,tpr,thresholds=roc_curve(Y,Y_pred)

plt.plot(fpr,tpr,marker='.', label='rfc')

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.0])

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.grid(True)
gmeans = np.sqrt(tpr * (1-fpr))

gmeans
ix = np.argmax(gmeans)

print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
thresholds
fpr
tpr
df_test=pd.read_csv(r'/kaggle/input/titanic/test.csv')
df_test.head(2)
df_test=df_test[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
df_test.isna().sum()
Sex_dummy=pd.get_dummies(df_test['Sex'],drop_first=True)

embarked_dummy=pd.get_dummies(df_test['Embarked'],drop_first=True)
embarked_dummy.head(1)
df_test.drop(['Sex','Embarked'],axis=1,inplace=True)
df_test=pd.concat([df_test,Sex_dummy,embarked_dummy],axis=1)
df_test.head()
df_test=impute_it.fit_transform(df_test)
df_test=pd.DataFrame(df_test)
df_test.columns=[['Age', 'Fare', 'Parch', 'Pclass', 'Q', 'S', 'SibSp','male']]
x=df_test[['Age', 'Fare', 'Parch', 'Pclass', 'Q', 'S', 'SibSp','male']]
Y_test_pred=rfc.predict(x)
Y_test_pred
Y_test_pred=pd.DataFrame(Y_test_pred)
df_test=pd.concat([df_test,Y_test_pred],axis=1)
df_test.rename(columns={0:'Survived'}).head()