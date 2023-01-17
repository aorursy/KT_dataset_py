import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
df = pd.read_csv("../input/titanic/train.csv")
# manipulating the DataFrame



print(str(round(df['Cabin'].isna().sum()/len(df['Cabin'])*100,2))+'% of nan values for Cabin column')  

print('too many null values. Cabin is not a good feature')



# categorical features



#binary

df['Sex_enc']=df['Sex'].map({'male':0,'female':1})  



#more than 2

df[['C','Q','S']]=pd.get_dummies(df['Embarked'])
# features from text   (this is shown for completeness, but the feature will not be used)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vec = TfidfVectorizer()    # get information from text, transforming in features

text_tfidf = tfidf_vec.fit_transform(df['Name'])

dfname=pd.DataFrame(text_tfidf.toarray())



# getting the names of features from text

dfname.columns=tfidf_vec.get_feature_names()

peppe=list(zip(dfname.columns,dfname.sum()))

peppedf=pd.DataFrame(peppe)



# taking the most significant ones

peppedf.sort_values(by=1,ascending=False)

dfname2=dfname.loc[:,['mr','miss','mrs']]





# concatenate the two dataset

dfcomplete=pd.concat([df,dfname2],axis=1)
sns.countplot(df['Sex_enc'],hue=df['Survived'])

plt.xlabel('Gender')

plt.xticks([0,1],['Male','Female'])

plt.show()
sns.countplot(df['Pclass'],hue=df['Survived'])

plt.xlabel('Class')

plt.show()
plt.figure(figsize=(20,10))

sns.countplot(df['Age'],hue=df['Survived'])

plt.xlabel('Class')

plt.show()
print('Below 10 years the survival probability is higher')

print('There is not a significant difference between having 2-3 years or 5-6 years')



df['new_age']=np.floor(df['Age']/10)



nanage=round(df['new_age'].isna().sum()/len(df['new_age'])*100,1)

print('nan values in the age columns '+str(nanage)+'%')
# the feature selection can be done automatically using PCE or RFE (Recursive Feature Elimination) 

# Let us do it manually in this simple case



print('The PassengerId and the name are not relevant features')

print('The Sex and the Age columns has been encoded. Let us remove the original ones')

print('The number of the Ticket and the number of the cabin are not relevant for the survival probability')

print('The fare is highly correlated with the Pclass and it can be removed')

print('The embarked place is not relevant for the survival probability')



dfcomplete2=df.drop(['PassengerId','Name','Sex','Age','Ticket','Cabin','Fare','Embarked','C','S','Q'],axis=1)

dfcomplete2=dfcomplete2.fillna(dfcomplete2.mean())

dfcomplete2.head()



X=dfcomplete2.drop('Survived',axis=1)

y=dfcomplete2['Survived']



from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_std=scaler.fit_transform(X)
# Our dataset after feature engineer and feature selection



X
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate

from sklearn.feature_selection import RFE
# We will use a RandomForestClassifier. What is the ideal number of n_estimators ? 



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV



rf_grid=GridSearchCV(RandomForestClassifier(),{'n_estimators':[10,50,100,200,500]},cv=5)



rf_grid.fit(X_std,y)



print(rf_grid.best_params_)

print(rf_grid.best_score_)



bestpar=rf_grid.best_params_['n_estimators']
X_train,X_test,y_train,y_test=train_test_split(X_std,y,test_size=0.25,stratify=y)



rf=RandomForestClassifier(n_estimators=bestpar)



rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)



print("----------------------------------------------------") 

print("Confusion matrix \n"+str(confusion_matrix(y_pred,y_test))) 

print("----------------------------------------------------") 

print("Report \n"+str(classification_report(y_test,y_pred))) 

print("----------------------------------------------------")



crosscv=cross_validate(rf,X_std,y,cv=10)['test_score'].mean()

print('The cross validation score is '+str(round(crosscv,2)))
# confusion matrix

confusion_matrix(y_test,y_pred)
print('With a very simple model we obtain a cross validation score above 80%')

print('Further feature engineer and feature selection can improve the result')