import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('../input/train.csv')
print(df.shape)
df.head()
df1 = pd.read_csv('../input/test.csv')
print (df.Pclass.unique())
print (df.Sex.unique())
print (df.SibSp.unique())
print (df.Parch.unique())
print (df.Embarked.unique())
print (df.Survived.unique())
def preprocess_data(df):
    df.Sex.replace(['male','female'],[0,1], inplace=True)
    separate = pd.get_dummies(df['Embarked'],prefix='Embarked')
    df = pd.concat([df,separate],axis=1)
    df['family_size'] = df.SibSp + df.Parch + 1 #minimum 1 family member
    median_age = df.Age.median()
    df.Age.fillna(median_age, inplace=True)
    mean_fare = df.Fare.mean()
    df.Fare.fillna(mean_fare, inplace=True)
#     df['normFare'] = StandardScaler().fit_transform(df['Fare'].values.reshape(-1, 1))
    df.drop(['PassengerId','Name','Ticket','Cabin','Embarked','SibSp','Parch'],axis=1, inplace=True)
    return df
df_copy  = df.copy()
df1_copy = df1.copy()
# df_copy.head()
df_copy = preprocess_data(df_copy)
df1_copy = preprocess_data(df1_copy)
df_copy.head()
corr=df_copy.corr()
print(corr)
def plot_corr(df):
    corr = df.corr()
    return sns.heatmap(np.abs(corr), xticklabels=corr.columns, yticklabels=corr.columns)
plot_corr(df_copy)
X = df_copy.iloc[:,1:]
Y = df_copy.iloc[:,0]
X_test = df1_copy.iloc[:,0:]
X.head()
X_train, X_val, Y_train, Y_val = train_test_split(X,Y,test_size=0.1,random_state=44)
X_train.shape , Y_train.shape, X_val.shape, Y_val.shape
model1 = LogisticRegression()
model2 = SVC()
model3 = RandomForestClassifier(n_estimators=100)
model4 = KNeighborsClassifier(n_neighbors=3)
model1.fit(X_train,Y_train)
print("Training accuracy:",round(model1.score(X_train,Y_train)*100,2))
print("Validation accuracy:",round(model1.score(X_val,Y_val)*100,2))
print("Confusion matrix:\n",classification_report(model1.predict(X_val),Y_val))
model2.fit(X_train,Y_train)
print("Training accuracy:",round(model2.score(X_train,Y_train)*100,2))
print("Validation accuracy:",round(model2.score(X_val,Y_val)*100,2))
print("Confusion matrix:\n",classification_report(model2.predict(X_val),Y_val))
model3.fit(X_train,Y_train)
print("Training accuracy:",round(model3.score(X_train,Y_train)*100,2))
print("Validation accuracy:",round(model3.score(X_val,Y_val)*100,2))
print("Confusion matrix:\n",classification_report(model3.predict(X_val),Y_val))
model4.fit(X_train,Y_train)
print("Training accuracy:",round(model4.score(X_train,Y_train)*100,2))
print("Validation accuracy:",round(model4.score(X_val,Y_val)*100,2))
print("Confusion matrix:\n",classification_report(model4.predict(X_val),Y_val))
Y_pred = model1.predict(X_test)
submission = pd.DataFrame({'PassengerId':df1.PassengerId, 'Survived':Y_pred})
submission.to_csv('submission.csv',index=False)
##oversampling using smote algorithm
# from imblearn.over_sampling import SMOTE
# def oversampling(X_train,Y_train):
#     smote = SMOTE(random_state=0)
#     X_train_bal, Y_train_bal = smote.fit_sample(X_train, Y_train.ravel())
#     return X_train_bal, Y_train_bal
