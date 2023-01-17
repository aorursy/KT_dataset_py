import pandas as pd
import numpy as np
df=pd.read_csv("../input/bank-marketing/bank-additional-full.csv",sep=';')
df.head()
df.shape
df.isnull().sum()
df.apply(lambda x : len(x.unique()))
import matplotlib.pyplot as plt
import seaborn as sns
#correlation_matrix
corr_m = df.corr() 
f, ax = plt.subplots(figsize =(7,6)) 
sns.heatmap(corr_m,annot=True, cmap ="YlGnBu", linewidths = 0.1)
sns.scatterplot(x = 'previous',y = 'pdays',data = df,alpha = 0.5);
#sns.scatterplot(x = 'day',y = 'campaign',data = df,alpha = 0.5);
sns.countplot(x = 'age', data = df)
sns.catplot('contact',kind = 'count',data = df,aspect =3)
sns.catplot('marital',kind = 'count',data = df,aspect =3)
sns.catplot('education',kind = 'count',data = df,aspect =3)
df['education'].replace({'unknown':'secondary'},inplace = True)
sns.catplot('education',kind = 'count',data = df,aspect =3)
sns.catplot('default',kind = 'count',data = df,aspect =3)
sns.catplot('housing',kind = 'count',data = df,aspect =3)
sns.catplot('loan',kind = 'count',data = df,aspect =3)
sns.catplot('y',kind = 'count',data = df,aspect =3)
df['contact'].replace({'unknown':'cellular'},inplace = True)
sns.catplot('contact',kind = 'count',data = df,aspect =3)
sns.catplot('month',kind = 'count',data = df,aspect =3)
sns.catplot('poutcome',kind = 'count',data = df,aspect =3)
cat_cols = df.select_dtypes(include=['object']).columns
num_cols = df.select_dtypes(exclude=['object']).columns
cat_cols,num_cols
#treat for age column 
age_q1=df['age'].quantile(q = 0.25)
age_q2=df['age'].quantile(q = 0.50)
age_q3=df['age'].quantile(q = 0.75)
age_q4=df['age'].quantile(q = 1.00)
print('Quartiles:',age_q1,age_q2,age_q3,age_q4)
outliers=age_q3+1.5*age_q3-1.5*age_q1
print('outliers:',outliers)
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
le=LabelEncoder()
def clusters(x):
    if x<=33:
        return 0
    elif x>32 & x<=49:
        return 1
    elif x>49 & x<=73:
        return 2
    elif x>73 & x<=87:
        return 4
df['age'] = df['age'].astype('int').apply(clusters)
#df['age_new'] = le.fit_transform(df['age_new'])
df.head()
#converting cat to conti...
df['marital'] = le.fit_transform(df['marital'])
df['education'] = le.fit_transform(df['education'])
df['default'] = le.fit_transform(df['default'])
df['housing'] = le.fit_transform(df['housing'])
df['loan'] = le.fit_transform(df['loan'])
df['contact'] = le.fit_transform(df['contact'])
df['poutcome'] = le.fit_transform(df['poutcome'])
df['y'] = le.fit_transform(df['y'])
df.head(10)
df.apply(lambda x : len(x.unique()))
#explore numerical columns
sns.catplot('job',kind = 'count',data = df,aspect =3)
df['job'].replace({'unknown':'self-employed'},inplace=True)
sns.catplot('job',kind = 'count',data = df,aspect =3)
#sns.catplot('day',kind = 'count',data = df,aspect =3)
sns.catplot('campaign',kind = 'count',data = df,aspect =3)
sns.catplot('previous',kind = 'count',data = df,aspect =3)
df['month'] = le.fit_transform(df['month'])
df['job'] = le.fit_transform(df['job'])
df.head()
#treat for age column 
dur_q1=df['duration'].quantile(q = 0.25)
dur_q2=df['duration'].quantile(q = 0.50)
dur_q3=df['duration'].quantile(q = 0.75)
dur_q4=df['duration'].quantile(q = 1.00)
print('Quartiles:',dur_q1,dur_q2,dur_q3,dur_q4)
outliers2=dur_q3+1.5*dur_q3-1.5*dur_q1
print('outliers:',outliers2)
def clusters2(y):
    if y<=104:
        return 0
    elif y>104 & y<=185:
        return 1
    elif y>185 & y<=329:
        return 2
    elif y>329 & y<=666.5:
        return 4
    elif y>666.5:
        return 5
df['duration'] = df['duration'].astype('int').apply(clusters2)
#df['age_new'] = le.fit_transform(df['age_new'])
df.head()
df.apply(lambda x : len(x.unique()))
df['marital'].value_counts()
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
min_max_scaler = preprocessing.MinMaxScaler()
df['pdays'] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(df['pdays'])))
df.head()
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
X=df.drop(['balance','y'],axis=1)
y=df['y']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=3)
#model_1 (LogisticRegression)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
lr=LogisticRegression()
lr.fit(X_train,y_train)
y_pred1=lr.predict(X_test)
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
confusion_matrix1=confusion_matrix(y_test, y_pred1)
precision_score1=precision_score(y_test, y_pred1)
recall_score1=recall_score(y_test, y_pred1)
accuracy_score1=accuracy_score(y_test, y_pred1)
f1_score1=f1_score(y_test, y_pred1)
print('confusion_matrix:\n',confusion_matrix1)
print('precision_score:',precision_score1)
print('recall_score:',recall_score1)
print('accuracy_score:',accuracy_score1)
print('f1_score:',f1_score1)
print(classification_report(y_test, y_pred1))

#model_2 (RandomForest)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
y_pred2=rfc.predict(X_test)
confusion_matrix2=confusion_matrix(y_test, y_pred2)
precision_score2=precision_score(y_test, y_pred2)
recall_score2=recall_score(y_test, y_pred2)
accuracy_score2=accuracy_score(y_test, y_pred2)
f1_score2=f1_score(y_test, y_pred2)
print('confusion_matrix:\n',confusion_matrix2)
print('precision_score:',precision_score2)
print('recall_score:',recall_score2)
print('accuracy_score:',accuracy_score2)
print('f1_score:',f1_score2)
print(classification_report(y_test, y_pred2))

#model_3 (DecisionTrees)
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='gini') #criterion = entopy, gini
dtc.fit(X_train, y_train)
y_pred3 = dtc.predict(X_test)
confusion_matrix3=confusion_matrix(y_test, y_pred3)
precision_score3=precision_score(y_test, y_pred3)
recall_score3=recall_score(y_test, y_pred3)
accuracy_score3=accuracy_score(y_test, y_pred3)
f1_score3=f1_score(y_test, y_pred3)
print('confusion_matrix:\n',confusion_matrix3)
print('precision_score:',precision_score3)
print('recall_score:',recall_score3)
print('accuracy_score:',accuracy_score3)
print('f1_score:',f1_score3)

#model_4 (xgboost)
from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X_train, y_train)
y_pred4 = xgbc.predict(X_test)
confusion_matrix4=confusion_matrix(y_test, y_pred4)
precision_score4=precision_score(y_test, y_pred4)
recall_score4=recall_score(y_test, y_pred4)
accuracy_score4=accuracy_score(y_test, y_pred4)
f1_score4=f1_score(y_test, y_pred4)
print('confusion_matrix:\n',confusion_matrix4)
print('precision_score:',precision_score4)
print('recall_score:',recall_score4)
print('accuracy_score:',accuracy_score4)
print('f1_score:',f1_score4)
F_scores = {'Model':  ['Log_R', 'RF','DT','XGB'],
         'conf_matrix': [confusion_matrix1, confusion_matrix2 , confusion_matrix3, confusion_matrix4],
         'precision': [precision_score1,precision_score2,precision_score3,precision_score4],
         'recall': [recall_score1,recall_score2,recall_score3,recall_score4],
         'accuracy': [accuracy_score1,accuracy_score2,accuracy_score3,accuracy_score4],
         'f1': [f1_score1,f1_score2,f1_score3,f1_score4] 
           }
df_scores = pd.DataFrame (F_scores, columns = ['Model','conf_matrix','precision','recall','accuracy','f1'])
df_scores
print(df_scores.to_markdown(tablefmt="grid"))

#Additional graphs to compare actual vs predicted
df1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred1})
ax1 = sns.distplot(df1['Actual'], hist=False, color="red", label="Actual Value")
sns.distplot(df1['Predicted'], hist=False, color="blue", label="Predicted Values" , ax=ax1)
df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})
ax1 = sns.distplot(df2['Actual'], hist=False, color="red", label="Actual Value")
sns.distplot(df2['Predicted'], hist=False, color="blue", label="Predicted Values" , ax=ax1)
df3 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred3})
ax1 = sns.distplot(df3['Actual'], hist=False, color="red", label="Actual Value")
sns.distplot(df3['Predicted'], hist=False, color="blue", label="Predicted Values" , ax=ax1)
df4 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred4})
ax1 = sns.distplot(df4['Actual'], hist=False, color="red", label="Actual Value")
sns.distplot(df4['Predicted'], hist=False, color="blue", label="Predicted Values" , ax=ax1)
