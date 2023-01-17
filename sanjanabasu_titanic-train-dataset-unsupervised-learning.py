import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv('../input/titanic-machine-learning-from-disaster/train.csv')
df.head()
print('Median fare of survived', df[df['Survived']==1]['Fare'].median())
print('Median fare of not survived', df[df['Survived']==0]['Fare'].median())
print('Number of Unique passenger ids=',len(df['PassengerId'].unique()))
print('Number of Unique Tickets=',len(df['Ticket'].unique()))
print('Number of Unique cabins=',len(df['Cabin'].unique())) ##Not a reliable number due to presence of missing information
def clean_name(x):
    l=[]
    if isinstance(x,str):
        l=x.split(", ")
        x=l[0]
    return(x)

df['Surname'] = df['Name'].apply(clean_name).astype('str')
def clean_title(x):
    l=[]
    t=[]
    if isinstance(x,str):
        l=x.split(", ")
        s=l[1]
        t=s.split(". ")
        x= t[0]
    return(x)

df['Title'] = df['Name'].apply(clean_title).astype('str')
df['Title'].value_counts()
def new_title(x,Sex,Age):
    a=''
    if isinstance(x,str):
        if x in ['Mr', 'Mrs', 'Miss', 'Master']:
            a=x
        else:
            if Sex=='female' and Age<30:
                a='Miss'
            elif Sex=='female' and Age>=30:
                a='Mrs'
            elif Sex=='male' and Age>=18:
                a='Mr'
            else:
                a='Master'
    return(a)            

df['Title'] = df.apply(lambda x: new_title(x['Title'], x['Sex'],x['Age']), axis=1)
sur= df.groupby('Surname').count()['Title']
df['Fam_count']=df['Surname'].map(sur)
def isfam(x):
    if x>1:
        a=1
    else:
        a=0
    return(a)

df['IsFamily']=df['Fam_count'].apply(isfam)
df['new_title']=df['Title'].replace({'Mr':0,'Mrs':1,'Master':2,'Miss':3})
df['Sex']=df['Sex'].replace({'male':0,'female':1})
df1= df.drop(['PassengerId','Name','Ticket','Cabin','Title','Surname','Fam_count'],axis=1)
df1.head()
df1.isnull().sum()
df1['Age'].fillna(df1['Age'].median(),inplace=True)
df1['Embarked'].fillna(df1['Embarked'].mode()[0],inplace=True)
df1.isnull().sum() #Recheck the data
df1['Embarked']=df1['Embarked'].replace({'C':0,'S':1,'Q':2})
df1.info()
df1.head()
x1=df1.iloc[:,[1,2,3,4,5,6,7,8,9]].values
from sklearn import preprocessing
X = preprocessing.scale(x1)
from sklearn.cluster import KMeans
y = np.array(df['Survived'])
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
correct = 0
for i in range(len(x1)):
    predict_me = np.array(x1[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))
pred=kmeans.predict(X)
plt.scatter(X[pred == 0, 0], X[pred == 0, 1], 
            s = 30, c = 'red', label = 'dead')
plt.scatter(X[pred == 1, 0], X[pred == 1, 1], 
            s = 30, c = 'blue', label = 'survived')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 30, c = 'yellow', label = 'Centroids')

plt.legend()
plt.show()
a=df['Survived'].value_counts().values
b=[len(pred[pred==0]),len(pred[pred==1])]
check=pd.DataFrame({'Actual':a,'Predicted':b},columns=['Actual','Predicted'])
check
print('Model accuracy is: %.2f'%((correct/len(X))*100),'%')