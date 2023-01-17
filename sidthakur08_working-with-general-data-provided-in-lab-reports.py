import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.shape
data['cp'].value_counts()
target = data.pop('target')
a = pd.get_dummies(data['cp'], prefix = "cp")
b = pd.get_dummies(data['sex'], prefix='sex')
c = pd.get_dummies(data['fbs'], prefix='fbs')
data = pd.concat([data,a,b,c,target],axis=1)
data.head(10)
drop_col = ['restecg','thalach','exang','oldpeak','slope','cp','ca','thal','sex','fbs']
data = data.drop(columns=drop_col)
data.head(10)
data.isna().sum()
X = data.iloc[:,0:11]
y = data.iloc[:,-1]
display(X.head())
display(y.head())
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.33)
accuracies = {}

#svm
svc = SVC(random_state=0,kernel ='linear')
svc.fit(X_train,y_train)

acc_svm = svc.score(X_test,y_test)*100
print(acc_svm)
#naive bayes
nb = GaussianNB()
nb.fit(X_train,y_train)

acc_nb = nb.score(X_test,y_test)*100
acc_nb
#logistic regression
lr = LogisticRegression(random_state=0)
lr.fit(X_train,y_train)

acc_lr = lr.score(X_test,y_test)*100
acc_lr
#decision trees
dt = DecisionTreeClassifier(random_state=4)
dt.fit(X_train,y_train)

acc_dt = dt.score(X_test,y_test)*100
acc_dt
#random forest
'''score_list = dict()
for i in range(100):'''
rf = RandomForestClassifier(random_state=69)
rf.fit(X_train,y_train)

acc_rf = rf.score(X_test,y_test)*100
'''    score_list[i]=acc_rf
k = max(score_list,key=score_list.get)
k'''
acc_rf
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train, y_train)
acc_knn = knn.score(X_test,y_test)*100
acc_knn
scoreList = []
for i in range(1,20):
    knn2 = KNeighborsClassifier(n_neighbors = i)  # n_neighbors means k
    knn2.fit(X_train, y_train)
    scoreList.append(knn2.score(X_test, y_test))
    
plt.plot(range(1,20), scoreList)
plt.xticks(np.arange(1,20,1))
plt.show()

acc = max(scoreList)*100
acc
test = pd.read_csv('/kaggle/input/heartdiseaseself/data_self.csv')
test['pain score'] = test['pain score'].fillna(test['pain score'].mode()[0])
test
def pred(model,d):
    d = d.dropna().reset_index(drop=True)
    d['cp_0']= None
    d['cp_1']= None
    d['cp_2']=None
    d['cp_3']=None
    d['sex_0']=None
    d['sex_1']=None
    d['fbs_0']=None
    d['fbs_1']=None
    for i in d.index:
        ps = d.loc[i]['pain score']
        fbs = d.loc[i]['fasting blood sugar']
        sex = d.loc[i]['sex']
        if ps>=0 and ps<3:
            d['cp_0'].loc[i] = 1
            d['cp_1'].loc[i] = 0
            d['cp_2'].loc[i] = 0
            d['cp_3'].loc[i] = 0
        elif ps>=3 and ps<6:
            d['cp_0'].loc[i] = 0
            d['cp_1'].loc[i] = 1
            d['cp_2'].loc[i] = 0
            d['cp_3'].loc[i] = 0
        elif ps>=6 and ps<9:
            d['cp_0'].loc[i] = 0
            d['cp_1'].loc[i] = 0
            d['cp_2'].loc[i] = 1
            d['cp_3'].loc[i] = 0
        elif ps>=9 and ps<=10:
            d['cp_0'].loc[i] = 0
            d['cp_1'].loc[i] = 0
            d['cp_2'].loc[i] = 0
            d['cp_3'].loc[i] = 1
        if sex == 'male':
            d['sex_0'].loc[i]=0
            d['sex_1'].loc[i]=1
        else:
            d['sex_0'].loc[i]=1
            d['sex_1'].loc[i]=0
        if fbs >=100:
            d['fbs_0'].loc[i]=0
            d['fbs_1'].loc[i]=1
        elif fbs<100:
            d['fbs_0'].loc[i]=1
            d['fbs_1'].loc[i]=0
    d = d.drop(columns = ['pain score','sex','fasting blood sugar'])
    display(d)
    p = model.predict(d)
    return p
import time
a = time.time()
print(pred(rf,test))
b = time.time()
print(str(b-a))
