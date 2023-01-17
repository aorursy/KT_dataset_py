import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

#Suppressing all warnings
warnings.filterwarnings("ignore")

%matplotlib inline
df=pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
#checking null values
df.isnull().sum()
df.info()
#converting age from float to int
df['age']=df['age'].astype(int)
corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
corr[abs(corr['DEATH_EVENT']) > 0.1]['DEATH_EVENT']
df.columns
x=df[['age', 'ejection_fraction', 'serum_creatinine', 'serum_sodium', 'time']]
y=df['DEATH_EVENT']
x.head()
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)
#Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr=LogisticRegression(max_iter=10000)
lr.fit(x_train,y_train)
lr_pred=lr.predict(x_test)
s1=accuracy_score(y_test,lr_pred)
print("Logistic Regression Success Rate :", "{:.2f}%".format(100*s1))
print(classification_report(y_test,lr.predict(x_test)))
print(confusion_matrix(y_test,lr.predict(x_test)))
from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(x_train,y_train)
p2=gbc.predict(x_test)
s2=accuracy_score(y_test,p2)
print("Gradient Booster Classifier Success Rate :", "{:.2f}%".format(100*s2))
print(classification_report(y_test,gbc.predict(x_test)))
print(confusion_matrix(y_test,gbc.predict(x_test)))
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
p3=rfc.predict(x_test)
s3=accuracy_score(y_test,p3)
print("Random Forest Classifier Success Rate :", "{:.2f}%".format(100*s3))
print(classification_report(y_test,rfc.predict(x_test)))
print(confusion_matrix(y_test,rfc.predict(x_test)))
from sklearn.svm import SVC
svm=SVC()
svm.fit(x_train,y_train)
p4=svm.predict(x_test)
s4=accuracy_score(y_test,p4)
print("Support Vector Machine Success Rate :", "{:.2f}%".format(100*s4))
print(classification_report(y_test,svm.predict(x_test)))
print(confusion_matrix(y_test,svm.predict(x_test)))
from sklearn.neighbors import KNeighborsClassifier
scorelist=[]
for i in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    p5=knn.predict(x_test)
    s5=accuracy_score(y_test,p5)
    scorelist.append(round(100*s5, 2))
print("K Nearest Neighbors Top 5 Success Rates:")
print(sorted(scorelist,reverse=True)[:5])
print(classification_report(y_test,knn.predict(x_test)))
print(confusion_matrix(y_test,knn.predict(x_test)))
## Pickle
import pickle

# save model
pickle.dump(svm, open('heart.pickle', 'wb'))

# load model
heart_failure_model = pickle.load(open('heart.pickle', 'rb'))
