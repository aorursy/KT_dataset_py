# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import export_graphviz

import seaborn as sns

import matplotlib.pyplot as plt
df= pd.read_csv('../input/HR_comma_sep.csv')

df.head(10)
df.isnull().sum()
df.dtypes
df['sales'].unique()
df['salary'].unique()
#Low=

#Medium=

#High=10

df['salary'].replace({'low':1,'medium':5,'high':10},inplace=True)
satisfaction_level=df['satisfaction_level']

last_evaluation=df['last_evaluation']

number_project=df['number_project']

average_montly_hours=df['average_montly_hours']

time_spend_company=df['time_spend_company']

Work_accident=df['Work_accident']
sns.pairplot(df, hue="left", vars=['satisfaction_level', 'last_evaluation', 'average_montly_hours','number_project'])

plt.show()




#X_train,y_train,X_test,y_test=train_test_split(X,y,data_size=0.2)





corr=df.corr()

sns.heatmap(corr,annot=True)
sns.pairplot(df, hue="salary", vars=['satisfaction_level','time_spend_company', 'last_evaluation', 'average_montly_hours','number_project'])

plt.show()
#Creating dummy variables for the column:sales

dummies=pd.get_dummies(df['sales'],prefix='sales')

df=pd.concat([df,dummies],axis=1)

df.drop(['sales'],axis=1,inplace=True)

df.head(10)
#Spilting data into test and train split:

X=df.drop(['left'],axis=1)

y=df['left']
#RandomForestClassifier

model=RandomForestRegressor(n_estimators=100,n_jobs=-1,oob_score=True,random_state=19)



model.fit(X,y)
#RandomForestClassifier

from sklearn.metrics import roc_auc_score





y_pred=model.oob_prediction_



acc=roc_auc_score(y,y_pred)

print('Accuracy :',acc)
#K-Fold Validation

#Evaluating the model for a Ten-Fold Cross Validation



acc=[]

for i in [50,100,150,200,250,300,350,400,500,600]:

    #Training the model

    clf=RandomForestRegressor(n_estimators=i,n_jobs=-1,oob_score=True,random_state=19)

    clf.fit(X,y)

    y_pred=clf.oob_prediction_

    k=roc_auc_score(y,y_pred)

    acc.append(k)

   

print(acc)

    

    

    
import matplotlib.pyplot as plt

xx=[50,100,150,200,250,300,350,400,500,600]

yy=acc

plt.plot(xx,yy)
model.n_outputs_
model.n_features_
#Spilting data into test and train split:

X=df.drop(['left'],axis=1)

y=df['left']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print(X_train.shape,y_train.shape,)

print(X_test.shape,y_test.shape)
def draw_roc_curve(fitted_c, x_test, y_test, title):

    from sklearn.metrics import roc_curve, auc

    import matplotlib.pyplot as plt

    c                                        = fitted_c

    probas                                   = c.predict_proba(x_test)

    false_positive_rate, recall_, thresholds = roc_curve(y_test, probas[:,1])

    roc_auc                                  = auc(false_positive_rate, recall_)

    

    plt.title  ('ROC %.2f %s'%(roc_auc, title))

    plt.legend (loc="lower right")

    plt.plot   ([0,1],[0,1], "r--")

    plt.plot   (false_positive_rate, recall_, 'b', label='AUC = %.2f'%roc_auc)

    plt.xlim   ([0.0,1.1])

    plt.ylim   ([0.0,1.1])

    plt.ylabel ('Recall')

    plt.xlabel ('Fall-out')

    plt.show()







from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score

#DecisionTreeClassifier

dt=DecisionTreeClassifier(max_depth=3,min_samples_leaf=int(0.05*len(X_train)),random_state=19)

boosted_dt=AdaBoostClassifier(dt,algorithm='SAMME',n_estimators=800,learning_rate=0.5)

boosted_dt.fit(X_train,y_train)

y_predicted=boosted_dt.predict(X_test)



print ("Area under ROC curve: %f"%(roc_auc_score(y_test, y_predicted)))



y_test
X_test