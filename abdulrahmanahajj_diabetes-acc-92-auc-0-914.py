import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

data=pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

data.head()
data.shape
data.isnull().sum()
data.describe()
data.loc[(data.SkinThickness<5)& (data.Outcome==0), 'SkinThickness']=int(data[(data.Outcome==0)]['SkinThickness'].mean())

data.loc[(data.SkinThickness<5)& (data.Outcome==1), 'SkinThickness']=int(data[(data.Outcome==1)]['SkinThickness'].mean())
data.loc[(data.Insulin==0)& (data.Outcome==0), 'Insulin']=int(data[(data.Outcome==0)]['Insulin'].mean())

data.loc[(data.Insulin==0)& (data.Outcome==1), 'Insulin']=int(data[(data.Outcome==1)]['Insulin'].mean())
data.columns
sns.countplot(data=data ,x="Outcome",hue="Outcome")

plt.title("Womans have diabetes")
fig, ax=plt.subplots(figsize=(5,5))

sns.boxplot(y="Age",x='Outcome',hue='Outcome',data=data)

plt.title(" Age ")
for i in data.columns:

    plt.figsize=(12,10)

    plt.hist(data[i])

    plt.title(i)

    plt.show()
sns.lineplot(data=data ,x='Age',hue='Age',y="Glucose")
sns.pairplot(x_vars=["Glucose","Pregnancies","BMI"],y_vars="Age",hue="Outcome",data=data)
sns.pairplot(data=data,hue="Outcome")
corr_matrix=data.corr()

corr_matrix
import seaborn as sns

#get correlations of each features in dataset

corr_matrix = data.corr()

top_corr_features = corr_matrix.index

plt.figure(figsize=(8,6))

#plot heat map

g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
X=np.array(data[["Pregnancies","BloodPressure","Glucose","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]])

y=np.array(data.Outcome)

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

log_clf = LogisticRegression()

rnd_clf = RandomForestClassifier()

svm_clf = SVC()

tree_clf = DecisionTreeClassifier()

knn_clf= KNeighborsClassifier()

bgc_clf=BaggingClassifier()

gbc_clf=GradientBoostingClassifier()

abc_clf= AdaBoostClassifier()



voting_clf = VotingClassifier(

estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf), ('tree', tree_clf),('knn', knn_clf),('bg', bgc_clf),

            ('gbc', gbc_clf),('abc', abc_clf)],voting='hard')

voting_clf.fit(x_train, y_train)



from sklearn.metrics import accuracy_score

for clf in  (log_clf, rnd_clf, svm_clf,tree_clf,knn_clf,bgc_clf,gbc_clf,abc_clf,voting_clf):

    clf.fit(x_train,y_train)

    y_pred = clf.predict(x_test)

    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
from sklearn.ensemble import GradientBoostingClassifier

gbc=GradientBoostingClassifier(random_state=0)

gbc.fit(x_train,y_train)

#predict x_test values

pred=gbc.predict(x_test)

#print accuracy for algorithm

print("Accuracy for GradientBoosting data: ",gbc.score(x_test,y_test))

# 0.8779888739049218
#import classification_report

from sklearn.metrics import classification_report

print(classification_report(y_test,pred))
from sklearn.metrics import roc_curve,auc

y_pred_proba = gbc.predict_proba(x_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test,pred)

auc_gbc = auc(fpr, tpr)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr, label='Gradient Boosting (auc = %0.3f)'% auc_gbc )

plt.xlabel('Tpr')

plt.xlabel('Fpr')

plt.title('Gradient Boosting ROC curve')

plt.legend()

plt.show()