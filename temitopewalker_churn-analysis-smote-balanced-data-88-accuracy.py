import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
data = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.head()
data.shape
data.info()
data.describe()
data.isnull().sum()
data.nunique()

num_cols = data.select_dtypes('object')

fig = plt.figure(figsize=(20,8))

for col in range(len(num_cols.columns)):
    fig.add_subplot(5,4,col+1)
    sns.countplot(x=num_cols.iloc[:,col])
    plt.xlabel(num_cols.columns[col])

plt.tight_layout()
sns.pairplot(data)

sns.heatmap(data.corr(),annot=True)

data.drop(['customerID'],axis=1,inplace=True)
data.head()
from sklearn.model_selection import train_test_split
# We remove the label values from our training data
X = data.drop(['Churn'], axis=1)

# We assigned those label values to our Y dataset
y = data['Churn']
X = pd.get_dummies(X,drop_first=True)

from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
y=lab.fit_transform(y)
num_int = data.select_dtypes(exclude=['object'])
num_int
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
log_clf=LogisticRegression()
rnd_clf = RandomForestClassifier()
gbr_clf=GradientBoostingClassifier()
svc_clf = SVC()
voting_clf = VotingClassifier([('lr', log_clf), ('rf', rnd_clf),('gbr', gbr_clf), ('SCV', svc_clf)])
voting_clf.fit(X_train, y_train)


for clf in (log_clf, rnd_clf, voting_clf, gbr_clf,svc_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confusion_matrix(y_test, y_pred)
    classification_report(y_test, y_pred)
    print(clf.__class__.__name__, 'r2_score', accuracy_score(y_test, y_pred))
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=2)
X , y =sm.fit_resample(X,y)
sns.countplot(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
voting_clf.fit(X_train, y_train)
for clf in (log_clf, rnd_clf, voting_clf, gbr_clf,svc_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    confusion_matrix(y_test, y_pred)
    classification_report(y_test, y_pred)
    print(clf.__class__.__name__, 'r2_score', accuracy_score(y_test, y_pred))

