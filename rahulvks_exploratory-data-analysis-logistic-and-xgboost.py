import pandas as pd

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

% matplotlib inline

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier, plot_importance



data = pd.read_csv('../input/xAPI-Edu-Data.csv')
len(data)
data.head(4)
data.columns
data['gender'].value_counts()
print('Percentage',data.gender.value_counts(normalize=True))

data.gender.value_counts(normalize=True).plot(kind='bar')
data['PlaceofBirth'].value_counts()
nationality = sns.countplot(x = 'PlaceofBirth', data=data, palette='Set3')

nationality.set(xlabel='PlaceofBirth',ylabel='count', label= "Students Birth Place")

plt.setp(nationality.get_xticklabels(), rotation=90)

plt.show()
pd.crosstab(data['Class'],data['Topic'])
sns.countplot(x='StudentAbsenceDays',data = data, hue='Class',palette='dark')

plt.show()

P_Satis = sns.countplot(x="ParentschoolSatisfaction",data=data,linewidth=2,edgecolor=sns.color_palette("dark"))
# gender comparison Relationship with Pare

plot = sns.countplot(x='Class', hue='Relation', data=data, order=['L', 'M', 'H'], palette='Set1')

plot.set(xlabel='Class', ylabel='Count', title='Gender comparison')

plt.show()



#educational level student belongs (nominal: ‘lowerlevel’,’MiddleSchool’,’HighSchool’)
sns.pairplot(data,hue='Class')
#Graph Analysis Gender vs Place of Birth
import networkx as nx



g= nx.Graph()

g = nx.from_pandas_dataframe(data,source='gender',target='PlaceofBirth')

print (nx.info(g))





plt.figure(figsize=(10,10)) 

nx.draw_networkx(g,with_labels=True,node_size=50, alpha=0.5, node_color="blue")

plt.show()
data.dtypes

Features = data.drop('Class',axis=1)

Target = data['Class']

label = LabelEncoder()

Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index

for col in Cat_Colums:

    Features[col] = label.fit_transform(Features[col])

    

X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=0.2, random_state=52)
Logit_Model = LogisticRegression()

Logit_Model.fit(X_train,y_train)

Prediction = Logit_Model.predict(X_test)

Score = accuracy_score(y_test,Prediction)

Report = classification_report(y_test,Prediction)



print(Prediction)
print(Score)
print(Report)
xgb = XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=100,seed=10)

xgb_pred = xgb.fit(X_train, y_train).predict(X_test)

print (classification_report(y_test,xgb_pred))
print(accuracy_score(y_test,xgb_pred))
plot_importance(xgb)