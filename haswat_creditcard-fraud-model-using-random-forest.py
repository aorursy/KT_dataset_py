import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
data= pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

data.head()
data.info()
plt.figure(figsize=(12,8))



total = float(len(data["Class"]) )

ax = sns.countplot(x="Class", data=data)

for p in ax.patches:

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}'.format((height/total)*100),

            ha="center") 

plt.title('Creditcard Fraud \n 0=non-fraud || 1=fraud',color='black',fontsize=15)

plt.show()
sns.distplot(data['Amount'],color='g')

plt.title('Distribution of transaction amount',fontsize=10)

plt.show()
sns.distplot(data['Time'],color='b')

plt.title('Distribution of transaction Time',fontsize=10)

plt.show()
correlation=data.drop(['Amount','Time','Class'],axis=1)
sns.heatmap(correlation.corr(),cmap='PuBu')
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
X=data.drop(['Class'],axis=1)

Y=data['Class']
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.3,random_state=30)
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42,

                                   max_features = 'auto', max_depth = 10)

classifier.fit(Xtrain, Ytrain)
tmp = pd.DataFrame({'Feature': X, 'Feature importance': classifier.feature_importances_})

tmp = tmp.sort_values(by='Feature importance',ascending=False)

plt.figure(figsize = (7,4))

plt.title('Features importance',fontsize=14)

s = sns.barplot(x='Feature',y='Feature importance',data=tmp)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.show() 
from sklearn.metrics import confusion_matrix

predicted = classifier.predict(Xtest)

predicted_proba = classifier.predict_proba(Xtest)



matrix = confusion_matrix(Ytest, predicted)



print(matrix)
from sklearn.metrics import accuracy_score, make_scorer

classifier.fit(Xtrain, Ytrain)

predictions = classifier.predict(Xtest)

accuracy_score(y_true = Ytest, y_pred = predictions)