import pandas as pd 

import os

from sklearn.cluster import KMeans

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv', delimiter=';')

test = pd.read_csv('../input/test.csv', delimiter=';')
print('There are', train.shape[0], 'rows and', train.shape[1], 'columns in the dataset.')
print('There are', test.shape[0], 'rows and', test.shape[1], 'columns in the dataset.')
train[' UNS'].value_counts()

#['Middle','Low','High','very_low']
test[' UNS'].value_counts()

#['Middle','Low','High','Very Low']
le = LabelEncoder()

train[' UNS'] = le.fit_transform(train[' UNS'])

test[' UNS'] = le.fit_transform(test[' UNS'])
X_train = train.iloc[:,:-1]

X_test = test.iloc[:,:-1]

y_train = train.iloc[:,-1]

y_test = test.iloc[:,-1]
for i in range(1,6):

    km = KMeans(n_clusters=i,random_state=0,n_jobs=4)

    km.fit(X_train,y_train)

    y_pred = km.predict(X_test)

    print('n_clusters = ',i)

    print('Accuracy Score: %.3f' % accuracy_score(y_test,y_pred))

    print('Precision Score: %.3f' % precision_score(y_test,y_pred,average='micro'))

    print('Recall Score: %.3f' % recall_score(y_test,y_pred, average='micro'))

    print('f-1 Score: %.3f' % f1_score(y_test,y_pred, average='micro'),'\n')