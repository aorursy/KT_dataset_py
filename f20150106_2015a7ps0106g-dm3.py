import pandas as pd

import sklearn as sk

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

'''set sns'''

sns.set()
opcode_frequency_benign=pd.read_csv('../input/opcode_frequency_benign.csv')

opcode_frequency_benign['Class']=0

opcode_frequency_benign.shape

'''change name later'''
opcode_frequency_malware=pd.read_csv("../input/opcode_frequency_malware.csv")

opcode_frequency_malware['Class']=1

opcode_frequency_malware.shape

#m=pd.read_csv("m.csv")

#m['Class']=1

#m.shape

'''small names'''
data_original=pd.concat([opcode_frequency_benign,opcode_frequency_malware])

data=data_original.drop_duplicates() 

data.head()

#data.head
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score

# clean cell

features=data.columns

features=features.drop(['Class','FileName'])

# new cell??

X=data[features]

Y=data['Class']

# X_train?

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.33,random_state=69)
from sklearn.tree import DecisionTreeClassifier



depths = [None,1,2,5,10,15,20,25]

# try more depths?

print("Entropy \n")

for depth in depths:

    decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)

    decision_tree.fit(X_train, Y_train)

    predictions = decision_tree.predict(X_test)

    print("AUC with maximum depth " + str(depth) + " = " + str(roc_auc_score(predictions,Y_test)))
from sklearn.ensemble import AdaBoostClassifier



adaboost = AdaBoostClassifier(random_state = 69)

adaboost.fit(X_train, Y_train)

# check

Y_prediction_adaboost = adaboost.predict(X_test)

roc_auc_score(Y_prediction_adaboost, Y_test)
from sklearn.neighbors import KNeighborsClassifier



neighbors = list(i for i in range(1, 20, 3))

for n in neighbors:

    knn = KNeighborsClassifier(n_neighbors=n, weights='uniform')

    knn.fit(X_train,Y_train)

    predictions=knn.predict(X_test)

    #check

    print("AUC with neihbors " + str(n) + " = " + str(roc_auc_score(predictions,Y_test)))

from sklearn.ensemble import RandomForestClassifier

depths = [1, 2, 5, 10, 15, 20, 25, 30, None]



for depth in depths:

    random_forest = RandomForestClassifier(n_estimators=50, max_depth=depth,random_state=69, warm_start=True)

    random_forest.fit(X_train, Y_train)

    predictions = random_forest.predict(X_test)

    print("AUC with maximum depth " + str(depth) + " = " + str(roc_auc_score(predictions,Y_test)))

# dataframe?

clf = RandomForestClassifier(n_estimators=50, max_depth=None, random_state=69, warm_start=True)

clf.fit(X, Y)
test_X = pd.read_csv("../input/Test_data.csv")

temp = test_X.drop(columns=['FileName', 'Unnamed: 1809'])

ans = pd.DataFrame(clf.predict(temp))

ans

# csv
kaggle = pd.DataFrame(columns=['FileName', 'Class'])

kaggle['FileName']=test_X['FileName']

kaggle['Class'] = ans

kaggle.to_csv('2015A7PS0106G.csv', index=False)

# clean
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(kaggle, title = "Download CSV file", filename = "data.csv"):

    csv = kaggle.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(kaggle)