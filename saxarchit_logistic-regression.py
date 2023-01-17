import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



dataset = pd.read_csv('../input/framingham.csv')

dataset.head()
count=0

for i in dataset.isnull().sum(axis=1):

    if i>0:

        count=count+1

if count>0:

    print(count, 'Rows(or', round((count/len(dataset.index))*100), '%) with missing values are dropped out of total', str(len(dataset.index)))

    dataset.dropna(axis=0,inplace=True)

    print('Now dataset has', len(dataset.index),' rows')
X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,-1].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
plt.figure(figsize = (20,10))

for i in dataset.columns:

    plt.hist(dataset[i], bins=2, alpha =0.2, label=i)

plt.legend()

plt.title("Data Distribution")
def plot_hists(dataframe, features, rows, cols):

    fig=plt.figure(figsize=(20,20))

    

    for i, feature in enumerate(features):

        ax=fig.add_subplot(rows,cols,i+1)

        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')

        ax.set_title(feature+" Distribution",color='DarkRed')

        

    fig.tight_layout()  

    plt.show()

plot_hists(dataset,dataset.columns,5,4)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()



classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

print(cm)
from sklearn.metrics import accuracy_score as score

print('Accuracy')

print(score(y_test,y_pred)*100)