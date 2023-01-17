import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df=pd.read_table('../input/fruit_data_with_colors.txt')
df.head()
df.describe()
df['fruit_name'].value_counts().plot(kind='bar')
def barchart(feature):

    orange=df[df['fruit_name']=='orange'][feature].value_counts()

    apple=df[df['fruit_name']=='apple'][feature].value_counts()

    lemon=df[df['fruit_name']=='lemon'][feature].value_counts()

    mandarin=df[df['fruit_name']=='mandarin'][feature].value_counts()

    #survived1=survived[1]

    #dead1=dead[0]

    df1 = pd.DataFrame([orange,apple,lemon,mandarin])

    df1.index=['orange','apple','lemon','mandarin']

    df1.plot(kind='bar',stacked=True,figsize=(10,5))
barchart('fruit_subtype')
df['fruit_subtype'].value_counts().plot(kind='bar')
df['color_score'].value_counts().plot(kind='bar')
df['fruit_label'].value_counts().plot(kind='bar')
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier
df.head()
X=df[['mass','width','height','color_score']]
Y=df[['fruit_label']]

X1=np.array(X)

Y1=np.array(Y)
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)
# testting with different types of nearest neighbour in KNN

from sklearn import metrics

klist = list(range(1,30))

scores = []

for k in klist:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    scores.append(metrics.accuracy_score(y_test, y_pred))

    

plt.plot(klist, scores)

plt.xlabel('Value of k for KNN')

plt.ylabel('Accuracy Score')

plt.title('Accuracy Scores for Values of k of k-Nearest-Neighbors')

plt.show()
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
sample_result=[]

MachineLearningAlgo=[]

X=['LinearSVC','DecisionTreeClassifier','KNeighborsClassifier','SVC','GradientBoostingClassifier','RandomForestClassifier']

Z=[LinearSVC(),DecisionTreeClassifier(),KNeighborsClassifier(),SVC(),GradientBoostingClassifier(),RandomForestClassifier()]
for model in Z:

    model.fit(X_train,y_train)      ## training the model this could take a little time

    accuracy=model.score(X_test,y_test)    ## comparing result with the test data set

    MachineLearningAlgo.append(accuracy) 

    ## saving the accuracy
d={'Accuracy':MachineLearningAlgo,'Algorithm':X}

df1=pd.DataFrame(d)
df1
model1=RandomForestClassifier()
model1.fit(X1,Y1)
model.predict(X1)
df2=pd.DataFrame(model.predict(X1))
df2.head()