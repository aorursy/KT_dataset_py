import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt





df=pd.read_table('../input/fruit_data_with_colors.txt')


plt.figure(figsize=(15,7))

sns.barplot('fruit_subtype','fruit_label',data=df)

plt.show()
fruit_type={'turkey_navel' : 3 ,'unknown' : 4, 'cripps_pink' : 1

            ,'selected_seconds' : 3, 'spanish_belsan' : 4, 'golden_delicious' : 1,

            'braeburn' : 1, 'mandarin' : 2, 'spanish_jumbo' : 3,'granny_smith' : 1}



df['fruit_subtype']=df['fruit_subtype'].map(fruit_type)

df['height'].groupby(df['fruit_label']).mean()
df['width'].groupby(df['fruit_label']).mean()
df['mass'].groupby(df['fruit_label']).mean()
df['color_score'].groupby(df['fruit_label']).mean()
from sklearn.model_selection import train_test_split

cols=['fruit_subtype', 'mass', 'width', 'height','color_score']

X=df[cols]



y=df['fruit_label']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.neighbors import KNeighborsClassifier



neighbours=np.arange(1,9)

train_accuracy=np.empty(len(neighbours))

test_accuracy=np.empty(len(neighbours))



for i in range(len(neighbours)):

    knn=KNeighborsClassifier(n_neighbors=i+1)

    knn.fit(X_train,y_train)

    train_accuracy[i]=knn.score(X_train,y_train)

    test_accuracy[i]=knn.score(X_test,y_test)

plt.title('k-NN Varying number of neighbors')

plt.plot(neighbours, test_accuracy, label='Testing Accuracy')

plt.plot(neighbours, train_accuracy, label='Training accuracy')

plt.legend()

plt.xlabel('Number of neighbors')

plt.ylabel('Accuracy')

plt.show()


knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train,y_train)

knn.score(X_test,y_test)