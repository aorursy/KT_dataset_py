import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv('../input/breast-cancer-wisconsin-prognostic-data-set/data 2.csv')

df.head()
df['diagnosis'] = df['diagnosis'].map({'M':0, 'B':1})

y = df['diagnosis']

df.drop(['id', 'Unnamed: 32', 'diagnosis'], axis = 1, inplace = True)

df.head()

cols = list(df.columns)
df.info()

df.describe()
fig, ax = plt.subplots(6,5)

k = 0

for i in range(6):

    for j in range(5):

        ax[i,j].hist(cols[k], data = df, edgecolor= 'black', linewidth= 1, color= 'red')

        plt.subplots_adjust(left=1, bottom=1, right=2, top=2, wspace=1, hspace=1)

        ax[i,j].set_title(str(cols[k]))

        k+=1

fig.set_size_inches(15,10)

plt.gcf()

fig.tight_layout()
from collections import Counter

Counter(y)

sns.countplot(x = y)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit_transform(df)
from sklearn.decomposition import PCA

pca = PCA()

pca.fit_transform(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_),'ro-')

plt.grid()
pca_new = PCA(n_components=8)

X_new = pca_new.fit_transform(df)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.25)
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import LinearSVC,SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score



from sklearn.metrics import confusion_matrix, accuracy_score
models=[LogisticRegression(),LinearSVC(),SVC(kernel='rbf'),KNeighborsClassifier(),RandomForestClassifier(),

        DecisionTreeClassifier(),GradientBoostingClassifier(),GaussianNB()]

model_names=['LogisticRegression','LinearSVM','rbfSVM','KNearestNeighbors','RandomForestClassifier','DecisionTree',

             'GradientBoostingClassifier','GaussianNB']

acc_score=[]



for model in range(len(models)):

    clf=models[model]

    clf.fit(X_train,y_train)

    pred=clf.predict(X_test)

    acc_score.append(accuracy_score(pred,y_test))

     

d={'Modelling Algo':model_names,'Accuracy':acc_score}



acc_table=pd.DataFrame(d)

acc_table
sns.barplot(y='Modelling Algo',x='Accuracy',data=acc_table)

plt.xlabel('Learning Models')

plt.ylabel('Accuracy scores')

plt.title('Accuracy levels of different classification models')