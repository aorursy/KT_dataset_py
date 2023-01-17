import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization library

import matplotlib.pyplot as plt # mathematical plotting library

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

%matplotlib inline
#read the dataset

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename));

        

wine_df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

wine_df.head(5)
wine_df.quality.value_counts()
from pandas_profiling import ProfileReport

profile = ProfileReport(wine_df);
profile.to_widgets()
wine_df.describe()
#unique values in each column

wine_df.nunique()
range_df = pd.concat([wine_df.max(),wine_df.min(),wine_df.max()-wine_df.min()],axis=1)

range_df.rename(columns={0:"max",1:"min",2:"range"},inplace=True)

range_df
matrix = np.triu(wine_df.corr())

plt.figure(figsize=(14,14))

#plot heat map

g=sns.heatmap(wine_df.corr(),mask= matrix,annot=True,cmap="bwr")
wine_df.hist(bins = 30, figsize=(12,10), color= 'orange');
col=wine_df.columns

for i in range(len(col)-1):

    dss=pd.DataFrame((wine_df.groupby('quality')[col[i]].max()-wine_df.groupby('quality')[col[i]].min()).sort_index())

    sns.barplot(x=dss.index, y=col[i], data=dss)

    plt.show()
#Converting quality into categorical data

#Dividing wine as bad, average, good and best by giving the limit for the quality

bins = [2, 3, 4, 7, 8]

group_names = ['bad', 'average', 'good', 'best']

wine_df['quality'] = pd.cut(wine_df['quality'], bins = bins, labels = group_names)
#converting categorical into binary data

from sklearn.preprocessing import StandardScaler, LabelEncoder

LE = LabelEncoder()

wine_df['quality'] = LE.fit_transform(wine_df['quality'])
# spitting the dataset

X = wine_df.iloc[:,:-1].values

y = wine_df.loc[:,"quality"].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier

knn_class = KNeighborsClassifier(n_neighbors = 4, metric = 'minkowski', p = 2)

knn_class.fit(X_train, y_train)



y_pred_knn = knn_class.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred_knn)

print("KNN accuracy: {:2.2f}%" .format(accuracy_score(y_test, y_pred_knn) * 100) )
from sklearn.ensemble import RandomForestClassifier

rfc_class = RandomForestClassifier(n_estimators=200)

rfc_class.fit(X_train, y_train)



y_pred_rfc = rfc_class.predict(X_test)
#from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred_rfc)

print("random forest accuracy: {:2.2f}%" .format(accuracy_score(y_test, y_pred_rfc) * 100) )
#Now lets try to do some evaluation for random forest model using cross validation.

rfc_eval = cross_val_score(estimator = rfc_class, X = X_train, y = y_train, cv = 5)

print(rfc_eval.mean()*100)