# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd 

#  reading train data

df=pd.read_csv('/kaggle/input/bnp-paribas-cardif-claims-management/train.csv')

%matplotlib inline

import matplotlib.pyplot as plt

# counting Null values

# plt.yticks

h=df.isna().sum().sort_values()

h=h.to_frame()

h.columns

h.plot(kind='barh',figsize=(150,150))

df.groupby('target')['target'].count().plot(kind='barh')


# identifying columns with categorical data in them

n_columns=df.columns.where(df.dtypes=='object')

n_columns=list(n_columns)

n_columns= [x for x in n_columns if str(x)!='nan']





# dropping columns with categorical data

df=df.drop(n_columns, axis=1)



#dropping null values in the rest of the dataset.

df=df.dropna()     

df=df.reset_index()

# standardizing the data

n_columns=list(df.columns)

del n_columns[0:2]

m=df[n_columns].mean()

s=df[n_columns].std()

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# plt.rcParams['figure.max_open_warning']=300



for col in df.columns:

    print(col, "\n", )

    try:

        df[col].hist()

        plt.show()

    except:

        df[col].value_counts().plot(kind="bar")

        plt.show()
from sklearn.preprocessing import StandardScaler

features = n_columns

# Separating out the features

x = df.loc[:, features].values

# Separating out the target

y = df.loc[:,['target']].values

# Standardizing the features

x = StandardScaler().fit_transform(x)



# showing the coveriance matrix

df[n_columns].cov()
# PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=35)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents , columns = ['principal component 1', 'principal component 2', 'principal component 3','principal component 4', 'principal component 5', 'principal component 6','principal component 7', 'principal component 8', 'principal component 9','principal component 10','principal component 11', 'principal component 12', 'principal component 13','principal component 14', 'principal component 15','principal component 16', 'principal component 17', 'principal component 18','principal component 19', 'principal component 20','principal component 21', 'principal component 22', 'principal component 23','principal component 24', 'principal component 25','principal component 26', 'principal component 27', 'principal component 28','principal component 29', 'principal component 30','principal component 31', 'principal component 32', 'principal component 33','principal component 34', 'principal component 35',])

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
print(pca.explained_variance_ratio_.sum())   #shows the over variance expressed by the first 35 principal components

print(pca.explained_variance_ratio_*100)   #shows variance expressed by each of the first 35 principal components
%matplotlib inline

a1=pca.explained_variance_ratio_*100

b1=a1

for i in range(len(a1)):

    if i>0:

        b1[i]=b1[i]+b1[i-1]

    else:

        continue







plt.plot(b1)

plt.xlabel('Number of PCs')

plt.ylabel('Variance explained')

plt.title('Variance explained by Principal Component Analysis')

plt.show()

# Implementing Random Forest



from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import MinMaxScaler
X = finalDf.iloc[:,0:35].values

y= finalDf.iloc[:,35:].values



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)



from sklearn.ensemble import RandomForestClassifier



regressor = RandomForestClassifier(n_estimators=200, random_state=0)

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
# y_pred1=y_pred

y_pred1= np.rint(y_pred)

y_pred1=y_pred1.astype(int)

# print(type(y_pred1[1]))



from sklearn import metrics

import numpy as np

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



print('Confusion Matrix\n',confusion_matrix(y_test,y_pred1),end='\n\n')

print(classification_report(y_test,y_pred1),end='\n\n\n')

print('Over all test accuracy with Random Forests {}%'.format(accuracy_score(y_test, y_pred1)*100))
# navie bayes classification



from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

y_pred = gnb.fit(X_train, y_train).predict(X_test)



print('Confusion Matrix\n',confusion_matrix(y_test,y_pred),end='\n\n')

print(classification_report(y_test,y_pred),end='\n\n\n')

print('Over all test accuracy with Gaussian Navie Bayes {}%'.format(accuracy_score(y_test, y_pred)*100))
from sklearn.svm import SVC

svclassifier = SVC(kernel='rbf')

svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)





print('Confusion Matrix\n',confusion_matrix(y_test,y_pred),end='\n\n')

print(classification_report(y_test,y_pred),end='\n\n\n')

print('Over all test accuracy with Support Vector Machine {}%'.format(accuracy_score(y_test, y_pred)*100))
import pandas as pd

sample_submission = pd.read_csv("../input/bnp-paribas-cardif-claims-management/sample_submission.csv")

test = pd.read_csv("../input/bnp-paribas-cardif-claims-management/test.csv")

train = pd.read_csv("../input/bnp-paribas-cardif-claims-management/train.csv")