import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import jaccard_similarity_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'../input/titanic/train.csv')      #Training set
test_df = pd.read_csv(r'../input/titanic/test.csv')  #test set




cdf = df[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']] #modified train df

cdf['Age'].fillna(29,inplace=True)
embarked_list = ['S','Q','C']
cdf['Embarked'].fillna(random.choice(embarked_list), inplace=True)
cdf['Fare'].fillna(35.4,inplace=True)




test_cdf = test_df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']] #modified test df

test_cdf['Age'].fillna(30,inplace=True)                               #fill nan position in age with avg age
test_cdf['Embarked'].fillna(random.choice(embarked_list), inplace=True)#fill nan positons in embarked with random location
test_cdf['Fare'].fillna(32.2,inplace=True)




x_train = np.asarray(cdf[['Pclass','Sex','Age','SibSp','Parch','Embarked','Fare']]) #array conversion train
y_train = np.asarray(cdf['Survived'])

x_test = np.asarray(test_cdf[['Pclass','Sex','Age','SibSp','Parch','Embarked','Fare']]) #array conversion test



sex = preprocessing.LabelEncoder()                          #label conversion F/M
sex.fit(['female','male'])
x_train[:,1] = sex.transform(x_train[:,1])
x_test[:,1] = sex.transform(x_test[:,1])

embarked = preprocessing.LabelEncoder()                     #label conversion S/C/Q
embarked.fit(['S','C','Q'])
x_train[:,5] = embarked.transform(x_train[:,5])
x_test[:,5] = embarked.transform(x_test[:,5])




x_train = preprocessing.StandardScaler().fit(x_train).transform(x_train)
x_test = preprocessing.StandardScaler().fit(x_test).transform(x_test)
#x_train1, x_test1, y_train1, y_test1 = train_test_split(x_train, y_train, test_size=0.2, random_state=4) #train-test split





########################################SVM########################################
param = {'C':[0,1,1,2,3,4,5,10,100,100],'gamma':[0.0001, 0.001, 0.005, 0.1, 1, 3, 5]}
gsc = GridSearchCV(svm.SVC(kernel='rbf'),param,cv=5,scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
grid_result = gsc.fit(x_train,y_train)
best_params = grid_result.best_params_

clf = svm.SVC(kernel='rbf', C=best_params["C"], gamma=best_params["gamma"],tol=0.001, cache_size=200, verbose=False, max_iter=-1)
clf.fit(x_train,y_train)

prediction = clf.predict(x_test)

print(prediction)

#print(classification_report(y_test1,prediction))
#print(accuracy_score(y_test1, prediction))

output = pd.DataFrame({'PassengerId':test_df.PassengerId, 'Survived':prediction})
output.to_csv('Titanic_Pavi4.csv', index=False)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import seaborn as sns

df = pd.read_csv(r'../input/titanic/train.csv')      #Training set

cdf = df[['Survived','Pclass','Sex','Age','Embarked']] #modified train df
cdf['Age'].fillna(29,inplace=True)
embarked_list = ['S','Q','C']
cdf['Embarked'].fillna(random.choice(embarked_list), inplace=True)
print(cdf.shape)

cdf['Sex'] = cdf['Sex'].astype('category')
cdf['Embarked'] = cdf['Embarked'].astype('category')

cdf['Sex'] = cdf['Sex'].cat.codes
cdf['Embarked'] = cdf['Embarked'].cat.codes

print(cdf.head(10))

g = sns.PairGrid(cdf,hue='Survived')
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)
g = g.add_legend()
plt.show()

#x_train = np.asarray(cdf[['Pclass','Sex','Age','SibSp','Parch','Embarked']]) #array conversion train
#y_train1 = np.asarray(cdf['Survived'])


