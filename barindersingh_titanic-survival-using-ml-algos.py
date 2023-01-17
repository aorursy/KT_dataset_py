import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#Reading All the Files

train = pd.read_csv('../input/titanic/train.csv')

test  = pd.read_csv('../input/titanic/test.csv')

gender_submission  = pd.read_csv('../input/titanic/gender_submission.csv')

#Merging of two files on the basis of Passenger ID 



merging_gender_submission = pd.merge(test,gender_submission , on = 'PassengerId' , how = 'inner')

Titanic_details = merging_gender_submission

Titanic_details.head(1)


#This dataset contains information about 11 different variables:

#1.Survival

#2.Pclass

#3.Name

#4.Sex

#5.Age

#6.SibSp

#7.Parch

#8.Ticket

#9.Fare

#10.Cabin

#11.Embarked
#Data Visualisation



#1 = Using Scatter Plot

plt.scatter(x = 'Pclass', y = 'Age', data = Titanic_details)

#sns.stripplot(x = 'Pclass', y = 'Age', data = Titanic_details, jitter = True, edgecolor = 'gray')

plt.legend()

plt.xlabel('Pclass')

plt.ylabel('Age')

plt.show()
#2.Box Plot



box_plot = sns.boxplot(x = 'Pclass', y = 'Age', data = Titanic_details)

strip_plot = sns.stripplot(x = 'Pclass', y = 'Age', data = Titanic_details, jitter = True , edgecolor = 'gray')

plt.show()
#3. Histogram

Titanic_details.hist(figsize = (20,25))

plt.show()
#4. Pair plot



#sns.pairplot(Titanic_details , hue = 'Sex')
#5. Violin Plots

f , ax=plt.subplots(1,2,figsize=(12,8))

sns.violinplot("Pclass","Age", hue="Survived", data=Titanic_details,split=True,ax=ax[0])

ax[0].set_title('Pclass and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data = Titanic_details,split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
#6. KDE plot

sns.kdeplot(Titanic_details['Fare'])

sns.FacetGrid(Titanic_details, hue="Survived", size = 10).map(sns.kdeplot, "Fare").add_legend()

#7. jointplot() allows you to basically match up two distplots for bivariate data. With your choice of what kindparameter to compare with:

#“scatter”

#“reg”

#“resid”

#“kde”

#“hex”



sns.jointplot(x= 'Fare', y = 'Age', data = Titanic_details)

sns.jointplot(x= 'Fare', y = 'Age', data = Titanic_details, kind = 'reg')

sns.jointplot(x= 'Fare', y = 'Age', data = Titanic_details , kind = 'resid')

sns.jointplot(x= 'Fare', y = 'Age', data = Titanic_details , kind = 'kde')

sns.jointplot(x= 'Fare', y = 'Age', data = Titanic_details , kind = 'hex')

sns.jointplot(x= 'Fare', y = 'Age', data = Titanic_details, kind = 'scatter')

#8. Swarm Plot

sns.swarmplot(x = 'Pclass' , y = 'Age' , data = Titanic_details, size =10)

#9. heatmap

sns.heatmap(Titanic_details.corr() , cmap = 'coolwarm' )

#10. Factor Plot



#sns.factorplot('Pclass' , 'Survived' , hue = 'Sex', data = Titanic_details)

#sns.factorplot('SibSp', 'Survived', hue = 'Pclass', data = Titanic_details)
print(Titanic_details.shape)

print(Titanic_details.size)
#Data Preprocessing Using PCA



Titanic_details.describe()
Titanic_details = Titanic_details.drop(columns = ['Cabin'])

Titanic_details = Titanic_details.drop(columns = ['Name'])

Titanic_details = Titanic_details.drop(columns = ['Ticket'])

Titanic_details.head(1)
train = train.drop(columns = ['Cabin'])

train = train.drop(columns = ['Name'])

train = train.drop(columns = ['Ticket'])

train.head(1)
Titanic_details['Embarked'].value_counts()

Titanic_details['Embarked'] = Titanic_details['Embarked'].fillna('S')
train['Embarked'].value_counts()

train['Embarked'] = train['Embarked'].fillna('S')

train.head()
Titanic_details = Titanic_details.drop(columns = ['Embarked'])

train = train.drop(columns = ['Embarked'])


from sklearn.impute import SimpleImputer

missingvalueimputer = SimpleImputer(missing_values = np.NaN, strategy = 'mean')

X=train.iloc[:,4].values

X=X.reshape(-1,1)

train.iloc[:,4] = missingvalueimputer.fit_transform(X)

train.isnull().sum()

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

X = train.iloc[:,3].values

train.iloc[:,3] = labelencoder.fit_transform(X)

X = X.reshape(-1,1)





Y = Titanic_details.iloc[:,2].values

Titanic_details.iloc[:,2] = labelencoder.fit_transform(Y)

Y = Y.reshape(-1,1)



train = train[['PassengerId' ,'Pclass' ,'Sex' ,'Age' ,'SibSp' ,'Parch' ,'Fare' , 'Survived']]

train.head(1)
from sklearn.impute import SimpleImputer

missingvalueimputer = SimpleImputer(missing_values = np.NaN, strategy = 'mean')

Y = Titanic_details.iloc[:,3].values

Y = Y.reshape(-1,1)

Titanic_details.iloc[:,3] = missingvalueimputer.fit_transform(Y)



Titanic_details['Fare'].value_counts()

Titanic_details['Fare'] = Titanic_details['Fare'].fillna('7.7500')

Titanic_details.isnull().sum()

Titanic_details
df = [ train , Titanic_details ]

final_sheet = pd.concat(df)

final_sheet
final_sheet = final_sheet.reset_index()

#final_sheet.to_csv('/home/barinder/Desktop/kaggle/titanic/final_sheet.csv')
X_train = final_sheet.iloc[:891,1:8]

Y_train = final_sheet.iloc[:891,8]

X_test = final_sheet.iloc[891:,1:8]

Y_test = final_sheet.iloc[891:,8]

Y_test

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

X_train_min= mms.fit_transform(X_train)

X_test_min = mms.transform(X_test)



from sklearn.preprocessing import StandardScaler

independent_scaler = StandardScaler()

X_train_norm = independent_scaler.fit_transform(X_train)

X_test_norm = independent_scaler.transform(X_test)



from sklearn.decomposition import PCA

pca = PCA(n_components =2)

X_train_pca = pca.fit_transform(X_train_norm)

X_train_pca =pca.transform(X_train_norm)

X_test_pca = pca.fit_transform(X_test_norm)

X_test_pca =pca.transform(X_test_norm)

pca.explained_variance_ratio_

#Logistic Regression MinMax

from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression()

lgr.fit(X_train_min , Y_train)



prediction = lgr.predict(X_test_min)

from sklearn import metrics

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,prediction)

accuracy



prediction.reshape(-1,1)

gender_submission['Survived'] = prediction

dt = gender_submission

dt

dt.to_csv('logisticregression_minmax.csv', index=False)
#Logistic Regression PCA

from sklearn.linear_model import LogisticRegression

lgr = LogisticRegression()

lgr.fit(X_train_pca , Y_train)



prediction = lgr.predict(X_test_pca)

from sklearn import metrics

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,prediction)

accuracy



prediction.reshape(-1,1)

gender_submission['Survived'] = prediction

dt = gender_submission

dt

dt.to_csv('logisticregression_pca.csv', index=False)
#Decision Tree MINMAX

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 1, random_state = 0)

dt.fit(X_train_min , Y_train)



prediction = dt.predict(X_test_min)



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,prediction)

accuracy



prediction.reshape(-1,1)

gender_submission['Survived'] = prediction

dt_pca = gender_submission

dt_pca

dt_pca.to_csv('decisiontree_pca.csv', index=False)
#Decision Tree PCA

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 1, random_state = 0)

dt.fit(X_train_pca , Y_train)



prediction = dt.predict(X_test_pca)



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,prediction)

accuracy
#Random Forest MINMAX

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators =100)

rfc.fit(X_train_min , Y_train)



prediction = rfc.predict(X_test_min)



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test , prediction)

accuracy



prediction.reshape(-1,1)

gender_submission['Survived'] = prediction

dt = gender_submission

dt

dt.to_csv('randomforest_minmax.csv', index=False)
#Random Forest PCA

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators =100)

rfc.fit(X_train_pca , Y_train)



prediction = rfc.predict(X_test_pca)



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test , prediction)

accuracy
#KNN Algorithm MINMAX

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 6)

knn.fit(X_train_min , Y_train)



prediction =knn.predict(X_test_min) 



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, prediction)

print(accuracy)



prediction.reshape(-1,1)

gender_submission['Survived'] = prediction

dt = gender_submission

dt

dt.to_csv('knn_minmax.csv', index=False)
#KNN Algorithm PCA

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 6)

knn.fit(X_train_pca , Y_train)



prediction =knn.predict(X_test_pca) 



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, prediction)

print(accuracy)

#KMeans MINMAX

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2  , n_init =10 , random_state =0 )



kmeans.fit(X_train_min,Y_train)

prediction = kmeans.predict(X_test_min)



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,prediction)

accuracy





prediction.reshape(-1,1)

gender_submission['Survived'] = prediction

kmeans_minmax = gender_submission

kmeans_minmax.to_csv('kMeans_minmax.csv', index=False)
#KMeans 

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2  , n_init =10 , random_state =0 )



kmeans.fit(X_train_pca,Y_train)

prediction = kmeans.predict(X_test_pca)



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test,prediction)

accuracy
#navbayes gaussian minmax

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train_min, Y_train)



prediction = model.predict(X_test_min)

prediction



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, prediction)

print(accuracy)



prediction.reshape(-1,1)

gender_submission['Survived'] = prediction

dt = gender_submission

dt

dt.to_csv('navbayes_minmax.csv', index=False)
#navbayes gaussian pca

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(X_train_pca, Y_train)



prediction = model.predict(X_test_pca)

prediction



from sklearn.metrics import accuracy_score

accuracy = accuracy_score(Y_test, prediction)

print(accuracy)



prediction.reshape(-1,1)

gender_submission['Survived'] = prediction

dt = gender_submission

dt

dt.to_csv('navbayes_pca.csv', index=False)
#SVM minmax

from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train_min, Y_train)



prediction = svc.predict(X_test_min)

prediction



accuracy = metrics.accuracy_score(Y_test , prediction)

accuracy



prediction.reshape(-1,1)

gender_submission['Survived'] = prediction

dt = gender_submission

dt

dt.to_csv('svm_minmax.csv', index=False)
#SVM pca

from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train_pca, Y_train)



prediction = svc.predict(X_test_pca)

prediction



accuracy = metrics.accuracy_score(Y_test , prediction)

accuracy