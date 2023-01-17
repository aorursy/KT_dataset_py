import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

import pandas as pd

testfile=pd.read_csv("../input/test.csv")

submissionfile=pd.read_csv("../input/gender_submission.csv")

trainfile=pd.read_csv("../input/train.csv")

testfile

submissionfile.head()

Merge=pd.merge(testfile, submissionfile, on= "PassengerId", how= "inner")

Merge
final=pd.concat([trainfile, Merge])

final=final.reset_index()

final=final.drop('index',axis=1)
final.info()

#final.describe()

#final.isnull().sum()

#final.columns

#final
X=final[['Age', 'Embarked', 'Fare', 'Parch', 'PassengerId','Pclass', 'Sex', 'SibSp']]

Y=final['Survived']

X['Embarked']=X['Embarked'].fillna('S')

X['Sex']=X['Sex'].fillna('Male')

X.isnull().sum()

Y.head(1)
from sklearn.preprocessing import LabelEncoder

X_labelencoder = LabelEncoder()

X.iloc[:,1]=X_labelencoder.fit_transform(X.iloc[:,1])

X.iloc[:,6]=X_labelencoder.fit_transform(X.iloc[:,6])

X.head(1)
from sklearn.preprocessing import Imputer

missingValueImputer = Imputer (missing_values = 'NaN', strategy = 'mean', axis = 0)  

missingValueImputer = missingValueImputer.fit (X.iloc[:,0:4])

X.iloc[:,0:4] = missingValueImputer.transform(X.iloc[:,0:4])

X.isnull().sum()
# 1.scatter plot and stripplot

plt.scatter(x = 'Pclass', y = 'Age', data = final)

sns.stripplot(x = 'Pclass', y = 'Age', data = final, jitter = True, edgecolor = 'gray')

plt.legend()

plt.xlabel('Age')

plt.ylabel('Survived')

plt.show()
#2. Plot a Histogram

final.hist(figsize = (20,25))

plt.show()
# 3. Plot a Boxplot

final.plot.box(figsize=(7,5))

plt.show()



# 4.pie chart

final['Survived'].value_counts().sort_values(ascending=False).head(5)

import matplotlib.pyplot as plt

from matplotlib import style

style.use('fivethirtyeight')

slices = [494,815]

activities = ['Survived','Non_survived']

cols = ['g','b']

outside = (0, 0.1) 

plt.pie(slices,labels=activities,colors=cols,startangle=90,explode=outside,shadow=True)

plt.title('Survival Graph')

plt.legend(loc='upper left')

plt.show()



# 5 .Plot Kernel Density Chart 

final["Survived"].plot.kde()

plt.show()
# 6. pairplot

sns.pairplot(final , hue = 'Survived')

plt.show()
# 7. displot

sns.distplot(final['Pclass'])

sns.distplot(final['Survived'])

plt.show()
# 8. Violin Plots

f , ax=plt.subplots(1,2,figsize=(12,8))

sns.violinplot("Embarked","Age", hue="Survived", data=final,split=True,ax=ax[0])

ax[0].set_title('Embarked and Age vs Survived')

ax[0].set_yticks(range(0,110,10))

sns.violinplot("Sex","Age", hue="Survived", data = final,split=True,ax=ax[1])

ax[1].set_title('Sex and Age vs Survived')

ax[1].set_yticks(range(0,110,10))

plt.show()
# 9. heatmap

sns.heatmap(final.corr() , cmap = 'coolwarm' )

plt.show()
#10. Swarm Plot

sns.swarmplot(x = 'Pclass' , y = 'Age' , data = final, size =10)

plt.show()
# 11. Factor Plot

sns.factorplot('Pclass' , 'Survived' , hue = 'Sex', data = final)

sns.factorplot('SibSp', 'Survived', hue = 'Pclass', data = final)

plt.show()
X_train=X.iloc[:891,:]

X_test=X.iloc[891:,:]

Y_train=Y[:891]

Y_test=Y[891:]

Y_train

# Fit the Logistic regression to the train data.

from sklearn.linear_model import LogisticRegression

LRClassifier = LogisticRegression (random_state = 0)

LRClassifier.fit (X_train, Y_train)

# Predict the values 

prediction = LRClassifier.predict (X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test, prediction))

prediction
#submissionfile=submissionfile.drop(['Survived'],axis=1)

#submissionfile['Survived']=prediction

#submissionfile.to_csv("../input/logistic_prediction.csv")

#logistic=pd.read_csv("../input/logistic_prediction.csv")

#logistic
import matplotlib.pyplot as plt

plt.scatter(X_test['Age'], Y_test, color = 'red')

plt.scatter(X_test['Age'],prediction, color = 'green')

plt.title ('Survival')

plt.xlabel('Age')

plt.ylabel('Survived')

plt.show()
import seaborn as sns

sns.countplot(Y)

plt.show()
#knn

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=4)

knn.fit(X_train,Y_train)

#let us get the predictions using the classifier we had fit above

prediction = knn.predict(X_test)

#print(prediction)

from sklearn.metrics import accuracy_score

print("Accuracy is",accuracy_score(Y_test, prediction))
#submissionfile=submissionfile.drop(['Survived'],axis=1)

#submissionfile['Survived']=prediction

#submissionfile.to_csv("../input/knn_prediction.csv")

#knn=pd.read_csv("../input/knn_prediction.csv")

#knn
#Lets make predictions with new data:

prediction1=knn.predict([['62.000000','1.0','9.6875','0.0','894','2','1','0']])  

print(prediction1) 
survived=final[final['Survived']==1]

not_survived=final[final['Survived']==0]

plt.scatter(survived['Sex'],survived['Age'],color='g')

plt.scatter(not_survived['Sex'],not_survived['Age'],color='b')

plt.xlabel('sex')

plt.ylabel('age')

plt.show()



import seaborn as sns

sns.countplot(survived['Sex'])

plt.xlabel('Survival rate')

plt.show()

sns.countplot(not_survived['Sex'])

plt.xlabel('Non_Survival rate')

plt.show()
x=X[['Age','Sex']]

x_train=x.iloc[:891,:]

x_test=x.iloc[891:,:]

#decision tree

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth=2)

decision_tree.fit(x_train, Y_train)

prediction =decision_tree.predict(x_test)

prediction
from sklearn import metrics

import numpy as np

from sklearn.metrics import accuracy_score

print("Accuracy is",accuracy_score(Y_test, prediction))
#submissionfile=submissionfile.drop(['Survived'],axis=1)

#submissionfile['Survived']=prediction

#submissionfile.to_csv("../input/decisiontree_prediction.csv")

#decisiontree=pd.read_csv("../input/decisiontree_prediction.csv")

#decisiontree


data_feature_names = [ 'Age','Sex']

from sklearn.tree import export_graphviz

from sklearn import tree

from graphviz import Source

from IPython.display import SVG

from IPython.display import display

graph = Source(tree.export_graphviz(decision_tree, out_file=None, feature_names=data_feature_names, filled = True,rounded=True))



display(SVG(graph.pipe(format='svg')))
final['Survived'].value_counts().sort_values(ascending=False).head(5)

import matplotlib.pyplot as plt

from matplotlib import style

style.use('fivethirtyeight')

slices = [494,815]

activities = ['Survived','Non_survived']

cols = ['c','m']

outside = (0, 0.1) 

plt.pie(slices,labels=activities,colors=cols,startangle=90,explode=outside,shadow=True)

plt.title('Survival Graph')

plt.legend(loc='upper left')

plt.show()



#svm

from sklearn.svm import SVC

svc=SVC() #Default hyperparameters

svc.fit(x_train,Y_train)

predict=svc.predict(x_test)

predict

from sklearn import metrics

print('Accuracy Score: of svc default parameters')

print(metrics.accuracy_score(Y_test,predict))
svc=SVC(kernel='linear')

svc.fit(x_train,Y_train)

predict=svc.predict(x_test)

print('Accuracy Score: with default linear kernel')

print(metrics.accuracy_score(Y_test,predict))
svc=SVC(kernel='rbf')

svc.fit(x_train,Y_train)

predict=svc.predict(x_test)

print('Accuracy Score: with default rbf kernel')

print(metrics.accuracy_score(Y_test,predict))
#submissionfile=submissionfile.drop(['Survived'],axis=1)

#submissionfile['Survived']=predict

#submissionfile.to_csv("../input/svm_prediction.csv")

#svm=pd.read_csv("../input/svm_prediction.csv")

#svm
#GaussianNB is specifically used when the features have continuous values.



from sklearn.naive_bayes import GaussianNB

model = GaussianNB()

model.fit(x_train, Y_train)

prediction = model.predict(x_test)



from sklearn.metrics import accuracy_score



print(accuracy_score(Y_test, prediction))

prediction

#submissionfile=submissionfile.drop(['Survived'],axis=1)

#submissionfile['Survived']=prediction

#submissionfile.to_csv("../input/gaussianNB_prediction.csv")

#S=pd.read_csv("../input/gaussianNB_prediction.csv")

#S