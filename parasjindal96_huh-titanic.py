import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")



train.describe(include="all")
train.sample(5)
print(pd.isnull(train).sum())
#survival by gender

sns.barplot(x='Sex', y='Survived', data=train)
# a method to print %age survived for particular class

#can anyone suggest a method so that they get printed in sorted order either by %age or by labels

def perSurvived(m):

    print(m)

    for i in train[m].unique():

        if len(train["Survived"][train[m]==i].value_counts())!=1:

            print(i,"survived:",train["Survived"][train[m]==i].value_counts(normalize=True)[1]*100)

        elif train["Survived"][train[m]==i].value_counts().index[0]==0:

            print(i,"survived:",0)

        else:

            print(i,"survived:",100)
#check percentage of male and female passenger survived

perSurvived("Sex")
#survival by pclass

sns.barplot(x="Pclass", y="Survived",data=train)

perSurvived("Pclass")
#survival by siblings/spouse

sns.barplot(x="SibSp", y="Survived", data=train)

perSurvived("SibSp")
#survival by parch

sns.barplot(x="Parch", y="Survived", data=train)
#categorising ages into classes like baby, adult etc.



def simplify_ages(df):

    bins=(0,5,12,18,24,35,60,120)

    group_names=['baby','child','teen','student','young adult','adult','senior']

    categories=pd.cut(df.Age,bins,labels=group_names)

    df.Age=categories

    return df



simplify_ages(train)

simplify_ages(test)



sns.barplot(x="Age", y="Survived", data=train)
train["CabinBool"]=(train["Cabin"].notnull().astype('int'))

test["CabinBool"]=(test["Cabin"].notnull().astype('int'))

perSurvived("CabinBool")

sns.barplot(x="CabinBool", y="Survived", data=train)
train=train.drop(['Cabin'], axis=1)

test=test.drop(['Cabin'],axis=1)

train=train.drop(['Ticket'],axis=1)

test=test.drop(['Ticket'],axis=1)
# first fill all the missing values= cabin, age, embarked

# we will fill the embarked with the mode of embarked

s=train["Embarked"].value_counts().index[0]

train=train.fillna({"Embarked":s})

# maximum of them are from southampton so we will fill the missing values with S



combine=[train,test]



for dataset in combine:

    dataset["Title"]=dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



def mapping(train,test,feature):

    featureMap=dict()

    count=1

    combine=[train,test]

    for i in train[feature].unique():

        featureMap[i]=count

        count=count+1

    #find percentage of each class survived and group them together who have close values

    #can anyone tell me the way to do it? One way is to sort them and 

    for dataset in combine:

        dataset[feature]=dataset[feature].map(featureMap)

        dataset[feature]=dataset[feature].fillna(0)

    return train,test



train,test=mapping(train,test,"Title")



sns.barplot(x="Title", y="Survived", data=train)

#here i am checking which values are close to each other and set there title as same

#sorry for doing this part manually, not sure how to do it automatically

for dataset in combine:

    dataset["Title"]=dataset["Title"].replace([5,6,15,17],5)

    dataset["Title"]=dataset["Title"].replace([8,9,11,12,13,16],6)

    dataset["Title"]=dataset["Title"].replace([10,14],8)

    
train=train.drop("Name",axis=1)

test=test.drop("Name",axis=1)

train.head()
train.describe(include="all")
test.describe(include="all")
#now we will try to predict the age from title

print(pd.crosstab(train["Title"], train["Age"]))

print(pd.crosstab(train["Title"], train["Sex"]))

print(pd.crosstab(train["Title"], train["Survived"]))
#we will fill the age as mode of age that particular title has.

age_title_map=dict()

for i in range(len(train["Title"].value_counts())):

    age_title_map[i+1]=train[train["Title"]==i+1]["Age"].mode()[0]



print(age_title_map)

for i in range(len(train["Age"])):

    if pd.isnull(train["Age"][i]):

        train["Age"][i]=age_title_map[train["Title"][i]]

        

for i in range(len(test["Age"])):

    if pd.isnull(test["Age"][i]):

        test["Age"][i]=age_title_map[test["Title"][i]]



#check which values are still missing

print(pd.isnull(train).sum())

print(pd.isnull(test).sum())
#map age to numerical value

train,test=mapping(train,test,"Age")
#map sex to numerical value

train,test=mapping(train,test,"Sex")
#map embarked to numerical value

train,test=mapping(train,test,"Embarked")
#fill in the missing fare value with the mean of fare of that class



for i in range(len(test["Fare"])):

    if pd.isnull(test["Fare"][i]):

        pclass=test["Pclass"][i]

        test["Fare"][i]=round(train[train["Pclass"]==pclass]["Fare"].mean(),4)

#divide fare into bands

train["FareBand"]=pd.qcut(train["Fare"],4,labels=[1,2,3,4])

test["FareBand"]=pd.qcut(test["Fare"],4,labels=[1,2,3,4])
train=train.drop("Fare",axis=1)

test=test.drop("Fare",axis=1)
sns.barplot(x="FareBand",y="Survived",data=train)
sns.pairplot(data=train)
from sklearn.model_selection import train_test_split



predictors=train.drop(["Survived", "PassengerId"], axis=1)

target=train["Survived"]



x_train, x_val, y_train, y_val=train_test_split(predictors,target, test_size=0.22, random_state=0)

#gaussian naive bayes



from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian=GaussianNB()

gaussian.fit(x_train, y_train)

y_pred=gaussian.predict(x_val)

acc_gaussian=round(accuracy_score(y_pred,y_val)*100,2)

print(acc_gaussian)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_val)

acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_logreg)
from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

y_pred = svc.predict(x_val)

acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_svc)
from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_val)

acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_linear_svc)
from sklearn.linear_model import Perceptron



perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred = perceptron.predict(x_val)

acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_perceptron)

from sklearn.tree import DecisionTreeClassifier



decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred = decisiontree.predict(x_val)

acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_decisiontree)
from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_val)

acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_randomforest)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_val)

acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_knn)
from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_val)

acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_sgd)
from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_val)

acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk)

from keras.models import Sequential

from keras.layers import Dense,Activation,Layer,Lambda,Dropout



def getModel(arr):

    model=Sequential()

    for i in range(len(arr)):

        if i!=0 and i!=len(arr)-1:

            if i==1:

                model.add(Dense(arr[i],input_dim=arr[0],kernel_initializer='normal', activation='relu'))

            else:

                model.add(Dense(arr[i],activation='relu',kernel_initializer='normal'))

                model.add(Dropout(0.1))

    model.add(Dense(arr[-1],kernel_initializer='normal',activation="sigmoid"))

    model.compile(loss="binary_crossentropy",optimizer='rmsprop',metrics=['accuracy'])

    return model
import keras

import matplotlib.pyplot as plt

from IPython.display import clear_output

class PlotLosses(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        

        self.fig = plt.figure()

        

        self.logs = []



    def on_epoch_end(self, epoch, logs={}):

        

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.i += 1

        

        clear_output(wait=True)

        plt.plot(self.x, self.losses, label="loss")

        plt.legend()

        plt.show();

        

plot_losses = PlotLosses()
firstModel=getModel([9,30,20,5,1])
x_train.shape
firstModel.fit(np.array(x_train),np.array(y_train),epochs=100,callbacks=[plot_losses])
sModel=getModel([9,30,100,5,1])

sModel.fit(np.array(x_train),np.array(y_train),epochs=100,callbacks=[plot_losses])
scores=firstModel.evaluate(np.array(x_val),np.array(y_val))

print(scores)

accNN=scores[1]*100
scores=sModel.evaluate(np.array(x_val),np.array(y_val))

print(scores)

accNN2=scores[1]*100
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier',"Neural Network"],

    'Score': [acc_svc, acc_knn, acc_logreg, 

              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,

              acc_sgd, acc_gbk,accNN]})

models.sort_values(by='Score', ascending=False)
predY=np.round(sModel.predict(np.array(test.drop('PassengerId', axis=1)))).astype(int).reshape(1,-1)[0]

#print(predY)
ids = test['PassengerId']

predictions = gbk.predict(test.drop('PassengerId', axis=1))

#print(predictions)

#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predY })

output.to_csv('submissionNN.csv', index=False)