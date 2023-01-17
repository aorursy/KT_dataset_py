import pandas as pd

import numpy as np



# import datasets

train = pd.read_csv('../input/titanic/train.csv')

test  = pd.read_csv('../input/titanic/test.csv')
for dataset in [train,test]:

    

    # Title : Gender / Married

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')

    dataset['Title'] = dataset['Title'].replace(

        ['Capt', 'Col', 'Countess', 'Don','Dona', 'Dr', 'Jonkheer','Lady','Major', 'Rev', 'Sir'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')    

    dataset['Title'] = dataset['Title'].astype(str)

    

    # Family: 0,1,2

    dataset["Family"] = dataset["Parch"] + dataset["SibSp"]

    dataset['Family'] = dataset['Family'].astype(int)

    dataset.loc[ dataset['Family'] >=  2, 'Family'] = 2

    

    # Embarked : Classification by marina

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    dataset['Embarked'] = dataset['Embarked'].astype(str)

    

    # Convert text to number.

    dataset["Title"] = dataset["Title"].map({'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Other':4})

    dataset["Sex"] = dataset["Sex"].map({'male':0, 'female':1})

    dataset["Embarked"] = dataset["Embarked"].map({'Q':0, 'S':1, "C":2})

       

    # Fare

    dataset.loc[(dataset['Fare'] <= 7.854), 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare']   = 3

    dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4

    dataset['Fare'] = dataset['Fare'].fillna(5)

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    

    # Age

    dataset.loc[(dataset['Age'] <=20), 'Age'] = 0

    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 40), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 60), 'Age']   = 2

    dataset.loc[ dataset['Age'] > 60, 'Age'] = 3

    dataset['Age'] = dataset['Age'].fillna(4)

    dataset['Age'] = dataset['Age'].astype(int)

    

train = train[['Survived',"Title", "Sex","Age","Pclass","Embarked","Family","Fare"]]

test = test[["Title", "Sex","Age","Pclass","Embarked","Family","Fare"]]

train
# The variables to be used only have the following values.

fitures = ["Title", "Sex","Pclass","Embarked","Family","Fare","Age"]



for fiture in fitures :

    white = " "*(10 - len(fiture))+":"

    print(fiture,white,train[fiture].unique())
# Splite The train dataset into train data and validation data.

fitures = ["Title", "Sex","Pclass","Embarked","Family","Fare","Age"]



num = int(len(train)*(7/10)) 

X_train,X_valid = train[fitures][:num],   train[fitures][num:]

y_train,y_valid = train['Survived'][:num],train['Survived'][num:]

print(f"train data : {len(y_train)}\nvalid data : {len(y_valid)}")
import time

import pandas as pd

from tqdm.notebook   import tqdm

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_predict



def train_report(*models,dataset=(X_train,y_train,X_valid, y_valid)):

    columns = ["Name", "Time(sec)","accuracy(%)", "precision(%)","recall(%)","f1-score(%)","confusion" ]

    df = pd.DataFrame(columns=columns)

    

    X_train,y_train,X_valid,y_valid = dataset



    for model in tqdm(models) :

        model_name = str(model.__class__.__name__)

        print(model_name, end="...")

        

        # Time measurement

        start = time.time()

        

        # Trainning start

        model.fit(X_train,y_train)

        

        # report

        y_pred     = cross_val_predict(model, X_valid, y_valid, cv=3)     

        clf_report = classification_report(y_valid,y_pred, output_dict =True)

        

        accuracy   = clf_report["accuracy"]                # accuracy

        precision  = clf_report['macro avg']['precision']  # precision

        recall     = clf_report['macro avg']['recall']     # recall

        f1_score   = clf_report['macro avg']['f1-score']

        confusion  = confusion_matrix(y_valid, y_pred)     # confusion_matrix

        

        accuracy,precision,recall,f1_score = [round(100*x,2) for x in [accuracy,precision,recall,f1_score]]

        

        train_time = round(time.time() - start,2)



        # save data

        new_row = {f"{columns[0]}":model_name, # name

                   f"{columns[1]}":train_time, # training time

                   f"{columns[2]}":accuracy,   # accuracy

                   f"{columns[3]}":precision,  # precision

                   f"{columns[4]}":recall,     # recall 

                   f"{columns[5]}":f1_score,   # f1_score 

                   f"{columns[6]}":confusion,  # confusion_matrix 

                  }

        

        df = df.append(new_row,ignore_index=True)    

        df = df.drop_duplicates(["Name"],keep="last")

        print("complite..!")

    return df
from sklearn.svm          import SVC

from sklearn.naive_bayes  import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors    import KNeighborsClassifier

from sklearn.tree         import DecisionTreeClassifier

from sklearn.ensemble     import RandomForestClassifier

from sklearn.ensemble     import ExtraTreesClassifier

from sklearn.ensemble     import BaggingClassifier

from sklearn.ensemble     import VotingClassifier



rand = 1234



svm_clf = SVC()

gnb_clf = GaussianNB()

sgd_clf = SGDClassifier()

log_clf = LogisticRegression()

knn_clf = KNeighborsClassifier(n_neighbors = 15)

rdf_clf = RandomForestClassifier(n_estimators=100)

ext_clf = ExtraTreesClassifier(n_estimators=5,random_state=rand)



bag_clf=BaggingClassifier(

    DecisionTreeClassifier(random_state=rand),n_estimators=20,

    max_samples=50,bootstrap=True,n_jobs=-1,random_state=rand)



voting_clf=VotingClassifier(estimators=[

    ('svm',svm_clf),

    ('knn',knn_clf),

    ('rdf',rdf_clf),

    ('ext',ext_clf)

], voting='hard')



models =  [svm_clf, gnb_clf, sgd_clf, log_clf, knn_clf, rdf_clf, ext_clf, bag_clf, voting_clf]
clf_data= train_report(*models)

clf_data
X = train[["Title", "Sex","Pclass","Embarked","Family","Fare","Age"]]

y = train['Survived']



for model in tqdm(models):

    model.fit(X,y)

    a = pd.DataFrame({"PassengerId":range(892,1310),"Survived":model.predict(test)})

    a = a.set_index("PassengerId")

    a.to_csv(f"{model.__class__.__name__}.csv")