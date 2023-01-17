import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')
# Pointer = train[['Pclass','Survived']].groupby(['Pclass']).sum()
# (Pointer/(len(train)/len(Pointer))).plot.bar()
# Pointer = train[['Sex','Survived']].groupby(['Sex']).sum()
# (Pointer/(len(train)/len(Pointer))).plot.bar()
# Pointer = train[['Age','Survived']].groupby(['Age']).sum()
# (Pointer/(len(train)/len(Pointer))).plot.bar()
# Pointer = train[['Sex','Survived']].groupby(['Sex']).sum()
# (Pointer/(len(train)/len(Pointer))).plot.bar()
# Pointer = train[['SibSp','Survived']].groupby(['SibSp']).sum()
# (Pointer/(len(train)/len(Pointer))).plot.bar()
# Pointer = train[['Parch','Survived']].groupby(['Parch']).sum()
# (Pointer/(len(train)/len(Pointer))).plot.bar()
# Pointer = train[['Fare','Survived']].groupby(['Fare']).sum()
# (Pointer/(len(train)/len(Pointer))).plot.bar()
# Pointer = train[['Embarked','Survived']].groupby(['Embarked']).sum()
# (Pointer/(len(train)/len(Pointer))).plot.bar()
def testFormat(test):
    test.columns = ['Survived']
    test.index.name = None
    test['PassengerId'] = test.index
    test = test[['PassengerId','Survived']]
    return test
def ClassificationAlgo(Function, train, test):
    train = train.replace(['male','female'],[1,0])
    train = train[['Survived','Pclass','Sex','Age','SibSp','Fare']].fillna(0)

    model = Function.fit(X=train.iloc[:,1:],y=train.iloc[:,0])

    test = test[['PassengerId','Pclass','Sex','Age','SibSp','Fare']].fillna(0).replace(['male','female'],[1,0])
    test = testFormat(pd.DataFrame(pd.Series(model.predict(test.iloc[:,1:]), index = test.iloc[:,0])))
    return test
#test['Survived'] = [1 if (x/2 == int(x)) else 0 for x in range(len(test))]
# from sklearn.linear_model import LogisticRegression

# test = ClassificationAlgo(LogReg(random_state=0, solver='lbfgs',multi_class='multinomial'), train, test)
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.datasets import make_classification

# test = ClassificationAlgo(RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0), train, test)

# from sklearn import svm

# test = ClassificationAlgo(svm.SVC(gamma='scale'), train, test)

from sklearn import neighbors

test = ClassificationAlgo(neighbors.KNeighborsClassifier(n_neighbors=5), train, test)
submission = test[['PassengerId','Survived']]
submission.to_csv('submission1.csv',index=False)
