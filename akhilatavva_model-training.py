import pandas as pd
cleandata=pd.read_csv("../input/cleancsv/clean.csv")
print(cleandata.shape)
print(cleandata.head())
data=pd.read_csv("../input/finalcsv/avgword2vectextsummary.csv")
print(data.shape)
print(data.head(2))
cleandata.drop("Summary",inplace=True,axis=1)
cleandata.drop("Text",inplace=True,axis=1)
cleandata.drop("Unnamed: 0",inplace=True,axis=1)


data.drop("Unnamed: 0",inplace=True,axis=1)
cleandata
data
final = pd.concat([data,cleandata.reindex(data.index)], axis=1)
final
final.isnull().sum()
def f(k):
    if k=="positive":
        return 1
    else:
        return 0
cleandata["Score"]=cleandata["Score"].apply(lambda x: f(x))
final
final.drop("Score",inplace=True,axis=1)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
x_train, x_test, y_train, y_test = train_test_split(final, cleandata["Score"], test_size=0.4, random_state=0)
x_train
y_train.shape
logis=LogisticRegression(max_iter=1000)
logis.fit(x_train,y_train)
pred=logis.predict(x_test)
print(pred)
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))
x_train1, x_test1, y_train1, y_test1 = train_test_split(final, cleandata["Score"], test_size=0.3, random_state=0)
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train1,y_train1)
y_pred=gnb.predict(x_test1)
y_pred
y_test1.value_counts()
print((accuracy_score(y_test1,y_pred)))
confusion_matrix(y_test1,y_pred)
print(classification_report(y_test1,y_pred))
import seaborn as sns
x_train2,x_test2,y_train2,y_test2=train_test_split(final,cleandata["Score"],test_size=0.3,random_state=0)
from sklearn.tree import DecisionTreeClassifier
gini=DecisionTreeClassifier(random_state=0)
gini.fit(x_train2,y_train2)
y_pred2=gini.predict(x_test2)
y_pred2
confusion_matrix(y_test2,y_pred2)
print(accuracy_score(y_test2,y_pred2))
gini1=DecisionTreeClassifier(criterion="gini",splitter="random",max_leaf_nodes=10,min_samples_leaf=5,max_depth=5,random_state=0)
gini1.fit(x_train2,y_train2)
y_pred21=gini1.predict(x_test2)
y_pred21
print(accuracy_score(y_test2,y_pred21))

