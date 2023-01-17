import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
stats.chisqprob=lambda chisq,df:stats.chi2.sf(chisq,df)
train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')
train.head()
data_list=[train,test]

def remove_col(data_list):
    trans_data=[]
    col_list=['PassengerId','Name','Ticket','Cabin']
    for data in data_list:
        Data=data.drop(col_list,axis=1)
        trans_data.append(Data)
    Data1=trans_data[0]
    Data2=trans_data[1]
    return Data1,Data2

Train,Test=remove_col(data_list)
Train
data_list2=[Train,Test]
def sex(data_list):
    trans_data=[]
    for data in data_list:
        data['Sex']=data['Sex'].map({'male':0,'female':1})
        trans_data.append(data)
    data1=trans_data[0]
    data2=trans_data[1]
        
    return data1,data2

Train,Test=sex(data_list2)
Train.info()
Test.info()
x_test=Test
sns.countplot(train['Survived'])
sns.countplot(train['Survived'],hue=train['Pclass'])
sns.countplot(train['Sex'],hue=train['Survived'])
sns.distplot(train['Age'])
sns.catplot(x='Survived',y='Age',data=train,kind='box',hue='Pclass',col='Sex')
sns.catplot(x='Survived',kind='count',data=train,height=5,hue='Pclass',col='Sex')
sns.countplot(train['Pclass'])
sns.boxplot(x=train['Pclass'],y=train['Age'])
a=train.groupby('Pclass')['Age']
b=train.groupby('Pclass')['Fare']
print('Median Age of people in Pclass 1 is : {} years \t Mean Fare of people in Pclass 1 is : {:.2f} '.format(a.get_group(1).median(),b.get_group(1).mean()))
print('Median Age of people in Pclass 2 is : {} years \t Mean Fare of people in Pclass 2 is : {:.2f} '.format(a.get_group(2).median(),b.get_group(2).mean()))
print('Median Age of people in Pclass 3 is : {} years \t Mean Fare of people in Pclass 3 is : {:.2f} '.format(a.get_group(3).median(),b.get_group(3).mean()))
def impute(cols):
    age=cols[0]
    pclass=cols[1]
    if pd.isnull(age):
        if pclass==1:
            return 37
        elif pclass==2:
            return 29
        else:
            return 24
    else:
        return age
    
Train['Age'] = Train[['Age','Pclass']].apply(impute,axis=1)
Test['Age'] = Test[['Age','Pclass']].apply(impute,axis=1)
sns.countplot(train['Embarked'])
Train['Embarked']= Train['Embarked'].fillna('S')
Train.info()
print(Test[Test['Fare'].isnull()])
print('\nFor Pclass =3 ,The mean fare was 13.68')
Test['Fare']=Test['Fare'].fillna(13.68)
train_with_dummies=pd.get_dummies(Train,drop_first=True)
test_with_dummies=pd.get_dummies(Test,drop_first=True)
train_with_dummies.head()
x_train=train_with_dummies.drop('Survived',axis=1)
y_train=train_with_dummies['Survived']
x_test=test_with_dummies
lda=LDA()
x_train_lda=lda.fit_transform(x_train,y_train)
x_test_lda=lda.transform(x_test)
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train_lda,y_train)
prediction=classifier.predict(x_test_lda)

results=pd.read_csv('../input/titanic/gender_submission.csv')
y_test=results['Survived']
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,prediction)
sns.heatmap(cm,annot=True)
print('Accurcay score is : {:.2f}%'.format(accuracy_score(y_test,prediction)*100))
classifier_tree=DecisionTreeClassifier(max_leaf_nodes=70000)
classifier_tree.fit(x_train_lda,y_train)
prediction_tree=classifier_tree.predict(x_test_lda)
cm=confusion_matrix(y_test,prediction_tree)
sns.heatmap(cm,annot=True)
accuracy_score(y_test,prediction_tree)
classifier_forest=RandomForestClassifier()
classifier_forest.fit(x_train_lda,y_train)
prediction_forest=classifier_forest.predict(x_test_lda)
cm=confusion_matrix(y_test,prediction_forest)
sns.heatmap(cm,annot=True)
accuracy_score(y_test,prediction_forest)
from sklearn.svm import SVC
classifier_svc=SVC()
param_grid=[
    {'C':[0.25,0.5,0.75,1],'kernel':['linear','poly','rbf'],'gamma':np.linspace(0.1,0.9,10)}
]
grid_search=GridSearchCV(estimator=classifier_svc,
                         param_grid=param_grid,
                         scoring='accuracy',
                         cv=10,
                         n_jobs=-1
)
grid_search.fit(x_train_lda,y_train)
best_accuracy=grid_search.best_score_
best_parameter=grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameter)
from sklearn.ensemble import AdaBoostClassifier
classifier_ada=AdaBoostClassifier(LogisticRegression(),
                                 n_estimators=200,
                                 algorithm='SAMME.R',
                                 learning_rate=0.5)
classifier_ada.fit(x_train_lda,y_train)
prediction_ada=classifier_ada.predict(x_test_lda)
cm=confusion_matrix(y_test,prediction_ada)
sns.heatmap(cm,annot=True)
accuracy_score(y_test,prediction_ada)
from xgboost import XGBClassifier
classifier_xgb=XGBClassifier()
classifier_xgb.fit(x_train_lda,y_train)
prediction_xgb=classifier_xgb.predict(x_test_lda)
cm=confusion_matrix(y_test,prediction_xgb)
sns.heatmap(cm,annot=True)
accuracy_score(y_test,prediction_xgb)
passenger=test['PassengerId']
prediction=pd.DataFrame(prediction)
final_prediction=pd.concat([passenger,prediction],axis=1)
final_prediction=final_prediction.rename(columns={0:'Survived'})
final_prediction