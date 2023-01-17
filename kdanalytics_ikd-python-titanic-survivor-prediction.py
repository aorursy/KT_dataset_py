%matplotlib inline
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# Importing Classifier Modules
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
df=pd.read_csv('train.csv')
df.head()
df.head()
#CleanData
df['Embarked']=df['Embarked'].fillna('S')
df['Age']=df['Age'].interpolate()
df['is_child'] = df['Age'].apply(lambda x:1 if x <= 15 else 0)
df['family'] = df['SibSp'] + df['Parch']
df['is_alone'] = df['family'].apply(lambda x:1 if x == 0  else 0)
df['is_male'] = df['Sex'].apply(lambda x:1 if x == 'male'  else 0)
df=df.drop(['PassengerId','Name','Ticket','Cabin','Sex'], axis=1)
df=pd.get_dummies(df,prefix=['is'])
X, y = df.drop(['Survived'],axis=1), df['Survived']
train_X, test_X, train_y, test_y = train_test_split(X,y, train_size = 0.80, test_size = 0.20, stratify = y)
dftest=pd.read_csv('test.csv')
dftest.head()
dftest.info()
#DescribeData
sns.distplot(dftest[dftest['Age'].notnull()]['Age'])
#CleanData
dftest['Age']=dftest['Age'].interpolate()
dftest['Fare']=dftest['Fare'].fillna(dftest['Fare'].mean())
dftest['is_child'] = dftest['Age'].apply(lambda x:1 if x <= 15 else 0)
dftest['family'] = dftest['SibSp'] + dftest['Parch']
dftest['is_alone'] = dftest['family'].apply(lambda x:1 if x == 0  else 0)
dftest['is_male'] = dftest['Sex'].apply(lambda x:1 if x == 'male'  else 0)
dftest=dftest.drop(['PassengerId','Name','Ticket','Cabin','Sex'], axis=1)
dftest=pd.get_dummies(dftest,prefix=['is'])
dftest.head()
dftest.info()
def sma_first_classifier(model):
    classifier = model()
    classifier.fit(train_X, train_y)
    #classifier.y_pred = classifier.predict(test_X)
    classifier.scr = round(classifier.score(test_X, test_y)*100, 2)
    return classifier
acc_log_reg = sma_first_classifier(lambda: LogisticRegression(penalty='l2'))
acc_gbc = sma_first_classifier(lambda: GradientBoostingClassifier(learning_rate=0.3))
acc_svc = sma_first_classifier(SVC)
acc_linear_svc = sma_first_classifier(LinearSVC)
acc_knn = sma_first_classifier(KNeighborsClassifier)
acc_decision_tree = sma_first_classifier(DecisionTreeClassifier)
acc_random_forest = sma_first_classifier(lambda: RandomForestClassifier(n_estimators = 1000))
acc_gnb = sma_first_classifier(GaussianNB)
acc_perceptron = sma_first_classifier(Perceptron)
acc_sgd = sma_first_classifier(SGDClassifier)
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Gradient Boosting Classifier', 'Support Vector Machines', 'Linear SVC', 
              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 
              'Perceptron', 'Stochastic Gradient Decent'],
    'Score': [acc_log_reg.scr, acc_gbc.scr, 
              acc_svc.scr, acc_linear_svc.scr, 
              acc_knn.scr, acc_decision_tree.scr, 
              acc_random_forest.scr, acc_gnb.scr, 
              acc_perceptron.scr, acc_sgd.scr]
    #'Pred_list': [acc_log_reg.y_pred, acc_gbc.y_pred, 
      #        acc_svc.y_pred, acc_linear_svc.y_pred, 
           #   acc_knn.y_pred, acc_decision_tree.y_pred, 
             # acc_random_forest.y_pred, acc_gnb.y_pred, 
        #      acc_perceptron.y_pred, acc_sgd.y_pred]
    })
models.sort_values(by='Score', ascending=False)
classifier = RandomForestClassifier(n_estimators=1000)
classifier.fit(train_X, train_y)
predictor_rand_forest = classifier.predict(dftest)
test=pd.read_csv('test.csv')
predict = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predictor_rand_forest
})

predict.to_csv('predictions', index = False)