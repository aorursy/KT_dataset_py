import numpy as np # linear algebra

import pandas as pd # data processing, CSV file.

import pandas_profiling #Profiling is a process that helps us on understanding the data.

import plotly as pltoty

import matplotlib.pyplot as plt

#plt.style.use('ggplot')

#%matplotlib

import seaborn as sns

#import plotly.offline #importing plotly in offline mode.

#import cufflinks as cf #importing cufflinks in offline mode.

#cf.go_offline()

#cf.set_config_file(offline=False,world_readable=True)

import pprint # “pretty printer” for producing aesthetically pleasing representations of data structures.

from sklearn.model_selection import train_test_split #seperate the data into training and test datasets.



from sklearn import tree

from sklearn.tree import DecisionTreeClassifier   #Trees can be used as classifier or regression models.

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import metrics

from sklearn.metrics import accuracy_score #Accuracy classification score.

from sklearn.metrics import confusion_matrix #

from sklearn.model_selection import cross_val_score,LeaveOneOut,KFold

#from scipy.stats import sem

from sklearn.model_selection import GridSearchCV

#from IPython.core.interactiveshell import InteractiveShell #Printing all the outputs of a cell.

#InteractiveShell.ast_node_interactivity='all'



import missingno as msno #missing data visualizations





#import pdb #python debugger

#pdb.pm()

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
train.profile_report()

test.profile_report()
train.head()
#remove unwanted columns for now

features_df = ['Pclass','Sex','Age','SibSp','Parch','Embarked']

titanic_df = train[features_df]

test_df = test[features_df]

#titanic_df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

#test_df = test.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
#visualize the missing data

msno.bar(titanic_df)
# Transform Sex into binary values 0 and 1

sex = pd.Series(np.where(titanic_df.Sex == 'male' , 1 , 0 ) , name = 'Sex')

sex_test = pd.Series(np.where(test_df.Sex == 'male' , 1 , 0 ) , name = 'Sex')
# Create a new variable for every unique value of Embarked

embarked = pd.get_dummies(titanic_df.Embarked , prefix='Embarked' )

embarked_test = pd.get_dummies(test_df.Embarked , prefix='Embarked')



# Create a new variable for every unique value of Embarked

pclass = pd.get_dummies( titanic_df.Pclass , prefix='Pclass' )

pclass_test = pd.get_dummies( test_df.Pclass , prefix='Pclass' )
# Create dataset

imputed = pd.DataFrame()

imputed_test = pd.DataFrame()



# Fill missing values of Age with the average of Age (mean)

imputed[ 'Age' ] = titanic_df.Age.fillna( titanic_df.Age.mean() )

imputed_test[ 'Age' ] = test_df.Age.fillna( test_df.Age.mean() )





# Fill missing values of Fare with the average of Fare (mean)

#imputed[ 'Fare' ] = titanic_df.Fare.fillna( titanic_df.Fare.mean() )

#imputed_test[ 'Fare' ] = test_df.Fare.fillna( test_df.Fare.mean() )
# Select which features/variables to include in the dataset from the list below:

# imputed , embarked , pclass , sex , family , cabin , ticket



clean_df = pd.concat( [ imputed , embarked , sex,  pclass ] , axis=1 )



clean_df_test = pd.concat( [ imputed_test , embarked_test, sex_test,  pclass_test ] , axis=1 )
clean_df.head()
clean_df_test.head()
X = clean_df

y = train.Survived



train_X , test_X , train_y , test_y = train_test_split( X , y , train_size = .7 )
dTree_model  = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,

            max_features=None, max_leaf_nodes=5,

            min_samples_leaf=1,

            min_samples_split=2, min_weight_fraction_leaf=0.0,

            presort=False, random_state=None, splitter='best')



dTree_model.fit(train_X,train_y)
#score the predicted output from model on our test data against our ground truth test data.

y_predict = dTree_model.predict(test_X)



#accuacy for Train Data

acc_decisiontree  = round(accuracy_score(test_y,y_predict)*100,2)

acc_decisiontree
#confusion Matrix

cm = confusion_matrix(test_y,y_predict)

cm_df = pd.DataFrame(cm,columns=['Predicted Result: Not Survival', 'Predicted Result: Survival'],index=['Not Survival', 'Survival'])

plt.figure(figsize=(5,5))

sns.heatmap(cm,annot=True,fmt='g')

cm_df
#accuracy for Test Data

y_predict_test = dTree_model.predict(test_X)

accuracy_score(test_y,y_predict_test)
#importance of features.

pd.DataFrame(dTree_model.feature_importances_,columns=["Importance"],index=train_X.columns)                                             
import graphviz 



dot_data = tree.export_graphviz(dTree_model,out_file=None,feature_names=clean_df.columns,class_names=True,filled=True,rounded=True)

treeview = graphviz.Source(dot_data)

treeview
loocv = LeaveOneOut()

results = cross_val_score(dTree_model,X,y,cv=loocv)

results.mean()
kfold = KFold(n_splits=10,random_state=5)



result = cross_val_score(dTree_model,X,y,cv=kfold)

result.mean()
#initiate model object

model_NB = GaussianNB()
#fit the model with Train data

model_NB.fit(train_X,train_y)



predict_train_NB = model_NB.predict(train_X)

predict_train_NB



#Accuracy of the model for Train data

accuracy_NB = accuracy_score(train_y,predict_train_NB)

print(" NavieBayes Accuracy")

print(" {0:4f}" .format(accuracy_NB))

#Accuracy of the model for Test data

pred_test_y = model_NB.predict(test_X)

accuracy_score(test_y, pred_test_y)
print("Classification Report")

print(metrics.classification_report(train_y,predict_train_NB,labels=[1, 0]))
clean_df_test.head()
predict_submission = dTree_model.predict(clean_df_test)

predict_submission
#DataFrame with the passengers ids

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predict_submission})



#to save 

submissionfile_titanic = 'Titanic Result V2.csv'

submission.to_csv(submissionfile_titanic,index=False)
