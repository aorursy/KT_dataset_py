# to view graphs inline
%matplotlib inline

# base imports
import pandas as pd
import numpy as np

# graphic librarys
import seaborn as sns
# import pydotplus # Kaggle kernel dont have pydotplus
from IPython.display import Image

# sklearn imports
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz

# set style to seaborn graphics
sns.set(style="darkgrid")

# import train and test dataframe
df_train = pd.read_csv("../input/train.csv", header=0)
df_test = pd.read_csv("../input/test.csv", header=0)
# describe the basic information about the dataframe
df_train.describe()
# View the first lines of train dataframe to try understanding data
df_train.head()
# View the first lines of test dataframe to try understanding data
df_test.head()
# Visualize the number of survivers and no survivers
df_train['Survived'].value_counts().plot(kind='bar')
# Visualize the numbers of male and female in dataset
df_train['Sex'].value_counts().plot(kind='bar')
# Visualize the number os male and femele versus survivers
sns.countplot(x="Sex", hue="Survived", data=df_train);
# Visualize the ticket class versus survivers
sns.countplot(x="Pclass", hue="Survived", data=df_train);
# View a brief information about the Age of passagers
df_train['Age'].describe()
# create a ranges of age to group the passagers
df_train['AgeRange'] = pd.cut(df_train['Age'], [0, 15, 40, 80], labels=['child', 'adult', 'aged'])

# visualize the number of passagers per age range
df_train['AgeRange'].value_counts().plot(kind='bar')
# Age Ranges versus survivers
sns.countplot(x="AgeRange", hue="Survived", data=df_train);
# Drop unused data
df_train.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin','Embarked', 'Age', 'Fare'], axis=1, inplace=True)
# view the first lines about the new dataframe
df_train.head()
# Randon forest only work with numerals, its necessary transform data to numbers
# its a lot of way to do this
# in this case i use label encoder to convert

le = preprocessing.LabelEncoder()

df_train['Sex'] = le.fit_transform(df_train['Sex'])
df_train['AgeRange'] = le.fit_transform(df_train['AgeRange'].astype(str))

df_train.head()
# get a 30% of train data to create a train and test data
# to see how perform my algorithm

Train,Test = train_test_split(df_train, test_size = 0.3, random_state=10)

Train_IndepentVars = Train.values[:, 1:4]
Train_TargetVar = Train.values[:,0]
Train_TargetVar=Train_TargetVar.astype('float')

Test_IndepentVars = Test.values[:, 1:4]
Test_TargetVar = Test.values[:,0]
Test_TargetVar = Test_TargetVar.astype('float')

ninstances_testing = Test_TargetVar.size
# make a test with one tree
# to see how perform
# randon_state is for the test can be reproducible

dtc = DecisionTreeClassifier(criterion='entropy', random_state=10, max_depth=5)
dtc.fit(Train_IndepentVars, Train_TargetVar)
predict_dtc = dtc.predict(Test_IndepentVars)

error_sum_dtc = sum(abs(Test_TargetVar - predict_dtc))
error_tax_dtc   = 100.0*error_sum_dtc/ninstances_testing
accuracy_score_dtc = round(accuracy_score(predict_dtc, Test_TargetVar) * 100, 2)

print ("(decision tree) Number of prediction errors: %d/%d\t Accuracy Score: %.2f%%" % (error_sum_dtc, ninstances_testing, accuracy_score_dtc))

# this is for create a tree visualization
# but kaggle dont have pydotplus in kernel and this dont work

# dot_data = StringIO()
# feature_names = list(df_train)
# class_names = ['True', 'False']

# export_graphviz(dtc, out_file=dot_data,  
#                  filled=True, rounded=True, feature_names=feature_names[1:4], proportion=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# display(Image(graph.create_png()))
# Now create a randon forest classifier with 10 tree
# And try predict de test data

rfc = RandomForestClassifier(criterion = "entropy", random_state=10, n_estimators=10, max_depth=5)
rfc.fit(Train_IndepentVars, Train_TargetVar)
predict_rfc = rfc.predict(Test_IndepentVars)

error_sum_rfc = sum(abs(Test_TargetVar - predict_rfc))
error_tax_rfc   = 100.0*error_sum_rfc/ninstances_testing
accuracy_score = round(accuracy_score(predict_rfc, Test_TargetVar) * 100, 2)

print ("(randon forest) Number of prediction errors: %d/%d\t Accuracy Score: %.2f%%" % (error_sum_rfc, ninstances_testing, accuracy_score))
# this is for create a tree visualization
# but kaggle dont have pydotplus in kernel and this dont work

# for dtc in rfc.estimators_:
#     dot_data = StringIO()
    
#     export_graphviz(dtc, out_file=dot_data,  
#                      filled=True, rounded=True, feature_names=feature_names[1:4], proportion=True)
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#     display(Image(graph.create_png()))
# This code if for generate submission file with test daframe 

Train_IndepentVars = df_train[['Pclass', 'Sex', 'AgeRange']]
Train_TargetVar = df_train['Survived']

ids = df_test['PassengerId']

df_test['AgeRange'] = pd.cut(df_test['Age'], [0, 15, 40, 80], labels=['child', 'adult', 'aged'])
df_test.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin','Embarked', 'Age', 'Fare'], axis=1, inplace=True)

df_test['Sex'] = le.fit_transform(df_test['Sex'])
df_test['AgeRange'] = le.fit_transform(df_test['AgeRange'].astype(str))

rfc = RandomForestClassifier(criterion = "entropy", random_state=10, n_estimators=10, max_depth=4)
rfc.fit(Train_IndepentVars, Train_TargetVar)
predict_rfc = rfc.predict(df_test)

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predict_rfc })
output.to_csv('submission.csv', index=False)
