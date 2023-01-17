# Libraries for data analysis
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir("../input"))

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

test_columns = test_data.columns.tolist()
train_columns = train_data.columns.tolist()
print('There are ',len(test_columns),' test columns ')
print('There are ',len(train_columns),' train columns \n')
print('***********columns in test data***********')
print(test_data.columns.values,'\n')
print('***********columns in train data***********')
print(train_data.columns.values,'\n')
diff= list(set(train_columns)-set(test_columns))
print('*************************************************')
print('The extra column in train data DataFrame is',diff)

# There are 891 values in every column in training data except for age which has 714 values 
print('*******The NaN values in each column******* \n')
print(train_data.isnull().sum(),'\n')

print('*******NaN values as % of 891 ******* \n')
print((train_data.isnull().sum()/891)*100)

train_data = pd.read_csv('../input/train.csv')
train_clean = train_data 
for col in train_clean.columns:
    if train_clean[col].isnull().sum()/891 >= 0.6:
        train_clean =train_clean.drop(col,axis =1)
        
# train_clean=train_clean.dropna(axis=0)        
#Need to check if there are any more NaNs in the new dataframe 
train_clean.isnull().sum()/891
train_clean.head()
train_clean.isnull().sum()
train_clean.describe()
# train_clean.isnull().values.any()
train_clean.sample(5)
'''
There are 177 NaN values in Age. This is because there are empty Age values for whatever reason. I am going to fill these
empty values with the mean age
'''
train_clean['Age'] =  train_clean['Age'].fillna(train_clean['Age'].mean())
train_clean.isnull().sum()

train_clean= train_clean.fillna(train_clean.mean())

train_clean.isnull().sum()
train_clean=train_clean.dropna(axis=0)
train_clean['Embarked'].isnull().sum()
test_data = pd.read_csv('../input/test.csv')
test_data.sample(5)
# test_data.isnull().sum()
test_data = test_data.drop('Cabin', axis=1)
test_data.describe()
# test_data['Age'].count()
# test_data.columns
print('*******NaN values as % of the count of the values in each column ******* \n')
# test_data.isnull().sum()/test_data[]
for col in test_data.columns:
    print(col,':',(test_data[col].isnull().sum()/test_data[col].count())*100,'% values are NaNs')
# FIX AGE AND FARE COLUMN IN THE TEST SET.

## AGE 
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
test_data.isnull().sum() # no NaNs in the Age column
# test_data.sample(5)

## FARE
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())
# test_data.isnull().sum()  # no NaNs in the Fare column

test_clean = test_data
test_clean.sample(10)

test_clean.isnull().sum()
# temp variables just so I don't have to reload training, testing data and then cleaning it
train_try = train_clean
test_try = test_clean
data_all = [train_try,test_try]
train_try['FareBand'] = pd.qcut(train_try['Fare'], 4)
test_try['FareBand'] = pd.qcut(test_try['Fare'],4)
train_try[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
# test_clean[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
# train_clean.sample(5)
# train_try  = train_clean
# test_try = test_clean
type(train_try)
test_try.sample(5)

for dataset in data_all:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


train_try = train_try.drop(['FareBand'], axis=1)
test_try = test_try.drop(['FareBand'], axis =1)
train_clean = train_try 

train_clean.sample(10)
test_clean = test_try 

test_clean.sample(10)
# pd.DataFrame(train_clean.corr()['Survived']).reset_index()
train_clean.corr()['Survived']
train_clean['Sex'] =  train_clean['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test_clean['Sex'] = test_clean['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
test_clean.sample(5)
# train_clean.sample(5)
train_clean.sample(5)
# train_clean.corr()['Survived']
sns.heatmap(train_clean.corr(),cmap='RdYlGn')
fig = plt.gcf()
plt.show()
# Passenger class with survived. 
class_surv = pd.DataFrame(train_clean[['Pclass','Survived']]).groupby('Pclass', as_index=False).count()
plt.hist(class_surv['Pclass'], weights=class_surv['Survived'])
plt.xticks(np.arange(1,3))
plt.grid()
plt.title('People survived based on their Passenger class')

train_clean.sample(5)
train_clean[["Survived", "Sex"]].groupby(['Sex'], as_index=False).count()#.mean().sort_values(by='Survived',ascending=False)
train_clean[["Survived", "Sex"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False)

jack_df = train_clean[["Pclass","Sex","Age", "Parch","Fare", "Survived"]]

jack_df.sample(5)
test_clean = test_clean[["Pclass","Sex","Age", "Parch","Fare"]]

test_clean.sample(5)
# jack_ = {'Pclass':3, 'Sex':0, 'Age': 20, 'Parch':0, 'Fare':0}
jack_ = {'Pclass':[3], 'Sex':[0], 'Age': [20], 'Parch':[0], 'Fare':[0]}
jack_test= pd.DataFrame(jack_, columns=['Pclass','Sex','Age','Parch','Fare'])
jack_test
## MACHINE LEARNING LIBRARIES 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Let's set up data
# all_features.head(5)
x_train = jack_df.drop('Survived', axis=1)
y_train = jack_df[["Survived"]]
# x_test  = test_clean.drop("PassengerId", axis=1).copy()
x_test = test_clean.copy()
x_train.shape,y_train.shape, x_test.shape#,  y_test.shape
x_train.head(5)

x_test.head(5)
y_train.head()
def jack_output(jack_p):
    for i in jack_p:
        print(jack_p[i])
        if jack_p == 0:
            result = "JACK DIDN'T SURVIVE :'(" 
        else:
            result = 'JACK SURVIVED!!!!'
    return result 

# Logistic Regression 
clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
jack_log = clf.predict(jack_test)
jack_logVal = jack_output(jack_log)
log_score = round(clf.score(x_train,y_train)*100,2)
print('\n Per Logistic Regression ', jack_logVal,'\n')
print('Logistic Regression score: ',log_score)

# K Nearest Neighbors 
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
jack_nn = knn.predict(jack_test)
jack_knn_out = jack_output(jack_nn)
k_score = round(knn.score(x_train, y_train) * 100, 2)
print('\nPer KNN ',jack_knn_out,'\n')
print('The score for KNN is: ',k_score)
# print(y_pred)
# Support Vector Classifiers
svc = SVC(gamma='auto')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
jack_svc = svc.predict(jack_test)
jack_svc_out = jack_output(jack_svc)
svc_score = round(svc.score(x_train,y_train)*100,2)
print('Per SVC ', jack_svc_out)
print('The score for SVC is: ', svc_score)

# Decision Tree Classifiers

tree= DecisionTreeClassifier(max_depth = 100, random_state = 42)
tree.fit(x_train,y_train)
tree_pred = tree.predict(x_test)
jack_tree = tree.predict(jack_test)
jack_tree_out = jack_output(jack_tree)
tree_score = round(tree.score(x_train, y_train)*100,2)
print('\nPer Decision Tree ', jack_tree_out,'\n')
print('The score for Decision Tree classifer is: ',tree_score)

algo_scores = {'Logistic Regression': [log_score], 'KNN':[k_score], 'SVC': [svc_score], 'Decision Tree':[tree_score]}
algo_df = (pd.DataFrame(algo_scores, columns = list(algo_scores.keys())).T)
algo_perform = algo_df.reset_index()
algo_perform.columns = ['Algorithm', 'Score']
algo_perform
plt.plot(algo_perform['Algorithm'],algo_perform['Score'])
plt.title('The performance of the algorithms')
plt.grid()

