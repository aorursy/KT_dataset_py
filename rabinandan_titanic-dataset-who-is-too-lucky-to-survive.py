
####IMPORT PACKAGES ####################################

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Common Model Algorithms
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#Configure Visualization Defaults
#%matplotlib inline = show plots in Jupyter Notebook browser
%matplotlib inline
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 12,8
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output

###LOAD DATA####################
data_train = pd.read_csv('../input/train.csv')
data_test  = pd.read_csv('../input/test.csv')
dataset = [data_train, data_test]
####DESCRIBE DATA AND ANALYSE
print(data_train.describe()) ##HELP
print(data_train.info())
print(data_train.sample())

#PassengerId does never contribute into servival... hence simply drop it
for data in dataset:
    data.drop(['PassengerId'],axis=1, inplace= True)
#data_test.drop(['PassengerId'],axis=1)
print(data_test.describe())
   
####Survive column is output/target and not contain any null entries, looks good

##Pclass- is passenger class 1st class passentger got higher precidence of servival respective to 3rd one. 
##Just remain as it is -(HELP)

##NAME does not any null value, we simply extract title from this column
for data in dataset:
    data['Title'] = data['Name'].str.split(",",expand=True)[1].str.split(".",expand=True)[0]
#data_train['Title'].unique()
data_train['Title'].value_counts()
##So, Just create Miss title and put all other titles except whose count >10
stat_min = 10 #while small is arbitrary, we'll use the common minimum in 
#statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/
for data in dataset:
    title_names = (data['Title'].value_counts() < stat_min) #this will create a true false series with title name as index
    data['Title'] = data['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
#print(data_train['Title'].unique())


##Now Remove NAME 
for data in dataset:
    data.drop(['Name'],axis=1, inplace=True)
#data_train.head()    
##NOW code categorical data
#code categorical Title column and remove Title column
label = LabelEncoder()
for data in dataset: 
    data['Title_Code'] = label.fit_transform(data['Title'])
    data.drop(['Title'], axis=1, inplace=True )
data_train.head()

###Sex column is categorical text, just code it similar to Title code
for data in dataset:
    data["Sex_Code"] = label.fit_transform(data['Sex'])
    data.drop(['Sex'], axis=1, inplace=True)
data_train.head()    
##Now, Age column : Hands on - has missing data as well as continuons data
## A basic approach to fill missing continuous data is to median 
for data in dataset:
    data['Age'].fillna(data['Age'].median(), inplace = True)
data['Age'].isnull().sum()
##Lets convert continuous age into bin and then categorize it using label encoder(HELP)
for data in dataset:
    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html
    data['AgeBin'] = pd.cut(data['Age'].astype(int), 5)
    data['AgeBin_Code'] = label.fit_transform(data['AgeBin'])
    data.drop(['Age'], axis=1, inplace=True)
    data.drop(['AgeBin'], axis=1, inplace=True)
    data_train.head()
##Here no. of sibling and parents sibsp and parch column makes family_size and IsAlone
for data in dataset:
    #Discrete variables
    data['FamilySize'] = data ['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = 1 #initialize to yes/1 is alone
    data.loc[data['FamilySize'] > 1, 'IsAlone'] = 0
    data.drop(['SibSp'], axis=1, inplace=True)
    data.drop(['Parch'], axis=1, inplace=True)
data_train.head()
##Ticket is almost not contributing in survival, you may consider, i am dopping for now
for data in dataset:
    data.drop(['Ticket'], axis=1, inplace=True)
data_train.head()
##Fare may contribute to survivavl, one paying more got nearby lifebot cabin and other facility
##For fare create bin and encode as categorical------------
##Just a quick fix data_test conains 1 missing entry, fill using median
for data in dataset:
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
#Now for rest part of fare
for data in dataset:
    #Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    data['FareBin'] = pd.qcut(data['Fare'], 4)
    #print(data['Fare'].isnull().sum())
    data['FareBin_Code'] = label.fit_transform(data['FareBin'])
    data.drop(['Fare'], axis=1, inplace=True)
    data.drop(['FareBin'], axis=1, inplace=True)
data_train.head()
##ofcourse cabin contribute into survival, however 78% of total is null entries in this data, so good
## is to drop it and somehow fare may help in place of it
for data in dataset:
    data.drop(['Cabin'], axis=1, inplace=True)
data_train.head()
##For Embarked- it is categorical char value treat as Sex
##Before it just only 2 missing entries (among all 891 entries) is filled with mode
for data in dataset:
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)
    
for data in dataset:
    data['Embarked_Code'] = label.fit_transform(data['Embarked'])
    data.drop(['Embarked'], axis=1, inplace=True)
data_train.head(10)
    
for data in dataset:
    print(data.describe())
    print("--"*10)
    print(data.head(8))
    print("--"*10)
#Quick check for corelation (HELP)
from scipy.stats.stats import pearsonr

print(pearsonr(data_train[['FamilySize']],data_train[['IsAlone']] ))
print(pearsonr(data_test[['FamilySize']],data_test[['IsAlone']] ))
# separating our independent and dependent variable
X = data_train.drop(['Survived'], axis=1)
Y = data_train["Survived"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size = .33, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
data_test_scaled = sc.transform(data_test)
#Logistic regression 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report, precision_recall_curve, confusion_matrix
logreg = LogisticRegression(solver='liblinear', penalty='l1')
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
logreg_accy = round(accuracy_score(y_pred,y_test), 3)
print (logreg_accy)
print (classification_report(y_test, y_pred, labels=logreg.classes_))
print (confusion_matrix(y_pred, y_test))
#KNN approach
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(weights="distance", n_neighbors=45,  metric='minkowski', p =2 )
#n_neighbors: specifies how many neighbors will vote on the class
#weights: uniform weights indicate that all neighbors have the same weight while "distance" indicates
        # that points closest to the 
#metric and p: when distance is minkowski (the default) and p == 2 (the default), this is equivalent to the euclidean distance metric
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
knn_accy = round(accuracy_score(y_test, y_pred), 3)
print (knn_accy)
#Naive Bayes
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
gaussian_accy = round(accuracy_score(y_pred, y_test), 3)
print(gaussian_accy)
# Support Vector Machines
from sklearn.svm import SVC

svc = SVC(probability=True)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
svc_accy = round(accuracy_score(y_pred, y_test), 3)
print(svc_accy)
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dectree = DecisionTreeClassifier( max_depth=5, 
                                class_weight = 'balanced',
                                min_weight_fraction_leaf = 0.01)
dectree.fit(x_train, y_train)
y_pred = dectree.predict(x_test)
dectree_accy = round(accuracy_score(y_pred, y_test), 3)
print(dectree_accy)
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators=100,max_depth=9,min_samples_split=6, min_samples_leaf=4)
#randomforest = RandomForestClassifier(class_weight='balanced', n_jobs=-1)
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_test)
random_accy = round(accuracy_score(y_pred, y_test), 3)
print (random_accy)
# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gradient = GradientBoostingClassifier()
gradient.fit(x_train, y_train)
y_pred = gradient.predict(x_test)
gradient_accy = round(accuracy_score(y_pred, y_test), 3)
print(gradient_accy)
##Here GradientBoostingClassifer has maximum accuracy.. go with it
#PassengerId1 = data_test['PassengerId']
test_result = gradient.predict(data_test_scaled)
data_test  = pd.read_csv('../input/test.csv')
test_df = pd.DataFrame(columns = ['PassengerId', 'Survived'])
test_df['PassengerId'] = data_test['PassengerId']
test_df['Survived'] = test_result
test_df.to_csv('test_result.csv', index=False)