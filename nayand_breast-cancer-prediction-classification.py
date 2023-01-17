# Import required libraries

from sklearn import datasets
import pandas as pd
# Load breast cancer dataset from sklearn bulit in datasets
df = datasets.load_breast_cancer()
#Check our target values in categorical value
df.target_names
# check our data values i.e. datasets feature values
df.data
# Check the total number of features in dataset
len(df.feature_names)
# Check the target or dependent variable
df.target
# There are three ways to convert data into dataframe
# 1 Method
# we can directly pass feature name as column name and its value as feature value

dataframe = pd.DataFrame(df.data, columns = df.feature_names)
dataframe.head()
# 2nd Method
# we can make key value pair dictionary and pass key as feature name and value as its data value

d1 = {}
count = 0
for col in df.feature_names :
    d1[col] = df.data[:,count]
    count+=1
dd = pd.DataFrame(d1)
dd.head()
# 3rd Method
# pass individual key value pair for dictionary 
# this is more time consuming to do 
# we can use this methd when we have less number of feature or we need specific features from data

import pandas as pd
d = {
    df.feature_names[0] : df.data[:,0],
    df.feature_names[1] : df.data[:,1],
    df.feature_names[2] : df.data[:,2],
    df.feature_names[3] : df.data[:,3],
    df.feature_names[4] : df.data[:,4],
    df.feature_names[5] : df.data[:,5],
    df.feature_names[6] : df.data[:,6],
    df.feature_names[7] : df.data[:,7],
    df.feature_names[8] : df.data[:,8],
    df.feature_names[9] : df.data[:,9],
    df.feature_names[10] : df.data[:,10],
    df.feature_names[11] : df.data[:,11],
    df.feature_names[12] : df.data[:,12],
    df.feature_names[13] : df.data[:,13],
    df.feature_names[14] : df.data[:,14],
    df.feature_names[15] : df.data[:,15],
    df.feature_names[16] : df.data[:,16],
    df.feature_names[17] : df.data[:,17],
    df.feature_names[18] : df.data[:,18],
    df.feature_names[19] : df.data[:,19],
    df.feature_names[20] : df.data[:,20],
    df.feature_names[21] : df.data[:,21],
    df.feature_names[22] : df.data[:,22],
    df.feature_names[23] : df.data[:,23],
    df.feature_names[24] : df.data[:,24],
    df.feature_names[25] : df.data[:,25],
    df.feature_names[26] : df.data[:,26],
    df.feature_names[27] : df.data[:,27],
    df.feature_names[28] : df.data[:,28],
    df.feature_names[29] : df.data[:,29],
    
    'target' : df.target
    
}

data = pd.DataFrame(d)
data.head()
# chech the column or feature name

d.keys()
data.head()
# chekc the number of rows and columns

data.shape
# Check for null value 

data.isnull().sum()
# check the info of columns or features 
# number of categorical or numeric data or any other formats data

data.info()
# Split dependent and Independent variable

x = data.drop('target',axis=1)
y = data['target']
data.info()
# Do Scaling of data using StandardScaler method

from sklearn.preprocessing import StandardScaler

for col in x :
    ss = StandardScaler()
    data[col] = ss.fit_transform(data[[col]])
# As we observe that data is scaled 
# it is good to go ahead

data.head()
# Let's make prediction using machine learning algorithms

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
# split the data into training and testing 

x_trn,x_tst,y_trn,y_tst = train_test_split(x, y, test_size=0.3)
# Create a funstion which calculate the model accuracy and roc score

def model(object1,name):
    object1.fit(x_trn,y_trn)
    y_pred = object1.predict(x_tst)
    
    print('accuracy of model using',name,'=',object1.score(x_tst,y_tst))
    print('roc score of model using',name,'=',roc_auc_score(y_tst,y_pred))
    print(' ')
# check the model accuracy using every algorithm

model(LogisticRegression(),'LogisticRegression')
model(DecisionTreeClassifier(criterion='entropy'),'DecisionTreeClassifier')
model(RandomForestClassifier(n_estimators=50),'RandomForestClassifier')
model(KNeighborsClassifier(n_neighbors=3),'KNeighborsClassifier')
# Import chi2 library and selectkbest for selecting number of feature

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
# create chi2 object and fit_transform the data

chi = SelectKBest(score_func=chi2,k=20)
x = chi.fit_transform(x,y)
x.shape
# after feature selection again split the data into train and test 

x_trn,x_tst,y_trn,y_tst = train_test_split(x, y, test_size=0.3)
# function to calculate the model accuracy 

def model1(object1,name):
    
    object1.fit(x_trn,y_trn)
    y_pred = object1.predict(x_tst)
    
    print('accuracy of model using',name,'=',object1.score(x_tst,y_tst))
    print('roc score of model using',name,'=',roc_auc_score(y_tst,y_pred))
    print(' ')
# Check the model accuracy using all different algorithms

model1(LogisticRegression(),'LogisticRegression')
model1(DecisionTreeClassifier(criterion='entropy'),'DecisionTreeClassifier')
model1(RandomForestClassifier(n_estimators=50),'RandomForestClassifier')
model1(KNeighborsClassifier(n_neighbors=3),'KNeighborsClassifier')
x = data.drop('target',axis=1)
y = data['target']
# Create PCA object and decide the number of features

from sklearn.decomposition import PCA
pca = PCA(n_components=20)
x = pca.fit_transform(x)
# we merge 10 features from 30 and make it into 20

x.shape
# Split the data into test and trian

x_trn,x_tst,y_trn,y_tst = train_test_split(x,y,test_size= 0.3)

# function to make prediction

def model2(object1,name):
    
    object1.fit(x_trn,y_trn)
    y_pred = object1.predict(x_tst)
    
    print('accuracy of model using',name,'=',object1.score(x_tst,y_tst))
    print('roc score of model using',name,'=',roc_auc_score(y_tst,y_pred))
    print(' ')
# check model accurcay after PCA process

model2(LogisticRegression(),'LogisticRegression')
model2(DecisionTreeClassifier(criterion='entropy'),'DecisionTreeClassifier')
model2(RandomForestClassifier(n_estimators=50),'RandomForestClassifier')
model2(KNeighborsClassifier(n_neighbors=3),'KNeighborsClassifier')

