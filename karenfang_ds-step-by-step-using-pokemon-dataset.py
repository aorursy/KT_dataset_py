# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# read the data
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
# check the info of the data
data.info()

# therea are 800 rows with 12 columns. 
# since there are many features, see the correlation between features by using heatmap plot
plt.figure(figsize=(14,8))
sns.heatmap(data.corr(),annot=True,linewidths=.5)
# for a quick view, we can see the pairplot too. 
# only use this if the data is small, otherwise it takes a long time to run
sns.pairplot(data)

# we can see some linear regression relationship between some features, such as attack vs. hp, attack vs. defense etc. 
# check the head of the data and see what they are
data.head(10)

# NaN noticed. Need to deal with missing number. 
# check the info 
data.info()
# check data describe
data.describe()
# check describe for categorical data
data.describe(include = ['O'])

# name is unique
# There are 18 unique type 1, most common is Water
# There are 18 unique type 2, most common is flying
# check missing value
data.isnull().any()

# Name and Type 2 have missing value
# list all the column name for easy using later
data.columns
# Line plot - just as an example since non of the data has time associated
plt.figure(figsize=(14,8))
data['Speed'].plot(kind='line',label='Speed',linestyle = ':',color = 'g',grid=True,alpha=0.5)
data['Attack'].plot(kind='line',label='Attack',linestyle = '-',color = 'r',grid=True,alpha=0.5)

plt.legend(loc = 'upper right')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Line Plot Example')
# scatter plot - see how attack associate with defense
plt.figure(figsize=(14,8))
plt.scatter(x = 'Attack', y = 'Defense',data=data,alpha = 0.5, marker ='o',c = 'r')
plt.xlabel('Attack')
plt.ylabel('Defense')
plt.title('Attack vs. Defense Scatter Plot')
# histogram - Speed
plt.figure(figsize=(14,8))
data.hist('Speed',bins=50, figsize=(14,8))
# clf() = cleans it up again you can start a fresh
data.hist('Speed',bins = 50)
plt.clf()
# We cannot see plot due to clf()
#create dictionary and look its keys and values
dictionary = { 'spain': 'madrid','usa': 'vegas'}
print(dictionary.keys())
print(dictionary.values())
# dictionary can be updated. 
# you can do update, add, remove, check and clear the whole dictionary

# update city name
dictionary['usa'] = 'San Francisco' 
print(dictionary) # you can see city for USA is updated to San Francisco from Vegas

# add new entry
dictionary['france'] = 'Paris'
print(dictionary)

# remove entry
del dictionary['spain']
print(dictionary)

#check if an entry is in a dictionary
print('france' in dictionary)

# clear the whole dictionary
dictionary.clear()
print(dictionary)
# read dataset
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
series = data['Defense']  #if we just grab a column, only use one [], we get a series
print(type(series))

data_frame = data[['Defense']] # if we use double [], we get a data frame
print(type(data_frame))
# comparison
print('Is 3 equal to 2? ', 3==2)
print('Is 3 greater than 2? ', 3>2)
print('Is 3 less than to 2? ', 3<2)
print('Is 3 not equal to 2? ', 3!=2)
# logical operation
print(True and True)
print(True or False)
# filtering 
data[data['Attack']>180] # there are 2 pokemons which attack is greater than 180
# filtering 2
data[(data['Attack']>180) & (data['Defense'] > 100)] 

# there is only one with attack > 180 and defense > 100
# while loop
i = 0

while i !=5: 
    print('i is ', i)
    i += 1
print('i is equal to 5')
# for loop 
list = [1,2,3,4,5]
for i in list:
    print('i is ',i)

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(list):
    print(index," : ",value)
print('')  
dictionary = {'spain':'madrid','france':'paris'}
for key,value in dictionary.items():
    print(key," : ",value)
# for pandas, we can achieve index and value: 
# use function of iterrows to iterate over DataFrame rows as (index, Series) pairs.
for index,value in data[['Attack']][0:1].iterrows():
    print(index," : ",value)
# How can we learn what is built in scope
import builtins
dir(builtins)
# nested function: 
def square():
    def add():
        x = 2
        y = 3
        z = x + y
        return z
    return add()**2

print(square())
square = lambda x: x**2
print(square(2))

tot = lambda x,y,z: x+y+z
print(tot(1,2,3))
name = 'karen'
it = iter(name)
print(next(it)) # print next iteration

print(*it) # print remaining iteration
num = [1,2,3]
num2 = [i + 1 for i in num]
print(num2)
num = [5,10,15]
num2 = [i**2 if i ==10 else i-5 if i < 7 else i +5 for i in num]
num2
# lets return pokemon csv and make one more list comprehension example
# lets classify pokemons whether they have high or low speed. Our threshold is average speed.

threshold = data['Speed'].mean()
data['speed_level'] = ['high' if i > threshold else 'low' for i in data['Speed']]

# check out first 10 rows
data.loc[:10,['speed_level','Speed']]
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
data.head()
data.tail()
data.columns
data.shape
data.info()
# value_counts()
data['Type 1'].value_counts(dropna =False)  # if there are nan values that also be counted)
data.describe()
# Box plots: visualize basic statistics like outliers, min/max or quantiles
data.boxplot(column = 'Attack',by = 'Legendary')
# first make a smaller dataset
data_new = data.head()
data_new
# let's melt
melted = pd.melt(frame = data_new, id_vars = 'Name', value_vars = ['Attack', 'Defense'])
melted
melted.pivot(index = 'Name',columns = 'variable',values ='value')
data1 = data.head()
data2 = data.tail()

con_data = pd.concat([data1, data2], axis=0,ignore_index = True)
con_data
data1 = data['Attack'].head()
data2 = data['Defense'].head()
con_data_col = pd.concat([data1,data2],axis = 1)
con_data_col

data.dtypes
# let's convert objects to categorical and int to float
data["Type 1"] = data['Type 1'].astype('category')
data['Speed'] = data['Speed'].astype('float')

data.dtypes
# we can check missing value using info()
data.info()

# we can see total rows is 800. Name has 799, one is missing. 
# Type 2 has 414 non-null, so many are missing 
data['Type 2'].value_counts(dropna=False)

# we can see that NaN=386
# method 1: drop NaN
data1 = data # make a copy of data to data1
data1['Type 2'].dropna(inplace = True)

# check if NaN is dropped - there is no more NaN 
data1['Type 2'].value_counts(dropna = False)
assert  data['Type 2'].notnull().all() # returns nothing because we drop nan values
data['Type 2'].fillna('empty',inplace = True)
assert  data['Type 2'].notnull().all() # returns nothing because we drop nan values
# data frames from dictionary
data_dict = {'country':['Spain','France'], 'population':['11','12']}
df = pd.DataFrame(data_dict)
df
# add a new column
df['capital']=['marid','paris']
df
# Broadcasting
df["income"] = 0 #Broadcasting entire column
df
# ploting several features in a plot graph
data1 = data[['Attack','Defense','Speed']]
data1.plot()

# very confusing 
# instead, we can do subplot
data1.plot(subplots=True)
# scatter plot
data1.plot(kind ='scatter',x = 'Attack',y='Defense')
# histogram
data1.plot(kind ='hist',y='Attack', bins = 50,range =(0,250),grid = False, normed = True)
data1['Attack'].plot.hist(bins = 50, normed=True, range=(0,250))
plt.legend()
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

fig, axes = plt.subplots(nrows = 2, ncols=1)

data1['Defense'].plot.hist(bins = 50, range = (0,250), normed = True, ax = axes[0],legend = 'Defense')
data1['Defense'].plot.hist(bins = 50, range = (0,250), normed = True, ax = axes[1], cumulative = True, legend = 'Defense')
time_list = ['2018-03-01', '2018-03-02']
print(type(time_list[1])) # you can see it's a string 

# we can convert string to datetime 
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
data2 = data.head()
data2

# add 5 date
time_list = ['2018-02-15', '2018-02-20', '2018-02-28', '2018-03-01', '2018-03-02']
datetime_object = pd.to_datetime(time_list)
data2['date'] = datetime_object
data2
# we can make date as index 
data2 = data2.set_index('date')
data2
# Now we can select according to our date index
print(data2.loc["2018-02-15"])
print(data2.loc["2018-02-28":"2018-03-02"])
data2
# summary by month using resampling
data2.resample(rule = 'M').mean()
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
data.head()
# we can use [] 
data['HP'][0]
# we can use columns
data.HP[0]
# use loc
data.loc[0,['HP']]
# select certain columns
data[['Attack','HP']].head()
# Difference between selecting columns: series and dataframes
print(type(data["HP"]))     # series
print(type(data[["HP"]]))   # data frames
# slicing and indexing series
data.loc[1:10,'HP': 'Defense']
# Reverse slicing 
data.loc[10:1:-1,"HP":"Defense"] 
# From something to end
data.loc[1:10,"Speed":] 
data[data['HP']>200]
# combining filters 
data[(data['HP']>200) & (data['Attack']>5)]
# filter column based on others
data.HP[data['Speed']<10]
def divident(item):
    return item/2

data.HP.apply(divident).head()
# for simple function, we can use lambda

data.HP.apply(lambda x: x/2).head()
# Defining column using other columns
data["total_power"] = data.Attack + data.Defense
data.head()
# current index name 
print(data.index.name)
# lets change it
data.index.name = "index_name"
data.head()
# Overwrite index
# if we want to modify index we need to change all of them.
data.head()

# first copy of our data to data3 then change index 
data3 = data.copy()

# lets make index start from 100. It is not remarkable change but it is just example
data3.index = range(100,900,1)
data3.head()
# We can make one of the column as index. I actually did it at the beginning of manipulating data frames with pandas section
# It was like this
# data= data.set_index("#")
# also you can use data.index = data["#"]
# lets read data frame one more time to start from beginning
data = pd.read_csv('../input/pokemon-challenge/pokemon.csv')
data.head()
# As you can see there is index. However we want to set one or more column to be index
# Setting index : type 1 is outer type 2 is inner index
data1 = data.set_index(["Type 1","Type 2"]) 
data1.head(100)

dic = {"treatment": ['A','A','B','B'], "gender": ['F','M','F','M'],'response': [10,45,9,10],'age':[15,4,74,28]}
df = pd.DataFrame(dic)
df
# piviting 
df.pivot(index = 'treatment',columns = 'gender',values = 'response')
# take a look at df
df
df1 = df.set_index(keys = ['treatment','gender'])
df1
# unstack 
# level determine indexes
df1.unstack(level = 0)

# treatement is the first level, so when we unstack the dataset, treatment got unstacked
df1.unstack(level = 1)

# now we can also unstack the second order or indexing, gender
# change order of indexing 
df2 = df1.swaplevel(0,1)
df2

# now gender is the first order and treatment is the second
df.pivot(index="treatment",columns = "gender",values="response")
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df
# according to treatment take means of other features
df.groupby('treatment').mean()
# there are other methods like sum, std,max or min
# we can only choose one of the feature
df.groupby('treatment').age.mean()
# Or we can choose multiple features
df.groupby('treatment')[['age','response']].mean()
df.info()
# as you can see gender is object
# However if we use groupby, we can convert it categorical data. 
# Because categorical data uses less memory, speed up operations like groupby
df["gender"] = df["gender"].astype("category")
df["treatment"] = df["treatment"].astype("category")
df.info()
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# read csv (comma separated value) into data
data = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
print(plt.style.available) # look at available plot styles
plt.style.use('ggplot')
# check the head of the data and get a taste of what it looks like
data.head()
data.info()

# check out how big is the data and see if there are any NaN
# there are 310 entries, 
# features are float type
# target is object type, such as string

data.describe()

# In order to visualize data, values should be closer each other. So we can check this by checking out describe()
# when data is not too large, we can quickly plot pairplot to see the relationship between each feature
sns.pairplot(data, hue = 'class')
# do some value counts: 
data['class'].value_counts()

sns.countplot(x = 'class',data=data)
# KNN
# import the library and define KNN object 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

# set x and y
x = data.loc[:,data.columns !='class']
y = data['class']

# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)

# fit the data
knn.fit(x_train, y_train)

# predict the data 
prediction = knn.predict(x_test)

# check the accuracy score
print("With KNN(N = 3), the accuracy is: ",knn.score(x_test,y_test))
# Model complexity 
neig = np.arange(1,25) # set an arange from 1 to 24

# create empty list to hold train and test accuracy
train_accuracy=[]
test_accuracy=[]

# loop over differen neig values and put the result to train and test accuracy list
for i, k in enumerate(neig):
    # k from 1 to 24
    knn = KNeighborsClassifier(n_neighbors=k)
    
    #fit
    knn.fit(x_train, y_train)
    
    # train accuracy
    train_accuracy.append(knn.score(x_train,y_train))
    
    # test accuracy
    test_accuracy.append(knn.score(x_test,y_test))

# make the plot
plt.figure(figsize=(13,8))
plt.plot(neig,test_accuracy, label = 'Testing Accuracy')
plt.plot(neig,train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Score of Accuracy')
plt.title('k value vs. Accuracy')

# keep the plot
plt.show()

# find the best score and # of k
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy), 1+test_accuracy.index(np.max(test_accuracy))))
# create data1 that includes pelvic_incidence that is feature and sacral_slope that is target variable

# pick out the abnormal one and see how pelvic_incidence and sacral_slope were correlative
data1 = data[data['class']=='Abnormal']
x = data1['pelvic_incidence'].reshape(-1,1)
y = data1['sacral_slope'].reshape(-1,1)

# plot the scatter plot
plt.figure(figsize=(8,8))
plt.scatter(x, y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
# linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

# fit
reg.fit(x,y)

#predict 
predicted = reg.predict(x)

print('R^2 scores: ',reg.score(x,y))

# plot the regression line and scatter plot
plt.plot(x,predicted, color = 'black', linewidth = 3)
plt.scatter(x,y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
# CV

from sklearn.model_selection import cross_val_score
reg = LinearRegression()
k = 5
cv_result = cross_val_score(reg, x,y,cv = k) # use R^2 score
print('CV scores: ', cv_result)
print('CV scores average: ', cv_result.mean())

# acceptable range is 0.39
# Ridge
from sklearn.linear_model import Ridge
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 2, test_size = 0.3)
ridge = Ridge(alpha = 0.1, normalize = True)
ridge.fit(x_train, y_train)
ridge_predict = ridge.predict(x_test)
print('Ridge socre: ', ridge.score(x_test,y_test))
# Lasso
from sklearn.linear_model import Lasso

# get some x variable 
x = data1[['pelvic_incidence','pelvic_tilt numeric','lumbar_lordosis_angle','pelvic_radius']]
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=3,test_size = 0.3)
lasso = Lasso(alpha=0.1,normalize = True)
lasso.fit(x_train,y_train)
lasso_predict = lasso.predict(x_test)
print('Lasso score: ', lasso.score(x_test,y_test))
print('Lasso coefficients: ', lasso.coef_)

# check the coefficients ,and you can see 'pelvic_incidence','pelvic_tilt numeric' are important features, but the rest are not
# Confusion matrix with random forest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
x = data.loc[:,data.columns != 'class']
y = data[['class']]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)
rf = RandomForestClassifier(random_state = 4)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n',cm)
print('Classification Report: \n', classification_report(y_test, y_pred))

# visualize with seaborn library
sns.heatmap(cm,annot=True, fmt = 'd')
# ROC Curve with logistic regression 
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# abnormal = 1, normal = 0
data['class_binary'] = [1 if i =='Abnormal' else 0 for i in data['class']]

x=data.loc[:,'pelvic_incidence':'degree_spondylolisthesis']
y=data['class_binary']

x_trian, x_test, y_train, y_test = train_test_split(x,y,test_size =0.3, random_state = 42)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

y_pred_prob = logreg.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test,y_pred_prob)

# make the ROC plot
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')

# grid search cross validation with 1 hyperparameter
from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors': np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv = 3)
knn_cv.fit(x,y)

# print hyperparameter
print('Tuned hyperparameter k: {}'.format(knn_cv.best_params_))
print('Best score: {}'.format(knn_cv.best_score_))
# grid search cross validation with 2 hyperparameter
# 1. hyperparameter is C:logistic regression regularization parameter
# 2. penalty l1 or l2
# Hyperparameter grid
param_grid = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 12)
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv = 3)
logreg_cv.fit(x_train, y_train)

print('Tuned hyperparameters: {}'.format(logreg_cv.best_params_))
print('Best Accuracy: {}'.format(logreg_cv.best_score_))
# load data
data = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

# get dummies
df = pd.get_dummies(data)
df.head(10)
# drop one of the dummy varaible to avoide duplicate
df.drop('class_Normal',axis = 1, inplace = True)
df.head(10)
#  SVM, pre-process and pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
steps = [('scalar', StandardScaler()),
         ('SVM', SVC())]
pipeline = Pipeline(steps)
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 1)
cv = GridSearchCV(pipeline,param_grid=parameters,cv=3)
cv.fit(x_train,y_train)

y_pred = cv.predict(x_test)

print("Accuracy: {}".format(cv.score(x_test, y_test)))
print("Tuned Model Parameters: {}".format(cv.best_params_))
data = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'])
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
# KMean clustering
data2 = data[['pelvic_radius','degree_spondylolisthesis']]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(data2)
labels = kmeans.predict(data2)
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = labels)
# cross tabulation table
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)
# inertia
inertia_list = np.empty(8)
for i in range(1,8):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(data2)
    inertia_list[i] = kmeans.inertia_

plt.plot(range(0,8),inertia_list,'-o')
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
data = pd.read_csv('../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
data3 = data.drop('class', axis = 1)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scalar = StandardScaler()
kmeans = KMeans(n_clusters = 2)
pipe = make_pipeline(scalar,kmeans)
pipe.fit(data3)
labels = pipe.predict(data3)
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)
from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(data3.loc[200:220,:], method = 'single')
dendrogram(merg, leaf_rotation=90, leaf_font_size = 6)
plt.show()
from sklearn.manifold import TSNE
model = TSNE(learning_rate=100)
transformed = model.fit_transform(data2)
x = transformed[:,0]
y = transformed[:,1]

color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]
plt.scatter(x,y,c = color_list )
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()
# PCA
from sklearn.decomposition import PCA
model = PCA()
model.fit(data3)
transformed = model.transform(data3)
print('Principle components: ',model.components_)
# PCA variance
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler,pca)
pipeline.fit(data3)

plt.bar(range(pca.n_components_), pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.show()
# apply PCA
pca = PCA(n_components = 2)
pca.fit(data3)
transformed = pca.transform(data3)
x = transformed[:,0]
y = transformed[:,1]
plt.scatter(x,y,c = color_list)
plt.show()
