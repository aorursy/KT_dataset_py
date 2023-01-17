import numpy as np # linear algebra 
import matplotlib.pyplot as plt
import scipy.linalg as la

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statistics

        
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.model_selection import KFold 

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import svm
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from scipy.io import loadmat
data = loadmat('/kaggle/input/s-mats/S2.mat')
data
mdata = data['training_data']
print(type(mdata))
mdata.shape
mdata[0,1]
training_data = data["training_data"]
# c=[]
# for i in range (0,3):
#     class1 = np.array(training_data[0][i])
#     c.append(class1)
# pc = pd.DataFrame(class1)
class1 = np.array(training_data[0][0])
class2 = np.array(training_data[0][1])
class3 = np.array(training_data[0][2])
class4 = np.array(training_data[0][3])
len(c)
m1,n1,r1 = class1.shape
m2,n2,r2 = class2.shape
m3,n3,r3 = class3.shape
m4,n4,r4 = class4.shape

out_arr1 = np.column_stack((np.repeat(np.arange(m1),n1),class1.reshape(m1*n1,-1)))
out_df1 = pd.DataFrame(out_arr1)
out_df1.drop(columns=[0])

out_arr2 = np.column_stack((np.repeat(np.arange(m2),n2),class2.reshape(m2*n2,-1)))
out_df2 = pd.DataFrame(out_arr2)
out_df2.drop(columns=[0])


out_arr3 = np.column_stack((np.repeat(np.arange(m3),n3),class3.reshape(m3*n3,-1)))
out_df3 = pd.DataFrame(out_arr3)
out_df3.drop(columns=[0])


out_arr4 = np.column_stack((np.repeat(np.arange(m4),n4),class1.reshape(m4*n4,-1)))
out_df4 = pd.DataFrame(out_arr4)
out_df4.drop(columns=[0])
out_df1.shape, out_df2.shape, out_df3.shape, out_df4.shape

# type(out_df1)
training_set = pd.concat([out_df1, out_df2, out_df3, out_df4], ignore_index=True)

training_set.drop(columns=[0])

training_set.shape
training_set.drop(training_set.columns[0], axis=1,inplace=True)
training_set.shape
testing_data = data["test_data"]
class_test_1=[]
for i in range (0,399):
    class_test = np.array(testing_data[0][i])
    class_test_1.append(class_test)
len(class_test_1)
class_test_1
testing_set = pd.DataFrame(class_test_1)
testing_set.shape
# import numpy as np
# from scipy.io import loadmat  # this is the SciPy module that loads mat-files
# import matplotlib.pyplot as plt
# from datetime import datetime, date, time
# import pandas as pd

# # mat = loadmat('/kaggle/input/mat-files/A01T.mat')  # load mat-file
# # mdata = mat['data']
# # type(mat)
# # mat
# # mdtype = mdata.dtype
# # mdtype
# # # print(mdtype.names)
# # mdtype.size
# # # ndata = {n: mdata[n][0, 0] for n in mdtype.names}
# # # ndata

# # import scipy.io
# # import mne.io.read_raw_edf
# # import pandas as pd
# # raw = mne.io.read_raw_edf('/kaggle/input/gdf-file/A01T.gdf', stim_channel=None)
# # # mat = scipy.io.loadmat('/kaggle/input/mat-files/A01T.mat')
# # # mat = {k:v for k, v in mat.items() if k[0] != '_'}
# # # data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.iteritems()})
# # # data.to_csv("/kaggle/output/example.csv")
# # import pandas as pd #pandas lib  
# url="/kaggle/input/trydata/data try.csv" # data 
# # load dataset into Pandas DataFrame

# df=pd.read_csv(url) # 4 features 5 target  reduce 4 features into 3
# # df=df.drop(['Id'])

# target = ['6/18/2020']
# # Separating out the target
# # y=df.loc[:,['target']].values
# df.shape
# xt=df.iloc[:,2:152].values
# yt=df.iloc[:,152:].values
# xt, yt
# # features=['Lat', 'Long'] # naming features in data array
# xafg=df.loc[1]
# # import pandas as pd #pandas lib  
# url="/kaggle/input/irisdata/Iris.csv" # data 
# # load dataset into Pandas DataFrame

# df=pd.read_csv(url, names=['Id', 'sepal length','sepal width','petal length','petal width','target']) # 4 features 5 target  reduce 4 features into 3
# # df=df.drop(['Id'])
# features=['sepal length', 'sepal width', 'petal length', 'petal width'] # naming features in data array
# target = ['target']
# # Separating out the target
# y=df.loc[:,['target']].values
# df
# df.describe()
# missing_values = ["n/a", "na", "--", '', ' ', 'NULL']
# # df = pd.read_csv('/kaggle/input/generic-food.csv', names = ['Food name', 'Scientific name', 'Group', 'Sub Group'], na_values = missing_values)
# df.tail(10)
# # print df['ST_NUM']
# dt = df.isnull()
# print(dt.shape)


# # # Detecting numbers or string
# # cnt=0
# # for row in df['OWN_OCCUPIED']:
# #     try:
# #         int(row)
# #         df.loc[cnt, 'OWN_OCCUPIED']=np.nan
# #     except ValueError:
# #         pass
# #     cnt+=1

# # Total missing values for each feature
# print(df.isnull().values.any())
# print('Values missing in features = ', df.isnull().sum())
# print('Total Missing values are = ', df.isnull().sum().sum())
# # print that class which has missing values
# # print(datain['Scientific name'])

# # # Replace using median 
# # #method 1: mode
# # print(datain['Scientific name'].mode())

# # mode1 = df['Scientific name'].mode()
# # data = mode1.iloc[0]
# # df.fillna(data, inplace=True)
# # Remove to top row
# # df = df.drop([0])
# df

# print('Values missing in features:\n', df.isnull().sum())
# df.isnull()
# x2=xt
# y2=yt
x2=training_set.loc[:,:].values #y series into arry

y2=testing_set.loc[:,:].values #data frame into 2 array

x2.shape, y2.shape

# y2=y2[:,0]
x2.shape, y2.shape
type(x2), type(y2)
x2.shape, y2.shape
# x2[0,144]
from sklearn import preprocessing
from sklearn import utils
x2 = preprocessing.scale(x2)
y2 = preprocessing.scale(y2)
x2 = x2*10
y2 = y2*10
# lab_enc = preprocessing.LabelEncoder()
# y_train = lab_enc.fit_transform(x2)
# print(y_train_encoded)
# X_train = lab_enc.fit_transform(X_train[:,0:1])
# print(y_train_encoded)
# y_test = lab_enc.fit_transform(y_test[:,0:1])
# print(y_train_encoded)
# X_test = lab_enc.fit_transform(X_test[:,0])
# # print(y_train_encoded)
# # print(utils.multiclass.type_of_target(y_train[:,0]))
# # print(utils.multiclass.type_of_target(y_train[:,0].astype('int')))
# # print(utils.multiclass.type_of_target(y_train_encoded))
x2 = x2[:y2.shape[0],:]
x2.shape
x2 = x2.astype('int')
y2 = y2.astype('int')
kf = KFold(n_splits=9)
kf.get_n_splits(x2)

for train_index, test_index in kf.split(x2):
    X_train, X_test = x2[train_index], x2[test_index]
for ytrain_index, ytest_index in kf.split(y2):
    y_train, y_test = y2[ytrain_index], y2[ytest_index]
X_train.shape, y_train.shape, X_test.shape, y_test.shape

# X_train, X_test, y_train, y_test = train_test_split(x2, y2, test_size= 0.33)
for i in range (0, x2.shape[1]):
    print(x2[0,i])
    plt.scatter(range (0, x2[:,0].size), x2[:,i])
    
x2
# for i in range (144, 149):
#     print(xt[i,0], xt[i,1], x2[0,i])
#     plt.bar(xt[i,1], x2[:,i])
# #     plt.scatter(x2[:,i], range (0, x2[:,0].size))

## Question 1 = PCA
# X_train1, X_test1, y_train1, y_test1 = train_test_split(x2, y2[:,0], test_size= 0.33)
X_train1, X_test1, y_train1, y_test1 = X_train, X_test, y_train, y_test
xt_count = []
for x in range (x2[:,0].size):
    xt_count.append(x)

pca=PCA(n_components=1)
# x=pca.fit_transform(X_train1, y_train1)
pca_t=pca.fit_transform(x2, y2)

# plt.scatter(xt_count,X_train1[:,0], s=20, marker='o')
# plt.scatter(xt_count,X_train1[:,1], s=40, marker='+')
# plt.scatter(xt_count,X_train1[:,2], s=60, marker='*')
# plt.scatter(xt_count,X_train1[:,3], s=80, marker='^')
# plt.scatter(xt_count,x, s=100, marker='o')
plt.scatter(range (0, x2[:,0].size),x2[:,0], s=20, marker='o')
plt.scatter(range (0, x2[:,1].size),x2[:,1], s=40, marker='+')
plt.scatter(range (0, x2[:,2].size),x2[:,2], s=60, marker='*')
plt.scatter(range (0, x2[:,3].size),x2[:,3], s=80, marker='^')
plt.scatter(range (0, pca_t.size),pca_t, s=100, marker='o')
# X_train, X_test, y_train, y_test = X_train*10, X_test*10, y_train*10, y_test*10
# X_train, X_test, y_train, y_test = X_train.astype('int'), X_test.astype('int'), y_train.astype('int'), y_test.astype('int')
## Question 2 = LDA
# X_train1, X_test1, y_train1, y_test1 = train_test_split(x2, y2, test_size= 0.33)
X_train1, X_test1, y_train1, y_test1 = X_train, X_test, y_train, y_test
clf=LDA(n_components=1)
x11=clf.fit_transform(X_train1, y_train1[:,0])

plt.scatter(range (0, X_train1[:,0].size),X_train1[:,0], s=20, marker='*')
plt.scatter(range (0, X_train1[:,0].size),X_train1[:,1], s=40, marker='+')
plt.scatter(range (0, X_train1[:,0].size),X_train1[:,2], s=60, marker='o')
plt.scatter(range (0, X_train1[:,0].size),X_train1[:,3], s=80, marker='^')
plt.scatter(range (0, x11.size),x11, s=100, marker='*')
# plt.legend()

# print(accuracy_score(y_test1, clf.predict(X_test1))*100)

clf=LDA(n_components=1)
lda_t=clf.fit_transform(x2, y2[:,1])

plt.scatter(range (0, x2[:,0].size),x2[:,0], s=20, marker='*')
plt.scatter(range (0, x2[:,1].size),x2[:,1], s=40, marker='+')
plt.scatter(range (0, x2[:,2].size),x2[:,2], s=60, marker='o')
plt.scatter(range (0, x2[:,3].size),x2[:,3], s=80, marker='^')
plt.scatter(range (0, lda_t.size),lda_t, s=100, marker='*')
# plt.legend()

print(accuracy_score(y2[:,1], clf.predict(x2))*100)
# plt.scatter(range (0, y_test1.size),y_test1)
# plt.scatter(range (0, clf.predict(X_test1).size),clf.predict(X_test1))
# plt.scatter(X_test1[:,0], y_test1, s=100, marker='^')
# plt.scatter(X_test1[:,0],clf.predict(X_test1), s=40, marker='o')
## Question 3 = Simple Regression
# # Splitting the Dataset 
X_train1, X_test1, y_train1, y_test1 = train_test_split(x2[:,2:3], x2[:,0], test_size= 0.33)

# # Instantiating LinearRegression() Model
lr = LinearRegression()

# # Training/Fitting the Model
lr.fit(X_train1, y_train1)

# # Making Predictions
lr.predict(X_test1)
lr_pred = lr.predict(X_test1)
# print(accuracy_score(y_test1, lr_pred)*100)
# plt.scatter(lr_count, y_test1)
plt.scatter(X_test1[:,0], y_test1)

#pred
# plt.scatter(lr_count,lr_pred)
plt.scatter(X_test1[:,0],lr_pred)

plt.plot(np.sort(X_test1[:,0]), np.sort(lr_pred), 'b')
## Question 4 = Multiple Linear Regression
# # Splitting the Dataset 
X_train1, X_test1, y_train1, y_test1 = train_test_split(x2[:,1:], x2[:,0], test_size= 0.33)

# # Instantiating LinearRegression() Model
lr4 = LinearRegression()

# # Training/Fitting the Model
lr4.fit(X_train1, y_train1)

# # Making Predictions
lr4.predict(X_test1)
lr4_pred = lr4.predict(X_test1)
# print(accuracy_score(y_test1, lr4_pred)*100)
# lr4_pred.shape, y_test1.shape
# plt.scatter(lr_count, y_test1)
plt.scatter(X_test1[:,0], y_test1)

#pred
# plt.scatter(lr_count,lr_pred)
plt.scatter(X_test1[:,0],lr4_pred)
X_test1.shape
plt.plot(np.sort(X_test1[:,0]), np.sort(lr4_pred), 'b')

from sklearn.preprocessing import PolynomialFeatures
# Question 5

poly = PolynomialFeatures(degree=2)

xp = poly.fit_transform(X_train1)
# predict_ = poly.fit_transform(X_test1)

llf=LinearRegression()
llf.fit(xp, y_train1)
# ppred=llf.predict(predict_)

# print clf.predict(predict_)
ppred=llf.predict(xp)

plt.plot(np.sort(X_train1[:,1]), np.sort(ppred), 'b')
y1d = y_train[:,0]
y1d.shape
## Quesioin 6 = SVM
svm_model=make_pipeline(StandardScaler(), SVC(gamma='auto'))
svm_model.fit(X_train, y_train[:,0])
svm_pred=svm_model.predict(X_test)
print(accuracy_score(y_test[:,0], svm_pred)*100)
plt.scatter(X_test[:,0], y_test[:,0], s=100, marker='^')
plt.scatter(X_test[:,0],svm_pred, s=40, marker='o')
## Question 7 = Decission Classifier
dtc_model=DecisionTreeClassifier(criterion='gini',max_leaf_nodes=3) #max_leaf_nodes = None , then we have high accuracy
dtc_model.fit(X_train,y_train[:,0])
dtc_pred=dtc_model.predict(X_test)

# print(pred)

plt.scatter(X_test[:,0], y_test[:,0], s=100, marker='^')
plt.scatter(X_test[:,0],dtc_pred, s=40, marker='o')
print(accuracy_score(y_test[:,0], dtc_pred)*100)
## Question 8 = Random Forest
rfc_model=RandomForestClassifier(n_estimators=100, criterion='gini')

rfc_model.fit(X_train,y_train[:,0])
rfc_pred=rfc_model.predict(X_test)
plt.scatter(X_test[:,0], y_test[:,0], s=100, marker='^')
plt.scatter(X_test[:,0],rfc_pred, s=40, marker='o')
print(accuracy_score(y_test[:,0], rfc_pred)*100)
## Question 9 = Logistic Regression
# X_train, X_test, y_train, y_test=train_test_split(x2, y2, test_size=0.3, random_state=42)
lr1_count = []
for x in range (X_test1[:,0].size):
    lr1_count.append(x)
# lr1_count

# y1d = y2.reshape(y2.shape[0],1)
y1d = y_train[:,0]
y1d.shape

lr1=LogisticRegression()
# 
lr1.fit(X_train, y_train[:,0])


lr1.predict(X_test)
lr1_pred=lr1.predict(X_test)

# print(pred)
# plt.scatter(lr1_count,y_test1)
# #pred
# plt.scatter(lr1_count,lr1_pred)
plt.scatter(X_test[:,0], y_test[:,0], s=100, marker='^')
plt.scatter(X_test[:,0],lr1_pred, s=40, marker='o')
print(accuracy_score(y_test[:,0], lr1_pred)*100)
## Question 10 = KNN
knn_model=KNeighborsClassifier(n_neighbors=5, algorithm='brute', p=2) # p=2 means euclidean dist


# Train the model using the training sets
knn_model.fit(X_train,y_train[:,0])

#Predict Output
knn_pred=knn_model.predict(X_test)
print(accuracy_score(y_test[:,0], knn_pred)*100)
plt.scatter(X_test[:,0], y_test[:,0], s=100, marker='^')
plt.scatter(X_test[:,0],knn_pred, s=40, marker='o')
## Question 11 = Naive Bayes
naive_model = GaussianNB()

# Train the model using the training sets
naive_model.fit(X_train,y_train[:,0])

#Predict Output
naive_pred= naive_model.predict(X_test)
print(accuracy_score(y_test[:,0], naive_pred)*100)
plt.scatter(X_test[:,0], y_test[:,0], s=100, marker='^')
plt.scatter(X_test[:,0],naive_pred, s=40, marker='o')
## Question 12 = KMeans
cluster=KMeans(n_clusters =3,random_state=1, max_iter=10, algorithm='auto')
cluster.fit(X_train)
labels=cluster.labels_
labels.size
print(labels)

# labels.tostring()
# for i in range (134):

#     if labels[i] == '1':
#         print(i)
#         labels[i]='Iris-setosa'
#     if labels[i] == '0':
#         labels[i]='Iris-versicolor'
#     if labels[i] == '2':
#         labels[i]='Iris-virginica'
# labels
# # print(accuracy_score(y_test, labels)*100)
# print(accuracy_score(y_test1, lr1_pred)*100, accuracy_score(y_test, knn_pred)*100, accuracy_score(y_test, naive_pred)*100,
#       accuracy_score(y_test, svm_pred)*100, accuracy_score(y_test, dtc_pred)*100, accuracy_score(y_test, rfc_pred)*100)
# ## Q 13
# # import pandas as pd #pandas lib  
# url="/kaggle/input/irisdata/Iris.csv" # data 
# # load dataset into Pandas DataFrame

# df=pd.read_csv(url, names=['Id', 'sepal length','sepal width','petal length','petal width','target']) # 4 features 5 target  reduce 4 features into 3
# # df=df.drop(['Id'])
# features=['sepal length', 'sepal width', 'petal length', 'petal width'] # naming features in data array
# df=df.drop(0)
# # Separating out the target
# y=df.loc[:,['target']].values
# x=df.loc[:,features].values
# x
# Compare Algorithms
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# load dataset
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# dataframe = pandas.read_csv(url, names=names)
# array = dataframe.values
# X = array[:,0:8]
# Y = array[:,8]
# prepare configuration for cross validation test harness
seed = 7
# prepare models
# models = ['svm_pred','dtc_pred','rfc_pred','lr1_pred','knn_pred','naive_pred']
models = []
models.append(('LDA', LinearDiscriminantAnalysis()));
models.append(('SVM', SVC()));
models.append(('DTC', DecisionTreeClassifier()));
models.append(('RFC', RandomForestClassifier()));
models.append(('LR', LogisticRegression()));
models.append(('KNN', KNeighborsClassifier()));
models.append(('NB', GaussianNB()));
# evaluate each model in turn
results = []
names=[]
scoring='accuracy'
acc=[]
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed);
    cv_results = model_selection.cross_val_score(model, x2, y2, cv=kfold, scoring=scoring);
    results.append(cv_results);
    names.append(name);
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std());
    acc.append(msg)
    print(msg)

results
# # boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
acc