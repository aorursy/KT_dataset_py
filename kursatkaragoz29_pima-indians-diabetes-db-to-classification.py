from mlxtend.plotting import plot_decision_regions
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#data read
data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
data.shape # sample,feature
data.info()

data.head() #first 5 data
data.describe()
data.describe().T # Changed rows and columns. (T)
data.isnull().sum()
print("Number of rows with Glucose == 0 => {}".format((data.Glucose==0).sum()))
print("Number of rows with BloodPressure == 0 => {}".format((data.BloodPressure==0).sum()))
print("Number of rows with SkinThickness == 0 => {}".format((data.SkinThickness==0).sum()))
print("Number of rows with Insulin == 0 => {}".format((data.Insulin==0).sum()))
print("Number of rows with BMI == 0 => {}".format((data.BMI==0).sum()))
#datayı bozmadan kopyaladık ve bağlılığı kaldırdık.
data_copy = data.copy(deep=True) 

#Featurelerdeki  0 valueleri Nan value yapalım.
data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]= data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)

print(data_copy.isnull().sum())
data_copy.hist(figsize=(15,15))
plt.show()
#Featurelerdeki missing valueleri mean ve median değerleri ile doldurduk.
#We filled missing values in features with mean and median values

data_copy.BloodPressure.fillna(data_copy.BloodPressure.mean(),inplace=True)
data_copy.Glucose.fillna(data_copy.Glucose.mean(),inplace=True)
data_copy.SkinThickness.fillna(data_copy.SkinThickness.median(),inplace=True)
data_copy.Insulin.fillna(data_copy.Insulin.median(),inplace=True)
data_copy.BMI.fillna(data_copy.BMI.median(),inplace=True)
data_copy.describe().T
data_copy.hist(figsize=(15,15))
plt.show()
data_copy.head()
# Data Normalize
#Yöntem 1(Medhod 1)
#from sklearn.preprocessing import MinMaxScaler
#norm = MinMaxScaler()
#x = pd.DataFrame(norm.fit_transform(data_copy.drop(['Outcome'],axis=1)),
#                 columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#       'BMI', 'DiabetesPedigreeFunction', 'Age'])
#x.head()

#Yöntem 2(Method 2)
y=  data_copy['Outcome'].values.reshape(-1,1)  #Dependent Features (Class)
x_data =data_copy.drop(['Outcome'],axis=1)     #Independent Features
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))   #normalized
x.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=29)
x_train.head()
y_train[:5]
from sklearn.neighbors import KNeighborsClassifier
train_score_list=[]
test_score_list=[]

for each in range (2,31):
    knn=KNeighborsClassifier(n_neighbors=each)
    knn.fit(x_train,y_train)
    test_score_list.append(knn.score(x_test,y_test)*100)
    train_score_list.append(knn.score(x_train,y_train)*100)

    
print("Best accuracy(test) is {:.3f} % with K = {}".format(np.max(test_score_list),test_score_list.index(np.max(test_score_list))+ 2))
print("Best accuracy(train) is {:.3f}% with K = {}".format(np.max(train_score_list),train_score_list.index(np.max(train_score_list))+2))

import plotly.graph_objs as go

arange=np.arange(2,31)
trace1=go.Scatter(
    x=arange,
    y=train_score_list,
    mode="lines + markers",
    name="Train_Score",
    marker=dict(color = 'rgba(16, 112, 2, 0.8)'),
    
)
trace2=go.Scatter(
    x=arange,
    y=test_score_list,
    mode="lines + markers",
    name="Test_Score",
    marker=dict(color = 'rgba(80, 26, 80, 0.8)'),
)
data=[trace1,trace2]
layout = dict(title = '-value VS Accuracy',
              xaxis= dict(title= 'K Value',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Score',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
# import confusion_matrix
from sklearn.metrics import confusion_matrix

k_value =test_score_list.index(np.max(test_score_list))+ 2   # K paramer
knn2 = KNeighborsClassifier(n_neighbors=k_value)
knn2.fit(x_train,y_train)
y_pred = knn2.predict(x_test)
y_true = y_test

# 0 value is negative
# 1 value is positive
cmatrix = confusion_matrix(y_true,y_pred,labels=[0,1])

f,ax = plt.subplots(figsize=(8,4))
sns.heatmap(cmatrix,annot=True,linewidths=0.5,cbar=True,linecolor="white",fmt='.0f',ax=ax)
plt.title("Model's Confusion Matrix")
plt.xlabel("y_predict (Predicted Label)")
plt.ylabel("y_true (Actual Label)")
plt.show()

#1="Pozitif"
#0="Negatif"

print('True negative = ', cmatrix[0][0])
print('False positive = ', cmatrix[0][1])
print('False negative = ', cmatrix[1][0])
print('True positive = ', cmatrix[1][1])

from sklearn.metrics import classification_report
print(classification_report(y_true,y_pred))
from sklearn import tree
dt = tree.DecisionTreeClassifier(random_state=1,criterion='entropy') # criterion default = gini
dt.fit(x_train,y_train)

print("Train Score: ",dt.score(x_train,y_train))
print("Test Score: ",dt.score(x_test,y_test))

print("\n\nDecision Tree default parameters : ",dt.get_params)

from sklearn import tree
dt = tree.DecisionTreeClassifier(random_state=1,criterion='gini')
dt.fit(x_train,y_train)

print("Train Score: ",dt.score(x_train,y_train))
print("Test Score: ",dt.score(x_test,y_test))

# Criterion gini visualization
import graphviz 
from graphviz import Source
plt.figure(figsize=(40,20))  
_ = tree.plot_tree(dt, feature_names = x.columns, 
             filled=True, fontsize=10, rounded = True)
plt.savefig('diabetes.png')
plt.show()
#confusion matrix for Criterion gini
y_pred = dt.predict(x_test)
y_true = y_test

# 0 value is negative
# 1 value is positive
cmatrix2 = confusion_matrix(y_true,y_pred,labels=[0,1])

f,ax = plt.subplots(figsize=(8,4))
sns.heatmap(cmatrix2,annot=True,linewidths=0.5,cbar=True,linecolor="white",fmt='.0f',ax=ax)
plt.title("Model's Confusion Matrix")
plt.xlabel("y_predict (Predicted Label)")
plt.ylabel("y_true (Actual Label)")
plt.show()

#1="Pozitif"
#0="Negatif"

print('True negative = ', cmatrix2[0][0])
print('False positive = ', cmatrix2[0][1])
print('False negative = ', cmatrix2[1][0])
print('True positive = ', cmatrix2[1][1])
from sklearn.metrics import classification_report
print(classification_report(y_true,y_pred))
from sklearn.ensemble import RandomForestClassifier

#*n_estimators: number of tree models to be used
#max_depth: The maximum depth of the tree. If None, 
#then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

rf = RandomForestClassifier(n_estimators=10,max_depth= 5, random_state=2, criterion="gini")
rf.fit(x_train,y_train)

print("Random Forest Score: ",rf.score(x_test,y_test))

print("Random Forest Parameters: ",rf.get_params)
# n_ estimators default = 100

pd.concat((pd.DataFrame(x_train.columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)
score_estimators=[]
for i in range(0,len(rf.estimators_)):
    estimator=rf.estimators_[i].score(x_test,y_test)
    print("{}. estimator score is {}".format(i+1,estimator))
    score_estimators.append(estimator)
    
print("\nBest Accuracy(test) is {:.3f}% with estimator number = {}".format(np.max(score_estimators)*100,score_estimators.index(np.max(score_estimators))+1))
# decision tree structure according to best estimator

best_estimator_number = score_estimators.index(np.max(score_estimators))+1
model_estimator = rf.estimators_[best_estimator_number]

plt.figure(figsize=(18,10))  
_ = tree.plot_tree(model_estimator, feature_names = x.columns, 
             filled=True, fontsize=10, rounded = True)
plt.savefig('diabetes.png')
plt.show()
import timeit
start = timeit.default_timer()
train_score_list=[]
test_score_list=[]

for i in range(1,101):
    rf2 = RandomForestClassifier(n_estimators=i,max_depth= 5, random_state=2, criterion="gini")
    rf2.fit(x_train,y_train)
    train_score_list.append(rf2.score(x_test,y_test))
    test_score_list.append(rf2.score(x_train,y_train))
    
plt.figure(figsize=(12,5))
p1=sns.lineplot(x=range(1,101),y=train_score_list,color='red',label="Train Scores")
p1=sns.lineplot(x=range(1,101),y=test_score_list,color='lime',label="Test Scores")
plt.legend()
plt.title('N-Estimator vs Accuracy')
plt.xlabel("Estimators")
plt.ylabel("Accuracy")
plt.show()

stop = timeit.default_timer()
print('Run Time: ', stop - start) 
print("\nBest Accuracy(test): {:.3f} with n_estimators: {} ".format(np.max(test_score_list)*100,test_score_list.index(np.max(test_score_list))+1))
print("\nBest Accuracy(train): {:.3f} with n_estimators: {} ".format(np.max(train_score_list)*100,train_score_list.index(np.max(train_score_list))+1))
best_n_estimators_parameter = test_score_list.index(np.max(test_score_list))+1
rf3 = RandomForestClassifier(n_estimators=best_n_estimators_parameter,max_depth= 5, random_state=2, criterion="gini")
rf3.fit(x_train,y_train)
y_true = y_test
y_pred = rf3.predict(x_test)
# 0 value is negative
# 1 value is positive
from sklearn.metrics import confusion_matrix
cmatrix3 = confusion_matrix(y_true,y_pred,labels=[0,1])

f,ax = plt.subplots(figsize=(8,4))
sns.heatmap(cmatrix3,annot=True,linewidths=0.5,cbar=True,linecolor="white",fmt='.0f',ax=ax)
plt.title("Model's Confusion Matrix")
plt.xlabel("y_predict (Predicted Label)")
plt.ylabel("y_true (Actual Label)")
plt.show()

#1="Pozitif"
#0="Negatif"

print('True negative = ', cmatrix3[0][0])
print('False positive = ', cmatrix3[0][1])
print('False negative = ', cmatrix3[1][0])
print('True positive = ', cmatrix3[1][1])
from sklearn.svm import SVC
c_range=np.arange(0.1,1.5,0.1)

for i in c_range:
    svm_linear = SVC(kernel="linear",C=i,random_state=29)
    svm_linear.fit(x_train,y_train)
    print("accuracy of svm(linear): {:.4f} with C(cost) parameter: {:.2f}".format(svm_linear.score(x_test,y_test),i))



#confusion matrix for Criterion gini
y_pred = svm_linear.predict(x_test)
y_true = y_test

# 0 value is negative
# 1 value is positive
cmatrix_linear = confusion_matrix(y_true,y_pred,labels=[0,1])

f,ax = plt.subplots(figsize=(8,4))
sns.heatmap(cmatrix_linear,annot=True,linewidths=0.5,cbar=True,linecolor="white",fmt='.0f',ax=ax)
plt.title("Model's Confusion Matrix")
plt.xlabel("y_predict (Predicted Label)")
plt.ylabel("y_true (Actual Label)")
plt.show()

#1="Pozitif"
#0="Negatif"

print('True negative = ', cmatrix_linear[0][0])
print('False positive = ', cmatrix_linear[0][1])
print('False negative = ', cmatrix_linear[1][0])
print('True positive = ', cmatrix_linear[1][1])





from sklearn.metrics import classification_report
print(classification_report(y_true,y_pred))
from sklearn.datasets import make_moons
# create spiral dataset
X, Y = make_moons(n_samples=1000, noise=0.15, random_state=42)
plt.figure(figsize=(10,5))
ax = plt.axes()
ax.scatter(X[:,0],X[:,1],c=Y);
plt.xlabel("1");
plt.ylabel("2");
X=(X - np.min(X)) / (np.max(X) - np.min(X))
from sklearn.svm import SVC
svm_poly = SVC(kernel="poly",C=1.0,degree=3,random_state=29)
svm_poly.fit(X,Y)

plt.figure(figsize=(15,5))
plot_decision_regions(X,Y,svm_poly)
plt.show()
from sklearn.svm import SVC
svm_polynomial = SVC(kernel="poly",C=1.0,degree=2.0,gamma='scale')
svm_polynomial.fit(x_train,y_train)
print("Accuracy of svm(poly)%: ",svm_polynomial.score(x_test,y_test)*100)
print("\n\nOther Default Parameters: ",svm_polynomial.get_params())

#confusion matrix for Criterion gini
y_pred = svm_polynomial.predict(x_test)
y_true = y_test

# 0 value is negative
# 1 value is positive
cmatrix_polynomial = confusion_matrix(y_true,y_pred,labels=[0,1])

f,ax = plt.subplots(figsize=(8,4))
sns.heatmap(cmatrix_polynomial,annot=True,linewidths=0.5,cbar=True,linecolor="white",fmt='.0f',ax=ax)
plt.title("Model's Confusion Matrix")
plt.xlabel("y_predict (Predicted Label)")
plt.ylabel("y_true (Actual Label)")
plt.show()

#1="Pozitif"
#0="Negatif"

print('True negative = ', cmatrix_polynomial[0][0])
print('False positive = ', cmatrix_polynomial[0][1])
print('False negative = ', cmatrix_polynomial[1][0])
print('True positive = ', cmatrix_polynomial[1][1])
print("\n\n")





from sklearn.metrics import classification_report
print(classification_report(y_true,y_pred))
from sklearn.datasets.samples_generator import make_circles
X,Y = make_circles(90, factor=0.2, noise=0.1) 
#noise = standard deviation of Gaussian noise added in data. 
#factor = scale factor between the two circles
plt.figure(figsize=(8,5))
plt.scatter(X[:,0],X[:,1], c=Y, s=50, cmap='seismic')
plt.show()
X=(X - np.min(X)) / (np.max(X) - np.min(X))
from sklearn.svm import SVC
import time

gamma_values=[0.1,1,10,100]
for i in gamma_values:
    start=time.time()
    svm_gausian = SVC(kernel="rbf",C=1.0,degree=3,gamma=i,random_state=29)
    svm_gausian.fit(X,Y)
    finish=time.time()

    plt.figure(figsize=(15,5))
    plt.title("Gamma = {}, Time Cost= {:.4f}".format(i,(finish-start)))
    plt.xlabel("x feature")
    plt.ylabel("y feature")
    plot_decision_regions(X,Y,svm_gausian)
    plt.show()
from sklearn.svm import SVC
svm_rbf = SVC(kernel="rbf",C=0.5,gamma="scale")
svm_rbf.fit(x_train,y_train)
print("Accuracy of svm(poly): %",svm_rbf.score(x_test,y_test)*100)
#confusion matrix for Criterion gini
y_pred = svm_rbf.predict(x_test)
y_true = y_test

# 0 value is negative
# 1 value is positive
cmatrix_rbf = confusion_matrix(y_true,y_pred,labels=[0,1])

f,ax = plt.subplots(figsize=(8,4))
sns.heatmap(cmatrix_rbf,annot=True,linewidths=0.5,cbar=True,linecolor="white",fmt='.0f',ax=ax)
plt.title("Model's Confusion Matrix")
plt.xlabel("y_predict (Predicted Label)")
plt.ylabel("y_true (Actual Label)")
plt.show()

#1="Pozitif"
#0="Negatif"

print('True negative = ', cmatrix_rbf[0][0])
print('False positive = ', cmatrix_rbf[0][1])
print('False negative = ', cmatrix_rbf[1][0])
print('True positive = ', cmatrix_rbf[1][1])
print("\n\n")




from sklearn.metrics import classification_report
print(classification_report(y_true,y_pred))
from sklearn.svm import SVC

svm_sigmoid = SVC(kernel="sigmoid",gamma=0.5)
svm_sigmoid.fit(x_train,y_train)
print("Accuracy of svm(sigmoid): %",svm_sigmoid.score(x_test,y_test)*100)
y_pred = svm_sigmoid.predict(x_test)
y_true = y_test

# 0 value is negative
# 1 value is positive
cmatrix_sigmoid  = confusion_matrix(y_true,y_pred,labels=[0,1])

f,ax = plt.subplots(figsize=(8,4))
sns.heatmap(cmatrix_sigmoid,annot=True,linewidths=0.5,cbar=True,linecolor="white",fmt='.0f',ax=ax)
plt.title("Model's Confusion Matrix")
plt.xlabel("y_predict (Predicted Label)")
plt.ylabel("y_true (Actual Label)")
plt.show()

#1="Pozitif"
#0="Negatif"

print('True negative = ', cmatrix_sigmoid[0][0])
print('False positive = ', cmatrix_sigmoid[0][1])
print('False negative = ', cmatrix_sigmoid[1][0])
print('True positive = ', cmatrix_sigmoid[1][1])
print("\n\n")




from sklearn.metrics import classification_report
print(classification_report(y_true,y_pred))
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)
print("print accuracy of naive bayes: % ",nb.score(x_test, y_test)*100)
#confusion matrix for Criterion gini
y_pred = nb.predict(x_test)
y_true = y_test

# 0 value is negative
# 1 value is positive
cmatrix_nb = confusion_matrix(y_true,y_pred,labels=[0,1])

f,ax = plt.subplots(figsize=(8,4))
sns.heatmap(cmatrix_nb,annot=True,linewidths=0.5,cbar=True,linecolor="white",fmt='.0f',ax=ax)
plt.title("Model's Confusion Matrix")
plt.xlabel("y_predict (Predicted Label)")
plt.ylabel("y_true (Actual Label)")
plt.show()

#1="Pozitif"
#0="Negatif"

print('True negative = ', cmatrix_nb[0][0])
print('False positive = ', cmatrix_nb[0][1])
print('False negative = ', cmatrix_nb[1][0])
print('True positive = ', cmatrix_nb[1][1])
print("\n\n")




from sklearn.metrics import classification_report
print(classification_report(y_true,y_pred))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print("Test Accuracy : %{}".format(lr.score(x_test,y_test)*100))
from sklearn import linear_model
train_score_list=[]
test_score_list=[]
arange=np.arange(1,20)
for each in range (1,len(arange)+1):
    logreg = linear_model.LogisticRegression(random_state = 29,max_iter= each)
    logreg.fit(x_train,y_train)
    train_score_list.append(logreg.score(x_train,y_train))
    test_score_list.append(logreg.score(x_test,y_test))    
print("Best accuracy(test) is {} with itereration n. = {}".format(np.max(test_score_list),test_score_list.index(np.max(test_score_list))+ 1))
print("Best accuracy(train) is {} with iteration n. = {}".format(np.max(train_score_list),train_score_list.index(np.max(train_score_list))+ 1))
best_max_iter_parameter = test_score_list.index(np.max(test_score_list))+ 1
logreg2 = linear_model.LogisticRegression(random_state = 29,max_iter= best_max_iter_parameter)
logreg2.fit(x_train,y_train)
y_pred = logreg2.predict(x_test)
y_true = y_test

print("Score: ",logreg2.score(x_train,y_train))
# 0 value is negative
# 1 value is positive
cmatrix_logreg = confusion_matrix(y_true,y_pred,labels=[0,1])

f,ax = plt.subplots(figsize=(8,4))
sns.heatmap(cmatrix_logreg,annot=True,linewidths=0.5,cbar=True,linecolor="white",fmt='.0f',ax=ax)
plt.title("Model's Confusion Matrix")
plt.xlabel("y_predict (Predicted Label)")
plt.ylabel("y_true (Actual Label)")
plt.show()

#1="Pozitif"
#0="Negatif"

print('True negative = ', cmatrix_logreg[0][0])
print('False positive = ', cmatrix_logreg[0][1])
print('False negative = ', cmatrix_logreg[1][0])
print('True positive = ', cmatrix_logreg[1][1])

print("\n\n")



from sklearn.metrics import classification_report
print(classification_report(y_true,y_pred))
#import library for parameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
x.shape #ALL DATASET drop('Outcome')
y.shape #Outcome
knn = KNeighborsClassifier(68)
knn.fit(x_train,y_train)


results = cross_validate(knn, x, y, cv=5,scoring='accuracy',return_train_score=True)
print("Cv:5, Test Score: {}".format(results['test_score']))
print("Cv:5, Train Score: {}".format(results['train_score']))

accuracy = cross_val_score(knn,x,y,cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(),accuracy.std() * 2))

f1 = cross_val_score(knn,x,y,cv=5,scoring='f1')
print('F1 Score : ',f1)

mse = cross_val_score(knn,x,y,cv=5,scoring='neg_mean_squared_error')
print('Negative Mean Squared Error: ', mse)


print("Cv = 5, recall = ",cross_val_score(knn, x, y, scoring='recall'))
print("Cv = 5, precision = ",cross_val_score(knn, x, y, scoring='precision'))


grid = {'n_neighbors': np.arange(1,100)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv=5) # cross validation (cv) default = 5
knn_cv.fit(x_train,y_train)# Fit

# Print hyperparameter

print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 
print("Best score: {}".format(knn_cv.best_score_))
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='gini',max_depth=3)
dtree.fit(x,y)
dtree_result = cross_validate(dtree,x, y, cv=5,scoring='accuracy',return_train_score=True)
print("Cv:5, Test Score: {}".format(dtree_result['test_score']))
print("Cv:5, Train Score: {}".format(dtree_result['train_score']))


accuracy = cross_val_score(dtree,x,y,cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(),accuracy.std() * 2))

f1 = cross_val_score(dtree,x,y,cv=5,scoring='f1')
print('F1 Score : ',f1)

mse = cross_val_score(dtree,x,y,cv=5,scoring='neg_mean_squared_error')
print('Negative Mean Squared Error: ', mse)
print("Cv = 5, recall = ",cross_val_score(knn, x, y, scoring='recall'))
print("Cv = 5, precision = ",cross_val_score(knn, x, y, scoring='precision'))
from sklearn.tree import DecisionTreeClassifier

#create a dictionary of all values we want to test
param_grid = {'criterion':['gini', 'entropy'],'max_depth': list(range(3,15))}
# decision tree model
dtree_model=DecisionTreeClassifier(random_state=29)
#use gridsearch to test all values
dtree_gscv = GridSearchCV(dtree_model, param_grid)# cv default = 5 = 5-fold cv
 #fit model to data
dtree_gscv.fit(x_train, y_train)
print("Tuned hyperparameters Criterion: {}, Max_depth: {}".format
      (dtree_gscv.best_params_['criterion'],dtree_gscv.best_params_['max_depth'])) 
print("Best score: {}".format(dtree_gscv.best_score_))
rftree = RandomForestClassifier(criterion='entropy',max_depth=4,n_estimators=100,max_features='auto',random_state=1)
rftree.fit(x,y)
rftree_result = cross_validate(rftree,x, y, cv=5,scoring='accuracy',return_train_score=True)
print("Cv:5, Test Score: {}".format(rftree_result['test_score']))
print("Cv:5, Train Score: {}".format(rftree_result['train_score']))


accuracy = cross_val_score(rftree,x,y,cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(),accuracy.std() * 2))

f1 = cross_val_score(rftree,x,y,cv=5,scoring='f1')
print('F1 Score : ',f1)

mse = cross_val_score(rftree,x,y,cv=5,scoring='neg_mean_squared_error')
print('Negative Mean Squared Error: ', mse)
print("Cv = 5, recall = ",cross_val_score(rftree, x, y, scoring='recall'))
print("Cv = 5, precision = ",cross_val_score(rftree, x, y, scoring='precision'))
from sklearn.ensemble import RandomForestClassifier
param_grid = { 
    'n_estimators': [10,50,100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7],
    'criterion' :['gini', 'entropy']
}
rfc =RandomForestClassifier(random_state=29)
rfc_gscv=GridSearchCV(estimator=rfc, param_grid=param_grid) # cv default = 5
rfc_gscv.fit(x_train,y_train)

print("Tuned best parameters for random forest: ",rfc_gscv.best_params_ ) 
print("Best score: {}".format(rfc_gscv.best_score_))

                       
# OUTPUT
#Tuned best parameters for random forest:  {'criterion': 'entropy', 'max_depth': 4, 'max_features': 'auto', 'n_estimators': 100}
#Best score: 0.780356524749048                     
                     
lin_svc = SVC(kernel='linear',C=10)
lin_svc.fit(x,y)
lin_svc_result = cross_validate(lin_svc,x, y, cv=5,scoring='accuracy',return_train_score=True)
print("Cv:5, Test Score: {}".format(lin_svc_result['test_score']))
print("Cv:5, Train Score: {}".format(lin_svc_result['train_score']))


accuracy = cross_val_score(lin_svc,x,y,cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(),accuracy.std() * 2))

f1 = cross_val_score(lin_svc,x,y,cv=5,scoring='f1')
print('F1 Score : ',f1)

mse = cross_val_score(lin_svc,x,y,cv=5,scoring='neg_mean_squared_error')
print('Negative Mean Squared Error: ', mse)
print("Cv = 5, recall = ",cross_val_score(lin_svc, x, y, scoring='recall'))
print("Cv = 5, precision = ",cross_val_score(lin_svc, x, y, scoring='precision'))
from sklearn.svm import SVC
param_grid = { 'C' : [0.001, 0.01, 0.1, 1, 10, 20]}
svm_linear = SVC(kernel="linear")
svm_gscv0 = GridSearchCV(svm_linear,param_grid=param_grid)
svm_gscv0.fit(x_train,y_train)

print("Tuned best parameters for kernel linear svm: ",svm_gscv0.best_params_ ) 
print("Best score: {}".format(svm_gscv0.best_score_))

rbf_svc = SVC(kernel='rbf',C=100,gamma=0.1)
rbf_svc.fit(x,y)
rbf_svc_result = cross_validate(rbf_svc,x, y, cv=5,scoring='accuracy',return_train_score=True)
print("Cv:5, Test Score: {}".format(rbf_svc_result['test_score']))
print("Cv:5, Train Score: {}".format(rbf_svc_result['train_score']))


accuracy = cross_val_score(rbf_svc,x,y,cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(),accuracy.std() * 2))

f1 = cross_val_score(rbf_svc,x,y,cv=5,scoring='f1')
print('F1 Score : ',f1)

mse = cross_val_score(rbf_svc,x,y,cv=5,scoring='neg_mean_squared_error')
print('Negative Mean Squared Error: ', mse)
print("Cv = 5, recall = ",cross_val_score(rbf_svc, x, y, scoring='recall'))
print("Cv = 5, precision = ",cross_val_score(rbf_svc, x, y, scoring='precision'))
from sklearn.svm import SVC
param_grid = { 
   'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6)
}
svm_rbf = SVC(kernel="rbf")
svm_gscv = GridSearchCV(svm_rbf,param_grid) #cv default = 5
svm_gscv.fit(x_train,y_train)

print("Tuned best parameters for kernel rbf svm: ",svm_gscv.best_params_ ) 
print("Best score: {}".format(svm_gscv.best_score_))
poly_svc = SVC(kernel='poly',C=1,degree=3)
poly_svc.fit(x,y)
poly_svc_result = cross_validate(poly_svc,x, y, cv=5,scoring='accuracy',return_train_score=True)
print("Cv:5, Test Score: {}".format(poly_svc_result['test_score']))
print("Cv:5, Train Score: {}".format(poly_svc_result['train_score']))


accuracy = cross_val_score(poly_svc,x,y,cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(),accuracy.std() * 2))

f1 = cross_val_score(poly_svc,x,y,cv=5,scoring='f1')
print('F1 Score : ',f1)

mse = cross_val_score(poly_svc,x,y,cv=5,scoring='neg_mean_squared_error')
print('Negative Mean Squared Error: ', mse)
print("Cv = 5, recall = ",cross_val_score(poly_svc, x, y, scoring='recall'))
print("Cv = 5, precision = ",cross_val_score(poly_svc, x, y, scoring='precision'))
### from sklearn.svm import SVC
param_grid = { 
    'C' : [0.001, 0.01, 0.1, 1, 10, 20],
    'degree':list(range(1,5))
}
svm_poly = SVC(kernel="poly")
svm_gscv3 = GridSearchCV(svm_poly,param_grid) #cv default = 5
svm_gscv3.fit(x_train,y_train)

print("Tuned best parameters for kernel poly svm: ",svm_gscv3.best_params_ ) 
print("Best score: {}".format(svm_gscv3.best_score_))
sigmoid_svc = SVC(kernel='sigmoid',gamma=0.5)
sigmoid_svc.fit(x,y)
sigmoid_svc_result = cross_validate(sigmoid_svc,x, y, cv=5,scoring='accuracy',return_train_score=True)
print("Cv:5, Test Score: {}".format(sigmoid_svc_result['test_score']))
print("Cv:5, Train Score: {}".format(sigmoid_svc_result['train_score']))


accuracy = cross_val_score(sigmoid_svc,x,y,cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (accuracy.mean(),accuracy.std() * 2))

f1 = cross_val_score(sigmoid_svc,x,y,cv=5,scoring='f1')
print('F1 Score : ',f1)

mse = cross_val_score(sigmoid_svc,x,y,cv=5,scoring='neg_mean_squared_error')
print('Negative Mean Squared Error: ', mse)
print("Cv = 5, recall = ",cross_val_score(sigmoid_svc, x, y, scoring='recall'))
print("Cv = 5, precision = ",cross_val_score(sigmoid_svc, x, y, scoring='precision'))
from sklearn.svm import SVC
param_grid = { 
   'gamma': np.arange(0,1,0.1),
    'coef0': np.logspace(-3, 2, 10)
}
svm_sigmoid = SVC(kernel="sigmoid")
svm_gscv4 = GridSearchCV(svm_sigmoid,param_grid) #cv default = 5
svm_gscv4.fit(x_train,y_train)

print("Tuned best parameters for kernel poly svm: ",svm_gscv4.best_params_ ) 
print("Best score: {}".format(svm_gscv4.best_score_))

#Output
#Tuned best parameters for kernel poly svm:  {'coef0': 0.003593813663804626, 'gamma': 0.5}
#Best score: 0.7747317410868813

classifier = ['Knn','Decision Tree(Gini)','Random Forest','Svm(Linear)','Svm(Rbf)','Svm(Poly)','Svm(Sigmoid)']
accuracy_score=[77.7,77.1,78,77.1,76.9,77.1,77.5]
conclusionf =dict(Model=classifier,Accuracy_Score=accuracy_score)
conclusionf = pd.DataFrame(conclusionf)
conclusionf
arange=np.arange(2,31)
trace1=go.Scatter(
    x=conclusionf.Model,
    y=conclusionf.Accuracy_Score,
    mode="lines + markers",
    name="Accuracy_Score-For CV 5",
    marker=dict(color = 'rgba(16, 0, 2, 0.8)'),
    
)
data=trace1
layout = dict(title = '-Model VS Accuracy',
              xaxis= dict(title= 'Model',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Accuracy_Score',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)