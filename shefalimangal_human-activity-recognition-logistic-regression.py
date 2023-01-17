import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
%matplotlib inline

from sklearn.preprocessing import LabelEncoder 
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
!pip3 install searchgrid
#read the file

train_data = pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/train.csv')
test_data = pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/test.csv')
train_data.head(), test_data.head()
print("Train Data: {}".format(train_data.shape))
print("Test Data: {}".format(test_data.shape))
print(train_data.info())
print()
print(test_data.info())
print("Null values in Train Data: {}".format(train_data.isnull().values.any()))
print("Null values in Test Data: {}".format(test_data.isnull().values.any()))
train_data_copy = train_data.copy()
test_data_copy = test_data.copy()
# Train Data
train_col = train_data.drop(columns=['Activity','subject'])
train_tarcol = train_data['Activity']


#Test Data

test_col = test_data.drop(columns=['Activity','subject'])
test_tarcol = test_data['Activity']
count_train_act = train_data['Activity'].value_counts()
count_test_act = test_data['Activity'].value_counts()
activities = sorted(train_tarcol.unique())
# activity count by plot

sns.set(rc={'figure.figsize':(13,6)})
fig = sns.countplot(x = "Activity" , data = train_data)
plt.xlabel("Activity")
plt.ylabel("Count")
plt.title("Activity Count")
plt.grid(True)
plt.show(fig)

Acc = 0
Gyro = 0
other = 0

for value in train_col.columns:
    if "Acc" in str(value):
        Acc += 1
    elif "Gyro" in str(value):
        Gyro += 1
    else:
        other += 1
plt.figure(figsize=(12,8))
plt.bar(['Accelerometer', 'Gyroscope', 'Others'],[Acc,Gyro,other],color=('r','g','b'))

sns.set(rc={'figure.figsize':(15,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in train_data.columns[0:10]:
    index = index + 1
    fig = sns.kdeplot(train_data[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)
sns.set(rc={'figure.figsize':(15,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in train_data.columns[10:20]:
    index = index + 1
    ax1 = sns.kdeplot(train_data[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)
sns.set(rc={'figure.figsize':(15,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in train_data.columns[20:30]:
    index = index + 1
    ax1 = sns.kdeplot(train_data[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)
sns.set(rc={'figure.figsize':(15,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in train_data.columns[30:40]:
    index = index + 1
    ax1 = sns.kdeplot(train_data[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)
sns.set(rc={'figure.figsize':(15,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in train_data.columns[40:50]:
    index = index + 1
    ax1 = sns.kdeplot(train_data[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)
sns.set(rc={'figure.figsize':(15,10)})
plt.subplot(221)
fig1 = sns.stripplot(x='Activity', y= train_data.loc[train_data['Activity']=="STANDING"].iloc[:,10], data= train_data.loc[train_data['Activity']=="STANDING"], jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig1)
plt.subplot(224)
fig2 = sns.stripplot(x='Activity', y= train_data.loc[train_data['Activity']=="STANDING"].iloc[:,11], data= train_data.loc[train_data['Activity']=="STANDING"], jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig2)
plt.subplot(223)
fig2 = sns.stripplot(x='Activity', y= train_data.loc[train_data['Activity']=="STANDING"].iloc[:,12], data= train_data.loc[train_data['Activity']=="STANDING"], jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig2)
plt.subplot(222)
fig2 = sns.stripplot(x='Activity', y= train_data.loc[train_data['Activity']=="STANDING"].iloc[:,13], data= train_data.loc[train_data['Activity']=="STANDING"], jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig2)
sns.set(rc={'figure.figsize':(15,5)})
fig1 = sns.stripplot(x='Activity', y= train_data.loc[train_data['subject']==15].iloc[:,7], data= train_data.loc[train_data['subject']==15], jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig1)
le = LabelEncoder() 

train_tarcol= le.fit_transform(train_tarcol)
#train_tarcol
test_tarcol= le.fit_transform(test_tarcol)
test_tarcol
# Train Data
#train_col = train_data.drop(columns=['Activity','subject'])
#train_tarcol = train_data['Activity']


#Test Data

#test_col = test_data.drop(columns=['Activity','subject'])
#test_tarcol = test_data['Activity']
#pca on train

#pca = PCA(n_components=2)
#pca.fit(train_col)
#train_col_pca = pca.transform(train_col)
#print("shape of PCA",train_col_pca.shape)
#print("train_col_PCA = ",train_col_pca)


pca_2 = PCA(0.95)
pca_2.fit(train_col)
pca_2.fit(test_col)
train_pca = pca_2.transform(train_col)
test_pca = pca_2.transform(test_col)
print(pca_2.n_components_)
print(test_pca)
ex_variance = np.var(train_pca,axis = 0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio) 
ex_variance = np.var(test_pca,axis = 0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio) 
#Classification models

lr = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=10000) 
lr.fit(train_pca, train_tarcol)
print(train_tarcol)
# Predicting the test set result
y_pred = lr.predict(test_pca)
y_pred
#confusion matrix
cm = confusion_matrix(test_tarcol, y_pred)
cm
print(classification_report(test_tarcol, y_pred, labels=[1, 2, 3]))

print("Accuracy",accuracy_score(test_tarcol,y_pred)*100)
rfc = RandomForestClassifier()
parameters = {'n_estimators': [10, 100, 1000], 'max_depth': [3, 6, 9], 'max_features' : ['auto', 'log2']}
model=GridSearchCV(rfc,parameters,n_jobs=-1,cv=4,scoring='accuracy')
model.fit(train_pca, train_tarcol)
from sklearn.metrics import accuracy_score
ypred=model.predict(test_pca)
accuracy=accuracy_score(test_tarcol,ypred)
accuracy

print(classification_report(test_tarcol, ypred, labels=[1, 2, 3]))
