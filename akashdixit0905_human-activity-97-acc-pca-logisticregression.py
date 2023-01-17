import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.cm import get_cmap
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
init_notebook_mode(connected=True)
train = pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/train.csv')
train.head()
test = pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/test.csv')
test.head()
#Removing Duplicate Rows : No duplicate rows found
train = train.drop_duplicates()
print(train.shape)

#Removing Duplicate Rows : No duplicate rows found
test = test.drop_duplicates()
print(test.shape)
train.info()
test.info()
train.describe()
test.describe()
## Finding Null values in the train dataset
train.isnull().sum()
print(train.isnull().sum()[train.isnull().sum()>0])

test.isnull().sum()
print(test.isnull().sum()[test.isnull().sum()>0])
train.columns
test.columns
# Activity is the target column which has to be pridict
print("train_columns =",train['Activity'].unique())
print("test_columns =",test['Activity'].unique())
# Subject is the person ID's
print("train_subject =",train['subject'].unique())
print("test_subject =",test['subject'].unique())
#Bar plot of train activity
plt.figure(figsize=(10,10))
plt.title('Barplot of Activity')
sns.countplot(train.Activity)
plt.xticks(rotation=90)
#Bar plot of train activity
plt.figure(figsize=(10,10))
plt.title('Barplot of Activity')
sns.countplot(test.Activity)
plt.xticks(rotation=90)
# plotting the visualization in the train distribution
label_counts = train['Activity'].value_counts()
n = label_counts.shape[0]
colormap = get_cmap('viridis')
colors = [rgb2hex(colormap(col)) for col in np.arange(0, 1.01, 1/(n-1))]

data = go.Bar(x = label_counts.index,
              y = label_counts,
              marker = dict(color = colors))

layout = go.Layout(title = 'Smart Acticity Label Distribution',
                  xaxis = dict(title = 'Activity'),
                  yaxis = dict(title = 'Count'))

fig = go.Figure(data=[data], layout = layout)
iplot(fig)
# plotting the visualization in the train distribution
label_counts = test['Activity'].value_counts()
n = label_counts.shape[0]
colormap = get_cmap('viridis')
colors = [rgb2hex(colormap(col)) for col in np.arange(0, 1.01, 1/(n-1))]

data = go.Bar(x = label_counts.index,
              y = label_counts,
              marker = dict(color = colors))

layout = go.Layout(title = 'Smart Acticity Label Distribution',
                  xaxis = dict(title = 'Activity'),
                  yaxis = dict(title = 'Count'))

fig = go.Figure(data=[data], layout = layout)
iplot(fig)
print("train \n",train['Activity'].value_counts())
print("\n")
print("test :\n",test['Activity'].value_counts())
plt.figure(figsize=(10,10))
plt.pie(np.array(train['Activity'].value_counts()), labels = sorted(train['Activity'].unique()), autopct = '%0.6f')
plt.figure(figsize=(10,10))
plt.pie(np.array(test['Activity'].value_counts()), labels = sorted(test['Activity'].unique()), autopct = '%0.6f')
sns.boxplot(train['subject'])
sns.boxplot(test['subject'])
# Type of sensor used to take the data
Acc = 0
Gyro = 0
other = 0

for value in train.columns:
    if "Acc" in str(value):
        Acc += 1
    elif "Gyro" in str(value):
        Gyro += 1
    else:
        other += 1
        
plt.figure(figsize=(10,10))
plt.bar(['Accelerometer', 'Gyroscope', 'Others'],[Acc,Gyro,other],color=('r','g','b'))
pd.crosstab(train.subject, train.Activity, margins = True).style.background_gradient(cmap='autumn_r')
pd.crosstab(test.subject, test.Activity, margins = True).style.background_gradient(cmap='autumn_r')
sns.set(rc={'figure.figsize':(15,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in train.columns[0:10]:
    index = index + 1
    fig = sns.kdeplot(train[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)
sns.set(rc={'figure.figsize':(15,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in train.columns[0:10]:
    index = index + 1
    fig = sns.kdeplot(test[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)
sns.set(rc={'figure.figsize':(15,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in train.columns[10:20]:
    index = index + 1
    fig = sns.kdeplot(train[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)
sns.set(rc={'figure.figsize':(15,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in train.columns[10:20]:
    index = index + 1
    fig = sns.kdeplot(test[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)
sns.set(rc={'figure.figsize':(15,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in train.columns[20:30]:
    index = index + 1
    fig = sns.kdeplot(train[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)
sns.set(rc={'figure.figsize':(15,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in train.columns[20:30]:
    index = index + 1
    fig = sns.kdeplot(test[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)
sns.set(rc={'figure.figsize':(15,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in train.columns[30:40]:
    index = index + 1
    fig = sns.kdeplot(train[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)
sns.set(rc={'figure.figsize':(15,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in train.columns[30:40]:
    index = index + 1
    fig = sns.kdeplot(test[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)
sns.set(rc={'figure.figsize':(15,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in train.columns[40:50]:
    index = index + 1
    fig = sns.kdeplot(train[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)
sns.set(rc={'figure.figsize':(15,7)})
colours = ["maroon","coral","darkorchid","goldenrod","purple","darkgreen","darkviolet","saddlebrown","aqua","olive"]
index = -1
for i in train.columns[40:50]:
    index = index + 1
    fig = sns.kdeplot(test[i] , shade=True, color=colours[index])
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig)
sns.set(rc={'figure.figsize':(15,10)})
plt.subplot(221)
fig1 = sns.stripplot(x='Activity', y= train.loc[train['Activity']=="STANDING"].iloc[:,10], data= train.loc[train['Activity']=="STANDING"], jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig1)
plt.subplot(224)
fig2 = sns.stripplot(x='Activity', y= train.loc[train['Activity']=="STANDING"].iloc[:,11], data= train.loc[train['Activity']=="STANDING"], jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig2)
plt.subplot(223)
fig2 = sns.stripplot(x='Activity', y= train.loc[train['Activity']=="STANDING"].iloc[:,12], data= train.loc[train['Activity']=="STANDING"], jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig2)
plt.subplot(222)
fig2 = sns.stripplot(x='Activity', y= train.loc[train['Activity']=="STANDING"].iloc[:,13], data= train.loc[train['Activity']=="STANDING"], jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig2)
sns.set(rc={'figure.figsize':(15,10)})
plt.subplot(221)
fig1 = sns.stripplot(x='Activity', y= test.loc[test['Activity']=="STANDING"].iloc[:,10], data= test.loc[test['Activity']=="STANDING"], jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig1)
plt.subplot(224)
fig2 = sns.stripplot(x='Activity', y= test.loc[test['Activity']=="STANDING"].iloc[:,11], data= test.loc[test['Activity']=="STANDING"], jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig2)
plt.subplot(223)
fig2 = sns.stripplot(x='Activity', y= test.loc[test['Activity']=="STANDING"].iloc[:,12], data= test.loc[test['Activity']=="STANDING"], jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig2)
plt.subplot(222)
fig2 = sns.stripplot(x='Activity', y= test.loc[test['Activity']=="STANDING"].iloc[:,13], data= test.loc[test['Activity']=="STANDING"], jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig2)
sns.set(rc={'figure.figsize':(15,5)})
fig1 = sns.stripplot(x='Activity', y= train.loc[train['subject']==15].iloc[:,7], data= train.loc[train['subject']==15], jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig1)
sns.set(rc={'figure.figsize':(15,5)})
fig1 = sns.stripplot(x='Activity', y= test.loc[train['subject']==15].iloc[:,7], data= train.loc[train['subject']==15], jitter=True)
plt.title("Feature Distribution")
plt.grid(True)
plt.show(fig1)
train.dtypes.tail()
test.dtypes.tail()
# Dropping the subject column because it will not affect the dataset
train = train.drop('subject',axis=1)
test = test.drop('subject',axis=1)
X_train = train.iloc[:,0:len(train.columns)-1]
Y_train = train.iloc[:,-1]
X_test = train.iloc[:,0:len(train.columns)-1]
Y_test = train.iloc[:,-1]
le = LabelEncoder()
Y_train = le.fit_transform(Y_train)

le = LabelEncoder()
Y_test = le.fit_transform(Y_test)
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)
from sklearn.decomposition import PCA

pca = PCA(0.95)

pca.fit(X_train)
pca.fit(X_test)

train_x_pca = pca.transform(X_train)
test_x_pca = pca.transform(X_test)

print(pca.n_components_)
print(pca.explained_variance_)
ex_variance = np.var(train_x_pca,axis=0)
print(ex_variance)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio)
ex_variance = np.var(test_x_pca,axis=0)
print(ex_variance)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio)
# Fitting Logistic Regression To the training set 
from sklearn.linear_model import LogisticRegression   
  
classifier = LogisticRegression(penalty='l2',solver='lbfgs',class_weight='balanced', max_iter=10000,random_state = 0) 
classifier.fit(train_x_pca, Y_train)
print(Y_train)
y_pred = classifier.predict(test_x_pca)
print(test_x_pca)
# making confusion matrix between 
#  test set of Y and predicted value. 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred) 
print (cm)
from sklearn.metrics import classification_report,accuracy_score
print(classification_report(Y_test,y_pred))
print("Accuracy:",accuracy_score(Y_test, y_pred)*100)

print(y_pred)