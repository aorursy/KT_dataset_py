

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

from subprocess import check_output

df = pd.read_csv('../input/gesture-recognition/ASL_DATA.csv')
df.head()
df.tail()
df.sample(10)
df.dtypes
df.info()
df.describe()
df.drop('Id',axis=1,inplace=True) #dropping the Id column as it is unecessary, axis=1 specifies that it should be column wise, inplace =1 means the changes should be reflected into the dataframe
fig = df[df.Letter=='A'].plot(kind='scatter',x='Index_Pitch',y='Index_Roll',color='orange', label='A')
df[df.Letter=='B'].plot(kind='scatter',x='Thumb_Pitch',y='Thumb_Roll',color='blue', label='B',ax=fig)
df[df.Letter=='C'].plot(kind='scatter',x='Index_Pitch',y='Index_Roll',color='green', label='C', ax=fig)
df[df.Letter=='D'].plot(kind='scatter',x='Index_Pitch',y='Index_Roll',color='red', label='D', ax=fig)
df[df.Letter=='K'].plot(kind='scatter',x='Index_Pitch',y='Index_Roll',color='yellow', label='K', ax=fig)
fig.set_xlabel("Index Pitch")
fig.set_ylabel("Index Roll")
fig.set_title("Index Pitch VS Index Roll")
fig=plt.gcf()
fig.set_size_inches(10,6)
plt.show()

plt.savefig('Index Pitch  vs Index Roll.png')
plt.savefig('books_read.png')
df.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(18,12)
plt.show()
plt.savefig('Plots.jpg')
plt.savefig('Value_Spread.png')
plt.figure(figsize=(15,10))
plt.subplot(4,4,1)
sns.violinplot(x='Letter',y='Thumb_Pitch',data=df)
plt.subplot(4,4,2)
sns.violinplot(x='Letter',y='Thumb_Roll',data=df)
plt.subplot(4,4,3)
sns.violinplot(x='Letter',y='Index_Pitch',data=df)
plt.subplot(4,4,4)
sns.violinplot(x='Letter',y='Index_Roll',data=df)
plt.subplot(4,4,5)
sns.violinplot(x='Letter',y='Middle_Pitch',data=df)
plt.subplot(4,4,6)
sns.violinplot(x='Letter',y='Middle_Roll',data=df)
plt.subplot(4,4,7)
sns.violinplot(x='Letter',y='Ring_Pitch',data=df)
plt.subplot(4,4,8)
sns.violinplot(x='Letter',y='Ring_Roll',data=df)
plt.subplot(4,4,9)
sns.violinplot(x='Letter',y='Pinky_Pitch',data=df)
plt.subplot(4,4,10)
sns.violinplot(x='Letter',y='Pinky_Roll',data=df)
plt.subplot(2,1,1)
sns.boxplot(x = "Letter", y = "Thumb_Pitch", data = df)
plt.subplot(2,1,2)
sns.boxplot(x = "Letter", y = "Thumb_Roll", data = df)
plt.subplot(2,1,1)
sns.boxplot(x = "Letter", y = "Index_Pitch", data = df)
plt.subplot(2,1,2)
sns.boxplot(x = "Letter", y = "Index_Roll", data = df)

plt.subplot(2,1,1)
sns.boxplot(x = "Letter", y = "Middle_Pitch", data = df)
plt.subplot(2,1,2)
sns.boxplot(x = "Letter", y = "Middle_Roll", data = df)

plt.subplot(2,1,1)
sns.boxplot(x = "Letter", y = "Ring_Roll", data = df)
plt.subplot(2,1,2)
sns.boxplot(x = "Letter", y = "Ring_Pitch", data = df)

plt.subplot(2,1,1)
sns.boxplot(x = "Letter", y = "Pinky_Pitch", data = df)
plt.subplot(2,1,2)
sns.boxplot(x = "Letter", y = "Pinky_Roll", data = df)

# importing alll the necessary packages to use the various classification algorithms
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

plt.figure(figsize=(30,20)) 
sns.heatmap(df.corr(),annot=True) #draws  heatmap with input as the correlation matrix calculted by(iris.corr())
plt.figure()


from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

# Generate a large random dataset
rs = np.random.RandomState(33)
#d = pd.DataFrame(data=rs.normal(size=(100, 26)),
#                 columns=list(ascii_letters[26:]))

# Compute the correlation matrix
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=np.bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 20))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio
ax = sns.heatmap(corr,annot=True, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, cbar_kws={"shrink": .5})

bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
train, test = train_test_split(df, test_size = 0.3)# in this our main data is split into train and test
# the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=30%
print(train.shape)
print(test.shape)
train_X = train[['Thumb_Pitch','Thumb_Roll','Index_Pitch','Index_Roll','Middle_Pitch','Middle_Roll','Ring_Pitch','Ring_Roll','Pinky_Pitch','Pinky_Roll']]# taking the training data features
train_y=train.Letter# output of our training data
test_X= test[['Thumb_Pitch','Thumb_Roll','Index_Pitch','Index_Roll','Middle_Pitch','Middle_Roll','Ring_Pitch','Ring_Roll','Pinky_Pitch','Pinky_Roll']] # taking test data features
test_y =test.Letter   #output value of test data
from sklearn import preprocessing
mm_scaler = preprocessing.RobustScaler()
train_X = mm_scaler.fit_transform(train_X)
test_X = mm_scaler.transform(test_X)
train_X
test_X
test_y.head()
import sklearn
model = LogisticRegression()
model.fit(train_X,train_y)
prediction=model.predict(test_X)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_y)*100)

print(sklearn.metrics.confusion_matrix(test_y,prediction))

print(classification_report(test_y, prediction))
model=DecisionTreeClassifier()
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_y))
model=KNeighborsClassifier(n_neighbors=500) #this examines 3 neighbours for putting the new data into a class
model.fit(train_X,train_y)
prediction=model.predict(test_X)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_y))
a_index=list(range(1,11))
a=pd.Series()
x=[1,2,3,4,5,6,7,8,9,10]
for i in list(range(1,11)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(train_X,train_y)
    prediction=model.predict(test_X)
    a=a.append(pd.Series(metrics.accuracy_score(prediction,test_y)))
plt.plot(a_index, a)
plt.xticks(x)
from sklearn.svm import SVC
model=SVC()
model.fit(train_X, train_y)
prediction=model.predict(test_X)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(test_y,prediction))

print(classification_report(test_y, prediction))