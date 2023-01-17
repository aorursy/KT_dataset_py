import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

import os

%matplotlib inline

from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('../input/diabetes.csv')
df.head()
df.tail()
df.info()
df.isnull().values.any()
df.dtypes
df.describe(include = 'all').T
#df.hist()

hist = df.hist(figsize=(10,8))
current_palette = sns.color_palette()

sns.countplot(x = 'Outcome', data = df)

df['Outcome'].value_counts()




sns.set_style('whitegrid')

sns.boxplot(x = 'Outcome',y = 'Age',data = df)
sns.barplot(x= 'Outcome', y='BloodPressure', data = df)
plt.figure(figsize=(10,10))

sns.boxplot(x= 'Outcome', y='SkinThickness', data = df)
sns.boxplot(x = 'Outcome',y = 'Pregnancies',data = df)
plt.figure(figsize = (10,10))

sns.countplot(df['Pregnancies'])
#Diabetes pedigree function

plt.figure(figsize=(10,12))

sns.factorplot(x= 'Outcome', y='DiabetesPedigreeFunction', data = df)
#Glucose Level and outcome

plt.figure(figsize = (20,8))

sns.countplot(df['Glucose'])
df.corr()
corr = df.corr()

plt.figure(figsize = (12,12))

sns.heatmap(corr,annot=True )
sns.swarmplot(x ='SkinThickness', data = df)
df.drop(df.index[579], inplace = True)
sns.swarmplot(x ='SkinThickness', data = df)
correlations = df.corr()

correlations['Outcome'].sort_values(ascending=False)
def visualise(df):

    fig, ax = plt.subplots()

    ax.scatter(df.iloc[:,1].values, df.iloc[:,5].values)

    ax.set_title('Highly Correlated Features')

    ax.set_xlabel('Plasma glucose concentration')

    ax.set_ylabel('Body mass index')



visualise(df)
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']



for column in zero_not_accepted:

    df[column] = df[column].replace(0, np.NaN)

    mean = int(df[column].mean(skipna=True))

    df[column] =df[column].replace(np.NaN, mean)
visualise(df)
X = df.drop('Outcome', axis =1)

#X.head()

y =df['Outcome']

#y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()



model.fit(X_train, y_train.ravel())

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

# creating object of LogisticRegression class

classifier_logis = LogisticRegression(random_state=0)

# fitting the model/ training the model on training data (X_train,t_train)

classifier_logis.fit(X_train,y_train)

# predicting whether the points (people/rows) in the test set (X_test) have the liver disease or not

y_pred_logis = classifier_logis.predict(X_test)

# evaluating model performance by confusion-matrix

cm_logis = confusion_matrix(y_test,y_pred_logis)

print(cm_logis)

# accuracy-result of LogisticRegression model

accuracy_logis = accuracy_score(y_test,y_pred_logis)

print('The accuracy of LogisticRegression is : ', str(accuracy_logis*100) , '%')
df.info()
#from sklearn.metrics import confusion_matrix

#cm = confusion_matrix(y_test, y_pred)

#print(cm)
def precision_recall(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)    

    tp = cm[0,0]

    fp = cm[0,1]

    fn = cm[1,0]

    tn= cm[1,1]

    prec = tp / (tp+fp)

    rec = tp / (tp+fn)

    accr= (tp+tn)/(tp+fp+fn+tn)

    return prec, rec, accr



precision, recall, accuracy = precision_recall(y_test, y_pred)

print('Precision: %f Recall %f' % (precision, recall))

print('Accuracy: %f' % (accuracy))
#model.coef_

#model.score(X_train, X_test)
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score



clf = DecisionTreeClassifier(random_state=0)

cross_val_score(clf, X, y, cv=10)
clf.fit(X_train, y_train)

clf.predict(X_test)

clf.score(X_test, y_test)
#from sklearn import svm

#clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

#pre = clf.predict(X_test) 

#print(clf.score(X_test, y_test))

#svm_accr = clf.score(X_test, y_test)

from sklearn.svm import SVC

# creating object of SVC class

classifier_svc = SVC(kernel='rbf', random_state=0, gamma='auto')

# fitting the model/ training the model on training data (X_train,t_train)

classifier_svc.fit(X_train,y_train)

# predicting whether the points (people/rows) in the test set (X_test) have the liver disease or not

y_pred_svc = classifier_svc.predict(X_test)

# evaluating model performance by confusion-matrix

cm_svc = confusion_matrix(y_test,y_pred_svc)

print(cm_svc)

# accuracy-result of SVC model

accuracy_svc = accuracy_score(y_test,y_pred_svc)

print('The accuracy of SupportVectorClassification is : ', str(accuracy_svc*100) , '%')


from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



# Define the model: Init K-NN

classifier = KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski')



# Fit Model

classifier.fit(X_train, y_train)



# Predict the test set results

y_pred = classifier.predict(X_test)



# Evaluate Model

cm = confusion_matrix(y_test, y_pred)

print (cm)

#print(f1_score(y_test, y_pred))

print(accuracy_score(y_test, y_pred))

# STEP 5: Evaluate Model

#########################

# Making the confusion matrix

#cm = confusion_matrix(y_test, y_pred)

#print (cm)

#print(f1_score(y_test, y_pred))

#print(accuracy_score(y_test, y_pred))
#from sklearn.ensemble import RandomForestClassifier

#from sklearn.model_selection import cross_val_score



#clf1 = RandomForestClassifier(max_depth=2, random_state=0)

#clf1.fit(X_train, y_train)

#clf1.predict(X_test)

#acc_random=clf1.score(X_test, y_test)

#print(acc_random)

from sklearn.ensemble import RandomForestClassifier

# creating object of RandomForestClassifier class

classifier_rfc = RandomForestClassifier(n_estimators=250, criterion='entropy',random_state=0 )

# fitting the model/ training the model on training data (X_train,t_train)

classifier_rfc.fit(X_train,y_train)

# predicting whether the points (people/rows) in the test set (X_test) have the liver disease or not

y_pred_rfc = classifier_rfc.predict(X_test)

# evaluating model performance by confusion-matrix

cm_rfc = confusion_matrix(y_test,y_pred_rfc)

print(cm_rfc)

# accuracy-result of RandomForestClassifier model

accuracy_rfc = accuracy_score(y_test,y_pred_rfc)

print('The accuracy of RandomForestClassifier is : ', str(accuracy_rfc*100) , '%')



from sklearn.neural_network import MLPClassifier

clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

clf2.fit(X_train, y_train)

clf2.predict(X_test)

clf2.score(X_test, y_test)
#help(SVC)

#from sklearn.svm import SVC

#model_linear = SVC(kernel = "linear")

#model_linear.fit(X_train,y_train)

#pred_test_linear = model_linear.predict(X_test)
#pred_test_linear = model_linear.predict(X_test)



#xyz=np.mean(pred_test_linear==y_test)

#print(xyz)
models_comparison = [['Logistic Regression',accuracy_logis*100],

                     ['Support Vector Classfication',accuracy_svc*100], 

                     ['Random Forest Classifiaction',accuracy_rfc*100]

                    ]

models_compaison_df = pd.DataFrame(models_comparison,columns=['Model','% Accuracy'])

models_compaison_df.head()
fig = plt.figure(figsize=(20,8))

sns.set()

sns.barplot(x='Model',y='% Accuracy',data=models_compaison_df,palette='Dark2')

plt.xticks(size=18)

plt.ylabel('% Accuracy',size=14)

plt.xlabel('Model',size=14)
Pregnancies = [5]

Glucose= [121]

BloodPressure= [72]

SkinThickness= [23]

Insulin= [112]

BMI= [26.2]

DiabetesPedigreeFunction= [0.245]

Age= [30]

#Outcome= [1, 2, 3, 4,5]



# Creating a data frame using explicits lists

X1 = pd.DataFrame(columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]) 

X1["Pregnancies"] = pd.Series(Pregnancies)

X1["Glucose"] = pd.Series(Glucose)

X1["BloodPressure"] = pd.Series(BloodPressure)

X1["SkinThickness"] = pd.Series(SkinThickness)

X1["Insulin"] = pd.Series(Insulin)

X1["BMI"] = pd.Series(BMI)

X1["DiabetesPedigreeFunction"] = pd.Series(DiabetesPedigreeFunction)

X1["Age"] = pd.Series(Age)

#X["X1"] = pd.Series(x1) 

X1.iloc[:,:]

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()



model.fit(X_train, y_train.ravel())

y_pred = model.predict(X1)



print(y_pred)