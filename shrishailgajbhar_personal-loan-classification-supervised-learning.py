# Importing the necessary libraries

import numpy as np #importing numpy library

import pandas as pd  # To read the dataset as dataframe

import seaborn as sns # For Data Visualization 

import matplotlib.pyplot as plt # Necessary module for plotting purpose

%matplotlib inline

from sklearn.model_selection import train_test_split # For train-test split

# getting methods for confusion matrix, F1 score, Accuracy Score

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression # For logistic Regression

from sklearn.naive_bayes import GaussianNB # For Naive Bayes classifier

from sklearn.neighbors import KNeighborsClassifier # For K-NN Classifier

from sklearn.svm import SVC # For support vector machine based classifier
# Reading the data as a data frame

df_orig = pd.read_excel('../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx',sheet_name = 'Data')

df_orig.head()
# Creating copy of original dataframe 

df = df_orig.copy()

# For our convinience, let us make target attribute "Personal Loan" as the last column 

# of our dataframe.

df.drop('Personal Loan',axis=1,inplace=True)

df['Personal Loan'] = df_orig['Personal Loan']
df.head()
# Shape of dataframe

df.shape
# More info about columns

df.info()
# The column (attribute) names are:

for name in df.columns:

    print(name)
# Let us see datatypes of the column attributes

df.dtypes
# Number of unique datatypes and their value count

df.dtypes.value_counts()
# Let us check the dataset for missing values

df.isnull().sum()
# Let us see the 5-point summary of for the attributes

df.describe()
# Let us see how many negative entries are there in the Experience column

df[df['Experience']<0].Experience.count()
# Let us see how many unique negative entries are there?

df[df['Experience']<0].Experience.value_counts()
df.corr()
# Above table represented more elegently using heatmap

plt.figure(figsize=(12,10))

sns.heatmap(df.corr(),annot=True)
# Let us find the unique ages which have -1, -2 and -3 entries in the Experience column

df[df['Experience'] == -1]['Age'].value_counts()
# We will find the mean of positive experience values for above ages and use it to replace all the experience entries 

# having -1 value

l1 = df[df['Experience'] == -1]['Age'].value_counts().index.tolist()

ind_1 = df[df['Experience'] == -1]['Experience'].index.tolist()

for i in ind_1:

    df.loc[i,'Experience'] = df[(df['Age'].isin(l1)) & (df.Experience > 0)].Experience.mean()
# Let us check the values are correctly replaced.

df[df['Experience'] == -1]['Age'].value_counts()
df[df['Experience'] == -2]['Age'].value_counts()
# We will find the mean of positive experience values for above ages and use it to replace all the experience entries 

# having -2 value

l2 = df[df['Experience'] == -2]['Age'].value_counts().index.tolist()

ind_2 = df[df['Experience'] == -2]['Experience'].index.tolist()

for i in ind_2:

    df.loc[i,'Experience'] = df[(df['Age'].isin(l2)) & (df.Experience > 0)].Experience.mean()
df[df['Experience'] == -3]['Age'].value_counts()
# We will find the mean of positive experience values for above ages and use it to replace all the experience entries 

# having -3 value

l3 = df[df['Experience'] == -3]['Age'].value_counts().index.tolist()

ind_3 = df[df['Experience'] == -3]['Experience'].index.tolist()

for i in ind_3:

    df.loc[i,'Experience'] = df[(df['Age'].isin(l3)) & (df.Experience > 0)].Experience.mean()
df.Experience.describe()
# Let us see the distribution 5000 entries in target column

df['Personal Loan'].value_counts()
# The column attribute "ID" doesn't provide any significant information about a customer

# buying a personal loan hence we will skip analysis of the same.
plt.figure(figsize=(12,8))

sns.distplot(df[df['Personal Loan'] == 0]['Age'],kde=False, color='b', label='Personal Loan=0')

sns.distplot(df[df['Personal Loan'] == 1]['Age'],kde=False, color='r',label='Personal Loan=1')

plt.legend()

plt.title("Age Distribution")
age_cut = pd.cut(df['Age'],bins=[20,30,40,50,60])

pd.crosstab(age_cut,df['Personal Loan']).apply(lambda r: r/r.sum()*100, axis=1)
plt.figure(figsize=(12,8))

sns.distplot(df[df['Personal Loan'] == 0]['Experience'],kde=False, color='b', label='Personal Loan=0')

sns.distplot(df[df['Personal Loan'] == 1]['Experience'],kde=False, color='r',label='Personal Loan=1')

plt.legend()

plt.title("Experience Distribution")
exp_cut = pd.cut(df['Experience'],bins=[0,10,20,30,40,50])

pd.crosstab(exp_cut,df['Personal Loan']).apply(lambda r: r/r.sum()*100, axis=1)
plt.figure(figsize=(12,8))

sns.distplot(df[df['Personal Loan'] == 0]['Income'],kde=False, color='b', label='Personal Loan=0')

sns.distplot(df[df['Personal Loan'] == 1]['Income'],kde=False, color='r',label='Personal Loan=1')

plt.legend()

plt.title("Income Distribution")
inc_cut = pd.cut(df['Income'],bins=[0,50,100,150,200,250])

pd.crosstab(inc_cut,df['Personal Loan']).apply(lambda r: r/r.sum()*100, axis=1)
# Since it is a ordinal categorical variable, we will use countplot

sns.countplot(x='Family',hue='Personal Loan',data=df)
pd.crosstab(df['Family'],df['Personal Loan']).apply(lambda r: r/r.sum()*100, axis=1)
# Similar to ID attribute, we drop ZIP Code column analysis since it is not relevant to customer 

# buying the personal loan 
plt.figure(figsize=(12,8))

sns.distplot(df[df['Personal Loan'] == 0]['CCAvg'],kde=False, color='b', label='Personal Loan=0')

sns.distplot(df[df['Personal Loan'] == 1]['CCAvg'],kde=False, color='r',label='Personal Loan=1')

plt.legend()

plt.title("CCAvg Distribution")
ccavg_cut = pd.cut(df['CCAvg'],bins=[0,2,4,6,8,10])

pd.crosstab(ccavg_cut,df['Personal Loan']).apply(lambda r: r/r.sum()*100, axis=1)
# Since Education is an ordinal categorical variable, we will use countplot

sns.countplot(df['Education'],hue=df['Personal Loan'])
pd.crosstab(df['Education'],df['Personal Loan']).apply(lambda r: r/r.sum()*100, axis=1)
plt.figure(figsize=(8,8))

sns.distplot(df[df['Personal Loan'] == 0]['Mortgage'],kde=False, color='b', label='Personal Loan=0')

sns.distplot(df[df['Personal Loan'] == 1]['Mortgage'],kde=False, color='r',label='Personal Loan=1')

plt.legend()

plt.title("Mortgage Distribution")
mort_cut = pd.cut(df['Mortgage'],bins=[0,100,200,300,400,500,600])

pd.crosstab(mort_cut,df['Personal Loan']).apply(lambda r: r/r.sum()*100, axis=1)
plt.figure(figsize=(12,12))

plt.subplot(2,2,1)

sns.countplot(df['Securities Account'],hue=df['Personal Loan'])

plt.subplot(2,2,2)

sns.countplot(df['CD Account'],hue=df['Personal Loan'])

plt.subplot(2,2,3)

sns.countplot(df['Online'],hue=df['Personal Loan'])

plt.subplot(2,2,4)

sns.countplot(df['CreditCard'],hue=df['Personal Loan'])
pd.crosstab(df['Securities Account'],df['Personal Loan']).apply(lambda r: r/r.sum()*100, axis=1)
pd.crosstab(df['CD Account'],df['Personal Loan']).apply(lambda r: r/r.sum()*100, axis=1)
pd.crosstab(df['Online'],df['Personal Loan']).apply(lambda r: r/r.sum()*100, axis=1)
pd.crosstab(df['CreditCard'],df['Personal Loan']).apply(lambda r: r/r.sum()*100, axis=1)
sns.FacetGrid(data=df,row='Education',col='Family',hue='Personal Loan').map(plt.hist,'Income').add_legend()
sns.boxplot(x='Education',y='Income',hue='Personal Loan',data=df)
sns.boxplot(x='Family',y='Income',hue='Personal Loan',data=df)
# Before moving further, let us plot the pairplot using all attributes

sns.pairplot(df_orig,hue='Personal Loan',diag_kind='hist')
# Let us see the correlation of all independent attributes with target attribute i.e., personal loan 

df.corr().loc['Personal Loan',:].sort_values(ascending=False)
# Significance test for numerical columns

import statsmodels.api as sm

df_num = df.loc[:,['Personal Loan', 'Income', 'CCAvg', 'CD Account', 'Mortgage', 'Education', 'Family', 'Securities Account', 'Age']]

df_num['intercept'] = 1

log_mod = sm.Logit(df_num['Personal Loan'], df_num[['intercept', 'Income', 'CCAvg', 'Mortgage', 'Age']]).fit()
log_mod.summary()
# Let see the statistical significance of ordinal categorical variables Family and Education

df_ordc = df.loc[:,['Personal Loan','Family','Education']]

df_ordc['intercept'] = 1

log_mod = sm.Logit(df_ordc['Personal Loan'], df_ordc[['intercept', 'Family', 'Education']]).fit()
log_mod.summary()
df_bc = df.loc[:,['Personal Loan','CD Account','Securities Account']]

df_bc['intercept'] = 1

log_mod = sm.Logit(df_bc['Personal Loan'], df_bc[['intercept', 'CD Account','Securities Account']]).fit()
log_mod.summary()
df.head()
# Since target attribute is binary in nature, let us see count for each class

df['Personal Loan'].value_counts()
# Converting above target class distribution as dataframe

df_target = df['Personal Loan'].value_counts()

df_target = pd.DataFrame({'class':df_target.index, 'count':df_target.values})
df_target
# barplot for target column distribution

sns.barplot(x='class',y = 'count',data=df_target);
# Let us add the percentage column to the dataframe.

df_target['Percentage'] = df_target['count']/df_target['count'].sum()*100

df_target
# Let us plot the Pie Plot

plt.pie(df_target['Percentage'],labels=['No Personal Loan','Personal Loan'],autopct= '%1.1f%%');
# Train test split

# We will drop the Age, ID columns from training as well as test dataset

X = df.iloc[:,2:-1]

y = df['Personal Loan']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 7 )
# create model using constructor

LogRegModel = LogisticRegression()

# fit the model to training set

LogRegModel.fit(X_train,y_train)

# Predict the test data to get y_pred

y_pred = LogRegModel.predict(X_test)

# get accuracy of model

lr_acc_score = accuracy_score(y_test,y_pred)

# get F1-score of model

lr_f1_score = f1_score(y_test,y_pred) 

# get the confusion matrix

lr_confmat = confusion_matrix(y_test,y_pred)

# get the classification report

lr_classrep = classification_report(y_test,y_pred)



print("The accuracy of the model is {} %".format(lr_acc_score*100))

print("The f1-score of the model is {} %".format(lr_f1_score*100))

print("The confusion matrix for logistic regression is: \n",lr_confmat)

print("Detailed classification report for logistic regression is: \n",lr_classrep)
# create model using constructor

NBModel = GaussianNB()

# fit the model to training set

NBModel.fit(X_train,y_train)

# Predict the test data to get y_pred

y_pred = NBModel.predict(X_test)

# get accuracy of model

nb_acc_score = accuracy_score(y_test,y_pred)

# get F1-score of model

nb_f1_score = f1_score(y_test,y_pred) 

# get the confusion matrix

nb_confmat = confusion_matrix(y_test,y_pred)

# get the classification report

nb_classrep = classification_report(y_test,y_pred)



print("The accuracy of the model is {} %".format(nb_acc_score*100))

print("The f1-score of the model is {} %".format(nb_f1_score*100))

print("The confusion matrix for Naive Bayes classifier is: \n",nb_confmat)

print("Detailed classification report for Naive Bayes classifier is: \n",nb_classrep)
# create model using constructor

KNNModel = KNeighborsClassifier() # Calling default constructor

# fit the model to training set

KNNModel.fit(X_train,y_train)

# Predict the test data to get y_pred

y_pred = KNNModel.predict(X_test)

# get accuracy of model

knn_acc_score = accuracy_score(y_test,y_pred)

# get F1-score of model

knn_f1_score = f1_score(y_test,y_pred) 

# get the confusion matrix

knn_confmat = confusion_matrix(y_test,y_pred)

# get the classification report

knn_classrep = classification_report(y_test,y_pred)



print("The accuracy of the model is {} %".format(knn_acc_score*100))

print("The f1-score of the model is {} %".format(knn_f1_score*100))

print("The confusion matrix for K-NN classifier is: \n",knn_confmat)

print("Detailed classification report for K-NN classifier is: \n",knn_classrep)
df_comp = pd.DataFrame({'Classification Algorithm':['Logistic Regression', 'Naive Bayes', 'K-Nearest Neighbor'],'Accuracy (%)':[lr_acc_score*100,nb_acc_score*100,knn_acc_score*100],'f1-score (%)':[lr_f1_score*100,nb_f1_score*100,knn_f1_score*100]})



print("Following table shows comparison of the classification algorithms (using unscaled data and default parameters): ")

df_comp
plt.figure(figsize=(12,10))

sns.heatmap(X.corr(),annot=True)
# KNN Accuracy for neighbors = 1,3,...99

knn_acc=[]

knn_f1 = []

for i in range(1,100,2):

    print("Calculating the K-NN classifier accuracy for {} neighbors.".format(i))

    # create model using constructor

    KNNModel = KNeighborsClassifier(n_neighbors=i) # Calling default constructor

    # fit the model to training set

    KNNModel.fit(X_train,y_train)

    # Predict the test data to get y_pred

    y_pred = KNNModel.predict(X_test)

    # get accuracy of model

    knn_acc_score = accuracy_score(y_test,y_pred)

    knn_acc.append(knn_acc_score*100)

    # get F1-score of model

    knn_f1_score = f1_score(y_test,y_pred) 

    knn_f1.append(knn_f1_score*100)

df_knn = pd.DataFrame({'n_neighbors':list(range(1,100,2)), 'Accuracy':knn_acc,'f1-score':knn_f1})  
df_knn
# Let us scale train as well as test data using MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)
# Repeating all three algorithms again on the scaled data
# create model using constructor

LogRegModel = LogisticRegression()

# fit the model to training set

LogRegModel.fit(X_train_scaled,y_train)

# Predict the test data to get y_pred

y_pred = LogRegModel.predict(X_test_scaled)

# get accuracy of model

lr_acc_score = accuracy_score(y_test,y_pred)

# get F1-score of model

lr_f1_score = f1_score(y_test,y_pred) 

# get the confusion matrix

lr_confmat = confusion_matrix(y_test,y_pred)

# get the classification report

lr_classrep = classification_report(y_test,y_pred)



print("The accuracy of the model is {} %".format(lr_acc_score*100))

print("The f1-score of the model is {} %".format(lr_f1_score*100))

print("The confusion matrix for logistic regression is: \n",lr_confmat)

print("Detailed classification report for logistic regression is: \n",lr_classrep)
# create model using constructor

NBModel = GaussianNB()

# fit the model to training set

NBModel.fit(X_train_scaled,y_train)

# Predict the test data to get y_pred

y_pred = NBModel.predict(X_test_scaled)

# get accuracy of model

nb_acc_score = accuracy_score(y_test,y_pred)

# get F1-score of model

nb_f1_score = f1_score(y_test,y_pred) 

# get the confusion matrix

nb_confmat = confusion_matrix(y_test,y_pred)

# get the classification report

nb_classrep = classification_report(y_test,y_pred)



print("The accuracy of the model is {} %".format(nb_acc_score*100))

print("The f1-score of the model is {} %".format(nb_f1_score*100))

print("The confusion matrix for Naive Bayes classifier is: \n",nb_confmat)

print("Detailed classification report for Naive Bayes classifier is: \n",nb_classrep)
# create model using constructor

KNNModel = KNeighborsClassifier() # Calling default constructor

# fit the model to training set

KNNModel.fit(X_train_scaled,y_train)

# Predict the test data to get y_pred

y_pred = KNNModel.predict(X_test_scaled)

# get accuracy of model

knn_acc_score = accuracy_score(y_test,y_pred)

# get F1-score of model

knn_f1_score = f1_score(y_test,y_pred) 

# get the confusion matrix

knn_confmat = confusion_matrix(y_test,y_pred)

# get the classification report

knn_classrep = classification_report(y_test,y_pred)



print("The accuracy of the model is {} %".format(knn_acc_score*100))

print("The f1-score of the model is {} %".format(knn_f1_score*100))

print("The confusion matrix for K-NN classifier is: \n",knn_confmat)

print("Detailed classification report for K-NN classifier is: \n",knn_classrep)
# KNN Accuracy for neighbors = 1,3,...99

knn_acc=[]

knn_f1 = []

for i in range(1,100,2):

    print("Calculating the K-NN classifier accuracy for {} neighbors.".format(i))

    # create model using constructor

    KNNModel = KNeighborsClassifier(n_neighbors=i) # Calling default constructor

    # fit the model to training set

    KNNModel.fit(X_train_scaled,y_train)

    # Predict the test data to get y_pred

    y_pred = KNNModel.predict(X_test_scaled)

    # get accuracy of model

    knn_acc_score = accuracy_score(y_test,y_pred)

    knn_acc.append(knn_acc_score*100)

    # get F1-score of model

    knn_f1_score = f1_score(y_test,y_pred) 

    knn_f1.append(knn_f1_score*100)

df_knn = pd.DataFrame({'n_neighbors':list(range(1,100,2)), 'Accuracy':knn_acc,'f1-score':knn_f1})
df_knn
# create model using constructor

SVMModel = SVC() # Calling default constructor

# fit the model to training set

SVMModel.fit(X_train_scaled,y_train)

# Predict the test data to get y_pred

y_pred = SVMModel.predict(X_test_scaled)

# get accuracy of model

svm_acc_score = accuracy_score(y_test,y_pred)

# get F1-score of model

svm_f1_score = f1_score(y_test,y_pred) 

# get the confusion matrix

svm_confmat = confusion_matrix(y_test,y_pred)

# get the classification report

svm_classrep = classification_report(y_test,y_pred)



print("The accuracy of the model is {} %".format(svm_acc_score*100))

print("The f1-score of the model is {} %".format(svm_f1_score*100))

print("The confusion matrix for SVM classifier is: \n",svm_confmat)

print("Detailed classification report for SVM classifier is: \n",svm_classrep)
svm_acc=[]

svm_f1 = []

for i in range(1,1000,100):

    print("Calculating the SVM classifier accuracy for C = {}.".format(i))

    # create model using constructor

    SVMModel = SVC(C=i) # Calling default constructor

    # fit the model to training set

    SVMModel.fit(X_train_scaled,y_train)

    # Predict the test data to get y_pred

    y_pred = SVMModel.predict(X_test_scaled)

    # get accuracy of model

    svm_acc_score = accuracy_score(y_test,y_pred)

    svm_acc.append(svm_acc_score*100)

    # get F1-score of model

    svm_f1_score = f1_score(y_test,y_pred) 

    svm_f1.append(svm_f1_score*100)

df_svm = pd.DataFrame({'C':list(range(1,1000,100)), 'Accuracy':svm_acc,'f1-score':svm_f1})
df_svm