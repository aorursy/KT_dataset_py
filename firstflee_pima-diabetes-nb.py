import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
diabetes= pd.read_csv('../input/diabetes.csv')
diabetes.describe(include = "all").transpose()
diabetes.info()
diabetes.columns = ['Pregnancies','Glucose','BloodPressure',

                     'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
diabetes.columns
diabetes.hist(figsize = (20,20))
#We find that there are '0' values in the dataset which can be missing data.  Before replacing this missing values 

#copy the dataframe and then make the required changes



pimadiabetes = diabetes.copy(deep = True)

pimadiabetes.head()
pimadiabetes.tail()
pimadiabetes[pimadiabetes['BloodPressure'] == 0].shape[0]
pimadiabetes[pimadiabetes['BloodPressure'] == 0].index.tolist()

pimadiabetes[pimadiabetes['BloodPressure'] == 0].groupby('Outcome')['BloodPressure'].count()
# Replacing '0' values with null values for column Glucose

pimadiabetes['BloodPressure'].replace(0,np.NaN,inplace = True)

# The median value calculation of Glucose grouped by outcome 



pimadiabetes_bp = pimadiabetes[pimadiabetes['BloodPressure'].notnull()]

pimadiabetes_bp = pimadiabetes_bp[['BloodPressure','Outcome']].groupby('Outcome')['BloodPressure'].median().reset_index()

pimadiabetes_bp                               

    
# Replacing the zero values of glucose with the corresponding median values of outcomes as calculated

pimadiabetes.loc[(pimadiabetes['BloodPressure'].isnull()) & (pimadiabetes['Outcome'] == 0),'BloodPressure'] = pimadiabetes_bp['BloodPressure'][0]

pimadiabetes.loc[(pimadiabetes['BloodPressure'].isnull()) & (pimadiabetes['Outcome'] == 1),'BloodPressure'] = pimadiabetes_bp['BloodPressure'][1]
pimadiabetes.head(10)
# checking for replacement of all '0' values

pimadiabetes[pimadiabetes['BloodPressure'] == 0].shape[0]
#zero values to be replaced

pimadiabetes[pimadiabetes['Glucose'] == 0].shape[0]



pimadiabetes[pimadiabetes['Glucose'] == 0].index.tolist()
pimadiabetes[pimadiabetes['Glucose'] == 0].groupby('Outcome')['Glucose'].count()
# Replacing '0' values with null values

pimadiabetes['Glucose'].replace(0,np.NaN,inplace = True)
# Calculation of median values grouped according to outcome

pimadiabetes_gl = pimadiabetes[pimadiabetes['Glucose'].notnull()]

pimadiabetes_gl = pimadiabetes_gl[['Glucose','Outcome']].groupby('Outcome').median()

pimadiabetes_gl

pimadiabetes.loc[(pimadiabetes['Glucose'].isnull()) & (pimadiabetes['Outcome'] == 0),'Glucose'] = pimadiabetes_gl['Glucose'][0]

pimadiabetes.loc[(pimadiabetes['Glucose'].isnull()) & (pimadiabetes['Outcome'] == 1),'Glucose'] = pimadiabetes_gl['Glucose'][1]

# checking for replacement of all '0' values

pimadiabetes[pimadiabetes['Glucose'] == 0].shape[0]
# Selection of data to refine for BMI

pimadiabetes[pimadiabetes['BMI'] == 0].shape[0]
pimadiabetes[pimadiabetes['BMI'] == 0].index.tolist()
pimadiabetes[pimadiabetes['BMI'] == 0].groupby('Outcome')['BMI'].count()
# Replacing zero values by NaN values

pimadiabetes['BMI'].replace(0,np.NaN,inplace = True)

# Calculating median of the BMI according to outcome

pimadiabetes_bmi = pimadiabetes[pimadiabetes['BMI'].notnull()]

pimadiabetes_bmi = pimadiabetes_bmi[['BMI','Outcome']].groupby('Outcome').median().reset_index()

pimadiabetes_bmi
# Replacing the Null Values with corresponding median values

pimadiabetes.loc[(pimadiabetes['Outcome'] == 0) & (pimadiabetes['BMI'].isnull()),'BMI'] = pimadiabetes_bmi['BMI'][0]

pimadiabetes.loc[(pimadiabetes['Outcome'] == 1) & (pimadiabetes['BMI'].isnull()),'BMI'] = pimadiabetes_bmi['BMI'][1]

pimadiabetes.head (10)
# checking for replacement of all '0' values

pimadiabetes[pimadiabetes['BMI'] == 0].shape[0]
# CALCULATING NUMBER OF ZERO VALUES

pimadiabetes[pimadiabetes['Insulin'] == 0].shape[0]

pimadiabetes[pimadiabetes['Insulin'] == 0].index.tolist()

pimadiabetes[pimadiabetes['Insulin'] == 0].groupby('Outcome')['Insulin'].count()



# Replace zero with Null Values

pimadiabetes['Insulin'].replace(0,np.NaN,inplace = True)
# Calculating the median grouped by Outcome

pimadiabetes_ins = pimadiabetes[pimadiabetes['Insulin'].notnull()]

pimadiabetes_ins = pimadiabetes_ins[['Insulin','Outcome']].groupby('Outcome')['Insulin'].median().reset_index()

pimadiabetes_ins
# Replacing Null Values with corresponding median values

pimadiabetes.loc[(pimadiabetes['Outcome'] == 0) & (pimadiabetes['Insulin'].isnull()),'Insulin'] = pimadiabetes_ins['Insulin'][0]

pimadiabetes.loc[(pimadiabetes['Outcome'] == 1) & (pimadiabetes['Insulin'].isnull()),'Insulin'] = pimadiabetes_ins['Insulin'][1]

pimadiabetes.head(10)
# checking for replacement of all '0' values

pimadiabetes[pimadiabetes['Insulin'] == 0].shape[0]
# Checking zero values to be replaced

pimadiabetes[pimadiabetes['SkinThickness'] == 0].shape[0]

pimadiabetes[pimadiabetes['SkinThickness'] == 0].index.tolist()

pimadiabetes[pimadiabetes['SkinThickness'] == 0].groupby('Outcome')['SkinThickness'].count()
# Replacing zero values with null values

pimadiabetes['SkinThickness'].replace(0,np.NaN,inplace = True)
# Calculating the median values grouped by Outcome

pimadiabetes_skin = pimadiabetes[pimadiabetes['SkinThickness'].notnull()]

pimadiabetes_skin = pimadiabetes_skin[['SkinThickness','Outcome']].groupby('Outcome').median().reset_index()

pimadiabetes_skin
# Replacing the Null Values with the medians calculated

pimadiabetes.loc[(pimadiabetes['Outcome'] == 0) & (pimadiabetes['SkinThickness'].isnull()),'SkinThickness'] = pimadiabetes_skin['SkinThickness'][0]

pimadiabetes.loc[(pimadiabetes['Outcome'] == 1) & (pimadiabetes['SkinThickness'].isnull()),'SkinThickness'] = pimadiabetes_skin['SkinThickness'][1]

pimadiabetes.head(10)
# checking for replacement of all '0' values

pimadiabetes[pimadiabetes['SkinThickness'] == 0].shape[0]


pimadiabetes.describe().transpose()
#Calculating the number of persons who are diabetic at an younger age (<30)

pimadiabetes[pimadiabetes['Age']<30].count()
pimadiabetes[(pimadiabetes['Age']<30) & (pimadiabetes['Outcome'] == 1)].count()
# Number of persons who are not diabetic at age > 50

pimadiabetes[pimadiabetes['Age']>50].count()

pimadiabetes[(pimadiabetes['Age']>50) & (pimadiabetes['Outcome'] == 0)].count()
sns.countplot(x = 'Outcome', data = pimadiabetes)
# Histogram

pimadiabetes.hist(figsize = (20,20))
#Boxplot for the refined data

pimadiabetes.plot(kind = 'box',figsize = (20,20), subplots = True, layout = (3,3))
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.boxplot(x='Outcome', y='Pregnancies', data=pimadiabetes, ax=axes[0])

sns.countplot(pimadiabetes['Pregnancies'], hue = pimadiabetes['Outcome'], ax=axes[1])
pimadiabetes['age_group'] = pd.cut(pimadiabetes['Age'], range(0, 100, 10))


g = sns.catplot(x="age_group", y="Pregnancies", hue="Outcome",

               data=pimadiabetes, kind="box"

              )

g.fig.set_figheight(4)

g.fig.set_figwidth(20)


g = sns.catplot(x="age_group", y="BMI", hue="Outcome",

               data=pimadiabetes, kind="box"

              )

g.fig.set_figheight(4)

g.fig.set_figwidth(20)


g = sns.catplot(x="age_group", y="SkinThickness", hue="Outcome",

               data=pimadiabetes, kind="box"

              )

g.fig.set_figheight(4)

g.fig.set_figwidth(20)


g = sns.catplot(x="age_group", y="Glucose", hue="Outcome",

               data=pimadiabetes, kind="box"

              )

g.fig.set_figheight(4)

g.fig.set_figwidth(20)
#For processing of data

from sklearn import preprocessing
# Normalization of all numerical columns



# Normalize Pregnancies column

x_array = np.array(pimadiabetes['Pregnancies'])

norm_Pregnancies = preprocessing.normalize([x_array])



# Normalize Glucose column

x_array1 = np.array(pimadiabetes['Glucose'])

norm_Glucose = preprocessing.normalize([x_array1])



# Normalize BloodPressure column

x_array2 = np.array(pimadiabetes['BloodPressure'])

norm_BloodPressure = preprocessing.normalize([x_array2])



# Normalize Skinthickness column

x_array3 = np.array(pimadiabetes['SkinThickness'])

norm_SkinThickness = preprocessing.normalize([x_array3])



# Normalize Insulin column

x_array4 = np.array(pimadiabetes['Insulin'])

norm_Insulin = preprocessing.normalize([x_array4])



# Normalize BMI column

x_array5 = np.array(pimadiabetes['BMI'])

norm_BMI = preprocessing.normalize([x_array5])



# Normalize DiabeticPedigreeFunction column

x_array6 = np.array(pimadiabetes['DiabetesPedigreeFunction'])

norm_DiabeticPedigreeFunc = preprocessing.normalize([x_array6])



# Normalize Age column

x_array7 = np.array(pimadiabetes['Age'])

norm_Age = preprocessing.normalize([x_array7])



# Outcome Variable

x_array8 = np.array(pimadiabetes['Outcome'])
#Preparing Normalized Dataset

pimadiabetes_norm = pd.DataFrame({'Pregnancies':norm_Pregnancies[0,:],

                            'Glucose':norm_Glucose[0,:],

                            'BloodPressure':norm_BloodPressure[0,:],

                            'SkinThickness':norm_SkinThickness[0,:],

                            'Insulin':norm_Insulin[0,:],

                            'BMI':norm_BMI[0,:],

                            'DiabetesPedigreeFunction':norm_DiabeticPedigreeFunc[0,:],

                            'Age':norm_Age[0,:],

                            'Outcome':x_array8

                            })



pimadiabetes_norm.head()
# To get a view of how the  variable are scattered with respect to each other scatter matrix is drawn

from pandas.tools.plotting import scatter_matrix

p = scatter_matrix(pimadiabetes_norm, figsize = (25,25))
corr = pimadiabetes_norm.corr()

corr
plt.figure(figsize=(12,10))

sns.heatmap(corr,annot = True,linewidth = 0.5,fmt = '.2f')
#To split the data into train and test data set

from sklearn.model_selection import train_test_split



# To model Gaussian Naive Bayers classifier

from sklearn.naive_bayes import GaussianNB



#To check the accuracy of the model

from sklearn.metrics import accuracy_score



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix



# for prepartation of classification report

from sklearn.metrics import classification_report



X = pimadiabetes.iloc[:,:8]

y = pimadiabetes.iloc[:,8]
# Split the data into train and test data

# Train data is 70%

# Test data us 30%

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.45, random_state= 1000 )

#implement Gausian Naive Baye's

clf = GaussianNB()

#Fitting GaussianNB into test data set

clf.fit(X_train, y_train, sample_weight = None)
#Predicting test and train data using gaussian function

y_predict_test1 = clf.predict(X_test)

y_predict_train1 = clf.predict(X_train)
#Accuracy of test data set

accuracy_score(y_test,y_predict_test1,normalize = True)
#Accuracy of train data set

accuracy_score(y_train,y_predict_train1,normalize = True)
#Confusion matrix for the test data set

cm_test1 = confusion_matrix(y_test, y_predict_test1)

cm_test1
# Confusion matrix for the train data set

cm_train1 = confusion_matrix(y_train,y_predict_train1)

cm_train1
print(classification_report(y_test, y_predict_test1))

print(classification_report(y_train,y_predict_train1))
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score,auc

import matplotlib.pyplot as plt



##Computing false and true positive rates

fpr, tpr,_=roc_curve(y_predict_test1,y_test,drop_intermediate=False)



plt.figure()

##Adding the ROC

plt.plot(fpr, tpr, color='red',

 lw=2, label='ROC curve')

##Random FPR and TPR

plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')

##Title and label

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.title('ROC curve - Iteration - 1')

plt.legend()

plt.show()
Performance()
X_features = pimadiabetes_norm.iloc[:,:8]

y = pimadiabetes_norm.iloc[:,8]
# Standardization



from sklearn.preprocessing import StandardScaler

rescaledX = StandardScaler().fit_transform(X_features)



X = pd.DataFrame(data = rescaledX, columns= X_features.columns)

X.head()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.45, random_state = 1000)
clf = GaussianNB()
clf.fit(X_train, y_train, sample_weight = None)
#Predicting test and train data using gaussian function

y_predict_test2 = clf.predict(X_test)

y_predict_train2 = clf.predict(X_train)
#Accuracy for test data

accuracy_score(y_test,y_predict_test2, normalize = True)
#Accuracy for train data

accuracy_score(y_train,y_predict_train2,normalize = True)
#Confusion matrix for the test data set

cm_test2 = confusion_matrix(y_test, y_predict_test2)

cm_test2
# Confusion matrix for the train data set

cm_train2 = confusion_matrix(y_train,y_predict_train2)

cm_train2
#Classification report for test data

print(classification_report(y_test, y_predict_test1))

#Classification report for test data

print(classification_report(y_train, y_predict_train1))

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score,auc

import matplotlib.pyplot as plt



##Computing false and true positive rates

fpr, tpr,_=roc_curve(y_predict_test2,y_test,drop_intermediate=False)

AUC = auc(fpr,tpr)

print('AUC is : %0.4f' % AUC)

plt.figure()

##Adding the ROC

plt.plot(fpr, tpr, color='red',

 lw=2, label='ROC curve')

##Random FPR and TPR

plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')

##Title and label

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.title('ROC curve - Iteration - 2')

plt.legend()

plt.show()
# As Age and Pregency has correlation of 0.54, SkinThickness and BMI has correlation of 0.57 we will take only one variable

# out of these two sets.  BMI is choosen as it has less number of null values



X_features = pd.DataFrame(data = pimadiabetes_norm, columns =  ['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age'])

X_features.head()
y = pimadiabetes_norm.iloc[:,8]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.45, random_state = 1000 )

clf = GaussianNB()
clf.fit(X_train, y_train, sample_weight = None)
# Predicting the test and train data set using the GaussianNB classifier

y_predict_test3 = clf.predict(X_test)

y_predict_train3 = clf.predict(X_train)
# Accuracy of test data by GaussianNB Classifier

accuracy_score(y_test, y_predict_test3, normalize = True)
# Accuracy of training data by Gaussian Classifier

accuracy_score(y_train, y_predict_train3, normalize = True)
#Confusion matrix for the test data

cm_test3 = confusion_matrix(y_test, y_predict_test3)

cm_test3
#Confusin matrix for the train data

cm_train3 = confusion_matrix(y_train, y_predict_train3)

cm_train3
# Classification report for the test data

print(classification_report(y_test, y_predict_test3))
#Classification report for the train data

print(classification_report(y_train, y_predict_train3))
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

import matplotlib.pyplot as plt



##Computing false and true positive rates

fpr, tpr,_=roc_curve(y_predict_test3,y_test,drop_intermediate=False)



plt.figure()

##Adding the ROC

plt.plot(fpr, tpr, color='red',

 lw=2, label='ROC curve')

##Random FPR and TPR

plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')

##Title and label

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.title('ROC curve - Iteration - 3')

plt.legend()

plt.show()
# As Age and Pregency has correlation of 0.54, SkinThickness and BMI has correlation of 0.57 we will take only one variable

# out of these two sets.  BMI is choosen as it has less number of null values



X_features = pd.DataFrame(data = pimadiabetes_norm, columns =  ['Pregnancies','Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age'])

X_features.head()
y = pimadiabetes_norm.iloc[:,8]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.45,random_state = 1000 )

clf = GaussianNB()
clf.fit(X_train, y_train, sample_weight = None)
# Predicting the test and train data set using the GaussianNB classifier

y_predict_test4 = clf.predict(X_test)

y_predict_train4 = clf.predict(X_train)
# Accuracy of test data by GaussianNB Classifier

accuracy_score(y_test, y_predict_test4, normalize = True)
# Accuracy of training data by Gaussian Classifier

accuracy_score(y_train, y_predict_train4, normalize = True)
#Confusion matrix for the test data

cm_test4 = confusion_matrix(y_test, y_predict_test4)

cm_test4
#Confusin matrix for the train data

cm_train4 = confusion_matrix(y_train, y_predict_train4)

cm_train4
# Classification report for the test data

print(classification_report(y_test, y_predict_test4))
#Classification report for the train data

print(classification_report(y_train, y_predict_train4))




##Computing false and true positive rates

fpr, tpr,_=roc_curve(y_predict_test4,y_test,drop_intermediate=False)



plt.figure()

##Adding the ROC

plt.plot(fpr, tpr, color='red',

 lw=2, label='ROC curve')

##Random FPR and TPR

plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')

##Title and label

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.title('ROC curve - Iteration - 4')

plt.legend()

plt.show()