# Import libraires 

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv") #Load the dataset
data.describe()
data.head(10)
data.tail(10) 
data.isnull().sum()

#There are no missing values in the dataset but there are 0 values.
plt.figure(figsize=(10,8))

sns.heatmap(data.corr(),annot=True,linewidths=3)



plt.show()
for col in data.columns:

    if col !="Pregnancies" and col !="DiabetesPedigreeFunction" and col !="Outcome":

        data[col] = data[col].replace(0,np.nan)



#We have replaced the inaccure values with nan values.Now let's look for the missing values        

total_missing_values = pd.DataFrame(data.isnull().sum()) # count the percentage of misisng values



percentage_mising_values = pd.DataFrame(data.isnull().sum() / data.shape[0] * 100) #Compute percentage of missing values



missing_df = pd.DataFrame(index=data.columns,columns = ['Total missing values','Total missing values %'])

missing_df['Total missing values'] = total_missing_values

missing_df['Total missing values %'] = percentage_mising_values 

missing_df
data.drop(['Insulin','SkinThickness'],inplace=True,axis=1)
data.hist(figsize=(10,10),grid=[3,3])

plt.show()
data['BMI'].fillna(data['BMI'].mean(),inplace=True)

data['Glucose'].fillna(data['Glucose'].mean(),inplace=True)

data['BloodPressure'].fillna(data['BloodPressure'].mean(),inplace=True)
data.hist(figsize=(10,10),grid=[4,4])

plt.show()
sns.distplot(data['Age'])

plt.show()
sns.boxplot(x='Outcome',y='Age',data=data)

plt.show()
data[data['Age'] > 68]
data.drop(data[data['Age'] > 68].Age,inplace=True)
sns.boxplot(y='DiabetesPedigreeFunction',x='Outcome',data=data)

plt.show()
sns.boxplot(y='Pregnancies',x='Outcome',data=data)

plt.show()
data.drop(data[data['Pregnancies'] >10].index,inplace=True)
data['Outcome'].value_counts()
sns.distplot(data['Outcome'])

plt.show()
from sklearn.utils import resample

from imblearn.over_sampling import SMOTE



imbalanced = data.copy(deep=True)

downsampled = data.copy(deep=True)

synthetic_dataset = data.copy(deep=True)



#Before we down sample the data we should split the data based on class labels.

not_diabetic = downsampled[downsampled.Outcome==0]

diabetic = downsampled[downsampled.Outcome==1]



not_diabetic_downsampled = resample(not_diabetic,

                                replace = False, # sample without replacement

                                n_samples = len(diabetic), # match minority n

                                random_state = 27) # reproducible results



# combine minority and downsampled majority

downsampled  = pd.concat([not_diabetic_downsampled, diabetic])



#split the data into x and y

synthetic_dataset_X = synthetic_dataset.iloc[:,:-1]

synthetic_dataset_y = synthetic_dataset.iloc[:,-1]

sm = SMOTE(random_state = 10) 

#Fit the SMOTE model to the data

synthetic_X, synthetic_y = sm.fit_sample(synthetic_dataset_X, synthetic_dataset_y.ravel()) 

df_synthetic_dataset_y = pd.DataFrame(columns=['Outcome'])

df_synthetic_dataset_y['Outcome'] = synthetic_y

synthetic_dataset_X = pd.DataFrame(synthetic_X, columns=synthetic_dataset.columns[:-1])

#Concat X and Y again into singel dataset

synthetic_dataset = pd.concat([synthetic_dataset_X,df_synthetic_dataset_y],axis=1) 



print("Class labels of imbalanced dataset has {} 0s and {} 1s.\n".format(imbalanced['Outcome'].value_counts()[0],imbalanced['Outcome'].value_counts()[1]))

print("Class labels of downsampled dataset has {} 0s and {} 1s.\n".format(downsampled['Outcome'].value_counts()[0],downsampled['Outcome'].value_counts()[1]))

print("Class labels of synthetic dataset has {} 0s and {} 1s.\n".format(synthetic_dataset['Outcome'].value_counts()[0],synthetic_dataset['Outcome'].value_counts()[1]))
plt.figure(figsize=(10,5))

sns.pairplot(data=data,hue='Outcome')

plt.show()
plt.figure(figsize=(10,8))

sns.heatmap(data.corr(),annot=True,linewidths=3)



plt.show()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,classification_report



datasets = [imbalanced,downsampled,synthetic_dataset]

dataset_names = ['imbalanced','downsampled','synthetic_dataset']

models = [KNeighborsClassifier,LogisticRegression,RandomForestClassifier]



columns = ['Dataset','Model','accuray_score','f1_score','TN','FP','FN','TP']

Model_details = pd.DataFrame(columns=columns)







for dataset_name in dataset_names:

    index_dataset = dataset_names.index(dataset_name)

    frame = {}

    for model in models:

        frame['Dataset'] = dataset_name

        frame['Model'] = model.__name__

        dataset = datasets[index_dataset]

        X= dataset.iloc[:,:-1]

        y = dataset.iloc[:,-1]

        

        #scale the values

        sc_X = StandardScaler()

        X = sc_X.fit_transform(X)

        #split the dataset into train and test

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42, stratify=y)

        clf = model()

        clf.fit(X_train,y_train)

        

        y_pred = clf.predict(X_test)

        

        frame['f1_score'] = f1_score(y_test,y_pred)

        frame['accuray_score'] = accuracy_score(y_test,y_pred)

        frame['TN'] = confusion_matrix(y_test,y_pred)[0][0]

        frame['FP'] = confusion_matrix(y_test,y_pred)[0][1]

        frame['FN'] = confusion_matrix(y_test,y_pred)[1][0]

        frame['TP'] = confusion_matrix(y_test,y_pred)[1][1]

        Model_details = Model_details.append(frame,ignore_index=True)
Model_details
from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {"n_neighbors": np.arange(1, 25),'weights':['uniform','distance']}





X= downsampled.iloc[:,:-1]

y = downsampled.iloc[:,-1]

        

#scale the values

sc_X = StandardScaler()

X = sc_X.fit_transform(X)

#split the dataset into train and test

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42, stratify=y)



clf = KNeighborsClassifier()

clf_cv= GridSearchCV(clf,param_grid,cv=3,n_jobs=-1)

clf_cv.fit(X,y)
clf_cv.best_params_
clf_cv.best_score_