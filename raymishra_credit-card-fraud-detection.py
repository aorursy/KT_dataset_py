# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn 
import seaborn as sns
import matplotlib.pyplot as plt
import imblearn
from sklearn.model_selection import GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv(r'''../input/creditcard.csv''')
data

data.describe()
data.isnull().any().max()
#no null values in our dataset
print('No Frauds', round(len(data[data["Class"]==0])/len(data)*100, 2))
print('Frauds', round(len(data[data["Class"]==1])/len(data)*100, 2))
#case of a classic imbalanced dataset, a classifier always predicting 0 wil also predict with an accuracy of 99.83%
sns.countplot('Class', data=data)
plt.title('Class Distributions \n (0: No Fraud || 1: Fraud)', fontsize=14)
plt.show()
#next up, we are going to scale the values of time and amount, because all the other columns V1: V27 are scaled
from sklearn.preprocessing import RobustScaler
#RobustScaler are less prone to outliers
rcf= RobustScaler()
data["scaled_amount"]= rcf.fit_transform(data["Amount"].values.reshape(-1,1))
data["scaled_time"]= rcf.fit_transform(data["Time"].values.reshape(-1,1))
data= data.drop(["Time", "Amount"] ,axis =1 )
data
scaled_amount = data['scaled_amount']
scaled_time = data['scaled_time']

data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
data.insert(0, 'scaled_amount', scaled_amount)
data.insert(1, 'scaled_time', scaled_time)
labels = data.columns
X= data.drop('Class', axis=1)
y=data["Class"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 0)

from imblearn.over_sampling import SMOTE
def sampling_func(X, y):
    smote= SMOTE( ratio= 'minority')
    x_sm, y_sm= smote.fit_sample(X, y)
    return x_sm, y_sm
    

def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()
X_sampled, y_sampled = sampling_func(X_train, y_train)
plot_2d_space(X_sampled, y_sampled, 'SMOTE oversampled data')

X_sampled= pd.DataFrame(X_sampled)
y_Sampled= pd.DataFrame(y_sampled)
df= pd.concat([X_sampled, y_Sampled], axis= 1)
df.columns
df.columns= data.columns
df
colors = ["#0101DF", "#DF0101"]
sns.countplot('Class', data=df, palette=colors)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()
f, ax2 = plt.subplots(1, 1, figsize=(24,20))
sampled_corr = df.corr()
sns.heatmap(sampled_corr, cmap='coolwarm_r', annot_kws={'size':20})
ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)
plt.show()

f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
sns.boxplot(x="Class", y="V17", data=df, palette=colors, ax=axes[0])
axes[0].set_title('V17 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V14", data=df, palette=colors, ax=axes[1])
axes[1].set_title('V14 vs Class Negative Correlation')


sns.boxplot(x="Class", y="V12", data=df, palette=colors, ax=axes[2])
axes[2].set_title('V12 vs Class Negative Correlation')


sns.boxplot(x="Class", y="V10", data=df, palette=colors, ax=axes[3])
axes[3].set_title('V10 vs Class Negative Correlation')

plt.show()
f, axes = plt.subplots(ncols=4, figsize=(20,4))

# Positive correlations (The higher the feature the probability increases that it will be a fraud transaction)
sns.boxplot(x="Class", y="V11", data=df, palette=colors, ax=axes[0])
axes[0].set_title('V11 vs Class Positive Correlation')

sns.boxplot(x="Class", y="V4", data=df, palette=colors, ax=axes[1])
axes[1].set_title('V4 vs Class Positive Correlation')


sns.boxplot(x="Class", y="V2", data=df, palette=colors, ax=axes[2])
axes[2].set_title('V2 vs Class Positive Correlation')


sns.boxplot(x="Class", y="V19", data=df, palette=colors, ax=axes[3])
axes[3].set_title('V19 vs Class Positive Correlation')

plt.show()
# # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)
v14_fraud = df['V14'].loc[df['Class'] == 1].values
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v14_iqr = q75 - q25
print('iqr: {}'.format(v14_iqr))

v14_cut_off = v14_iqr * 1.92
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
print('Cut Off: {}'.format(v14_cut_off))
print('V14 Lower: {}'.format(v14_lower))
print('V14 Upper: {}'.format(v14_upper))

outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V14 outliers:{}'.format(outliers))

df = df.drop(df[(df['V14'] > v14_upper) & (df['V14'] < v14_lower)].index)


# # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)
v14_fraud = df['V14'].loc[df['Class'] == 0].values
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v14_iqr = q75 - q25
print('iqr: {}'.format(v14_iqr))

v14_cut_off = v14_iqr * 5.4
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
print('Cut Off: {}'.format(v14_cut_off))
print('V14 Lower: {}'.format(v14_lower))
print('V14 Upper: {}'.format(v14_upper))

outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V14 outliers:{}'.format(outliers))

df = df.drop(df[(df['V14'] > v14_upper) & (df['V14'] < v14_lower)].index)


# # -----> V11 Removing Outliers (Highest Positive Correlated with Labels)
# Removing outliers i genuine cases
v11_fraud = df['V11'].loc[df['Class'] == 0].values
q25, q75 = np.percentile(v11_fraud, 25), np.percentile(v11_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v11_iqr = q75 - q25
print('iqr: {}'.format(v11_iqr))

v11_cut_off = v11_iqr * 2
v11_lower, v11_upper = q25 - v11_cut_off, q75 + v11_cut_off
print('Cut Off: {}'.format(v11_cut_off))
print('V11 Lower: {}'.format(v11_lower))
print('V11 Upper: {}'.format(v11_upper))

outliers = [x for x in v11_fraud if x < v11_lower or x > v11_upper]
print('Feature V11 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V11 outliers:{}'.format(outliers))

df = df.drop(df[(df['V11'] > v11_upper) & (df['V11'] < v11_lower)].index)

# # -----> V11 Removing Outliers (Highest Postive Correlated with Labels)
# Removing outliers in genuine target cases
v11_fraud = df['V11'].loc[df['Class'] == 1].values
q25, q75 = np.percentile(v11_fraud, 25), np.percentile(v11_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v11_iqr = q75 - q25
print('iqr: {}'.format(v11_iqr))

v11_cut_off = v11_iqr * 2.35
v11_lower, v11_upper = q25 - v11_cut_off, q75 + v11_cut_off
print('Cut Off: {}'.format(v11_cut_off))
print('V11 Lower: {}'.format(v11_lower))
print('V11 Upper: {}'.format(v11_upper))

outliers = [x for x in v11_fraud if x < v11_lower or x > v11_upper]
print('Feature V11 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V11 outliers:{}'.format(outliers))

df = df.drop(df[(df['V11'] > v11_upper) & (df['V11'] < v11_lower)].index)
X_train_final= df.drop('Class', axis=1)
y_train_final=df["Class"] 

from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X_train_final)
explained_variance=pca.explained_variance_ratio_
explained_variance
with plt.style.context('dark_background'):
    plt.figure(figsize=(10, 10))

    plt.bar(range(30), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
pca=PCA(n_components=5)
X_new=pca.fit_transform(X_train_final)
X_train_pca= pd.DataFrame(X_new)
X_train_pca
X_test_pca= pca.transform(X_test)
X_test_pca= pd.DataFrame(X_test_pca)
X_test_pca
#from sklearn.ensemble import RandomForestClassifier
#parameters = { 
#    'n_estimators': [200, 500],
#    'max_features': ['auto', 'sqrt', 'log2'],
#    'max_depth' : [4,5,6,7,8],
#   'criterion' :['gini', 'entropy']
#}
#classifier= RandomForestClassifier()
#grid_search= GridSearchCV(estimator=classifier, param_grid=parameters, cv= 5, n_jobs= -1)
#using Logistic Regression to classify the tweets 

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
parameters = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(estimator= classifier,param_grid= parameters, cv=5,  n_jobs= -1)

#from sklearn import svm, datasets
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#svc = svm.SVC()
#grid_search = GridSearchCV(estimator= svc, param_grid= parameters, cv=5, n_jobs= -1)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report



    
grid_search.fit(X_train_pca, y_train_final)


    
y_pred = grid_search.predict(X_test_pca)
cm= confusion_matrix(y_test, y_pred)
labels = ['Not relevant', 'Relevant']
print(classification_report(y_test, y_pred, target_names=labels))
cm
import keras
from keras.models import Sequential
from keras.layers import Dense

#creating an object of sequential class
classifier= Sequential()

n_inputs= X_train_pca.shape[1]

#now we'll use methods of the object to add layers in our neural network model
classifier.add(Dense(output_dim = 3, init='uniform' , activation = 'relu', input_dim = n_inputs))
classifier.add(Dense(output_dim = 1, init='uniform' , activation = 'softmax'))


#compile the ANN model
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics= ['accuracy']) 
#fit the ANN model
classifier.fit(X_train_pca, y_train_final, batch_size= 10, nb_epoch= 100)

#prediction on test set
y_pred = classifier.predict(X_test_pca)
       
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred>0.5)
cm
labels = ['Not relevant', 'Relevant']
print(classification_report(y_test, y_pred>0.5, target_names=labels))