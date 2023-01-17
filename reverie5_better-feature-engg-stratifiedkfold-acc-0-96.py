#Import libraries

import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns
#reading data

data= pd.read_csv('/kaggle/input/drug-classification/drug200.csv')

print("Dataframe Shape: ",data.shape)
#check data

data.head()
# Target variable analysis

data['Drug'].value_counts()
#Feature variables analysis

#Check for missing values

data.isnull().sum()
data.describe()
# col-Age

sns.distplot(data['Age'])
pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)
data.groupby(['Age', 'Drug']).size()
# col- Sex

data.Sex.value_counts()
data.groupby(['Sex', 'Drug']).size()
# col- BP

data.BP.value_counts()
sns.catplot(x="Drug", y="BP", data=data)
data.groupby(['BP', 'Drug']).size()
# col- Cholesterol

data.Cholesterol.value_counts()
data.groupby(['Cholesterol', 'Drug']).size()
# col- Na_to_K

print(data.Na_to_K.nunique())

sns.distplot(data['Na_to_K'])
sns.catplot(x="Drug", y="Na_to_K", data=data)
# Positive skewness also tells, (mean and median) > mode

#mean, median, mode: lets check

print(data.Na_to_K.mean())

print(data.Na_to_K.median())

print(data.Na_to_K.mode()[0])
#skewness and kurtosis

print("Skewness= ", data['Na_to_K'].skew())

print("Kurtosis= ", data['Na_to_K'].kurt())
data.Age.max()
# feature engg

# Binning Age into Age groups

bins= [13,18,65,80]

labels = ['Teen','Adult','Elderly']

data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)

data.drop('Age', axis=1, inplace=True)

print (data.head())
data.AgeGroup.value_counts()
data['is_Na2K_greater15'] = [1 if x>15 else 0 for x in data['Na_to_K']]
# Na_to_K groups

data['Na_to_K_groups'] = pd.qcut(data['Na_to_K'],

                            q=[0, .2, .4, .6, .8, 1],

                            labels=False)

data.drop('Na_to_K', axis=1, inplace=True)

data.Na_to_K_groups.value_counts()
# Binarize Sex variable

data['Sex'].replace(['F','M'],[0,1],inplace=True)
#Label encoding

from sklearn import preprocessing 

  

le = preprocessing.LabelEncoder() 

data['BP']= le.fit_transform(data['BP']) 

data['Cholesterol']= le.fit_transform(data['Cholesterol'])

data['AgeGroup']= le.fit_transform(data['AgeGroup']) 

data.head()
data.columns
#features

features = ['Sex', 'BP', 'Cholesterol', 'AgeGroup','is_Na2K_greater15', 'Na_to_K_groups']
#model

from sklearn import tree

from sklearn import ensemble

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn import metrics
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

X = data[features]

y = data.Drug



scores= []

i=1

for train_index,test_index in kf.split(X, y):

    print('Fold no. = ', i)

    

    x_train, x_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    

    #model

    model1 = tree.DecisionTreeClassifier(random_state=42)

    model1.fit(x_train, y_train)

     

    test_pred= model1.predict(x_test)

    test_acc = metrics.accuracy_score(y_test, test_pred)

    print('Accuracy score over test set:',test_acc)

    scores.append(test_acc)    

    

    i+=1

    

#mean score

print()

print('Mean Accuracy for Decision Tree: ', np.mean(scores))
#RandomForestClassifier

kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

X = data[features]

y = data.Drug



scores= []

i=1

for train_index,test_index in kf.split(X, y):

    print('Fold no. = ', i)

    

    x_train, x_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    

    #model

    model2 = ensemble.RandomForestClassifier(random_state=42)

    model2.fit(x_train, y_train)

     

    test_pred= model2.predict(x_test)

    test_acc = metrics.accuracy_score(y_test, test_pred)

    print('Accuracy score over test set:',test_acc)

    scores.append(test_acc)    

    

    i+=1

    

#mean score

print()

print('Mean Accuracy for Random Forest Classifier: ', np.mean(scores))
# model-random forest classifier feature importance

feat_importances = pd.Series(model2.feature_importances_, index=features)

feat_importances.plot(kind='barh')