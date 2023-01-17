# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing.imputation import Imputer
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
my_age_imputer = Imputer(strategy = 'median')
# Any results you write to the current directory are saved as output.
#loading data into dataframe variable 
path = '../input/train.csv'
test_path = '../input/test.csv'
test_data = pd.read_csv(test_path)
train_data = pd.read_csv(path)
total_data = train_data.append(test_data)
#exploring the data
print ((total_data.isnull().sum())) # finding columns that have null values
#getting rid of Cabin since most of its values are missing (687)
data = total_data.drop('Cabin', axis = 1) # drop Cabin because it is mostly blank
# replacing missing values in age with median age
droplist = ['PassengerId','Name','Sex','Ticket','Embarked', 'Survived', 'Pclass', 'Parch', 'Fare', 'SibSp' ]
data1=data.drop(droplist, axis = 1)
imputed_age = my_age_imputer.fit_transform(data1)
#imputer outputs a multi-D array so we need to convert it into a dataframe before we can use it
age_corrected = pd.DataFrame({'ImputedAge':imputed_age[:,0]})
data['ImputedAge'] = age_corrected
data.Embarked.fillna('S', inplace = True) # filling na with mode of location Embarked
data.Embarked = data.Embarked.replace(['S', 'Q', 'C'], [0,1,2])
corr = data.corr()
print (corr.Survived)#checking data correlation
from matplotlib import pyplot as plt 
#plotting histograms of Age and ImputedAge to see if the distribution is similar - histogram shows the imputed data in 25-30 is somewhat higher
plt.hist(data.ImputedAge, range = [0, data.ImputedAge.max()], density = True, alpha = 0.5)
plt.hist(data.Age[~np.isnan(data.Age)], range = [data.Age.min(), data.Age.max()], density = True, alpha = 0.5)
plt.show()
#exploring family size as a feature 
data['fsize']  = data.Parch+data.SibSp+1 #Parch denotes children and sibsp is for siblings while added 1 for self
data.groupby('Survived').fsize.value_counts()# shows that singles died more whereas 2-4 membered families had more survivors
#data['fsize'==1]  = 'Single'
uniq_fsize = list(data.fsize.unique())
print ((uniq_fsize))
uniq_fsize_2 = []
for i in uniq_fsize :
    if i == 1:
        uniq_fsize_2.append('Single')
    elif i > 4:
        uniq_fsize_2.append('Large')
    elif i<=4 and i>1:
        uniq_fsize_2.append('Medium')
print (uniq_fsize_2)
data.fsize.replace(uniq_fsize,uniq_fsize_2, inplace= True)
data.groupby('Survived').fsize.value_counts() # checking if categorization worked
#creating variables child to check if children were more likely to survive
data['isChild'] =np.where(data.ImputedAge<18, 'Child','Adult')
data.groupby('Survived').isChild.value_counts() # more adults died than children
#exploring name as dataset
#name has 3 parts - Title (Mr/Ms etc), Last name (the one before ','), remaining name - lets split them
data['lastName'] = (data.Name.str.split(',', expand = True))[:][0]
data['name_title'] = (data.Name.str.split('.',expand = True))[:][0].str.split(',', expand = True)[:][1]
titles = data.name_title.value_counts().keys().tolist()# some title are quite common while others are not
title_freq = data.name_title.value_counts().tolist()
# creating - categories: Mr, Miss - includes Mlle, Mme and Ms, Mrs, Master and RareTitles
miss_titles = [' Mme', ' Ms', ' Mlle'] 
rare_titles = [' Dr', ' Rev', ' Major', ' Col', ' Don', ' Capt', ' Sir', ' Lady', ' Jonkheer', ' the Countess', ' Dona']
data.name_title.replace(rare_titles,' Rare Titles', inplace = True)
data.name_title.replace(miss_titles,' Miss', inplace = True)
(data.groupby('Sex').name_title.value_counts()) # titles replaced

#creating variables child to check if children were more likely to survive
#be careful to put () between & operators in below
data['isMother'] =np.where((data.ImputedAge>18) & (data.name_title !=' Miss') & (data.Parch >0) & (data.Sex == 'female'), 'Mother','Not Mother')
value = data.groupby('Survived').isMother.value_counts() # more not mothers died than mothers - though not significantly much
import seaborn as sns # plotting result using seaborn - what a beuatiful graph :)
sns.set(style='whitegrid')
ax = sns.barplot (x=data.isMother, y=data.Survived)
# first change categorical values to numbers like 0,1,2
print(list(data)) # seeing column headers - Sex, isChild, name_title and isMother are categorical; lastName has many different values so cecking it first
#data.groupby('Survived').lastName.value_counts() #- doesnt look like much correlation
data.Sex.replace(['male','female'],[0,1], inplace = True)
data.isChild.replace(['Adult','Child'],[0,1], inplace = True)
data.isMother.replace(['Not Mother','Mother'],[0,1], inplace = True)
data.name_title.replace([' Mr',' Master', ' Miss', ' Mrs', ' Rare Titles'],[0,1,2,3,4], inplace = True)
data.fsize.replace(['Single','Medium','Large'],[0,1,2], inplace = True)
data.Fare.fillna(data.Fare.median(), inplace = True)
#binning fare and age as they have larger values then the rest
data['Fare_bin'] = pd.cut(data.Fare, 4, labels = [0,1,2,3])
data['ImputedAge_bin'] = pd.cut(data.ImputedAge.astype(int), 4, labels = [0,1,2,3])
#changing category valeus to numerical values - interpreter takes it as float
data.Fare_bin.replace([0,1,2,3],[0,1,2,3], inplace = True)
data.ImputedAge_bin.replace([0,1,2,3],[0,1,2,3], inplace = True)
#then do corr again
corr = data.corr()
print (corr.Survived)
#using appropriate features and removing redundant ones - for example using bins instead of fare and age
features = ['Pclass', 'Sex', 'Fare_bin', 'ImputedAge_bin', 'fsize', 'isChild', 'name_title', 'isMother']
   
#splitting data back to test and train
train_data = data.iloc[0:891,:] 
test_data = data.iloc[891:1310,:]
#moving Survived to y as that is our result
y = train_data.Survived
#keeping parameters with more 1% correlation
X = train_data[features]     
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_val, y_train, y_val = train_test_split(X,y, random_state = 1)
#getting the train and validate data
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.grid_search import GridSearchCV

#Modeling K Nearest Neighbor 
from sklearn.neighbors import KNeighborsClassifier 
from matplotlib import pyplot as plt
#using GridCV to optimise KNN parameters
knn = KNeighborsClassifier()
n_neigh = [i for i in range (1,50,2)]#arbitrary
algorithm = ['auto']
weight = ['uniform','distance']
leaf = [i for i in range(1,50,5)]#arbitrary
#defining dictionary for Grid CV
param = {'n_neighbors':n_neigh, 'algorithm':algorithm, 'weights':weight, 'leaf_size':leaf}
grid = GridSearchCV(knn,param, cv = 10, scoring = 'accuracy')
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_estimator_)
grid.best_estimator_.fit(X_train,y_train)
grid_pred = grid.best_estimator_.predict(X_val)
#checking how well the KNN is predicting
print("=== Confusion Matrix for Grid KNN ===")
print((confusion_matrix(y_val, grid_pred)))
print("=== Classification Report ===")
print(classification_report(y_val, grid_pred))
t_data = test_data.drop('Age', axis =1)
t_data.Fare.fillna(t_data.Fare.median(), inplace = True)
t_data_X = t_data[features]
predict_t_data = (grid.best_estimator_.predict(t_data_X).astype(int)) # important or the output will be 1.0 style
submission = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": predict_t_data})
submission.to_csv('submission_GridCV_KNN.csv', index=False)