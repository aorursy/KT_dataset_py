#%% Importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pickle

#%%
Train_set = pd.read_csv("train.csv")
Test_set = pd.read_csv("test.csv")
#Ori_Train = Train_set
#Ori_Test = Test_set
pd.set_option('display.max_columns', None)
Train_set.head(8)
Train_set = Train_set.drop(['Id'], axis = 1)

#Train_set.describe()
Train_set.isnull().sum() #check for missing values
"""
Thats great no missing values and all values belong 
to integer data type.Also the data contains 
binary columns of data for qualitative independent 
variables such as wilderness areas and soil type.
"""

Train_set['Cover_Type'].value_counts()
sns.countplot(data=Train_set,x=Train_set['Cover_Type'])
"""
So we see that all the cover types are equal in number
i.e, 2160
"""
for i in range(10,54):
    print (Train_set.iloc[:,i].value_counts())
"""
Soil_Type7 and Soil_type25 are constant with '0' 
binary value.
We can drop those.
"""
Train_set = Train_set.drop(['Soil_Type7', 'Soil_Type15'], axis = 1)
Test_set = Test_set.drop(['Soil_Type7', 'Soil_Type15'], axis = 1)

#%% Visualizations
colnames = Train_set.columns


"""
We can actually plot for [0:52] columns but the wilderness areas 
and soil type data points are in binary data form.Later we will be 
grouping them into single variable  
"""
for i in colnames[0:10]:
    plt.figure()
    sns.violinplot(data=Train_set,x=Train_set['Cover_Type'],y=Train_set[i])
    plt.show()
    
"""
From Visualizations we observe that:
1.Elevation seems to be an important factor for prediction as 
  each Cover_Type has different type of distribution
2.Aspect and slope plots shows normal distribution for most of the 
  classes
3.Horizontal distance to hydrology and roadways plots are quite similar
4.Hillshade 9am and 12pm plots are left skewed
"""

#%% Correlation
corrdata = Train_set.iloc[:,:10]
data_corr = corrdata.corr()
level = 0.6
#print(data_corr) # Are the highly correlated ones

corrmat = Train_set.iloc[:,:10].corr()
sns.heatmap(corrmat,vmax=0.8,square=True) #for better visualization

high_corr = [] #to get highly correlated ones
for i in range(0, 10):
    for j in range(i+1, 10):
        if data_corr.iloc[i,j]>= level and data_corr.iloc[i,j]<1\
        or data_corr.iloc[i,j] <0 and data_corr.iloc[i,j]<=-level:
            high_corr.append([i,j,data_corr.iloc[i,j]])
sorted_high_corr = sorted(high_corr,key= lambda x: abs(x[2]),reverse = True)

cols = corrdata.columns
for i,j,corr in sorted_high_corr:
    print("%s and %s = %.2f" % (cols[i], cols[j], corr))
    
#visualization
for i,j,corr in sorted_high_corr:
    sns.pairplot(data = Train_set, hue='Cover_Type', x_vars=cols[i], y_vars=cols[j])
    plt.show()

  
#%% Combining the One-Hot Encoded Variables
"""
Now we are going to group the one-hot encoded variables of a 
Wilderness_Area', 'Soil_Type into one single variable
"""
row,column = Train_set.shape
grp_data = pd.DataFrame(index= np.arange(0,row), columns=['Wilderness_Area', 'Soil_Type', 'Cover_Type'])
for i in range(0,row):
    area_class = 0;
    soil_class = 0;
    for j in range(10,14):
        if (Train_set.iloc[i,j] == 1):
            area_class = j-9
            break
    for k in range(14,54):
        if (Train_set.iloc[i,k] == 1):
            soil_class = k-13
            break
    grp_data.iloc[i] = [area_class,soil_class,Train_set.iloc[i, column-1]]

plt.figure()    
sns.countplot(x = 'Wilderness_Area', hue = 'Cover_Type', data = grp_data)
plt.show()

plt.figure(figsize=(20,10))
sns.countplot(x='Soil_Type', hue = 'Cover_Type', data= grp_data)
plt.show()

"""
1.Wilderness_Area 1,3,4 show presence of class distinction
2.Few Soil_Types does not show much class distinction
"""

#%%
Id=Test_set['Id']
y=Train_set['Cover_Type']
Train_set=Train_set.drop(['Cover_Type'],1)
Test_set=Test_set.drop(['Id'],1)


#Train-Test split for Cross-validation

x_train, x_test, y_train, y_test = train_test_split(Train_set, y, test_size=0.3, random_state=42)

#%% Using RandomForestClassifier 

"""
We are using the Random Forest Classifier to predict because:
1.The overfitting problem will never come when we use the 
  random forest algorithm.
2.The random forest algorithm can be used for feature engineering.
  As we see there is a strong correlation between different features.
  The RandomForest algorithm does the feature engineering and choose
  the best features for prediction

and of course Random Forest is known for its accuracy and missing value
treatment (We do not have any missing values here)
we will also perform feature selection and build another model
and compare the accuracies 
"""

rf=RandomForestClassifier(n_estimators=150,class_weight='balanced',n_jobs=2,random_state=42)
rf.fit(x_train,y_train)
pred=rf.predict(x_test)
confusion_matrix(pred,y_test)
acc=rf.score(x_test,y_test)
print(acc)


#rf.fit(Train_set,y)
#res=rf.predict(Test_set)
#res

#Result=pd.DataFrame(Id)
#Result['Cover_Type']=res
#Result.head()


"""
So our model is about 86% accurate and in the next step feature 
selection is done by taking top 20 important features.

We will be creating test and train data sets using those features
and test our model accuracy

"""
#%% Feature selection

#Please remove the comment quotes, in case if u need to run the feature selection model

"""
from sklearn.feature_selection import SelectFromModel

colnames = Train_set.columns
imp_fea = []
for feature in zip(colnames, rf.feature_importances_):
    imp_fea.append(feature)
    imp_fea = sorted(imp_fea , key = lambda x:x[1], reverse = True)
    
imp_fea[0:20] #Top 20 important features

sfm = SelectFromModel(rf, threshold=0.008) #Selecting those top 20 features
sfm.fit(x_train, y_train)
for i in sfm.get_support(indices=True):
    print(colnames[i])

#Creating Test and Train Data sets using those TOP FEATURES
    
X_important_train = sfm.transform(x_train)
X_important_test = sfm.transform(x_test)
rf_important = RandomForestClassifier(n_estimators=150,class_weight='balanced', random_state=42, n_jobs=2)
rf_important.fit(X_important_train, y_train)
y_important_pred = rf_important.predict(X_important_test)
confusion_matrix(y_important_pred,y_test)
rf_important.score(X_important_test,y_test)

"""
"""
As can be seen by the accuracy scores, our original model which 
contained all the features is 86.1% accurate while the our 
top features model which contained only the top 20 features is 85.1% 
accurate. Thus, for a 1% cost in accuracy we reduced the 
number of features in the model but I didnt find any appreciable
training time difference between the two models
"""
"""
rf_important.fit(Train_set,y)
res=rf_important.predict(Test_set)
res

"""
#%% Into the pickle file
with open(r"model.pkl","wb") as output_file:
    pickle.dump(rf,output_file,-1)
#rf2 = pickle.load(open('model.pkl','rb'))
#rf2.predict(Test_set[0:8]) #pickle file is working properly

model_columns = list(Train_set.columns)
pickle.dump(model_columns, open('model_columns.pkl','wb'))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
