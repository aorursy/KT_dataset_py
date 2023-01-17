#Basic module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Preparation module
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

#Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
df= pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()
#Info
df.info()
#numerical variables summary statistics
df.describe()
#Check for null values in the dataset
df.isnull().sum()
#return the random guess if we had to predict if and employee will leave or not
random_guess = 1-len(df[df['Attrition']=='Yes'])/df.shape[0]
random_guess
#plot the object features #
plt.style.use('ggplot')
#Create a loop that print all categorical variable against the attrition variable
for col in df.select_dtypes('object'):
    plt.figure(figsize=(8,6))
    sns.countplot(x=col,hue='Attrition',data=df)
#plot the int features 
for col in df.select_dtypes('int64'):
    plt.figure(figsize=(10,8))
    sns.boxplot(x='Attrition', y=col,data=df)
#Create heat map to see correlated features
plt.figure(figsize=(10,8))
sns.heatmap(df.corr())
#Look for variables with low variances
df.var(axis=0)
#Drop the features with low variance
to_drop = ['StandardHours','EmployeeCount','EmployeeNumber','Over18','PerformanceRating','StockOptionLevel','JobInvolvement']
df.drop(to_drop,axis=1,inplace=True)
#Split X and y 
X= df.drop('Attrition',axis=1)
y=df['Attrition'].replace({'Yes':1,'No':0})

#split categorical , numerical and ordinal features
categorical = list(X.columns[X.dtypes=='object'])
ordinal = ['Education','EnvironmentSatisfaction','JobLevel','JobSatisfaction','WorkLifeBalance','RelationshipSatisfaction']
numerical = list(X.drop(categorical + ordinal,axis=1))

#Transform numerical and categorical features
X_cat = pd.get_dummies(X[categorical]) #Transform categorical into 0 and 1
X_num = StandardScaler().fit_transform(X[numerical])
X_num = pd.DataFrame(X_num,columns=X[numerical].columns) #Transform the array back to a dataframe for future use

#Create the new X object and look at it
X_new = pd.concat([X_num,X_cat],axis=1)
X_new
X_train,X_test,y_train,y_test= train_test_split(X_new,y,test_size=0.40,shuffle=True)

print('X_train shape',X_train.shape)
print('X_test shape',X_test.shape)
print('y_train shape',y_train.shape)
print('y_test shape',y_test.shape)

#I decided to put the test_size at 40% because the dataset have very few observation.
# PCA to reduce the dimension and plot the graph
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d=np.argmax(cumsum>0.95)+1
#Plot the variance curve against the number of features
plt.plot(cumsum)
plt.xlabel('Number of features')
plt.ylabel('Explained Variance')
plt.plot(d,0.95,marker='d')
#check the number of features to keept
d
k=21
#Changing the Train set
selector = SelectKBest(f_classif,k=k)
selector.fit(X_train,y_train)

# Keep only the selected features into a new variable X_train_reduced
col=selector.get_support(indices=True)
X_train_reduced = X_train.iloc[:,col]

#Changing the Test set
selector.fit(X_test,y_test)

#Same as above
col=selector.get_support(indices=True)
X_test_reduced = X_test.iloc[:,col]
#Quick look at the new data
X_train_reduced
#Create fit and score the model
rfc = RandomForestClassifier(n_estimators=700,max_depth=10,n_jobs=-1,random_state=123)
rfc_model = rfc.fit(X_train_reduced,y_train)

rfc_scores = cross_val_score(rfc,X_train_reduced,y_train,scoring='accuracy',cv=5)
print('This is train score',rfc_scores.mean())
#Predict the model
y_pred_rfc = rfc_model.predict(X_test_reduced)
print('This is test score: ',accuracy_score(y_pred_rfc,y_test))

#Print the confusion_matrix
print('Confusion matrix:')
print(confusion_matrix(y_test,y_pred_rfc))
print(classification_report(y_test,y_pred_rfc))
# Create model that 
def plot_feature_importance(importance,names,model_type): 
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data) 
    
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True) 
    
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
plot_feature_importance(rfc.feature_importances_,X_train_reduced.columns,'RANDOM FOREST')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
