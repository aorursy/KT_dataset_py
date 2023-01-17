# importing pandas, numpy
import pandas as pd
import numpy as np
import random
# importing packages for plots 
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from pandas import read_csv
#from sklearn.preprocessing import Imputer
import numpy
# reading the training dataset
titanic_df = pd.read_csv("../input/titanic/train.csv")
# analyzing missing values

def missing_value_analysis(data,columns):
    df = data[columns]
    missing_value_perc = df.isnull().sum() *100 /df.shape[0] 
    missing_value_perc = pd.DataFrame({'id':missing_value_perc.index, 'missing_value_perc':missing_value_perc.values})

    missing_data = missing_value_perc[missing_value_perc.missing_value_perc > 0]
    sns.set(style="whitegrid")
    plt.figure(figsize=(10,8))
    ax = sns.barplot(x="id",y="missing_value_perc", data=missing_data, color='steelblue')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    print(missing_data.shape)
    return missing_value_perc
missing_value_analysis(titanic_df,columns=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
# filling mising values of age with mean of columns

titanic_df.iloc[:,5]=titanic_df.iloc[:,5].fillna(titanic_df.mean(axis=0)[3])
# filling nans of embarked with S, as it is the mostly used value

titanic_df['Embarked']=titanic_df['Embarked'].fillna('S')

# dropping cabin column as it ahs 77% of nas

titanic_df = titanic_df.drop(["Cabin"],axis=1)
# lets check for any missing values
titanic_df.isnull().any()
# dropping the columns which are not useful as they are customer specific 

titanic_df = titanic_df.drop(["Ticket"],axis=1)
titanic_df = titanic_df.drop(["Name"],axis=1)

# encoding to convert string columns to numeric
titanic_df['Sex']=titanic_df["Sex"].map({'female':1, 'male':0})
titanic_df['Embarked']=titanic_df['Embarked'].map({'S':0, 'C':1,'Q':2})
# Importing kmeans pacjage for clustering
from sklearn import preprocessing
from sklearn.cluster import KMeans

random.seed(30)
X = np.array(titanic_df.drop(['Survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(titanic_df['Survived'])
X
# applying clustering algorithm for four clusters

clf = KMeans(n_clusters=4)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))
# convering labels to dataframe
pp1=clf.fit(X)
predicted_class1=pp1.labels_
predicted_class1=pd.DataFrame(predicted_class1)
predicted_class1=predicted_class1.rename(columns={0: 'output'})
predicted_class1=predicted_class1.reset_index()
titanic_df=titanic_df.reset_index()
data1=titanic_df.merge(predicted_class1,on='index')
data1['output'].value_counts()
# convering labels to dataframe
pp1=clf.fit(X)
predicted_class1=pp1.labels_
predicted_class1=pd.DataFrame(predicted_class1)

predicted_class1=predicted_class1.rename(columns={0: 'output'})
predicted_class1=predicted_class1.reset_index()
titanic_df=titanic_df.reset_index()
data1=titanic_df.merge(predicted_class1,on='index')
data1['output'].value_counts()
def distplot(data, columns, title=None, xlabel = None, ylabel = None, 
             num_bins = 30, edge_color = 'SteelBlue', fill_color = None,
             line_width = 1.0, fig_size = (8,6), title_font = 14, adjust_top = 0.85, adjust_wspace = 0.3):
    df = data
    fig = plt.figure(figsize = fig_size)
    if title != None:
        title = fig.suptitle(title, fontsize = title_font)
    fig.subplots_adjust(top = adjust_top, wspace = adjust_wspace)

    ax1 = fig.add_subplot(1, 1, 1)
    if xlabel != None:
        ax1.set_xlabel(xlabel)
    if ylabel != None:
        ax1.set_ylabel(ylabel) 
    sns.kdeplot(df[columns], ax=ax1, shade=True, color = edge_color)
# lets visualize teh cluster across the survival rates
distplot(data=data1,columns=['output','Survived'],xlabel = 'clusters',ylabel='survival')
sns.countplot(x='output',data=data1)
class0=data1[data1['output']==0]
class1=data1[data1['output']==1]
class2=data1[data1['output']==2]
class3=data1[data1['output']==3]
#Distribution of Gender 
sns.distplot(class0['Sex']);
# Distribution of Age 
sns.set_color_codes()
sns.distplot(class0['Age'], color="r")  
# Distribution of Pclass 
sns.set_color_codes()
sns.countplot(x='Pclass',data=class0)
# Distribution of Gender 
sns.distplot(class1['Sex'],kde=False);
#Distribution of Age 
sns.distplot(class1['Age'], color="r")  
#Distribution of class 
sns.countplot(x='Pclass',data=class1)
#Distribution of Gender
sns.distplot(class2['Sex']);
#Distribution of Age
sns.distplot(class2['Age'], color="r") 
#Distribution of Class
sns.countplot(x='Pclass',data=class2)
#Distribution of Gender
sns.distplot(class3['Sex']);
#Distribution of Age
sns.distplot(class3['Age'], color="r")  
#Distribution of Class
sns.countplot(x='Pclass',data=class3)
sns.countplot(data=data1,x='output',hue='SibSp')
sns.countplot(data=data1,x='output',hue='Parch')
