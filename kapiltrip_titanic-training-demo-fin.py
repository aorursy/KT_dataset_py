# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
sns.set_style('whitegrid')

%matplotlib inline
###Loading the data
titanic_df = pd.read_csv('../input/train.csv')

titanic_df.head(10)
titanic_df.info()
titanic_df.shape
titanic_df['PassengerId'].nunique()
###Gender Plot
sns.factorplot('Sex',data=titanic_df,kind='count')


### Class plot
sns.factorplot('Pclass',data=titanic_df,kind='count')
sns.factorplot('Pclass',data=titanic_df,hue='Sex',kind='count')

# Set style of scatterplot
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")

# Create scatterplot of dataframe
sns.lmplot('Age', # Horizontal axis
           'Fare', # Vertical axis
           data=titanic_df, # Data source
           fit_reg=False, # Don't fix a regression line
           hue="Survived", # Set color
           scatter_kws={"marker": "D","s": 50}) # S marker size

# Set title
plt.title('Fare ')

# Set x-axis label
plt.xlabel('Age')

# Set y-axis label
plt.ylabel('Fare')
#Creating Child as a feature
def titanic_children(passenger):
    
    age , sex = passenger
    if age <16:
        return 'child'
    else:
        return sex

titanic_df['person'] = titanic_df[['Age','Sex']].apply(titanic_children,axis=1)
        
### Plotting a graph to check the ratio of male,female and children in each category of class

sns.factorplot('Pclass',data=titanic_df,hue='person',kind='count')
###Mean age of the passengers
titanic_df['Age'].mean()
as_fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=5)

as_fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

as_fig.set(xlim=(0,oldest))

as_fig.add_legend()
titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())
sns.factorplot('Embarked',data=titanic_df,hue='Pclass',kind='count')

titanic_df['Alone'] = titanic_df.Parch + titanic_df.SibSp

## if Alone value is >0 then they are with family else they are Alone

titanic_df['Alone'].loc[titanic_df['Alone']>0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Without Family'

#Let us visualise the Alone column

sns.factorplot('Alone',kind='count',data=titanic_df)
# let us see who are alone according to class
sns.factorplot('Alone',kind='count',data=titanic_df,hue='Pclass')
sns.factorplot('Survived',data=titanic_df,kind='count')
sns.factorplot('Survived',data=titanic_df,kind='count',hue='Pclass')
sns.factorplot('Survived',data=titanic_df,kind='count',hue='Sex')
sns.factorplot('Pclass','Survived',data=titanic_df,hue='Sex')
sns.factorplot('Pclass','Survived',data=titanic_df,hue='Alone')
sns.lmplot('Age','Survived',data=titanic_df,logistic=True)
sns.lmplot('Age','Survived',data=titanic_df,hue='Pclass',logistic=True)
sns.lmplot('Age','Survived',data=titanic_df,hue='Sex',logistic=True)
sns.lmplot('Age','Survived',data=titanic_df,hue='Alone',logistic=True)
sns.lmplot('Age','Survived',data=titanic_df,hue='Embarked',logistic=True)
AgeBucket = ['less2', '2-18', '18-35','35-55','55-65', '65plus']
bins = [-1, 2, 18, 35,55, 65, np.inf]
    #combined['AgeBin'] = pd.cut(combined['Age'], bins, labels=names)
    
titanic_df['AgeBucket'] = pd.cut(titanic_df['Age'],bins, labels = AgeBucket)
sns.factorplot('Survived',data=titanic_df,kind='count',hue='AgeBucket')

bins = [-1, 20, 40, 80, 120, 200,300, np.inf]
FareBucket = ['0-20',' 20-40', '40-80', '80-120', '120-200','200-300','300+']
titanic_df['FareBucket'] = pd.cut(titanic_df['Fare'], bins,
                                 labels=FareBucket).astype('str')
sns.factorplot('Survived',data=titanic_df,kind='count',hue='FareBucket')

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
titanic_df = pd.read_csv("../input/train.csv")
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

clf = KMeans(n_clusters=4,random_state=30)
clf.fit(X)

# convering labels to dataframe
pp1=clf.fit(X)
predicted_class1=pp1.labels_
predicted_class1=pd.DataFrame(predicted_class1)
predicted_class1=predicted_class1.rename(columns={0: 'output'})
predicted_class1=predicted_class1.reset_index()
titanic_df=titanic_df.reset_index()
data1=titanic_df.merge(predicted_class1,on='index')
# Getting the number of passengers in different clusters
data1['output'].value_counts()
# Getting the surivival rates in various clusters
data1[['output','Survived']].groupby('output').agg({'Survived':'mean','output':'count'})
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
cluster0=data1[data1['output']==0]
cluster1=data1[data1['output']==1]
cluster2=data1[data1['output']==2]
cluster3=data1[data1['output']==3]
# Cluster0
cluster0['Sex'].value_counts(normalize=True)

cluster0['Pclass'].value_counts(normalize=True)

sns.distplot(cluster0['Age'], color="r",)  

kwargs={'cumulative': True}
sns.distplot(class2['Age'], color="r",hist_kws=kwargs,bins=10)  
# Cluster1
cluster1['Sex'].value_counts(normalize=True)

cluster1['Pclass'].value_counts(normalize=True)

sns.distplot(cluster1['Age'], color="r",)  

kwargs={'cumulative': True}
sns.distplot(class1['Age'], color="r",hist_kws=kwargs,bins=10)  
# Cluster2
cluster2['Sex'].value_counts(normalize=True)

cluster2['Pclass'].value_counts(normalize=True)

sns.distplot(cluster2['Age'], color="r",)  

kwargs={'cumulative': True}
sns.distplot(class2['Age'], color="r",hist_kws=kwargs,bins=10)  
# Cluster3
cluster3['Sex'].value_counts(normalize=True)

cluster3['Pclass'].value_counts(normalize=True)

sns.distplot(cluster3['Age'], color="r",)  

kwargs={'cumulative': True}
sns.distplot(class3['Age'], color="r",hist_kws=kwargs,bins=10)  
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale
def clean_data(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())
    
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1
    
    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2

titanic_df['Alone'] = titanic_df.Parch + titanic_df.SibSp

## if Alone value is >0 then they are with family else they are Alone

titanic_df['Alone'].loc[titanic_df['Alone']>0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Without Family'

# Check score with Decision Tree Model
import pandas as pd
from sklearn import tree
train = pd.read_csv("/kaggle/input/train.csv")
train['Alone'] = train.Parch + train.SibSp
train['Alone'].loc[train['Alone']>0] = 1
train['Alone'].loc[train['Alone'] == 0] = 0
clean_data(train)
target = train["Survived"].values
features = train[["Pclass", "Age", "Fare", "Embarked", "Sex", "SibSp", "Parch",'Alone']].values
decision_tree = tree.DecisionTreeClassifier(random_state = 42)
decision_tree_ = decision_tree.fit(features, target)
print(decision_tree_.score(features, target)) 
# Making the Decision Tree more generalized to reduce overfitting
from sklearn import model_selection
generalized_tree = tree.DecisionTreeClassifier(
                    random_state = 1,
                    max_depth = 7,
                    min_samples_split = 2)
generalized_tree_ = generalized_tree.fit(features, target)
scores = model_selection.cross_val_score(generalized_tree, features, target, scoring = 'accuracy', cv = 50)
print(scores)
print(scores.mean())
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import StandardScaler as scaler

data = export_graphviz(DecisionTreeClassifier(max_depth=3).fit(features, target), out_file=None, 
                       feature_names = ['Pclass', 'Age', 'Fare', 'Embarked', 'Sex', 'SibSp', 'Parch','Alone'],
                       class_names = ['Survived (0)', 'Survived (1)'], 
                       filled = True, rounded = True, special_characters = True)
# we have intentionally kept max_depth short here to accommodate the entire visual-tree
graph = graphviz.Source(data)
graph
#Perform Grid Search to tune hyperparameters of the Random Forest model
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state = 1)
n_estimators = [1740]
max_depth = [6]
min_samples_split = [4 ]
min_samples_leaf = [5] 
oob_score = ['True']

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, oob_score = oob_score)

gridF = GridSearchCV(forest, hyperF, verbose = 1, n_jobs = 4)
bestF = gridF.fit(features, target)
print(bestF)
# Check score with Random Forest Model having the best hyperparameters
from sklearn.ensemble import RandomForestClassifier

r_forest = RandomForestClassifier(criterion='gini',bootstrap=True,
                                    n_estimators=1745,
                                    max_depth=9,
                                    min_samples_split=6,
                                    min_samples_leaf=6,
                                    max_features='auto',
                                    oob_score=True,
                                    random_state=123,
                                    n_jobs=-1,
                                    verbose=0)
rf_clf = r_forest.fit(features, target)
print(rf_clf.score(features, target)) 
rf_clf.oob_score_

