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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn.preprocessing as skpe

import sklearn.metrics as sklm

import sklearn.model_selection as ms

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier
path="../input/titanicdataset-traincsv/train.csv"

df=pd.read_csv(path)

df.head()
df.shape
df.columns
# Variable Identification

df.dtypes
# or you can use info() method

df.info()    # Name,Sex,Ticket,Cabin and Embarked are categorical variables els are continuous 
# Univariate Analysis 

# Continuous Varaible

df.describe()   # Describe method shows only continuous variables
# Note:-Frequency table is used for categorical - categorical variables; scatter plot is used for continuous – continuous variables; and histogram is used to plot single variable.

plt.hist(df['Age'],alpha=0.9, rwidth=0.8)

plt.xlabel('Age')     # This is a bit right-skewed
sns.boxplot(y=df['Age'],data=df,width=.2,palette='autumn')   # There are a few points lying outside the max value sns.boxplot(y=df['Age'],data=df,width=.2,palette='autumn')   # There are a few points lying outside the max value 
plt.hist(df['Fare'],alpha=0.9, rwidth=0.8)

plt.xlabel('Fare')
sns.boxplot(y=df['Fare'],data=df,width=.2,palette='autumn')   # There are many points lying outside the max range
# Categrical Variable

df['Sex'].value_counts()/len(df['Sex'])*100
(df['Sex'].value_counts()/len(df['Sex'])*100).plot.bar()   # Frequency Chart
# Bivariate Analysis

# Continuous-Continuous Variable

sns.scatterplot(x='Age',y='Fare',data=df,legend='brief')   # There are two points present above 500 mark
# Checking correlation bw different variables

sns.heatmap(df.drop('PassengerId',axis=1).corr(),vmax=.7,cbar=True,annot=True)
sns.set_style(style='whitegrid')

sns.scatterplot(x='Fare',y='Survived',data=df)
# Categorical -Continuous Bivariate Analysis

df.groupby('Sex')['Age'].mean().plot.bar()
from scipy import stats

# But there could be sample indifferences so we perform a ttest to or know if the mean age of both male and female are statistically different or not

males=df[df['Sex']=='male']

females=df[df['Sex']=='female']

stats.ttest_ind(males['Age'],females['Age'],nan_policy='omit') # nan policy is to ignore missing values

# For the two groups to be statistically diiferent pvalue should be < 0.05
# Categorical -Categorical Bivariate Analysis

# Now we want to know the relationship b/w gender and survival rate,so,we make a two-way table

pd.crosstab(df['Sex'],df['Survived'])
# This doesn't tell us the whole story so we perform chi-square test

stats.chi2_contingency(pd.crosstab(df['Sex'],df['Survived']))  # First value is chi-square statistic and second is the p-value which is less than 0.05
sns.boxplot(data=df, y='Age', x='Pclass')
sns.boxplot(data=df, y='Age', x='Survived')
sns.boxplot(data=df, y='Age', x='SibSp')
sns.boxplot(data=df, y='Age', x='Parch')
# Trating Missing Values

df.isnull().sum() 
# Deletion of rows and columns for this much missing values might not be useful,so ,we can impute them with mean of their column

df['Age']=df['Age'].fillna(df['Age'].mean())
# Outlier Detection

# Univariate Outlier Detection

sns.boxplot(y='Age',data=df,width=.4)
# Bivariate Outlier Detection

df.plot.scatter('Age','Fare')
# Removing outliers from dataset

df = df.drop(df[(df['Fare']>400) & (df['Age']>30)].index)
df = df.drop(df[(df['Fare']>500) & (df['Survived']>0.8)].index)   # Dropping that person who gave more than 500 bucks as fare
df = df.drop(df[(df['Age']>79) & (df['Survived']>0.8)].index)  # Dropping that 80 year old person
df.plot.scatter('Age','Fare')     # This is better than the previous scatter plot
# Variable Transformation

sns.distplot(df['Age'],color='Black')

df['Age'].skew()   # This is not a normal distribution
sns.distplot(np.log(df['Age']),color='Black')

np.log(df['Age']).skew()
sns.distplot(np.sqrt(df['Age']),color='Black')

np.sqrt(df['Age']).skew()
df.isnull().sum()
df['Cabin'].unique()
df['Cabin'].fillna(value="NA",inplace=True) #Filling Cabin
def take_section(code):

    return code[0]

df['Cabin']=df['Cabin'].apply(take_section)
df['Cabin'].unique()
df['Embarked'].unique()
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  #Filling Embarked
df.dropna(subset=['Age'],axis=0,inplace=True)     # Treating a single missing entry which i don't know what it is so i dropped that whole row

df.reset_index(drop=True,inplace=True)      # Resetting index
df=df.drop(['PassengerId'],axis=1)
df=df.drop(['Ticket'],axis=1)
# Using expression pattern to extract the Title of the passenger

df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



# Changing to common category

df['Title'] = df['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')

df['Title'] = df['Title'].replace('Ms', 'Miss')

df['Title'] = df['Title'].replace('Mme', 'Mrs')

df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df.drop(['Name'],axis=1,inplace=True)
df.head()
df.dtypes
df['Parch']=df['Parch'].astype('object')
df['SibSp']=df['SibSp'].astype('object')
df=pd.get_dummies(df)

df.head()
# Segregating Features and Labels

x=df.drop(['Survived'],axis=1)

y=df['Survived']
train_x,test_x,train_y,test_y=ms.train_test_split(x,y,random_state=1,stratify=y)
scaler=skpe.StandardScaler()

train_x_scaled=scaler.fit_transform(train_x)

train_x_scaled=pd.DataFrame(train_x_scaled,columns=x.columns)

test_x_scaled=scaler.fit_transform(test_x)

test_x_scaled=pd.DataFrame(test_x_scaled,columns=x.columns)
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(train_x,train_y)
# Making predictions using predict_proba function

probabilities=logreg.predict_proba(test_x)

probabilities
test_preds=probabilities[:,1]

test_preds
def plot_auc(labels, probs):

    ## Compute the false positive rate, true positive rate

    ## and threshold along with the AUC

    fpr, tpr, threshold = sklm.roc_curve(labels, probs[:,1])

    auc = sklm.auc(fpr, tpr)

    

    ## Plot the result

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

    

plot_auc(test_y, probabilities)  
for i in range(len(test_preds)):

    if(test_preds[i]>.55):

        test_preds[i]=1

    else:

        test_preds[i]=0
# Confusion Matrix

cf=sklm.confusion_matrix(test_y,test_preds)

print(cf)
print(sklm.classification_report(test_y,test_preds))
logreg.coef_
x=range(len(train_x.columns))

c=logreg.coef_.reshape(-1)

plt.bar(x,c)

plt.xlabel('Variables')

plt.ylabel('Coeffecients')

plt.title('Coeffecient plot')
Coeffecients=pd.DataFrame({'Variable':train_x.columns,'Coeffecient':abs(c)})

Coeffecients.head()
# Selecting variables with high coeffecients

sign_var=Coeffecients[Coeffecients.Coeffecient>.3]
subset=df[sign_var['Variable'].values]

subset.head()
subset.shape
train_x,test_x,train_y,test_y=ms.train_test_split(subset,y,random_state=5,stratify=y)
score=ms.cross_val_score(LogisticRegression(),X=train_x,y=train_y,cv=10)

score
param_dist={'C':[.1,1,10,100,1000]}

clf=LogisticRegression()

clf_cv=ms.RandomizedSearchCV(clf,param_distributions=param_dist,cv=5)

clf_cv.fit(train_x,train_y)

print("Tuned Logistic Regression Parameters: {}".format(clf_cv.best_params_)) 

print("Best score is {}".format(clf_cv.best_score_)) 
clf=LogisticRegression(C=10)

clf.fit(train_x,train_y)

clf.score(train_x,train_y),clf.score(test_x,test_y)