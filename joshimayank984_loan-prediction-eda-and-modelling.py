# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv(r'/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

test = pd.read_csv(r'/kaggle/input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')
##checking the shape of both train and test dataset

print(train.shape)

print(test.shape)
##get the column names in the data

#although its quiet understood that the test data will not contain the label i.e. Loan Status , but still

#we shall take a look



cols_train = train.columns

cols_test = test.columns

print("Train data Column names : ")

print(cols_train)

print('_____________________________')

print("Train data Column names : ")

print(cols_test)
train.head()
test.head()
## Info about the data using .info and .describe

train.info()

test.info()
##describe

train.describe()
test.describe()
#converting Credit History to Object datatype as its of for 0 and 1

train['Credit_History'] = train['Credit_History'].astype('O')

test['Credit_History'] = test['Credit_History'].astype('O')
print(train.info())

print(test.info())
## describe object type columns

train.describe(include = 'O')
# we will drop ID because it's not important for our model and it will just mislead the model



train.drop('Loan_ID', axis=1, inplace=True)

test.drop('Loan_ID', axis=1, inplace=True)
train.head()
#check if we have any duplicate rows 

print(train.duplicated().any())

print(test.duplicated().any())
len(test)-len(test.drop_duplicates())



## dropping the duplicates in test data

test = test.drop_duplicates()

train['Loan_Status'].value_counts()
# let's look at the target percentage



plt.figure(figsize=(8,6))

sns.set(style="darkgrid")

sns.countplot(train['Loan_Status']);



print('The percentage of Y class : %.2f' % (train['Loan_Status'].value_counts()[0] / len(train)))

print('The percentage of N class : %.2f' % (train['Loan_Status'].value_counts()[1] / len(train)))



# We can consider it as imbalanced data, we shall use F1 Score, Precision and Recall to evaluate the Prediction
train.head()
## finding the null values and treating them

heat = sns.heatmap(train.isnull(), cbar=False)

plt.show()

Null_percent = train.isna().mean().round(4)*100



Null_percent = pd.DataFrame({'Null_percentage' : Null_percent})

Null_percent.head()

Null_percent = Null_percent[Null_percent.Null_percentage > 0].sort_values(by = 'Null_percentage', ascending = False)

print("Percentage of Null cells : \n \n " , Null_percent)
null_klmns = Null_percent.index

null_klmns = list(null_klmns)

train[null_klmns].info()
#we shall separate the categorical and numeric columns

cat_data = []

num_data = []



for i,c in enumerate(train.dtypes):

    if c == object:

        cat_data.append(train.iloc[:, i])

    else :

        num_data.append(train.iloc[:, i])



cat_data = pd.DataFrame(cat_data).transpose()

num_data = pd.DataFrame(num_data).transpose()

cat_data.head()
num_data.head()
##EDA for numerical data

for i in num_data.columns:

    print(i)

    sns.set(style="whitegrid")

    sns.boxplot(num_data[i])

    plt.show()

    

##EDA for categorical data

for i in cat_data.columns:

    print(i)

    total = float(len(cat_data))

    plt.figure(figsize=(8,10))

    sns.set(style="whitegrid")

    ax = sns.countplot(cat_data[i])

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format(height/total),ha="center") 

    plt.show()
## EDA on categorical data relative to Loan Status

##EDA for categorical data

for i in cat_data.columns:

    print(i)

    total = float(len(cat_data))

    plt.figure(figsize=(8,10))

    sns.set(style="darkgrid")

    ax = sns.countplot(x = i, hue = 'Loan_Status', data = cat_data)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format(height/total),ha="center") 

    plt.show()

    

#from the corresponding charts one can gain a lot of important info
##creating a pair plot

sns.pairplot(train)

plt.show()
## Data Imputation

# for cat_data



# If you want to fill every column with its own most frequent value you can use



cat_data = cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))

cat_data.isnull().sum().any() # no more missing data
#num_data

##as we have many outliers in columns with Numerical data, we shall impute the blank cells with the median of their respective columns



num_data.fillna(num_data.median(), inplace=True)

num_data.isnull().sum().any() # no more missing data 

num_data.head()
## num_data has certain columns with some very high valued columns and some very low, thus we 

#should standardize the values of these columns



for col in num_data.columns:

    num_data[col] = (num_data[col]-num_data[col].min())/(num_data[col].max() - num_data[col].min())

    

num_data.head()
##Label Encoding

from sklearn.preprocessing import LabelEncoder  

le = LabelEncoder()

cat_data.head()
# transform the target column



target_values = {'Y': 1 , 'N' : 0}



target = cat_data['Loan_Status']

cat_data.drop('Loan_Status', axis=1, inplace=True)



target = target.map(target_values)
# transform other columns



for i in cat_data:

    cat_data[i] = le.fit_transform(cat_data[i])
cat_data.head()
df = pd.concat([cat_data, num_data, target], axis=1)

df.head()
#confirming if we have any null values



df.isna().any()



#so we are good to model
## Creating our variable and target dataset

X = pd.concat([cat_data, num_data], axis=1)

y = target
# we will use 4 different models for training



## ---------------------------All in one modelling---------------------------



from sklearn.model_selection import train_test_split  #to split the dataset for training and testing

from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import metrics #for checking the model accuracy

from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import StratifiedShuffleSplit

# we will use StratifiedShuffleSplit to split the data Taking into consideration that we will get the same ratio on the target column



from sklearn.model_selection import StratifiedShuffleSplit



sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train, test in sss.split(X, y):

    X_train, X_test = X.iloc[train], X.iloc[test]

    y_train, y_test = y.iloc[train], y.iloc[test]

    

print('X_train shape', X_train.shape)

print('y_train shape', y_train.shape)

print('X_test shape', X_test.shape)

print('y_test shape', y_test.shape)



# almost same ratio

print('\nratio of target in y_train :',y_train.value_counts().values/ len(y_train))

print('ratio of target in y_test :',y_test.value_counts().values/ len(y_test))

print('ratio of target in original_data :',df['Loan_Status'].value_counts().values/ len(df))
models = {

    'LogisticRegression': LogisticRegression(random_state=34),

    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors= 5),

    'SVC': SVC(random_state=34),

    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=1, random_state=34)

}
# loss



from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score



def loss(y_true, y_pred, retu=False):

    pre = precision_score(y_true, y_pred)

    rec = recall_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred)

    loss = log_loss(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)

    

    if retu:

        return pre, rec, f1, loss, acc

    else:

        print('  pre: %.3f\n  rec: %.3f\n  f1: %.3f\n  loss: %.3f\n  acc: %.3f' % (pre, rec, f1, loss, acc))
# train_eval_train



def train_eval_train(models, X, y):

    for name, model in models.items():

        print(name,':')

        model.fit(X, y)

        loss(y, model.predict(X))

        print('-'*30)

        

train_eval_train(models, X_train, y_train)



# we can see that best model is LogisticRegression at least for now, SVC is just memorizing the data so it is overfitting .
X_train.shape
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)



def train_eval_cross(models, X, y, folds):

    # we will change X & y to dataframe because we will use iloc (iloc don't work on numpy array)

    X = pd.DataFrame(X) 

    y = pd.DataFrame(y)

    idx = [' pre', ' rec', ' f1', ' loss', ' acc']

    for name, model in models.items():

        ls = []

        print(name,':')



        for train, test in folds.split(X, y):

            model.fit(X.iloc[train], y.iloc[train]) 

            y_pred = model.predict(X.iloc[test]) 

            ls.append(loss(y.iloc[test], y_pred, retu=True))

        print(pd.DataFrame(np.array(ls).mean(axis=0), index=idx)[0])  #[0] because we don't want to show the name of the column

        print('-'*30)

        

train_eval_cross(models, X, y, skf)
