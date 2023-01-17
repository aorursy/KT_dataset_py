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
#importing Libraries



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression



from sklearn.metrics import classification_report



from sklearn.model_selection import train_test_split
pwd

#reading the file 



data = pd.read_csv('../input/train.csv')
data.head()
#number of missing values



data.isnull().sum()
#percentage of missing values



data.isnull().mean()
#Let us check if the data missing in cabin and Age is at random 

data['cabin_null'] = np.where(data.Cabin.isnull(), 1, 0)

data['age_null'] = np.where(data.Age.isnull(), 1, 0)
data.groupby(['Survived'])['cabin_null'].mean()

data.groupby(['Survived'])['age_null'].mean()
#lets replace missing age by median, now that we have captured the missingess of Age in Age_null



data['Age'].fillna(data.Age.median(), inplace=True)
#lets check the two null values in Embarked column



data[data.Embarked.isnull()]
data['Embarked'].dropna(inplace= True)
#Next let us check missing values for Cabin



data.Cabin.isnull().mean()
# As we can see that 77% of data is missing in cabin variable so we can think of removing this column but the remaining columns

# might help us in predicting

# we will try random sampling imputation and also add a column to show missingness of the data



data['Cabin'+'_NA'] = np.where(data['Cabin'].isnull(), 1, 0)
data.head()
# selecting random sample for filling the na values

random_sample = data['Cabin'].dropna().sample(data['Cabin'].isnull().sum(), random_state=0,replace=True)



# pandas needs to have the same index in order to merge datasets

random_sample.index = data[data['Cabin'].isnull()].index



# map the random sample to fill in the null values



data.loc[data['Cabin'].isnull(), 'Cabin'] = random_sample


data.Cabin.unique()



#also we can see that there are alot of variables in this column, we can only take the initials letter of the variable to reduce

#the column length
# let's capture the first letter



data['Cabin'] = data['Cabin'].astype(str).str[0]



data.Cabin.unique()



# we can see that the variables are reduced to only 8 variables.
data.Name.head(10)



# we can see that the name column consist of title variables which can be utilized in our predictions as well
# Extract Title from Name, store in column and plot barplot



import re



data['Title'] = data.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))



sns.countplot(x='Title', data=data);

plt.xticks(rotation=45);
#You can see that there are several titles in the above plot and there are many that don't occur so often. So, it makes sense to put them in fewer buckets.

#For example, you probably want to replace 'Mlle' and 'Ms' with 'Miss' and 'Mme' by 'Mrs', as these are French titles and ideally, you want all your data to be in one language. Next, you also take a bunch of titles that you can't immediately categorize and put them in a bucket called 'Rare'.



data['Title'] = data['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})

data['Title'] = data['Title'].replace(['Don', 'Dona', 'Rev', 'Dr',

                                            'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Rare')

sns.countplot(x='Title', data=data);

plt.xticks(rotation=45);
# we have Title, Sex, and Cabin as Columns which have few variables and can be used in prediction 

#lets enumerate them so that we can use the for modelling 



for col in ['Sex', 'Cabin', 'Title','Embarked']:

    labels_dict = {k:i for i, k in enumerate(data[col].unique(), 0)}

    data[col]=data[col].map(labels_dict)
data.head()
# lets drop the unwanted columns

data.drop(['PassengerId','Name','Ticket','Cabin_NA','SibSp','Parch',],axis=1,inplace=True)
# let's now calculate the range to have an idea of the magnitude of the data set



for col in ['Pclass', 'Age', 'Fare']:

    print(col, '_range: ', data[col].max()-data[col].min())
X_train, X_test, y_train, y_test = train_test_split(data.drop('Survived',axis=1), 

                                                    data['Survived'], test_size=0.30, 

                                                    random_state=101)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
X_train_scaled
X_test_scaled
logmodel = LogisticRegression()

logmodel.fit(X_train_scaled,y_train)
predictions = logmodel.predict(X_test_scaled)

print(classification_report(y_test,predictions))