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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn import linear_model
animals1 = pd.read_csv('../input/train.csv')

animals2 = pd.read_csv('../input/test.csv')

print (animals1.shape)

print (animals2.shape)
print (animals1.head())
print (animals2.head())
print (animals1.info())

print (animals2.info())
animals1.rename(columns = {'AnimalID':'ID'}, inplace=True)

animals = pd.merge(animals1, animals2, how='outer')
print (animals.info())
# Name column has missing values

# Lets replace the missing values with 'noname' instead of null

animals['Name'] = animals.loc[animals.Name.isnull(), 'Name']='Noname'
# Lets see all the unique values in AgeuponOutcome column

print (animals.AgeuponOutcome.unique())
# Lets convert the AgeuponOutcome into days and create a new column 

def agetodays(x):

        try:

            y = x.split()

        except:

            return None 

        if 'year' in y[1]:

            return float(y[0]) * 365

        elif 'month' in y[1]:

            return float(y[0]) * (365/12)

        elif 'week' in y[1]:

            return float(y[0]) * 7

        elif 'day' in y[1]:

            return float(y[0])

        

animals['AgeInDays'] = animals['AgeuponOutcome'].apply(agetodays)

print (animals.AgeInDays.unique())
# Lets impute the missing values with median value

animals.loc[(animals['AgeInDays'].isnull()),'AgeInDays'] = animals['AgeInDays'].median()



# Lets drop the AgeuponOutcome column

animals.drop('AgeuponOutcome', axis=1, inplace=True)
# Lets impute the missing value for SexuponOutcome based on the most repeated value

animals.loc[(animals['SexuponOutcome'].isnull()), 'SexuponOutcome'] = animals['SexuponOutcome'].fillna(animals['SexuponOutcome'].value_counts().index[0])
# Lets drop the outcomesubtype as we don't need it for our prediction

animals.drop('OutcomeSubtype', axis=1, inplace=True)
def timetoday(x):

    y = x.split(' ')[1].split(':')[0]

    y = int(y)

    if (y>5) & (y<11):

        return 'morning'

    elif (y>10) & (y<16):

        return 'afternoon'

    elif (y>15) & (y<20):

        return 'night'

    else:

        return 'latenight'   

    

animals['Timeofday'] = animals.DateTime.apply(timetoday)



animals['hours'] = animals.DateTime.str[11:13].astype('int')
# Lets drop the columns we don't need for prediction

animals.drop(['ID', 'DateTime'], axis=1, inplace=True)

animals.info()
# Lets convert the categotical to numerical for prediction 

le = LabelEncoder()

col_num = animals.select_dtypes(include=['O']).columns.values

col_num_list = list(col_num)

col_num_list.remove('OutcomeType')



for col in col_num_list:

    animals[col] = le.fit_transform(animals[col])

print(animals.head())
# Lets have training and testing data



train = animals[animals['OutcomeType'].isnull()==False]

test = animals[animals['OutcomeType'].isnull()==True]

print (train.shape)

print (test.shape)
train['OutcomeType'] = le.fit_transform(train['OutcomeType'])
# Initialize the target and attribute features

target_train = ['OutcomeType']

features_train = ['Name', 'AnimalType', 'SexuponOutcome', 'Breed', 'Color', 'AgeInDays', 'Timeofday', 'hours']



# Initialize logistic regression model

log_model = linear_model.LogisticRegression()



# Train the model

log_model.fit(X = train[features_train],

              y = train[target_train])



# Check trained model intercept

print(log_model.intercept_)



# Check trained model coefficients

print(log_model.coef_)
# Make predictions

preds = log_model.predict(X= test[features_train])

print (preds)

preds = le.inverse_transform(preds)

print (preds)
# Retransform the AnimalType 

animals.loc[animals['AnimalType']==0, 'AnimalType']='Cat'

animals.loc[animals['AnimalType']==1, 'AnimalType']='Dog'



# Retransform the SexuponOutcome



animals.loc[animals['SexuponOutcome']==2, 'SexuponOutcome']='Neutered Male'

animals.loc[animals['SexuponOutcome']==3, 'SexuponOutcome']='Spayed Female'

animals.loc[animals['SexuponOutcome']==1, 'SexuponOutcome']='Intact Male'

animals.loc[animals['SexuponOutcome']==0, 'SexuponOutcome']='Intact Female'

animals.loc[animals['SexuponOutcome']==4, 'SexuponOutcome']='Unknown'
# Impute the predicted values

animals.loc[animals['OutcomeType'].isnull()==True, 'OutcomeType']=preds
animals.info()
sns.countplot(data = animals, x='AnimalType', hue='OutcomeType')

plt.show()
from statsmodels.graphics.mosaicplot import mosaic

plt.rcParams['font.size'] = 8.0

mosaic(animals, ['AnimalType', 'SexuponOutcome','OutcomeType'])

plt.xticks(rotation=90)

plt.show()