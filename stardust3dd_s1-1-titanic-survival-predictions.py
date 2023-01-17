import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings('ignore')



from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier



print('All modules & libraries imported!')
train= pd.read_csv('../input/titanic/train.csv', index_col=  'PassengerId')

test= pd.read_csv('../input/titanic/test.csv', index_col=  'PassengerId')

df= pd.concat([train, test], axis= 0)

df.head()
print(df.info()) # get the datatypes of each column

print(df.isna().sum()) # 177 missing values for Age, 687 for Cabin, 2 for Embarked
# first column Pclass is passenger class wihtout any missing values & proper datatype. Nothing to do

# second column Name requires some feature engineering. We can extract titles (Mr/Mrs etc) from name

df['Title']= '' # create empty column for storing the appropriate titles



df['Title'][df['Name'].str.contains('Mr. ')]= 'Mr'

df['Title'][df['Name'].str.contains('Mrs. ')]= 'Mrs'

df['Title'][df['Name'].str.contains('Miss. ')]= 'Miss'

df['Title'][df['Name'].str.contains('Mlle. ')]= 'Miss'

df['Title'][df['Name'].str.contains('Ms. ')]= 'Miss'

df['Title'][df['Name'].str.contains('Master. ')]= 'Master'

df['Title'][df['Name'].str.contains('Don. ')]= 'Mr'

df['Title'][df['Name'].str.contains('Dona. ')]= 'Madam'

df['Title'][df['Name'].str.contains('Rev. ')]= 'Rev'

df['Title'][df['Name'].str.contains('Dr. ')]= 'Mr'

df['Title'][df['Name'].str.contains('Mme. ')]= 'Madam'

df['Title'][df['Name'].str.contains('Capt. ')]= 'Mr'

df['Title'][df['Name'].str.contains('Col. ')]= 'Mr'

df['Title'][df['Name'].str.contains('Major. ')]= 'Mr'

df['Title'][df['Name'].str.contains('Countess. ')]= 'Madam'

df['Title'][df['Name'].str.contains('Sir. ')]= 'Mr'

df['Title'][df['Name'].str.contains('Jonkheer. ')]= 'Master'



# column Name is not of use to us anymore so we drop it

df.drop(['Name'], axis= 1, inplace= True)

df.head()
# column Age has some missing values. Let us handle them

age_mean= df['Age'].mean()

age_sd= df['Age'].std()

c= df['Age'].isna().sum()

age_random=  np.random.randint(age_mean-age_sd, age_mean+age_sd, c)

df['Age'][np.isnan(df['Age'])] = age_random

df.isna().sum()
# column Embarked has 2 missing values. Those can be imputed by the mode value

df.fillna({'Embarked': 'S'}, inplace= True)

df.isna().sum()
# column Fare has 1 missing value. The passenger in question was a male of about 61 years age,

# was a 3rd class passenger, & had no spouse/sibling/children/parent aboard the ship. He was not

# allotted any cabin, embarked from Southampton & had ticket number 3701. We try to find the

# most likely fare for a person with these above characteristics. Passenger having ticket number

# 345364 displays all identical characteristics, & has fare 6.2375. We impute this value.

df[(df['Embarked']=='S')&(df['Pclass']==3)&(df['SibSp']== 0)&(df['Parch']== 0)&(df['Age']<=61)&(df['Age']>=59)&(df['Sex']==1)]

df['Fare'][(df['Fare'].isna())]= 6.2375

df.isna().sum()
# Regarding the absence of values in column Cabin of course we cannot make any estimation

# as to which passenger was alootted which cabin. We perform a different sort of feature

# engineering here. We create another column; if Cabin value is present then the new column

# has 1 else 0 value. Then we drop Cabin column.

df['Has_Cabin']= ''

df['Has_Cabin'][df['Cabin'].isna()]= 0

df['Has_Cabin'][df['Cabin'].notna()]= 1

df.drop('Cabin', axis= 1, inplace= True)

df.isna().sum() # no more missing values in dataframe
# Column SibSp specifies number of siblings & spouses travelling with said passenger.

# Column Parch specifies number of parents or children of said passenger.

# We can feature engineer a new column called Family that is SibSp+Parch instead of

# retaining both of them.

df['Family']= df['Parch'] + df['SibSp']

df.drop(['Parch', 'SibSp'], axis= 1, inplace= True)

df['Has']= ''

df['Has'][(df['Family']==0)]= 0

df['Has'][(df['Family']!=0)]= 1

df.head()
# we need to categorize age. Column Age having value below 18 should be a category. 19-39 another,

# 40-60 another, 61 to 80 another. We add an additional column Age_scale, then drop Age.

df['Age_scale']= ''

df['Age_scale'][(df['Age']<=18)]= 0

df['Age_scale'][(df['Age']>18)&(df['Age']<=39)]= 1

df['Age_scale'][(df['Age']>39)&(df['Age']<=60)]= 2

df['Age_scale'][(df['Age']>60)&(df['Age']<=80)]= 3

df['Age_scale']= df['Age_scale'].apply(pd.to_numeric)

df.drop(['Age'], axis= 1, inplace= True)
# lastly we need to drop the column Ticket.

df.drop('Ticket', axis= 1, inplace= True)

df.isna().sum()
df['Fare_range']= ''

df['Fare_range'][(df['Fare'] <= 7.91)] = 0

df['Fare_range'][(df['Fare'] > 7.91) & (df['Fare'] <= 14.454)] = 1

df['Fare_range'][(df['Fare'] > 14.454) & (df['Fare'] <= 31)]   = 2

df['Fare_range'][(df['Fare'] > 31)] = 3

df['Fare_range']= df['Fare_range'].apply(pd.to_numeric)

df.drop(['Fare'], axis= 1, inplace= True)

#train['Fare_range']= train['Fare_range'].astype(int)
# we need to integer encode the column Sex

le= LabelEncoder()

df['Sex']= le.fit_transform(df['Sex'])

df['Embarked']= le.fit_transform(df['Embarked'])

df['Title']= le.fit_transform(df['Title'])



# now we need to one-hot encode the columns Embarked, Pclass, Title

#df= pd.get_dummies(df, prefix=['Embarked'], columns= ['Embarked'])

#df= pd.get_dummies(df, prefix=['Title'], columns= ['Title'])
# let us separate the df dataframe into train & test dataframes now

test= df[df['Survived'].isna()]

train= df[df['Survived'].notna()]
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

# Most survivors were from 1st passenger class
train[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
train[['Fare_range', 'Survived']].groupby(['Fare_range'], as_index=False).mean()
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()

# Most survivors were women
train[['Age_scale', 'Survived']].groupby(['Age_scale'], as_index=False).mean()

# Most survivors were under the age of 18
# let us examine the correlation matrix first to check for any extreme cases, multicollinearity etc.

pd.options.display.max_columns = None

train.corr()

# we can safely assume that high multicollinearity is absent in our data
lr= LogisticRegression()

svm= SVC()

lsvm= LinearSVC()

sgdc= SGDClassifier()

rf= RandomForestClassifier(n_estimators= 1000)

perc= Perceptron(shuffle= True)

skf= StratifiedKFold(n_splits= 10, shuffle= True)

dt= DecisionTreeClassifier()

knn= KNeighborsClassifier()
target= train['Survived']

train.drop(['Survived'], axis= 1, inplace= True)

test.head()
train.head()
scores= {'name':[], 'cl': [], 'r2': []}

from sklearn import model_selection

results= model_selection.cross_validate(lr, train, target, cv= skf, scoring= ['accuracy', 'r2'])

scores['name'].append('Logistic Regression')

scores['cl'].append(results['test_accuracy'].mean())

scores['r2'].append(results['test_r2'].mean())

print('Classification accuracy using Logistic Regression: %.4f' % (results['test_accuracy'].mean()))

print('R squared value using Logistic Regression: %.4f' % (results['test_r2'].mean()))
results= model_selection.cross_validate(rf, train, target, cv= skf, scoring= ['accuracy', 'r2'])

scores['name'].append('Random Forest')

scores['cl'].append(results['test_accuracy'].mean())

scores['r2'].append(results['test_r2'].mean())

print('Classification accuracy using Random Forest: %.4f' % (results['test_accuracy'].mean()))

print('R squared value using Random Forest: %.4f' % (results['test_r2'].mean()))
results= model_selection.cross_validate(svm, train, target, cv= skf, scoring= ['accuracy', 'r2'])

scores['name'].append('SVM')

scores['cl'].append(results['test_accuracy'].mean())

scores['r2'].append(results['test_r2'].mean())

print('Classification accuracy using Support Vector Machines: %.4f' % (results['test_accuracy'].mean()))

print('R squared value using  Support Vector Machines: %.4f' % (results['test_r2'].mean()))
results= model_selection.cross_validate(perc, train, target, cv= skf, scoring= ['accuracy', 'r2'])

scores['name'].append('Perceptron')

scores['cl'].append(results['test_accuracy'].mean())

scores['r2'].append(results['test_r2'].mean())

print('Classification accuracy using Perceptron: %.4f' % (results['test_accuracy'].mean()))

print('R squared value using Perceptron: %.4f' % (results['test_r2'].mean()))
results= model_selection.cross_validate(sgdc, train, target, cv= skf, scoring= ['accuracy', 'r2'])

scores['name'].append('Stochastic Gradient Descent')

scores['cl'].append(results['test_accuracy'].mean())

scores['r2'].append(results['test_r2'].mean())

print('Classification accuracy using Stochastic Gradient Descent: %.4f' % (results['test_accuracy'].mean()))

print('R squared value using Stochastic Gradient Descent: %.4f' % (results['test_r2'].mean()))
results= model_selection.cross_validate(dt, train, target, cv= skf, scoring= ['accuracy', 'r2'])

scores['name'].append('Decision Tree')

scores['cl'].append(results['test_accuracy'].mean())

scores['r2'].append(results['test_r2'].mean())

print('Classification accuracy using Decision Tree: %.4f' % (results['test_accuracy'].mean()))

print('R squared value using Decision Tree: %.4f' % (results['test_r2'].mean()))
results= model_selection.cross_validate(knn, train, target, cv= skf, scoring= ['accuracy', 'r2'])

scores['name'].append('KNN')

scores['cl'].append(results['test_accuracy'].mean())

scores['r2'].append(results['test_r2'].mean())

print('Classification accuracy using K Nearest Neighbors: %.4f' % (results['test_accuracy'].mean()))

print('R squared value using K Nearest Neighbors: %.4f' % (results['test_r2'].mean()))
scores_df= pd.DataFrame(scores)

scores_df.sort_values(by=['r2'], ascending= False)

# we see from the below dataframe that SVM is the best classifier currenty for our purposes
test.drop('Survived', axis= 1, inplace= True)
test.head()
svm= SVC()

svm.fit(train, target)

score_svc= svm.predict(test)
sub = pd.DataFrame({

        "PassengerId": test.index,

        "Survived": score_svc

})

sub['Survived']= sub['Survived'].astype(int)

sub.head()

sub.to_csv('titanic1.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_link(df, title = "Download", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_link(sub)