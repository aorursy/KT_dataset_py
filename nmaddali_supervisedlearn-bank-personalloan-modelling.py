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
import pandas as pd

import numpy as np

import matplotlib as mp

import seaborn as sns

%matplotlib inline

sns.set(style="ticks")



from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import model_selection



from scipy.stats import zscore
df = pd.read_csv('/kaggle/input/Bank_Personal_Loan_Modelling.csv')

df.columns = ["ID","Age","Experience","Income","ZIPCode","Family","CCAvg","Education","Mortgage","PersonalLoan","SecuritiesAccount","CDAccount","Online","CreditCard"]
df.head(2)
df.columns
df.shape
df.info()
# No columns have null data in the file

df.apply(lambda x : sum(x.isnull()))
# finding unique data

df.apply(lambda x: len(x.unique()))
df.describe().transpose()
plt = sns.pairplot(df[['Age','Experience','Income','ZIPCode','Family','CCAvg' ,'Education' , 'Mortgage','PersonalLoan','SecuritiesAccount','CDAccount','Online','CreditCard']] )

df.head(1)
# Observation on given data



# Age feature is normally distributed with majority of customers falling between 30 years and 60 years of age. 

# We can confirm this by looking at the describe statement above, which shows mean is almost equal to median



# Experience is normally distributed with more customer having experience starting from 8 years. 

# Here the mean is equal to median. There are negative values in the Experience. 

# This could be a data input error as in general it is not possible to measure negative years of experience. 

# We can delete these values.



# Income is positively skewed. Majority of the customers have income between 45K and 55K. 

# We can confirm this by saying the mean is greater than the median



# CCAvg is also a positively skewed variable and average spending is between 0K to 10K and 

# majority spends less than 2.5K



# Mortgage 70% of the individuals have a mortgage of less than 40K. However the max value is 635K



# The variables family and education are ordinal variables. The distribution of families is evenly distributes
plt = sns.boxplot(df[['Income']])#,'Experience','Income','ZIP Code','Family','CCAvg' ,'Education' , 'Mortgage','Personal Loan','Securities Account','CD Account','Online','CreditCard']] )
sns.distplot( df['Age'], color = 'r')
# Before "Negative Experience Cleaning"

# there are 52 records with negative value "Experience"

b4negExp = df.Experience < 0

b4negExp.value_counts()
dfposExp = df.loc[df['Experience'] >0]

mask = df.Experience < 0

column_name = 'Experience'

mylist = df.loc[mask]['ID'].tolist()
for id in mylist:

    age = df.loc[np.where(df['ID']==id)]["Age"].tolist()[0]

    education = df.loc[np.where(df['ID']==id)]["Education"].tolist()[0]

    df_filtered = dfposExp[(dfposExp.Age == age) & (dfposExp.Education == education)]

    exp = df_filtered['Experience'].median()

    df.loc[df.loc[np.where(df['ID']==id)].index, 'Experience'] = exp
# After "Negative Experience Cleaning"

# there are 0 records with negative value "Experience"

aftrnegExp = df.Experience < 0

aftrnegExp.value_counts()
df.describe().transpose()
sns.boxplot(x="Education", y="Income", hue="PersonalLoan", data=df)
sns.boxplot(x="Education", y='Mortgage', hue="PersonalLoan", data=df)
sns.countplot(x="ZIPCode", data=df[df.PersonalLoan==1], hue ="PersonalLoan",orient ='v')
zipcode_top5 = df[df.PersonalLoan==1]['ZIPCode'].value_counts().head(5)

zipcode_top5
sns.countplot(x="Family", data=df,hue="PersonalLoan")
familysize_no = np.mean( df[df.PersonalLoan == 0]['Family'] )

familysize_no
familysize_yes = np.mean( df[df.PersonalLoan == 1]['Family'] )

familysize_yes
from scipy import stats



stats.ttest_ind(df[df.PersonalLoan == 1]['Family'], df[df.PersonalLoan == 1]['Family'])
sns.countplot(x="SecuritiesAccount", data=df,hue="PersonalLoan")
# Observation : Majority of customers who does not have loan have securities account
sns.countplot(x="CDAccount", data=df,hue="PersonalLoan")
# Observation: Customers who does not have CD account , does not have loan as well. 

# This seems to be majority. But almost all customers who has CD account has loan as well
sns.countplot(x="CreditCard", data=df,hue="PersonalLoan")
sns.distplot( df[df.PersonalLoan == 0]['CCAvg'], color = 'r')

sns.distplot( df[df.PersonalLoan == 1]['CCAvg'], color = 'g')
print('Credit card spending of Non-Loan customers: ',df[df.PersonalLoan == 0]['CCAvg'].median()*1000)

print('Credit card spending of Loan customers    : ', df[df.PersonalLoan == 1]['CCAvg'].median()*1000)
sns.distplot( df[df.PersonalLoan == 0]['Income'], color = 'r')

sns.distplot( df[df.PersonalLoan == 1]['Income'], color = 'g')
sns.distplot( df[df.PersonalLoan == 0]['Education'], color = 'r')

sns.distplot( df[df.PersonalLoan == 1]['Education'], color = 'g')
from matplotlib import pyplot as plot

fig, ax = plot.subplots()

colors = {1:'red',2:'yellow',3:'green'}

ax.scatter(df['Experience'],df['Age'],c=df['Education'].apply(lambda x:colors[x]))

plot.xlabel('Experience')

plot.ylabel('Age')
# Observation The above plot shows experience and age have a positive correlation. 

# As experience increase age also increases. Also the colors show the education level. 

# There is gap in the mid forties of age and also more people in the under graduate level
from matplotlib import pyplot as plt

plt.figure(figsize=(25, 25))

ax = sns.heatmap(df.corr(), vmax=.8, square=True, fmt='.2f', annot=True, linecolor='white', linewidths=0.01)

plt.title('Correlation')

plt.show()
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(df.drop(['Experience' ,'ID' ,'CCAvg'], axis=1), test_size=0.3 , random_state=100)
train_set.describe().transpose()
test_set.describe().transpose()
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

train_labels = train_set.pop("PersonalLoan")

test_labels = test_set.pop("PersonalLoan")
dt_model = DecisionTreeClassifier(criterion = 'entropy' , max_depth = 3)
dt_model.fit(train_set, train_labels)
dt_model.score(test_set , test_labels)
y_predict = dt_model.predict(test_set)

y_predict[:5]
test_set.head(5)
naive_model = GaussianNB()

naive_model.fit(train_set, train_labels)



prediction = naive_model.predict(test_set)

naive_model.score(test_set,test_labels)
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error

cm = pd.DataFrame(confusion_matrix(test_labels, prediction).T, index=['No', 'Yes'], columns=['No', 'Yes'])

cm.index.name = 'Predicted'

cm.columns.name = 'True'

cm
randomforest_model = RandomForestClassifier(max_depth=2, random_state=0)

randomforest_model.fit(train_set, train_labels)
Importance = pd.DataFrame({'Importance':randomforest_model.feature_importances_*100}, index=train_set.columns)

Importance.sort_values('Importance', axis=0, ascending=True).plot(kind='barh', color='r', )
predicted_random=randomforest_model.predict(test_set)
randomforest_model.score(test_set,test_labels)
train_set_indep = df.drop(['Experience' ,'ID' ,'CCAvg'] , axis = 1).drop(labels= "PersonalLoan" , axis = 1)

train_set_indep_z = train_set_indep.apply(zscore)

train_set_dep = df["PersonalLoan"]

X = np.array(train_set_indep_z)

Y = np.array(train_set_dep)

X_Train = X[ :3500, :]

X_Test = X[3501: , :]

Y_Train = Y[:3500, ]

Y_Test = Y[3501:, ]
knn = KNeighborsClassifier(n_neighbors= 21 , weights = 'uniform', metric='euclidean')

knn.fit(X_Train, Y_Train)    

predicted = knn.predict(X_Test)

from sklearn.metrics import accuracy_score

acc = accuracy_score(Y_Test, predicted)
print(acc)
X=df.drop(['PersonalLoan','Experience','ID'],axis=1)

y=df.pop('PersonalLoan')
models = []

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('RF', RandomForestClassifier()))

# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

	kfold = model_selection.KFold(n_splits=10, random_state=12345)

	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)

# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()