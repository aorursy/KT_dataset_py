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
data = pd.read_excel('/kaggle/input/u8-updated/ANLT5030_U8_AirBNB.xlsx')

data.head(5)
data.dtypes
print('Number of Listings Considered to be Not rented on monthly basis', data['Month_Target'].sum())

print('Number of Listings Considered to be rented on monthly basis', len(data) - data['Month_Target'].sum())

print('Total Number of Listings', len(data))
df = data[['Month_Target',

    'host_id',

'neighbourhood_group',

'neighbourhood',

'latitude',

'longitude',

'room_type',

'price',

'minimum_nights',

'calculated_host_listings_count',

'availability_365']]



df.head()
df.describe()
df.room_type.unique()
df.loc[df['room_type'] == 'Private room', 'Private' ] = True

df.loc[df['room_type'] != 'Private room', 'Private' ] = False



df.loc[df['room_type'] == 'Entire home/apt', 'Entire' ] = True

df.loc[df['room_type'] != 'Entire home/apt', 'Entire' ] = False



df.loc[df['room_type'] == 'Shared room', 'Shared' ] = True

df.loc[df['room_type'] != 'Shared room', 'Shared' ] = False



df['Private'] = df.loc[:, 'Private'].astype(int)

df['Entire'] = df.loc[:, 'Entire'].astype(int)

df['Shared'] = df.loc[:, 'Shared'].astype(int)
print('Sum of Private Rooms', df.Private.sum())

print('Sum of Entire Apts', df.Entire.sum())

print('Sum of Shared Rooms', df.Shared.sum())

df = df.drop(columns = 'room_type')

df.head()
df.neighbourhood_group.unique()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(df['neighbourhood_group'])

le.transform(df['neighbourhood_group'])
df['neighbor_group_val'] = le.transform(df['neighbourhood_group'])
df = df.drop(columns = 'neighbourhood_group')

df.head()
df.neighbourhood.unique()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

le.fit(df['neighbourhood'])

le.transform(df['neighbourhood'])
df['neighbourhood_value'] = le.transform(df['neighbourhood'])
df = df.drop(columns = 'neighbourhood')

df.head()
from matplotlib import pyplot as plt

import seaborn as sns

fig, ax = plt.subplots(figsize=(6,6));

sns.heatmap(df.corr(), ax=ax, annot=False, linewidths=.1, cmap = "YlGnBu");

plt.title('Pearson Correlation matrix heatmap');
plt.hist(df.price)

plt.title('Histogram of Price')

plt.xlabel('Price - Outlier at 3,000')

plt.ylabel('Count of Distribution')

plt.show()

plt.hist(df.minimum_nights)

plt.title('Histogram of Min Nights')

plt.xlabel('Min Nights - Outlier at 370')

plt.ylabel('Count of Distribution')

plt.show()

plt.hist(df.calculated_host_listings_count)

plt.title('Histogram of Sum of Host Listing')

plt.xlabel('Total Host Listings - Outliers')

plt.ylabel('Count of Distribution')

plt.show()

plt.hist(df.availability_365)

plt.title('Histogram of availability_365')

plt.xlabel('availability_365')

plt.ylabel('Count of Distribution')

plt.show()
plt.hist(np.log(df['price']))

plt.title('Histogram of Price Log Transform')

plt.xlabel('Price Log Transform')

plt.ylabel('Count of Distribution')

plt.show()

plt.hist(np.log(df.minimum_nights))

plt.title('Histogram of Min Nights Log Transform')

plt.xlabel('Min Nights Log Transform')

plt.ylabel('Count of Distribution')

plt.show()

plt.hist(np.log(df.calculated_host_listings_count))

plt.title('Histogram of Sum of Host Listing Log Transform')

plt.xlabel('Total Host Listings Log Transform')

plt.ylabel('Count of Distribution')

plt.show()
df['Price_log'] = np.log(df['price'])

df['Min_Nights_log'] = np.log(df['minimum_nights'])

df['Host_Count_log'] = np.log(df['calculated_host_listings_count'])

df.head()
from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

pax_data = df.loc[:, ('Price_log', 'Min_Nights_log', 'Host_Count_log',  

                     'availability_365', 'Private', 'Entire','neighbor_group_val')].values



scaler = StandardScaler()

scaler.fit(pax_data)



y = df.iloc[:,0].values

x = scaler.transform(pax_data)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.2, random_state = 42)
print('Shape of x_train', x_train.shape)

print('Shape of y_train', y_train.shape)

print('Shape of x_test', x_test.shape)

print('Shape of y_test', y_test.shape)
import statsmodels.api as sm

logit_model=sm.Logit(y_train, x_train)

result=logit_model.fit()

print(result.summary())
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

print('Accuracy of logistic regression  classifier on test set: {:.2f}'.format(logreg.score(x_test, y_test)))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print('Confusion Matrix')

print(confusion_matrix)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(x_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.svm import LinearSVC



models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('LSVM', LinearSVC()))
from sklearn.metrics import accuracy_score



for lbl, model in models:

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    print(accuracy_score(y_test, predictions))

    