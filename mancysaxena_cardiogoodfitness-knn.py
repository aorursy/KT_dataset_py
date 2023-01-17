



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import missingno as msno

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

customer_fit_df = pd.read_csv('../input/cardiogoodfitness/CardioGoodFitness.csv')
customer_fit_df.head()
customer_fit_df.info()
## The info() method in pandas prints the summary of a dataframe and returns None

## More Info https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.info.html
p=msno.bar(customer_fit_df)
customer_fit_df.describe(include='all')

#By Default, the describe method omits the non numeric values when .describe() method is called. 

#include='all' here, also includes categorical value.



#Following observations can be made :

#    1. No value is null (Or values like Age, income,are all > 0 indicating that all are genuine values)

#    2. TM195 is the highest purchased product (In the describe method, if multiple categories are highest, arbitrarily one of them is chosen)

#    3. freq is the frequency of the 'top' value mentioned above

#    4. More number of Males are present in the data than female

#    5. Marital Status that is highest in the data is Parterned



#    For more info on .descibe() method, refer : https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
product_names=customer_fit_df['Product'].unique()

product_count=customer_fit_df['Product'].value_counts()

plt.bar(product_names,product_count)
gender_type = customer_fit_df['Gender'].unique()

gender_distibution =customer_fit_df['Gender'].value_counts()

plt.bar(gender_type, gender_distibution)
customer_fit_df['Education'].nunique()
customer_fit_df['Education'].value_counts().plot(kind='bar')
marital_status = customer_fit_df['MaritalStatus'].unique()

marital_status_values = customer_fit_df['MaritalStatus'].value_counts()

plt.bar(marital_status,marital_status_values)
plt.hist(customer_fit_df['Income'])
plt.boxplot(customer_fit_df['Income'])
customer_fit_df_copy = customer_fit_df.copy(deep=True) 

lower_bound = customer_fit_df_copy.quantile(0.25)

upper_bound = customer_fit_df_copy.quantile(0.75)

IQR = upper_bound - lower_bound

customer_fit_df = customer_fit_df_copy[~((customer_fit_df_copy < lower_bound- 1.5* IQR ) |  (customer_fit_df_copy > upper_bound+ 1.5* IQR )).any(axis=1)]
usage_measures = customer_fit_df['Usage'].unique()

usage_measures_values = customer_fit_df['Usage'].value_counts()

plt.bar(usage_measures,usage_measures_values)
Fitness_measure = customer_fit_df['Fitness'].unique()

Fitness_measures_values = customer_fit_df['Fitness'].value_counts()

plt.bar(Fitness_measure,Fitness_measures_values)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(customer_fit_df['MaritalStatus'])
customer_fit_df['MaritalStatus'] = le.transform(customer_fit_df['MaritalStatus'])
le1 = LabelEncoder()

le1.fit(customer_fit_df['Product'])

customer_fit_df['Product'] = le1.transform(customer_fit_df['Product'])
le1 = LabelEncoder()

customer_fit_df['Gender'] = le1.fit_transform(customer_fit_df['Gender'])
customer_fit_df['Gender']
customer_fit_df['Product']
customer_fit_df.head(50)
plt.figure(figsize=(12,10))

p = sns.heatmap(customer_fit_df.corr(), annot=True, vmin=-1, vmax=1)
X = customer_fit_df

X.head()

X = customer_fit_df.iloc[:,1:]

Y = customer_fit_df.iloc[:,0].to_frame()

Y.head()

customer_fit_df
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=42, test_size=0.3)
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()

scalar.fit(X_train)



X_train = scalar.transform(X_train)

X_test = scalar.transform(X_test)

X_train
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=4)

classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(Y_test, y_pred))

print(classification_report(Y_test, y_pred))
Expected = Y_test['Product'].values
X_test[6]
my_submission = pd.DataFrame({'Predicted': y_pred, 'Expected' :Expected})

my_submission.to_csv('./submission.csv', index=False)
