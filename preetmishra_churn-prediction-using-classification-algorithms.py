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
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report
df = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()
print('Rows: {}, Columns: {}'.format(df.shape[0], df.shape[1]))

features = df.columns.to_list()

features.remove('Churn')

print('Features:\n', features, sep='')
df.drop(["customerID"], axis = 1,inplace = True)
df.isnull().sum()
def check_values():

    for i in range(df.columns.size):

        print(df.columns[i] + ':')

        for j in range(df[df.columns[i]].size):

            if df[df.columns[i]][j] == ' ' :

                print('Found space')

            elif df[df.columns[i]][j] == '-' :

                print('Found hyphen')

            elif df[df.columns[i]][j] == 'NA' :

                print('Found NA')

        print('Done!')
check_values()
# replacing spaces with null values

df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
check_values()
df.isnull().sum()
df = df[df["TotalCharges"].notnull()]
df.isnull().sum()
df.reset_index(drop = True, inplace = True)
df.dtypes
df.TotalCharges = df.TotalCharges.astype(float)
df.dtypes
df.nunique()
for i in range(df.columns.size) :

    if df[df.columns[i]].nunique() <= 4:

        print(df[df.columns[i]].unique())
col_map = ['Partner', 

          'Dependents', 

          'PhoneService', 

          'MultipleLines',

          'OnlineSecurity',

          'OnlineBackup',

          'DeviceProtection',

          'TechSupport',

          'StreamingTV',

          'StreamingMovies',

          'PaperlessBilling', 

          'Churn']

for col in col_map:

    df[col] = [1 if val == "Yes" else 0 if val == "No" else -1 for val in df[col]]
for i in range(df.columns.size) :

    if df[df.columns[i]].nunique() <= 4:

        print(df[df.columns[i]].unique())
df['gender'] = [1 if gen == 'Male' else 0 for gen in df['gender']]
df.head()
plt.figure(figsize = [15, 6])

plt.pie(df['Churn'].value_counts(), 

        labels = ['No', 'Yes'], 

        startangle = 90, 

        autopct='%1.1f%%', 

        wedgeprops = {'width' : 0.2},

        counterclock = True);

plt.title('Customer churn')

plt.legend()

plt.axis('equal');
plt.figure(figsize = [15, 6])

plt.suptitle('Gender distribution')



plt.subplot(1, 2, 1)

plt.pie(df[df['Churn'] == 1]['gender'].value_counts(), 

        labels = ['Female', 'Male'], 

        startangle = 90, 

        autopct='%1.1f%%', 

        wedgeprops = {'width' : 0.2},

        counterclock = True);

plt.legend()

plt.text(-0.13,-0.03, 'Churn',fontsize = 14)

plt.axis('equal')



plt.subplot(1, 2, 2)

plt.pie(df[df['Churn'] == 0]['gender'].value_counts(), 

        labels = ['Male', 'Female'], 

        startangle = 90, 

        autopct='%1.1f%%', 

        wedgeprops = {'width' : 0.2},

        counterclock = True);

plt.legend()

plt.text(-0.22,-0.03, 'Not Churn',fontsize = 14)

plt.axis('equal');
bluish = sns.color_palette()[0]

orangish = sns.color_palette()[1]
plt.figure(figsize = [15, 8])

ten_dist = sns.kdeplot(df['tenure'][df["Churn"] == 0], color = bluish, shade = True)

ten_dist = sns.kdeplot(df['tenure'][df["Churn"] == 1], color = orangish, shade= True)

ten_dist.legend(['Not Churn', 'Churn'])

ten_dist.set_xlabel('Tenure')

ten_dist.set_ylabel('Frequency')

plt.xticks(np.arange(0, 80, 5))

plt.title('Distribution of tenure for churn and not churn customers');
plt.figure(figsize = [15, 8])

ten_dist = sns.kdeplot(df['MonthlyCharges'][df["Churn"] == 0], color = bluish, shade = True)

ten_dist = sns.kdeplot(df['MonthlyCharges'][df["Churn"] == 1], color = orangish, shade= True)

ten_dist.legend(['Not Churn', 'Churn'])

ten_dist.set_xlabel('Monthly charges')

ten_dist.set_ylabel('Frequency')

plt.title('Distribution of monthly charges for churn and not churn customers');
plt.figure(figsize = [15, 8])

ten_dist = sns.kdeplot(df['TotalCharges'][df["Churn"] == 0], color = bluish, shade = True)

ten_dist = sns.kdeplot(df['TotalCharges'][df["Churn"] == 1], color = orangish, shade= True)

ten_dist.legend(['Not Churn', 'Churn'])

ten_dist.set_xlabel('Total charges')

ten_dist.set_ylabel('Frequency')

plt.title('Distribution of total charges for churn and not churn customers');
plt.figure(figsize = [10,6])

sns.countplot(data = df, x = 'Contract', hue = 'Churn')

plt.legend(['Not Churn', 'Churn'])

plt.title('Contracts against churn and not churn customers', fontsize = 14);
plt.figure(figsize = [10,6])

sns.countplot(data = df, x = 'InternetService', hue = 'Churn')

plt.legend(['Not Churn', 'Churn'])

plt.title('Internet service against churn and not churn customers', fontsize = 14);
df = pd.get_dummies(data = df)

df.head()
corr = df.corr()

fig = plt.figure(figsize = (8, 8))

ax = fig.add_subplot(111)

p = ax.matshow(corr, vmin = -1, vmax = 1)

fig.colorbar(p)

ticks = np.arange(0, 27, 1) 

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(df.columns.to_list(), rotation = 90)

ax.set_yticklabels(df.columns.to_list());
df.corr()['Churn'].sort_values()
df.describe()
X = df.drop(["Churn"], axis = 1)

X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)

X.describe()
y = df['Churn'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print('Test: ', X_train.shape[0], ', ', y_train.shape[0], sep = '')

print('Train: ', X_test.shape[0],',', y_test.shape[0], sep = '')
lr_model = LogisticRegression()

lr_model.fit(X_train,y_train)

lr_train_acc = lr_model.score(X_train, y_train)

lr_test_acc = lr_model.score(X_test, y_test)

print('Logistic Regression')

print('Training accuracy:', lr_train_acc)

print('Testing accuracy:', lr_test_acc) 
svc_model = SVC(random_state = 1)

svc_model.fit(X_train, y_train)

svm_train_acc = svc_model.score(X_train,y_train)

svm_test_acc = svc_model.score(X_test,y_test)

print('SVM')

print('Training accuracy:', svm_train_acc)

print('Testing accuracy:', svm_test_acc)
plt.figure(figsize = (15, 6))

acc = []

acc_k = []

for k in range(1, 25):

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X_train, y_train)

    acc.append(knn.score(X_test,y_test))

    acc_k.append([knn.score(X_test,y_test), k])

    

plt.plot(range(1, 25), acc)

plt.xticks(np.arange(1, 26, 1))

plt.xlabel("Range")

plt.ylabel("Score")

plt.title('Finding k for KNN');
max(acc_ind)
knn_model = KNeighborsClassifier(n_neighbors = 9)

knn_model.fit(X_train, y_train)

knn_train_acc = knn_model.score(X_train, y_train)

knn_test_acc = knn_model.score(X_test, y_test)

print('KNN for k = 15')

print('Training accuracy:', knn_train_acc)

print('Testing accuracy:', knn_test_acc)
def scores(name, y, y_hat):

    acc = accuracy_score(y, y_hat)

    precision = precision_score(y, y_hat)

    recall = recall_score(y, y_hat)    

    f1 = f1_score(y, y_hat, average='weighted')

    

    print(name)    

    print('Accuracy:', acc)

    print('Precision: ', precision)                   

    print('Recall:', recall)

    print('F1_score:', f1)

    print()
scores("Logistic Regression",y_test, lr_model.predict(X_test))

scores("Support Vector Machine", y_test, svc_model.predict(X_test))

scores("K-Nearest Neighbors", y_test, knn_model.predict(X_test))
lr_matrix = confusion_matrix(y_test, lr_model.predict(X_test))

f, ax = plt.subplots(figsize = (8, 8))

sns.heatmap(lr_matrix, annot = True, color = "red", fmt = ".0f")

plt.xlabel("Predicted")

plt.ylabel("True")

plt.title("Confusion Matrix");