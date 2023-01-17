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
df = pd.read_csv("../input/adult.csv")

df.head()
# replace "?" to NaN

df.replace("?", np.nan, inplace = True)

df.head(5)
missing_data = df.isnull()

missing_data.sum()
missing_col = []

for column in missing_data.columns.values.tolist():

    if(missing_data[column].sum() > 0):

        print("Column: ",column)

        print("Missing Data: {} ({:.2f}%)".format(missing_data[column].sum(), (missing_data[column].sum() * 100/ len(df))))

        print("Data Type: ",df[column].dtypes)

        print("")

        missing_col.append(column)
import matplotlib.pyplot as plt

%matplotlib inline
fig1 = plt.figure(figsize=(18,5))

i = 0

for column in missing_col:

    bad = missing_data[column].sum()

    good = len(df) - missing_data[column].sum()

    x = [bad, good]

    labels = ["Missing Data", "Good Data"]

    explode = (0.1, 0)

    i = i+1

    ax = fig1.add_subplot(1,3,i)

    ax.pie(x,explode = explode, labels = labels, shadow = True,autopct='%1.1f%%', colors = ['#ff6666', '#99ff99'],rotatelabels = True, textprops={'fontsize': 18})

    centre_circle = plt.Circle((0,0),0.4,color='black', fc='white',linewidth=0)

    fig = plt.gcf()

    fig.gca().add_artist(centre_circle)

    ax.axis('equal')

    ax.set_title(column, fontsize = 25)

plt.tight_layout()

plt.show()

# Calculate Mode

workclass_mode = df['workclass'].value_counts().idxmax()

occupation_mode = df['occupation'].value_counts().idxmax()

native_country_mode = df['native.country'].value_counts().idxmax()
print("Mode of workclass: ",workclass_mode)

print("Mode of Occupation: ",occupation_mode)

print("Mode of natice.country: ",native_country_mode)
df_manual = df
#replace the missing categorical values by the most frequent value

df_manual["workclass"].replace(np.nan, workclass_mode, inplace = True)

df_manual["occupation"].replace(np.nan, occupation_mode, inplace = True)

df_manual["native.country"].replace(np.nan, native_country_mode, inplace = True)
df_manual.isnull().sum()
count = 0

for column in df_manual.columns.values.tolist():

    if df_manual[column].dtype == 'object':

        print("Column Name: ",column)

        print("Data Type: ", df_manual[column].dtypes)

        print("")

        count = count + 1

print("Count : ",count)

dummy = pd.get_dummies(df_manual["workclass"])

dummy.head()
#Rename column names

dummy.rename(columns={'Federal-gov':'work-Federal-gov', 

                      'Local-gov':'work-Local-gov',

                      'Private': 'work-Private',

                      'Self-emp-inc': 'work-Self-emp-inc',

                      'Self-emp-not-inc': 'Self-emp-not-inc',

                      'State-gov': 'work-State-gov',

                      'Without-pay' : 'work-Without-pay'}, inplace=True)

dummy.head()
dummy.drop("Never-worked", axis = 1, inplace=True)

dummy.head()
# merge data frame "df" and "dummy" 

df_manual = pd.concat([df_manual, dummy], axis=1)



# drop original column "workplace" from "df"

df_manual.drop("workclass", axis = 1, inplace=True)

df_manual.head()
dummy = pd.get_dummies(df_manual["education"])

dummy.head()
dummy.drop("Some-college", axis = 1, inplace=True)

dummy.head()
# merge data frame "df" and "dummy_variable_1" 

df_manual = pd.concat([df_manual, dummy], axis=1)



# drop original column "fuel-type" from "df"

df_manual.drop("education", axis = 1, inplace=True)

df_manual.head()
dummy = pd.get_dummies(df_manual["marital.status"])

dummy.head()
dummy.drop("Never-married", axis = 1, inplace=True)

# merge data frame "df" and "dummy_variable_1" 

df_manual = pd.concat([df_manual, dummy], axis=1)



# drop original column "fuel-type" from "df"

df_manual.drop("marital.status", axis = 1, inplace=True)

df_manual.head()
dummy = pd.get_dummies(df_manual["occupation"])

dummy.head()
dummy.drop("Other-service", axis = 1, inplace=True)

# merge data frame "df" and "dummy_variable_1" 

df_manual = pd.concat([df_manual, dummy], axis=1)



# drop original column "fuel-type" from "df"

df_manual.drop("occupation", axis = 1, inplace=True)

df_manual.head()
dummy = pd.get_dummies(df_manual["relationship"])

dummy.head()
dummy.drop("Other-relative", axis = 1, inplace=True)

# merge data frame "df" and "dummy_variable_1" 

df_manual = pd.concat([df_manual, dummy], axis=1)



# drop original column "fuel-type" from "df"

df_manual.drop("relationship", axis = 1, inplace=True)

df_manual.head()
dummy = pd.get_dummies(df_manual["race"])

dummy.head()
dummy.drop("Other", axis = 1, inplace=True)

# merge data frame "df" and "dummy_variable_1" 

df_manual = pd.concat([df_manual, dummy], axis=1)



# drop original column "fuel-type" from "df"

df_manual.drop("race", axis = 1, inplace=True)

df_manual.head()
dummy = pd.get_dummies(df_manual["sex"])

dummy.head()
dummy.drop("Male", axis = 1, inplace=True)

dummy.rename(columns={ 'Female' : 'Female/Male'}, inplace = True)

# merge data frame "df" and "dummy_variable_1" 

df_manual = pd.concat([df_manual, dummy], axis=1)

# drop original column "fuel-type" from "df"

df_manual.drop("sex", axis = 1, inplace=True)

df_manual.head()
dummy = pd.get_dummies(df_manual["native.country"])

dummy.head()
# merge data frame "df" and "dummy_variable_1" 

df_manual = pd.concat([df_manual, dummy], axis=1)

# drop original column "fuel-type" from "df"

df_manual.drop("native.country", axis = 1, inplace=True)

df_manual.head()
dummy = pd.get_dummies(df_manual["income"])

dummy.head()
dummy.rename(columns={ '>50K' : 'Income > 50K'}, inplace = True)

dummy.drop('<=50K', axis = 1, inplace = True)
df_manual = pd.concat([df_manual, dummy], axis=1)

df_manual.drop("income", axis = 1, inplace=True)

df_manual.head()
df_manual.dtypes
X = df_manual.iloc[:,:-1].values

y = df_manual["Income > 50K"].iloc[:].values
df_manual.describe(include = 'all')
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)
import seaborn as sns

sns.countplot(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from xgboost import XGBClassifier

classifier = XGBClassifier(learning_rate = 0.1, n_estimators = 100)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Model Accuracy = {:.2f}%".format(accuracies.mean()* 100))
from mlxtend.plotting import plot_confusion_matrix

fig, ax = plot_confusion_matrix(conf_mat=cm,

                                show_absolute=True,

                                show_normed=True,

                                colorbar=True,

                               cmap = 'Dark2')

plt.show()
# import lightgbm as lgb

# d_train = lgb.Dataset(X_train, label = y_train)

# params = {}

# params['learning_rate'] = 0.01

# params['boosting_type'] = 'gbdt'

# params['objective'] = 'binary'

# params['metric'] = 'binary_logloss'

# params['sub_feature'] = 0.5

# params['num_leaves'] = 10

# params['min_data'] = 50

# params['max_depth'] = 10

# clf = lgb.train({},train_set = d_train)
# #Prediction

# y_pred=clf.predict(X_test)
# for i in range(0,y_pred.shape[0]):

#     if y_pred[i]>=0.5:       # setting threshold to .5

#        y_pred[i]=1

#     else:  

#        y_pred[i]=0
missing_col = []

for column in missing_data.columns.values.tolist():

    if(missing_data[column].sum() > 0):

        print("Column: ",column)

        print("Missing Data: {} ({:.2f}%)".format(missing_data[column].sum(), (missing_data[column].sum() * 100/ len(df))))

        print("Data Type: ",df[column].dtypes)

        print("")

        missing_col.append(column)
df_dl_method = pd.read_csv("../input/adult.csv")

# replace "?" to NaN

df_dl_method.replace("?", np.nan, inplace = True)

df_dl_method.head(5)
df_without_null = df_dl_method.dropna()
# reset index, because we droped two rows

df_without_null.reset_index(drop = True, inplace = True)

df_without_null.drop(["occupation", "native.country"], axis = 1, inplace = True)

df_without_null.head()
def encoder(dataframe, col, drop_dummy_trap = ""):

    dummy = pd.get_dummies(dataframe[col])

    dataframe = pd.concat([dataframe, dummy], axis=1)

    if(len(drop_dummy_trap) != 0):

        # drop original column "fuel-type" from "df"

        dataframe.drop(drop_dummy_trap, axis = 1, inplace=True)

    dataframe.drop(col, axis = 1, inplace = True)

    return dataframe
df_without_null = encoder(dataframe = df_without_null, col = "education", drop_dummy_trap = "Some-college")

#df_without_null = encoder(df = df_without_null, col = "occupation", drop_dummy_trap = "Other-service")

df_without_null = encoder(dataframe = df_without_null, col = "marital.status", drop_dummy_trap = "Never-married")

df_without_null = encoder(dataframe = df_without_null, col = "relationship", drop_dummy_trap = "Other-relative")

df_without_null = encoder(dataframe = df_without_null, col = "race", drop_dummy_trap = "Other")

df_without_null = encoder(dataframe = df_without_null, col = "sex", drop_dummy_trap = "Male")

df_without_null = encoder(dataframe = df_without_null, col = "income", drop_dummy_trap = "<=50K")
X_train = df_without_null.drop(["workclass"], axis = 1).iloc[:].values

y_train = df_without_null.iloc[:,1].values
df_test=df_dl_method.loc[pd.isnull(df_dl_method["workclass"])]

df_test.head()
df_test.drop(["workclass", "occupation", "native.country"], axis = 1, inplace = True)

df_test.head()
df_test = encoder(df_test, col = "education", drop_dummy_trap = "Some-college")

#df_test = encoder(df_test, col = "occupation", drop_dummy_trap = "Other-service")

df_test = encoder(df_test, col = "marital.status", drop_dummy_trap = "Never-married")

df_test = encoder(df_test, col = "relationship", drop_dummy_trap = "Other-relative")

df_test = encoder(df_test, col = "race", drop_dummy_trap = "Other")

df_test = encoder(df_test, col = "sex", drop_dummy_trap = "Male")

#df_test = encoder(df_test, col = "native.country", drop_dummy_trap = "")

df_test = encoder(df_test, col = "income", drop_dummy_trap = "<=50K")
X_test = df_test.iloc[:].values
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
y_train[:] = labelencoder.fit_transform(y_train[:])

y_train
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from xgboost import XGBClassifier

classifier = XGBClassifier(learning_rate = 0.1, n_estimators = 100)

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)
decode = dict(zip(labelencoder.transform(labelencoder.classes_), labelencoder.classes_))
for i in range(0,y_pred.shape[0]):

    y_pred[i] = decode[y_pred[i]]

y_pred    
df_dl_method.head()
fill = pd.DataFrame(y_pred, columns = ["workclass"])
j = 0

for i in range(0, df_dl_method.shape[0]):

    if(pd.isnull(df_dl_method.workclass[i])):

        df_dl_method.workclass[i] = y_pred[j]

        j = j+1
df_dl_method.workclass.isnull().sum()
df_viz = pd.read_csv("../input/adult.csv")

# replace "?" to NaN

df_viz.replace("?", np.nan, inplace = True)
fig1 = plt.figure(figsize=(20,5))

i = 1

column = "workclass"



bad = df_viz[column].isnull().sum()

good = len(df_viz) - df_viz[column].isnull().sum()

x = [bad, good]

labels = ["Missing Data", "Good Data"]

explode = (0.1, 0)

ax = fig1.add_subplot(1,2,i)

ax.pie(x,explode = explode, labels = labels, shadow = True,autopct='%1.1f%%', colors = ['#ff6666', '#99ff99'],rotatelabels = True, textprops={'fontsize': 18})

centre_circle = plt.Circle((0,0),0.4,color='black', fc='white',linewidth=0)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

ax.axis('equal')

ax.set_title(column + "(before)", fontsize = 25) 

i = i+1



bad = df_dl_method[column].isnull().sum()

good = len(df) - df_dl_method[column].isnull().sum()

x = [bad, good]

labels = ["Missing Data", "Good Data"]

explode = (0.1, 0)

ax = fig1.add_subplot(1,2,i)

ax.pie(x,explode = explode, labels = labels, shadow = True,autopct='%1.1f%%', colors = ['#ff6666', '#99ff99'],rotatelabels = True, textprops={'fontsize': 18})

centre_circle = plt.Circle((0,0),0.4,color='black', fc='white',linewidth=0)

fig = plt.gcf()

fig.gca().add_artist(centre_circle)

ax.axis('equal')

ax.set_title(column + "(after)", fontsize = 25)





plt.tight_layout()

plt.show()