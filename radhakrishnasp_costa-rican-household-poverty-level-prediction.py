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
data = pd.read_csv('../input/costa-rican-household-poverty-prediction/train.csv')
data.head()
data.info()
for label, content in data.items():

    if pd.api.types.is_string_dtype(content):

        print(label)
df = data.copy()
print(df['Target'].value_counts())

df['Target'].value_counts().plot(kind='bar')
def preprocess_data(df):    

    # Fill numeric rows with the median

    df.drop('Id', axis=1)

    df.set_index('Id', inplace=True)

    for label, content in df.items():

        if pd.api.types.is_numeric_dtype(content):

            if pd.isnull(content).sum():

                #df[label+"_is_missing"] = pd.isnull(content)

                df[label] = content.fillna(content.median())

                

        # Turn categorical variables into numbers

        if not pd.api.types.is_numeric_dtype(content):

            #df[label+"_is_missing"] = pd.isnull(content)

            # We add the +1 because pandas encodes missing categories as -1

            df[label] = pd.Categorical(content).codes+1        

    

    return df

preprocess_data(df)
X = df.drop('Target', axis=1)

y = df['Target']
X
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier().fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print('R2 score is : {:.2f}'.format(accuracy_score(y_test, rf_pred)))

print('\n')

print("Classification Report : ")

print(classification_report(y_test,rf_pred))
from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier().fit(X_train, y_train)

etc_pred = etc.predict(X_test)

print('R2 score is : {:.2f}'.format(accuracy_score(y_test, etc_pred)))

print('\n')

print("Classification Report : ")

print(classification_report(y_test,etc_pred))
etc.feature_importances_
import seaborn as sns



# Helper function for plotting feature importance

def plot_features(columns, importances, n=10):

    df = (pd.DataFrame({"features": columns,

                        "feature_importance": importances})

          .sort_values("feature_importance", ascending=False)

          .reset_index(drop=True))

    

    sns.barplot(x="feature_importance",

                y="features",

                data=df[:n],

                orient="h")

plot_features(X_train.columns, etc.feature_importances_)
new_data = df[['meaneduc','SQBmeaned','hogar_nin','SQBhogar_nin','cielorazo',

               'qmobilephone','idhogar','overcrowding','r4t1','SQBdependency']]
new_data
X_train, X_test, y_train, y_test = train_test_split(new_data,y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
etc = ExtraTreesClassifier().fit(X_train, y_train)

etc_pred = etc.predict(X_test)

print('R2 score is : {:.2f}'.format(accuracy_score(y_test, etc_pred)))

print('\n')

print("Classification Report : ")

print(classification_report(y_test,etc_pred))
test = pd.read_csv('../input/costa-rican-household-poverty-prediction/test.csv')
test_df = test.copy()
test_df.info()
for label, content in test_df.items():

    if pd.api.types.is_string_dtype(content):

        print(label)
preprocess_data(test_df)
pred = rf.predict(test_df)

pred = pd.DataFrame(pred)

pred.to_csv('submission.csv')
pred
Id = test['Id']

Id = pd.DataFrame(Id)
Id
subs = pd.concat([id, pred], ignore_index=True, axis=1)
subs
subs.rename(columns={'0':'ID','1':'Target'}, inplace=True)
subs.columns = ['Id','Target']
subs.drop
s = subs.copy()
s.reset_index(drop=True, inplace=True)
s.set_index('Id', inplace=True)
subs = s
subs
subs.to_csv('submission.csv')