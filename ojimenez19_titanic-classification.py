# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import plot_confusion_matrix

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer

import seaborn as sns

from sklearn.pipeline import make_pipeline

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn import svm

import plotly.express as px

from sklearn.preprocessing import PowerTransformer

from sklearn.model_selection import GridSearchCV

# Input data files are available ifrom sklearn.preprocessing import QuantileTransformern the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def show_confusion_matrix(model,x,y):

    disp = plot_confusion_matrix(model, x, y,

                             cmap=plt.cm.Blues,

                             normalize=None)

    plt.show()

    

def get_pipeline(model,preprocessor):

    return Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', model)

                             ])



def get_data(path):

    data = pd.read_csv(path)

    data = feature_engineering(data)

    return data



def feature_engineering(data):

    encoder = LabelEncoder()

    imputer = SimpleImputer()

    age_ranges = [1,19,30,50,100]

    labels = ['minor','young','old','very_old']

    cols_to_remove = ['Ticket','PassengerId','Age_Range','Embarked','Age','Sex']

        

    data["Name"] = data["Name"].str.split(',').str[1]

    data["Name"] = data["Name"].str.split('.').str[0]

    data["Name"] = data["Name"].str.strip()

    

    

    data['Age_Range'] = pd.cut(data['Age'],bins = age_ranges, labels = labels)

        

    data['Minor_Male'] =  (data['Age_Range']=='minor') & (data['Sex']=='male')

    data['Minor_Female'] =  (data['Age_Range']=='minor') & (data['Sex']=='female')

        

    data['Young_Male'] =  (data['Age_Range']=='young') & (data['Sex']=='male')

    data['Young_Female'] =  (data['Age_Range']=='young') & (data['Sex']=='female')

    

    data['Old_Male'] =  (data['Age_Range']=='old') & (data['Sex']=='male')

    data['Old_Female'] =  (data['Age_Range']=='old') & (data['Sex']=='female')

        

    data['Very_Old_Male'] =  (data['Age_Range']=='very_old') & (data['Sex']=='male')

    data['Very_Old_Female'] =  (data['Age_Range']=='very_old') & (data['Sex']=='female')



    data["Name_Age_Range"] = data['Age_Range'].str.cat(data['Name'].values.astype(str), sep='_')

    

    

    data["Name"] = encoder.fit_transform(data['Name'])

    data["Name_Age_Range"] = encoder.fit_transform(data['Name_Age_Range'].astype(str))

    data["Embarked"] = encoder.fit_transform(data['Embarked'].astype(str))

    data["Cabin"] = encoder.fit_transform(data['Cabin'].astype(str))

    data = data.drop(cols_to_remove,axis=1)

    

    data = data.replace(True, 1)

    data = data.replace(False, 0)

     

    cols = data.columns

        

    fig = px.histogram(data, x="Fare",labels={"Fare":"Fare unnormalized"})

    fig.show()

     

    fare = data['Fare']

       

    pt = PowerTransformer()

    fare_transformed = pt.fit_transform(np.array(fare).reshape(-1,1))



    data.drop(['Fare'],axis=1)

    data['Fare']= imputer.fit_transform(fare_transformed)

    

    fig_normalized = px.histogram(data, x="Fare",labels={"Fare":"Fare normalized"})

    fig_normalized.show()

        

    return data
train_data = get_data('/kaggle/input/titanic/train.csv')

train_data.head()
y = train_data['Survived']

X = train_data.drop(['Survived'],axis=1)

X
model = RandomForestClassifier(min_samples_leaf=10)



model.fit(X,y)



scores = cross_val_score(model, X, y, cv=5)

print('Scores:',scores)

print('Score:',scores.mean())

show_confusion_matrix(model,X,y)
#/kaggle/input/titanic/gender_submission.csv

#/kaggle/input/titanic/test.csv

test_data = get_data('/kaggle/input/titanic/test.csv')

test_data.head()
test_predictions = model.predict(test_data)

test_predictions
# SAVING OUTPUT TO CSV

#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_predictions})

output = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

output['Survived']=test_predictions

output.to_csv('/kaggle/working/titanic_submission.csv', index=False)