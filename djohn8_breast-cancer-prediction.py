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



        

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.express as px

import plotly.graph_objects as go

import plotly.io as pio

from plotly.subplots import make_subplots



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

display(df.head())

display(df.info())
# columns to drop

df.columns[0], df.columns[-1]

df.drop(columns=['id', 'Unnamed: 32'], axis=1, inplace=True)

# class lables

df.diagnosis.value_counts()
# use labelencoder to convert the labels to '1' and '0'

# from sklearn import preprocessing

# le = preprocessing.LabelEncoder()

# le.fit_transform(df['diagnosis'])



# change the labels to '1' and '0'

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
px.histogram(df, x= 'diagnosis', color='diagnosis', hover_data=df.columns)
df.head()
display(df.columns)



# feature to examine for variations in either clases

examine = df.columns[1:11]

display(examine)
pio.templates
pio.templates.default='plotly_white'
dfM = df[df['diagnosis']==1]

dfB = df[df['diagnosis']==0]



for i in examine:

    fig = go.Figure()

    fig.add_trace(go.Histogram(

                x = dfM['radius_mean'],

                name='malignant',

                marker_color='red',

                xbins=dict(size=0.5)

                ))

    fig.add_trace(go.Histogram(

                x = dfB['radius_mean'],

                name='benign',

                marker_color='blue',

                xbins=dict(size=0.5)

                ))

    fig.update_layout(barmode='overlay', title_text='Distribution of feature: '+str(i) + ' for malignant and benign cases')

#reduce opacity

    fig.update_traces(opacity=0.75)

    fig.show()
from sklearn.model_selection import train_test_split

# 70% train, 30% test

df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
examine.tolist()
from sklearn.linear_model import LogisticRegression

from sklearn import linear_model

#logreg = LogisticRegression()

logreg = linear_model.LogisticRegressionCV(cv=10,verbose=1,n_jobs=-1,scoring='accuracy',solver='lbfgs')



features=examine.tolist()

to_predict= 'diagnosis'



logreg.fit(df_train[features], df_train['diagnosis'])

prediction = logreg.predict(df_test[features])

prediction



from sklearn import metrics

metrics.accuracy_score(prediction, df_test['diagnosis'])

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score



def classification_models(model, df, df_train, df_test, cv=7, features=None):

    if features==None:

        model.fit(df_train[examine], df_train['diagnosis'])

        # prediction on the test

        prediction = model.predict(df_test[examine])

    else:

        model.fit(df_train[features], df_train['diagnosis'])

        # prediction on the test

        prediction = model.predict(df_test[features])

    

    # prediction on the test

    #prediction = model.predict(df_test[examine])

    

    # accuracy

    accuracy = accuracy_score(prediction, df_test['diagnosis'])

    print('Accuracy for model: {} is {}'.format(model, accuracy))

    

    # cross validation

    score= cross_val_score(estimator=model, X = df[examine], y=df['diagnosis'], cv=7, n_jobs=-1)

    for i in range(len(score)):

        print('Cross_validation score for for fold:{} is {}'.format(i,round(score[i],2)))
#from sklearn.exceptions import ConvergenceWarning

logreg = LogisticRegression()

classification_models(logreg, df, df_train, df_test, cv=7)
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()



classification_models(dtc, df, df_train, df_test, cv=7)

# for depth in range(3,7):

#     dtc = DecisionTreeClassifier(max_depth=depth)

#     classification_models(dtc, df, df_train, df_test, cv=7)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=150, n_jobs=-1, min_samples_split=10)



classification_models(rf, df, df_train, df_test, cv=7)
#rf.feature_importances_, features

imp_features =pd.Series(rf.feature_importances_, index= features).sort_values(ascending=False)

imp_features
#top 5 features

display(imp_features.index[:5].tolist())

classification_models(rf, df, df_train, df_test, cv=7, features=imp_features.index[:5].tolist())
import xgboost as xgb

xgb_class = xgb.XGBClassifier()

classification_models(xgb_class, df, df_train, df_test, cv=7, features=None)