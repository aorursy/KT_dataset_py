import os

os.chdir('/kaggle/input/red-wine-quality-cortez-et-al-2009/')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 



from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import cross_val_score



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier





%matplotlib inline

plt.style.use('ggplot')



import warnings

warnings.filterwarnings("ignore")
df_red = pd.read_csv("winequality-red.csv")
df = df_red

df.head()
df.info()
df['quality'].value_counts(sort = False)
df['quality'].hist()
def gen_labels(df):

    labels = ['bad', 'ok', 'good']

    

    if 1 <= df.loc['quality'] <= 5:

        label = labels[0]

    elif 5 < df.loc['quality'] < 7:

        label = labels[1]

    elif 7 <= df.loc['quality']<= 10:

        label = labels[2]

        

    return label
df['label'] = df.apply(gen_labels, axis = 1)



df['label'] = df['label'].astype('category')



df['label'].value_counts()

#### Taking too long

# df['label'].hist()
df.columns
df.groupby('label').mean()
sns.catplot(x='label', y='pH', hue='quality', data=df, kind = 'swarm')
sns.catplot(x='label', y='fixed acidity', hue='quality', data=df, kind = 'swarm')
def scale_and_split(df, test_sizre=0.3):

    

    target = df[['label']]

    features = df.drop(['label', 'quality'], axis = 1)

    labels = list(target.label.unique())

    

    scaler = StandardScaler()

    features = scaler.fit_transform(features)

    

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 42)

    

    return X_train, X_test, y_train, y_test, labels
def evaluate_model(model, df):

    

    X_train, X_test, y_train, y_test, labels = scale_and_split(df)

    

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    print('Cross validation score - ', scores.mean()*100)

    

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)



    accuracy = accuracy_score(y_test, y_pred) 

    print('Test accuracy - ',accuracy*100)

    print('Confusion Matrix -\n', confusion_matrix(y_test, y_pred, labels))
lr = LogisticRegression()

dt = DecisionTreeClassifier(criterion='gini', max_depth=12, random_state=42)

rc = RandomForestClassifier(n_estimators=100, max_depth=12 ,random_state=42)



print('\nEvaluation results - Logistic Regression')

evaluate_model(lr, df)



print('\nEvaluation results - Decision Tree Classifier')

evaluate_model(dt, df)



print('\nEvaluation results - Random Forest Classifier')

evaluate_model(rc, df)

from sklearn.utils import resample
df_majority = df[df['label']!='good']

df_minority = df[df['label']=='good']

 

df_minority_upsampled = resample(df_minority, replace=True, n_samples=700, random_state=42)



df_upsampled = pd.concat([df_majority, df_minority_upsampled])



df_upsampled['label'].value_counts()
lr = LogisticRegression()

dt = DecisionTreeClassifier(criterion='gini', max_depth=12, random_state=42)

rc = RandomForestClassifier(n_estimators=250, max_depth=12 ,random_state=42)



print('\nEvaluation results on upsampled data - Logistic Regression')

evaluate_model(lr, df_upsampled)



print('\nEvaluation results on upsampled data - Decision Tree Classifier')

evaluate_model(dt, df_upsampled)



print('\nEvaluation results on upsampled data - Random Forest Classifier')

evaluate_model(rc, df_upsampled)