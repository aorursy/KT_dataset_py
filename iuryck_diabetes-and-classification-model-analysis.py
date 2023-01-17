# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas_profiling

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings("ignore")



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/diabetes/diabetes.csv')

df.tail()
print('____NULL____')

for column in df.columns :

    print(column,' - ', df[f'{column}'].isnull().sum())
print('____ZERO____')

for column in df.columns :

    percent =  df[df[f'{column}']==0].shape[0]/df.shape[0]

    print(column,' - ',df[df[f'{column}']==0].shape[0], end = ' - ')

    print( "%.2f" % percent, end='%\n')

    
df.describe()
import seaborn as sns

import plotly.express as px



sns.heatmap(df.corr())
px.histogram(df, x='Outcome')
#getting copy of dataframe

fixdf = df.copy()

#getting a portion of rows to remove from copy dataframe

indexdrop = fixdf.index[fixdf['Outcome']==0][:230]

#dropping the rows to make the classes equal

fixdf = fixdf.drop(indexdrop)

px.histogram(fixdf, x='Outcome')



for column in fixdf.columns:

    fig =  px.violin(fixdf, y=f'{column}',color='Outcome',violinmode='overlay')

    fig.show()
import sklearn

from sklearn import svm

from sklearn.feature_selection import RFECV

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import f1_score







X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

svc = svm.SVC(max_iter=2000)

svc.fit(X_train,y_train)

print('Normalized SVC score: ', svc.score(X_test, y_test))

print('Normalized SVC f1 score:  ',f1_score(y_test, svc.predict(X_test)))



print('____________________________________')

print('__________Selected Features_________')

print('____________________________________')



X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

svc = svm.LinearSVC(max_iter=2000)

select = RFECV(svc,scoring='accuracy',min_features_to_select=1)

select.fit(X_train,y_train)

i=0

for column in df.drop('Outcome', axis=1):

    print(column, ' - ', select.ranking_[i], '\t\t selected = ', select.get_support()[i])

    i=i+1



    

fig = px.line(x = range(1, len(select.grid_scores_) + 1), y = select.grid_scores_)

fig.show()





print('______________________________')

print('Normalized Selected SVC score : ',select.score(X_test, y_test))

print('Normalized f1 score:  ',f1_score(y_test, select.predict(X_test)))







X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

lg = LogisticRegression()

lg.fit(X_train,y_train)





print('______________________________')

print('Normalized LG score : ',lg.score(X_test, y_test))

print('Normalized LG f1 score:  ',f1_score(y_test, lg.predict(X_test)))
from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import plot_precision_recall_curve



import matplotlib.pyplot as plt





X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

svc = svm.LinearSVC(max_iter=2000)

select = RFECV(svc,scoring='accuracy',min_features_to_select=1)

select.fit(X_train,y_train)

disp = plot_precision_recall_curve(select, X_test, y_test)



X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

svc = svm.SVC(max_iter=2000)

svc.fit(X_train,y_train)



disp = plot_precision_recall_curve(svc, X_test, y_test)







disp = plot_precision_recall_curve(lg, X_test, y_test)





from sklearn.model_selection import learning_curve

import plotly.graph_objects as go



fig = go.Figure()



train_steps=[c for c in range(1,613)]

train_steps = train_steps[0::10]



X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

svc = svm.SVC(max_iter=2000)



train_sizes, train_scores, valid_scores = learning_curve(svc, X, y, train_sizes=train_steps, cv=5, shuffle=True)



index = [c for c in range(0,len(train_scores))]

train_scores = train_scores.mean(axis = 1)

valid_scores = valid_scores.mean(axis =1)



fig.add_trace(go.Scatter(x=index, y=train_scores,name='Training'))

fig.add_trace(go.Scatter(x=index, y=valid_scores ,name='Validation'))

fig.update_layout(title='SVC')

fig.show()



X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

svc = svm.LinearSVC(max_iter=2000)

select = RFECV(svc,scoring='accuracy',min_features_to_select=1)



train_sizes, train_scores, valid_scores = learning_curve(select, X, y,train_sizes=train_steps, cv=5, shuffle=True)



index = [c for c in range(0,len(train_scores))]

train_scores = train_scores.mean(axis = 1)

valid_scores = valid_scores.mean(axis =1)



fig = go.Figure()

fig.add_trace(go.Scatter(x=index, y=train_scores,name='Training'))

fig.add_trace(go.Scatter(x=index, y=valid_scores ,name='Validation'))

fig.update_layout(title='RFECV SVC')

fig.show()





X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

lg = LogisticRegression()



train_sizes, train_scores, valid_scores = learning_curve(lg, X, y, train_sizes=train_steps, cv=5, shuffle=True)



index = [c for c in range(0,len(train_scores))]

train_scores = train_scores.mean(axis = 1)

valid_scores = valid_scores.mean(axis =1)





fig = go.Figure()

fig.add_trace(go.Scatter(x=index, y=train_scores,name='Training'))

fig.add_trace(go.Scatter(x=index, y=valid_scores ,name='Validation'))

fig.update_layout(title='Logistic Regression')

fig.show()



from numpy import mean

import seaborn as sns



plt.subplots(figsize=(20,7))

sns.barplot(x="Age", y="Insulin", data=df, estimator=mean)

plt.show()
fix_columns = ['Insulin','SkinThickness','BloodPressure']

for fix_column in fix_columns:





    for age in df['Age'].unique():

        small = df.copy()[(df['Age'] > age-5) & (df['Age'] < age+5)]

        try:

            sample = df.loc[df['Age']==age]

            sample = sample.loc[sample[fix_column] == 0]

        except:

            continue



        for row in range(0,sample.shape[0]):



            if sample.iloc[row]['Outcome'] ==0:



                df.loc[sample.iloc[row].name, fix_column] = small[small['Outcome']==0][fix_column].mean()



            else: df.loc[sample.iloc[row].name, fix_column] = small[small['Outcome']==1][fix_column].mean()



print('____ZERO____')

for column in df.columns :

    percent =  df[df[f'{column}']==0].shape[0]/df.shape[0]

    print(column,' - ',df[df[f'{column}']==0].shape[0], end = ' - ')

    print( "%.2f" % percent, end='%\n')
plt.subplots(figsize=(20,7))

sns.barplot(x="Age", y="Insulin", data=df, estimator=mean)

plt.show()
for column in df.columns:

    fig =  px.violin(df, y=f'{column}',color='Outcome',violinmode='overlay')

    fig.show()




X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

svc = svm.SVC(max_iter=2000)

svc.fit(X_train,y_train)

print('Normalized SVC score: ', svc.score(X_test, y_test))

print('Normalized SVC f1 score:  ',f1_score(y_test, svc.predict(X_test)))



print('____________________________________')

print('__________Selected Features_________')

print('____________________________________')



X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

svc = svm.LinearSVC(max_iter=2000)

select = RFECV(svc,scoring='accuracy',min_features_to_select=1)

select.fit(X_train,y_train)

i=0

for column in df.drop('Outcome', axis=1):

    print(column, ' - ', select.ranking_[i], '\t\t selected = ', select.get_support()[i])

    i=i+1



    

fig = px.line(x = range(1, len(select.grid_scores_) + 1), y = select.grid_scores_)

fig.show()





print('______________________________')

print('Normalized Selected SVC score : ',select.score(X_test, y_test))

print('Normalized f1 score:  ',f1_score(y_test, select.predict(X_test)))







X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

lg = LogisticRegression()

lg.fit(X_train,y_train)





print('______________________________')

print('Normalized LG score : ',lg.score(X_test, y_test))

print('Normalized LG f1 score:  ',f1_score(y_test, lg.predict(X_test)))
X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

svc = svm.LinearSVC(max_iter=2000)

select = RFECV(svc,scoring='accuracy',min_features_to_select=1)

select.fit(X_train,y_train)

disp = plot_precision_recall_curve(select, X_test, y_test)



X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

svc = svm.SVC(max_iter=2000)

svc.fit(X_train,y_train)



disp = plot_precision_recall_curve(svc, X_test, y_test)







disp = plot_precision_recall_curve(lg, X_test, y_test)





from sklearn.model_selection import learning_curve

import plotly.graph_objects as go



fig = go.Figure()



train_steps=[c for c in range(1,613)]

train_steps = train_steps[0::10]



X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

svc = svm.SVC(max_iter=2000)



train_sizes, train_scores, valid_scores = learning_curve(svc, X, y, train_sizes=train_steps, cv=5, shuffle=True)



index = [c for c in range(0,len(train_scores))]

train_scores = train_scores.mean(axis = 1)

valid_scores = valid_scores.mean(axis =1)



fig.add_trace(go.Scatter(x=index, y=train_scores,name='Training'))

fig.add_trace(go.Scatter(x=index, y=valid_scores ,name='Validation'))

fig.update_layout(title='SVC')

fig.show()



X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

svc = svm.LinearSVC(max_iter=2000)

select = RFECV(svc,scoring='accuracy',min_features_to_select=1)



train_sizes, train_scores, valid_scores = learning_curve(select, X, y,train_sizes=train_steps, cv=5, shuffle=True)



index = [c for c in range(0,len(train_scores))]

train_scores = train_scores.mean(axis = 1)

valid_scores = valid_scores.mean(axis =1)



fig = go.Figure()

fig.add_trace(go.Scatter(x=index, y=train_scores,name='Training'))

fig.add_trace(go.Scatter(x=index, y=valid_scores ,name='Validation'))

fig.update_layout(title='RFECV SVC')

fig.show()





X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

lg = LogisticRegression()



train_sizes, train_scores, valid_scores = learning_curve(lg, X, y, train_sizes=train_steps, cv=5, shuffle=True)



index = [c for c in range(0,len(train_scores))]

train_scores = train_scores.mean(axis = 1)

valid_scores = valid_scores.mean(axis =1)





fig = go.Figure()

fig.add_trace(go.Scatter(x=index, y=train_scores,name='Training'))

fig.add_trace(go.Scatter(x=index, y=valid_scores ,name='Validation'))

fig.update_layout(title='Logistic Regression')

fig.show()



from sklearn.metrics import precision_recall_curve

from numpy import argmax



X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

svc = svm.SVC(max_iter=2000, probability=True)

svc.fit(X_train,y_train)



yhat = svc.predict_proba(X_test)

# keep probabilities for the positive outcome only

yhat = yhat[:, 1]

# calculate pr-curve

precision, recall, thresholds = precision_recall_curve(y_test, yhat)





disp = plot_precision_recall_curve(svc, X_test, y_test)



fscore = (2 * precision * recall) / (precision + recall)

# locate the index of the largest f score

ix = argmax(fscore)



print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(svc, X_test, y_test)
from sklearn.metrics import confusion_matrix



thresh = thresholds[ix]

probs = svc.predict_proba(X_test)[:, 1]

def prediction(prob):

    if prob >= thresh: return 1

    else: return 0





    

predic = [(prediction(c)) for c in probs]



data = confusion_matrix(y_test, predic)

df_cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, cmap="Blues", annot=True, fmt='g')# font size
X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

svc = svm.LinearSVC(max_iter=2000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=22)

svc.fit(X_train,y_train)



columns = [c for c in df.drop('Outcome', axis=1).columns]







figw = px.histogram(x=columns, y=svc.coef_.reshape(svc.coef_.shape[1])).update_xaxes(categoryorder="total descending")



figw.show()

from sklearn.inspection import permutation_importance



X = np.array(df.drop('Outcome', axis=1))

scaler = StandardScaler()

X = scaler.fit_transform(X)

y = np.array(df['Outcome'])

svc = svm.SVC(max_iter=2000, probability=True)

svc.fit(X_train,y_train)



r = permutation_importance(svc, X_test, y_test,

                           n_repeats=30,

                            random_state=0)



for i in r.importances_mean.argsort()[::-1]:

   

    print(f"{columns[i]:<8} "

              f"{r.importances_mean[i]:.3f}"

             f" +/- {r.importances_std[i]:.3f}")



fig3=px.histogram(y=r.importances_mean, x=columns).update_xaxes(categoryorder="total descending")

fig3.show()