import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, confusion_matrix

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import VotingClassifier

from sklearn.svm import SVC

import math



plt.style.use('seaborn')

seed = 6467

np.random.seed(seed)



%matplotlib inline
df = pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
df.head()
df.drop(columns = ['PassengerId','Firstname','Lastname'], inplace=True)

len(df)
df.isnull().sum()
df.nunique()
fig = px.histogram(df, x="Age", color="Sex", marginal="box",

                   hover_data=df.columns, title="Age Distribution accross gender")

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Survived', y=df[df['Survived']==1]['Country'].value_counts().values, x=df[df['Survived']==1]['Country'].value_counts().index),

    go.Bar(name='Not Survived', y=df[df['Survived']==0]['Country'].value_counts().values, x=df[df['Survived']==0]['Country'].value_counts().index)

])

# Change the bar mode

fig.update_layout(barmode='group', title_text = 'Survival count accross countries')

fig.show()
fig = px.sunburst(df, path=['Category', 'Survived'], title='Survival count accross category')

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Survived', y=df[df['Survived']==1]['Sex'].value_counts().values, x=df[df['Survived']==1]['Sex'].value_counts().index),

    go.Bar(name='Not Survived', y=df[df['Survived']==0]['Sex'].value_counts().values, x=df[df['Survived']==0]['Sex'].value_counts().index)

])

# Change the bar mode

fig.update_layout(barmode='group', title_text='Survival count accross gender')

fig.show()
fig = px.pie(df, names='Survived', title='Survival class distribution')

fig.show()
features = ['Country','Sex','Category']

df[features] = df[features].apply(preprocessing.LabelEncoder().fit_transform)
df.head()
corr = df.corr()

fig = px.imshow(corr)

fig.show()
X = df.drop(columns=['Survived'])

y = df['Survived']

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=seed)
rf_model = RandomForestClassifier(n_estimators=10, class_weight={0:1, 1:5})

gb_model = GradientBoostingClassifier(n_estimators=10)

svm_model = SVC(class_weight={0:1, 1:6},probability=True)

logreg_model = LogisticRegression(class_weight={0:1, 1:5})

ensemble = VotingClassifier([('rf',rf_model),('gb',gb_model),('svc',svm_model),('lg',logreg_model)], voting='soft')



models = {'Random Forest':rf_model, 'Gradient Boosting' :gb_model, 'SVC' :svm_model, 'Logistic Regression': logreg_model, 'Ensemble': ensemble}

names = ['Random Forest','Gradient Boosting','SVC','Logistic Regression','Ensemble']

n_repeats = 10

rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=n_repeats, random_state=seed)
i=0

roc_auc_scores = np.zeros(5)

for train_index, val_index in rskf.split(train_x, train_y):

    

    x_train, x_val = train_x.iloc[train_index], train_x.iloc[val_index]

    y_train, y_val = train_y.iloc[train_index], train_y.iloc[val_index]

    

    models[names[i]].fit(x_train, y_train)

    y_pred = models[names[i]].predict_proba(x_val)   

    y_proba = [p[1] for p in y_pred]

    roc_auc_scores[i] += roc_auc_score(y_val, y_proba)

  

    i += 1

    

    if i == 5:

        i=0



roc_auc_scores = roc_auc_scores/(n_repeats-2)

for i in range(len(names)):

    print(f"Validation ROC AUC score for {names[i]}: {roc_auc_scores[i]}")
for i in names:

    y_pred = models[i].predict_proba(test_x)

    fpr1, tpr1, thresh1 = roc_curve(test_y, y_pred[:,1], pos_label=1)

    plt.plot(fpr1, tpr1, linestyle='--', label=i)



plt.plot([0,1],[0,1], linestyle='--', color='blue')

plt.title('ROC curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive rate')

plt.legend(loc='best')

plt.show()

    
pred_y = ensemble.predict_proba(test_x)

prob_y = [p[1] for p in pred_y]

print("ROC AUC score: ",roc_auc_score(test_y, prob_y))



pred_y = ensemble.predict(test_x)

print("Confusion matrix: \n", confusion_matrix(test_y, pred_y))

print("Accuracy :", accuracy_score(test_y, pred_y))
