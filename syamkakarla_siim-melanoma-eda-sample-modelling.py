import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly as py
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import iplot, init_notebook_mode
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

from sklearn import preprocessing

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
path =  '/kaggle/input/siim-isic-melanoma-classification/'

train = pd.read_csv(path+'train.csv')

test = pd.read_csv(path+'test.csv')
train.info()
train.head()
train.dtypes
train.isna().sum()
train.sex.fillna('Not Provoded', inplace = True)

train.age_approx.fillna(train.age_approx.mean(), inplace = True)

train.anatom_site_general_challenge.fillna('UnKnown' , inplace=True)
train.isna().sum()
b = train[train['target']==0]
n = 16
fig = plt.figure(figsize = (15,15))
for i, ind in zip(range(1, 1+n), [b.index[np.random.randint(b.shape[0])] for _ in range(n)]):
    fig.add_subplot(4,4,i)   
    plt.imshow(plt.imread(path+'jpeg/train/'+train.image_name[ind]+'.jpg'))
    plt.axis('off')
    plt.title('Benign'if train.target[ind] == 0 else 'Malignant')
plt.show()
b = train[train['target']==1]
n = 16
fig = plt.figure(figsize = (15,15))
for i, ind in zip(range(1, 1+n), [b.index[np.random.randint(b.shape[0])] for _ in range(n)]):
    fig.add_subplot(4,4,i)   
    plt.imshow(plt.imread(path+'jpeg/train/'+train.image_name[ind]+'.jpg'))
    plt.axis('off')
    plt.title('Benign'if train.target[ind] == 0 else 'Malignant')
plt.show()
x = train.sex.value_counts()
x = pd.DataFrame(data={'sex': x.index.tolist(), 'Count': x.values.tolist()})
fig = px.pie(x, values='Count', names='sex', title='Gender Affected Most')
fig.show()
x = train.age_approx.value_counts()

df = pd.DataFrame({'Age':x.index, 
                  'Count':x.values})
px.bar(df, x = 'Age', y = 'Count', color='Age', barmode='group')
x = train.diagnosis.value_counts()
x = pd.DataFrame(data={'sex': x.index.tolist(), 'Count': x.values.tolist()})
fig = px.pie(x, values='Count', names='sex', title='Gender Affected Most')
fig.show()
tr = train[['sex', 'age_approx', 'anatom_site_general_challenge', 'target']]
tr.head()
tr.dtypes
label_encoder = preprocessing.LabelEncoder()

tr['sex']= label_encoder.fit_transform(tr['sex']) 

tr['anatom_site_general_challenge']= label_encoder.fit_transform(tr['anatom_site_general_challenge']) 

tr.head()
test.anatom_site_general_challenge.fillna('UnKnown' , inplace=True)
test.isna().sum()
ts = test[['sex', 'age_approx', 'anatom_site_general_challenge']]

label_encoder = preprocessing.LabelEncoder()

ts['sex']= label_encoder.fit_transform(ts['sex']) 

ts['anatom_site_general_challenge']= label_encoder.fit_transform(ts['anatom_site_general_challenge']) 

ts.head()
X_train, X_test, y_train, y_test = train_test_split(tr.iloc[:, :-1], tr.iloc[:, -1], test_size=0.3, random_state=11)

model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_test, predictions))
x = confusion_matrix(y_test, predictions)
ff.create_annotated_heatmap(
    z=x,
    x=[0,1],
    y=[0,1],
    annotation_text=x,
    showscale=False, colorscale='Peach')
pred = model.predict(ts)
sub = pd.read_csv(path+'sample_submission.csv')
sub.info()
sub.head()
sub.target = pred
sub.to_csv('submission_XGBoost.csv', index=False)