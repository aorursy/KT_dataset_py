# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import keras
from keras.layers import Dense
from keras.models import Sequential
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_auc_score,roc_curve,scorer,auc
from sklearn.metrics import f1_score
import statsmodels.api as sm
from sklearn.metrics import precision_score,recall_score

import plotly.offline as py
py.init_notebook_mode(connected=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.DataFrame(df)
df.head()
sns.pairplot(df[['tenure','MonthlyCharges','TotalCharges','Churn','Contract','SeniorCitizen']], hue='Churn')
### checking counts of types in each columns
print(df.InternetService.value_counts())
print(df.Contract.value_counts())
print(df.PaymentMethod.value_counts())

### empty data replace with np.nan
df['TotalCharges'] = df['TotalCharges'].replace(" ",np.nan)

### drop rows including nan
df = df[df['TotalCharges'].notnull()]

### change str type into float
df['TotalCharges'] = df['TotalCharges'].astype('float')
### checking distribution of MonthlyCharges and TotalCharges
df.MonthlyCharges.plot(kind='hist',color='c')
plt.xlabel('dollar')
plt.title('MonthlyCharges')
plt.show()

df.TotalCharges.plot(kind='hist',color='coral')
plt.xlabel('dollar')
plt.title('TotalCharges')
plt.show()
#df[['MonthlyCharges','TotalCharges']].plot(kind='hist', subplots=True)

### making dummy variables except customerID column
df_dum = pd.get_dummies(df.iloc[:,1:])
df_dum.head()
plt.hist(data=df_dum, x='SeniorCitizen')
plt.title('SeniorCitizen or Not?')
plt.xlim(-1,2)
plt.xticks([0,1])
plt.show()
### chekcing columns
df_dum.columns
dum_cols = df.nunique()[df.nunique()<5].keys().tolist()
dum_cols = dum_cols[:-1]
num_cols = df.nunique()[df.nunique()>=5].keys().tolist()
num_cols = num_cols[1:]

print("Total number of dum_cols: "+ str(len(dum_cols)) + "\n" + str(dum_cols)+"\n")
print("Total number of num_cols: "+ str(len(num_cols)) + "\n" + str(num_cols))
def pie_plot(Column):    
    ct1 = pd.crosstab(df[Column],df['Churn'])
    trace1 = go.Pie(labels = ct1.index,
                    values = ct1.iloc[:,0],
                    hole=0.3,
                    domain=dict(x=[0,.45]))
    trace2 = go.Pie(labels = ct1.index,
                    values = ct1.iloc[:,1],
                    domain=dict(x=[.55,1]),
                    hole=0.3)

    layout = go.Layout(dict(title = Column + " distribution in customer attrition ",
                                plot_bgcolor  = "rgb(243,243,243)",
                                paper_bgcolor = "rgb(243,243,243)",
                                annotations = [dict(text = "churn customers",
                                                    font = dict(size = 13),
                                                    showarrow = False,
                                                    x = .15, y = 1),
                                               dict(text = "Non churn customers",
                                                    font = dict(size = 13),
                                                    showarrow = False,
                                                    x = .88,y = 1)

                                              ]
                               )
                          )

    fig = go.Figure(data=[trace1,trace2],layout=layout)
    py.iplot(fig)
for i in dum_cols:
    pie_plot(i)
### shorter the contract period, higher probability of churn? lets check crosstab

ct  = pd.crosstab(df.Contract, df.Churn)
ct.plot(kind='bar')
plt.title('Churn by Contract')
plt.ylabel("# of chrun")
plt.show()
### spliting into train and test set. fitting into linear regression model
X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(df_dum.iloc[:,:-2],df_dum.iloc[:,-1],test_size=0.2,random_state=5)
ln = LinearRegression()
ln_fit = ln.fit(X_train, Y_train)


###  showing relationship between monthly charges, TotalCharges between churn
plt.scatter(df['MonthlyCharges'], ln.predict(df_dum.iloc[:,:-2]), c='lawngreen', alpha=0.05)
plt.title('Monthly Charges vs Churn')
plt.show()

plt.scatter(df['TotalCharges'], ln.predict(df_dum.iloc[:,:-2]), c='c',alpha=0.05)
plt.title('Total Charges vs Churn')
plt.show()
df
df.head()
### correlation heatmap

plt.figure(figsize=(10,10))
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

### correlation matrix checking

rs = np.random.RandomState(0)
df = pd.DataFrame(rs.rand(10, 10))
corr = df_dum.corr()
corr.style.background_gradient()
### model fitting using logistic regression

lm = LogisticRegression(max_iter=10000)
fitted = lm.fit(X_train, Y_train)
pred = lm.predict(X_test)
scores = lm.score(X_test,Y_test)
print("Logistic Regression score is.:" + str(lm.score(X_test,Y_test)))

fpr, tpr, thres = roc_curve(Y_test, pred)
roc_auc = auc(fpr,tpr)
fpr, tpr, thres = roc_curve(Y_test, pred)
roc_auc = auc(fpr,tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
cf_matrix = confusion_matrix(Y_test,pred)
roc = roc_auc_score(Y_test,pred)
print(cf_matrix)
print(roc)
sns.heatmap(cf_matrix, annot=True, linewidths=1.0)
### model fitting with SVC

svc = SVC(max_iter=10000, gamma='auto')
svc_fitted = svc.fit(X_train, Y_train)
svc_pred = svc.predict(X_test)
svc_scores = svc.score(X_test,Y_test)
print("SVC Regression scores is..:" + str(svc.score(X_test,Y_test)))
### model fitting using SVM svc

svm = LinearSVC(max_iter=100000)
svm_fitted = svm.fit(X_train, Y_train)
svm_pred = svm.predict(X_test)
svm_scores = svm.score(X_test,Y_test)
print("SVM Regression scores is..:" + str(svm.score(X_test,Y_test)))
fpr, tpr, thres = roc_curve(Y_test, svm_pred)
roc_auc = auc(fpr,tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
### model fitting using deep learning
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=10)

predictors = np.array(df_dum.iloc[:,:-2])
target = np.array(df_dum.iloc[:,-1])
# Specify the model
n_cols = predictors.shape[1]
model_1 = Sequential()
model_1.add(Dense(10, activation='relu', input_shape = (n_cols,)))
model_1.add(Dense(10, activation='relu'))
model_1.add(Dense(2, activation='softmax'))
model_1.add(Dense(1))

model_2 = Sequential()
model_2.add(Dense(45, activation='relu', input_shape = (n_cols,)))
model_2.add(Dense(100, activation='relu'))
model_2.add(Dense(100, activation='relu'))
model_2.add(Dense(2, activation='softmax'))
model_2.add(Dense(1))

# Compile the model
model_1.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
model_2.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

# Fit the model
model_1_training = model_1.fit(predictors, target, batch_size=100, epochs=100, validation_split=0.2,callbacks=[early_stopping_monitor],verbose=False)
model_2_training = model_2.fit(predictors, target, batch_size=100, epochs=100, validation_split=0.2,callbacks=[early_stopping_monitor],verbose=False)
predictions = model_1.predict(X_test)
print(predictions)
model_1.summary()
model_1.evaluate(X_test,Y_test)

### compare model1 and model2
plt.plot(model_1_training.history['val_loss'], 'r', model_2_training.history['val_loss'], 'b')
plt.xlabel('Epochs')
plt.ylabel('Validation score')
plt.show()
