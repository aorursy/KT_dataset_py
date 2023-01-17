import numpy as np 

import pandas as pd 

import sklearn

import scipy

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import classification_report,accuracy_score

from pylab import rcParams

import plotly

import plotly.graph_objs as go

import plotly

import plotly.figure_factory as ff



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

dataset = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

dataset.shape
dataset.columns
dataset.isnull().values.any()
dataset.describe().T
dataset.info()
dataset.head()
dataset.shape
import plotly.express as px

df = pd.value_counts(dataset['Class'], sort = True).sort_index()

fig = px.bar(df)

fig.show()
import plotly.express as px

df = dataset

fig = px.histogram(df, x=dataset.Amount, color=dataset.Class)

fig.show()
import plotly.express as px

df = dataset

fig = px.histogram(df, x=dataset.Time, color=dataset.Class)

fig.show()
f,ax = plt.subplots(figsize=(25, 15))

sns.heatmap(dataset.corr(), annot=True, linewidths=.3, fmt= '.2f',ax=ax, cmap='viridis')

plt.savefig("corr.png")

plt.show()
fig, ax = plt.subplots(1, 2, figsize=(30,7))



amount = dataset['Amount'].values

time = dataset['Time'].values



sns.distplot(amount, ax=ax[0], color='green')

ax[0].set_title('Distribution of Transaction Amount', fontsize=14)

ax[0].set_xlim([min(amount), max(amount)])



sns.distplot(time, ax=ax[1], color='yellow')

ax[1].set_title('Distribution of Transaction Time', fontsize=14)

ax[1].set_xlim([min(time), max(time)])





plt.savefig("dis")

plt.show()
import plotly.express as px

df = dataset

fig = px.histogram(df, x=dataset.Time, y=dataset.Amount, color=dataset.Class,

                   marginal="box", # or violin, rug

                )

fig.show()
dataset.hist(figsize=(20,20))

plt.show()
import plotly.express as px

df = dataset

fig = px.box(df, x=dataset.Class, y=dataset.Amount, points="all")

fig.show()
import plotly.express as px

df = dataset

fig = px.box(df, x=dataset.Class, y=dataset.Time, points="all")

fig.show()
Fraud = dataset[dataset['Class']==1]

Normal = dataset[dataset['Class']==0]
pd.concat([Normal.Amount.describe(), Normal.Time.describe()],  axis=1)
pd.concat([Fraud.Amount.describe(), Fraud.Time.describe()],  axis=1)
print('Fraud Shape:\t', Fraud.shape)

print('Normal Shape:\t', Normal.shape)
f, (axes1, axes2) = plt.subplots(1,2,sharey=True)

f.set_figheight(15)

f.set_figwidth(40)

sns.heatmap(Fraud.corr(), annot=False, linewidths=.3, fmt= '.2f',ax=axes1, cmap='viridis')

axes1.title.set_text('Fraud Correlation')

sns.heatmap(Normal.corr(), annot=False, linewidths=.3, fmt= '.2f',ax=axes2, cmap='viridis')

axes2.title.set_text('Normal Correlation')

plt.savefig("corr_fr_nr.png")

plt.show()
import plotly.express as px

df = Fraud

fig = px.histogram(df,Fraud.Amount,

                   title='Amount of Transaction (Fraud)',

                   opacity=0.8,

                   color_discrete_sequence=['red'] # color of histogram bars

                   )

fig.show()
import plotly.express as px

df = Normal

fig = px.histogram(df,Normal.Amount,

                   title='Amount of Transactions (Normal)',

                   opacity=0.8,

                   color_discrete_sequence=['green'] # color of histogram bars

                   )

fig.show()
import plotly.graph_objects as go

from plotly.subplots import make_subplots



fig = make_subplots(rows=2, cols=1)





trace1 = go.Scatter(x=Fraud.Time, y=Fraud.Amount,

                    mode='markers',

                    name='Fraud',

                    )

trace2 = go.Scatter(x=Normal.Time,y=Normal.Amount,

                    mode='markers',

                    name='Normal')



fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 2, 1)



fig.show()
from sklearn.preprocessing import StandardScaler, RobustScaler



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



dataset['amount_scale'] = rob_scaler.fit_transform(dataset['Amount'].values.reshape(-1,1))

dataset['time_scale'] = rob_scaler.fit_transform(dataset['Time'].values.reshape(-1,1))



dataset.drop(['Time','Amount'], axis=1, inplace=True)
amount_scale = dataset['amount_scale']

time_scale = dataset['time_scale']



dataset.drop(['amount_scale', 'time_scale'], axis=1, inplace=True)

dataset.insert(0, 'amount_scale', amount_scale)

dataset.insert(1, 'time_scale', time_scale)



dataset.head()
dataset = dataset.sample(frac=1)



fraud = dataset.loc[dataset['Class'] == 1]

normal = dataset.loc[dataset['Class'] == 0][:492]



normal_distributed_data = pd.concat([fraud, normal])



sample_data = normal_distributed_data.sample(frac=1, random_state=42)



sample_data.head()
sample_data.shape
f,ax = plt.subplots(figsize=(25, 15))

sns.heatmap(sample_data.corr(), annot=True, linewidths=.3, fmt= '.2f',ax=ax, cmap='viridis')

plt.savefig("corr_sample.png")

plt.show()
f, axes = plt.subplots(nrows=2, ncols=5, figsize=(30,20))



sns.boxplot(x="Class", y="V2", data=sample_data, ax=axes[0][0])

axes[0][0].set_title('V2 vs Class Pozitif Correlation')

sns.boxplot(x="Class", y="V4", data=sample_data, ax=axes[0][1])

axes[0][1].set_title('V4 vs Class Pozitif Correlation')

sns.boxplot(x="Class", y="V11", data=sample_data, ax=axes[0][2])

axes[0][2].set_title('V11 vs Class Pozitif Correlation')

sns.boxplot(x="Class", y="V3", data=sample_data, ax=axes[0][3])

axes[0][3].set_title('V3 vs Class Negatif Correlation')

sns.boxplot(x="Class", y="V9", data=sample_data, ax=axes[0][4])

axes[0][4].set_title('V9 vs Class Negative Correlation')



sns.boxplot(x="Class", y="V10", data=sample_data, ax=axes[1][0])

axes[1][0].set_title('V10 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V12", data=sample_data, ax=axes[1][1])

axes[1][1].set_title('V12 vs Class Negatif Correlation')

sns.boxplot(x="Class", y="V14", data=sample_data, ax=axes[1][2])

axes[1][2].set_title('V14 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V16", data=sample_data, ax=axes[1][3])

axes[1][3].set_title('V16 vs Class Negative Correlation')

sns.boxplot(x="Class", y="V17", data=sample_data, ax=axes[1][4])

axes[1][4].set_title('V17 vs Class Negative Correlation')





plt.show()
X = sample_data.drop('Class', axis=1)

y = sample_data['Class']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=21)
from sklearn.svm import SVC

svm_model = SVC()
svm_params = {"C": np.arange(1,10), "kernel":["linear", "rbf"]}
from sklearn.model_selection import GridSearchCV

svm_cv_model = GridSearchCV(svm_model, svm_params, cv=7, n_jobs=-1, verbose=7).fit(X_train, y_train)
svm_cv_model.best_score_
best_params = svm_cv_model.best_params_

print(best_params)
svm = SVC(C = best_params['C'], kernel=best_params['kernel'], probability=True).fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
accuracy_score(y_test, y_pred_svm)
from sklearn.model_selection import cross_val_score

cross_val_score(svm, X_test, y_test, cv=21).mean()
print(classification_report(y_test, y_pred_svm))
from imblearn.metrics import classification_report_imbalanced, sensitivity_specificity_support

print('sensitivity and specificity:', sensitivity_specificity_support(y_test, y_pred_svm, average='micro', labels=pd.unique(dataset.Class)))

print(classification_report_imbalanced(y_test, y_pred_svm))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_svm)

sns.heatmap(cm, annot=True, fmt="d", cbar=False)

plt.title('SVC Confusion Matrix')

plt.savefig('svc_con_mat')

plt.show()
from sklearn.metrics import roc_auc_score, roc_curve

svm_roc_auc = roc_auc_score(y_test, svm.predict(X_test))

fpr , tpr, thresholds = roc_curve(y_test, svm.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % svm_roc_auc)

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.legend(loc='lower right')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=21)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf_params = {'n_estimators': [100,200,500],

            'max_features': [3,5,7],

            'min_samples_split':[5,10,20]}
rf_cv_model = GridSearchCV(rf, rf_params, cv=7, n_jobs=-1, verbose=1).fit(X_train, y_train)
rf_cv_model
best_params = rf_cv_model.best_params_

print(best_params)
rf = RandomForestClassifier(max_features=best_params['max_features'], min_samples_split=best_params['min_samples_split'], n_estimators=best_params['n_estimators']).fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_score(y_test, y_pred_rf)
rf.feature_importances_
feature_imp = pd.Series(rf.feature_importances_,

                       index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(15, 13))

sns.barplot(x=feature_imp, y=feature_imp.index)

plt.xlabel('Değisken Önem Skorları')

plt.ylabel('Değişkenler')

plt.title('Değişken Önem Düzeyleri')

plt.show()
cross_val_score(rf, X_test, y_test, cv=21).mean()
print(classification_report(y_test, y_pred_rf))
from imblearn.metrics import classification_report_imbalanced, sensitivity_specificity_support

print('sensitivity and specificity:', sensitivity_specificity_support(y_test, y_pred_rf, average='micro', labels=pd.unique(dataset.Class)))

print(classification_report_imbalanced(y_test, y_pred_rf))
cm = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm, annot=True, fmt="d", cbar=False)

plt.title('RF Confusion Matrix')

plt.savefig('rf_con_mat')

plt.show()
rf_roc_auc = roc_auc_score(y_test, rf.predict(X_test))

fpr , tpr, thresholds = roc_curve(y_test, rf.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % rf_roc_auc)

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.legend(loc='lower right')

plt.show()
from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split





x_orjinal_train , x_orjinal_test, y_orjinal_train, y_orjinal_test = train_test_split(X, y, test_size=0.33,random_state=21)



y_train = y_orjinal_train

y_test = y_orjinal_test



min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x_orjinal_train)

x_train = pd.DataFrame(x_scaled)

x_scaled1 = min_max_scaler.fit_transform(x_orjinal_test)

x_test = pd.DataFrame(x_scaled1)





x_train = x_train.values

x_test = x_test.values



x_train = np.asarray(x_train)

x_test = np.asarray(x_test)



x_train_mean = np.mean(x_train)

x_train_std = np.std(x_train)



x_test_mean = np.mean(x_test)

x_test_std = np.std(x_test)



x_train = (x_train - x_train_mean)/x_train_std

x_test = (x_test - x_test_mean)/x_test_std



print(x_train.shape)



x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size = 0.33, random_state = 21)
x_train.shape
import keras

from keras.models import Sequential

from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard



cnn = Sequential()

cnn.add(Conv1D(32, 2, activation='relu', input_shape=(30,1)))

cnn.add(Dropout(0.1))



cnn.add(Conv1D(64, 2, activation='relu'))

cnn.add(Dropout(0.2))





cnn.add(Flatten())

cnn.add(Dropout(0.4))

cnn.add(Dense(64, activation='relu'))

cnn.add(Dropout(0.5))



cnn.add(Dense(1, activation='sigmoid'))



cnn.summary()



cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])



epochs = 7

batch_size = 10

history = cnn.fit(x_train , y_train , verbose=1 , batch_size=batch_size , epochs=epochs ,validation_data=(x_test, y_test) )





from keras.utils import plot_model

plot_model(cnn)
loss, accuracy = cnn.evaluate(x_test, y_test, verbose=1)

loss_v, accuracy_v = cnn.evaluate(x_validate, y_validate, verbose=1)

print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))

print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
fig, ax1 = plt.subplots(figsize= (10, 5) )

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("acc.png")

plt.show()
fig, ax1 = plt.subplots(figsize= (10, 5) )

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.savefig("loss.png")

plt.show()
from sklearn.metrics import confusion_matrix

y_pred_cnn = cnn.predict_classes(x_validate)

cm = confusion_matrix(y_validate, y_pred_cnn)

sns.heatmap(cm, annot=True, fmt="d", cbar=False)

plt.title('CNN Confusion Matrix')

plt.savefig('cnn_con_mat')

plt.show()
accuracy_score(y_validate, y_pred_cnn)
print(classification_report(y_validate, y_pred_cnn))
from imblearn.metrics import classification_report_imbalanced, sensitivity_specificity_support

print('sensitivity and specificity:', sensitivity_specificity_support(y_validate, y_pred_cnn, average='micro', labels=pd.unique(dataset.Class)))

print(classification_report_imbalanced(y_validate, y_pred_cnn))
dataset = dataset.sample(frac=0.5)



fraud = dataset.loc[dataset['Class'] == 1]

normal = dataset.loc[dataset['Class'] == 0][:492]



normal_distributed_data = pd.concat([fraud, normal])



sample_data = normal_distributed_data.sample(frac=0.5, random_state=42)



sample_data.head()
X = sample_data.drop('Class', axis=1)

y = sample_data['Class']
min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(X)

x = pd.DataFrame(x_scaled)

x = x.values

x = np.asarray(x)

x_mean = np.mean(x)

x_std = np.std(x)

x = (x - x_mean)/x_std

print(x.shape)



x = np.reshape(x, (x.shape[0], x.shape[1], 1))

from sklearn.metrics import confusion_matrix



y_pred_svm = svm.predict(X)

y_pred_rf = rf.predict(X)

y_pred_cnn = cnn.predict_classes(x)



f, (axes1, axes2, axes3) = plt.subplots(1,3,sharey=True)

f.set_figheight(7)

f.set_figwidth(35)

sns.heatmap(confusion_matrix(y, y_pred_svm), annot=True, linewidths=.3, fmt= 'd',ax=axes1, cmap='viridis', cbar=False)

axes1.title.set_text('SVC Confusion Matrix')

sns.heatmap(confusion_matrix(y, y_pred_rf), annot=True, linewidths=.3, fmt= 'd',ax=axes2, cmap='viridis', cbar=False)

axes2.title.set_text('Random Forest Confusion Matrix')

sns.heatmap(confusion_matrix(y, y_pred_cnn), annot=True, linewidths=.3, fmt= 'd',ax=axes3, cmap='viridis', cbar=False)

axes3.title.set_text('CNN Confusion Matrix')

plt.savefig("cm_all.png")

plt.show()
models = [svm, rf, cnn]

result = []

results = pd.DataFrame(columns=['Models', "Accuracy"])



for model in models:

    names = model.__class__.__name__

    print(names)

    if names == 'Sequential':

        y_pred = model.predict_classes(x)

    else:

        y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)

    result = pd.DataFrame([[names, acc*100]], columns=['Models', 'Accuracy'])

    results = results.append(result)
plt.figure(figsize=(15,5))

sns.barplot(x='Accuracy', y='Models', data=results, color='purple')

plt.xlabel('Accuracy %')

plt.title('Modellerin Doğruluk Oranları');
results