import pandas as pd

import numpy as np



from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix 

from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import classification_report, confusion_matrix



from sklearn.neural_network import MLPClassifier



import seaborn as sn

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')
df_freq = df.count()

variables = df_freq.loc[df_freq>0.1*df.shape[0]].index

df = df[variables]
df2 = df.copy()

df2.drop(['Patient addmited to regular ward (1=yes, 0=no)','Patient addmited to semi-intensive unit (1=yes, 0=no)','Patient addmited to intensive care unit (1=yes, 0=no)'],axis=1,inplace=True)

df2['SARS-Cov-2 exam result'] = df2['SARS-Cov-2 exam result'].map(lambda x: 1 if x == 'positive' else 0)

corrs = df2.corr()['SARS-Cov-2 exam result']

cont_out = list(corrs.loc[(corrs<0.1)|(corrs>-0.1)].index)

cont_vars = list(corrs.loc[(corrs>=0.1)|(corrs<=-0.1)].index)
corrs[cont_vars]
cont_vars = list(set(cont_vars)-set(['SARS-Cov-2 exam result','Patient addmited to regular ward (1=yes, 0=no)']))

variables = [var for var in variables if var not in cont_out]

cat_vars = list(set(variables) - set(cont_vars)-set(['Patient ID','Patient addmited to regular ward (1=yes, 0=no)','Patient addmited to semi-intensive unit (1=yes, 0=no)','Patient addmited to intensive care unit (1=yes, 0=no)']))

variables = cont_vars+cat_vars

yCol = ['SARS-Cov-2 exam result']

df = df[variables+yCol]
cont_vars
cat_vars
df = df.loc[~df[cont_vars].isna().all(axis=1)]

df[cat_vars] = df[cat_vars].replace(np.nan,'not_tested')

df_dummies = pd.get_dummies(df[cat_vars])



dumm_col = list(df_dummies.columns)

cols = []

for col in dumm_col:

    if "not_tested" not in col:

        cols.append(col)

df_dummies = df_dummies[cols]



df = pd.concat([df,df_dummies],axis=1)

df.drop(columns=cat_vars,axis=1,inplace=True)

variables = list(set(df.columns)-set(yCol))



df.dropna(inplace=True)

df.shape
scaler = MinMaxScaler(copy=False, feature_range=(0, 1))

df[variables] = scaler.fit_transform(df[variables])



df[yCol] = df[yCol].replace('negative',0)

df[yCol] = df[yCol].replace('positive',1)



x = df[variables]

y = df[yCol]

rus = RandomUnderSampler(random_state=42)

x_rus, y_rus = rus.fit_resample(x, y)



x_train, x_test, y_train, y_test = train_test_split(x_rus, y_rus, test_size=0.20, random_state=1)
clf = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-3,

                    hidden_layer_sizes=(28, 2), random_state=1)



clf.fit(x_train, y_train)

y_pred_nn1 = clf.predict(x_test)



clf = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-3,

                    hidden_layer_sizes=(32, 2), random_state=1)



clf.fit(x_train, y_train)

y_pred_nn2 = clf.predict(x_test)



clf = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-3,

                    hidden_layer_sizes=(36, 2), random_state=1)



clf.fit(x_train, y_train)

y_pred_nn3 = clf.predict(x_test)



clf = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-3,

                    hidden_layer_sizes=(40, 2), random_state=1)



clf.fit(x_train, y_train)

y_pred_nn4 = clf.predict(x_test)



clf = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-3,

                    hidden_layer_sizes=(44, 2), random_state=1)



clf.fit(x_train, y_train)

y_pred_nn5 = clf.predict(x_test)
df_ensemble = pd.DataFrame(columns=['class_nn1','class_nn2','class_nn3','class_nn4','class_nn5','y_true'])

df_ensemble['class_nn1'] = y_pred_nn1

df_ensemble['class_nn2'] = y_pred_nn2

df_ensemble['class_nn3'] = y_pred_nn3

df_ensemble['class_nn4'] = y_pred_nn4

df_ensemble['class_nn5'] = y_pred_nn5

y_test.reset_index(inplace=True,drop=True)

df_ensemble['y_true'] = y_test

df_ensemble['y_pred'] = df_ensemble[['class_nn1','class_nn2','class_nn3','class_nn4','class_nn5']].mode(axis=1)
cm = confusion_matrix(df_ensemble['y_true'],df_ensemble['y_pred'])



df_cm = pd.DataFrame(cm, index = ['negative','positive'],

                  columns = ['negative','positive'])

sn.set(font_scale=1.4) # for label size

sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
accuracy_score(df_ensemble['y_true'],df_ensemble['y_pred'])
precision_score(df_ensemble['y_true'],df_ensemble['y_pred'])
recall_score(df_ensemble['y_true'],df_ensemble['y_pred'])
f1_score(df_ensemble['y_true'],df_ensemble['y_pred'], average='macro')