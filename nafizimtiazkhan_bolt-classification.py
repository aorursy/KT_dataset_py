%matplotlib inline

!pip install --upgrade pip

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

import numpy as np

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

! pip install -q scikit-plot

import scikitplot as skplt

import pickle



df = pd.read_csv('../input/bolt-dataset/final_dataset.csv')



df["type"] = df["type"].astype('category')

df['type']=df['type'].cat.codes

df["type"]=df["type"].astype('float')

df.dtypes

df.head()
from imblearn.over_sampling import SMOTE

# for reproducibility purposes

seed = 100

# SMOTE number of neighbors

k = 1



#df = pd.read_csv('df_imbalanced.csv', encoding='utf-8', engine='python')

# make a new df made of all the columns, except the target class

X = df.loc[:, df.columns != 'type']

y = df.type

sm = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=seed)

X_res, y_res = sm.fit_resample(X, y)



# plt.title('base')

# plt.xlabel('x')

# plt.ylabel('y')

# plt.scatter(X_res[:, 0], X_res[:, 1], marker='o', c=y_res,

#            s=25, edgecolor='k', cmap=plt.cm.coolwarm)

# plt.show()



df = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res)], axis=1)

# rename the columns

df.columns = ['e1/do']+['e2/do']+['fu/fy']+['fmx/fndt']+['type']

df.to_csv('df_smoted.csv', index=False, encoding='utf-8')

print(df.columns)

print(df.info())

print(df.head())
# df['e1/do'] = np.log(df['e1/do'])

# df['e2/do'] = np.log(df['e2/do'])

# df['fu/fy'] = np.log(df['fu/fy'])

# df['fmx/fndt'] = np.log(df['fmx/fndt'])

# df['type'] = np.log(df['type'])

df = df.replace([np.inf, -np.inf], np.nan)

df = df.fillna(df.mean())

data= df[['e1/do']+['e2/do']+['fu/fy']+['fmx/fndt']+['type']]

print(data.head())

print(data.shape)
print(data.info())

features = list(data.columns.values)

print(features)



corr = data.corr() 

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.0) | (corr <= -0.0)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);

# Plottinf correlation above or below 0.5

corr = data.corr() # We already examined SalePrice correlations

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.5) | (corr <= -0.5)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);



import seaborn as sns

quantitative_features_list1 = ['e1/do', 'e2/do', 'fu/fy', 'fmx/fndt', 'type']

#quantitative_features_list1 = ['a/d', 'p', 'sqrt(fc)', 'lf/df', 'Vf', 'F', 'Type', 'Vu']

data_plot_data=data_mod_num = data[quantitative_features_list1]

sns.pairplot(data_plot_data)

print(data.head())





X = data.loc[:, data.columns != 'type']

y=data['type']

print(X.head())

print(y.head())
evaluation = pd.DataFrame({'Model': [],

                           'Accuracy(train)':[],

                           'Precision(train)':[],

                           'Recall(train)':[],

                           'F1_score(train)':[],

                           'Accuracy(test)':[],

                           'Precision(test)':[],

                           'Recalll(test)':[],

                           'F1_score(test)':[]})



evaluation2 = pd.DataFrame({'Model': [],

                           'Test':[],

                           '1':[],

                           '2':[],

                           '3':[],

                           '4':[],

                           '5':[],

                           '6':[],

                           '7':[],

                           '8':[],

                           '9':[],

                           '10':[],

                           'Mean':[]})

import numpy as np

from sklearn import linear_model

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

sm = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=seed)

X_train, y_train = sm.fit_resample(X_train, y_train)

X_test, y_test = sm.fit_resample(X_test, y_test)



pickle.dump(scaler, open('scaler.pkl','wb'))
from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification

clf = AdaBoostClassifier(n_estimators=895, random_state=500)

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['ADABOOST',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)









complex_model_1=clf



cv_train_acc=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy')

cv_train_acc_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy').mean(),'.3f'))



cv_train_pre=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro')

cv_train_pre_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro').mean(),'.3f'))



cv_train_re=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro')

cv_train_re_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro').mean(),'.3f'))



cv_train_f1=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro')

cv_train_f1_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro').mean()



cv_test_acc=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy')

cv_test_acc_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy').mean()



cv_test_pre=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro')

cv_test_pre_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro').mean()



cv_test_re=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro')

cv_test_re_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro').mean()



cv_test_f1=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro')

cv_test_f1_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro').mean()





r = evaluation2.shape[0]

evaluation2.loc[r] = ['AdaBoost','Train accuracy',float(format(cv_train_acc[0],'.3f')),float(format(cv_train_acc[1],'.3f')),float(format(cv_train_acc[2],'.3f')),float(format(cv_train_acc[3],'.3f')),float(format(cv_train_acc[4],'.3f')),float(format(cv_train_acc[5],'.3f')),float(format(cv_train_acc[6],'.3f')),float(format(cv_train_acc[7],'.3f')),float(format(cv_train_acc[8],'.3f')),float(format(cv_train_acc[9],'.3f')),float(format(cv_train_acc_m,'.3f'))]

evaluation2.loc[r+1] = ['AdaBoost','Train precision',float(format(cv_train_pre[0],'.3f')),float(format(cv_train_pre[1],'.3f')),float(format(cv_train_pre[2],'.3f')),float(format(cv_train_pre[3],'.3f')),float(format(cv_train_pre[4],'.3f')),float(format(cv_train_pre[5],'.3f')),float(format(cv_train_pre[6],'.3f')),float(format(cv_train_pre[7],'.3f')),float(format(cv_train_pre[8],'.3f')),float(format(cv_train_pre[9],'.3f')),float(format(cv_train_pre_m,'.3f'))]

evaluation2.loc[r+2] = ['AdaBoost','Train recall',float(format(cv_train_re[0],'.3f')),float(format(cv_train_re[1],'.3f')),float(format(cv_train_re[2],'.3f')),float(format(cv_train_re[3],'.3f')),float(format(cv_train_re[4],'.3f')),float(format(cv_train_re[5],'.3f')),float(format(cv_train_re[6],'.3f')),float(format(cv_train_re[7],'.3f')),float(format(cv_train_re[8],'.3f')),float(format(cv_train_re[9],'.3f')),float(format(cv_train_re_m,'.3f'))]

evaluation2.loc[r+3] = ['AdaBoost','Train f1_score',float(format(cv_train_f1[0],'.3f')),float(format(cv_train_f1[1],'.3f')),float(format(cv_train_f1[2],'.3f')),float(format(cv_train_f1[3],'.3f')),float(format(cv_train_f1[4],'.3f')),float(format(cv_train_f1[5],'.3f')),float(format(cv_train_f1[6],'.3f')),float(format(cv_train_f1[7],'.3f')),float(format(cv_train_f1[8],'.3f')),float(format(cv_train_f1[9],'.3f')),float(format(cv_train_f1_m,'.3f'))]

evaluation2.loc[r+4] = ['AdaBoost','Test accuracy',float(format(cv_test_acc[0],'.3f')),float(format(cv_test_acc[1],'.3f')),float(format(cv_test_acc[2],'.3f')),float(format(cv_test_acc[3],'.3f')),float(format(cv_test_acc[4],'.3f')),float(format(cv_test_acc[5],'.3f')),float(format(cv_test_acc[6],'.3f')),float(format(cv_test_acc[7],'.3f')),float(format(cv_test_acc[8],'.3f')),float(format(cv_test_acc[9],'.3f')),float(format(cv_test_acc_m,'.3f'))]

evaluation2.loc[r+5] = ['AdaBoost','Test precision',float(format(cv_test_pre[0],'.3f')),float(format(cv_test_pre[1],'.3f')),float(format(cv_test_pre[2],'.3f')),float(format(cv_test_pre[3],'.3f')),float(format(cv_test_pre[4],'.3f')),float(format(cv_test_pre[5],'.3f')),float(format(cv_test_pre[6],'.3f')),float(format(cv_test_pre[7],'.3f')),float(format(cv_test_pre[8],'.3f')),float(format(cv_test_pre[9],'.3f')),float(format(cv_test_pre_m,'.3f'))]

evaluation2.loc[r+6] = ['AdaBoost','Test recall',float(format(cv_test_re[0],'.3f')),float(format(cv_test_re[1],'.3f')),float(format(cv_test_re[2],'.3f')),float(format(cv_test_re[3],'.3f')),float(format(cv_test_re[4],'.3f')),float(format(cv_test_re[5],'.3f')),float(format(cv_test_re[6],'.3f')),float(format(cv_test_re[7],'.3f')),float(format(cv_test_re[8],'.3f')),float(format(cv_test_re[9],'.3f')),float(format(cv_test_re_m,'.3f'))]

evaluation2.loc[r+7] = ['AdaBoost','Train f1_score',float(format(cv_test_f1[0],'.3f')),float(format(cv_test_f1[1],'.3f')),float(format(cv_test_f1[2],'.3f')),float(format(cv_test_f1[3],'.3f')),float(format(cv_test_f1[4],'.3f')),float(format(cv_test_f1[5],'.3f')),float(format(cv_test_f1[6],'.3f')),float(format(cv_test_f1[7],'.3f')),float(format(cv_test_f1[8],'.3f')),float(format(cv_test_f1[9],'.3f')),float(format(cv_test_f1_m,'.3f')) ]
pickle.dump(complex_model_1, open('bolt_ab_c.pkl','wb'))
importances = clf.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)
p=y_train

q=y_test

y_train=y_train.replace([0,1,2,3], ["B","N","SP","TO"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0,1,2,3], ["B","N","SP","TO"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0,1,2,3], ["B","N","SP","TO"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1,2,3], ["B","N","SP","TO"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
!pip install catboost

from catboost import CatBoostClassifier



clf = CatBoostClassifier(

    iterations=1000, 

    learning_rate=0.1, 

    #verbose=5,

    #loss_function='CrossEntropy'

)

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['CATBOOST',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)











complex_model_1=clf



cv_train_acc=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy')

cv_train_acc_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy').mean(),'.3f'))



cv_train_pre=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro')

cv_train_pre_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro').mean(),'.3f'))



cv_train_re=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro')

cv_train_re_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro').mean(),'.3f'))



cv_train_f1=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro')

cv_train_f1_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro').mean()



cv_test_acc=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy')

cv_test_acc_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy').mean()



cv_test_pre=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro')

cv_test_pre_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro').mean()



cv_test_re=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro')

cv_test_re_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro').mean()



cv_test_f1=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro')

cv_test_f1_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro').mean()





r = evaluation2.shape[0]

evaluation2.loc[r] = ['CatBoost','Train accuracy',float(format(cv_train_acc[0],'.3f')),float(format(cv_train_acc[1],'.3f')),float(format(cv_train_acc[2],'.3f')),float(format(cv_train_acc[3],'.3f')),float(format(cv_train_acc[4],'.3f')),float(format(cv_train_acc[5],'.3f')),float(format(cv_train_acc[6],'.3f')),float(format(cv_train_acc[7],'.3f')),float(format(cv_train_acc[8],'.3f')),float(format(cv_train_acc[9],'.3f')),float(format(cv_train_acc_m,'.3f'))]

evaluation2.loc[r+1] = ['CatBoost','Train precision',float(format(cv_train_pre[0],'.3f')),float(format(cv_train_pre[1],'.3f')),float(format(cv_train_pre[2],'.3f')),float(format(cv_train_pre[3],'.3f')),float(format(cv_train_pre[4],'.3f')),float(format(cv_train_pre[5],'.3f')),float(format(cv_train_pre[6],'.3f')),float(format(cv_train_pre[7],'.3f')),float(format(cv_train_pre[8],'.3f')),float(format(cv_train_pre[9],'.3f')),float(format(cv_train_pre_m,'.3f'))]

evaluation2.loc[r+2] = ['CatBoost','Train recall',float(format(cv_train_re[0],'.3f')),float(format(cv_train_re[1],'.3f')),float(format(cv_train_re[2],'.3f')),float(format(cv_train_re[3],'.3f')),float(format(cv_train_re[4],'.3f')),float(format(cv_train_re[5],'.3f')),float(format(cv_train_re[6],'.3f')),float(format(cv_train_re[7],'.3f')),float(format(cv_train_re[8],'.3f')),float(format(cv_train_re[9],'.3f')),float(format(cv_train_re_m,'.3f'))]

evaluation2.loc[r+3] = ['CatBoost','Train f1_score',float(format(cv_train_f1[0],'.3f')),float(format(cv_train_f1[1],'.3f')),float(format(cv_train_f1[2],'.3f')),float(format(cv_train_f1[3],'.3f')),float(format(cv_train_f1[4],'.3f')),float(format(cv_train_f1[5],'.3f')),float(format(cv_train_f1[6],'.3f')),float(format(cv_train_f1[7],'.3f')),float(format(cv_train_f1[8],'.3f')),float(format(cv_train_f1[9],'.3f')),float(format(cv_train_f1_m,'.3f'))]

evaluation2.loc[r+4] = ['CatBoost','Test accuracy',float(format(cv_test_acc[0],'.3f')),float(format(cv_test_acc[1],'.3f')),float(format(cv_test_acc[2],'.3f')),float(format(cv_test_acc[3],'.3f')),float(format(cv_test_acc[4],'.3f')),float(format(cv_test_acc[5],'.3f')),float(format(cv_test_acc[6],'.3f')),float(format(cv_test_acc[7],'.3f')),float(format(cv_test_acc[8],'.3f')),float(format(cv_test_acc[9],'.3f')),float(format(cv_test_acc_m,'.3f'))]

evaluation2.loc[r+5] = ['CatBoost','Test precision',float(format(cv_test_pre[0],'.3f')),float(format(cv_test_pre[1],'.3f')),float(format(cv_test_pre[2],'.3f')),float(format(cv_test_pre[3],'.3f')),float(format(cv_test_pre[4],'.3f')),float(format(cv_test_pre[5],'.3f')),float(format(cv_test_pre[6],'.3f')),float(format(cv_test_pre[7],'.3f')),float(format(cv_test_pre[8],'.3f')),float(format(cv_test_pre[9],'.3f')),float(format(cv_test_pre_m,'.3f'))]

evaluation2.loc[r+6] = ['CatBoost','Test recall',float(format(cv_test_re[0],'.3f')),float(format(cv_test_re[1],'.3f')),float(format(cv_test_re[2],'.3f')),float(format(cv_test_re[3],'.3f')),float(format(cv_test_re[4],'.3f')),float(format(cv_test_re[5],'.3f')),float(format(cv_test_re[6],'.3f')),float(format(cv_test_re[7],'.3f')),float(format(cv_test_re[8],'.3f')),float(format(cv_test_re[9],'.3f')),float(format(cv_test_re_m,'.3f'))]

evaluation2.loc[r+7] = ['CatBoost','Train f1_score',float(format(cv_test_f1[0],'.3f')),float(format(cv_test_f1[1],'.3f')),float(format(cv_test_f1[2],'.3f')),float(format(cv_test_f1[3],'.3f')),float(format(cv_test_f1[4],'.3f')),float(format(cv_test_f1[5],'.3f')),float(format(cv_test_f1[6],'.3f')),float(format(cv_test_f1[7],'.3f')),float(format(cv_test_f1[8],'.3f')),float(format(cv_test_f1[9],'.3f')),float(format(cv_test_f1_m,'.3f')) ]

pickle.dump(complex_model_1, open('bolt_cb_c.pkl','wb'))
features = list(X.columns.values)

importances = clf.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)
p=y_train

q=y_test

y_train=y_train.replace([0,1,2,3], ["B","N","SP","TO"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0,1,2,3], ["B","N","SP","TO"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0,1,2,3], ["B","N","SP","TO"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1,2,3], ["B","N","SP","TO"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
import xgboost as xgb

clf = xgb.XGBClassifier(random_state=700)

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['XGBOOST',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)



complex_model_1=clf



cv_train_acc=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy')

cv_train_acc_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy').mean(),'.3f'))



cv_train_pre=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro')

cv_train_pre_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro').mean(),'.3f'))



cv_train_re=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro')

cv_train_re_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro').mean(),'.3f'))



cv_train_f1=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro')

cv_train_f1_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro').mean()



cv_test_acc=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy')

cv_test_acc_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy').mean()



cv_test_pre=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro')

cv_test_pre_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro').mean()



cv_test_re=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro')

cv_test_re_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro').mean()



cv_test_f1=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro')

cv_test_f1_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro').mean()





r = evaluation2.shape[0]

evaluation2.loc[r] = ['XgBoost','Train accuracy',float(format(cv_train_acc[0],'.3f')),float(format(cv_train_acc[1],'.3f')),float(format(cv_train_acc[2],'.3f')),float(format(cv_train_acc[3],'.3f')),float(format(cv_train_acc[4],'.3f')),float(format(cv_train_acc[5],'.3f')),float(format(cv_train_acc[6],'.3f')),float(format(cv_train_acc[7],'.3f')),float(format(cv_train_acc[8],'.3f')),float(format(cv_train_acc[9],'.3f')),float(format(cv_train_acc_m,'.3f'))]

evaluation2.loc[r+1] = ['XgBoost','Train precision',float(format(cv_train_pre[0],'.3f')),float(format(cv_train_pre[1],'.3f')),float(format(cv_train_pre[2],'.3f')),float(format(cv_train_pre[3],'.3f')),float(format(cv_train_pre[4],'.3f')),float(format(cv_train_pre[5],'.3f')),float(format(cv_train_pre[6],'.3f')),float(format(cv_train_pre[7],'.3f')),float(format(cv_train_pre[8],'.3f')),float(format(cv_train_pre[9],'.3f')),float(format(cv_train_pre_m,'.3f'))]

evaluation2.loc[r+2] = ['XgBoost','Train recall',float(format(cv_train_re[0],'.3f')),float(format(cv_train_re[1],'.3f')),float(format(cv_train_re[2],'.3f')),float(format(cv_train_re[3],'.3f')),float(format(cv_train_re[4],'.3f')),float(format(cv_train_re[5],'.3f')),float(format(cv_train_re[6],'.3f')),float(format(cv_train_re[7],'.3f')),float(format(cv_train_re[8],'.3f')),float(format(cv_train_re[9],'.3f')),float(format(cv_train_re_m,'.3f'))]

evaluation2.loc[r+3] = ['XgBoost','Train f1_score',float(format(cv_train_f1[0],'.3f')),float(format(cv_train_f1[1],'.3f')),float(format(cv_train_f1[2],'.3f')),float(format(cv_train_f1[3],'.3f')),float(format(cv_train_f1[4],'.3f')),float(format(cv_train_f1[5],'.3f')),float(format(cv_train_f1[6],'.3f')),float(format(cv_train_f1[7],'.3f')),float(format(cv_train_f1[8],'.3f')),float(format(cv_train_f1[9],'.3f')),float(format(cv_train_f1_m,'.3f'))]

evaluation2.loc[r+4] = ['XgBoost','Test accuracy',float(format(cv_test_acc[0],'.3f')),float(format(cv_test_acc[1],'.3f')),float(format(cv_test_acc[2],'.3f')),float(format(cv_test_acc[3],'.3f')),float(format(cv_test_acc[4],'.3f')),float(format(cv_test_acc[5],'.3f')),float(format(cv_test_acc[6],'.3f')),float(format(cv_test_acc[7],'.3f')),float(format(cv_test_acc[8],'.3f')),float(format(cv_test_acc[9],'.3f')),float(format(cv_test_acc_m,'.3f'))]

evaluation2.loc[r+5] = ['XgBoost','Test precision',float(format(cv_test_pre[0],'.3f')),float(format(cv_test_pre[1],'.3f')),float(format(cv_test_pre[2],'.3f')),float(format(cv_test_pre[3],'.3f')),float(format(cv_test_pre[4],'.3f')),float(format(cv_test_pre[5],'.3f')),float(format(cv_test_pre[6],'.3f')),float(format(cv_test_pre[7],'.3f')),float(format(cv_test_pre[8],'.3f')),float(format(cv_test_pre[9],'.3f')),float(format(cv_test_pre_m,'.3f'))]

evaluation2.loc[r+6] = ['XgBoost','Test recall',float(format(cv_test_re[0],'.3f')),float(format(cv_test_re[1],'.3f')),float(format(cv_test_re[2],'.3f')),float(format(cv_test_re[3],'.3f')),float(format(cv_test_re[4],'.3f')),float(format(cv_test_re[5],'.3f')),float(format(cv_test_re[6],'.3f')),float(format(cv_test_re[7],'.3f')),float(format(cv_test_re[8],'.3f')),float(format(cv_test_re[9],'.3f')),float(format(cv_test_re_m,'.3f'))]

evaluation2.loc[r+7] = ['XgBoost','Train f1_score',float(format(cv_test_f1[0],'.3f')),float(format(cv_test_f1[1],'.3f')),float(format(cv_test_f1[2],'.3f')),float(format(cv_test_f1[3],'.3f')),float(format(cv_test_f1[4],'.3f')),float(format(cv_test_f1[5],'.3f')),float(format(cv_test_f1[6],'.3f')),float(format(cv_test_f1[7],'.3f')),float(format(cv_test_f1[8],'.3f')),float(format(cv_test_f1[9],'.3f')),float(format(cv_test_f1_m,'.3f')) ]



pickle.dump(complex_model_1, open('bolt_xb_c.pkl','wb'))
features = list(X.columns.values)

importances = clf.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)
p=y_train

q=y_test

y_train=y_train.replace([0,1,2,3], ["B","N","SP","TO"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0,1,2,3], ["B","N","SP","TO"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0,1,2,3], ["B","N","SP","TO"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1,2,3], ["B","N","SP","TO"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(random_state=1000, learning_rate=0.1,n_estimators=500)

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['GradientBoosting',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)





complex_model_1=clf



cv_train_acc=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy')

cv_train_acc_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy').mean(),'.3f'))



cv_train_pre=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro')

cv_train_pre_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro').mean(),'.3f'))



cv_train_re=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro')

cv_train_re_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro').mean(),'.3f'))



cv_train_f1=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro')

cv_train_f1_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro').mean()



cv_test_acc=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy')

cv_test_acc_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy').mean()



cv_test_pre=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro')

cv_test_pre_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro').mean()



cv_test_re=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro')

cv_test_re_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro').mean()



cv_test_f1=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro')

cv_test_f1_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro').mean()





r = evaluation2.shape[0]

evaluation2.loc[r] = ['GB','Train accuracy',float(format(cv_train_acc[0],'.3f')),float(format(cv_train_acc[1],'.3f')),float(format(cv_train_acc[2],'.3f')),float(format(cv_train_acc[3],'.3f')),float(format(cv_train_acc[4],'.3f')),float(format(cv_train_acc[5],'.3f')),float(format(cv_train_acc[6],'.3f')),float(format(cv_train_acc[7],'.3f')),float(format(cv_train_acc[8],'.3f')),float(format(cv_train_acc[9],'.3f')),float(format(cv_train_acc_m,'.3f'))]

evaluation2.loc[r+1] = ['GB','Train precision',float(format(cv_train_pre[0],'.3f')),float(format(cv_train_pre[1],'.3f')),float(format(cv_train_pre[2],'.3f')),float(format(cv_train_pre[3],'.3f')),float(format(cv_train_pre[4],'.3f')),float(format(cv_train_pre[5],'.3f')),float(format(cv_train_pre[6],'.3f')),float(format(cv_train_pre[7],'.3f')),float(format(cv_train_pre[8],'.3f')),float(format(cv_train_pre[9],'.3f')),float(format(cv_train_pre_m,'.3f'))]

evaluation2.loc[r+2] = ['GB','Train recall',float(format(cv_train_re[0],'.3f')),float(format(cv_train_re[1],'.3f')),float(format(cv_train_re[2],'.3f')),float(format(cv_train_re[3],'.3f')),float(format(cv_train_re[4],'.3f')),float(format(cv_train_re[5],'.3f')),float(format(cv_train_re[6],'.3f')),float(format(cv_train_re[7],'.3f')),float(format(cv_train_re[8],'.3f')),float(format(cv_train_re[9],'.3f')),float(format(cv_train_re_m,'.3f'))]

evaluation2.loc[r+3] = ['GB','Train f1_score',float(format(cv_train_f1[0],'.3f')),float(format(cv_train_f1[1],'.3f')),float(format(cv_train_f1[2],'.3f')),float(format(cv_train_f1[3],'.3f')),float(format(cv_train_f1[4],'.3f')),float(format(cv_train_f1[5],'.3f')),float(format(cv_train_f1[6],'.3f')),float(format(cv_train_f1[7],'.3f')),float(format(cv_train_f1[8],'.3f')),float(format(cv_train_f1[9],'.3f')),float(format(cv_train_f1_m,'.3f'))]

evaluation2.loc[r+4] = ['GB','Test accuracy',float(format(cv_test_acc[0],'.3f')),float(format(cv_test_acc[1],'.3f')),float(format(cv_test_acc[2],'.3f')),float(format(cv_test_acc[3],'.3f')),float(format(cv_test_acc[4],'.3f')),float(format(cv_test_acc[5],'.3f')),float(format(cv_test_acc[6],'.3f')),float(format(cv_test_acc[7],'.3f')),float(format(cv_test_acc[8],'.3f')),float(format(cv_test_acc[9],'.3f')),float(format(cv_test_acc_m,'.3f'))]

evaluation2.loc[r+5] = ['GB','Test precision',float(format(cv_test_pre[0],'.3f')),float(format(cv_test_pre[1],'.3f')),float(format(cv_test_pre[2],'.3f')),float(format(cv_test_pre[3],'.3f')),float(format(cv_test_pre[4],'.3f')),float(format(cv_test_pre[5],'.3f')),float(format(cv_test_pre[6],'.3f')),float(format(cv_test_pre[7],'.3f')),float(format(cv_test_pre[8],'.3f')),float(format(cv_test_pre[9],'.3f')),float(format(cv_test_pre_m,'.3f'))]

evaluation2.loc[r+6] = ['GB','Test recall',float(format(cv_test_re[0],'.3f')),float(format(cv_test_re[1],'.3f')),float(format(cv_test_re[2],'.3f')),float(format(cv_test_re[3],'.3f')),float(format(cv_test_re[4],'.3f')),float(format(cv_test_re[5],'.3f')),float(format(cv_test_re[6],'.3f')),float(format(cv_test_re[7],'.3f')),float(format(cv_test_re[8],'.3f')),float(format(cv_test_re[9],'.3f')),float(format(cv_test_re_m,'.3f'))]

evaluation2.loc[r+7] = ['GB','Train f1_score',float(format(cv_test_f1[0],'.3f')),float(format(cv_test_f1[1],'.3f')),float(format(cv_test_f1[2],'.3f')),float(format(cv_test_f1[3],'.3f')),float(format(cv_test_f1[4],'.3f')),float(format(cv_test_f1[5],'.3f')),float(format(cv_test_f1[6],'.3f')),float(format(cv_test_f1[7],'.3f')),float(format(cv_test_f1[8],'.3f')),float(format(cv_test_f1[9],'.3f')),float(format(cv_test_f1_m,'.3f')) ]

pickle.dump(complex_model_1, open('bolt_gb_c.pkl','wb'))
features = list(X.columns.values)

importances = clf.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)
p=y_train

q=y_test

y_train=y_train.replace([0,1,2,3], ["B","N","SP","TO"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0,1,2,3], ["B","N","SP","TO"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0,1,2,3], ["B","N","SP","TO"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1,2,3], ["B","N","SP","TO"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from sklearn import svm

clf =svm.SVC(kernel='rbf',degree=100)

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['SVM',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)





complex_model_1=clf



cv_train_acc=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy')

cv_train_acc_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy').mean(),'.3f'))



cv_train_pre=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro')

cv_train_pre_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro').mean(),'.3f'))



cv_train_re=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro')

cv_train_re_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro').mean(),'.3f'))



cv_train_f1=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro')

cv_train_f1_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro').mean()



cv_test_acc=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy')

cv_test_acc_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy').mean()



cv_test_pre=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro')

cv_test_pre_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro').mean()



cv_test_re=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro')

cv_test_re_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro').mean()



cv_test_f1=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro')

cv_test_f1_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro').mean()





r = evaluation2.shape[0]

evaluation2.loc[r] = ['SVM','Train accuracy',float(format(cv_train_acc[0],'.3f')),float(format(cv_train_acc[1],'.3f')),float(format(cv_train_acc[2],'.3f')),float(format(cv_train_acc[3],'.3f')),float(format(cv_train_acc[4],'.3f')),float(format(cv_train_acc[5],'.3f')),float(format(cv_train_acc[6],'.3f')),float(format(cv_train_acc[7],'.3f')),float(format(cv_train_acc[8],'.3f')),float(format(cv_train_acc[9],'.3f')),float(format(cv_train_acc_m,'.3f'))]

evaluation2.loc[r+1] = ['SVM','Train precision',float(format(cv_train_pre[0],'.3f')),float(format(cv_train_pre[1],'.3f')),float(format(cv_train_pre[2],'.3f')),float(format(cv_train_pre[3],'.3f')),float(format(cv_train_pre[4],'.3f')),float(format(cv_train_pre[5],'.3f')),float(format(cv_train_pre[6],'.3f')),float(format(cv_train_pre[7],'.3f')),float(format(cv_train_pre[8],'.3f')),float(format(cv_train_pre[9],'.3f')),float(format(cv_train_pre_m,'.3f'))]

evaluation2.loc[r+2] = ['SVM','Train recall',float(format(cv_train_re[0],'.3f')),float(format(cv_train_re[1],'.3f')),float(format(cv_train_re[2],'.3f')),float(format(cv_train_re[3],'.3f')),float(format(cv_train_re[4],'.3f')),float(format(cv_train_re[5],'.3f')),float(format(cv_train_re[6],'.3f')),float(format(cv_train_re[7],'.3f')),float(format(cv_train_re[8],'.3f')),float(format(cv_train_re[9],'.3f')),float(format(cv_train_re_m,'.3f'))]

evaluation2.loc[r+3] = ['SVM','Train f1_score',float(format(cv_train_f1[0],'.3f')),float(format(cv_train_f1[1],'.3f')),float(format(cv_train_f1[2],'.3f')),float(format(cv_train_f1[3],'.3f')),float(format(cv_train_f1[4],'.3f')),float(format(cv_train_f1[5],'.3f')),float(format(cv_train_f1[6],'.3f')),float(format(cv_train_f1[7],'.3f')),float(format(cv_train_f1[8],'.3f')),float(format(cv_train_f1[9],'.3f')),float(format(cv_train_f1_m,'.3f'))]

evaluation2.loc[r+4] = ['SVM','Test accuracy',float(format(cv_test_acc[0],'.3f')),float(format(cv_test_acc[1],'.3f')),float(format(cv_test_acc[2],'.3f')),float(format(cv_test_acc[3],'.3f')),float(format(cv_test_acc[4],'.3f')),float(format(cv_test_acc[5],'.3f')),float(format(cv_test_acc[6],'.3f')),float(format(cv_test_acc[7],'.3f')),float(format(cv_test_acc[8],'.3f')),float(format(cv_test_acc[9],'.3f')),float(format(cv_test_acc_m,'.3f'))]

evaluation2.loc[r+5] = ['SVM','Test precision',float(format(cv_test_pre[0],'.3f')),float(format(cv_test_pre[1],'.3f')),float(format(cv_test_pre[2],'.3f')),float(format(cv_test_pre[3],'.3f')),float(format(cv_test_pre[4],'.3f')),float(format(cv_test_pre[5],'.3f')),float(format(cv_test_pre[6],'.3f')),float(format(cv_test_pre[7],'.3f')),float(format(cv_test_pre[8],'.3f')),float(format(cv_test_pre[9],'.3f')),float(format(cv_test_pre_m,'.3f'))]

evaluation2.loc[r+6] = ['SVM','Test recall',float(format(cv_test_re[0],'.3f')),float(format(cv_test_re[1],'.3f')),float(format(cv_test_re[2],'.3f')),float(format(cv_test_re[3],'.3f')),float(format(cv_test_re[4],'.3f')),float(format(cv_test_re[5],'.3f')),float(format(cv_test_re[6],'.3f')),float(format(cv_test_re[7],'.3f')),float(format(cv_test_re[8],'.3f')),float(format(cv_test_re[9],'.3f')),float(format(cv_test_re_m,'.3f'))]

evaluation2.loc[r+7] = ['SVM','Train f1_score',float(format(cv_test_f1[0],'.3f')),float(format(cv_test_f1[1],'.3f')),float(format(cv_test_f1[2],'.3f')),float(format(cv_test_f1[3],'.3f')),float(format(cv_test_f1[4],'.3f')),float(format(cv_test_f1[5],'.3f')),float(format(cv_test_f1[6],'.3f')),float(format(cv_test_f1[7],'.3f')),float(format(cv_test_f1[8],'.3f')),float(format(cv_test_f1[9],'.3f')),float(format(cv_test_f1_m,'.3f')) ]

pickle.dump(complex_model_1, open('bolt_svm_c.pkl','wb'))
# features = list(X.columns.values)

# importances = clf.feature_importances_

# import numpy as np

# indices = np.argsort(importances)

# plt.title('Feature Importances')

# plt.barh(range(len(indices)), importances[indices], color='b', align='center')

# plt.yticks(range(len(indices)), [features[i] for i in indices])

# plt.xlabel('Relative Importance')

# plt.show()



# print(importances)
p=y_train

q=y_test

y_train=y_train.replace([0,1,2,3], ["B","N","SP","TO"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0,1,2,3], ["B","N","SP","TO"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0,1,2,3], ["B","N","SP","TO"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1,2,3], ["B","N","SP","TO"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from sklearn.ensemble import RandomForestClassifier

clf =RandomForestClassifier()

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['Random Forest',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)





complex_model_1=clf



cv_train_acc=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy')

cv_train_acc_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy').mean(),'.3f'))



cv_train_pre=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro')

cv_train_pre_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro').mean(),'.3f'))



cv_train_re=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro')

cv_train_re_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro').mean(),'.3f'))



cv_train_f1=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro')

cv_train_f1_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro').mean()



cv_test_acc=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy')

cv_test_acc_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy').mean()



cv_test_pre=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro')

cv_test_pre_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro').mean()



cv_test_re=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro')

cv_test_re_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro').mean()



cv_test_f1=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro')

cv_test_f1_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro').mean()





r = evaluation2.shape[0]

evaluation2.loc[r] = ['RF','Train accuracy',float(format(cv_train_acc[0],'.3f')),float(format(cv_train_acc[1],'.3f')),float(format(cv_train_acc[2],'.3f')),float(format(cv_train_acc[3],'.3f')),float(format(cv_train_acc[4],'.3f')),float(format(cv_train_acc[5],'.3f')),float(format(cv_train_acc[6],'.3f')),float(format(cv_train_acc[7],'.3f')),float(format(cv_train_acc[8],'.3f')),float(format(cv_train_acc[9],'.3f')),float(format(cv_train_acc_m,'.3f'))]

evaluation2.loc[r+1] = ['RF','Train precision',float(format(cv_train_pre[0],'.3f')),float(format(cv_train_pre[1],'.3f')),float(format(cv_train_pre[2],'.3f')),float(format(cv_train_pre[3],'.3f')),float(format(cv_train_pre[4],'.3f')),float(format(cv_train_pre[5],'.3f')),float(format(cv_train_pre[6],'.3f')),float(format(cv_train_pre[7],'.3f')),float(format(cv_train_pre[8],'.3f')),float(format(cv_train_pre[9],'.3f')),float(format(cv_train_pre_m,'.3f'))]

evaluation2.loc[r+2] = ['RF','Train recall',float(format(cv_train_re[0],'.3f')),float(format(cv_train_re[1],'.3f')),float(format(cv_train_re[2],'.3f')),float(format(cv_train_re[3],'.3f')),float(format(cv_train_re[4],'.3f')),float(format(cv_train_re[5],'.3f')),float(format(cv_train_re[6],'.3f')),float(format(cv_train_re[7],'.3f')),float(format(cv_train_re[8],'.3f')),float(format(cv_train_re[9],'.3f')),float(format(cv_train_re_m,'.3f'))]

evaluation2.loc[r+3] = ['RF','Train f1_score',float(format(cv_train_f1[0],'.3f')),float(format(cv_train_f1[1],'.3f')),float(format(cv_train_f1[2],'.3f')),float(format(cv_train_f1[3],'.3f')),float(format(cv_train_f1[4],'.3f')),float(format(cv_train_f1[5],'.3f')),float(format(cv_train_f1[6],'.3f')),float(format(cv_train_f1[7],'.3f')),float(format(cv_train_f1[8],'.3f')),float(format(cv_train_f1[9],'.3f')),float(format(cv_train_f1_m,'.3f'))]

evaluation2.loc[r+4] = ['RF','Test accuracy',float(format(cv_test_acc[0],'.3f')),float(format(cv_test_acc[1],'.3f')),float(format(cv_test_acc[2],'.3f')),float(format(cv_test_acc[3],'.3f')),float(format(cv_test_acc[4],'.3f')),float(format(cv_test_acc[5],'.3f')),float(format(cv_test_acc[6],'.3f')),float(format(cv_test_acc[7],'.3f')),float(format(cv_test_acc[8],'.3f')),float(format(cv_test_acc[9],'.3f')),float(format(cv_test_acc_m,'.3f'))]

evaluation2.loc[r+5] = ['RF','Test precision',float(format(cv_test_pre[0],'.3f')),float(format(cv_test_pre[1],'.3f')),float(format(cv_test_pre[2],'.3f')),float(format(cv_test_pre[3],'.3f')),float(format(cv_test_pre[4],'.3f')),float(format(cv_test_pre[5],'.3f')),float(format(cv_test_pre[6],'.3f')),float(format(cv_test_pre[7],'.3f')),float(format(cv_test_pre[8],'.3f')),float(format(cv_test_pre[9],'.3f')),float(format(cv_test_pre_m,'.3f'))]

evaluation2.loc[r+6] = ['RF','Test recall',float(format(cv_test_re[0],'.3f')),float(format(cv_test_re[1],'.3f')),float(format(cv_test_re[2],'.3f')),float(format(cv_test_re[3],'.3f')),float(format(cv_test_re[4],'.3f')),float(format(cv_test_re[5],'.3f')),float(format(cv_test_re[6],'.3f')),float(format(cv_test_re[7],'.3f')),float(format(cv_test_re[8],'.3f')),float(format(cv_test_re[9],'.3f')),float(format(cv_test_re_m,'.3f'))]

evaluation2.loc[r+7] = ['RF','Train f1_score',float(format(cv_test_f1[0],'.3f')),float(format(cv_test_f1[1],'.3f')),float(format(cv_test_f1[2],'.3f')),float(format(cv_test_f1[3],'.3f')),float(format(cv_test_f1[4],'.3f')),float(format(cv_test_f1[5],'.3f')),float(format(cv_test_f1[6],'.3f')),float(format(cv_test_f1[7],'.3f')),float(format(cv_test_f1[8],'.3f')),float(format(cv_test_f1[9],'.3f')),float(format(cv_test_f1_m,'.3f')) ]

pickle.dump(complex_model_1, open('bolt_rf_c.pkl','wb'))


features = list(X.columns.values)

importances = clf.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)
p=y_train

q=y_test

y_train=y_train.replace([0,1,2,3], ["B","N","SP","TO"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0,1,2,3], ["B","N","SP","TO"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0,1,2,3], ["B","N","SP","TO"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1,2,3], ["B","N","SP","TO"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from sklearn import tree

clf =tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['Decision Tree',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)





complex_model_1=clf



cv_train_acc=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy')

cv_train_acc_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy').mean(),'.3f'))



cv_train_pre=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro')

cv_train_pre_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro').mean(),'.3f'))



cv_train_re=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro')

cv_train_re_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro').mean(),'.3f'))



cv_train_f1=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro')

cv_train_f1_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro').mean()



cv_test_acc=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy')

cv_test_acc_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy').mean()



cv_test_pre=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro')

cv_test_pre_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro').mean()



cv_test_re=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro')

cv_test_re_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro').mean()



cv_test_f1=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro')

cv_test_f1_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro').mean()





r = evaluation2.shape[0]

evaluation2.loc[r] = ['DT','Train accuracy',float(format(cv_train_acc[0],'.3f')),float(format(cv_train_acc[1],'.3f')),float(format(cv_train_acc[2],'.3f')),float(format(cv_train_acc[3],'.3f')),float(format(cv_train_acc[4],'.3f')),float(format(cv_train_acc[5],'.3f')),float(format(cv_train_acc[6],'.3f')),float(format(cv_train_acc[7],'.3f')),float(format(cv_train_acc[8],'.3f')),float(format(cv_train_acc[9],'.3f')),float(format(cv_train_acc_m,'.3f'))]

evaluation2.loc[r+1] = ['DT','Train precision',float(format(cv_train_pre[0],'.3f')),float(format(cv_train_pre[1],'.3f')),float(format(cv_train_pre[2],'.3f')),float(format(cv_train_pre[3],'.3f')),float(format(cv_train_pre[4],'.3f')),float(format(cv_train_pre[5],'.3f')),float(format(cv_train_pre[6],'.3f')),float(format(cv_train_pre[7],'.3f')),float(format(cv_train_pre[8],'.3f')),float(format(cv_train_pre[9],'.3f')),float(format(cv_train_pre_m,'.3f'))]

evaluation2.loc[r+2] = ['DT','Train recall',float(format(cv_train_re[0],'.3f')),float(format(cv_train_re[1],'.3f')),float(format(cv_train_re[2],'.3f')),float(format(cv_train_re[3],'.3f')),float(format(cv_train_re[4],'.3f')),float(format(cv_train_re[5],'.3f')),float(format(cv_train_re[6],'.3f')),float(format(cv_train_re[7],'.3f')),float(format(cv_train_re[8],'.3f')),float(format(cv_train_re[9],'.3f')),float(format(cv_train_re_m,'.3f'))]

evaluation2.loc[r+3] = ['DT','Train f1_score',float(format(cv_train_f1[0],'.3f')),float(format(cv_train_f1[1],'.3f')),float(format(cv_train_f1[2],'.3f')),float(format(cv_train_f1[3],'.3f')),float(format(cv_train_f1[4],'.3f')),float(format(cv_train_f1[5],'.3f')),float(format(cv_train_f1[6],'.3f')),float(format(cv_train_f1[7],'.3f')),float(format(cv_train_f1[8],'.3f')),float(format(cv_train_f1[9],'.3f')),float(format(cv_train_f1_m,'.3f'))]

evaluation2.loc[r+4] = ['DT','Test accuracy',float(format(cv_test_acc[0],'.3f')),float(format(cv_test_acc[1],'.3f')),float(format(cv_test_acc[2],'.3f')),float(format(cv_test_acc[3],'.3f')),float(format(cv_test_acc[4],'.3f')),float(format(cv_test_acc[5],'.3f')),float(format(cv_test_acc[6],'.3f')),float(format(cv_test_acc[7],'.3f')),float(format(cv_test_acc[8],'.3f')),float(format(cv_test_acc[9],'.3f')),float(format(cv_test_acc_m,'.3f'))]

evaluation2.loc[r+5] = ['DT','Test precision',float(format(cv_test_pre[0],'.3f')),float(format(cv_test_pre[1],'.3f')),float(format(cv_test_pre[2],'.3f')),float(format(cv_test_pre[3],'.3f')),float(format(cv_test_pre[4],'.3f')),float(format(cv_test_pre[5],'.3f')),float(format(cv_test_pre[6],'.3f')),float(format(cv_test_pre[7],'.3f')),float(format(cv_test_pre[8],'.3f')),float(format(cv_test_pre[9],'.3f')),float(format(cv_test_pre_m,'.3f'))]

evaluation2.loc[r+6] = ['DT','Test recall',float(format(cv_test_re[0],'.3f')),float(format(cv_test_re[1],'.3f')),float(format(cv_test_re[2],'.3f')),float(format(cv_test_re[3],'.3f')),float(format(cv_test_re[4],'.3f')),float(format(cv_test_re[5],'.3f')),float(format(cv_test_re[6],'.3f')),float(format(cv_test_re[7],'.3f')),float(format(cv_test_re[8],'.3f')),float(format(cv_test_re[9],'.3f')),float(format(cv_test_re_m,'.3f'))]

evaluation2.loc[r+7] = ['DT','Train f1_score',float(format(cv_test_f1[0],'.3f')),float(format(cv_test_f1[1],'.3f')),float(format(cv_test_f1[2],'.3f')),float(format(cv_test_f1[3],'.3f')),float(format(cv_test_f1[4],'.3f')),float(format(cv_test_f1[5],'.3f')),float(format(cv_test_f1[6],'.3f')),float(format(cv_test_f1[7],'.3f')),float(format(cv_test_f1[8],'.3f')),float(format(cv_test_f1[9],'.3f')),float(format(cv_test_f1_m,'.3f')) ]

pickle.dump(complex_model_1, open('bolt_dt_c.pkl','wb'))
features = list(X.columns.values)

importances = clf.feature_importances_

import numpy as np

indices = np.argsort(importances)

plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()



print(importances)
p=y_train

q=y_test

y_train=y_train.replace([0,1,2,3], ["B","N","SP","TO"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0,1,2,3], ["B","N","SP","TO"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0,1,2,3], ["B","N","SP","TO"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1,2,3], ["B","N","SP","TO"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from sklearn.neighbors import KNeighborsClassifier

clf =KNeighborsClassifier(n_neighbors=1)

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['KNN',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)





complex_model_1=clf



cv_train_acc=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy')

cv_train_acc_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy').mean(),'.3f'))



cv_train_pre=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro')

cv_train_pre_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro').mean(),'.3f'))



cv_train_re=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro')

cv_train_re_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro').mean(),'.3f'))



cv_train_f1=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro')

cv_train_f1_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro').mean()



cv_test_acc=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy')

cv_test_acc_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy').mean()



cv_test_pre=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro')

cv_test_pre_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro').mean()



cv_test_re=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro')

cv_test_re_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro').mean()



cv_test_f1=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro')

cv_test_f1_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro').mean()





r = evaluation2.shape[0]

evaluation2.loc[r] = ['KNN','Train accuracy',float(format(cv_train_acc[0],'.3f')),float(format(cv_train_acc[1],'.3f')),float(format(cv_train_acc[2],'.3f')),float(format(cv_train_acc[3],'.3f')),float(format(cv_train_acc[4],'.3f')),float(format(cv_train_acc[5],'.3f')),float(format(cv_train_acc[6],'.3f')),float(format(cv_train_acc[7],'.3f')),float(format(cv_train_acc[8],'.3f')),float(format(cv_train_acc[9],'.3f')),float(format(cv_train_acc_m,'.3f'))]

evaluation2.loc[r+1] = ['KNN','Train precision',float(format(cv_train_pre[0],'.3f')),float(format(cv_train_pre[1],'.3f')),float(format(cv_train_pre[2],'.3f')),float(format(cv_train_pre[3],'.3f')),float(format(cv_train_pre[4],'.3f')),float(format(cv_train_pre[5],'.3f')),float(format(cv_train_pre[6],'.3f')),float(format(cv_train_pre[7],'.3f')),float(format(cv_train_pre[8],'.3f')),float(format(cv_train_pre[9],'.3f')),float(format(cv_train_pre_m,'.3f'))]

evaluation2.loc[r+2] = ['KNN','Train recall',float(format(cv_train_re[0],'.3f')),float(format(cv_train_re[1],'.3f')),float(format(cv_train_re[2],'.3f')),float(format(cv_train_re[3],'.3f')),float(format(cv_train_re[4],'.3f')),float(format(cv_train_re[5],'.3f')),float(format(cv_train_re[6],'.3f')),float(format(cv_train_re[7],'.3f')),float(format(cv_train_re[8],'.3f')),float(format(cv_train_re[9],'.3f')),float(format(cv_train_re_m,'.3f'))]

evaluation2.loc[r+3] = ['KNN','Train f1_score',float(format(cv_train_f1[0],'.3f')),float(format(cv_train_f1[1],'.3f')),float(format(cv_train_f1[2],'.3f')),float(format(cv_train_f1[3],'.3f')),float(format(cv_train_f1[4],'.3f')),float(format(cv_train_f1[5],'.3f')),float(format(cv_train_f1[6],'.3f')),float(format(cv_train_f1[7],'.3f')),float(format(cv_train_f1[8],'.3f')),float(format(cv_train_f1[9],'.3f')),float(format(cv_train_f1_m,'.3f'))]

evaluation2.loc[r+4] = ['KNN','Test accuracy',float(format(cv_test_acc[0],'.3f')),float(format(cv_test_acc[1],'.3f')),float(format(cv_test_acc[2],'.3f')),float(format(cv_test_acc[3],'.3f')),float(format(cv_test_acc[4],'.3f')),float(format(cv_test_acc[5],'.3f')),float(format(cv_test_acc[6],'.3f')),float(format(cv_test_acc[7],'.3f')),float(format(cv_test_acc[8],'.3f')),float(format(cv_test_acc[9],'.3f')),float(format(cv_test_acc_m,'.3f'))]

evaluation2.loc[r+5] = ['KNN','Test precision',float(format(cv_test_pre[0],'.3f')),float(format(cv_test_pre[1],'.3f')),float(format(cv_test_pre[2],'.3f')),float(format(cv_test_pre[3],'.3f')),float(format(cv_test_pre[4],'.3f')),float(format(cv_test_pre[5],'.3f')),float(format(cv_test_pre[6],'.3f')),float(format(cv_test_pre[7],'.3f')),float(format(cv_test_pre[8],'.3f')),float(format(cv_test_pre[9],'.3f')),float(format(cv_test_pre_m,'.3f'))]

evaluation2.loc[r+6] = ['KNN','Test recall',float(format(cv_test_re[0],'.3f')),float(format(cv_test_re[1],'.3f')),float(format(cv_test_re[2],'.3f')),float(format(cv_test_re[3],'.3f')),float(format(cv_test_re[4],'.3f')),float(format(cv_test_re[5],'.3f')),float(format(cv_test_re[6],'.3f')),float(format(cv_test_re[7],'.3f')),float(format(cv_test_re[8],'.3f')),float(format(cv_test_re[9],'.3f')),float(format(cv_test_re_m,'.3f'))]

evaluation2.loc[r+7] = ['KNN','Train f1_score',float(format(cv_test_f1[0],'.3f')),float(format(cv_test_f1[1],'.3f')),float(format(cv_test_f1[2],'.3f')),float(format(cv_test_f1[3],'.3f')),float(format(cv_test_f1[4],'.3f')),float(format(cv_test_f1[5],'.3f')),float(format(cv_test_f1[6],'.3f')),float(format(cv_test_f1[7],'.3f')),float(format(cv_test_f1[8],'.3f')),float(format(cv_test_f1[9],'.3f')),float(format(cv_test_f1_m,'.3f')) ]

pickle.dump(complex_model_1, open('bolt_knn_c.pkl','wb'))
# features = list(X.columns.values)

# importances = clf.feature_importances_

# import numpy as np

# indices = np.argsort(importances)

# plt.title('Feature Importances')

# plt.barh(range(len(indices)), importances[indices], color='b', align='center')

# plt.yticks(range(len(indices)), [features[i] for i in indices])

# plt.xlabel('Relative Importance')

# plt.show()



# print(importances)
p=y_train

q=y_test

y_train=y_train.replace([0,1,2,3], ["B","N","SP","TO"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0,1,2,3], ["B","N","SP","TO"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0,1,2,3], ["B","N","SP","TO"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1,2,3], ["B","N","SP","TO"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from sklearn.neural_network import MLPClassifier

clf =MLPClassifier(solver='lbfgs', alpha=1e-5,

                     hidden_layer_sizes=(16, 16), random_state=100)

# clf =MLPClassifier()

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['ANN',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)





complex_model_1=clf



cv_train_acc=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy')

cv_train_acc_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy').mean(),'.3f'))



cv_train_pre=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro')

cv_train_pre_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro').mean(),'.3f'))



cv_train_re=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro')

cv_train_re_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro').mean(),'.3f'))



cv_train_f1=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro')

cv_train_f1_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro').mean()



cv_test_acc=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy')

cv_test_acc_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy').mean()



cv_test_pre=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro')

cv_test_pre_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro').mean()



cv_test_re=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro')

cv_test_re_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro').mean()



cv_test_f1=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro')

cv_test_f1_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro').mean()





r = evaluation2.shape[0]

evaluation2.loc[r] = ['ANN','Train accuracy',float(format(cv_train_acc[0],'.3f')),float(format(cv_train_acc[1],'.3f')),float(format(cv_train_acc[2],'.3f')),float(format(cv_train_acc[3],'.3f')),float(format(cv_train_acc[4],'.3f')),float(format(cv_train_acc[5],'.3f')),float(format(cv_train_acc[6],'.3f')),float(format(cv_train_acc[7],'.3f')),float(format(cv_train_acc[8],'.3f')),float(format(cv_train_acc[9],'.3f')),float(format(cv_train_acc_m,'.3f'))]

evaluation2.loc[r+1] = ['ANN','Train precision',float(format(cv_train_pre[0],'.3f')),float(format(cv_train_pre[1],'.3f')),float(format(cv_train_pre[2],'.3f')),float(format(cv_train_pre[3],'.3f')),float(format(cv_train_pre[4],'.3f')),float(format(cv_train_pre[5],'.3f')),float(format(cv_train_pre[6],'.3f')),float(format(cv_train_pre[7],'.3f')),float(format(cv_train_pre[8],'.3f')),float(format(cv_train_pre[9],'.3f')),float(format(cv_train_pre_m,'.3f'))]

evaluation2.loc[r+2] = ['ANN','Train recall',float(format(cv_train_re[0],'.3f')),float(format(cv_train_re[1],'.3f')),float(format(cv_train_re[2],'.3f')),float(format(cv_train_re[3],'.3f')),float(format(cv_train_re[4],'.3f')),float(format(cv_train_re[5],'.3f')),float(format(cv_train_re[6],'.3f')),float(format(cv_train_re[7],'.3f')),float(format(cv_train_re[8],'.3f')),float(format(cv_train_re[9],'.3f')),float(format(cv_train_re_m,'.3f'))]

evaluation2.loc[r+3] = ['ANN','Train f1_score',float(format(cv_train_f1[0],'.3f')),float(format(cv_train_f1[1],'.3f')),float(format(cv_train_f1[2],'.3f')),float(format(cv_train_f1[3],'.3f')),float(format(cv_train_f1[4],'.3f')),float(format(cv_train_f1[5],'.3f')),float(format(cv_train_f1[6],'.3f')),float(format(cv_train_f1[7],'.3f')),float(format(cv_train_f1[8],'.3f')),float(format(cv_train_f1[9],'.3f')),float(format(cv_train_f1_m,'.3f'))]

evaluation2.loc[r+4] = ['ANN','Test accuracy',float(format(cv_test_acc[0],'.3f')),float(format(cv_test_acc[1],'.3f')),float(format(cv_test_acc[2],'.3f')),float(format(cv_test_acc[3],'.3f')),float(format(cv_test_acc[4],'.3f')),float(format(cv_test_acc[5],'.3f')),float(format(cv_test_acc[6],'.3f')),float(format(cv_test_acc[7],'.3f')),float(format(cv_test_acc[8],'.3f')),float(format(cv_test_acc[9],'.3f')),float(format(cv_test_acc_m,'.3f'))]

evaluation2.loc[r+5] = ['ANN','Test precision',float(format(cv_test_pre[0],'.3f')),float(format(cv_test_pre[1],'.3f')),float(format(cv_test_pre[2],'.3f')),float(format(cv_test_pre[3],'.3f')),float(format(cv_test_pre[4],'.3f')),float(format(cv_test_pre[5],'.3f')),float(format(cv_test_pre[6],'.3f')),float(format(cv_test_pre[7],'.3f')),float(format(cv_test_pre[8],'.3f')),float(format(cv_test_pre[9],'.3f')),float(format(cv_test_pre_m,'.3f'))]

evaluation2.loc[r+6] = ['ANN','Test recall',float(format(cv_test_re[0],'.3f')),float(format(cv_test_re[1],'.3f')),float(format(cv_test_re[2],'.3f')),float(format(cv_test_re[3],'.3f')),float(format(cv_test_re[4],'.3f')),float(format(cv_test_re[5],'.3f')),float(format(cv_test_re[6],'.3f')),float(format(cv_test_re[7],'.3f')),float(format(cv_test_re[8],'.3f')),float(format(cv_test_re[9],'.3f')),float(format(cv_test_re_m,'.3f'))]

evaluation2.loc[r+7] = ['ANN','Train f1_score',float(format(cv_test_f1[0],'.3f')),float(format(cv_test_f1[1],'.3f')),float(format(cv_test_f1[2],'.3f')),float(format(cv_test_f1[3],'.3f')),float(format(cv_test_f1[4],'.3f')),float(format(cv_test_f1[5],'.3f')),float(format(cv_test_f1[6],'.3f')),float(format(cv_test_f1[7],'.3f')),float(format(cv_test_f1[8],'.3f')),float(format(cv_test_f1[9],'.3f')),float(format(cv_test_f1_m,'.3f')) ]

pickle.dump(complex_model_1, open('bolt_ann_c.pkl','wb'))
# features = list(X.columns.values)

# importances = clf.feature_importances_

# import numpy as np

# indices = np.argsort(importances)

# plt.title('Feature Importances')

# plt.barh(range(len(indices)), importances[indices], color='b', align='center')

# plt.yticks(range(len(indices)), [features[i] for i in indices])

# plt.xlabel('Relative Importance')

# plt.show()



# print(importances)
p=y_train

q=y_test

y_train=y_train.replace([0,1,2,3], ["B","N","SP","TO"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0,1,2,3], ["B","N","SP","TO"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0,1,2,3], ["B","N","SP","TO"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1,2,3], ["B","N","SP","TO"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from sklearn.naive_bayes import GaussianNB

clf =GaussianNB()

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train),'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='macro'),'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='macro'),'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='macro'),'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test),'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='macro'),'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='macro'),'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='macro'),'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['Naive Bayes',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)







complex_model_1=clf



cv_train_acc=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy')

cv_train_acc_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy').mean(),'.3f'))



cv_train_pre=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro')

cv_train_pre_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision_macro').mean(),'.3f'))



cv_train_re=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro')

cv_train_re_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall_macro').mean(),'.3f'))



cv_train_f1=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro')

cv_train_f1_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro').mean()



cv_test_acc=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy')

cv_test_acc_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy').mean()



cv_test_pre=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro')

cv_test_pre_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision_macro').mean()



cv_test_re=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro')

cv_test_re_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall_macro').mean()



cv_test_f1=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro')

cv_test_f1_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro').mean()





r = evaluation2.shape[0]

evaluation2.loc[r] = ['GNB','Train accuracy',float(format(cv_train_acc[0],'.3f')),float(format(cv_train_acc[1],'.3f')),float(format(cv_train_acc[2],'.3f')),float(format(cv_train_acc[3],'.3f')),float(format(cv_train_acc[4],'.3f')),float(format(cv_train_acc[5],'.3f')),float(format(cv_train_acc[6],'.3f')),float(format(cv_train_acc[7],'.3f')),float(format(cv_train_acc[8],'.3f')),float(format(cv_train_acc[9],'.3f')),float(format(cv_train_acc_m,'.3f'))]

evaluation2.loc[r+1] = ['GNB','Train precision',float(format(cv_train_pre[0],'.3f')),float(format(cv_train_pre[1],'.3f')),float(format(cv_train_pre[2],'.3f')),float(format(cv_train_pre[3],'.3f')),float(format(cv_train_pre[4],'.3f')),float(format(cv_train_pre[5],'.3f')),float(format(cv_train_pre[6],'.3f')),float(format(cv_train_pre[7],'.3f')),float(format(cv_train_pre[8],'.3f')),float(format(cv_train_pre[9],'.3f')),float(format(cv_train_pre_m,'.3f'))]

evaluation2.loc[r+2] = ['GNB','Train recall',float(format(cv_train_re[0],'.3f')),float(format(cv_train_re[1],'.3f')),float(format(cv_train_re[2],'.3f')),float(format(cv_train_re[3],'.3f')),float(format(cv_train_re[4],'.3f')),float(format(cv_train_re[5],'.3f')),float(format(cv_train_re[6],'.3f')),float(format(cv_train_re[7],'.3f')),float(format(cv_train_re[8],'.3f')),float(format(cv_train_re[9],'.3f')),float(format(cv_train_re_m,'.3f'))]

evaluation2.loc[r+3] = ['GNB','Train f1_score',float(format(cv_train_f1[0],'.3f')),float(format(cv_train_f1[1],'.3f')),float(format(cv_train_f1[2],'.3f')),float(format(cv_train_f1[3],'.3f')),float(format(cv_train_f1[4],'.3f')),float(format(cv_train_f1[5],'.3f')),float(format(cv_train_f1[6],'.3f')),float(format(cv_train_f1[7],'.3f')),float(format(cv_train_f1[8],'.3f')),float(format(cv_train_f1[9],'.3f')),float(format(cv_train_f1_m,'.3f'))]

evaluation2.loc[r+4] = ['GNB','Test accuracy',float(format(cv_test_acc[0],'.3f')),float(format(cv_test_acc[1],'.3f')),float(format(cv_test_acc[2],'.3f')),float(format(cv_test_acc[3],'.3f')),float(format(cv_test_acc[4],'.3f')),float(format(cv_test_acc[5],'.3f')),float(format(cv_test_acc[6],'.3f')),float(format(cv_test_acc[7],'.3f')),float(format(cv_test_acc[8],'.3f')),float(format(cv_test_acc[9],'.3f')),float(format(cv_test_acc_m,'.3f'))]

evaluation2.loc[r+5] = ['GNB','Test precision',float(format(cv_test_pre[0],'.3f')),float(format(cv_test_pre[1],'.3f')),float(format(cv_test_pre[2],'.3f')),float(format(cv_test_pre[3],'.3f')),float(format(cv_test_pre[4],'.3f')),float(format(cv_test_pre[5],'.3f')),float(format(cv_test_pre[6],'.3f')),float(format(cv_test_pre[7],'.3f')),float(format(cv_test_pre[8],'.3f')),float(format(cv_test_pre[9],'.3f')),float(format(cv_test_pre_m,'.3f'))]

evaluation2.loc[r+6] = ['GNB','Test recall',float(format(cv_test_re[0],'.3f')),float(format(cv_test_re[1],'.3f')),float(format(cv_test_re[2],'.3f')),float(format(cv_test_re[3],'.3f')),float(format(cv_test_re[4],'.3f')),float(format(cv_test_re[5],'.3f')),float(format(cv_test_re[6],'.3f')),float(format(cv_test_re[7],'.3f')),float(format(cv_test_re[8],'.3f')),float(format(cv_test_re[9],'.3f')),float(format(cv_test_re_m,'.3f'))]

evaluation2.loc[r+7] = ['GNB','Train f1_score',float(format(cv_test_f1[0],'.3f')),float(format(cv_test_f1[1],'.3f')),float(format(cv_test_f1[2],'.3f')),float(format(cv_test_f1[3],'.3f')),float(format(cv_test_f1[4],'.3f')),float(format(cv_test_f1[5],'.3f')),float(format(cv_test_f1[6],'.3f')),float(format(cv_test_f1[7],'.3f')),float(format(cv_test_f1[8],'.3f')),float(format(cv_test_f1[9],'.3f')),float(format(cv_test_f1_m,'.3f')) ]

pickle.dump(complex_model_1, open('bolt_gnb_c.pkl','wb'))
# features = list(X.columns.values)

# importances = clf.feature_importances_

# import numpy as np

# indices = np.argsort(importances)

# plt.title('Feature Importances')

# plt.barh(range(len(indices)), importances[indices], color='b', align='center')

# plt.yticks(range(len(indices)), [features[i] for i in indices])

# plt.xlabel('Relative Importance')

# plt.show()



# print(importances)
p=y_train

q=y_test

y_train=y_train.replace([0,1,2,3], ["B","N","SP","TO"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0,1,2,3], ["B","N","SP","TO"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0,1,2,3], ["B","N","SP","TO"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1,2,3], ["B","N","SP","TO"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(8,7),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
evaluation.to_csv('eval.csv')

evaluation2.to_csv("cross_val_results.csv")