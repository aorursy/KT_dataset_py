%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

import numpy as np

from sklearn import linear_model

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

!pip install scikit-plot

import scikitplot as skplt

import pickle

! pip install -q scikit-plot

font = {'family' : 'Times New Roman',

        'size'   : 26}



plt.rc('font', **font)

plt.rcParams.update({'font.size': 12})

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)

df = pd.read_csv('../input/thesis-conference/Original (1).csv')

print(df.head(10))

print(df.info())

pd.value_counts(df['TYPE']).plot.bar()

df['TYPE'].value_counts()
application = df

categorical_list = []

numerical_list = []

for i in application.columns.tolist():

    if application[i].dtype=='object':

        categorical_list.append(i)

    else:

        numerical_list.append(i)

print('Number of categorical features:', str(len(categorical_list)))

print('Number of numerical features:', str(len(numerical_list)))

# for each column, get value counts in decreasing order and take the index (value) of most common class

df[categorical_list] = df[categorical_list].apply(lambda x: x.fillna(x.value_counts().index[0]))

df[categorical_list]

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()   # creating instance of labelencoder



for attr in categorical_list:

    df[attr]= labelencoder.fit_transform(df[attr].astype(str))



df[numerical_list] = df[numerical_list].replace([np.inf, -np.inf], np.nan)

df[numerical_list] = df[numerical_list].fillna(df[numerical_list].mean())

df[numerical_list] 

#df=df.drop(columns=['TIP', 'œ','Unnamed: 160','Unnamed: 39','Unnamed: 41','EMERGENCYCESAREANSECTION','apgarfetus1fivemin','weightfetus1','MATERNALEDUCATION','ALIVENEWBORNS','ANESTHESIA'])

df=df.drop(columns=['TIP', 'œ','Unnamed: 160','Unnamed: 39','Unnamed: 41','OXYTOCIN','HOURSOFRUPTUREDMEMBRANESATDELIVERY','EMERGENCYCESAREANSECTION','FETUS1ADMISSIONICU','TEAR','ANESTHESIA','EPISIOTOMY','INDUCTION','ALIVENEWBORNS','apgarfetus1fivemin','FETUS1RECOVERY','apgarfetus1'])
data=df

X = data.loc[:, data.columns != 'TYPE']

y=data['TYPE']

print(X.head())

feature_name = X.columns.tolist()

print(y.head())

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

from sklearn.preprocessing import MinMaxScaler

X_norm = MinMaxScaler().fit_transform(X)

chi_selector = SelectKBest(chi2, k=11)

chi_selector.fit(X_norm, y)

chi_support = chi_selector.get_support()

chi_feature = X.loc[:,chi_support].columns.tolist()

print(str(len(chi_feature)), 'selected features')

chi_feature
df_filtered_sec = df[df['TYPE'] == 5] 

df_filtered_natural = df[df['TYPE'] != 5] 

df_filtered_sec['TYPE']=0

df_filtered_natural['TYPE']=1



frames = [df_filtered_sec, df_filtered_natural]

df = pd.concat(frames)

df.info()

pd.value_counts(df['TYPE']).plot.bar()

df['TYPE'].value_counts()
y=df['TYPE']

y
# df = df.rename(columns={'GAGE': 'gestational age', 'PARITY': 'parity', 'ORISK':'obsetrick risk','COMORBIDITY':'comorbidity','NPREVC':'number of previous cesarean','PREVC':'previous cesarean? (t/f)','INCREASED':'weight increased during pregnency','CARRE':'start week of antenatal care','HEIGHT':'height','WEIGHT':'weight','BMI':'body mass index'})



df=df[chi_feature]

df = df.rename(columns={'MEDICALINDICATION':'Medical Indication','AMNIOTICLIQUID':'Amniotic Liquid','OXYTOCIN':'Oxytocin','ARTMODE':'ART Mode','FetalINTRAPARTUMpH':'Fetal Intrapartum pH','PREINDUCTION':'Pre-Induction','HOURSOFRUPTUREDMEMBRANESATDELIVERY':'Rupture Membrane', 'AMNIOCENTESIS':'Amniocentesis', 'PARITY': 'parity', 'ORISK':'Obsetrick Risk','COMORBIDITY':'Comorbidity','NPREVC':'Number of Previous Cesarean','PREVC':'Previous Cesarean? (t/f)'})



df.head()

data=df

X = data.loc[:, data.columns != 'TYPE']

X.head()



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



print('Train Data')

bsome = pd.DataFrame(y_train)

print(bsome.value_counts())



print()

print('Test data')

tdata = pd.DataFrame(y_test)

print(tdata.value_counts())





from imblearn.over_sampling import SMOTE 

sm = SMOTE()

X_train, y_train = sm.fit_resample(X_train, y_train)

print(X_train.shape)



print('After SMOTE')



print('Train Data')

asome = pd.DataFrame(y_train)

print(asome.value_counts())



print()

print('Test data')

tdata = pd.DataFrame(y_test)

print(tdata.value_counts())

y_train
evaluation = pd.DataFrame({'Model': [],

                           'Accuracy(train)':[],

                           'Precision(train)':[],

                           'Recall(train)':[],

                           'F1_score(train)':[],

                           'Accuracy(test)':[],

                           'Precision(test)':[],

                           'Recall(test)':[],

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



box_train =  pd.DataFrame({'algorithm': [],

                           'accuracy':[]})



box_test =  pd.DataFrame({'algorithm': [],

                           'accuracy':[]})





#features = list(data.columns.values)



features=  chi_feature

print(features)
from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification

from sklearn.ensemble import RandomForestClassifier

svc = RandomForestClassifier(max_depth=2, random_state=0)

clf = AdaBoostClassifier(n_estimators=500,base_estimator=svc, random_state=43)

clf.fit(X_train, y_train)





acc_train=format(accuracy_score(clf.predict(X_train), y_train)*100,'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='weighted')*100,'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='weighted')*100,'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='weighted')*100,'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test)*100,'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='weighted')*100,'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='weighted')*100,'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='weighted')*100,'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['AB',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation









complex_model_1=clf



cv_train_acc=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy')

cv_train_acc_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy').mean(),'.3f'))



cv_train_pre=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision')

cv_train_pre_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision').mean(),'.3f'))



cv_train_re=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall')

cv_train_re_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall').mean(),'.3f'))



cv_train_f1=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro')

cv_train_f1_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro').mean()



cv_test_acc=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy')

cv_test_acc_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy').mean()



cv_test_pre=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision')

cv_test_pre_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision').mean()



cv_test_re=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall')

cv_test_re_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall').mean()



cv_test_f1=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro')

cv_test_f1_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro').mean()





r = evaluation2.shape[0]

evaluation2.loc[r] = ['AB','Train accuracy',100*float(format(cv_train_acc[0],'.3f')),100*float(format(cv_train_acc[1],'.3f')),100*float(format(cv_train_acc[2],'.3f')),100*float(format(cv_train_acc[3],'.3f')),100*float(format(cv_train_acc[4],'.3f')),100*float(format(cv_train_acc[5],'.3f')),100*float(format(cv_train_acc[6],'.3f')),100*float(format(cv_train_acc[7],'.3f')),100*float(format(cv_train_acc[8],'.3f')),100*float(format(cv_train_acc[9],'.3f')),100*float(format(cv_train_acc_m,'.3f'))]

evaluation2.loc[r+1] = ['AB','Train precision',100*float(format(cv_train_pre[0],'.3f')),100*float(format(cv_train_pre[1],'.3f')),100*float(format(cv_train_pre[2],'.3f')),100*float(format(cv_train_pre[3],'.3f')),100*float(format(cv_train_pre[4],'.3f')),100*float(format(cv_train_pre[5],'.3f')),100*float(format(cv_train_pre[6],'.3f')),100*float(format(cv_train_pre[7],'.3f')),100*float(format(cv_train_pre[8],'.3f')),100*float(format(cv_train_pre[9],'.3f')),100*float(format(cv_train_pre_m,'.3f'))]

evaluation2.loc[r+2] = ['AB','Train recall',100*float(format(cv_train_re[0],'.3f')),100*float(format(cv_train_re[1],'.3f')),100*float(format(cv_train_re[2],'.3f')),100*float(format(cv_train_re[3],'.3f')),100*float(format(cv_train_re[4],'.3f')),100*float(format(cv_train_re[5],'.3f')),100*float(format(cv_train_re[6],'.3f')),100*float(format(cv_train_re[7],'.3f')),100*float(format(cv_train_re[8],'.3f')),100*float(format(cv_train_re[9],'.3f')),100*float(format(cv_train_re_m,'.3f'))]

evaluation2.loc[r+3] = ['AB','Train f1_score',100*float(format(cv_train_f1[0],'.3f')),100*float(format(cv_train_f1[1],'.3f')),100*float(format(cv_train_f1[2],'.3f')),100*float(format(cv_train_f1[3],'.3f')),100*float(format(cv_train_f1[4],'.3f')),100*float(format(cv_train_f1[5],'.3f')),100*float(format(cv_train_f1[6],'.3f')),100*float(format(cv_train_f1[7],'.3f')),100*float(format(cv_train_f1[8],'.3f')),100*float(format(cv_train_f1[9],'.3f')),100*float(format(cv_train_f1_m,'.3f'))]

evaluation2.loc[r+4] = ['AB','Test accuracy',100*float(format(cv_test_acc[0],'.3f')),100*float(format(cv_test_acc[1],'.3f')),100*float(format(cv_test_acc[2],'.3f')),100*float(format(cv_test_acc[3],'.3f')),100*float(format(cv_test_acc[4],'.3f')),100*float(format(cv_test_acc[5],'.3f')),100*float(format(cv_test_acc[6],'.3f')),100*float(format(cv_test_acc[7],'.3f')),100*float(format(cv_test_acc[8],'.3f')),100*float(format(cv_test_acc[9],'.3f')),100*float(format(cv_test_acc_m,'.3f'))]

evaluation2.loc[r+5] = ['AB','Test precision',100*float(format(cv_test_pre[0],'.3f')),100*float(format(cv_test_pre[1],'.3f')),100*float(format(cv_test_pre[2],'.3f')),100*float(format(cv_test_pre[3],'.3f')),100*float(format(cv_test_pre[4],'.3f')),100*float(format(cv_test_pre[5],'.3f')),100*float(format(cv_test_pre[6],'.3f')),100*float(format(cv_test_pre[7],'.3f')),100*float(format(cv_test_pre[8],'.3f')),100*float(format(cv_test_pre[9],'.3f')),100*float(format(cv_test_pre_m,'.3f'))]

evaluation2.loc[r+6] = ['AB','Test recall',100*float(format(cv_test_re[0],'.3f')),100*float(format(cv_test_re[1],'.3f')),100*float(format(cv_test_re[2],'.3f')),100*float(format(cv_test_re[3],'.3f')),100*float(format(cv_test_re[4],'.3f')),100*float(format(cv_test_re[5],'.3f')),100*float(format(cv_test_re[6],'.3f')),100*float(format(cv_test_re[7],'.3f')),100*float(format(cv_test_re[8],'.3f')),100*float(format(cv_test_re[9],'.3f')),100*float(format(cv_test_re_m,'.3f'))]

evaluation2.loc[r+7] = ['AB','Test f1_score',100*float(format(cv_test_f1[0],'.3f')),100*float(format(cv_test_f1[1],'.3f')),100*float(format(cv_test_f1[2],'.3f')),100*float(format(cv_test_f1[3],'.3f')),100*float(format(cv_test_f1[4],'.3f')),100*float(format(cv_test_f1[5],'.3f')),100*float(format(cv_test_f1[6],'.3f')),100*float(format(cv_test_f1[7],'.3f')),100*float(format(cv_test_f1[8],'.3f')),100*float(format(cv_test_f1[9],'.3f')),100*float(format(cv_test_f1_m,'.3f')) ]

pickle.dump(clf, open('AdaBoost.pkl','wb'))

print(evaluation2)



r = box_train.shape[0]

box_train.loc[r] = ['AB',float(format(cv_train_acc[0]*100,'.3f'))]

box_train.loc[r+1] = ['AB',float(format(cv_train_acc[1]*100,'.3f'))]

box_train.loc[r+2] = ['AB',float(format(cv_train_acc[2]*100,'.3f'))]

box_train.loc[r+3] = ['AB',float(format(cv_train_acc[3]*100,'.3f'))]

box_train.loc[r+4] = ['AB',float(format(cv_train_acc[4]*100,'.3f'))]

box_train.loc[r+5] = ['AB',float(format(cv_train_acc[5]*100,'.3f'))]

box_train.loc[r+6] = ['AB',float(format(cv_train_acc[6]*100,'.3f'))]

box_train.loc[r+7] = ['AB',float(format(cv_train_acc[7]*100,'.3f'))]

box_train.loc[r+8] = ['AB',float(format(cv_train_acc[8]*100,'.3f'))]

box_train.loc[r+9] = ['AB',float(format(cv_train_acc[9]*100,'.3f'))]





r = box_test.shape[0]

box_test.loc[r] = ['AB',float(format(cv_test_acc[0]*100,'.3f'))]

box_test.loc[r+1] = ['AB',float(format(cv_test_acc[1]*100,'.3f'))]

box_test.loc[r+2] = ['AB',float(format(cv_test_acc[2]*100,'.3f'))]

box_test.loc[r+3] = ['AB',float(format(cv_test_acc[3]*100,'.3f'))]

box_test.loc[r+4] = ['AB',float(format(cv_test_acc[4]*100,'.3f'))]

box_test.loc[r+5] = ['AB',float(format(cv_test_acc[5]*100,'.3f'))]

box_test.loc[r+6] = ['AB',float(format(cv_test_acc[6]*100,'.3f'))]

box_test.loc[r+7] = ['AB',float(format(cv_test_acc[7]*100,'.3f'))]

box_test.loc[r+8] = ['AB',float(format(cv_test_acc[8]*100,'.3f'))]

box_test.loc[r+9] = ['AB',float(format(cv_test_acc[9]*100,'.3f'))]

box_test
features = list(X.columns.values)

importances = clf.feature_importances_

importances=importances*100

import numpy as np

indices = np.argsort(importances)

#plt.title('Feature Importances')

plt.figure(figsize=(6,8))

plt.barh(range(len(indices)), importances[indices], color='gbry', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

#plt.xlabel('Relative Importance')



plt.show()

print(importances)
p=y_train

q=y_test



y_train = pd.DataFrame(y_train)

y_train=y_train.replace([0,1], ["Cesarean","Non Cesarean"])



pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)

pred_train=pred_train.replace([0,1], ["Cesarean","Non Cesarean"])





pred_test=clf.predict(X_test)

y_test = pd.DataFrame(y_test)

y_test=y_test.replace([0,1], ["Cesarean","Non Cesarean"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1], ["Cesarean","Non Cesarean"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )

y_train=p

y_test=q
from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification

from catboost import CatBoostClassifier



clf = CatBoostClassifier(

    iterations=1000, 

    learning_rate=0.1, 

    #verbose=5,

    #loss_function='CrossEntropy'

)

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train)*100,'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='weighted')*100,'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='weighted')*100,'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='weighted')*100,'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test)*100,'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='weighted')*100,'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='weighted')*100,'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='weighted')*100,'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['CB',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation









complex_model_1=clf



cv_train_acc=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy')

cv_train_acc_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy').mean(),'.3f'))



cv_train_pre=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision')

cv_train_pre_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision').mean(),'.3f'))



cv_train_re=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall')

cv_train_re_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall').mean(),'.3f'))



cv_train_f1=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro')

cv_train_f1_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro').mean()



cv_test_acc=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy')

cv_test_acc_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy').mean()



cv_test_pre=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision')

cv_test_pre_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision').mean()



cv_test_re=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall')

cv_test_re_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall').mean()



cv_test_f1=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro')

cv_test_f1_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro').mean()





r = evaluation2.shape[0]

evaluation2.loc[r] = ['CB','Train accuracy',100*float(format(cv_train_acc[0],'.3f')),100*float(format(cv_train_acc[1],'.3f')),100*float(format(cv_train_acc[2],'.3f')),100*float(format(cv_train_acc[3],'.3f')),100*float(format(cv_train_acc[4],'.3f')),100*float(format(cv_train_acc[5],'.3f')),100*float(format(cv_train_acc[6],'.3f')),100*float(format(cv_train_acc[7],'.3f')),100*float(format(cv_train_acc[8],'.3f')),100*float(format(cv_train_acc[9],'.3f')),100*float(format(cv_train_acc_m,'.3f'))]

evaluation2.loc[r+1] = ['CB','Train precision',100*float(format(cv_train_pre[0],'.3f')),100*float(format(cv_train_pre[1],'.3f')),100*float(format(cv_train_pre[2],'.3f')),100*float(format(cv_train_pre[3],'.3f')),100*float(format(cv_train_pre[4],'.3f')),100*float(format(cv_train_pre[5],'.3f')),100*float(format(cv_train_pre[6],'.3f')),100*float(format(cv_train_pre[7],'.3f')),100*float(format(cv_train_pre[8],'.3f')),100*float(format(cv_train_pre[9],'.3f')),100*float(format(cv_train_pre_m,'.3f'))]

evaluation2.loc[r+2] = ['CB','Train recall',100*float(format(cv_train_re[0],'.3f')),100*float(format(cv_train_re[1],'.3f')),100*float(format(cv_train_re[2],'.3f')),100*float(format(cv_train_re[3],'.3f')),100*float(format(cv_train_re[4],'.3f')),100*float(format(cv_train_re[5],'.3f')),100*float(format(cv_train_re[6],'.3f')),100*float(format(cv_train_re[7],'.3f')),100*float(format(cv_train_re[8],'.3f')),100*float(format(cv_train_re[9],'.3f')),100*float(format(cv_train_re_m,'.3f'))]

evaluation2.loc[r+3] = ['CB','Train f1_score',100*float(format(cv_train_f1[0],'.3f')),100*float(format(cv_train_f1[1],'.3f')),100*float(format(cv_train_f1[2],'.3f')),100*float(format(cv_train_f1[3],'.3f')),100*float(format(cv_train_f1[4],'.3f')),100*float(format(cv_train_f1[5],'.3f')),100*float(format(cv_train_f1[6],'.3f')),100*float(format(cv_train_f1[7],'.3f')),100*float(format(cv_train_f1[8],'.3f')),100*float(format(cv_train_f1[9],'.3f')),100*float(format(cv_train_f1_m,'.3f'))]

evaluation2.loc[r+4] = ['CB','Test accuracy',100*float(format(cv_test_acc[0],'.3f')),100*float(format(cv_test_acc[1],'.3f')),100*float(format(cv_test_acc[2],'.3f')),100*float(format(cv_test_acc[3],'.3f')),100*float(format(cv_test_acc[4],'.3f')),100*float(format(cv_test_acc[5],'.3f')),100*float(format(cv_test_acc[6],'.3f')),100*float(format(cv_test_acc[7],'.3f')),100*float(format(cv_test_acc[8],'.3f')),100*float(format(cv_test_acc[9],'.3f')),100*float(format(cv_test_acc_m,'.3f'))]

evaluation2.loc[r+5] = ['CB','Test precision',100*float(format(cv_test_pre[0],'.3f')),100*float(format(cv_test_pre[1],'.3f')),100*float(format(cv_test_pre[2],'.3f')),100*float(format(cv_test_pre[3],'.3f')),100*float(format(cv_test_pre[4],'.3f')),100*float(format(cv_test_pre[5],'.3f')),100*float(format(cv_test_pre[6],'.3f')),100*float(format(cv_test_pre[7],'.3f')),100*float(format(cv_test_pre[8],'.3f')),100*float(format(cv_test_pre[9],'.3f')),100*float(format(cv_test_pre_m,'.3f'))]

evaluation2.loc[r+6] = ['CB','Test recall',100*float(format(cv_test_re[0],'.3f')),100*float(format(cv_test_re[1],'.3f')),100*float(format(cv_test_re[2],'.3f')),100*float(format(cv_test_re[3],'.3f')),100*float(format(cv_test_re[4],'.3f')),100*float(format(cv_test_re[5],'.3f')),100*float(format(cv_test_re[6],'.3f')),100*float(format(cv_test_re[7],'.3f')),100*float(format(cv_test_re[8],'.3f')),100*float(format(cv_test_re[9],'.3f')),100*float(format(cv_test_re_m,'.3f'))]

evaluation2.loc[r+7] = ['CB','Test f1_score',100*float(format(cv_test_f1[0],'.3f')),100*float(format(cv_test_f1[1],'.3f')),100*float(format(cv_test_f1[2],'.3f')),100*float(format(cv_test_f1[3],'.3f')),100*float(format(cv_test_f1[4],'.3f')),100*float(format(cv_test_f1[5],'.3f')),100*float(format(cv_test_f1[6],'.3f')),100*float(format(cv_test_f1[7],'.3f')),100*float(format(cv_test_f1[8],'.3f')),100*float(format(cv_test_f1[9],'.3f')),100*float(format(cv_test_f1_m,'.3f')) ]

pickle.dump(clf, open('CB.pkl','wb'))

print(evaluation2)



r = box_train.shape[0]

box_train.loc[r] = ['CB',float(format(cv_train_acc[0]*100,'.3f'))]

box_train.loc[r+1] = ['CB',float(format(cv_train_acc[1]*100,'.3f'))]

box_train.loc[r+2] = ['CB',float(format(cv_train_acc[2]*100,'.3f'))]

box_train.loc[r+3] = ['CB',float(format(cv_train_acc[3]*100,'.3f'))]

box_train.loc[r+4] = ['CB',float(format(cv_train_acc[4]*100,'.3f'))]

box_train.loc[r+5] = ['CB',float(format(cv_train_acc[5]*100,'.3f'))]

box_train.loc[r+6] = ['CB',float(format(cv_train_acc[6]*100,'.3f'))]

box_train.loc[r+7] = ['CB',float(format(cv_train_acc[7]*100,'.3f'))]

box_train.loc[r+8] = ['CB',float(format(cv_train_acc[8]*100,'.3f'))]

box_train.loc[r+9] = ['CB',float(format(cv_train_acc[9]*100,'.3f'))]





r = box_test.shape[0]

box_test.loc[r] = ['CB',float(format(cv_test_acc[0]*100,'.3f'))]

box_test.loc[r+1] = ['CB',float(format(cv_test_acc[1]*100,'.3f'))]

box_test.loc[r+2] = ['CB',float(format(cv_test_acc[2]*100,'.3f'))]

box_test.loc[r+3] = ['CB',float(format(cv_test_acc[3]*100,'.3f'))]

box_test.loc[r+4] = ['CB',float(format(cv_test_acc[4]*100,'.3f'))]

box_test.loc[r+5] = ['CB',float(format(cv_test_acc[5]*100,'.3f'))]

box_test.loc[r+6] = ['CB',float(format(cv_test_acc[6]*100,'.3f'))]

box_test.loc[r+7] = ['CB',float(format(cv_test_acc[7]*100,'.3f'))]

box_test.loc[r+8] = ['CB',float(format(cv_test_acc[8]*100,'.3f'))]

box_test.loc[r+9] = ['CB',float(format(cv_test_acc[9]*100,'.3f'))]

features = list(X.columns.values)

importances = clf.feature_importances_

importances=importances*100

import numpy as np

indices = np.argsort(importances)

#plt.title('Feature Importances')

plt.figure(figsize=(6,8))

plt.barh(range(len(indices)), importances[indices], color='gbry', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

#plt.xlabel('Relative Importance')



plt.show()

print(importances)
p=y_train

q=y_test



y_train = pd.DataFrame(y_train)

y_train=y_train.replace([0,1], ["Cesarean","Non Cesarean"])



pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)

pred_train=pred_train.replace([0,1], ["Cesarean","Non Cesarean"])





pred_test=clf.predict(X_test)

y_test = pd.DataFrame(y_test)

y_test=y_test.replace([0,1], ["Cesarean","Non Cesarean"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1], ["Cesarean","Non Cesarean"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )

y_train=p

y_test=q
from sklearn.ensemble import AdaBoostClassifier

from sklearn.datasets import make_classification

import xgboost as xgb

clf = xgb.XGBClassifier(n_estimators=2000)

clf.fit(X_train, y_train)



acc_train=format(accuracy_score(clf.predict(X_train), y_train)*100,'.3f')

precision_train=format(precision_score(y_train, clf.predict(X_train), average='weighted')*100,'.3f')

recall_train=format(recall_score(y_train,clf.predict(X_train), average='weighted')*100,'.3f')

f1_train=format(f1_score(y_train,clf.predict(X_train), average='weighted')*100,'.3f')





acc_test=format(accuracy_score(clf.predict(X_test), y_test)*100,'.3f')

precision_test=format(precision_score(y_test, clf.predict(X_test), average='weighted')*100,'.3f')

recall_test=format(recall_score(y_test,clf.predict(X_test), average='weighted')*100,'.3f')

f1_test=format(f1_score(y_test,clf.predict(X_test), average='weighted')*100,'.3f')



r = evaluation.shape[0]

evaluation.loc[r] = ['XB',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation









complex_model_1=clf



cv_train_acc=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy')

cv_train_acc_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='accuracy').mean(),'.3f'))



cv_train_pre=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision')

cv_train_pre_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='precision').mean(),'.3f'))



cv_train_re=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall')

cv_train_re_m=float(format(cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='recall').mean(),'.3f'))



cv_train_f1=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro')

cv_train_f1_m=cross_val_score(complex_model_1,X_train, y_train,cv=10,scoring='f1_macro').mean()



cv_test_acc=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy')

cv_test_acc_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='accuracy').mean()



cv_test_pre=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision')

cv_test_pre_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='precision').mean()



cv_test_re=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall')

cv_test_re_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='recall').mean()



cv_test_f1=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro')

cv_test_f1_m=cross_val_score(complex_model_1,X_test, y_test,cv=10,scoring='f1_macro').mean()





r = evaluation2.shape[0]

evaluation2.loc[r] = ['XB','Train accuracy',100*float(format(cv_train_acc[0],'.3f')),100*float(format(cv_train_acc[1],'.3f')),100*float(format(cv_train_acc[2],'.3f')),100*float(format(cv_train_acc[3],'.3f')),100*float(format(cv_train_acc[4],'.3f')),100*float(format(cv_train_acc[5],'.3f')),100*float(format(cv_train_acc[6],'.3f')),100*float(format(cv_train_acc[7],'.3f')),100*float(format(cv_train_acc[8],'.3f')),100*float(format(cv_train_acc[9],'.3f')),100*float(format(cv_train_acc_m,'.3f'))]

evaluation2.loc[r+1] = ['XB','Train precision',100*float(format(cv_train_pre[0],'.3f')),100*float(format(cv_train_pre[1],'.3f')),100*float(format(cv_train_pre[2],'.3f')),100*float(format(cv_train_pre[3],'.3f')),100*float(format(cv_train_pre[4],'.3f')),100*float(format(cv_train_pre[5],'.3f')),100*float(format(cv_train_pre[6],'.3f')),100*float(format(cv_train_pre[7],'.3f')),100*float(format(cv_train_pre[8],'.3f')),100*float(format(cv_train_pre[9],'.3f')),100*float(format(cv_train_pre_m,'.3f'))]

evaluation2.loc[r+2] = ['XB','Train recall',100*float(format(cv_train_re[0],'.3f')),100*float(format(cv_train_re[1],'.3f')),100*float(format(cv_train_re[2],'.3f')),100*float(format(cv_train_re[3],'.3f')),100*float(format(cv_train_re[4],'.3f')),100*float(format(cv_train_re[5],'.3f')),100*float(format(cv_train_re[6],'.3f')),100*float(format(cv_train_re[7],'.3f')),100*float(format(cv_train_re[8],'.3f')),100*float(format(cv_train_re[9],'.3f')),100*float(format(cv_train_re_m,'.3f'))]

evaluation2.loc[r+3] = ['XB','Train f1_score',100*float(format(cv_train_f1[0],'.3f')),100*float(format(cv_train_f1[1],'.3f')),100*float(format(cv_train_f1[2],'.3f')),100*float(format(cv_train_f1[3],'.3f')),100*float(format(cv_train_f1[4],'.3f')),100*float(format(cv_train_f1[5],'.3f')),100*float(format(cv_train_f1[6],'.3f')),100*float(format(cv_train_f1[7],'.3f')),100*float(format(cv_train_f1[8],'.3f')),100*float(format(cv_train_f1[9],'.3f')),100*float(format(cv_train_f1_m,'.3f'))]

evaluation2.loc[r+4] = ['XB','Test accuracy',100*float(format(cv_test_acc[0],'.3f')),100*float(format(cv_test_acc[1],'.3f')),100*float(format(cv_test_acc[2],'.3f')),100*float(format(cv_test_acc[3],'.3f')),100*float(format(cv_test_acc[4],'.3f')),100*float(format(cv_test_acc[5],'.3f')),100*float(format(cv_test_acc[6],'.3f')),100*float(format(cv_test_acc[7],'.3f')),100*float(format(cv_test_acc[8],'.3f')),100*float(format(cv_test_acc[9],'.3f')),100*float(format(cv_test_acc_m,'.3f'))]

evaluation2.loc[r+5] = ['XB','Test precision',100*float(format(cv_test_pre[0],'.3f')),100*float(format(cv_test_pre[1],'.3f')),100*float(format(cv_test_pre[2],'.3f')),100*float(format(cv_test_pre[3],'.3f')),100*float(format(cv_test_pre[4],'.3f')),100*float(format(cv_test_pre[5],'.3f')),100*float(format(cv_test_pre[6],'.3f')),100*float(format(cv_test_pre[7],'.3f')),100*float(format(cv_test_pre[8],'.3f')),100*float(format(cv_test_pre[9],'.3f')),100*float(format(cv_test_pre_m,'.3f'))]

evaluation2.loc[r+6] = ['XB','Test recall',100*float(format(cv_test_re[0],'.3f')),100*float(format(cv_test_re[1],'.3f')),100*float(format(cv_test_re[2],'.3f')),100*float(format(cv_test_re[3],'.3f')),100*float(format(cv_test_re[4],'.3f')),100*float(format(cv_test_re[5],'.3f')),100*float(format(cv_test_re[6],'.3f')),100*float(format(cv_test_re[7],'.3f')),100*float(format(cv_test_re[8],'.3f')),100*float(format(cv_test_re[9],'.3f')),100*float(format(cv_test_re_m,'.3f'))]

evaluation2.loc[r+7] = ['XB','Test f1_score',100*float(format(cv_test_f1[0],'.3f')),100*float(format(cv_test_f1[1],'.3f')),100*float(format(cv_test_f1[2],'.3f')),100*float(format(cv_test_f1[3],'.3f')),100*float(format(cv_test_f1[4],'.3f')),100*float(format(cv_test_f1[5],'.3f')),100*float(format(cv_test_f1[6],'.3f')),100*float(format(cv_test_f1[7],'.3f')),100*float(format(cv_test_f1[8],'.3f')),100*float(format(cv_test_f1[9],'.3f')),100*float(format(cv_test_f1_m,'.3f')) ]

pickle.dump(clf, open('XB.pkl','wb'))

print(evaluation2)



r = box_train.shape[0]

box_train.loc[r] = ['XB',float(format(cv_train_acc[0]*100,'.3f'))]

box_train.loc[r+1] = ['XB',float(format(cv_train_acc[1]*100,'.3f'))]

box_train.loc[r+2] = ['XB',float(format(cv_train_acc[2]*100,'.3f'))]

box_train.loc[r+3] = ['XB',float(format(cv_train_acc[3]*100,'.3f'))]

box_train.loc[r+4] = ['XB',float(format(cv_train_acc[4]*100,'.3f'))]

box_train.loc[r+5] = ['XB',float(format(cv_train_acc[5]*100,'.3f'))]

box_train.loc[r+6] = ['XB',float(format(cv_train_acc[6]*100,'.3f'))]

box_train.loc[r+7] = ['XB',float(format(cv_train_acc[7]*100,'.3f'))]

box_train.loc[r+8] = ['XB',float(format(cv_train_acc[8]*100,'.3f'))]

box_train.loc[r+9] = ['XB',float(format(cv_train_acc[9]*100,'.3f'))]





r = box_test.shape[0]

box_test.loc[r] = ['XB',float(format(cv_test_acc[0]*100,'.3f'))]

box_test.loc[r+1] = ['XB',float(format(cv_test_acc[1]*100,'.3f'))]

box_test.loc[r+2] = ['XB',float(format(cv_test_acc[2]*100,'.3f'))]

box_test.loc[r+3] = ['XB',float(format(cv_test_acc[3]*100,'.3f'))]

box_test.loc[r+4] = ['XB',float(format(cv_test_acc[4]*100,'.3f'))]

box_test.loc[r+5] = ['XB',float(format(cv_test_acc[5]*100,'.3f'))]

box_test.loc[r+6] = ['XB',float(format(cv_test_acc[6]*100,'.3f'))]

box_test.loc[r+7] = ['XB',float(format(cv_test_acc[7]*100,'.3f'))]

box_test.loc[r+8] = ['XB',float(format(cv_test_acc[8]*100,'.3f'))]

box_test.loc[r+9] = ['XB',float(format(cv_test_acc[9]*100,'.3f'))]
features = list(X.columns.values)

importances = clf.feature_importances_

importances=importances*100

import numpy as np

indices = np.argsort(importances)

#plt.title('Feature Importances')

plt.figure(figsize=(6,8))

plt.barh(range(len(indices)), importances[indices], color='gbry', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

#plt.xlabel('Relative Importance')



plt.show()

print(importances)
p=y_train

q=y_test



y_train = pd.DataFrame(y_train)

y_train=y_train.replace([0,1], ["Cesarean","Non Cesarean"])



pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)

pred_train=pred_train.replace([0,1], ["Cesarean","Non Cesarean"])





pred_test=clf.predict(X_test)

y_test = pd.DataFrame(y_test)

y_test=y_test.replace([0,1], ["Cesarean","Non Cesarean"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0,1], ["Cesarean","Non Cesarean"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(7,4),

    title_fontsize='18',

    text_fontsize='16',

    title =' ',

    cmap='BuGn'

    )

y_train=p

y_test=q
evaluation.to_csv('model_results.csv')

evaluation2.to_csv('cross_validation.csv')

evaluation
# evaluation['Precision(test)']= evaluation['Precision(test)'].astype(float)*100

acc_train= evaluation['Accuracy(train)'].astype(float)

pre_train=evaluation['Precision(train)'].astype(float)

recall_train=evaluation['Recall(train)'].astype(float)

f1_train=evaluation['F1_score(train)'].astype(float)



acc_test= evaluation['Accuracy(test)'].astype(float)

pre_test=evaluation['Precision(test)'].astype(float)

recall_test =evaluation['Recall(test)'].astype(float)

f1_train=evaluation['F1_score(test)'].astype(float)
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))



# line 1 points

x1 = ['AdaBoost','CatBoost','XgBoost']

# plotting the line 1 points 

plt.plot(x1, acc_train, label = "Accuracy")

plt.plot(x1, pre_train, label = "Precision")

plt.plot(x1, recall_train, label = "Recall")

plt.plot(x1, f1_train, label = "F1-Score")

plt.legend()

# Display a figure.



plt.show()
import matplotlib.pyplot as plt

plt.figure(figsize=(6,6))



# line 1 points

x1 = ['AdaBoost','CatBoost','XgBoost']

# plotting the line 1 points 

plt.plot(x1, acc_test, label = "Accuracy")

plt.plot(x1, pre_test, label = "Precision")

plt.plot(x1, recall_test, label = "Recall")

plt.plot(x1, f1_test, label = "F1-Score")

plt.legend()

# Display a figure.



plt.show()
import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

df = box_train

df['accuracy']=df['accuracy']

sns.boxplot( x=df["algorithm"], y=df["accuracy"], width=0.8, saturation=0.95)



plt.xlabel('Models')

plt.ylabel('Cross-Validation Accuracy(%)(Train Data)')



plt.show()

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd

df = box_test

df['accuracy']=df['accuracy']

sns.boxplot( x=df["algorithm"], y=df["accuracy"], width=0.8, saturation=0.95)



plt.xlabel('Models')

plt.ylabel('Cross-Validation Accuracy(%)(Test Data)')



plt.show()