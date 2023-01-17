%matplotlib inline

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

from sklearn import linear_model

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn import metrics

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier





df = pd.read_csv('../input/thesis-childbirth/Original.csv')



print(df.head())

print(df.info())
df['AGE']=df['AGE'].astype(float)

df['GAGE']=df['GAGE'].astype(float)

df['PARITY']=df['PARITY'].astype(float)

df['NPREVC']=df['NPREVC'].astype(float)

df['INCREASED']=df['INCREASED'].astype(float)

df['MISCARRIAGES']=df['MISCARRIAGES'].astype(float)

df['PREVIOUSTERMPREGNANCIES']=df['PREVIOUSTERMPREGNANCIES'].astype(float)

df['PREVIOUSPRETERMPREGNANCIES']=df['PREVIOUSPRETERMPREGNANCIES'].astype(float)



df["TYPE"] = df["TYPE"].astype('category')

df['TYPE']=df['TYPE'].cat.codes

df["TYPE"]=df["TYPE"].astype('float')



df["ORISK"] = df["ORISK"].astype('category')

df['ORISK']=df['ORISK'].cat.codes

df["ORISK"]=df["ORISK"].astype('float')



df["COMORBIDITY"] = df["COMORBIDITY"].astype('category')

df['COMORBIDITY']=df['COMORBIDITY'].cat.codes

df["COMORBIDITY"]=df["COMORBIDITY"].astype('float')



df["PREVC"] = df["PREVC"].astype('category')

df['PREVC']=df['PREVC'].cat.codes

df["PREVC"]=df["PREVC"].astype('float')



df["CARRE"] = df["CARRE"].astype('category')

df['CARRE']=df['CARRE'].cat.codes

df["CARRE"]=df["CARRE"].astype('float')



df["COMPLICATIONS"] = df["COMPLICATIONS"].astype('category')

df['COMPLICATIONS']=df['COMPLICATIONS'].cat.codes

df["COMPLICATIONS"]=df["COMPLICATIONS"].astype('float')



df["ROBSONGROUP"] = df["ROBSONGROUP"].astype('category')

df['ROBSONGROUP']=df['ROBSONGROUP'].cat.codes

df["ROBSONGROUP"]=df["ROBSONGROUP"].astype('float')



df["ART"] = df["ART"].astype('category')

df['ART']=df['ART'].cat.codes

df["ART"]=df["ART"].astype('float')



df["ARTMODE"] = df["ARTMODE"].astype('category')

df['ARTMODE']=df['ARTMODE'].cat.codes

df["ARTMODE"]=df["ARTMODE"].astype('float')



df["AMNIOTICLIQUID"] = df["AMNIOTICLIQUID"].astype('category')

df['AMNIOTICLIQUID']=df['AMNIOTICLIQUID'].cat.codes

df["AMNIOTICLIQUID"]=df["AMNIOTICLIQUID"].astype('float')



df["CARDIOTOCOGRAPHY"] = df["CARDIOTOCOGRAPHY"].astype('category')

df['CARDIOTOCOGRAPHY']=df['CARDIOTOCOGRAPHY'].cat.codes

df["CARDIOTOCOGRAPHY"]=df["CARDIOTOCOGRAPHY"].astype('float')



df["AMNIOCENTESIS"] = df["AMNIOCENTESIS"].astype('category')

df['AMNIOCENTESIS']=df['AMNIOCENTESIS'].cat.codes

df["AMNIOCENTESIS"]=df["AMNIOCENTESIS"].astype('float')



df["MATERNALEDUCATION"] = df["MATERNALEDUCATION"].astype('category')

df['MATERNALEDUCATION']=df['MATERNALEDUCATION'].cat.codes

df["MATERNALEDUCATION"]=df["MATERNALEDUCATION"].astype('float')



df["SUBSTANCEABUSE"] = df["SUBSTANCEABUSE"].astype('category')

df['SUBSTANCEABUSE']=df['SUBSTANCEABUSE'].cat.codes

df["SUBSTANCEABUSE"]=df["SUBSTANCEABUSE"].astype('float')



df["SMOKING"] = df["SMOKING"].astype('category')

df['SMOKING']=df['SMOKING'].cat.codes

df["SMOKING"]=df["SMOKING"].astype('float')



df["ALCOHOL"] = df["ALCOHOL"].astype('category')

df['ALCOHOL']=df['ALCOHOL'].cat.codes

df["ALCOHOL"]=df["ALCOHOL"].astype('float')



df["ALCOHOL"] = df["ALCOHOL"].astype('category')

df['ALCOHOL']=df['ALCOHOL'].cat.codes

df["ALCOHOL"]=df["ALCOHOL"].astype('float')



df["ANESTHESIA"] = df["ANESTHESIA"].astype('category')

df['ANESTHESIA']=df['ANESTHESIA'].cat.codes

df["ANESTHESIA"]=df["ANESTHESIA"].astype('float')



df["EPISIOTOMY"] = df["EPISIOTOMY"].astype('category')

df['EPISIOTOMY']=df['EPISIOTOMY'].cat.codes

df["EPISIOTOMY"]=df["EPISIOTOMY"].astype('float')



df["OXYTOCIN"] = df["OXYTOCIN"].astype('category')

df['OXYTOCIN']=df['OXYTOCIN'].cat.codes

df["OXYTOCIN"]=df["OXYTOCIN"].astype('float')



df["FetalINTRAPARTUMpH"] = df["FetalINTRAPARTUMpH"].astype('category')

df['FetalINTRAPARTUMpH']=df['FetalINTRAPARTUMpH'].cat.codes

df["FetalINTRAPARTUMpH"]=df["FetalINTRAPARTUMpH"].astype('float')



df["PREINDUCTION"] = df["PREINDUCTION"].astype('category')

df['PREINDUCTION']=df['PREINDUCTION'].cat.codes

df["PREINDUCTION"]=df["PREINDUCTION"].astype('float')



df["INDUCTION"] = df["INDUCTION"].astype('category')

df['INDUCTION']=df['INDUCTION'].cat.codes

df["INDUCTION"]=df["INDUCTION"].astype('float')

df.replace(-1, df.mean(), inplace=True) 

print(df.head())

print(df.info())

all_df=df[['TYPE','GAGE','PARITY','ORISK','COMORBIDITY','NPREVC','PREVC','INCREASED','CARRE','HEIGHT','WEIGHT', 'BMI','AGE','COMPLICATIONS','ROBSONGROUP','ART', 'ARTMODE', 'PREVIOUSTERMPREGNANCIES', 'PREVIOUSPRETERMPREGNANCIES', 'AMNIOTICLIQUID','MISCARRIAGES','CARDIOTOCOGRAPHY','AMNIOCENTESIS','MATERNALEDUCATION','SUBSTANCEABUSE','SMOKING','ALCOHOL','PREINDUCTION', 'INDUCTION','ANESTHESIA','EPISIOTOMY','OXYTOCIN', 'FetalINTRAPARTUMpH']]

all_df.head()

df=all_df



df = df.replace([np.inf, -np.inf], np.nan)

df = df.fillna(df.mean())

print(df.head())
# df['TYPE']=df['TYPE'].astype(str)

pd.value_counts(df['TYPE']).plot.bar()

# plt.title('Fraud class histogram')

# plt.xlabel('Class')

# plt.ylabel('Frequency')

df['TYPE'].value_counts()
from imblearn.over_sampling import SMOTE

# for reproducibility purposes

seed = 100

# SMOTE number of neighbors

k = 3

from matplotlib import pyplot

from imblearn.over_sampling import ADASYN

from collections import Counter



X = df.loc[:, df.columns != 'TYPE']

y = df.TYPE

# oversample = SMOTE()

oversample = ADASYN()

X_res, y_res = oversample.fit_resample(X, y)



counter = Counter(y_res)

for k,v in counter.items():

	per = v / len(y) * 100

	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

# plot the distribution

pyplot.bar(counter.keys(), counter.values())

pyplot.show()



# plt.title('base')

# plt.xlabel('x')

# plt.ylabel('y')

# plt.scatter(X_res[:, 0], X_res[:, 1], marker='o', c=y_res,

#            s=25, edgecolor='k', cmap=plt.cm.coolwarm)

# plt.show()



df = pd.concat([pd.DataFrame(y_res), pd.DataFrame(X_res) ], axis=1)

df.columns = ['TYPE']+['GAGE']+['PARITY']+['ORISK']+['COMORBIDITY']+['NPREVC']+['PREVC']+['INCREASED']+['CARRE']+['HEIGHT']+['WEIGHT']+[ 'BMI']+['AGE']+['COMPLICATIONS']+['ROBSONGROUP']+['ART']+[ 'ARTMODE']+[ 'PREVIOUSTERMPREGNANCIES']+[ 'PREVIOUSPRETERMPREGNANCIES']+[ 'AMNIOTICLIQUID']+['MISCARRIAGES']+['CARDIOTOCOGRAPHY']+['AMNIOCENTESIS']+['MATERNALEDUCATION']+['SUBSTANCEABUSE']+['SMOKING']+['ALCOHOL']+['PREINDUCTION']+['INDUCTION']+['ANESTHESIA']+['EPISIOTOMY']+['OXYTOCIN']+['FetalINTRAPARTUMpH']

df.to_csv('df_smoted.csv', index=False, encoding='utf-8')

print(df.columns)

print(df.info())

print(df.head())
X.head()
class_1_df = df[['TYPE','GAGE','PARITY','ORISK','COMORBIDITY','NPREVC','PREVC','INCREASED','CARRE','HEIGHT','WEIGHT', 'BMI']]

class_2_df = df[['TYPE','AGE','COMPLICATIONS','ROBSONGROUP','ART', 'ARTMODE', 'PREVIOUSTERMPREGNANCIES', 'PREVIOUSPRETERMPREGNANCIES', 'AMNIOTICLIQUID','MISCARRIAGES','CARDIOTOCOGRAPHY','AMNIOCENTESIS']]

class_3_df = df[['TYPE','MATERNALEDUCATION','SUBSTANCEABUSE','SMOKING','ALCOHOL','PREINDUCTION', 'INDUCTION','ANESTHESIA','EPISIOTOMY','OXYTOCIN', 'FetalINTRAPARTUMpH']]

class_4_df = df[['TYPE','GAGE','PARITY','ORISK','COMORBIDITY','NPREVC','PREVC','INCREASED','CARRE','HEIGHT','WEIGHT', 'BMI','AGE','COMPLICATIONS','ROBSONGROUP','ART', 'ARTMODE', 'PREVIOUSTERMPREGNANCIES', 'PREVIOUSPRETERMPREGNANCIES', 'AMNIOTICLIQUID','MISCARRIAGES','CARDIOTOCOGRAPHY','AMNIOCENTESIS']]

class_5_df = df[['TYPE','GAGE','PARITY','ORISK','COMORBIDITY','NPREVC','PREVC','INCREASED','CARRE','HEIGHT','WEIGHT', 'BMI','MATERNALEDUCATION','SUBSTANCEABUSE','SMOKING','ALCOHOL','PREINDUCTION', 'INDUCTION','ANESTHESIA','EPISIOTOMY','OXYTOCIN', 'FetalINTRAPARTUMpH']]

class_6_df = df[['TYPE','AGE','COMPLICATIONS','ROBSONGROUP','ART', 'ARTMODE', 'PREVIOUSTERMPREGNANCIES', 'PREVIOUSPRETERMPREGNANCIES', 'AMNIOTICLIQUID','MISCARRIAGES','CARDIOTOCOGRAPHY','AMNIOCENTESIS','MATERNALEDUCATION','SUBSTANCEABUSE','SMOKING','ALCOHOL','PREINDUCTION', 'INDUCTION','ANESTHESIA','EPISIOTOMY','OXYTOCIN', 'FetalINTRAPARTUMpH']]

class_7_df = df[['TYPE','GAGE','PARITY','ORISK','COMORBIDITY','NPREVC','PREVC','INCREASED','CARRE','HEIGHT','WEIGHT', 'BMI','AGE','COMPLICATIONS','ROBSONGROUP','ART', 'ARTMODE', 'PREVIOUSTERMPREGNANCIES', 'PREVIOUSPRETERMPREGNANCIES', 'AMNIOTICLIQUID','MISCARRIAGES','CARDIOTOCOGRAPHY','AMNIOCENTESIS','MATERNALEDUCATION','SUBSTANCEABUSE','SMOKING','ALCOHOL','PREINDUCTION', 'INDUCTION','ANESTHESIA','EPISIOTOMY','OXYTOCIN', 'FetalINTRAPARTUMpH']]

# all_df=df[['TYPE','GAGE','PARITY','ORISK','COMORBIDITY','NPREVC','PREVC','INCREASED','CARRE','HEIGHT','WEIGHT', 'BMI','AGE','COMPLICATIONS','ROBSONGROUP','ART', 'ARTMODE', 'PREVIOUSTERMPREGNANCIES', 'PREVIOUSPRETERMPREGNANCIES', 'AMNIOTICLIQUID','MISCARRIAGES','CARDIOTOCOGRAPHY','AMNIOCENTESIS','MATERNALEDUCATION','SUBSTANCEABUSE','SMOKING','ALCOHOL','PREINDUCTION', 'INDUCTION','ANESTHESIA','EPISIOTOMY','OXYTOCIN', 'FetalINTRAPARTUMpH']]

# all_df.head()

print(class_1_df)

print(class_2_df)

print(class_3_df)

print(class_4_df)

print(class_5_df)

print(class_6_df)

print(class_7_df)

# print(all_df)
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
X = class_1_df.loc[:, class_1_df.columns != 'TYPE']

y = class_1_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn import tree

clf = tree.DecisionTreeClassifier()

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

evaluation.loc[r] = ['Decision Tree (Model-1)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_2_df.loc[:, class_2_df.columns != 'TYPE']

y = class_2_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn import tree

clf = tree.DecisionTreeClassifier()

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

evaluation.loc[r] = ['Decision Tree (Model-2)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_3_df.loc[:, class_3_df.columns != 'TYPE']

y = class_3_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn import tree

clf = tree.DecisionTreeClassifier()

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

evaluation.loc[r] = ['Decision Tree (Model-3)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_4_df.loc[:, class_4_df.columns != 'TYPE']

y = class_4_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn import tree

clf = tree.DecisionTreeClassifier()

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

evaluation.loc[r] = ['Decision Tree (Model-4)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_5_df.loc[:, class_5_df.columns != 'TYPE']

y = class_5_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn import tree

clf = tree.DecisionTreeClassifier()

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

evaluation.loc[r] = ['Decision Tree (Model-5)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_6_df.loc[:, class_6_df.columns != 'TYPE']

y = class_6_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn import tree

clf = tree.DecisionTreeClassifier()

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

evaluation.loc[r] = ['Decision Tree (Model-6)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_7_df.loc[:, class_7_df.columns != 'TYPE']

y = class_7_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn import tree

clf = tree.DecisionTreeClassifier()

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

evaluation.loc[r] = ['Decision Tree (Model-7)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_1_df.loc[:, class_1_df.columns != 'TYPE']

y = class_1_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

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

evaluation.loc[r] = ['Random Forest (Model-1)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_2_df.loc[:, class_2_df.columns != 'TYPE']

y = class_2_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

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

evaluation.loc[r] = ['Random Forest (Model-2)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_3_df.loc[:, class_3_df.columns != 'TYPE']

y = class_3_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

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

evaluation.loc[r] = ['Random Forest (Model-3)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_4_df.loc[:, class_4_df.columns != 'TYPE']

y = class_4_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

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

evaluation.loc[r] = ['Random Forest (Model-4)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_5_df.loc[:, class_5_df.columns != 'TYPE']

y = class_5_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

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

evaluation.loc[r] = ['Random Forest (Model-5)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_6_df.loc[:, class_6_df.columns != 'TYPE']

y = class_6_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

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

evaluation.loc[r] = ['Random Forest (Model-6)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_7_df.loc[:, class_7_df.columns != 'TYPE']

y = class_7_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()

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

evaluation.loc[r] = ['Random Forest (Model-7)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_1_df.loc[:, class_1_df.columns != 'TYPE']

y = class_1_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.neighbors import KNeighborsClassifier



clf = KNeighborsClassifier(n_neighbors=1)

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

evaluation.loc[r] = ['KNN (Model-1)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_2_df.loc[:, class_2_df.columns != 'TYPE']

y = class_2_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.neighbors import KNeighborsClassifier



clf = KNeighborsClassifier(n_neighbors=1)

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

evaluation.loc[r] = ['KNN (Model-2)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_3_df.loc[:, class_3_df.columns != 'TYPE']

y = class_3_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.neighbors import KNeighborsClassifier



clf = KNeighborsClassifier(n_neighbors=1)

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

evaluation.loc[r] = ['KNN (Model-3)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_4_df.loc[:, class_4_df.columns != 'TYPE']

y = class_4_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.neighbors import KNeighborsClassifier



clf = KNeighborsClassifier(n_neighbors=1)

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

evaluation.loc[r] = ['KNN (Model-4)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_5_df.loc[:, class_5_df.columns != 'TYPE']

y = class_5_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.neighbors import KNeighborsClassifier



clf = KNeighborsClassifier(n_neighbors=1)

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

evaluation.loc[r] = ['KNN (Model-5)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_6_df.loc[:, class_6_df.columns != 'TYPE']

y = class_6_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.neighbors import KNeighborsClassifier



clf = KNeighborsClassifier(n_neighbors=1)

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

evaluation.loc[r] = ['KNN (Model-6)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_7_df.loc[:, class_7_df.columns != 'TYPE']

y = class_7_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn.neighbors import KNeighborsClassifier



clf = KNeighborsClassifier(n_neighbors=1)

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

evaluation.loc[r] = ['KNN (Model-7)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_1_df.loc[:, class_1_df.columns != 'TYPE']

y = class_1_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn import svm

clf =svm.SVC(kernel='rbf')

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

evaluation.loc[r] = ['SVM (Model-1)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_2_df.loc[:, class_2_df.columns != 'TYPE']

y = class_2_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn import svm

clf =svm.SVC(kernel='rbf')

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

evaluation.loc[r] = ['SVM (Model-2)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_3_df.loc[:, class_3_df.columns != 'TYPE']

y = class_3_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn import svm

clf =svm.SVC(kernel='rbf')

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

evaluation.loc[r] = ['SVM (Model-3)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_4_df.loc[:, class_4_df.columns != 'TYPE']

y = class_4_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn import svm

clf =svm.SVC(kernel='rbf')

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

evaluation.loc[r] = ['SVM (Model-4)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_5_df.loc[:, class_5_df.columns != 'TYPE']

y = class_5_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn import svm

clf =svm.SVC(kernel='rbf')

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

evaluation.loc[r] = ['SVM (Model-5)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_6_df.loc[:, class_6_df.columns != 'TYPE']

y = class_6_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn import svm

clf =svm.SVC(kernel='rbf')

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

evaluation.loc[r] = ['SVM (Model-6)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
X = class_7_df.loc[:, class_7_df.columns != 'TYPE']

y = class_7_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)



from sklearn import svm

clf =svm.SVC(kernel='rbf')

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

evaluation.loc[r] = ['SVM (Model-7)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from numpy import mean

from numpy import std

from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import StackingClassifier

from matplotlib import pyplot

 



X = class_1_df.loc[:, class_1_df.columns != 'TYPE']

y = class_1_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)





# get a stacking ensemble of models

def get_stacking():

	# define the base models

	level0 = list()

	level0.append(('knn', KNeighborsClassifier(n_neighbors=1)))

	level0.append(('dt', DecisionTreeClassifier()))

	level0.append(('svm', SVC()))

	level0.append(('rf', RandomForestClassifier()))

	# define meta learner model

	level1 = LogisticRegression(max_iter=10000)

	# define the stacking ensemble

	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

	return model





clf=get_stacking()



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

evaluation.loc[r] = ['Stacked (Model-1)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from numpy import mean

from numpy import std

from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import StackingClassifier

from matplotlib import pyplot

 



X = class_2_df.loc[:, class_2_df.columns != 'TYPE']

y = class_2_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)





# get a stacking ensemble of models

def get_stacking():

	# define the base models

	level0 = list()

	level0.append(('knn', KNeighborsClassifier(n_neighbors=1)))

	level0.append(('dt', DecisionTreeClassifier()))

	level0.append(('svm', SVC()))

	level0.append(('rf', RandomForestClassifier()))

	# define meta learner model

	level1 = LogisticRegression(max_iter=10000)

	# define the stacking ensemble

	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

	return model





clf=get_stacking()



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

evaluation.loc[r] = ['Stacked (Model-2)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from numpy import mean

from numpy import std

from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import StackingClassifier

from matplotlib import pyplot

 



X = class_3_df.loc[:, class_3_df.columns != 'TYPE']

y = class_3_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)





# get a stacking ensemble of models

def get_stacking():

	# define the base models

	level0 = list()

	level0.append(('knn', KNeighborsClassifier(n_neighbors=1)))

	level0.append(('dt', DecisionTreeClassifier()))

	level0.append(('svm', SVC()))

	level0.append(('rf', RandomForestClassifier()))

	# define meta learner model

	level1 = LogisticRegression(max_iter=10000)

	# define the stacking ensemble

	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

	return model





clf=get_stacking()



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

evaluation.loc[r] = ['Stacked (Model-3)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from numpy import mean

from numpy import std

from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import StackingClassifier

from matplotlib import pyplot

 



X = class_4_df.loc[:, class_4_df.columns != 'TYPE']

y = class_4_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)





# get a stacking ensemble of models

def get_stacking():

	# define the base models

	level0 = list()

	level0.append(('knn', KNeighborsClassifier(n_neighbors=1)))

	level0.append(('dt', DecisionTreeClassifier()))

	level0.append(('svm', SVC()))

	level0.append(('rf', RandomForestClassifier()))

	# define meta learner model

	level1 = LogisticRegression(max_iter=10000)

	# define the stacking ensemble

	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

	return model





clf=get_stacking()



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

evaluation.loc[r] = ['Stacked (Model-4)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from numpy import mean

from numpy import std

from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import StackingClassifier

from matplotlib import pyplot

 



X = class_5_df.loc[:, class_5_df.columns != 'TYPE']

y = class_5_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)





# get a stacking ensemble of models

def get_stacking():

	# define the base models

	level0 = list()

	level0.append(('knn', KNeighborsClassifier(n_neighbors=1)))

	level0.append(('dt', DecisionTreeClassifier()))

	level0.append(('svm', SVC()))

	level0.append(('rf', RandomForestClassifier()))

	# define meta learner model

	level1 = LogisticRegression(max_iter=10000)

	# define the stacking ensemble

	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

	return model





clf=get_stacking()



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

evaluation.loc[r] = ['Stacked (Model-5)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train,

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from numpy import mean

from numpy import std

from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import StackingClassifier

from matplotlib import pyplot



X = class_6_df.loc[:, class_6_df.columns != 'TYPE']

y = class_6_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)





# get a stacking ensemble of models

def get_stacking():

	# define the base models

	level0 = list()

	level0.append(('knn', KNeighborsClassifier(n_neighbors=1)))

	level0.append(('dt', DecisionTreeClassifier()))

	level0.append(('svm', SVC()))

	level0.append(('rf', RandomForestClassifier()))

	# define meta learner model

	level1 = LogisticRegression(max_iter=10000)

	# define the stacking ensemble

	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

	return model





clf=get_stacking()



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

evaluation.loc[r] = ['Stacked (Model-6)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
from numpy import mean

from numpy import std

from sklearn.datasets import make_classification

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import StackingClassifier

from matplotlib import pyplot



X = class_7_df.loc[:, class_7_df.columns != 'TYPE']

y = class_7_df['TYPE']

print(X.head())

print(y.head())





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)





# get a stacking ensemble of models

def get_stacking():

	# define the base models

	level0 = list()

	level0.append(('knn', KNeighborsClassifier(n_neighbors=1)))

	level0.append(('dt', DecisionTreeClassifier()))

	level0.append(('svm', SVC()))

	level0.append(('rf', RandomForestClassifier()))

	# define meta learner model

	level1 = LogisticRegression(max_iter=10000)

	# define the stacking ensemble

	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)

	return model





clf=get_stacking()



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

evaluation.loc[r] = ['Stacked (Model-7)',acc_train,precision_train,recall_train,f1_train,acc_test,precision_test,recall_test,f1_test]

evaluation.sort_values(by = 'Accuracy(test)', ascending=False)
p=y_train

q=y_test

y_train=y_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_train=clf.predict(X_train)

pred_train=pd.DataFrame(pred_train)



pred_train=pred_train.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



pred_test=clf.predict(X_test)

y_test=y_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])

pred_test=pd.DataFrame(pred_test)



pred_test=pred_test.replace([0.0,1.0,2.0,3.0,4.0,5.0,6.0], ["CES","ESP","EUT","FORC","VAGINAL","C-SEC","VACUM"])



skplt.metrics.plot_confusion_matrix(

    y_train, 

    pred_train,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )



skplt.metrics.plot_confusion_matrix(

    y_test, 

    pred_test,

    figsize=(20,10),

    title_fontsize='20',

    text_fontsize='20',

    )

y_train=p

y_test=q
evaluation.to_csv('result.csv')