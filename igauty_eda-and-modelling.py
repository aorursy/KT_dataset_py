# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

import datetime

import time

import warnings

warnings.filterwarnings('ignore')





from scipy.stats import norm

from scipy import stats

from sklearn import preprocessing



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

sns.set_style('whitegrid')

%matplotlib inline



#VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor

from patsy import dmatrices



#Modelling

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier

from lightgbm import LGBMClassifier

from sklearn.neural_network import MLPClassifier

import xgboost



from sklearn.tree import export_graphviz #plot tree

from subprocess import call



#Neural Nets

from keras.models import Sequential

from keras.layers import Dense

from keras.preprocessing import text, sequence

from keras import layers, models, optimizers

# Evaluation metrics

from sklearn import metrics 
#data = pd.read_csv('/media/vishwadeepg/New Volume/Work/0. Gauty/Kernal/heart_disease/heart.csv')

data = pd.read_csv('../input/heart.csv')
data.shape
data.info()
data.describe().T
data.dtypes.value_counts()
data.columns
data.head()
def missing_check(df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) 

    #print("Missing check:",missing_data )

    return missing_data

missing_check(data)
print(data['target'].value_counts())

ax = sns.countplot(x=data['target'], data=data)
data['age_bin'] = pd.cut(data.age,[29,30,35,40,45,50,55,60],labels=[30,35,40,45,50,55,60])

print(pd.DataFrame(data['age_bin'].value_counts()))

ax = sns.countplot(x=data['age_bin'], data=data)
print(pd.DataFrame(data['sex'].value_counts()))

ax = sns.countplot(x=data['sex'], data=data)
print(pd.DataFrame(data['cp'].value_counts()))

ax = sns.countplot(x=data['cp'], data=data)
data['chol_bin'] = pd.cut(data.chol,[125,150,200,250,300,350,400,450,500,550,600],

                              labels=[150,200,250,300,350,400,450,500,550,600])

print(pd.DataFrame(data['chol_bin'].value_counts()))

ax = sns.countplot(x=data['chol_bin'], data=data)
data['trestbps_bin'] = pd.cut(data.trestbps,[93,110,120,130,140,150,160,205],labels=[110,120,130,140,150,160,205])

print(pd.DataFrame(data['trestbps_bin'].value_counts()))

ax = sns.countplot(x=data['trestbps_bin'], data=data)
print(pd.DataFrame(data['fbs'].value_counts()))

ax = sns.countplot(x=data['fbs'], data=data)
print(pd.DataFrame(data['restecg'].value_counts()))

ax = sns.countplot(x=data['restecg'], data=data)
data['thalach_bin'] = pd.cut(data.thalach,[70,90,110,130,150,170,180,200,203],labels=[90,110,130,150,170,180,200,203])

print(pd.DataFrame(data['thalach_bin'].value_counts()))

ax = sns.countplot(x=data['thalach_bin'], data=data)
print(pd.DataFrame(data['exang'].value_counts()))

ax = sns.countplot(x=data['exang'], data=data)
data['oldpeak_bin']=pd.cut(data.oldpeak,[-0.1,0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,6.5],

                                    labels=[0.0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,6.5])

print(pd.DataFrame(data['oldpeak_bin'].value_counts()))

ax = sns.countplot(x=data['oldpeak_bin'], data=data)
print(pd.DataFrame(data['slope'].value_counts()))

ax = sns.countplot(x=data['slope'], data=data)
print(pd.DataFrame(data['ca'].value_counts()))

ax = sns.countplot(x=data['ca'], data=data)
print(pd.DataFrame(data['thal'].value_counts()))

ax = sns.countplot(x=data['thal'], data=data)
target_1 = data[data['target']==1]['age_bin'].value_counts()

target_0 = data[data['target']==0]['age_bin'].value_counts()

df = pd.DataFrame([target_1, target_0])

df.index = ['target_1','target_0']

print(df)

print('------------------------------------------------------------------------------------------------------------------------')

df.plot(kind='bar',stacked=True, figsize=(10,6))
target_1 = data[data['target']==1]['sex'].value_counts()

target_0 = data[data['target']==0]['sex'].value_counts()

df = pd.DataFrame([target_1, target_0])

df.index = ['target_1','target_0']

print(df)

print('------------------------------------------------------------------------------------------------------------------------')

df.plot(kind='bar',stacked=True, figsize=(10,6))
target_1 = data[data['target']==1]['cp'].value_counts()

target_0 = data[data['target']==0]['cp'].value_counts()

df = pd.DataFrame([target_1, target_0])

df.index = ['target_1','target_0']

print(df)

print('------------------------------------------------------------------------------------------------------------------------')

df.plot(kind='bar',stacked=True, figsize=(10,6))
target_1 = data[data['target']==1]['trestbps_bin'].value_counts()

target_0 = data[data['target']==0]['trestbps_bin'].value_counts()

df = pd.DataFrame([target_1, target_0])

df.index = ['target_1','target_0']

print(df)

print('------------------------------------------------------------------------------------------------------------------------')

df.plot(kind='bar',stacked=True, figsize=(10,6))
target_1 = data[data['target']==1]['chol_bin'].value_counts()

target_0 = data[data['target']==0]['chol_bin'].value_counts()

df = pd.DataFrame([target_1, target_0])

df.index = ['target_1','target_0']

print(df)

print('------------------------------------------------------------------------------------------------------------------------')

df.plot(kind='bar',stacked=True, figsize=(10,6))
target_1 = data[data['target']==1]['fbs'].value_counts()

target_0 = data[data['target']==0]['fbs'].value_counts()

df = pd.DataFrame([target_1, target_0])

df.index = ['target_1','target_0']

print(df)

print('------------------------------------------------------------------------------------------------------------------------')

df.plot(kind='bar',stacked=True, figsize=(10,6))
target_1 = data[data['target']==1]['restecg'].value_counts()

target_0 = data[data['target']==0]['restecg'].value_counts()

df = pd.DataFrame([target_1, target_0])

df.index = ['target_1','target_0']

print(df)

print('------------------------------------------------------------------------------------------------------------------------')

df.plot(kind='bar',stacked=True, figsize=(10,6))
target_1 = data[data['target']==1]['thalach_bin'].value_counts()

target_0 = data[data['target']==0]['thalach_bin'].value_counts()

df = pd.DataFrame([target_1, target_0])

df.index = ['target_1','target_0']

print(df)

print('------------------------------------------------------------------------------------------------------------------------')

df.plot(kind='bar',stacked=True, figsize=(10,6))
target_1 = data[data['target']==1]['exang'].value_counts()

target_0 = data[data['target']==0]['exang'].value_counts()

df = pd.DataFrame([target_1, target_0])

df.index = ['target_1','target_0']

print(df)

print('------------------------------------------------------------------------------------------------------------------------')

df.plot(kind='bar',stacked=True, figsize=(10,6))
target_1 = data[data['target']==1]['oldpeak_bin'].value_counts()

target_0 = data[data['target']==0]['oldpeak_bin'].value_counts()

df = pd.DataFrame([target_1, target_0])

df.index = ['target_1','target_0']

print(df)

print('------------------------------------------------------------------------------------------------------------------------')

df.plot(kind='bar',stacked=True, figsize=(10,6))
target_1 = data[data['target']==1]['slope'].value_counts()

target_0 = data[data['target']==0]['slope'].value_counts()

df = pd.DataFrame([target_1, target_0])

df.index = ['target_1','target_0']

print(df)

print('------------------------------------------------------------------------------------------------------------------------')

df.plot(kind='bar',stacked=True, figsize=(10,6))
target_1 = data[data['target']==1]['ca'].value_counts()

target_0 = data[data['target']==0]['ca'].value_counts()

df = pd.DataFrame([target_1, target_0])

df.index = ['target_1','target_0']

print(df)

print('------------------------------------------------------------------------------------------------------------------------')

df.plot(kind='bar',stacked=True, figsize=(10,6))
target_1 = data[data['target']==1]['thal'].value_counts()

target_0 = data[data['target']==0]['thal'].value_counts()

df = pd.DataFrame([target_1, target_0])

df.index = ['target_1','target_0']

print(df)

print('------------------------------------------------------------------------------------------------------------------------')

df.plot(kind='bar',stacked=True, figsize=(10,6))
data.plot(kind="scatter", x="age", y="chol", alpha= 0.5, color="g", figsize=(12,8))
data.plot(kind="scatter", x="age", y="trestbps", alpha= 0.5, color="r", figsize=(12,8))
data.plot(kind="scatter", x="age", y="thalach", alpha= 0.5, color="b", figsize=(12,8))
data.plot(kind="scatter", x="age", y="oldpeak", alpha= 0.5, color="m", figsize=(12,8))
data.groupby(['age_bin', 'chol_bin'])['target'].value_counts()
data.groupby(['age_bin', 'sex'])['target'].value_counts()
data.groupby(['age_bin', 'trestbps_bin'])['target'].value_counts()
data.groupby(['age_bin', 'fbs'])['target'].value_counts()
data.groupby(['age_bin', 'restecg'])['target'].value_counts()
data.groupby(['age_bin', 'thalach_bin'])['target'].value_counts()
data.groupby(['age_bin', 'exang'])['target'].value_counts()
data.groupby(['age_bin', 'slope'])['target'].value_counts()
data.groupby(['age_bin', 'ca'])['target'].value_counts()
data.groupby(['age_bin', 'thal'])['target'].value_counts()
pd.DataFrame(data.groupby(['sex', 'fbs', 'exang', 'slope'])['target'].value_counts())
data.chol.plot(kind="line",color="green",label="chol",grid=True,linestyle=":", figsize= (20,10))

data.thalach.plot(kind="line",color="purple",label="thalach",grid=True, figsize= (20,10))

data.age.plot(kind="line",color="pink",label="age",grid=True, figsize= (20,10))

data.trestbps.plot(kind="line",color="orange",label="trestbps",grid=True, figsize= (20,10))

plt.legend(loc="upper right") #legend: puts feature label into plot

plt.xlabel("indexes")

plt.ylabel("Features")

plt.title("Heart Diseases")

plt.show()
sns.pairplot(data.loc[:,["chol","age","ca","oldpeak"]])

plt.show()
data_corr = data.corr()['target'][:-1] # -1 because the latest row is Target

golden_features_list = data_corr[abs(data_corr) > 0.1].sort_values(ascending=False)

print("There is {} strongly correlated values with Target:\n{}".format(len(golden_features_list), golden_features_list))
corr = data.corr() 

plt.figure(figsize=(12, 10))



sns.heatmap(corr[(corr >= 0.1) | (corr <= -0.1)], 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);
def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   

#Outliers_to_drop = detect_outliers(train,2,[col])
def printContingencyTable(y_cv, Y_pred):

    confusion_matrix = metrics.confusion_matrix(y_cv, Y_pred)

    plt.matshow(confusion_matrix)

    plt.title('Confusion matrix')

    plt.colorbar()

    plt.ylabel('Churned')

    plt.xlabel('Predicted')

    plt.show()

    print("precision_score : ", metrics.precision_score(y_cv, Y_pred))

    print("recall_score : ", metrics.recall_score(y_cv, Y_pred))

    print("f1_score : ", metrics.f1_score(y_cv, Y_pred))

    print(confusion_matrix)
data = data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',

       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']]
Y = data['target']

X = data.drop(['target'], axis=1)



train_x, X_cv, train_y, y_cv = train_test_split(X, Y, test_size=0.30, random_state=42)
logreg = LogisticRegression()

logreg.fit(train_x, train_y)

Y_pred = logreg.predict(X_cv)



printContingencyTable(y_cv, Y_pred)
DT = DecisionTreeClassifier()

DT.fit(train_x, train_y)

Y_pred = DT.predict(X_cv)





printContingencyTable(y_cv, Y_pred)
RF = RandomForestClassifier()

RF.fit(train_x, train_y)

Y_pred = RF.predict(X_cv)



printContingencyTable(y_cv, Y_pred)


estimator = RF.estimators_[1]

feature_names = [i for i in train_x.columns]



y_train_str = train_y.astype('str')

y_train_str[y_train_str == '0'] = 'no disease'

y_train_str[y_train_str == '1'] = 'disease'

y_train_str = y_train_str.values



export_graphviz(estimator, out_file='tree.dot', 

                feature_names = feature_names,

                class_names = y_train_str,

                rounded = True, proportion = True, 

                label='root',

                precision = 2, filled = True)





call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



from IPython.display import Image

Image(filename = 'tree.png')

GB = GradientBoostingClassifier()

GB.fit(train_x, train_y)

Y_pred = GB.predict(X_cv)



printContingencyTable(y_cv, Y_pred)
et = ExtraTreesClassifier()

et.fit(train_x, train_y)

Y_pred = et.predict(X_cv)



printContingencyTable(y_cv, Y_pred)
adb = AdaBoostClassifier()

adb.fit(train_x, train_y)

Y_pred = adb.predict(X_cv)



printContingencyTable(y_cv, Y_pred)
import lightgbm as lgbm

params = {

    'objective' :'binary',

    'learning_rate' : 0.02,

    'num_leaves' : 76,

    'feature_fraction': 0.64, 

    'bagging_fraction': 0.8, 

    'bagging_freq':1,

    'boosting_type' : 'gbdt',

    'metric': 'binary_logloss'

}

d_train = lgbm.Dataset(train_x, train_y)

d_valid = lgbm.Dataset(X_cv, y_cv)

bst = lgbm.train(params, d_train, 5000, valid_sets=[d_valid], verbose_eval=50, early_stopping_rounds=100)
prediction = bst.predict(X_cv)

#convert into binary values

for i in range(0,len(prediction)):

    if prediction[i]>=.5:

        prediction[i]=1

    else:  

        prediction[i]=0



printContingencyTable(y_cv,prediction)
xgb = xgboost.XGBClassifier()

xgb.fit(train_x, train_y)

Y_pred = xgb.predict(X_cv)



printContingencyTable(y_cv, Y_pred)