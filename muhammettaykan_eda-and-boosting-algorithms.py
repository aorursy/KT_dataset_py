import numpy as np

import pandas as pd



#Visualization

import matplotlib.pyplot as plt

import seaborn as sns



#preprocessing & metrics

from sklearn import preprocessing

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from scipy import stats



#Model

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier
df=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

df.head()
df.columns
df.rename(columns=lambda x : str(x).replace(" ","_"), inplace=True)

df.shape
df.info()
df.describe()
fig, axs = plt.subplots(figsize=[8, 6])

sns.set(style="whitegrid")



sns.countplot(x="class",

              data=df,

              order = df['class'].value_counts().index,

              )

plt.suptitle('Class Count',

          color='black',

          fontsize=14,

          fontweight='bold')

plt.show()
labels=df['class'].value_counts().index

values=df['class'].value_counts().values

explode = (0.03, 0)

#visualization

plt.figure(figsize=(7,7))

plt.pie(values,

        explode=explode,

        labels=labels,

        autopct='%1.1f%%')

plt.suptitle('Class Rate',

          color='black',

          fontsize=14,

          fontweight='bold')

plt.show()
fig, axs = plt.subplots(figsize=[14, 8], ncols=1)



ax = sns.boxplot(data=df, 

                 orient="v", 

                 palette="Set2")

plt.suptitle('Box Plot of Each Feature',

          color='black',

          fontsize=14,

          fontweight='bold')

plt.show()
columns=df.columns

k=1



plt.figure(figsize=(20, 15))

for i in columns[:-1]:

    plt.subplot(3, 3, k)

    sns.boxplot(x="class", 

                y=i, 

                hue="class", 

                data=df)

    k+=1

plt.suptitle('Boxplot of Each Feature by Class',

          color='black',

          fontsize=14,

          fontweight='bold')

plt.show()
k=1

plt.figure(figsize=(20, 15))

for i in columns[:-1]:

    plt.subplot(3, 3, k)

    sns.distplot((df[i]), 

                 bins=30, 

                 norm_hist=True)

    k+=1



plt.show()
fig, (ax) = plt.subplots(1, 1, figsize=(10,6))

matrix = np.triu(df.corr())

hm = sns.heatmap(df.corr(), 

                 ax=ax,           

                 cmap="coolwarm", 

                 linewidths=1,

                 annot=True,

                 center=0,

                 mask=matrix

          

                )

fig.subplots_adjust(top=0.93)

fig.suptitle('Data Correlation Heatmap', 

              fontsize=14, 

              fontweight='bold')

plt.show()
# Label Encoder for Target Variable

label_encoder = preprocessing.LabelEncoder()

df['class']=label_encoder.fit_transform(df['class'].values)
#Normalization

df.iloc[:,:-1] = preprocessing.StandardScaler().fit_transform(df.iloc[:,:-1])



#Train-Test Split

X = df.drop(['class'],axis=1)

y = df['class']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#AdaBoost Classifier

ada = AdaBoostClassifier()

ada.fit(X_train, y_train)

y_pred = ada.predict(X_test)



test_score = round(accuracy_score(y_test,y_pred) * 100, 2)

print ('The accuracy for the AdaBoost model is',test_score,"%")



conf_matrix= pd.crosstab(y_pred, y_test)

print('\n Confision matrix')

print(conf_matrix)



clf_report=classification_report(y_test , y_pred , target_names=['Abnormal','Normal'])

print('\n Classification report')

print(clf_report)
#Gradient Boosting Classifier

gdb = GradientBoostingClassifier()

gdb.fit(X_train, y_train)

y_pred = gdb.predict(X_test)



test_score = round(accuracy_score(y_test,y_pred) * 100, 2)

print ('The accuracy for the AdaBoost model is',test_score,"%")



conf_matrix= pd.crosstab(y_pred, y_test)

print('\n Confision matrix')

print(conf_matrix)



clf_report=classification_report(y_test , y_pred , target_names=['Abnormal','Normal'])

print('\n Classification report')

print(clf_report)
#XGBoost Classifier

xgb = XGBClassifier()

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)



test_score = round(accuracy_score(y_test,y_pred) * 100, 2)

print ('The accuracy for the XGBoost model is',test_score,"%")



conf_matrix= pd.crosstab(y_pred, y_test)

print('\n Confision matrix')

print(conf_matrix)



clf_report=classification_report(y_test , y_pred , target_names=['Abnormal','Normal'])

print('\n Classification report')

print(clf_report)
#Cross Validation for Boosting Classifiers

boost_array = [ada, gdb, xgb]

labels = ['Ada Boosting', 'Gradient Boosting', 'XG Boosting']

results=[[],[]]



for clf in boost_array:

    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

    results[0].append(round(scores.mean() * 100, 2))

    results[1].append(round(scores.std(), 3))
# CV results by mean and standard deviation

results=pd.DataFrame(np.array(results).T,columns=['Accuracy Score','Standart Deviations(+/-)'])

results.insert(loc = 0,

              column = 'Classifiers',

              value = labels)

results
k=1



plt.figure(figsize=(12, 15))

for i in boost_array:

    plt.subplot(3, 1, k)

    feat_importances = pd.Series(i.feature_importances_, index=X.columns)

    sns.barplot(x=feat_importances.values, 

                y=feat_importances.index, 

                )

    

    plt.title(labels[k-1] + ' Feature Importance')

    plt.ylabel('features')

    k+=1



plt.show()
#for unique value

def unique(list_): 

    outlier = np.array(list_) 

    outlier=np.unique(outlier)

    return outlier
outlier_index=[]

for i in columns:

    if i!='class':

        Q1 = df[i].quantile(0.25)

        Q3 = df[i].quantile(0.75)

        IQR = Q3 - Q1

        K=(df[(df[i]<(Q1-1.5*IQR)) | (df[i]>(Q3+(1.5*IQR)))].index)

        for i in K:

            outlier_index.append(i)



outlier_iqr =unique(outlier_index)
outlier_index=[]

for i in columns:

    if i!='class':

        K=df[stats.zscore(df[i])>=3].index

        for i in K:

            outlier_index.append(i)



outlier_zscore =unique(outlier_index)
print('Outlier count for iqr test: ',len(outlier_iqr))

print('Outlier count for z score: ',len(outlier_zscore))
def outlier_accuracy(data,outlier):

    df_outlier=data

    df_outlier=df_outlier.drop(df_outlier.index[outlier])

    #Train test

    X = df_outlier.drop(['class'],axis=1)

    y = df_outlier['class']

    

    results=[[],[]]



    for clf in boost_array:

        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

        results[0].append(round(scores.mean() * 100, 2))

        results[1].append(round(scores.std(), 3))

    

    results=pd.DataFrame(np.array(results).T,columns=['Accuracy Score','Standart Deviations(+/-)'])

    results.insert(loc = 0,

                   column = 'Classifiers',

                   value = labels)

    

    return results
outlier_accuracy(df,outlier_iqr)
outlier_accuracy(df,outlier_zscore)