#Importing the necessary libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import warnings

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

%matplotlib inline



warnings.filterwarnings("ignore")

pd.set_option('display.max_columns',500)
dataset = pd.read_csv("../input/ckdisease/kidney_disease.csv")

dataset.head
#check No of records and columns

dataset.shape
#Columns in the Dataset

dataset.columns
#Dataset Info

dataset.info()
dataset.select_dtypes(include = 'object').head()
dataset.isnull().any()
# unique value analysis

for col in dataset:

    print (f"{col}: has {dataset[col].nunique()} total unique value")

    print(f"Unique Values: {dataset[col].unique()}")

    print("\n")
#Check Value Count for Final Feature

dataset.classification.value_counts()
#Prerequiste to use RegEx with string, We need to replace the Null Value in String

dataset['pcv'].fillna('0', inplace=True)

dataset['wc'].fillna('0', inplace=True)

dataset['rc'].fillna('0.0', inplace=True)
#Check the bar graph of categorical Data using factorplot

sns.set_style("whitegrid")

sns.factorplot(data=dataset, x='rbc', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='pc', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='pcc', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='ba', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='pcv', kind= 'count',size=6,aspect=2)

sns.factorplot(data=dataset, x='wc', kind= 'count',size=10,aspect=2)

sns.factorplot(data=dataset, x='rc', kind= 'count',size=6,aspect=2)

sns.factorplot(data=dataset, x='htn', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='dm', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='cad', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='appet', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='pe', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='ane', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='classification', kind= 'count',size=4,aspect=2)
#Created Function to convert the Data into Correct Format
import re

def get_number(val):

    if val.strip():

        val = val.strip()

        txt= re.findall("\d+", val)

        if len(txt) > 0:

            return txt[0]

        else:

            return '0'
import re

def get_dec_number(val):

    if val.strip():

        val = val.strip()

        txt= re.findall("\d+\.\d+", val)

        if len(txt) > 0:

            return txt[0]

        else:

            return '0'
#Loop on the Dataset and convert the Non-Numeric Field to Numeric

for ind in dataset.index: 

    if dataset['rbc'] [ind] == 'normal':

        dataset['rbc'] [ind] = '1'

    elif dataset['rbc'] [ind] == 'abnormal':

        dataset['rbc'] [ind] = '0'

        

    if dataset['pc'] [ind] == 'normal':

        dataset['pc'] [ind] = '1'

    elif dataset['pc'] [ind] == 'abnormal':

        dataset['pc'] [ind] = '0' 

        

    if dataset['pcc'] [ind] == 'present':

        dataset['pcc'] [ind] = '1'

    elif dataset['pcc'] [ind] == 'notpresent':

        dataset['pcc'] [ind] = '0'  



    if dataset['ba'] [ind] == 'present':

        dataset['ba'] [ind] = '1'

    elif dataset['ba'] [ind] == 'notpresent':

        dataset['ba'] [ind] = '0'  

    

    if dataset['htn'] [ind] == 'yes':

        dataset['htn'] [ind] = '1'

    elif dataset['htn'] [ind] == 'no':

        dataset['htn'] [ind] = '0'

    

    if dataset['appet'] [ind] == 'good':

        dataset['appet'] [ind] = '1'

    elif dataset['appet'] [ind] == 'poor':

        dataset['appet'] [ind] = '0'

        

    if dataset['pe'] [ind] == 'yes':

        dataset['pe'] [ind] = '1'

    elif dataset['pe'] [ind] == 'no':

        dataset['pe'] [ind] = '0'   



    if dataset['ane'] [ind] == 'yes':

        dataset['ane'] [ind] = '1'

    elif dataset['ane'] [ind] == 'no':

        dataset['ane'] [ind] = '0'  

        

    if dataset['dm'] [ind] == 'yes' or dataset['dm'] [ind] == ' yes':

        dataset['dm'] [ind] = '1'

    elif dataset['dm'] [ind] == 'no':

        dataset['dm'] [ind] = '0'  

    elif dataset['dm'] [ind] == '\tyes':

        dataset['dm'] [ind] = '1'

    elif dataset['dm'] [ind] == '\tno':

        dataset['dm'] [ind] = '0' 

        

    if dataset['cad'] [ind] == 'yes':

        dataset['cad'] [ind] = '1'

    elif dataset['cad'] [ind] == 'no':

        dataset['cad'] [ind] = '0'  

    elif dataset['cad'] [ind] == '\tno':

        dataset['cad'] [ind] = '0'   

        

    if dataset['classification'] [ind] == 'ckd':

        dataset['classification'] [ind] = '1'

    elif dataset['classification'] [ind] == 'notckd':

        dataset['classification'] [ind] = '0'  

    elif dataset['classification'] [ind] == 'ckd\t':

        dataset['classification'] [ind] = '1'

    

    dataset['pcv'] [ind] = get_number(dataset['pcv'] [ind])

    dataset['wc'] [ind] = get_number(dataset['wc'] [ind])

    dataset['rc'] [ind] = get_dec_number(dataset['rc'] [ind])



#Check the bar graph of categorical Data using factorplot

sns.set_style("whitegrid")

sns.factorplot(data=dataset, x='rbc', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='pc', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='pcc', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='ba', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='pcv', kind= 'count',size=6,aspect=2)

sns.factorplot(data=dataset, x='wc', kind= 'count',size=10,aspect=2)

sns.factorplot(data=dataset, x='rc', kind= 'count',size=6,aspect=2)

sns.factorplot(data=dataset, x='htn', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='dm', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='cad', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='appet', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='pe', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='ane', kind= 'count',size=4,aspect=2)

sns.factorplot(data=dataset, x='classification', kind= 'count',size=4,aspect=2)
100*(dataset.isnull().sum()/dataset.shape[0])
#Change the Datatype to Int and Float from Object

dataset[['pcv','wc','classification']].astype(dtype = 'int64')

dataset[['rbc','pc','rc','pcc']].astype(dtype = 'float64')
#Check before Imputing Null Values

dataset.isnull().any()
#Imputing Null Values in Numeric Fields using Mean and in Categorical using Mode

dataset['age'].fillna(dataset['age'].mean(), inplace=True)

dataset['bp'].fillna(dataset['bp'].mean(), inplace=True)

dataset['sg'].fillna(dataset['sg'].mean(), inplace=True)

dataset['al'].fillna(dataset['al'].mean(), inplace=True)

dataset['su'].fillna(dataset['su'].mean(), inplace=True)

dataset['bgr'].fillna(dataset['bgr'].mean(), inplace=True)

dataset['bu'].fillna(dataset['bu'].mean(), inplace=True)

dataset['sc'].fillna(dataset['sc'].mean(), inplace=True)

dataset['sod'].fillna(dataset['sod'].mean(), inplace=True)

dataset['pot'].fillna(dataset['pot'].mean(), inplace=True)

dataset['hemo'].fillna(dataset['hemo'].mean(), inplace=True)

dataset['rbc'].fillna('1', inplace=True)

dataset['pc'].fillna('1', inplace=True)

dataset['pcc'].fillna('0', inplace=True)

dataset['ba'].fillna('0', inplace=True)

dataset['htn'].fillna('0', inplace=True)

dataset['dm'].fillna('0', inplace=True)

dataset['cad'].fillna('0', inplace=True)

dataset['appet'].fillna('1', inplace=True)

dataset['pe'].fillna('0', inplace=True)

dataset['ane'].fillna('0', inplace=True)
#Check After Imputing Null Values

dataset.isnull().any()
#Transform non-numeric columns into numerical columns

#Pre-requiste no null Values should be there in column

for column in dataset.columns:

        if dataset[column].dtype == np.number:

            continue

        print(column)

        dataset[column] = LabelEncoder().fit_transform(dataset[column])
# descriptive statistics

dataset.describe()
#Identify Columns with Missing Values

dataset.isna().sum(axis=0)
def draw_histograms(dataframe, features, rows, cols):

    fig=plt.figure(figsize=(20,20))

    for i, feature in enumerate(features):

        ax=fig.add_subplot(rows,cols,i+1)

        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')

        ax.set_title(feature+" Distribution",color='DarkRed')

        

    fig.tight_layout()  

    plt.show()

draw_histograms(dataset,dataset.columns,10,10)
dataset.corr()
f,ax=plt.subplots(figsize=(30,25))

sns.heatmap(dataset.corr(),annot=True,fmt=".2f",ax=ax,linewidths=0.5,linecolor="orange")

plt.xticks(rotation=45)

plt.yticks(rotation=45)

plt.show()
#Drop Id and RC Column

#Id is not related to Classification

#PCV and RC are highly corelated

#Also Drop High Negatively Corelated Features with Classification i.e sg,rbc,pc,sod,hemo,pcv,wc,rc,appet

dataset.drop(['id','sg','rbc','pc','sod','hemo','pcv','wc','rc','appet'],axis = 1,inplace=True)
#New Shape of Dataset

dataset.shape
#Create Box Plot to Identify Outliers

dataset.boxplot(return_type='dict')

plt.plot()
#Plot Separate BoxPlot for Features where max Outliers Identified

dataset.boxplot(column=['bgr'],return_type='dict')

plt.plot()
#Replace Outlier Values with Mean and again Plot the BoxPlot

dataset['bgr'] = dataset['bgr'].apply(lambda x: dataset['bgr'].mean() if x>220 else x)

dataset.boxplot(column=['bgr'],return_type='dict')

plt.plot()
#RBC against Classification

plt.figure(figsize = (30,15))

sns.barplot(x='age', y ='classification', data=dataset,estimator=sum,orient="v")

plt.title('Age against Classification')

plt.xlabel("Age")

plt.ylabel("Classification")
#Check the distribution of Age and BP against Classification using scatter plot

fig, axs = plt.subplots(1,2, figsize=(15, 5), sharey=True)

axs[0].scatter(data=dataset, x='al', y='classification')

plt.xlabel("AL")

plt.ylabel("Classification")

axs[1].scatter(data=dataset, x='bp', y='classification', color = 'red')

fig.suptitle('Scatter plot for Al and BP against Classification')

plt.xlabel("Bp")

plt.ylabel("Classification")
#From Above Scatter Plot we can see that for Al amore than 1 clearly end with CKD

#Similarly for BP more than 80 also results in CKD
#Determining the predictors/features(X) and response/Class Label (Y)

X = dataset.iloc[:,0:14].values

X
Y = dataset.iloc[:,[15]].values

Y
#Min Max Scaling

scaler = MinMaxScaler(feature_range=(0, 1))

X = scaler.fit_transform(X)
from sklearn.linear_model import LogisticRegression

#Making Confusion Matrix

from sklearn.metrics import confusion_matrix,classification_report



def get_score(model, X_train, X_test, Y_train, Y_test):

    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    cm = confusion_matrix(Y_test,Y_pred)

    print("Confusion Matrix /n", cm)

    print(classification_report(Y_test,Y_pred))

    return model.score(X_test, Y_test)
scores_l = []

#Perform 10 fold cross validation

from sklearn.model_selection import StratifiedKFold 

kf = StratifiedKFold(n_splits=10, random_state=None) 



for train_index, test_index in kf.split(X,Y):

      print("\n\nTrain:", train_index, "\nValidation:",test_index)

      X_train, X_test = X[train_index], X[test_index] 

      Y_train, Y_test = Y[train_index], Y[test_index]

      scores_l.append(get_score(LogisticRegression(), X_train, X_test, Y_train, Y_test))

      print(get_score(LogisticRegression(), X_train, X_test, Y_train, Y_test))                
scores_l
# Visualising the Test set results,

from sklearn.decomposition import PCA 

from matplotlib.colors import ListedColormap

pca = PCA(n_components = 2) 

  

X_train = pca.fit_transform(X_train) 

X_test = pca.transform(X_test) 

  

explained_variance = pca.explained_variance_ratio_ 



X_set, y_set = X_test, Y_test,

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

model = LogisticRegression() 

model.fit(X_train, Y_train)

plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('red', 'green'))),

plt.xlim(X1.min(), X1.max()),

plt.ylim(X2.min(), X2.max()),

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('red', 'green'))(i), label = j)  

plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)

plt.title('Logistic Regression (Test set)')

plt.xlabel('Features PC1')

plt.ylabel('Predicted Values PC2')

plt.legend()

plt.show()