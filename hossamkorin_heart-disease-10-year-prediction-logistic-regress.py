import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import plotly.express as px
import matplotlib.pyplot as plt
%matplotlib inline
df=pd.read_csv('../input/heart-disease-prediction-using-logistic-regression/framingham.csv')
df

#checking if we have any missing values
df.info()
#we have 7 columns that contain missing values
#let's do a quick pandas profiling report to verify if there are any missing data and help understand what values we
#should use to fill in any of these missing values
from pandas_profiling import ProfileReport
rpt=ProfileReport(df)
rpt
#replacing the missing education values witht the most frequent value
df['education'].replace(np.nan, 1.0, inplace= True)
#replacing all other values with either the mean or mode of the coulumn values
df['BPMeds'].replace(np.nan, 0.0, inplace= True)
df['totChol'].replace(np.nan, df['totChol'].mean(), inplace= True)
df['glucose'].replace(np.nan, df['glucose'].mean(), inplace= True)
df['cigsPerDay'].replace(np.nan, df['cigsPerDay'].mean(), inplace= True)
df['BMI'].replace(np.nan, df['BMI'].mean(), inplace= True)
df['heartRate'].replace(np.nan, df['heartRate'].mean(), inplace= True)
df.info()
#the data now contains no missing values
#defining the predictor and target variables
x=df.loc[:, df.columns != 'TenYearCHD']
y=df['TenYearCHD']
print(x,y)
#let's take a look at the correlation between the different features
import seaborn as sns
cor=x.corr()
sns.heatmap(cor, annot=True, cmap='coolwarm')
#we can tell that there is strong relation between some of our features (cigsPerDay & currentSmoker) and (PrevalentHyp&sysBHP)

#let's try to see if we can reduce the data dimensionality and reduce feature dependencies using PCA
#standardizaing the features with continuous variables
scl=StandardScaler()
x[['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]=scl.fit_transform(x[['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']])
x
#let's split the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=1)

#now let's try to perform some PCA to keep only the features that contribute most to our data
pca=PCA(0.95)   #0.95 is the amount of data variance/information that we would like to retain after doing PCA

pca.fit(x_train)
pca.n_components_ #we've reduced the number of columns no down to 9 from 15
#transforming our train and test features
x_train= pca.transform(x_train)
x_test=pca.transform(x_test)

x_train.shape

#now we're going to try and use Logistic Regression
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression(C=0.001, solver='liblinear')
LR.fit(x_train, y_train)
LR.score(x_test, y_test) #good model accuracy with almost 87%
y_hat=LR.predict(x_test)
y_hat_prob=LR.predict_proba(x_test)
y_hat_prob
#let's also take a look at the confusion matrix to understand the accuracy
from sklearn.metrics import classification_report, confusion_matrix
import itertools
#this is to define the actual function to plot a confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

print(confusion_matrix(y_test, y_hat, labels=[1,0]))
cnf_mtx=confusion_matrix(y_test, y_hat, labels=[1,0])
np.set_printoptions(precision=2)

#plotting the actual matrix
plt.figure()
plot_confusion_matrix(cnf_mtx, classes=['TenYearCHD=1','TenYearCHD=0'],normalize= False,  title='Confusion matrix')
print(classification_report(y_test, y_hat))