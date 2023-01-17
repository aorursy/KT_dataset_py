#importing libraries

                                             

import pandas as pd                                    # for dataframe

import numpy as np                                     # for numerical operations

from fancyimpute import KNN                            # for knn imputations

from scipy.stats import chi2_contingency               # for scientific calculations

import matplotlib.pyplot as plt                        # for visualisations

import seaborn as sns                                  # for visualisatons

from random import randrange,uniform                   # to generate random number

from sklearn.model_selection import train_test_split   # for implementing stratified sampling

from sklearn import tree                               # for implementing decision tree algorithm in data

from sklearn.tree import export_graphviz               #  plot tree

from sklearn.metrics import accuracy_score             # for implementing decision tree algorithm in data

from sklearn.metrics import confusion_matrix           # for calculating error metrics of various models

from sklearn.ensemble import RandomForestClassifier    # for implementing random forest model on data

import statsmodels.api as sn                           # for applying logistic model on data set

from sklearn.neighbors import KNeighborsClassifier     # for implementing knn model

from sklearn.naive_bayes import GaussianNB             # for implementing naive bayes

from sklearn import model_selection                    # for selecting model

from sklearn.metrics import classification_report,roc_auc_score,roc_curve # for model evaluation

from sklearn.metrics import classification_report      # for model evaluation

import pickle                                          # for saving the final modelimport seaborn as sns #for plotting

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor  # for calculating VIF

from statsmodels.tools.tools import add_constant

np.random.seed(123) #ensure reproducibility

pd.options.mode.chained_assignment = None  #hide any pandas warnings
hdata = pd.read_csv("../input/heart.csv")
hdata.head(10)
hdata.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',

       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

hdata['sex'][hdata['sex'] == 0] = 'female'

hdata['sex'][hdata['sex'] == 1] = 'male'



hdata['chest_pain_type'][hdata['chest_pain_type'] == 1] = 'typical angina'

hdata['chest_pain_type'][hdata['chest_pain_type'] == 2] = 'atypical angina'

hdata['chest_pain_type'][hdata['chest_pain_type'] == 3] = 'non-anginal pain'

hdata['chest_pain_type'][hdata['chest_pain_type'] == 4] = 'asymptomatic'



hdata['fasting_blood_sugar'][hdata['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'

hdata['fasting_blood_sugar'][hdata['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'



hdata['rest_ecg'][hdata['rest_ecg'] == 0] = 'normal'

hdata['rest_ecg'][hdata['rest_ecg'] == 1] = 'ST-T wave abnormality'

hdata['rest_ecg'][hdata['rest_ecg'] == 2] = 'left ventricular hypertrophy'



hdata['exercise_induced_angina'][hdata['exercise_induced_angina'] == 0] = 'no'

hdata['exercise_induced_angina'][hdata['exercise_induced_angina'] == 1] = 'yes'



hdata['st_slope'][hdata['st_slope'] == 1] = 'upsloping'

hdata['st_slope'][hdata['st_slope'] == 2] = 'flat'

hdata['st_slope'][hdata['st_slope'] == 3] = 'downsloping'



hdata['thalassemia'][hdata['thalassemia'] == 1] = 'normal'

hdata['thalassemia'][hdata['thalassemia'] == 2] = 'fixed defect'

hdata['thalassemia'][hdata['thalassemia'] == 3] = 'reversable defect'



hdata['target'][hdata['target'] == 0] = 'no'

hdata['target'][hdata['target'] == 1] = 'yes'



#Encoding Variable

#Assigning levels to the categories

lis = []

for i in range(0, hdata.shape[1]):

    if(hdata.iloc[:,i].dtypes == 'object'):

        hdata.iloc[:,i] = pd.Categorical(hdata.iloc[:,i])

        hdata.iloc[:,i] = hdata.iloc[:,i].cat.codes 

        hdata.iloc[:,i] = hdata.iloc[:,i].astype('object')

        lis.append(hdata.columns[i])

sns.countplot(x="target", data=hdata, palette="bwr")

plt.show()
countNoDisease = len(hdata[hdata.target == 0])

countHaveDisease = len(hdata[hdata.target == 1])

print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(hdata.target))*100)))

print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(hdata.target))*100)))

countFemale = len(hdata[hdata.sex == 0])

countMale = len(hdata[hdata.sex == 1])

print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(hdata.sex))*100)))

print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(hdata.sex))*100)))

hdata.groupby('target').mean()

pd.crosstab(hdata.age,hdata.target).plot(kind="bar",figsize=(20,6))

plt.title('Heart Disease Frequency for Ages')

plt.xlabel('Age')

plt.ylabel('Frequency')

plt.savefig('heartDiseaseAndAges.png')

plt.show()
pd.crosstab(hdata.sex,hdata.target).plot(kind="bar",figsize=(15,6),color=['blue','#AA1111' ])

plt.title('Heart Disease Frequency for Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency')

plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(x='cholesterol',y='thalassemia',data=hdata,hue='target')

plt.show()
plt.figure(figsize=(8,6))

sns.scatterplot(x='thalassemia',y='resting_blood_pressure',data=hdata,hue='target')

plt.show()
plt.scatter(x=hdata.age[hdata.target==1], y=hdata.thalassemia[(hdata.target==1)], c="green")

plt.scatter(x=hdata.age[hdata.target==0], y=hdata.thalassemia[(hdata.target==0)])

plt.legend(["Disease", "Not Disease"])

plt.xlabel("Age")

plt.ylabel("Maximum Heart Rate")

plt.show()
pd.crosstab(hdata.fasting_blood_sugar,hdata.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])

plt.title('Heart Disease Frequency According To FBS')

plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')

plt.xticks(rotation = 0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency of Disease or Not')

plt.show()
hdata.dtypes
#### Missing Value Analysis

hdata.isnull().sum()
hdata.head(10)
# checking statistical values of dataset

hdata.describe()
# store numeric variables in cnames

cnames=['age','resting_blood_pressure','cholesterol','max_heart_rate_achieved','st_depression','num_major_vessels']
# Plot boxplot to visualise outliers

%matplotlib inline

plt.boxplot(hdata['resting_blood_pressure'])
# Detect outliers and replace with NA



for i in cnames:

    #print(i)

    q75,q25=np.percentile(hdata.loc[:,i],[75,25])  # extract quartiles 

    iqr=q75-q25                                         # calculate IQR

    minimum=q25-(iqr*1.5)                               # calculate inner and outer frames

    maximum=q75+(iqr*1.5)

    

    #print(minimum)

    #print(maximum)

    hdata.loc[hdata.loc[:,i] < minimum, i] = np.nan

    hdata.loc[hdata.loc[:,i] > maximum, i] = np.nan



    missing_value=pd.DataFrame(hdata.isnull().sum())   # calculating missing values
hdata=pd.DataFrame(KNN(k=3).fit_transform(hdata),columns=hdata.columns)  #performing knn imputation

hdata.isnull().sum()    
##Correlation analysis

#Correlation plot

df_corr = hdata.loc[:,cnames]

df_corr
#Set the width and hieght of the plot

f, ax = plt.subplots(figsize=(7, 5))



#Generate correlation matrix

corr = df_corr.corr()



#Plot using seaborn library

sns.heatmap(corr, annot = True, cmap='coolwarm',linewidths=.1)

plt.show()
X = add_constant(hdata)

pd.Series([variance_inflation_factor(X.values, i) 

               for i in range(X.shape[1])], 

              index=X.columns)

# Normality Check

%matplotlib inline

plt.hist(hdata['chest_pain_type'],bins='auto')

plt.hist(hdata['age'],bins='auto')
#Normalisation

for i in cnames:

    #print(i)

    hdata[i]=(hdata[i]-np.min(hdata[i]))/(np.max(hdata[i])-np.min(hdata[i]))
hdata.head(10)
# replace target variable  with yes or no

hdata['target'] = hdata['target'].replace(0, 'No')

hdata['target'] = hdata['target'].replace(1, 'Yes')
# to handle data imbalance issue we are dividing our dataset on basis of stratified sampling

# divide data into train and test

X=hdata.values[:,0:13]

Y=hdata.values[:,13]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2)
# Decision tree - we will build the model on train data and test it on test data

C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)

# predict new test cases

C50_Predictions = C50_model.predict(X_test) # applying decision tree model on test data set

data1=hdata.drop(['target'],axis=1)
#Create dot file to visualise tree  #http://webgraphviz.com/

dotfile = open("pt.dot", 'w')

df = tree.export_graphviz(C50_model, out_file=dotfile,feature_names=data1.columns)
# Confusion matrix of decision tree

CM = pd.crosstab(y_test, C50_Predictions)

CM
#let us save TP, TN, FP, FN

TN=CM.iloc[0,0]

FP=CM.iloc[0,1]

FN=CM.iloc[1,0]

TP=CM.iloc[1,1]
#check accuracy of model

accuracy=((TP+TN)*100)/(TP+TN+FP+FN)

accuracy
# check false negative rate of the model

fnr=FN*100/(FN+TP)

fnr
print(classification_report(y_test,C50_Predictions))
RF_model = RandomForestClassifier(n_estimators = 700).fit(X_train, y_train)

RF_model
# Apply RF on test data to check accuracy

RF_Predictions = RF_model.predict(X_test)

# To evaluate performance of any classification model we built confusion metrics

CM =pd.crosstab(y_test, RF_Predictions)

CM
#let us save TP, TN, FP, FN

TN=CM.iloc[0,0]

FP=CM.iloc[0,1]

FN=CM.iloc[1,0]

TP=CM.iloc[1,1]
#check accuracy of model

accuracy=((TP+TN)*100)/(TP+TN+FP+FN)

accuracy
# check  of the model

fnr=FN*100/(FN+TP)

fnr
print(classification_report(y_test,RF_Predictions))
# knn implementation

knn_model=KNeighborsClassifier(n_neighbors=4).fit(X_train,y_train)

# predict knn_predictions 

knn_predictions=knn_model.predict(X_test)
# build confusion metrics

CM=pd.crosstab(y_test,knn_predictions)

CM
# try K=1 through K=25 and record testing accuracy

k_range = range(1, 26)



# We can create Python dictionary using [] or dict()

scores = []

from sklearn import metrics

# We use a loop through the range 1 to 26

# We append the scores in the dictionary

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    scores.append(metrics.accuracy_score(y_test, y_pred))



print(scores)
#let us save TP, TN, FP, FN

TN=CM.iloc[0,0]

FP=CM.iloc[0,1]

FN=CM.iloc[1,0]

TP=CM.iloc[1,1]
#check accuracy of model

accuracy=((TP+TN)*100)/(TP+TN+FP+FN)

accuracy
# check false negative rate of the model

fnr=FN*100/(FN+TP)

fnr
print(classification_report(y_test,knn_predictions))
# Naive Bayes implementation

NB_model=GaussianNB().fit(X_train,y_train)
# predict test cases 

NB_predictions=NB_model.predict(X_test)
# build confusion metrics

CM=pd.crosstab(y_test,NB_predictions)

CM
#let us save TP, TN, FP, FN

TN=CM.iloc[0,0]

FP=CM.iloc[0,1]

FN=CM.iloc[1,0]

TP=CM.iloc[1,1]
#check accuracy of model

accuracy=((TP+TN)*100)/(TP+TN+FP+FN)

accuracy
# check false negative rate of the model

fnr=FN*100/(FN+TP)

fnr
print(classification_report(y_test,NB_predictions))
methods = [ "C50_model","RF_model","knn_model","NB_model"]

accuracy = [75.4,75.4,77.0, 73.7]

colors = ["purple", "magenta","#CFC60E","#0FBBAE"]



sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,10))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

sns.barplot(x=methods, y=accuracy, palette=colors)

plt.show()