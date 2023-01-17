import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(color_codes =True)

%matplotlib inline
diab=pd.read_csv("../input/pima-diabetes/diabetes.csv")

diab.head()
diab.isnull().values.any()

## To check if data contains null values
diab.describe()

## To run numerical descriptive stats for the data set
(diab.Pregnancies == 0).sum(),(diab.Glucose==0).sum(),(diab.BloodPressure==0).sum(),(diab.SkinThickness==0).sum(),(diab.Insulin==0).sum(),(diab.BMI==0).sum(),(diab.DiabetesPedigreeFunction==0).sum(),(diab.Age==0).sum()

## Counting cells with 0 Values for each variable and publishing the counts below
## Creating a dataset called 'dia' from original dataset 'diab' with excludes all rows with have zeros only for Glucose, BP, Skinthickness, Insulin and BMI, as other columns can contain Zero values.

drop_Glu=diab.index[diab.Glucose == 0].tolist()

drop_BP=diab.index[diab.BloodPressure == 0].tolist()

drop_Skin = diab.index[diab.SkinThickness==0].tolist()

drop_Ins = diab.index[diab.Insulin==0].tolist()

drop_BMI = diab.index[diab.BMI==0].tolist()

c=drop_Glu+drop_BP+drop_Skin+drop_Ins+drop_BMI

dia=diab.drop(diab.index[c])
dia.info()
dia.describe()
dia1 = dia[dia.Outcome==1]

dia0 = dia[dia.Outcome==0]
dia1
dia0
## creating count plot with title using seaborn

sns.countplot(x=dia.Outcome)

plt.title("Count Plot for Outcome")
# Computing the %age of diabetic and non-diabetic in the sample

Out0=len(dia[dia.Outcome==1])

Out1=len(dia[dia.Outcome==0])

Total=Out0+Out1

PC_of_1 = Out1*100/Total

PC_of_0 = Out0*100/Total

PC_of_1, PC_of_0
## Creating 3 subplots - 1st for histogram, 2nd for histogram segmented by Outcome and 3rd for representing same segmentation using boxplot

plt.figure(figsize=(20, 6))

plt.subplot(1,3,1)

sns.set_style("dark")

plt.title("Histogram for Pregnancies")

sns.distplot(dia.Pregnancies,kde=False)

plt.subplot(1,3,2)

sns.distplot(dia0.Pregnancies,kde=False,color="Blue", label="Preg for Outome=0")

sns.distplot(dia1.Pregnancies,kde=False,color = "Gold", label = "Preg for Outcome=1")

plt.title("Histograms for Preg by Outcome")

plt.legend()

plt.subplot(1,3,3)

sns.boxplot(x=dia.Outcome,y=dia.Pregnancies)

plt.title("Boxplot for Preg by Outcome")
plt.figure(figsize=(20, 6))

plt.subplot(1,3,1)

plt.title("Histogram for Glucose")

sns.distplot(dia.Glucose, kde=False)

plt.subplot(1,3,2)

sns.distplot(dia0.Glucose,kde=False,color="Gold", label="Gluc for Outcome=0")

sns.distplot(dia1.Glucose, kde=False, color="Blue", label = "Gloc for Outcome=1")

plt.title("Histograms for Glucose by Outcome")

plt.legend()

plt.subplot(1,3,3)

sns.boxplot(x=dia.Outcome,y=dia.Glucose)

plt.title("Boxplot for Glucose by Outcome")
plt.figure(figsize=(20, 6))

plt.subplot(1,3,1)

sns.distplot(dia.BloodPressure, kde=False)

plt.title("Histogram for Blood Pressure")

plt.subplot(1,3,2)

sns.distplot(dia0.BloodPressure,kde=False,color="Gold",label="BP for Outcome=0")

sns.distplot(dia1.BloodPressure,kde=False, color="Blue", label="BP for Outcome=1")

plt.legend()

plt.title("Histogram of Blood Pressure by Outcome")

plt.subplot(1,3,3)

sns.boxplot(x=dia.Outcome,y=dia.BloodPressure)

plt.title("Boxplot of BP by Outcome")
plt.figure(figsize=(20, 6))

plt.subplot(1,3,1)

sns.distplot(dia.SkinThickness, kde=False)

plt.title("Histogram for Skin Thickness")

plt.subplot(1,3,2)

sns.distplot(dia0.SkinThickness, kde=False, color="Gold", label="SkinThick for Outcome=0")

sns.distplot(dia1.SkinThickness, kde=False, color="Blue", label="SkinThick for Outcome=1")

plt.legend()

plt.title("Histogram for SkinThickness by Outcome")

plt.subplot(1,3,3)

sns.boxplot(x=dia.Outcome, y=dia.SkinThickness)

plt.title("Boxplot of SkinThickness by Outcome")
plt.figure(figsize=(20, 6))

plt.subplot(1,3,1)

sns.distplot(dia.Insulin,kde=False)

plt.title("Histogram of Insulin")

plt.subplot(1,3,2)

sns.distplot(dia0.Insulin,kde=False, color="Gold", label="Insulin for Outcome=0")

sns.distplot(dia1.Insulin,kde=False, color="Blue", label="Insuline for Outcome=1")

plt.title("Histogram for Insulin by Outcome")

plt.legend()

plt.subplot(1,3,3)

sns.boxplot(x=dia.Outcome, y=dia.Insulin)

plt.title("Boxplot for Insulin by Outcome")
plt.figure(figsize=(20, 6))

plt.subplot(1,3,1)

sns.distplot(dia.BMI, kde=False)

plt.title("Histogram for BMI")

plt.subplot(1,3,2)

sns.distplot(dia0.BMI, kde=False,color="Gold", label="BMI for Outcome=0")

sns.distplot(dia1.BMI, kde=False, color="Blue", label="BMI for Outcome=1")

plt.legend()

plt.title("Histogram for BMI by Outcome")

plt.subplot(1,3,3)

sns.boxplot(x=dia.Outcome, y=dia.BMI)

plt.title("Boxplot for BMI by Outcome")
plt.figure(figsize=(20, 6))

plt.subplot(1,3,1)

sns.distplot(dia.DiabetesPedigreeFunction,kde=False)

plt.title("Histogram for Diabetes Pedigree Function")

plt.subplot(1,3,2)

sns.distplot(dia0.DiabetesPedigreeFunction, kde=False, color="Gold", label="PedFunction for Outcome=0")

sns.distplot(dia1.DiabetesPedigreeFunction, kde=False, color="Blue", label="PedFunction for Outcome=1")

plt.legend()

plt.title("Histogram for DiabetesPedigreeFunction by Outcome")

plt.subplot(1,3,3)

sns.boxplot(x=dia.Outcome, y=dia.DiabetesPedigreeFunction)

plt.title("Boxplot for DiabetesPedigreeFunction by Outcome")
plt.figure(figsize=(20, 6))

plt.subplot(1,3,1)

sns.distplot(dia.Age,kde=False)

plt.title("Histogram for Age")

plt.subplot(1,3,2)

sns.distplot(dia0.Age,kde=False,color="Gold", label="Age for Outcome=0")

sns.distplot(dia1.Age,kde=False, color="Blue", label="Age for Outcome=1")

plt.legend()

plt.title("Histogram for Age by Outcome")

plt.subplot(1,3,3)

sns.boxplot(x=dia.Outcome,y=dia.Age)

plt.title("Boxplot for Age by Outcome")
## importing stats module from scipy

from scipy import stats

## retrieving p value from normality test function

PregnanciesPVAL=stats.normaltest(dia.Pregnancies).pvalue

GlucosePVAL=stats.normaltest(dia.Glucose).pvalue

BloodPressurePVAL=stats.normaltest(dia.BloodPressure).pvalue

SkinThicknessPVAL=stats.normaltest(dia.SkinThickness).pvalue

InsulinPVAL=stats.normaltest(dia.Insulin).pvalue

BMIPVAL=stats.normaltest(dia.BMI).pvalue

DiaPeFuPVAL=stats.normaltest(dia.DiabetesPedigreeFunction).pvalue

AgePVAL=stats.normaltest(dia.Age).pvalue

## Printing the values

print("Pregnancies P Value is " + str(PregnanciesPVAL))

print("Glucose P Value is " + str(GlucosePVAL))

print("BloodPressure P Value is " + str(BloodPressurePVAL))

print("Skin Thickness P Value is " + str(SkinThicknessPVAL))

print("Insulin P Value is " + str(InsulinPVAL))

print("BMI P Value is " + str(BMIPVAL))

print("Diabetes Pedigree Function P Value is " + str(DiaPeFuPVAL))

print("Age P Value is " + str(AgePVAL))
sns.pairplot(dia, vars=["Pregnancies", "Glucose","BloodPressure","SkinThickness","Insulin", "BMI","DiabetesPedigreeFunction", "Age"],hue="Outcome")

plt.title("Pairplot of Variables by Outcome")
cor = dia.corr(method ='pearson')

cor
sns.heatmap(cor)
cols=["Pregnancies", "Glucose","BloodPressure","SkinThickness","Insulin", "BMI","DiabetesPedigreeFunction", "Age"]

X=dia[cols]

y=dia.Outcome
## Importing stats models for running logistic regression

import statsmodels.api as sm

## Defining the model and assigning Y (Dependent) and X (Independent Variables)

logit_model=sm.Logit(y,X)

## Fitting the model and publishing the results

result=logit_model.fit()

print(result.summary())
cols2=["Pregnancies", "Glucose","BloodPressure","SkinThickness","BMI"]

X=dia[cols2]

logit_model=sm.Logit(y,X)

result=logit_model.fit()

print(result.summary2())
cols3=["Pregnancies", "Glucose","BloodPressure","SkinThickness"]

X=dia[cols3]

logit_model=sm.Logit(y,X)

result=logit_model.fit()

print(result.summary())
cols4=["Pregnancies", "Glucose","BloodPressure"]

X=dia[cols4]

logit_model=sm.Logit(y,X)

result=logit_model.fit()

print(result.summary())
## Importing LogisticRegression from Sk.Learn linear model as stats model function cannot give us classification report and confusion matrix

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

cols4=["Pregnancies", "Glucose","BloodPressure"]

X=dia[cols4]

y=dia.Outcome

logreg.fit(X,y)

## Defining the y_pred variable for the predicting values. I have taken 392 dia dataset. We can also take a test dataset

y_pred=logreg.predict(X)

## Calculating the precision of the model

from sklearn.metrics import classification_report

print(classification_report(y,y_pred))

from sklearn.metrics import confusion_matrix

## Confusion matrix gives the number of cases where the model is able to accurately predict the outcomes.. both 1 and 0 and how many cases it gives false positive and false negatives

confusion_matrix = confusion_matrix(y, y_pred)

print(confusion_matrix)