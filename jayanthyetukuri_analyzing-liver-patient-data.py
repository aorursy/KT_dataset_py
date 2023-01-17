import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



liverData=pd.read_csv("../input/indian_liver_patient.csv")

liverData.head(3)
liverData.info()
liverData['Albumin_and_Globulin_Ratio'].fillna(value=0, inplace=True) #Fill the NaN's with 0's

liverData['Dataset'].replace(to_replace=1, value='patient', inplace=True) #Replacing the class labels

liverData['Dataset'].replace(to_replace=2, value='nonpatient', inplace=True) #Replacing the class labels

liverData.head(3)
fig = plt.figure()

ax = fig.add_subplot(111)

ax.spines["top"].set_visible(False)

ax.spines["bottom"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["left"].set_visible(False)



df1 = pd.value_counts(liverData.loc[liverData['Gender'] == 'Male']['Dataset'], sort = True).sort_index()

df2 = pd.value_counts(liverData.loc[liverData['Gender'] == 'Female']['Dataset'], sort = True).sort_index()

df1.plot(kind='bar', color='salmon', ax=ax, position=0, width=0.25)

df2.plot(kind='bar', color='mediumturquoise', ax=ax, position=1, width=0.25)



plt.title("Patient frequency histogram", fontsize=16)

plt.text(-0.4, 240, "Male", color='salmon', fontweight='bold', fontsize=14)

plt.text(-0.4, 210, "Female", color='mediumturquoise', fontweight='bold', fontsize=14)

    
liverData.describe()
from sklearn.preprocessing import MinMaxScaler

simpleScaler=MinMaxScaler()

cols=list(liverData.drop(['Gender', 'Dataset'],axis=1).columns)

liverDataScaled=pd.DataFrame(data=liverData)

liverDataScaled[cols]=simpleScaler.fit_transform(liverData[cols])

liverDataScaled.head(3)
liverDataScaledEncoded=pd.get_dummies(liverDataScaled)

liverDataScaledEncoded.head(3)
boxprops = dict(linestyle='-', color='k')

medianprops = dict(linestyle='-', color='k')

plt.figure()

ax = plt.subplot(111)

ax.spines["top"].set_visible(False)

ax.spines["bottom"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["left"].set_visible(False)

plt.title("Scaled feature (patients)", fontsize=16)

liverDataScaledEncodedBoxPlotDF=liverDataScaledEncoded.loc[liverDataScaledEncoded['Dataset_patient'] == 1].drop(

                                ['Gender_Male', 'Gender_Female', 'Dataset_patient', 'Dataset_nonpatient'],axis=1)

#liverDataScaledEncodedBoxPlotDF = liverDataScaledEncodedBoxPlotDF.sort_values(by=['Total_Bilirubin'], ascending=[True])

bp = liverDataScaledEncodedBoxPlotDF.boxplot(vert=False, showmeans=True, showfliers=False,

                boxprops=boxprops,

                medianprops=medianprops)


import matplotlib.pyplot as plt

plt.figure()

ax = plt.subplot(111)

ax.spines["top"].set_visible(False)

ax.spines["bottom"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["left"].set_visible(False)

plt.title("Scaled feature (non-patients)", fontsize=16)

liverDataScaledEncodedBoxPlotDF=liverDataScaledEncoded.loc[liverDataScaledEncoded['Dataset_nonpatient'] == 1].drop(

                                ['Gender_Male', 'Gender_Female', 'Dataset_patient', 'Dataset_nonpatient'],axis=1)

bp = liverDataScaledEncodedBoxPlotDF.boxplot(vert=False, showmeans=True, showfliers=False,

                boxprops=boxprops,

                medianprops=medianprops)
from sklearn.cross_validation import train_test_split

train_x, test_x, train_y, test_y = train_test_split(liverDataScaledEncoded.drop(['Gender_Male', 'Gender_Female', 

                                                                            'Dataset_patient', 'Dataset_nonpatient'], axis=1), 

                                                    liverDataScaledEncoded['Dataset_patient'], train_size=0.75) 

print("Test data size: " + str(train_x.shape))

print("Test data size: " + str(test_x.shape))
from sklearn import metrics

from sklearn import linear_model

mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(train_x, train_y)

print("Multinomial Logistic regression Train Accuracy :: "+ str(metrics.accuracy_score(train_y, mul_lr.predict(train_x))))

print("Multinomial Logistic regression Test Accuracy :: "+ str(metrics.accuracy_score(test_y, mul_lr.predict(test_x))))