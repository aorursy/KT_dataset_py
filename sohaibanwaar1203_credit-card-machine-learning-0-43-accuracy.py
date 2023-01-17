# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/creditcard.csv")

df.describe()
df.head(10)
import seaborn as sns

sns.heatmap(df.isnull())

print("NULL VALUES COUNT:\n",df.isnull().sum())
import seaborn as sns

sns.heatmap(df.corr())
import matplotlib.pyplot as plt

fig = plt.figure(figsize = (20, 25))

j = 0

#Droping_Characters and string coloums because graph donot support them



for i in df.columns:

    plt.subplot(6, 6, j+1)

    j += 1

    sns.distplot(df[i][df['Class']==1], color='g', label = 'Normal')

    sns.distplot(df[i][df['Class']==0], color='r', label = 'Fruad')

    plt.legend(loc='best')

fig.suptitle('Fruad detection ')

fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show()
from matplotlib.pyplot import pie, axis, show

fruad=len(df[df.Class==1])

normal=len(df[df.Class==0])

pie([fruad,normal], labels=["fruad:  "+str(fruad/(fruad+normal)),"Normal: "+str(normal/(fruad+normal))], pctdistance=1.1, labeldistance=1.2);

show()
for column in df:

    plt.figure()

    sns.boxplot(x=df[column])


Normal_df=df.drop("Class",axis=1)

Normal_df=Normal_df.drop("Amount",axis=1)



for column in Normal_df:

    plt.figure()

    sns.boxplot(x=Normal_df[column])
print(Normal_df.shape)

z_Scored_df=pd.DataFrame(Normal_df)

from scipy import stats

z_Scored_df=z_Scored_df[(np.abs(stats.zscore(Normal_df)) <1).all(axis=1)]

z_Scored_df.shape
df_y=df["Class"]

z_Scored_df=z_Scored_df.merge(df_y.to_frame(), left_index=True, right_index=True)

z_Scored_df.shape



for i in z_Scored_df:

    plt.figure()

    sns.boxplot(x=z_Scored_df[i])
z_Scored_df=z_Scored_df.merge(df["Amount"].to_frame(), left_index=True, right_index=True)

z_Scored_df.shape
#appending Fraud examples with normal

df_fruad=df[df.Class ==1]

z_Scored_df=z_Scored_df.append(df_fruad)

len(z_Scored_df)
from matplotlib.pyplot import pie, axis, show

fruad=len(z_Scored_df[z_Scored_df.Class==1])

normal=len(z_Scored_df[z_Scored_df.Class==0])

pie([fruad,normal], labels=["fruad:  "+str(fruad/(fruad+normal)),"Normal: "+str(normal/(fruad+normal))], pctdistance=1.1, labeldistance=1.2);

show()
x=z_Scored_df.drop("Class",axis=1)

y=z_Scored_df["Class"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

   x,y, test_size=0.3, random_state=0)
sns.heatmap(x.isnull())
from sklearn import ensemble



clf=ensemble.AdaBoostClassifier()

clf.fit(X_train,y_train)
from sklearn import metrics

y_pred=clf.predict(X_test)

df_confusion=metrics.confusion_matrix(y_test, y_pred)
from sklearn.metrics import confusion_matrix

prediction=clf.predict(X_test)

y_pred=[]

for i in prediction:

    y_pred.append(i.argmax())

y_pred=np.asarray(y_pred)

true_negative,false_positive,false_negative,true_positive=confusion_matrix(y_test, y_pred).ravel()



print("true_negative: ",true_negative)

print("false_positive: ",false_positive)

print("false_negative: ",false_negative)

print("true_positive: ",true_positive)

print("\n\n Accuracy Measures\n\n")

Sensitivity=true_positive/(true_positive+false_negative)

print("Sensitivity: ",Sensitivity)



False_Positive_Rate=false_positive/(false_positive+true_negative)

print("False_Positive_Rate: ",False_Positive_Rate)



Specificity=true_negative/(false_positive + true_negative)

print("Specificity: ",Specificity)



#FDR à 0 means that very few of our predictions are wrong

False_Discovery_Rate=false_positive/(false_positive+true_positive)

print("False_Discovery_Rate: ",False_Discovery_Rate)



Positive_Predictive_Value =true_positive/(true_positive+false_positive)

print("Positive_Predictive_Value: ",Positive_Predictive_Value)
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import f1_score

from sklearn.metrics import average_precision_score

from matplotlib import pyplot

from sklearn.metrics import auc

f1 = f1_score(y_test,prediction)

# calculate precision-recall AUC

precision, recall, thresholds = precision_recall_curve(y_test, prediction)

auc = auc(recall, precision)



ap = average_precision_score(y_test,prediction)

print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))

# plot no skill

pyplot.plot([0, 1], [0.5, 0.5], linestyle='--')

# plot the roc curve for the model

pyplot.plot(recall, precision, marker='.')

# show the plot

pyplot.show()
from sklearn.metrics import classification_report,accuracy_score

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor

from sklearn.svm import OneClassSVM

df=pd.read_csv("../input/creditcard.csv")

Fraud = df[df['Class']==1]



Valid = df[df['Class']==0]



outlier_fraction = len(Fraud)/float(len(Valid))

from sklearn.model_selection import train_test_split

x=df.drop("Class",axis=1)

y=df["Class"]

state = np.random.RandomState(42)

X_outliers = state.uniform(low=0, high=1, size=(x.shape[0], x.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(

   x,y, test_size=0.1, random_state=0)
classifiers = {

    

    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=10, algorithm='auto', 

                                              leaf_size=30, metric='euclidean',

                                              p=2, metric_params=None,  novelty=True,contamination=0.023)

    

}
n_outliers = len(Fraud)

for i, (clf_name,clf) in enumerate(classifiers.items()):

    #Fit the data and tag outliers

    if clf_name == "Local Outlier Factor":

        clf.fit(X_train)

        y_pred =clf.predict(X_train)

        scores_prediction = clf.negative_outlier_factor_

        

        

   #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions

    y_pred[y_pred == 1] = 0

    y_pred[y_pred == -1] = 1

    n_errors = (y_pred != y_train).sum()

    # Run Classification Metrics

    print("{}: {}".format(clf_name,n_errors))

    print("Accuracy Score :")

    print(accuracy_score(y_train,y_pred))

    print("Classification Report :")

    print(classification_report(y_train,y_pred))
from sklearn.metrics import confusion_matrix

y_pred=clf.predict(X_test)



y_pred[y_pred == 1] = 0

y_pred[y_pred == -1] = 1

true_negative,false_positive,false_negative,true_positive=confusion_matrix(y_test,y_pred).ravel()



print("true_negative: ",true_negative)

print("false_positive: ",false_positive)

print("false_negative: ",false_negative)

print("true_positive: ",true_positive)

print("\n\n Accuracy Measures\n\n")

Accuracy=(true_positive+true_negative)/(true_positive+false_negative+true_negative+false_positive)

print("Accuracy: ",Sensitivity)



Sensitivity=true_positive/(true_positive+false_negative)

print("Sensitivity: ",Sensitivity)



False_Positive_Rate=false_positive/(false_positive+true_negative)

print("False_Positive_Rate: ",False_Positive_Rate)



Specificity=true_negative/(false_positive + true_negative)

print("Specificity: ",Specificity)



#FDR à 0 means that very few of our predictions are wrong

False_Discovery_Rate=false_positive/(false_positive+true_positive)

print("False_Discovery_Rate: ",False_Discovery_Rate)



Positive_Predictive_Value =true_positive/(true_positive+false_positive)

print("Positive_Predictive_Value: ",Positive_Predictive_Value)
plt.title("Local Outlier Factor (LOF)")

plt.scatter(X_train.iloc[:, 1], X_train.iloc[:, 2], color='k', s=3., label='Data points')

# plot circles with radius proportional to the outlier scores

radius = (scores_prediction.max() - scores_prediction) / (scores_prediction.max() - scores_prediction.min())

plt.scatter(X_train.iloc[:, 1], X_train.iloc[:, 2], s=1000 * radius, edgecolors='r',

            facecolors='none', label='Outlier scores')

plt.axis('tight')



plt.xlabel("prediction errors: %d" % (n_errors))

legend = plt.legend(loc='upper left')

legend.legendHandles[0]._sizes = [10]

legend.legendHandles[1]._sizes = [20]

plt.show()