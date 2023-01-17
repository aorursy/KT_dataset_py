# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

import matplotlib 

import matplotlib.pyplot as plt

import csv
census_data = pd.read_csv('../input/adult-census-income/adult.csv')

census_data.head()
census_data.isnull().sum()
census_data.describe()
census_data.info()
census_data.workclass = census_data.workclass.replace({'?':'Not-Known'})

census_data.occupation = census_data.occupation.replace({'?':'Not-Known'})

census_data = census_data.rename(columns = {'education.num':'education_num'})

census_data = census_data.rename(columns ={'marital.status':'marital_status'})

census_data = census_data.rename(columns ={'capital.gain':'capital_gain'})

census_data = census_data.rename(columns ={'capital.loss':'capital_loss'})

census_data = census_data.rename(columns = {'hours.per.week':'hours_per_week'})

census_data = census_data.rename(columns ={'native.country':'native_country'})

census_data.head()
sns.countplot(x = 'sex',data = census_data)

plt.title("Gender")
census_data4 =census_data.groupby('workclass').sex.count().sort_values()

plt.title('Work - class')

census_data4.plot.bar()
census_data5 =census_data.groupby('occupation').sex.count().sort_values()

plt.title('Occupation')

census_data5.plot.bar()
census_data6 =census_data.groupby('marital_status').sex.count().sort_values()

plt.title('Marital_status')

census_data6.plot.bar()
census_data7 =census_data.groupby('relationship').sex.count().sort_values()

plt.title('Relationship')

census_data7.plot.bar()
census_data8 =census_data.groupby('race').sex.count().sort_values()

plt.title('Race')

census_data8.plot.bar()
census_data9 = pd.crosstab(census_data.sex , census_data.income)

print("Following is contigency table")

census_data9
a1 = [9592,1179]

a2 = [15128,6662]



a3 = np.array([a1,a2])



from scipy import stats

stats.chi2_contingency(a3)



chi2_stat, p_val, dof, ex = stats.chi2_contingency(a3)

print("Chisquare test value is : ",chi2_stat)

print("\nDegree of freedom is : ",dof)

print("\nP-Value is : ",p_val)

print("\nExpected observation contiggency table\n")

print(ex)
x,y,z = a3[0][1]+a3[0][0],a3[1][1]+a3[1][0],a3[0][0]+a3[1][0]+a3[0][1]+a3[1][1]

print('Number of female earning less than <=50K is ',a3[0][0])

print('Number of female observation is ',a3[0][1]+a3[0][0])

print('Number of male ',a3[1][1]+a3[1][0])

print('Total observation is ',a3[0][0]+a3[1][0]+a3[0][1]+a3[1][1])

print("Value of evaluation metric is ",((x*y)/z))
census_data10 = (census_data.groupby(['sex','income']).workclass.count()/census_data.groupby(['sex']).workclass.count())*100

census_data10
census_data11 = (census_data.groupby(['sex','income','workclass']).workclass.count()/census_data.groupby(['sex','income']).workclass.count())*100

census_data11
census_data11 = (census_data.groupby(['sex','income','marital_status']).workclass.count()/census_data.groupby(['sex','income']).workclass.count())*100

census_data11
census_data_v1 = census_data

census_data_v1.head()
dummy = pd.get_dummies(census_data_v1['workclass'])

census_data_v1 = pd.concat([census_data_v1 ,dummy],axis = 1)

census_data_v1 = census_data_v1.drop(['workclass'],axis = 1)

dummy = pd.get_dummies(census_data_v1['education'])

census_data_v1 = pd.concat([census_data_v1 ,dummy],axis = 1)

census_data_v1 = census_data_v1.drop(['education'],axis = 1)

dummy = pd.get_dummies(census_data_v1['marital_status'])

census_data_v1 = pd.concat([census_data_v1 ,dummy],axis = 1)

census_data_v1 = census_data_v1.drop(['marital_status'],axis = 1)

dummy = pd.get_dummies(census_data_v1['occupation'])

census_data_v1 = pd.concat([census_data_v1 ,dummy],axis = 1)

census_data_v1 = census_data_v1.drop(['occupation'],axis = 1)

dummy = pd.get_dummies(census_data_v1['relationship'])

census_data_v1 = pd.concat([census_data_v1 ,dummy],axis = 1)

census_data_v1 = census_data_v1.drop(['relationship'],axis = 1)

dummy = pd.get_dummies(census_data_v1['race'])

census_data_v1 = pd.concat([census_data_v1 ,dummy],axis = 1)

census_data_v1 = census_data_v1.drop(['race'],axis = 1)

dummy = pd.get_dummies(census_data_v1['sex'])

census_data_v1 = pd.concat([census_data_v1 ,dummy],axis = 1)

census_data_v1 = census_data_v1.drop(['sex'],axis = 1)

dummy = pd.get_dummies(census_data_v1['native_country'])

census_data_v1 = pd.concat([census_data_v1 ,dummy],axis = 1)

census_data_v1 = census_data_v1.drop(['native_country'],axis = 1)
X = census_data_v1[census_data_v1.columns.difference(['income'])]

y = census_data_v1['income']

y = y.replace('>50K',1)

y = y.replace('<=50K',0)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.metrics import mean_squared_error
X_train,val_X,y_train,val_y = train_test_split(X,y,test_size = 0.2,random_state = 0)

census_data_model = RandomForestClassifier(n_estimators = 500,bootstrap = True,max_features = 'sqrt')

census_data_model.fit(X_train,y_train)
print("Average absolute error value is " ,mean_absolute_error(val_y,census_data_model.predict(val_X)))

print("Average error square value is" ,mean_squared_error(val_y,census_data_model.predict(val_X)))

print("Root mean square error value is",np.sqrt(mean_squared_error(val_y,census_data_model.predict(val_X))))
y_pred_test = census_data_model.predict_proba(val_X)[:,1]

y_pred_train = census_data_model.predict_proba(X_train)[:,1]
from sklearn.metrics import roc_auc_score,average_precision_score,auc,roc_curve,precision_recall_curve
print("ROC Curve")

fpr , tpr ,thresold = roc_curve(val_y,y_pred_test)

roc_auc = auc(fpr,tpr)

plt.plot(fpr,tpr,label = 'ROC curve (area = %0.2f)'% roc_auc)

plt.xlabel("False Positve rate")

plt.ylabel("True Positive rate")

plt.legend(loc = 'lower right')
print("Precision Vs Recall Plot")

precision , recall , threshold = precision_recall_curve(val_y,y_pred_test)

average_precision =  average_precision_score(val_y,y_pred_test)

plt.plot(recall,precision,label = 'Precision recall curve (area = %0.2f)'% average_precision)

plt.xlabel("recall")

plt.ylabel("Precision")

plt.legend(loc = 'lower right')
max(y_pred_test[(y_pred_test >= 0.7) & (y_pred_test < 0.8)])
y_pred_test = np.where(y_pred_test > 0.798,1,0)

y_pred_train = np.where(y_pred_train > 0.798,1,0)
print("Confusion Matrix using test values")

matrix = confusion_matrix(val_y,y_pred_test)

sns.heatmap(matrix ,annot = True,cbar = True)
print("Confusion Matrix using train values")

matrix = confusion_matrix(y_train,y_pred_train)

sns.heatmap(matrix ,annot = True,cbar = True)
print("Following is Actual and predicted value table")

prediction_data = pd.DataFrame(val_X['age'])

prediction_data['Predicted_value'] = y_pred_test

prediction_data['Actual_value'] = val_y

prediction_data = prediction_data.sort_index(axis = 0)

prediction_data
prediction_data_v1 = pd.crosstab(prediction_data.Predicted_value , prediction_data.Actual_value)

print("Following is contigency table")

prediction_data_v1
import numpy as np
a1 = [4892,1015]

a2 = [74,532]



a3 = np.array([a1,a2])

print("recall is ",(a3[0][0])/(a3[0][0]+a3[1][0]))

print("precision is ",(a3[0][0])/(a3[0][0]+a3[0][1]))
print("Number of wrong prediction is ",prediction_data[prediction_data['Predicted_value'] != prediction_data['Actual_value']].Predicted_value.count()," out of total ",prediction_data['Predicted_value'].count(),"\nAnd Percentage of wrong prediction is ",round(prediction_data[prediction_data['Predicted_value'] != prediction_data['Actual_value']].Predicted_value.count()/prediction_data['Predicted_value'].count(),4),"\nNote Yes = 1 and No = 0 ")