import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
data = pd.read_csv('/kaggle/input/lending-club-loan-data/loan.csv')
data['loan_status'] = data['loan_status'].astype(str)

datanew = data[(data['loan_status']=='Fully Paid')| (data['loan_status']=='Charged Off') | (data['loan_status']=='Default')]

datanew['Status'] = 0

datanew['Status'][data['loan_status']=='Fully Paid'] = 1

datanew.drop(['loan_status'],1,inplace=True)
na_columns = datanew.isna().sum()

na_columns = na_columns[na_columns < 100000]

datanew = datanew[list(na_columns.index)]
datanew.groupby(['Status']).size()
numericCols = list(datanew._get_numeric_data().columns)

categoricalCols = list(set(datanew.columns) - set(numericCols))
datanew[categoricalCols] = datanew[categoricalCols].replace({ np.nan:'missing'})

datanew[numericCols] = datanew[numericCols].replace({ np.nan:-1})



y = datanew['Status']

datanew.drop(['Status'],1,inplace=True)
from sklearn.preprocessing import LabelEncoder

for c in categoricalCols:

    lb = LabelEncoder()

    datanew[c] = lb.fit_transform(datanew[c])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(datanew, y, test_size = 0.2, random_state = 0)
from lightgbm import LGBMClassifier

classifier = LGBMClassifier()

classifier.fit(X_train, y_train)

y_predict = classifier.predict_proba(X_test)[:,1]
y1 = pd.DataFrame(y_predict,columns=['prob'])

y1['hard'] = np.where(y1['prob']<0.5,0,1)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y1['hard'])

print(cm)

rf_real_auc = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])

print("CM Accuracy : "+str(rf_real_auc))

rf_pred = pd.DataFrame(y_predict,columns=['self_pay_status'])
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

import seaborn as sns



fpr, tpr, thresholds = roc_curve( y_test, y_predict)

auc_score = auc(fpr, tpr)

lgb_auc_score = auc_score

print("AUC : "+str(lgb_auc_score))

plt.figure(figsize=(16, 6))

lw = 2

plt.plot(fpr, tpr, lw=lw, label='ROC curve (area = %0.5f)' % lgb_auc_score)  

plt.plot([0, 1], [0, 1], color='g', lw=lw, linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.show() 
feature_importances = pd.DataFrame()

feature_importances['feature'] = X_train.columns



feature_importances['average'] = classifier.feature_importances_

plt.figure(figsize=(15, 4))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(10), x='average', y='feature');

plt.title(' TOP 7 feature importance',fontsize=15)

plt.xticks(fontsize=15)

plt.ylabel('Features',fontsize=20)

plt.yticks(fontsize=15)