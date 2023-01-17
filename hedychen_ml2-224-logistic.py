import pandas as pd

import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score

from sklearn.model_selection import train_test_split 

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/framingham-heart-study-dataset/framingham.csv')
data.dtypes
data.head(15)
data.describe()
data.isnull().sum()
data.shape
data[data['cigsPerDay'].isna()]['currentSmoker'].drop_duplicates()
data[['currentSmoker','TenYearCHD']].corr()
data.dropna(inplace=True,how='any')
data.shape
corr_result = data[['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose','TenYearCHD']].corr()
corr_result
corr_result['TenYearCHD'].describe()
corr_result[corr_result['TenYearCHD']>0.12].index
sns.pairplot(data[['education','age', 'sysBP', 'diaBP', 'glucose', 'TenYearCHD']])
x = data.loc[:,['age', 'sysBP','male','BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']].values

y = data['TenYearCHD'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=15)
model = LogisticRegression()

model.fit(x_train, y_train)

model_ypred_prob = model.predict_proba(x_test)

model_ypred = model.predict(x_test)
roc_auc_score(y_test, model_ypred_prob[:,1])
cm=confusion_matrix(y_test,model_ypred)

conf_matrix=pd.DataFrame(cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])

plt.figure(figsize = (7,6))

sns.heatmap(conf_matrix, annot=True,fmt='d')
accuracy_score(y_test,model_ypred)
fpr, tpr, thresholds = roc_curve(y_test, model_ypred_prob[:,1])

plt.plot(fpr,tpr)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for Heart disease classifier')

plt.xlabel('False positive rate (1-Specificity)')

plt.ylabel('True positive rate (Sensitivity)')

plt.grid(True)