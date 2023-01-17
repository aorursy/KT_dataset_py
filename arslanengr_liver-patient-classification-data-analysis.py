import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
patient_data = pd.read_csv('../input/indian_liver_patient.csv')
patient_data.columns.groupby(patient_data.dtypes)
patient_data['Gender'].unique()
gender = pd.get_dummies(patient_data['Gender'], drop_first=True)
patient_data.drop(['Gender'],axis =1 , inplace=True)
patient_data=pd.concat([patient_data,gender],axis=1)
patient_data.dtypes
patient_data.head()
plt.figure(figsize=(10,4))
sns.heatmap(patient_data.isnull(),yticklabels=False, cbar=False, cmap='viridis')
df = patient_data.rename(index=str,columns={"Male":"G"})
from sklearn.model_selection import train_test_split
X = df[['Age', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
        'Aspartate_Aminotransferase', 'Total_Bilirubin', 'Direct_Bilirubin', 'Total_Protiens', 'Albumin',
        'Albumin_and_Globulin_Ratio','G']]
y = df['Dataset']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test,predictions))
print('MSE:', metrics.mean_squared_error(y_test,predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,predictions)))
