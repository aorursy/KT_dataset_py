# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
data  = pd.read_csv('../input/diabetes.csv')
data.head()
# data type of all variables of data set
data.info()
# describe function in python return basic statistical measurements
data.describe()
# checking null value in the data set
data.isnull().sum()
data_columns = data.columns
data_columns = data_columns[:-1]

for col in data_columns:
    plt.figure(figsize = (4,4))
    plt.xlabel(col)
    sns.distplot(data[col],hist = True,kde = True)
print('''
If skewness is less than −1 or greater than +1, the distribution is highly skewed. 
If skewness is between −1 and −½ or between +½ and +1, the distribution is moderately skewed.
If skewness is between −½ and +½, the distribution is approximately symmetric.
Also from below output and above histogram graph 
Variables like - Preganancies,Glucose,SkinThickness are approximately symmetric - normally distributed
variables like - BloodPressure,Insulin,DiabetesPedigreeFunction,Age are highly Skeweed
and variables like - BMI is Moderately skewed
''')
data.skew(axis = 0, skipna = True) 
print("detecting outlier for each variables")
for col in data_columns:
    plt.figure(figsize = (5,5))
    data.boxplot(col)
data_1 = data.drop('Outcome', axis=1)
# From above boxplot it seems that data contain many outliers so we need to apply outlier treatment
def outlier_detect(data_1):
    for i in data_1.describe().columns:
        Q1=data_1.describe().at['25%',i]
        Q3=data_1.describe().at['75%',i]
        IQR=Q3 - Q1
        LTV=Q1 - 1.5 * IQR
        UTV=Q3 + 1.5 * IQR
        x=np.array(data_1[i])
        p=[]
        for j in x:
            if(j > UTV):
                p.append(data_1[i].quantile(0.75) + 1.5 *(data_1[i].quantile(0.75) - data_1[i].quantile(0.25)))
                if(j < LTV):
                    p.append(data_1[i].quantile(0.25) - 1.5 *(data_1[i].quantile(0.75) - data_1[i].quantile(0.25)))  
            else:
                p.append(j)            
        data_1[i]=p
    return data_1
data_1 = outlier_detect(data_1)
data_1 = round(data_1,2)
# Use normalization to convert all variables in one scale from 0 to 1
data_columns = data_1.columns[:-1]
for col in data_columns:
    data_1[col] = (data_1[col] - data_1[col].min())/(data_1[col].max() - data_1[col].min())
    
data_1.head()
data_1["Outcome"] = data.Outcome
# find correlation for each variables
corr_x = data_1.corr()
corr_target = abs(corr_x['Outcome'])
corr_df = pd.DataFrame(corr_target)
corr_df['VIF'] = 1/(1-(corr_df)**2)
corr_df
print('''from above vif value for each variable is less than 10 
so there is no multi collinearity present in the data set this means that all variables are independent.
Now we will procede to build model on given data set. Here outcome variable is dependent variable 
while others are independent variables.
''')
print("seperating data into the two part \n1)Input variables which X and 2)Output variable which y ")
X = data_1[data_1.columns.difference(['Outcome'])]
y = data_1['Outcome']
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.metrics import mean_squared_error
data_1['Outcome'].value_counts()/data_1['Outcome'].count()
print("Seperating X and y data into two part \n80% belongs to trian while 20% belongs to test \nAnd droping string variables")
X_train,val_X,y_train,val_y = train_test_split(X,y,test_size = 0.3,random_state = 0)
model = RandomForestClassifier(n_estimators = 100,bootstrap = True,max_features = 'sqrt')
model.fit(X_train,y_train)
print("Average absolute error value is " ,mean_absolute_error(val_y,model.predict(val_X)))
print("Average error square value is" ,mean_squared_error(val_y,model.predict(val_X)))
print("Root mean square error value is",np.sqrt(mean_squared_error(val_y,model.predict(val_X))))
print(classification_report(val_y,model.predict(val_X)))
y_pred_test = model.predict_proba(val_X)[:,1]
y_pred_train = model.predict_proba(X_train)[:,1]
from sklearn.metrics import roc_auc_score,average_precision_score,auc,roc_curve,precision_recall_curve
print("ROC Curve")
fpr , tpr ,thresold = roc_curve(val_y,y_pred_test)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,label = 'ROC curve (area = %0.2f)'% roc_auc)
plt.xlabel("False Positve rate")
plt.ylabel("True Positive rate")
plt.legend(loc = 'lower right')
y_pred_test = np.where(y_pred_test > 0.5,1,0)
y_pred_train = np.where(y_pred_train > 0.5,1,0)
print("Confusion Matrix using test values")
matrix = confusion_matrix(val_y,y_pred_test)
sns.heatmap(matrix ,annot = True,cbar = True)
importances = model.feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Rearrange feature names so they match the sorted feature importances
names = [data_1.columns[i] for i in indices]

# Create plot
plt.figure()

# Create plot title
plt.title("Feature Importance")

# Add bars
plt.bar(range(X.shape[1]), importances[indices])

# Add feature names as x-axis labels
plt.xticks(range(X.shape[1]), names, rotation=90)

# Show plot
plt.show()