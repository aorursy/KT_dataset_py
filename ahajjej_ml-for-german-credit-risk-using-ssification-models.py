# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
!pip install plotly
import plotly.offline as py 
import plotly.graph_objs as go
import plotly.express as px
from collections import Counter # To do counter of some features
from subprocess import call
from IPython.display import Image
############################################################################################
%matplotlib inline 
from sklearn.datasets import make_classification 
#from sklearn.learning_curve import learning_curve 
#from sklearn.cross_validation import train_test_split 
#from sklearn.grid_search import GridSearchCV
#from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import ShuffleSplit,train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
credit=pd.read_csv("../input/german-credit-data-with-risk/german_credit_data.csv")
print("The dataset is {} credit record".format(len(credit)))
credit.head(2)
credit=credit.iloc[:, 1:]
credit.info()
credit.describe()
credit['Risk'] = credit['Risk'].map({'bad':0, 'good':1})
credit['Saving accounts'] = credit['Saving accounts'].fillna('Other')
credit['Checking account'] = credit['Checking account'].fillna('Other')
credit_clean=credit.copy()
cat_features = ['Sex','Housing', 'Saving accounts', 'Checking account','Purpose']
num_features=['Age', 'Job', 'Credit amount', 'Duration','Risk']
for variable in cat_features:
    dummies = pd.get_dummies(credit_clean[cat_features])
    df1= pd.concat([credit_clean[num_features], dummies],axis=1)

Risk= df1['Risk']          
df2=df1.drop(['Risk'],axis=1)
X_train,X_test,Y_train,Y_test = train_test_split(df2,Risk,test_size=0.20,random_state = 30)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_test_pred = lr.predict(X_test)

confusion_matrix = confusion_matrix(Y_test, Y_test_pred)
confusion_matrix
y_true = ["bad", "good"]
y_pred = ["bad", "good"]
df_cm = pd.DataFrame(confusion_matrix, columns=np.unique(y_true), index = np.unique(y_true))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
df_cm.dtypes

plt.figure(figsize = (8,5))
plt.title('Confusion Matrix')
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 12})# font size
total=sum(sum(confusion_matrix))

sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
print('Sensitivity : ', sensitivity )

Recall_Specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
print('Recall: ', Recall_Specificity)

precision = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])
print('precision: ', precision)
fpr, tpr, thresholds = roc_curve(Y_test, Y_test_pred)

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12

plt.xlabel('False Positive Rate (1 - Recall)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

print("\n")
print ("Area Under Curve: %.2f" %auc(fpr, tpr))
print("\n")