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
df = pd.read_csv("/kaggle/input/fish-market/Fish.csv")
df.head()
df.isnull().sum()
df["Species"].value_counts()
print(f"percent distribution of Species:\n",(df["Species"].value_counts()/len(df))*100)
from sklearn.model_selection import train_test_split
# Split the data into features and targets

X = df[['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']]

y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, stratify=y, random_state = 0)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
import warnings

warnings.filterwarnings('ignore')
LR.fit(X_train,y_train)
y_pred=LR.predict(X_test)
output = pd.DataFrame({

    'y_test': y_test,

    'y_predict': y_pred

})
output
#confusion Matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix
import matplotlib.pyplot as plt

%matplotlib inline
import seaborn as sns
import seaborn as sns

sns.heatmap(cnf_matrix, annot=True, cmap="viridis")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
from sklearn.metrics import classification_report 
report = classification_report(y_test, y_pred) 
print(report)
from sklearn.metrics import accuracy_score 

print ('Accuracy Score :',accuracy_score(y_test, y_pred) )