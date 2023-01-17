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
df = pd.read_csv('../input/titanic-extended/full.csv')
df
df.head(n=11) #1st 11 rows
df.head(n=11).isnull() #looking for missing values
df.head(n=11).loc[5] # Displaying the 6th Row 
df.head(n=11)[5:6] #Alternative
df.head(n=11)[5:6].fillna("NaN") # filna() fills null or missing values
df.head(n=11)[5:6] 
import matplotlib.pyplot as plt
teams = ['CSK', 'KKR', 'DC', 'MI']
scores = [182,225,164,175]
plt.bar(teams,scores,width=0.8,color=['green','red','green','green'])
plt.xlabel('TEAMS')
plt.ylabel('SCORES')
plt.title('MATCH RESULTS')
plt.show()
print('KKR scored the Highest Run (Highlighted RED)')
import numpy as np
import pandas as pd
a=np.array([2,4,6,8,10,12])
b=np.array([1,6,5,8,9,10])
c=np.intersect1d(a,b)
print('Printing the common elements of the two numpy arrays:')
print(c)
print('Removing the common elemnts from the first array ')
for i in b:
    for k in a:
        if i==k:
            a=a[a!=k]
            
print(a)
print("Second array will remains unaffected:")
print('The second array: ')
print(b)
import numpy as np
import pandas as pd
train = pd.read_csv('../input/iris/Iris.csv')
train
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

x = train.drop("Species",axis=1)
y = train["Species"]

x_train, x_test, y_train, y_test = train_test_split(x,y, text_size = 0.3, random_state = 101)

logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)

predictions = logmodel.predict(x_test)

print("F1 Score: ", f1_score(y_test,predictions))
print("Confuson Matrix: \n")
confusion_matrix(y_test,predictions)
#Importing Necessary Libraries
#Preprocessing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

#Feature Selection Libraries
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# ML Libraries (Random Forest, Naive Bayes, SVM)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
 
# Evaluation Metrics
from yellowbrick.classifier import ClassificationReport
from sklearn import metrics
df = pd.read_csv('../input/iris/Iris.csv', error_bad_lines=False)
df.head()
df = df.drop(['Id'], axis=1)
df['Species'].unique() #target
df['Species'] = pd.factorize(df["Species"])[0] 
Target = 'Species'
df['Species'].unique()
Features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
print('Full Features: ', Features)
X_fs = df[Features]
Y_fs = df[Target]

model = LogisticRegression(solver='lbfgs', multi_class='auto')

rfe = RFE(model, 3) 
fit = rfe.fit(X_fs, Y_fs)

print("Number of Features Selected : %s" % (fit.n_features_))
print("Feature Ranking             : %s" % (fit.ranking_))
print("Selected Features           : %s" % (fit.support_))

Features = ['SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
print("Selected Features           :", Features)

#Split dataset to Training Set & Test Set
x, y = train_test_split(df, 
                        test_size = 0.2, 
                        train_size = 0.8, 
                        random_state= 3)

x1 = x[Features]    
x2 = x[Target]      
y1 = y[Features]    
y2 = y[Target]      
print('Feature Set Used    : ', Features)
print('Target Class        : ', Target)
print('Training Set Size   : ', x.shape)
print('Test Set Size       : ', y.shape)
nb_model = GaussianNB() 

# Model Training
nb_model.fit(X=x1, y=x2)

# Prediction with Test Set
result= nb_model.predict(y[Features]) 
ac_sc = accuracy_score(y2, result)
rc_sc = recall_score(y2, result, average="weighted")
pr_sc = precision_score(y2, result, average="weighted")
f1_sc = f1_score(y2, result, average='micro')
confusion_m = confusion_matrix(y2, result)
print("Accuracy    : ", ac_sc)
print("Recall      : ", rc_sc)
print("Precision   : ", pr_sc)
print("F1 Score    : ", f1_sc)
print("Confusion Matrix: ")
print(confusion_m)
