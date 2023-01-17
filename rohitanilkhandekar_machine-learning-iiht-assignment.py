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
df=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.head()

df.columns
df.describe()
df.info()
df.shape
df['quality'].unique()
df['quality'].value_counts()
# Features V/s Target 
#features -> predictor(columns used for prediction)
#target -> predicted

features=list(df.columns)[:-1]
print(features)
target=list(df.columns)[-1:][0]
print(target)
import seaborn as sns
sns.pairplot(df,hue ='quality') 

# Separates features and corresponding labels/target 
# by dropping Quality - we get data frame with all features 

X = df.drop(['quality'], axis=1)  #  X will hold all features
y = df['quality'] # y will hold target/labels

print(X.shape) #dimensions of input data
print(y.shape) #dimensions of output data


#train is assigned 70% of data and test is assigned 30% of data, as test_size=0.3. 

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.3, random_state=0)
print(X_train.shape)
print(X_test.shape)
from sklearn.tree import DecisionTreeClassifier
classifier1=DecisionTreeClassifier(criterion='gini')
classifier1.fit(X_train,y_train)
print(classifier1)
classifier2 = DecisionTreeClassifier(criterion='entropy')  
classifier2.fit(X_train, y_train) 
y_pred_1 = classifier1.predict(X_test)  
print(y_pred_1)

y_pred_2 = classifier2.predict(X_test)  
print(y_pred_2)
from sklearn.metrics import accuracy_score
acc_1=accuracy_score(y_test,y_pred_1)
print('Accuracy for Gini model:{} %'.format(acc_1*100))

acc_2=accuracy_score(y_test,y_pred_2)
print('Accuracy for Entropy model:{} %'.format(acc_2*100))
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred_1))
print(classification_report(y_test, y_pred_1)) 
print(classifier1.feature_importances_)

#print(classifier1.feature_importances_) is explained below
import pandas as pd
feature_importances = pd.DataFrame(classifier1.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)
feature_importances
from sklearn import tree
from sklearn.tree import export_graphviz

tree.export_graphviz(classifier1, out_file='tree.dot', feature_names=['alcohol','sulphates','total sulfur dioxide','volatile acidity',
                                                                     'pH','residual sugar','chlorides','free sulfur dioxide', 'density',
                                                                     'critic acid','fixed acidity'],class_names = 'Quality',rounded = True, 
                     proportion = False, precision = 2, filled = True)

!dot -Tpng tree.dot -o tree.png
from IPython.display import Image
Image(filename = 'tree.png')
