# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/phishing-website-dataset/dataset.csv')
df.head()
df['Result'].unique()
col=df.columns
for i in col:print(i)
df.shape
import seaborn as sns
sns.countplot(df['Result'])
for i in col:
     if  i!='index':
        print(i,df[i].unique())
print(df.corr()['Result'].sort_values())      # Print correlation with target variable
# Remove features having correlation coeff. between +/- 0.03
df.drop(['Favicon','Iframe','Redirect','popUpWidnow','RightClick','Submitting_to_email'],axis=1,inplace=True)
print(len(df.columns))
plt.figure(figsize=(15, 15))
sns.heatmap(df.corr(), linewidths=.5)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score

X= df.drop(columns='Result')
Y=df['Result']

train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.3,random_state=2)
knn=KNeighborsClassifier(n_neighbors=3)
model= knn.fit(train_X,train_Y)
knn_predict=model.predict(test_X)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

accuracy_score(knn_predict,test_Y)
neighbors = np.arange(1, 15)

train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors= k )

    # Fit the classifier to the training data
    knn.fit(train_X, train_Y)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(train_X, train_Y)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(test_X, test_Y)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()