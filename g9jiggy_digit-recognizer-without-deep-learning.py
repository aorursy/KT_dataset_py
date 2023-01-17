# Importing the usual libraries and filter warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pyplot import xticks
import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#train = pd.read_csv('train.csv')
#test = pd.read_csv('test.csv')

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

print(train.shape,test.shape)
#In the beginning it's important to check the size of your train and test data which later helps in 
#deciding the sample size while testing your model on train data
train.head(5)
test.head(5)
# Lets see if we have a null value in the whole dataset
#Usuall we will check isnull().sum() but here in our dataset number of columns are huge
print(np.unique([train.isnull().sum()]))
print(np.unique([test.isnull().sum()]))
y = train['label']
df_train = train.drop(columns=["label"],axis=1)
print(y.shape,df_train.shape)
#Looks like the values are equally distributed in the dataset
y.value_counts()
sns.countplot(y)
df_train = df_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
#Lets display first 50 images
plt.figure(figsize=(15,8))
for i in range(50):
    plt.subplot(5,10,i+1)
    plt.imshow(df_train[i].reshape((28,28)),cmap='binary')
    plt.axis("off")
plt.tight_layout()
plt.show()

y = train['label']
df_train = train.drop(columns=["label"],axis=1)
print(y.shape,df_train.shape)
# Normalize the dataset
df_train = df_train / 255
test = test / 255
# Loading 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
seed = 2
test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(df_train,y, test_size = test_size , random_state = seed)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#KNN
# we use n_neighbours-10 since we know our target variables are in the range of [0-9]
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print('Accuracy: %f' % accuracy)

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
test = test / 255
y_pred_test = knn.predict(test)
submission = pd.DataFrame({"ImageId": list(range(1, len(y_pred_test)+1)),"Label": y_pred_test})

submission.to_csv("submission_digit1.csv", index=False)
