#19BDA71036 Desmond Gracian Rebello
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import sklearn.model_selection as ms
from sklearn import preprocessing
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn import metrics
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score,precision_score,recall_score,f1_score
from sklearn.tree import DecisionTreeClassifier            #importing the library for decision tree model
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Loading the dataset
df= pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv")
X = df.drop("flag",1)   #Feature Matrix
y = df["flag"]          #Target Variable
df.shape
#Checking for class imbalance
print(df['flag'].value_counts())
sns.countplot(x='flag',data=df)
plt.show()
count_anomaly = len(df[df['flag']==0])
count_normal = len(df[df['flag']==1])
pct_anomaly = count_anomaly/(count_anomaly+count_normal)
print("percentage of anomaly:", pct_anomaly*100)
pct_normal = count_normal/(count_anomaly+count_normal)
print("percentage of normal:", pct_normal*100)
print("There is no class imbalance")
#Checking for missing values
if df.shape==df.notnull().shape:
    print (" No missing values")
else:
    print(" There are missing values")
#Backward Elimination to remove unimportant features
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
#The new feature matrix
new_x = pd.DataFrame(df[selected_features_BE],columns=selected_features_BE)
new_x.head()
#Check for multicollinearity
cor= new_x.corr()
sns.heatmap(cor)
print("Multicollinearity exists")
#Fitting a logistic model
X_train, X_test, y_train, y_test = train_test_split(new_x, y, test_size=0.3)
#Principal Component Analysis to combat multicollinearity
model_pca = PCA(n_components=5)
new_train = model_pca.fit_transform(X_train)
new_test  = model_pca.fit_transform(X_test)
logreg=LogisticRegression()
logreg.fit(new_train,y_train)
#f1 score of logistic model
y_pred=logreg.predict(new_test)
print(classification_report(y_test,y_pred))
#Implementing model on test data
test=pd.read_csv("../input/bda-2019-ml-test/Test_Mask_Dataset.csv")
model_pca = PCA(n_components=5)
test= model_pca.fit_transform(test)
pred=logreg.predict(test)
#Sample imputation
sub=pd.read_csv("../input/bda-2019-ml-test/Sample Submission.csv")
sub['flag']=pred
sub
#Creation of submission file
sub.to_csv("Sample Submission5.csv",index=False)
data=pd.read_csv("/kaggle/input/bda-2019-ml-test/Train_Mask.csv")                   #reading the csv file for analysis
data.head()  
df.drop('timeindex',axis = 1,inplace =True)
data.info()         #getting information about the data
data.isnull().sum()      #cheching for missing values
data.describe()         #getting the summary of data
plt.figure(figsize = (20,10))
matrix = np.triu(data.corr())
sns.heatmap(data.corr(), annot=True, mask=matrix,fmt=".1g",vmin=-1, vmax=1, center= 0,cmap= 'coolwarm')   #getting corrleation matrix to understand the relationship between the variables
X=data[['currentBack', 'motorTempBack', 'positionBack',
       'refPositionBack', 'refVelocityBack', 'trackingDeviationBack',
       'velocityBack', 'currentFront', 'motorTempFront', 'positionFront',                      #selecting the appropriate variable to split and train the data
       'refPositionFront', 'refVelocityFront', 'trackingDeviationFront',
       'velocityFront']]
X.head()
Y=data[['flag']]                #taking flag as a dependent variable according to the problem statement 
Y.head()
x_train,x_test,y_train,y_test=ms.train_test_split(X,Y,test_size=0.2)         #splitting the data to train and test for fitting models and to train the dataset for prediction.
x_train.shape,x_test.shape,y_train.shape,y_test.shape        
dt = DecisionTreeClassifier()              #running the decision tree model
dt.fit(x_train,y_train)                 #fits the decision tree model for train data
y_predict = dt.predict(x_test)                 #storing the predicted values to the variable y_predict
y_predict

cm=confusion_matrix(y_predict,y_test)                              #getting confusion matrix.
acs=accuracy_score(y_predict,y_test)                               #checking for accuracy
ps=precision_score(y_predict,y_test)                               #checking for precison score
rs=recall_score(y_predict,y_test)                                  #checking for recall score
print("Confusion Matrix :\n", cm)                                  #prints confusion matrix        
print("Accuracy :" , accuracy_score(y_test,y_predict))             #prints the accuracy score
print("Recall Score :",recall_score(y_test,y_predict))             #prints the recall score
print("Precision Score :",precision_score(y_test,y_predict))       #prints the precison score
f1_score(y_predict,y_test)                #getting f1 score to know the performance of the model to check how good the model is behaving.
print("The F1 score is:",f1_score(y_predict,y_test))
test_data=pd.read_csv("/kaggle/input/bda-2019-ml-test/Test_Mask_Dataset.csv")             #reading test data for predictions. 
test_data=test_data.drop(columns=['timeindex'])             #dropping the timeindex column since it's not much important for the analysis
test_data.head()                #displays first 5 rows of the test dataset
test_data['flag']=dt.predict(test_data)              #adding a column to store the predicted flag values
test_data.head()
Sample_Submission = pd.read_csv("/kaggle/input/bda-2019-ml-test/Sample Submission.csv")          #submitting the work to get the score
Sample_Submission['flag'] = test_data['flag']
Sample_Submission.to_csv("decision tree model.csv",index=False)