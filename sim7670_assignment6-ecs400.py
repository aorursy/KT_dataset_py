#Name: SUMAN SHAW
#SRN: 01FB16ECS400
#Assignment-6 Data Analytics
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
#importing the csv file
import os
print(os.listdir("../input")) #the csv file for absenteeism at work is present in the input directory

#viewing the dataset
data_file = "../input/Absenteeism_at_work.csv"
df= pd.read_csv(data_file)
df.head()
#removing the outliers
sns.boxplot(df['Absenteeism time in hours'])
median = np.median(df['Absenteeism time in hours'])
q75, q25 = np.percentile(df['Absenteeism time in hours'], [75 ,25])
iqr = q75 - q25
print("Lower outlier bound:",q25 - (1.5*iqr))
print("Upper outlier bound:",q75 + (1.5*iqr))
#setting the lower and upper bounds for outliers
df= df[df['Absenteeism time in hours']<=17]
df= df[df['Absenteeism time in hours']>=-7]
#Splitting data into training and testing
from sklearn.model_selection import train_test_split
y=df['Absenteeism time in hours']
X=df.drop('Absenteeism time in hours',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

#scaling the data
from sklearn import preprocessing
X_scaled_train = preprocessing.scale(X_train)
X_scaled_test = preprocessing.scale(X_test)
X_scaled_train.shape

#Support Vector Machine (SVM)
import pandas as pd
from sklearn.svm import SVC
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv("../input/Absenteeism_at_work.csv")
X = df.iloc[:, :-1].values  #labels and attributes separated here
y = df.iloc[:, 14].values
from sklearn.model_selection import train_test_split  #splitting the dataset for SVM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

#Scaling the data
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 
svclassifier = SVC(kernel='sigmoid')  
svclassifier.fit(X_train, y_train) 
y_pred = svclassifier.predict(X_test)  

from sklearn import metrics
print("Accuracy of SVM Model:",metrics.accuracy_score(y_test, y_pred)*100, "\n\n")
from sklearn.metrics import classification_report, confusion_matrix 
print("Confusion matrix:\n")
print(confusion_matrix(y_test, y_pred),"\n\nComputing the performance measures:")  
print(classification_report(y_test, y_pred)) 
#Decision Tree Classifier
import matplotlib.pyplot as plt  
%matplotlib inline

dataset = pd.read_csv("../input/Absenteeism_at_work.csv")  

X = dataset.drop('Absenteeism time in hours', axis=1)  
y = dataset['Absenteeism time in hours']  

#splitting for decision tree
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=42)  

from sklearn.tree import DecisionTreeClassifier  
classifier = DecisionTreeClassifier()  
classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test)  

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score 
print("Accuracy for Decision Tree:")
print(accuracy_score(y_test, y_pred)*100)
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
print("\n")
print("Computing the Performance measures: \n")
print(classification_report(y_test, y_pred))  

""""OBSERVATIONS: 1.According to the results,both SVM and Decision trees are quite close in terms of accuracy,with SVM being the slightly more accurate counterpart with 45.045% accuracy and decision tree having 43.243% accuracy,but as we know, accuracy is not the best measure for assessing classification models, so we cannot conclude anything on the basis of accuracy alone. 2.The overall precision of decision tree classifier is more than that of SVM with 0.44 and 0.34 respectively. Precision tells about the exactness of the model, which is more in the case of Decision trees.3.Also the F1-score and support values for Decision tree is higher than that of SVM.CONCLUSION:From this analysis, we can conclude that Decision Tree Classifier is a better fit for this data."""

