# import libraries 
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import seaborn as sns # Statistical data visualization
%matplotlib inline
#import dataset
df = pd.read_csv("/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv")
df.head()# it will show the 1st 5 rows 
df.shape # no of rows & columns
df.info() #will give the data type and no of null values
df.describe() #basic statistics about the data
#checking for any null values in the dataset
df.isnull().sum()
count = df.diagnosis.value_counts()
count
count.plot(kind='bar')
plt.title("no of malignants(1) and benigns(0) ")
plt.xlabel("Diagnosis")
plt.ylabel("count");
sns.pairplot(df, hue = 'diagnosis', vars = ['mean_radius', 'mean_texture', 'mean_area', 'mean_perimeter', 'mean_smoothness'] );
# Let's check the correlation between the variables 
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter
plt.figure(figsize=(20,10)) 
sns.heatmap(df.corr(), annot=True) ;
# Let's drop the target label coloumns
X = df.drop(['diagnosis'],axis=1)
X
y = df['diagnosis']
y
#now we will prepare the dataset for Machine learning by splitting the data into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
#we use logistic regression model to classify whether a tumor is malignant or benign based on the characteristics or features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred
acc = accuracy_score(y_test, y_pred)
print("Accuracy score using Logistic Regression:", acc*100)
#confusion matrix to check no of right predictions we got
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True);
print(classification_report(y_test, y_pred))