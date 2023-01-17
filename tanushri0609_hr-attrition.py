# Import the files
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

# Read the file
d_frame = pd.read_csv('../input/hr-attrition/HR_comma_sep.csv', sep=',', header=0)
d_frame.head()
# Look for null values in the data set
d_frame.info()
sb.pairplot(d_frame, palette='husl')
plt.show()
# find the correlation
sb.heatmap(d_frame.corr(),annot=True)
plt.show()
# Removing 'Department' and 'Salary' from data set for making model
data_for_training=d_frame.drop(['Department','salary'],axis=1)
data_for_training.head()
# Preparing data set for Random forest classifier
y=data_for_training['left']
training_set=data_for_training.drop(['left'],axis=1)
training_set.head()
# Split the data set
X_train, X_test, y_train, y_test=train_test_split(training_set, y, test_size=0.30)

#Training the Random forest classifier
classifier=RandomForestClassifier(n_estimators=30)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
# Calculate the accuracy
result=confusion_matrix(y_test,y_pred)
print('Confusion Matrix:')
print(result)
result1=accuracy_score(y_test,y_pred)
print('Accuracy:',result1)
# Convert the tuple(test data set) into data frame 
test_data=pd.DataFrame(X_test)
test_data.head()
#Add the prediction to the dataset
test_data['left_pred']=y_pred
# These employees will stay in the company 
test_data[test_data['left_pred']==0]