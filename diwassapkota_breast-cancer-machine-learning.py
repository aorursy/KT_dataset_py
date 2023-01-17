# this is unrelated to the class .. It just helps displaying all outputs in a cell instead of just last one
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'
# importing the breast cancer dataset in tensorflow from UCI.edu
import tensorflow as tf
dataset= tf.keras.utils.get_file("breast_cancer_data", "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data")
dataset
# Loading the files by creating the columns as mentioned in the data above.
import pandas as pd
column_name=['ID_number','Diagnosis','Radius', 'Texture', 'Perimeter', 'Area','Smoothness','Compactness','Concavity','Concave_points','Symmetry','Fractal_dimension','Radius_se', 'Texture_se', 'Perimeter_se', 'Area_se','Smoothness_se','Compactness_se','Concavity_se','Concave_points_se','Symmetry_se','Fractal_dimension_se', 'Radius_worst','Texture_worst', 'Perimeter_worst', 'Area_worst','Smoothness_worst','Compactness_worst','Concavity_worst','Concave_points_worst','Symmetry_worst','Fractal_dimension_worst']
df=pd.read_csv(dataset, names=column_name)
# Exploring some of the data
df.head()
df.describe()
# Calculating the number of Maliginant and Benign patients in the data
print("Number of Maliginant: ",len(df[df["Diagnosis"]=='M']))
print("Number of Benign: ",len(df[df["Diagnosis"]=='B']))
# Using seaborn to plot the data
import seaborn as sns
import matplotlib.pylab as plt
%matplotlib inline
df.hist(figsize=(20,20),color='navajowhite')
plt.show()
# As in the dataset we can see that diagnosis column consists of 'M'and 'B' but we need to convert them into numericals
df['Diagnosis']=df['Diagnosis'].astype('category').cat.codes
# Plot the Diagnosis data
sns.distplot(df["Diagnosis"],bins=20, kde=False, rug=True);
# dropping the unnecessary columns
data=df.drop(['ID_number'], axis = 1)
# Split datasets, one for training & test, and using scaler to scale the datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X = data.loc[:, data.columns != 'Diagnosis']
Y = data['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=60)
X_train_scale = scalar.fit_transform(X_train)
X_test_scale = scalar.transform(X_test)
# Using Logistic regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train_scale, y_train)
# Showing the training score. 
train_score = logreg.score(X_train_scale, y_train)
print('Training accuracy is ', train_score)
# Showing the testing score. 
test_score = logreg.score(X_test_scale, y_test)
print('Testing accuracy is ', test_score)
# Calculating the coefficient and intercept from above linear regression model
logreg.coef_
logreg.intercept_
#Calculating the probability
pred= logreg.predict(X_test_scale)
prob= logreg.predict_proba(X_test_scale)
# Looking at the data set prediction and actual label
rel=pd.DataFrame(prob)
rel["pred"]= pred
rel["actual_Label"]= y_test.to_list()
rel
# Calculating the recall, precision and F1 Score
from sklearn.metrics import classification_report, confusion_matrix
# Creating the confusion Matrix
print(confusion_matrix(y_test, pred))
# Creating the classification report
print(classification_report(y_test, pred))
#Plotting the RoC Curve
from sklearn import metrics
fpr,tpr,thre= metrics.roc_curve(y_test, prob[:,1])
plt.plot(fpr, tpr)

# Calculate the area under the curve
metrics.auc(fpr,tpr)
plt.plot([0,1], [0,1], 'k--')
plt.title('ROC curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')

# Higher the AUC, better the model is at predicting 0s as 0s and 1s as 1s.

cfm = confusion_matrix(y_test, pred)

true_negative = cfm[0][0]
false_positive = cfm[0][1]
false_negative = cfm[1][0]
true_positive = cfm[1][1]

print('Confusion Matrix: \n', cfm, '\n')

print('True Negative:', true_negative)
print('False Positive:', false_positive)
print('False Negative:', false_negative)
print('True Positive:', true_positive)
print('Correct Predictions', 
      round((true_negative + true_positive) / len(pred) * 100, 1), '%')

# Using normalization in the dataset
import tensorflow as tf
train_set_x=tf.keras.utils.normalize(X, axis=1)
train_set_x.shape
# # Using tensflow to build a model using various hidden layers
from tensorflow.keras import models, layers, regularizers
model = models.Sequential()
model.add(layers.Dense(200, activation='relu', input_dim=30)) # Since we have 30 features we will get input_dim=30
model.add(layers.Dense(120, activation = 'relu', kernel_initializer='uniform'))
model.add(layers.Dense(55, activation = 'relu', kernel_initializer='uniform'))
model.add(layers.Dense(1, kernel_initializer='uniform', activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer = tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
model.fit(train_set_x, Y, epochs=500,batch_size=50)# epochs is the iteration
scores = model.evaluate(train_set_x, Y)
print(model.metrics_names[1], scores[1]*100)
prob_keras = model.predict(train_set_x[:5])
pred_keras = model.predict_classes(train_set_x[:5])
print(prob_keras)
pred_keras
from sklearn.metrics import classification_report
pred = model.predict_classes(train_set_x)
print(confusion_matrix(Y, pred))
print(classification_report(Y, pred))
from sklearn import metrics
prob_keras = model.predict(train_set_x)
#Plotting the RoC Curve
fpr,tpr,thre= metrics.roc_curve(Y, prob_keras)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')

# Calculate the area under the curve
metrics.auc(fpr,tpr)
# Building the random forest model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train_scale, y_train)
# Calculate the Train Accuracy
train_score = rfc.score(X_train_scale, y_train)
print("Train Accuracy:",train_score)

# Calculate the Test Accuracy
test_score = rfc.score(X_test_scale, y_test)
print("Test Accuracy:",test_score)
# Predicting the model
rfc_pred = rfc.predict(X_test_scale)
rfc_pred
# Creating the classification report
print(confusion_matrix(y_test, rfc_pred))
print(classification_report(y_test, rfc_pred))
from sklearn import metrics
prob=rfc.predict_proba(X_test_scale)

# Plot the ROC Curve for test

fpr,tpr,thre=metrics.roc_curve(y_test, prob[:,1]) 
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.title('ROC curve')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')

# Area under curve
metrics.auc(fpr,tpr)
