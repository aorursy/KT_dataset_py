import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
%matplotlib inline
data = pd.read_csv(
    'https://raw.githubusercontent.com/saidRaiss/dataset/master/advertising.csv'
)
data.info()
data.head()
classes = data['Clicked on Ad']
ax = sns.countplot(x=classes, data=data)
# Click on Ad according to gender
plt.figure(figsize=(7, 5))
sns.countplot(x='Clicked on Ad', data=data, hue='Male', palette='coolwarm')
# Change datetime type from object to datetime64[ns]
data['Timestamp']=pd.to_datetime(data['Timestamp'])
# Now, let's create Hour, DayOfWeek, Month and Date columns from Timestamp
data['Hour']=data['Timestamp'].apply(lambda time : time.hour)
data['DayofWeek'] = data['Timestamp'].apply(lambda time : time.dayofweek)
data['Month'] = data['Timestamp'].apply(lambda time : time.month)
data['Date'] = data['Timestamp'].apply(lambda t : t.date())
# Ad clicked hourly distribution
plt.figure(figsize=(15,6))
sns.countplot(
    x='Hour', data=data[data['Clicked on Ad']==1],
    hue='Male', palette='rainbow',
    )
plt.title('Ad clicked hourly distribution')
# Ad clicked daily distribution
plt.figure(figsize=(12,5))
sns.countplot(x='DayofWeek',data=data[data['Clicked on Ad']==1],hue='Male',palette='rainbow')
plt.title('Ad clicked daily distribution')
# Ad clicked monthly distribution
plt.figure(figsize=(12,5))
sns.countplot(x='Month',data=data[data['Clicked on Ad']==1],hue='Male',palette='rainbow')
plt.title('Ad clicked monthly distribution')
data[data['Clicked on Ad']==1]['Date'].value_counts().head(5)
# Age wise distribution of Ad clicks
plt.figure(figsize=(10,6))
sns.swarmplot(x=data['Clicked on Ad'],y= data['Age'],data=data,palette='coolwarm')
plt.title('Age wise distribution of Ad clicks')
#Let's see Daily internet usage and daily time spent on site based on age
fig, axes = plt.subplots(figsize=(10, 6))
ax = sns.kdeplot(
    data['Daily Time Spent on Site'], data['Age'], cmap="Reds",
    shade=True, shade_lowest=False
    )
ax = sns.kdeplot(
    data['Daily Internet Usage'],data['Age'] ,cmap="Blues",
    shade=True, shade_lowest=False
    )
ax.set_xlabel('Time')
ax.text(20, 20, "Daily Time Spent on Site", size=16, color='r')
ax.text(200, 60, "Daily Internet Usage", size=16, color='b')
# Clicked on Ad distribution based on area distribution
plt.figure(figsize=(10,6))
sns.violinplot(x=data['Male'],y=data['Area Income'],data=data,palette='viridis',hue='Clicked on Ad')
plt.title('Clicked on Ad distribution based on area distribution')
# Convert a categorical variable to dummy variables
country = pd.get_dummies(data['Country'])
# Let's drop the columns not required for building the model
data.drop(
    ['Ad Topic Line','City','Country','Timestamp','Date']
    ,axis=1, inplace=True
    )
# Now let's join the dummy values
data = pd.concat([data,country],axis=1)
# Display final result
data.head()
y = data['Clicked on Ad'].values.reshape(-1, 1)
X = data.drop(['Clicked on Ad'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
    )
def plot_roc(roc_auc, false_positive_rate, true_positive_rate):
  plt.figure(figsize=(6, 6))
  plt.title('Receiver Operating Characteristics')
  plt.plot(false_positive_rate, true_positive_rate, color='red', label='AUC = {:.2f}'.format( roc_auc))
  plt.legend(loc = 'lower right')
  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.axis('tight')
  plt.ylabel('True Positive Rtae')
  plt.xlabel('False Positive Rtae')
# Create a model
log_reg = LogisticRegression()
# Training
log_reg.fit(X_train, y_train.ravel())
# Prediction
y_prob_log_reg = log_reg.predict_proba(X_test)[:, 1]
y_pred_log_reg = np.where(y_prob_log_reg > 0.5, 1, 0)
# Confusion matrix
confusion_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
confusion_matrix_log_reg
false_positive_rate_reg, true_positive_rate_reg, thresholds = roc_curve(
    y_test, y_prob_log_reg
    )
roc_auc_log_reg = auc(false_positive_rate_reg, true_positive_rate_reg)
roc_auc_log_reg
# Draw the ROC curve and get the area under the curve
plot_roc(roc_auc_log_reg, false_positive_rate_reg, true_positive_rate_reg)
# Create a model
lda = LinearDiscriminantAnalysis()
# Training
lda.fit(X_train, y_train.ravel())
# Prediction
y_prob_lda = lda.predict_proba(X_test)[:, 1]
y_pred_lda = np.where(y_prob_lda > 0.5, 1, 0)
# Confusion matrix
confusion_matrix_lda = confusion_matrix(y_test, y_pred_lda)
confusion_matrix_lda
false_positive_rate_lda, true_positive_rate_lda, thresholds = roc_curve(
    y_test, y_prob_lda
    )
roc_auc_lda = auc(false_positive_rate_lda, true_positive_rate_lda)
roc_auc_lda
# Draw the ROC curve and get the area under the curve
plot_roc(roc_auc_lda, false_positive_rate_lda, true_positive_rate_lda)
# Create a model
qda = QuadraticDiscriminantAnalysis()
# Training
qda.fit(X_train, y_train.ravel())
# Prediction
y_prob_qda = qda.predict_proba(X_test)[:, 1]
y_pred_qda = np.where(y_prob_qda > 0.5, 1, 0)
# Confusion matrix
confusion_matrix_qda = confusion_matrix(y_test, y_pred_qda)
false_positive_rate_qda, true_positive_rate_qda, thresholds = roc_curve(
    y_test, y_prob_qda
    )
roc_auc_qda = auc(false_positive_rate_qda, true_positive_rate_qda)
roc_auc_qda
# Draw the ROC curve and get the area under the curve
plot_roc(roc_auc_qda, false_positive_rate_qda, true_positive_rate_qda)
plt.figure(figsize=(6, 6))
plt.title('Receiver Operating Characteristics')
plt.plot(false_positive_rate_reg, true_positive_rate_reg, color='blue', label="Log reg AUC = {:.2f}".format(roc_auc_log_reg))
plt.plot(false_positive_rate_lda, true_positive_rate_lda, color='green', label='LDA AUC = {:.2f}'.format(roc_auc_lda))
plt.plot(false_positive_rate_qda, true_positive_rate_qda, color='red', label='QDA AUC = {:.2f}'.format(roc_auc_qda))
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rtae')
plt.xlabel('False Positive Rtae')