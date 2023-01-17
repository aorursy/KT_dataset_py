# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import flask as flsk

from pylab import rcParams
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
df=pd.read_csv('../input/credit-card-customer-churn-prediction/Churn_Modelling.csv')
df.tail()
df.info()
df.describe()
df.skew()
plt.hist(df['Age'])
plt.xlabel("Age")
plt.title("Age Distribution")
plt.show()
plt.scatter(df['Age'], df['Balance'],edgecolors='Red')
plt.title("Age Vs Balance")
plt.show()
# Data to plot
sizes = df['Geography'].value_counts(sort = True)
labels=df['Geography']
colors = ["grey","purple","red"] 
rcParams['figure.figsize'] = 5,5

# Plot
plt.pie(sizes,colors=colors,autopct='%1.1f%%',shadow=True,startangle=270)
plt.title('Geographical Area - Churn in Dataset')
plt.legend(labels)
plt.show()
# Data to plot
sizes = df['Exited'].value_counts(sort = True)
labels=df['Exited']
colors = ["purple","red"] 
rcParams['figure.figsize'] = 5,5

# Plot
plt.pie(sizes,colors=colors,autopct='%1.1f%%',shadow=True,startangle=270)
plt.title('Exit Customers - Churn in Dataset')
plt.legend(labels)
plt.show()
plt.boxplot(df['Age'])
plt.xlabel("Tenure")
plt.show()
corr_data = pd.DataFrame(df)
plt.figure(figsize=(10,7))
sns.heatmap(corr_data.corr(),annot=True,linewidths=2)
dataset = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
dataset =  dataset.drop(['Geography', 'Gender'], axis=1)
Geography = pd.get_dummies(df.Geography).iloc[:,1:]
Gender = pd.get_dummies(df.Gender).iloc[:,1:]
dataset = pd.concat([dataset,Geography,Gender], axis=1)
X = dataset.drop(['Exited'], axis=1)
y = dataset['Exited']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,max_depth=10, random_state=100)
print(X_test)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
predictions
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print("Accuracy",accuracy_score(y_test, predictions))
result = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(result)
print("Classification Report\n",classification_report(y_test,predictions)) 
feat_importances = pd.Series(classifier.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()