import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
ad_data = pd.read_csv("../input/advertising/advertising.csv")
ad_data.head()
ad_data.info()
ad_data.describe()
# sns.distplot(ad_data["Age"], kde=False, bins=30)
ad_data["Age"].plot.hist(bins=30)
sns.jointplot(x="Age", y="Area Income", data=ad_data)
sns.jointplot(x="Age", y="Daily Time Spent on Site", data=ad_data, kind="kde")
sns.jointplot(x="Daily Time Spent on Site", y="Daily Internet Usage", data=ad_data)
sns.pairplot(ad_data, hue="Clicked on Ad")
ad_data.head()
ad_data.columns
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.solver = "liblinear"
log.fit(X_train, y_train)
predictions = log.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))