import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

data.head()
print("The shape of the dataset:{}".format(data.shape))
print("The summary of the dataset:\t{}".format(data.info()))
print("The summary of the dataset:\t{}".format(data.describe().transpose))
print("The count of missing data :", data.isnull().sum())
data['Class'].value_counts()
print("The Fraud {} %".format(data['Class'].value_counts()[1]/len(data) *100))
sns.countplot('Class', data=data)
sns.scatterplot(x = 'Class', y = 'Amount', data=data)
sns.distplot(data['Amount'])
sns.distplot(data['Time'])
from sklearn.preprocessing import StandardScaler, RobustScaler



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



data['Scaled_Amount'] = std_scaler.fit_transform(data['Amount'].values.reshape(-1,1))

data['Scaled_Time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))



data.drop(['Time','Amount'], inplace=True, axis=1)

data.head(

)
data_fraud = data.loc[data['Class']==1]

data_nonfraud = data.loc[data['Class']== 0][:492]
equal_data = pd.concat([data_fraud, data_nonfraud])



#shuffle the data

new_data = equal_data.sample(frac=1, random_state=42)
new_data.head()
sns.countplot('Class', data=new_data)
plt.figure(figsize=(10,7))

corr = new_data.corr()

sns.heatmap(corr,cmap='coolwarm_r', annot_kws={'size':20})
X = new_data.drop('Class', axis=1)

y = new_data['Class']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Let's implement simple classifiers



# Classifier Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import collections





classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}
from sklearn.model_selection import cross_val_score



for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train, y_train, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
log_Reg = LogisticRegression()

log_Reg.fit(X_train, y_train)
log_Reg_Score = cross_val_score(log_Reg, X_train, y_train, cv=5)

print('Logistic Regression Cross Validation Score: ', round(log_Reg_Score.mean() * 100, 2).astype(str) + '%')
from sklearn.model_selection import cross_val_predict



log_Reg_Pred = cross_val_predict(log_Reg, X_train, y_train, cv=5,

                             method="decision_function")
from sklearn.metrics import precision_recall_curve



precision, recall, threshold = precision_recall_curve(y_train, log_Reg_Pred)
print("Precision Score: {:.2f}".format(np.mean(precision)))

print("Recall Score: {:.2f}".format(np.mean(recall)))
