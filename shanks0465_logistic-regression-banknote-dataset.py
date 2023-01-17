# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import pandas for Dataset Manupilation and matplotlib and seaborn for Visualization
import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
#Import functions for Model, Dataset Splitting and Evaluation
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
#Read the csv into a Dataframe
df=pd.read_csv('/kaggle/input/banknoteauthentication/data_banknote_authentication.csv') 
df.shape # To view the shape of our dataset (1371 rows and 5 columns)
df.columns=['VarianceWT','SkewnessWT','CurtosisWT','Image Entropy','Class'] # Set the column names for the Dataframe Object
df.head() # Returns first five rows in the Dataset
df.isnull().any() #To check if any column has null values. Column returns False if no null values
# Create X attributes and Y labels from dataframe object
X=df[['VarianceWT','SkewnessWT','CurtosisWT','Image Entropy']].values
y=df['Class'].values
corr=df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
# Exactly –1. A perfect downhill (negative) linear relationship

# –0.70. A strong downhill (negative) linear relationship

# –0.50. A moderate downhill (negative) relationship

# –0.25. A weak downhill (negative) linear relationship

# 0. No linear relationship


# +0.25. A weak uphill (positive) linear relationship

# +0.50. A moderate uphill (positive) relationship

# +0.70. A strong uphill (positive) linear relationship

# Exactly +1. A perfect uphill (positive) linear relationship
print(df['Class'].value_counts()) # Number of Authentic(0) and Fake(1) Bank Notes
sns.countplot(x="Class", data=df) # Count Plot of Class
sns.pairplot(data=df,kind='scatter',diag_kind='kde') #Shows relationships among all pairs of features
# Create the training and test sets using 0.2 as test size (i.e 75% of data for training rest 25% for model testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
logregressor = LogisticRegression() #Create the Logistic Regressor Object and fit the training set to the regressor object
logregressor.fit(X_train,y_train)
logregressor.get_params() #View the parameters of the Regressor (Here default parameters are used since none of them are intialized)
y_pred=logregressor.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
# Plot the Confusion Matrix as a HeatMap
class_names=[0,1] # Name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred),' - Classification Rate')
print("Precision:",metrics.precision_score(y_test, y_pred),' - Precision - What proportion of positive identifications was actually correct? ')
print("Recall:",metrics.recall_score(y_test, y_pred),' - Recall - What proportion of actual positives was identified correctly?')
#Receiver Operating Characteristic(ROC) curve is a plot of the true positive rate against the false positive rate.
y_pred_proba = logregressor.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.title('ROC Curve')
plt.show()
print('The AUC Score provides an aggregate measure of performance across all possible classification thresholds')
print('AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.')
print(metrics.classification_report(y, logregressor.predict(X))) # Displays a comprehensive Report of the Logistic Regression Model
