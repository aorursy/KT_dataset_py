import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')
penguin = pd.read_csv('../input/penguin-data-set/Penguins_data.csv')
penguin.head()
penguin.shape
penguin.info()
mn.matrix(penguin)
round(100*(penguin.isnull().sum()/(len(penguin))),2)
# We will predict the sex of the penguins therefore we will drop all the rows with null values in 'Sex' column
penguin.dropna(subset=['sex'],axis = 0,inplace = True)
# Checking the shape after dropping the rows
penguin.shape
penguin.info()
# All the missing values are gone
round(100*(penguin.isnull().sum()/(len(penguin))),)
penguin['species'].value_counts().plot.bar()
penguin['island'].value_counts().plot.bar()
# Creating dummy variables for species
species = pd.get_dummies(penguin['species'], drop_first=True)
species.head()
# Creating dummy variables for island
island = pd.get_dummies(penguin['island'], drop_first=True)
island.head()
penguin = pd.concat([penguin,species,island],axis = 1)
penguin.head()
# Dropping the 'species' and 'island' columns
penguin.drop(['species','island'],axis = 1,inplace = True)
penguin.head()
var = ['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_kg']
scaler = StandardScaler()
penguin[var] = scaler.fit_transform(penguin[var])
penguin.head()
penguin['sex'] = penguin['sex'].map({'male':1,'female':0})
penguin.head()
X = penguin.drop(['sex'],1)
y = penguin.sex
X.head()
y.head()
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.7,random_state = 100)
log_reg1 = LogisticRegression()
log_reg1.fit(X_train,y_train)
# lets make predictions
y_train = pd.DataFrame(y_train)
prediction = log_reg1.predict(X_train)
prediction
# Confusion_matrix
confusion_matrix(prediction,y_train)
# Precision_score
precision_score(prediction,y_train)
# Recall_score
recall_score(prediction,y_train)
# F1 score
f1_score(prediction,y_train)
# Accuracy score
accuracy_score(prediction,y_train)
# Using Statsmodel
X_train_lr = sm.add_constant(X_train)
log_reg2 = sm.GLM(y_train,X_train,family = sm.families.Binomial()).fit()
log_reg2.summary()
# Getting the predicted values on the train set
y_train_pred = log_reg2.predict(X_train)
y_train_pred[:10]
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]
def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None
fpr, tpr, thresholds = metrics.roc_curve(y_train['sex'], y_train_pred, drop_intermediate = False )
draw_roc(y_train['sex'], y_train_pred)
