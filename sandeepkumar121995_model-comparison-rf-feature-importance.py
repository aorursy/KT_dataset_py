import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

#classifier libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import f1_score,log_loss,roc_auc_score, roc_curve,recall_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc
df = pd.read_csv("../input/mushrooms.csv")
print(df.shape)
df.head()
# check for missing value
df.isnull().sum()
# check data types
df.dtypes
# check for distinct type of value in each columns
df.nunique()
# veil type has only one entry , we can delete this column
df = df.drop('veil-type',1)
# lets check the each entry in the column
for i in df.columns:
    print("distribution of column:",i)
    print(df[i].value_counts())
# stalk root has missing value,lets impute it with new category (n)
df.loc[df['stalk-root']=='?','stalk-root'] = 'n'
# encoding labels 
labelencoder=LabelEncoder()
for col in df.columns:
    df[col] = labelencoder.fit_transform(df[col])

df.head()
#train test split
X = df.drop('class',1)
y = df['class']

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# K -fold cross validation

n_folds = 5

def accuracy_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=0).get_n_splits(X.values)
    accuracy= cross_val_score(model, X.values, y.values, scoring="accuracy", cv = kf)
    return(accuracy)
logreg = LogisticRegression(C =1000, max_iter= 100000)
score = accuracy_cv(logreg)
print("1st Fold Score:",score[0])
print("2nd Fold Score:",score[1])
print("3rd Fold Score:",score[2])
print("4th Fold Score:",score[3])
print("5th Fold Score:",score[4])
print("\nLogistic regression score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
knn = KNeighborsClassifier(n_neighbors=1)
score = accuracy_cv(knn)
print("1st Fold Score:",score[0])
print("2nd Fold Score:",score[1])
print("3rd Fold Score:",score[2])
print("4th Fold Score:",score[3])
print("5th Fold Score:",score[4])
print("\nKNN score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
tree = DecisionTreeClassifier(criterion='entropy', max_depth= 12, random_state= 0)
score = accuracy_cv(tree)
print("1st Fold Score:",score[0])
print("2nd Fold Score:",score[1])
print("3rd Fold Score:",score[2])
print("4th Fold Score:",score[3])
print("5th Fold Score:",score[4])
print("\nDecision Tree score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
rf = RandomForestClassifier(n_estimators= 12, random_state=0, criterion='entropy')
score = accuracy_cv(rf)
print("1st Fold Score:",score[0])
print("2nd Fold Score:",score[1])
print("3rd Fold Score:",score[2])
print("4th Fold Score:",score[3])
print("5th Fold Score:",score[4])
print("\nDecision Tree score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
ada = AdaBoostClassifier(n_estimators= 50, learning_rate=1, random_state=0)
score = accuracy_cv(ada)
print("1st Fold Score:",score[0])
print("2nd Fold Score:",score[1])
print("3rd Fold Score:",score[2])
print("4th Fold Score:",score[3])
print("5th Fold Score:",score[4])
print("\nDecision Tree score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
xgb = XGBClassifier(max_depth=3,learning_rate=1,n_estimators=20, random_state=0)
score = accuracy_cv(xgb)
print("1st Fold Score:",score[0])
print("2nd Fold Score:",score[1])
print("3rd Fold Score:",score[2])
print("4th Fold Score:",score[3])
print("5th Fold Score:",score[4])
print("\nDecision Tree score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# Random Forest
rf.fit(X_train,y_train)
print(rf.score(X_test,y_test))
confusion_matrix(y_test,rf.predict(X_test))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, rf.predict(X_test))
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
## Feature importance 
a = list(rf.feature_importances_)
score = pd.DataFrame({'value':a,'columns':X.columns})
score = score.sort_values('value', ascending= False)
score
plt.figure(figsize= (25,12))
sns.barplot(score['columns'], score['value'])
plt.show()