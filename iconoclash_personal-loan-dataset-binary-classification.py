import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
original = pd.read_excel('../input/Bank_Personal_Loan_Modelling.xlsx',"Data")
feature=original.drop("Personal Loan",axis=1)
target=original["Personal Loan"]

loans = feature.join(target)
loans.head(5)
loans.tail(5)
listItem = []
for col in loans.columns :
    listItem.append([col,loans[col].dtype,
                     loans[col].isna().sum(),
                     round((loans[col].isna().sum()/len(loans[col])) * 100,2),
                    loans[col].nunique(),
                     list(loans[col].sample(5).drop_duplicates().values)]);

dfDesc = pd.DataFrame(columns=['dataFeatures', 'dataType', 'null', 'nullPct', 'unique', 'uniqueSample'],
                     data=listItem)
dfDesc
sns.heatmap(loans.isna(),yticklabels=False,cbar=False,cmap='viridis')
loans.describe().transpose()
outvis = loans.copy()
def fungsi(x):
    if x<0:
        return np.NaN
    else:
        return x
    
outvis["Experience"] = outvis["Experience"].apply(fungsi)
sns.heatmap(outvis.isnull(),yticklabels=False,cbar=False,cmap='viridis')
pd.DataFrame(loans.groupby("Education").mean()["Experience"])
pd.DataFrame(loans.groupby("Age").mean()["Experience"]).tail(8)
pltdf = pd.DataFrame(loans.groupby("Age").mean()["Experience"]).reset_index()
sns.lmplot(x='Age',y='Experience',data=pltdf)
plt.ylabel("Experience (Average)")
plt.title("Average of Experience by Age")
plt.show()
pd.DataFrame(loans[loans["Experience"]<0][["Age","Experience"]].sort_values("Age")).head()
pd.DataFrame(loans[loans["Experience"]<0][["Age","Experience"]].sort_values("Age"))["Age"].unique()
pd.DataFrame(loans[loans["Experience"]<0][["Age","Experience"]].sort_values("Age"))["Experience"].unique()
loans["Experience"] = loans["Experience"].apply(abs)
# def fungsi(x):
#     if x<0:
#         return np.NaN
#     else:
#         return x
    
# loans["Experience"] = loans["Experience"].apply(fungsi)

# loans.dropna(inplace=True)
# def fungsi(x):
#     if x== -1:
#         return 2
#     elif x== -2:
#         return 1
#     elif x== -3:
#         return 0
#     else:
#         return x
    
# loans["Experience"] = loans["Experience"].apply(fungsi)
loans.describe().transpose()
# loans.info()
loans[["Age","Experience","Income","CCAvg","Mortgage"]] = loans[["Age","Experience","Income","CCAvg","Mortgage"]].astype(float)
loans.info()
feature = loans.drop(["ID","Personal Loan"],axis=1)
target = loans["Personal Loan"]
# plt.figure(figsize=(10, 10))
# sns.heatmap(feature.corr(),annot=True,square=True)
#https://seaborn.pydata.org/generated/seaborn.heatmap.html

corr = feature.corr()

mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(10, 10))
sns.heatmap(corr, mask=mask,annot=True,square=True)
# plt.figure(figsize=(10, 10))
# sns.heatmap(feature.join(target).corr(),annot=True,square=True)
# plt.figure(figsize=(20, 20))
# sns.pairplot(feature.join(target).drop(["ZIP Code"],axis=1),hue="Personal Loan")
loans_corr = feature.join(target).corr()

mask = np.zeros((13,13))
mask[:12,:]=1

plt.figure(figsize=(10, 10))
with sns.axes_style("white"):
    sns.heatmap(loans_corr, annot=True,square=True,mask=mask)
sns.distplot(feature["Mortgage"])
plt.title("Mortgage Distribution with KDE")
SingleLog_y = np.log1p(feature["Mortgage"])              # Log transformation of the target variable
sns.distplot(SingleLog_y, color ="r")
plt.title("Mortgage Distribution with KDE First Transformation")
DoubleLog_y = np.log1p(SingleLog_y)
sns.distplot(DoubleLog_y, color ="g")
plt.title("Mortgage Distribution with KDE Second Transformation")
loans["Mortgage"] = DoubleLog_y
source_counts =pd.DataFrame(loans["Personal Loan"].value_counts()).reset_index()
source_counts.columns =["Labels","Personal Loan"]
source_counts
#https://matplotlib.org/gallery/pie_and_polar_charts/pie_features.html

fig1, ax1 = plt.subplots()
explode = (0, 0.15)
ax1.pie(source_counts["Personal Loan"], explode=explode, labels=source_counts["Labels"], autopct='%1.1f%%',
        shadow=True, startangle=70)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title("Personal Loan Percentage")
plt.show()
plt.figure(figsize=(10,6))
sns.distplot(loans[loans["Personal Loan"] == 0]['Income'], color = 'r',label='Personal Loan=0',kde=False)
sns.distplot(loans[loans["Personal Loan"] == 1]['Income'], color = 'b',label='Personal Loan=1',kde=False)
plt.legend()
plt.title("Income Distribution")
plt.figure(figsize=(10,6))
sns.distplot(loans[loans["Personal Loan"] == 0]['CCAvg'], color = 'r',label='Personal Loan=0',kde=False)
sns.distplot(loans[loans["Personal Loan"] == 1]['CCAvg'], color = 'b',label='Personal Loan=1',kde=False)
plt.legend()
plt.title("CCAvg Distribution")
plt.figure(figsize=(10,6))
sns.distplot(loans[loans["Personal Loan"] == 0]['Experience'], color = 'r',label='Personal Loan=0',kde=False)
sns.distplot(loans[loans["Personal Loan"] == 1]['Experience'], color = 'b',label='Personal Loan=1',kde=False)
plt.legend()
plt.title("Experience Distribution")
sns.countplot(x='Securities Account',data=loans,hue='Personal Loan')
plt.title("Securities Account Countplot")
sns.countplot(x='Family',data=loans,hue='Personal Loan')
plt.title("Family Countplot")
sns.boxplot(x='Education',data=loans,hue='Personal Loan',y='Income')
plt.legend(loc='lower right')
plt.title("Education and Income Boxplot")
sns.boxplot(x='Family',data=loans,hue='Personal Loan',y='Income')
plt.legend(loc='upper center')
plt.title("Family and Income Boxplot")
feature = loans.drop(["ID","Personal Loan"],axis=1)
target = loans["Personal Loan"]
feature = feature.drop(["ZIP Code","Age","Online","CreditCard"],axis=1)
# feature["Combination"] = (feature["Income"]/12)-feature["CCAvg"]
# feature["Combination"] = (feature["Income"]/12)/feature["CCAvg"]
# feature["Combination"] = (feature["Income"]/12)*feature["CCAvg"]
feature["Combination"] = (feature["Income"]/12)**feature["CCAvg"]
from sklearn.preprocessing import MinMaxScaler,StandardScaler,robust_scale
scaler = StandardScaler();



colscal=["Experience","Mortgage","Income","CCAvg","Combination"]

scaler.fit(feature[colscal])
scaled_features = pd.DataFrame(scaler.transform(feature[colscal]),columns=colscal)

feature =feature.drop(colscal,axis=1)
feature = scaled_features.join(feature)
from sklearn.model_selection import train_test_split,cross_val_score
X_train, X_test, y_train, y_test = train_test_split(feature,target,
                                                    test_size=0.30,
                                                    random_state=101)
y_train.value_counts()
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state=101)
xgb.fit(X_train, y_train)
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score, confusion_matrix, accuracy_score,recall_score
predict = xgb.predict(X_test)
predictProb = xgb.predict_proba(X_test)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print("confusion_matrix :\n",confusion_matrix(y_test, predict))
print("\nclassification_report :\n",classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict))
print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=101)

X_ros, y_ros = ros.fit_sample(X_train, y_train)
pd.Series(y_ros).value_counts()
xgb = XGBClassifier(n_estimators=97,random_state=101)
xgb.fit(X_ros, y_ros)
predict = xgb.predict(X_train.values)
predictProb = xgb.predict_proba(X_train.values)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print("confusion_matrix :\n",confusion_matrix(y_train, predict))
print("\nclassification_report :\n",classification_report(y_train, predict))
print('Recall Score',recall_score(y_train, predict))
print('ROC AUC :', roc_auc_score(y_train, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_train, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_train, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())
predict = xgb.predict(X_test.values)
predictProb = xgb.predict_proba(X_test.values)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print("confusion_matrix :\n",confusion_matrix(y_test, predict))
print("\nclassification_report :\n",classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict))
print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(estimator=xgb,
                                                       X=feature,
                                                       y=target,
                                                       train_sizes=np.linspace(0.01, 1.0, 10),
                                                       cv=10)

print(train_scores)
# Mean value of accuracy against training data
train_mean = np.mean(train_scores, axis=1)
print(train_mean)
print(train_sizes)
# Standard deviation of training accuracy per number of training samples
train_std = np.std(train_scores, axis=1)

# Same as above for test data
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# Plot training accuracies 
plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')
# Plot the variance of training accuracies
plt.fill_between(train_sizes,
                train_mean + train_std,
                train_mean - train_std,
                alpha=0.15, color='red')

# Plot for test data as training data
plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 
        label='Test Accuracy')
plt.fill_between(train_sizes,
                test_mean + test_std,
                test_mean - test_std,
                alpha=0.15, color='blue')

plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Train and Test Accuracy Comparison")
plt.show()
coef1 = pd.Series(xgb.feature_importances_,feature.columns).sort_values(ascending=False)

pd.DataFrame(coef1,columns=["Features"]).transpose().plot(kind="bar",title="Feature Importances") #for the legends

coef1.plot(kind="bar",title="Feature Importances")
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(min_samples_leaf=10)
dtree.fit(X_ros,y_ros)
predict = dtree.predict(X_test.values)
predictProb = dtree.predict_proba(X_test.values)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print("confusion_matrix :\n",confusion_matrix(y_test, predict))
print("\nclassification_report :\n",classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict))
print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=101,max_depth=250,max_leaf_nodes=50,random_state=101)
rfc.fit(X_ros,y_ros)
predict = rfc.predict(X_train.values)
predictProb = rfc.predict_proba(X_train.values)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print("confusion_matrix :\n",confusion_matrix(y_train, predict))
print("\nclassification_report :\n",classification_report(y_train, predict))
print('Recall Score',recall_score(y_train, predict))
print('ROC AUC :', roc_auc_score(y_train, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_train, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_train, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())
predict = rfc.predict(X_test.values)
predictProb = rfc.predict_proba(X_test.values)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print("confusion_matrix :\n",confusion_matrix(y_test, predict))
print("\nclassification_report :\n",classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict))
print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(estimator=rfc,
                                                       X=feature,
                                                       y=target,
                                                       train_sizes=np.linspace(0.01, 1.0, 10),
                                                       cv=10)

print(train_scores)
# Mean value of accuracy against training data
train_mean = np.mean(train_scores, axis=1)
print(train_mean)
print(train_sizes)
# Standard deviation of training accuracy per number of training samples
train_std = np.std(train_scores, axis=1)

# Same as above for test data
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
# Plot training accuracies 
plt.plot(train_sizes, train_mean, color='red', marker='o', label='Training Accuracy')
# Plot the variance of training accuracies
plt.fill_between(train_sizes,
                train_mean + train_std,
                train_mean - train_std,
                alpha=0.15, color='red')

# Plot for test data as training data
plt.plot(train_sizes, test_mean, color='blue', linestyle='--', marker='s', 
        label='Test Accuracy')
plt.fill_between(train_sizes,
                test_mean + test_std,
                test_mean - test_std,
                alpha=0.15, color='blue')

plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Train and Test Accuracy Comparison")
plt.show()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_ros,y_ros)
predict = knn.predict(X_test.values)
predictProb = knn.predict_proba(X_test.values)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict))
print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())
error_rate = []

# Will take some time
for i in range(1,100):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_ros,y_ros)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,100),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_ros,y_ros)
predict = logmodel.predict(X_test.values)
predictProb = logmodel.predict_proba(X_test.values)

score1 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="recall")
score2 =cross_val_score(X=X_train,y=y_train,estimator=xgb,scoring="roc_auc")

print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict))
print('ROC AUC :', roc_auc_score(y_test, predictProb[:,1]))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))
print("\nCross Validation Recall :",score1.mean())
print("Cross Validation Roc Auc :",score2.mean())
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
features=original.drop("Personal Loan",axis=1)
targets=original["Personal Loan"]

loans = features.join(targets)
loans.shape
import random


for i in range(1,7):
    copy = loans
    copy['Income']=copy['Income']+random.gauss(1,10) # add noice to income
    loans=loans.append(copy,ignore_index=True) # make voice df 2x as big
    print("shape of df after {0}th intertion of this loop is {1}".format(i,loans.shape))
loans["Experience"] = loans["Experience"].apply(abs)
feature = loans.drop(["ID","Personal Loan"],axis=1)
target = loans["Personal Loan"]

feature = feature.drop(["ZIP Code","Age","Online","CreditCard"],axis=1)

# feature["Combination"] = (feature["Income"]/12)-feature["CCAvg"]
# feature["Combination"] = (feature["Income"]/12)/feature["CCAvg"]
# feature["Combination"] = (feature["Income"]/12)*feature["CCAvg"]
feature["Combination"] = (feature["Income"]/12)**feature["CCAvg"]

from sklearn.preprocessing import MinMaxScaler,StandardScaler,robust_scale
scaler = StandardScaler();



colscal=["Experience","Mortgage","Income","CCAvg","Combination"]

scaler.fit(feature[colscal])
scaled_features = pd.DataFrame(scaler.transform(feature[colscal]),columns=colscal)

feature =feature.drop(colscal,axis=1)
feature = scaled_features.join(feature)

from sklearn.model_selection import train_test_split,cross_val_score
X_train, X_test, y_train, y_test = train_test_split(feature,target,
                                                    test_size=0.30,
                                                    random_state=101)

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=101)

X_ros, y_ros = ros.fit_sample(X_train, y_train)
clf = Sequential([
    Dense(units=72, kernel_initializer='uniform', input_dim=9, activation='relu'),
    Dense(72, kernel_initializer='uniform', activation='relu'),
    Dense(72, kernel_initializer='uniform', activation='relu'),
    Dropout(0.1),
    Dense(72, kernel_initializer='uniform', activation='relu'),
    Dropout(0.1),
    Dense(72, kernel_initializer='uniform', activation='relu'),
    Dense(1, kernel_initializer='uniform', activation='sigmoid')
])

clf.summary()
# from tensorflow.python.keras.callbacks import EarlyStopping
import keras.backend as K
def recall(y_true, y_pred): 

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1))) 
    recall = true_positives / (possible_positives + K.epsilon()) 
    return recall
clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=[recall,'accuracy'])
history= clf.fit(X_ros, y_ros, batch_size=50, epochs=25)
original = pd.read_excel('../input/Bank_Personal_Loan_Modelling.xlsx',"Data")

feature=original.drop("Personal Loan",axis=1)
target=original["Personal Loan"]

loans = feature.join(target)

loans["Experience"] = loans["Experience"].apply(abs)

feature = loans.drop(["ID","Personal Loan"],axis=1)
target = loans["Personal Loan"]

feature = feature.drop(["ZIP Code","Age","Online","CreditCard"],axis=1)

# feature["Combination"] = (feature["Income"]/12)-feature["CCAvg"]
# feature["Combination"] = (feature["Income"]/12)/feature["CCAvg"]
# feature["Combination"] = (feature["Income"]/12)*feature["CCAvg"]
feature["Combination"] = (feature["Income"]/12)**feature["CCAvg"]

from sklearn.preprocessing import MinMaxScaler,StandardScaler,robust_scale
scaler = StandardScaler();



colscal=["Experience","Mortgage","Income","CCAvg","Combination"]

scaler.fit(feature[colscal])
scaled_features = pd.DataFrame(scaler.transform(feature[colscal]),columns=colscal)

feature =feature.drop(colscal,axis=1)
feature = scaled_features.join(feature)

from sklearn.model_selection import train_test_split,cross_val_score
X_train, X_test, y_train, y_test = train_test_split(feature,target,
                                                    test_size=0.30,
                                                    random_state=101)
score = clf.evaluate(X_test, y_test, batch_size=128)
print('\nAnd the Test Score is ',"\nRecall :", score[1],"\nAccuracy :",score[2])
predictProb = pd.DataFrame(clf.predict(X_test.values))


def fungsi(x):
    if x<0.5:
        return 0
    else:
        return 1

predict = predictProb[0].apply(fungsi)
# score1 =cross_val_score(X=X_train,y=y_train,estimator=clf,scoring="recall")
# score2 =cross_val_score(X=X_train,y=y_train,estimator=clf,scoring="roc_auc")

print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))
print('Recall Score',recall_score(y_test, predict))
print('ROC AUC :', roc_auc_score(y_test, predictProb[0]))
print('Accuracy :',accuracy_score(y_test, predict))
print('Matthews Corr_coef :',matthews_corrcoef(y_test, predict))
history_dict = history.history
history_dict.keys()
#plot the metrics during training. 
epochs = range(1, len(history_dict['loss']) + 1)

plt.plot(epochs, history_dict['acc'], 'r',label='acc')
plt.plot(epochs, history_dict['recall'], 'b',label='recall')

plt.xlabel('Epochs')
plt.grid()
plt.legend()
plt.show()