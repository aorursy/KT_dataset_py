import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import cross_validation
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

%matplotlib inline
df = pd.read_csv("../input/diabetes.csv")
df.head()
df.shape
df.describe()
df.info()
df.columns
Counter(df.Outcome)
sns.countplot(x='Outcome',data=df)
df_1 = df[df.Outcome == 1]
df_0 = df[df.Outcome == 0]
columns = df.columns[:-1]

plt.subplots(figsize=(16,10))
number_features = len(columns)
for i,j,  in zip(columns, range(number_features) ):
    plt.subplot(3,3,j+1)
    plt.subplots_adjust(wspace=0.5,hspace=0.5)
    df_0[i].hist(bins=20, color='b', edgecolor='black')
    df_1[i].hist(bins=20, color='r', edgecolor='black')
    plt.title(i)
# get features and labels
X = df.iloc[:,:-1]
labels= df.iloc[:,-1]
# Standarize features
X = StandardScaler().fit_transform(X)
# Divide Data into train and test set  (test set will only be used in section 3.)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=0, stratify=labels)
# reset y_train index
y_train= y_train.reset_index(drop=True)
# Random Forest

RF_model = RandomForestClassifier(n_estimators=600, random_state=123456, class_weight="balanced")

acc=[]
sen=[]
spe=[]
kf = KFold(n_splits=5, random_state= 123)
kf.get_n_splits(X_train)

for train_index, test_index in kf.split(X_train):
    Features_train, Features_test = X_train[train_index], X_train[test_index]
    Labels_train, Labels_test = y_train[train_index], y_train[test_index]

    RF_model.fit(Features_train, Labels_train)
    cm = confusion_matrix(Labels_test, RF_model.predict(Features_test))
    tn, fp, fn, tp = confusion_matrix(Labels_test, RF_model.predict(Features_test)).ravel()
    sensitivity = tp/(tp+fn)
    specificity  = tn/(tn+fp)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    acc.append(accuracy)
    sen.append(sensitivity)
    spe.append(specificity)
    print(accuracy, sensitivity, specificity)

global_acc = np.mean(acc)
acc_std = np.std(acc)
global_sen = np.mean(sen)
sen_std = np.std(sen)
global_spe = np.mean(spe)
spe_std = np.std(spe)

print("_________________________________")
print('Accuracy:', global_acc, "+/-", acc_std)
print('Sensitivity:', global_sen, "+/-", sen_std)
print('Specificity:', global_spe, "+/-", spe_std)
# Logistic Regression

C_param_range = [0.001,0.01,0.1,1,10,100]

for i in C_param_range:
    LR_model = LogisticRegression(random_state=0, C=i, class_weight='balanced')
    print("\n C= ", i)
    
    acc=[]
    sen=[]
    spe=[]
    kf = KFold(n_splits=5, random_state= 123)
    kf.get_n_splits(X_train)

    for train_index, test_index in kf.split(X_train):
        Features_train, Features_test = X_train[train_index], X_train[test_index]
        Labels_train, Labels_test = y_train[train_index], y_train[test_index]

        LR_model.fit(Features_train, Labels_train)
        cm = confusion_matrix(Labels_test, LR_model.predict(Features_test))
        tn, fp, fn, tp = confusion_matrix(Labels_test, LR_model.predict(Features_test)).ravel()
        sensitivity = tp/(tp+fn)
        specificity  = tn/(tn+fp)
        accuracy = (tp+tn)/(tp+fp+tn+fn)
        acc.append(accuracy)
        sen.append(sensitivity)
        spe.append(specificity)
        
        print(accuracy, sensitivity, specificity)
      

    global_acc = np.mean(acc)
    acc_std = np.std(acc)
    global_sen = np.mean(sen)
    sen_std = np.std(sen)
    global_spe = np.mean(spe)
    spe_std = np.std(spe)

    print("_________________________________")
    print('Accuracy:', global_acc, "+/-", acc_std)
    print('Sensitivity:', global_sen, "+/-", sen_std)
    print('Specificity:', global_spe, "+/-", spe_std, "\n")
# check how the RF model would perform on the test set
print(classification_report(y_test, RF_model.predict(X_test)))
# Check results of chosen model (Logistic Regression) for unseen data, i.e. data that was not used for model creation
cm = confusion_matrix(y_test, LR_model.predict(X_test))
tn, fp, fn, tp = confusion_matrix(y_test, LR_model.predict(X_test)).ravel()
sensitivity = tp/(tp+fn)
specificity  = tn/(tn+fp)
accuracy = (tp+tn)/(tp+fp+tn+fn)

print(" Sensitivity:", sensitivity, "\n Specificity", specificity, "\n Accuracy:", accuracy)
print(classification_report(y_test, LR_model.predict(X_test)))