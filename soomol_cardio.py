import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
os.getcwd()
cardio = pd.read_csv("cardio.csv")
cardio.head()
# drop 'id' column 
cardio.drop('id',axis=1,inplace=True)
cardio.head()
cardio.describe()
cardio.dtypes
missing_data = cardio.isnull()
missing_data.head()
# Count missing values in each column
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 
#printing the correlations
correlations = cardio.corr()['cardio'].drop('cardio')
print(correlations*100)
cardio.cardio.value_counts()
diseased=(len(cardio[cardio.cardio==1])/len(cardio.cardio))*100
diseased_male=len(cardio[(cardio.cardio==1) & (cardio.gender==1)])/len(cardio.cardio)*100
diseased_female=len(cardio[(cardio.cardio==1) & (cardio.gender==2)])/len(cardio.cardio)*100

print("{:.2f}% of the total count were diseased, amoung which {:.2f}% were male and {:.2f}% were female".format(diseased,diseased_male,diseased_female))

non_diseased=(len(cardio[cardio.cardio==0])/len(cardio.cardio))*100
non_diseased_male=len(cardio[(cardio.cardio==0) & (cardio.gender==1)])/len(cardio.cardio)*100
non_diseased_female=len(cardio[(cardio.cardio==0) & (cardio.gender==2)])/len(cardio.cardio)*100
print("\n{:.2f}% of the total count were  not diseased, amoung which {:.2f}% were male and {:.2f}% were female".format(non_diseased,non_diseased_male,non_diseased_female))
col=['cholesterol','gluc', 'smoke', 'alco', 'active']
data_value=pd.melt(cardio,id_vars="cardio",value_vars=cardio[col])
sns.catplot(x="variable",hue="value",col="cardio",data=data_value,kind="count")
# Creating one input and output
column=["cardio"]
x=cardio.drop(column,axis = 1)
y=cardio["cardio"]
y=y.astype(int)
x.head()
# Scaler function
features_list = ["age", "height", "weight", "ap_hi", "ap_lo"]
def standartization(x):
    x_std = x.copy(deep=True)
    for column in features_list:
        x_std[column] = (x_std[column]-x_std[column].mean())/x_std[column].std()
    return x_std 
x_std=standartization(x)
x_std.head()
# Splitting the data into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.15,random_state=40)
# Data normalization
from sklearn.preprocessing import normalize
x_train = normalize(x_train)
x_test = normalize(x_test)
x = normalize(x)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(x_train,y_train)
print(tree.score(x_train, y_train))
print(tree.score(x_test, y_test))
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(x_train, y_train)
print(clf.score(x_train, y_train))
print(clf.score(x_test, y_test))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100)
rf.fit(x_train,y_train)

print(rf.score(x_train,y_train))
print(rf.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
print(classifier.score(x_train,y_train))
print(classifier.score(x_test,y_test))
# dropping less correlated columns and predicting results.
column=["alco","smoke"]
x_r=cardio.drop(column,axis = 1)
from sklearn.model_selection import train_test_split
x_r_train, x_r_test, y_train, y_test = train_test_split(x_r,y, random_state=12,stratify = y)
from sklearn.neighbors import KNeighborsClassifier
clf1 = KNeighborsClassifier(n_neighbors=5)
clf1.fit(x_r_train, y_train)
print(clf1.score(x_r_train,y_train))
print(clf1.score(x_r_test,y_test))
y_true = y_test
y_pred = clf1.predict(x_r_test)
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_true, y_pred)
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm1,fmt=".0f", annot=True,linewidths=0.2, linecolor="blue", ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
# F- Score Calculation
TN = cm1[0,0]
TP = cm1[1,1]
FN = cm1[1,0]
FP = cm1[0,1]
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
pd.DataFrame([[Precision, Recall, F1_Score]],columns=["Precision", "Recall", "F1 Score"], index=["Results"])

