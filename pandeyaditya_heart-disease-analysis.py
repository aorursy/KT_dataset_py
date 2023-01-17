import sklearn
import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
all_data = pd.read_csv("../input/heart-disease-uci/heart.csv")
all_data
data_pat= all_data[all_data["target"]==1]
from matplotlib import style
style.use("seaborn")
bins = np.arange(29,77,5)
plt.hist(data_pat["age"],bins,edgecolor="w")
plt.xticks(np.arange(20,90,5))
plt.title("Heart Patient analysis")
plt.xlabel("Age Group")
plt.ylabel("No of Cases")
plt.savefig("Heart Patient analysis AGE-wise.png",dpi=500)
plt.show()
gender_data = data_pat.groupby("sex").count()
gender_data
labels = 'Male', 'Female'
sizes = [72, 93]
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
plt.title("Heart Patient analysis")
plt.savefig("Heart Patient analysis GENDER-wise.png",dpi=500)
plt.show()
data_pat
style.use('fast')
bins = np.arange(200,600,25)
plt.hist(data_pat["chol"],edgecolor="w")
plt.xticks(np.arange(150,600,50))
plt.title("Heart Patient analysis")
plt.xlabel("Serum Cholestoral in mg/dl")
plt.ylabel("No of Cases")
plt.savefig("Heart Patient analysis Cholestoral-wise.png",dpi=500)
plt.show()
y= all_data.pop("target")
X = all_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2)
clf = LogisticRegression()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
y_pred = clf.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix
sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
print('Sensitivity : ', sensitivity )

specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
print('Specificity : ', specificity)
perm = PermutationImportance(clf, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())
