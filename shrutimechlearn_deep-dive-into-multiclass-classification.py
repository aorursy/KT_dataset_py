import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
full_data = pd.read_csv('/kaggle/input/faults.csv')
print(full_data.shape)

print("Number of rows: "+str(full_data.shape[0]))

print("Number of columns: "+str(full_data.shape[1]))
full_data.head()
full_data.columns
full_data.describe().T
fig, ax=plt.subplots(1,2,figsize=(15,6))

_ = sns.countplot(x='target', data=full_data, ax=ax[0])

_ = full_data['target'].value_counts().plot.pie(autopct="%1.1f%%", ax=ax[1])
full_data.hist(figsize=(15,15))

plt.show()
full_data.plot(kind="density", layout=(6,5), 

             subplots=True,sharex=False, sharey=False, figsize=(15,15))

plt.show()
full_data.isnull().sum()
full_data.X_Maximum.fillna(full_data.X_Maximum.median(),inplace=True)

full_data.Steel_Plate_Thickness.fillna(full_data.Steel_Plate_Thickness.median(),inplace=True)

full_data.Empty_Index.fillna(np.mean(full_data.Empty_Index),inplace=True)
full_data.isnull().sum()
def draw_univariate_plot(dataset, rows, cols, plot_type):

    column_names=dataset.columns.values

    number_of_column=len(column_names)

    fig, axarr=plt.subplots(rows,cols, figsize=(30,35))



    counter=0

    

    for i in range(rows):

        for j in range(cols):



            if column_names[counter]=='target':

                break

            if 'violin' in plot_type:

                sns.violinplot(x='target', y=column_names[counter],data=dataset, ax=axarr[i][j])

            elif 'box'in plot_type :

                #sns.boxplot(x='target', y=column_names[counter],data=dataset, ax=axarr[i][j])

                sns.boxplot(x=None, y=column_names[counter],data=dataset, ax=axarr[i][j])



            counter += 1

            if counter==(number_of_column-1,):

                break
draw_univariate_plot(dataset=full_data, rows=7, cols=4,plot_type="box")
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

X=full_data.drop('target',axis=1)

Y=le.fit_transform(full_data['target'])
le.classes_
le.inverse_transform([0,1,2,3,4,5,6])
dict(zip(le.inverse_transform([0,1,2,3,4,5,6]),[0,1,2,3,4,5,6]))
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, Y, stratify=Y, test_size = 0.3,random_state = 42)
def draw_confusion_matrix(cm):

    plt.figure(figsize=(12,8))

    sns.heatmap(cm,annot=True,fmt="d", center=0, cmap='autumn') 

    plt.title("Confusion Matrix")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix



logreg = LogisticRegression(random_state=42)

logreg.fit(X_train, y_train)



y_predict_train_logreg = logreg.predict(X_train)

y_predict_test_logreg = logreg.predict(X_test)



train_accuracy_score_logreg = accuracy_score(y_train, y_predict_train_logreg)

test_accuracy_score_logreg = accuracy_score(y_test, y_predict_test_logreg)



print(train_accuracy_score_logreg)

print(test_accuracy_score_logreg)
cm_logreg = confusion_matrix(y_test,y_predict_test_logreg)

draw_confusion_matrix(cm_logreg)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix



rf = RandomForestClassifier(random_state=42, n_estimators=50, max_depth=6, criterion = 'entropy', 

                            min_samples_leaf= 1,min_samples_split= 2)

rf.fit(X_train, y_train)



y_predict_train_rf = rf.predict(X_train)

y_predict_test_rf = rf.predict(X_test)



train_accuracy_score_rf = accuracy_score(y_train, y_predict_train_rf)

test_accuracy_score_rf = accuracy_score(y_test, y_predict_test_rf)



print(train_accuracy_score_rf)

print(test_accuracy_score_rf)
cm_rf = confusion_matrix(y_test,y_predict_test_rf)

draw_confusion_matrix(cm_rf)