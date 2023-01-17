#Првенствено ги внесуваме сите пакети што ќе ни бидат потребни за анализа на овој датасет



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score



import os

import os

print(os.listdir("../input"))



# Последниве три реда се автоматски генерирани од Kaggle
# Ги читаме податоците

heart = pd.read_csv("../input/heart-disease-uci/heart.csv")
# Првите 5 реда од податоците

heart.head()
#Број на луѓе што ја имаат/немаат заболување

heart.target.value_counts()
# На график ги прикажуваме бројот на луѓето што немаат и имаат заболување

sns.countplot(x="target", data=heart)

plt.xlabel("Заболување (0 = нема, 1 = има)")

plt.ylabel('Број')

plt.show()
# Го наоѓаме процентот на пациенти што имаат и немаат заболување

nema_bolest = len(heart[heart.target == 0])

ima_bolest = len(heart[heart.target == 1])

broj_na_zaboleni = len(heart.target)

print("Процент на пациенти што немаат заболување: {:.2f}%".format((nema_bolest / broj_na_zaboleni)*100))

print("Процент на пациенти што имаат заболување: {:.2f}%".format((ima_bolest / broj_na_zaboleni)*100))
# Ги претставуваме машките и женските пациенти на график

sns.countplot(x='sex', data=heart)

plt.xlabel("Пол (0 = женски, 1= машки)")

plt.ylabel('Број')

plt.show()
# Го наоѓаме процентот на машки и женски пациенти

broj_zenski = len(heart[heart.sex == 0])

broj_maski = len(heart[heart.sex == 1])

broj_pacienti = len(heart.sex)

print("Процент на женски пациенти: {:.2f}%".format((broj_zenski / broj_pacienti)*100))

print("Процент на машки пациенти: {:.2f}%".format((broj_maski / broj_pacienti)*100))
pd.crosstab(heart.age,heart.target).plot(kind="bar",figsize=(20,6))

plt.title('Зачестеност на срцеви заболувања споредбено со возраста на пациентот')

plt.xlabel('Возраст')

plt.ylabel('Зачестеност')

plt.legend(["Нема заболување", "Има заболување "])

plt.show()
pd.crosstab(heart.sex,heart.target).plot(kind="bar",figsize=(20,6))

plt.title('Зачестеност на срцеви заболувања споредбено со полот на пациентот')

plt.xlabel('Пол (0 = женски, 1 = машки)')

plt.xticks(rotation=0)

plt.legend(["Нема заболување", "Има заболување"])

plt.ylabel('Зачестеност')

plt.show()
y = heart.target.values

x_data = heart.drop(['target'], axis = 1)
#Ги менуваме вредностите да бидат во опсегот од 0 до 1

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
# Логистичка регресија (accuracy и f1_score)

accuracies = {}

f1_scores = {}



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

lr = LogisticRegression()

lr.fit(x_train,y_train)

pred_lr = lr.predict(x_test)

acc = lr.score(x_test,y_test)*100

accuracies['Logistic Regression'] = acc

print("Прецизност на моделот со логистичка регресија {:.2f}%".format(acc))

f1_score_lr = f1_score(pred_lr, y_test)*100



f1_scores['Logistic Regression F1'] = f1_score_lr

print('F1 Score from Logistic Regression: {:.2f}%'.format(f1_score_lr))



lr_scores_cvs = cross_val_score(lr, x_data, y, cv=10, scoring='accuracy')

print('Scores from 10 Fold Cross Validation with Logistic Regression',lr_scores_cvs)



print(max(lr_scores_cvs))

print(min(lr_scores_cvs))
# Random Forest Класификација (accuracy, f1_score и feature importances)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state = 0)

rf.fit(x_train, y_train)

pred_rf = rf.predict(x_test)

acc = rf.score(x_test,y_test)*100

accuracies['Random Forest'] = acc

print("Прецизност на моделот со Random Forest : {:.2f}%".format(acc))

f1_score_rf = f1_score(pred_rf, y_test)*100

f1_scores['Random Forest F1'] = f1_score_rf

print('F1 Score from Logistic Regression: {:.2f}%'.format(f1_score_rf))



rf_scores_cvs = cross_val_score(rf, x_data, y, cv=10, scoring='accuracy')

print('Scores from 10 Fold Cross Validation with Random Forest',rf_scores_cvs)



print(max(rf_scores_cvs))

print(min(rf_scores_cvs))
#Feature importances

print('The feature importances using Random Forest Classifier are:',rf.feature_importances_)



#Секоја колона со соодветната важност

for i in range(0, len(heart.columns)-1):

    print(heart.columns[i], rf.feature_importances_[i]*100)
# Support Vector Machine Класификација (accuracy, и f1_score



from sklearn.svm import SVC

svm = SVC(random_state = 0)

svm.fit(x_train, y_train)

pred_svc = svm.predict(x_test)

acc = svm.score(x_test,y_test)*100

accuracies['SVM'] = acc

print("Прецизност на моделот со SVM Algorithm: {:.2f}%".format(acc))

f1_score_svc = f1_score(pred_svc, y_test)*100

f1_scores['SVC F1'] = f1_score_svc

print('F1 Score from Logistic Regression: {:.2f}%'.format(f1_score_svc))



svc_scores_cvs = cross_val_score(svm, x_data, y, cv=10, scoring='accuracy')

print('Scores from 10 Fold Cross Validation with SVM',svc_scores_cvs)



print(max(svc_scores_cvs))

print(min(svc_scores_cvs))
colors = ["purple", "green", "orange"]



sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,2))

plt.ylabel("Accuracy %")

plt.xlabel("Algorithms")

plt.ylim(80, 90)

sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)

plt.show()
colors = ["purple", "green", "orange"]



sns.set_style("whitegrid")

plt.figure(figsize=(16,5))

plt.yticks(np.arange(0,100,2))

plt.ylim(80, 90)

plt.ylabel("Accuracy %")

plt.xlabel("F1 Scores")

sns.barplot(x=list(f1_scores.keys()), y=list(f1_scores.values()), palette=colors)

plt.show()
# Исцртување на confusion матрица за сите три методи



from sklearn.metrics import confusion_matrix



y_lr = lr.predict(x_test)

y_svm = svm.predict(x_test)

y_rf = rf.predict(x_test)



cm_lr = confusion_matrix(y_test,y_lr)

cm_svm = confusion_matrix(y_test,y_svm)

cm_rf = confusion_matrix(y_test,y_rf)



plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrices",fontsize=24)



plt.subplot(1,3,1)

plt.title("Logistic Regression Confusion Matrix")

sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(1,3,2)

plt.title("Support Vector Machine Confusion Matrix")

sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.subplot(1,3,3)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})



plt.show()