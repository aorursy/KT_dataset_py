

import numpy as np 

import pandas as pd 

from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, roc_auc_score, roc_curve

import seaborn as sns

sns.set(style="whitegrid")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df = pd.read_excel("../input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx")



df.shape
df.dtypes
df.select_dtypes(include=['object']).head()
print(df.select_dtypes(include=['object']).isnull().sum())
df.select_dtypes(exclude=['object']).isnull().sum()
df.dropna().shape
df_cat = df.select_dtypes(include=['object'])

df_numeric = df.select_dtypes(exclude=['object'])



imp = SimpleImputer(missing_values=np.nan, strategy='mean')



idf = pd.DataFrame(imp.fit_transform(df_numeric))

idf.columns = df_numeric.columns

idf.index = df_numeric.index





idf.isnull().sum()

idf.drop(["PATIENT_VISIT_IDENTIFIER"],1)

idf = pd.concat([idf,df_cat ], axis=1)



cor = idf.corr()

cor_target = abs(cor["ICU"])

relevant_features = cor_target[cor_target>0.2]

relevant_features.index
data = pd.concat([idf[relevant_features.index],idf["WINDOW"]],1)



data.ICU.value_counts()
plt.figure(figsize=(10,7))

count = sns.countplot(x = "ICU",data=data)

count.set_xticklabels(["Not Admitted","Admitted"])

plt.xlabel("ICU Admission")

plt.ylabel("Patient Count")

plt.show()
plt.figure(figsize=(10,7))

age = sns.countplot(data.AGE_ABOVE65, hue='ICU', data=data)

age.set_xticklabels(["Under 65","65 and Above"])

plt.title("COVID-19 ICU Admissions by Age Range")

plt.xlabel("Patient Age")

plt.ylabel("Patient Count")

plt.xticks(rotation = 0)

plt.legend(title = "ICU Admission",labels=['Not Admitted', 'Admitted'])

plt.show()
plt.figure(figsize=(10,7))

window = sns.countplot(data.WINDOW, hue='ICU', data=data)

window.set_xticklabels(["0-2","2-4","4-6","6-12","12+"])

plt.xticks(rotation = 45)

plt.title("Patient Event Window and ICU Admission Counts")

plt.ylabel("Patient Count")

plt.xlabel("Window (hours)")

plt.legend(title = "ICU Admission",labels=['Not Admitted', 'Admitted'])
plt.figure(figsize=(15,7))

percentile = age = sns.countplot(sorted(idf.AGE_PERCENTIL), hue='ICU', data=idf)

plt.xticks(rotation=40)

plt.xlabel("Age Percentile")

plt.ylabel("Patient Count")

plt.title("COVID-19 ICU Admissions by Age Percentile")

plt.legend(title = "ICU Admission",labels=['Not Admitted', 'Admitted'], loc = 0)
data = pd.get_dummies(data)



data.AGE_ABOVE65 = data.AGE_ABOVE65.astype(int)

data.ICU = data.ICU.astype(int)



data.head()
y = data.ICU

X = data.drop("ICU", 1)



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42, shuffle = True)
LR = LogisticRegression(max_iter = 500)

LR.fit(X_train,y_train)



y_hat = LR.predict(X_test)
confusion_matrix = pd.crosstab(y_test, y_hat, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix, annot=True, fmt = 'g', cmap = 'Blues')



print(classification_report(y_test, y_hat))

print("AUC = ",roc_auc_score(y_test, y_hat))



yhat_probs = LR.predict_proba(X_test)

yhat_probs = yhat_probs[:, 1]

fpr, tpr, _ = roc_curve(y_test, yhat_probs)





plt.figure(figsize=(10,7))

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label = "Base")

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc="best")

plt.show()
data2 = pd.concat([idf[relevant_features.index],idf["WINDOW"]],1)

data2.AGE_ABOVE65 = data2.AGE_ABOVE65.astype(int)

data2.ICU = data2.ICU.astype(int)



X2 = data2.drop("ICU",1)

y2 = data2.ICU
from sklearn.preprocessing import LabelEncoder



label_encoder = LabelEncoder()



X2.WINDOW = label_encoder.fit_transform(np.array(X2["WINDOW"].astype(str)).reshape((-1,)))

X2.WINDOW
X2_train,X2_test,y2_train,y2_test = train_test_split(X2,y2,test_size=0.25,random_state=42, shuffle = True)



LR.fit(X2_train,y2_train)



y2_hat = LR.predict(X2_test)
confusion_matrix2 = pd.crosstab(y2_test, y2_hat, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix2, annot=True, fmt = 'g', cmap = 'Reds')



print("ORIGINAL")

print(classification_report(y_test, y_hat))

print("AUC = ",roc_auc_score(y_test, y_hat),'\n\n')

print("LABEL ENCODING")

print(classification_report(y2_test, y2_hat))

print("AUC = ",roc_auc_score(y2_test, y2_hat))





y2hat_probs = LR.predict_proba(X2_test)

y2hat_probs = y2hat_probs[:, 1]



fpr2, tpr2, _ = roc_curve(y2_test, y2hat_probs)



plt.figure(figsize=(10,7))

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label="Base")

plt.plot(fpr2,tpr2,label="Label Encoded")

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc="best")

plt.show()
print("Original 1:0 ratio =",y2.value_counts()[1]/y2.value_counts()[0])

print("Training 1:0 ratio =",y2_train.value_counts()[1]/y2_train.value_counts()[0])

print("Testing 1:0 ratio =",y2_test.value_counts()[1]/y2_test.value_counts()[0])
X3_train,X3_test,y3_train,y3_test = train_test_split(X2,y2,test_size=0.25,random_state=42, stratify = y2, shuffle = True)



print("Original 1:0 ratio =",y2.value_counts()[1]/y2.value_counts()[0])

print("Training 1:0 ratio =",y3_train.value_counts()[1]/y3_train.value_counts()[0])

print("Testing 1:0 ratio =",y3_test.value_counts()[1]/y3_test.value_counts()[0])
LR.fit(X3_train,y3_train)

y3_hat = LR.predict(X3_test)
confusion_matrix3 = pd.crosstab(y3_test, y3_hat, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix3, annot=True, fmt = 'g', cmap = 'Greens')



print("LABEL ENCODING")

print(classification_report(y2_test, y2_hat))

print("AUC = ",roc_auc_score(y2_test, y2_hat),'\n\n')



print("LABEL ENCODING + STRATIFY")

print(classification_report(y3_test, y3_hat))

print("AUC = ",roc_auc_score(y3_test, y3_hat))



y3hat_probs = LR.predict_proba(X3_test)

y3hat_probs = y3hat_probs[:, 1]



fpr3, tpr3, _ = roc_curve(y3_test, y3hat_probs)



plt.figure(figsize=(10,7))

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label="Base")

plt.plot(fpr2,tpr2,label="Label Encoded")

plt.plot(fpr3,tpr3,label="Stratify")

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc="best")

plt.show()
from imblearn.over_sampling import SMOTE



sm = SMOTE(random_state = 42)

X_train_res, y_train_res = sm.fit_sample(X3_train,y3_train.ravel())

LR.fit(X_train_res, y_train_res)

y_res_hat = LR.predict(X3_test)



confusion_matrix3 = pd.crosstab(y3_test, y_res_hat, rownames=['Actual'], colnames=['Predicted'])

sns.heatmap(confusion_matrix3, annot=True, fmt = 'g', cmap="YlOrBr")



print("LABEL ENCODING + STRATIFY")

print(classification_report(y3_test, y3_hat))

print("AUC = ",roc_auc_score(y3_test, y3_hat),'\n\n')



print("SMOTE")

print(classification_report(y3_test, y_res_hat))

print("AUC = ",roc_auc_score(y3_test, y_res_hat))



y_res_hat_probs = LR.predict_proba(X3_test)

y_res_hat_probs = y_res_hat_probs[:, 1]



fpr_res, tpr_res, _ = roc_curve(y3_test, y_res_hat_probs)



plt.figure(figsize=(10,10))

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr, label="Base")

plt.plot(fpr2,tpr2,label="Label Encoded")

plt.plot(fpr3,tpr3,label="Stratify")

plt.plot(fpr_res,tpr_res,label="SMOTE")

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc="best")

plt.show()
from sklearn.metrics import average_precision_score

from sklearn.metrics import precision_recall_curve, auc

from sklearn.metrics import plot_precision_recall_curve



precision, recall, _ = precision_recall_curve(y3_test, y3hat_probs)

precision_sm, recall_sm, _ = precision_recall_curve(y3_test, y_res_hat_probs)



plt.figure(figsize=(10,7))

plt.plot(recall, precision, label="w/out SMOTE")

plt.plot(recall_sm, precision_sm, label="SMOTE")

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.title("Precision-Recall Curves")

plt.legend(loc="best")



auc_score = auc(recall, precision)

auc_score_sm = auc(recall_sm, precision_sm)



print("P-R AUC:", auc_score)

print("P-R AUC (SMOTE):", auc_score_sm)
#y3hat_probs

#y_res_hat_probs

import matplotlib.patches as mpatches

y3 = np.asarray(y3_test)



predict_df = pd.DataFrame({'Probability of Admission':y3hat_probs, 'Admission':y3})

predict_df = predict_df.sort_values('Probability of Admission')



predict_df['Rank'] = np.arange(1,483)



plt.figure(figsize=(12,8))



classes = ['Not Admitted','Admitted']

scatter = plt.scatter(predict_df['Rank'],predict_df['Probability of Admission'],

                      c=predict_df['Admission'], 

                      cmap = 'seismic', 

                      marker = '^')

plt.title("COVID-19 ICU Admission Predictions\n(Without SMOTE)")

plt.xlabel("Index")

plt.ylabel("Predicted Probability of ICU Admission")

plt.legend(handles=scatter.legend_elements()[0], labels=classes,title="Actual ICU Admission", loc='best')

plt.show()





predict_df2= pd.DataFrame({'SMOTE_prob':y_res_hat_probs, 'Admission':y3})

predict_df2 = predict_df2.sort_values("SMOTE_prob")

predict_df2['Rank'] = np.arange(1,483)



plt.figure(figsize=(12,8))



scatter = plt.scatter(predict_df2['Rank'],predict_df2['SMOTE_prob'],

                      c=predict_df2['Admission'], 

                      cmap = 'seismic', 

                      marker = '^')

plt.title("COVID-19 ICU Admission Predictions\n(SMOTE)")

plt.xlabel("Index")

plt.ylabel("Predicted Probability of ICU Admission")

plt.legend(handles=scatter.legend_elements()[0], labels=classes,title="Actual ICU Admission", loc='best')

plt.show()






