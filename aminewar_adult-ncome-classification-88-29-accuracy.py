import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv("../input/adult-census-income/adult.csv")

df.columns = df.columns.str.replace(" ","")
df.head()
df.info()
df.describe().T
df.isna().values.any()
for sutun_adi in df.columns:

    print("{} sütundaki benzersiz değerler".format(sutun_adi))

    print("{}".format(df[sutun_adi].unique()),"\n")
df["native.country"] = df["native.country"].apply(str.strip).replace("?",np.nan)

liste_1 =df["native.country"]



for i in range(0,len(liste_1)):

    if pd.isnull(liste_1[i]):

        liste_1[i] = liste_1[i-1]

        

df["native.country"].unique()        

                
df["occupation"] = df["occupation"].apply(str.strip).replace("?",np.nan)



liste_2 =df["occupation"]



for i in range(0,len(liste_2)):

    if pd.isnull(liste_2[i]):

        liste_2[i] = liste_2[i+1]

        

df["occupation"].unique() 
df["workclass"] = df["workclass"].apply(str.strip).replace("?",np.nan)

liste_3 =df["workclass"]



for i in range(0,len(liste_3)):

    if pd.isnull(liste_3[i]):

        liste_3[i] = liste_3[i+1]

        

df["workclass"].unique() 
plt.figure(figsize=(19,12))





num_feat = df.select_dtypes(include=['int64']).columns



for i in range(6):

    plt.subplot(2,3,i+1)

    plt.boxplot(df[num_feat[i]])

    plt.title(num_feat[i],color="g",fontsize=20)

    plt.yticks(fontsize=14)

    plt.xticks(fontsize=14)





plt.show()
from scipy.stats.mstats import winsorize

df["age"]           = winsorize(df["age"],(0,0.15))

df["fnlwgt"]        = winsorize(df["fnlwgt"],(0,0.15))

df["capital.gain"]  = winsorize(df["capital.gain"],(0,0.099))

df["capital.loss"]  = winsorize(df["capital.loss"],(0,0.099))

df["hours.per.week"]= winsorize(df["hours.per.week"],(0.12,0.18))
plt.rcParams['figure.figsize'] = (25,7)



baslik_font = {'family':'arial','color':'purple','weight':'bold','size':25}



col_list=['age',"fnlwgt",'capital.gain', 'capital.loss', 'hours.per.week']



for i in range(5):

    plt.subplot(1,5,i+1)

    plt.boxplot(df[col_list[i]])

    plt.title(col_list[i],fontdict=baslik_font)



plt.show()
con_var=['age', 'fnlwgt', 'education.num','hours.per.week']
plt.figure(figsize=(15,10))

plt.subplot(221)



i=0

for x in con_var:

    plt.subplot(2, 2, i+1)

    i += 1

    ax1=sns.kdeplot(df[df['income'] == '<=50K'][x], shade=True,label="income <=50K")

    sns.kdeplot(df[df['income'] == '>50K'][x], shade=False,label="income   >50K", ax=ax1)

    plt.title(x,fontsize=15)



plt.show()
plt.figure(figsize=(15,7))



deg=["race","sex"]



for i in range(2):

    plt.subplot(1,2,i+1)

    sns.countplot(x=deg[i],data=df,hue='income')

    plt.xlabel(deg[i],color="darkorange",fontsize=18)

    plt.ylabel("Count",color="darkorange",fontsize=18)

    plt.yticks(fontsize=13)

    plt.xticks(rotation=90,fontsize=13)



plt.show()
plt.figure(figsize=(15,7))



deg=["occupation","hours.per.week"]



for i in range(2):

    plt.subplot(1,2,i+1)

    sns.countplot(x=deg[i],data=df,hue="income")

    plt.xlabel(deg[i],color="darkorange",fontsize=20)

    plt.ylabel("Count",color="darkorange",fontsize=20)

    plt.yticks(fontsize=13)

    plt.xticks(rotation=90,fontsize=13)



plt.show()
plt.figure(figsize=(16,7))



deg=["education","education.num"]



for i in range(2):

    plt.subplot(1,2,i+1)

    sns.countplot(x=deg[i],data=df,hue="income")

    plt.xlabel(deg[i],color="darkorange",fontsize=20)

    plt.ylabel("Count",color="darkorange",fontsize=20)

    plt.yticks(fontsize=13)

    plt.xticks(rotation=90,fontsize=13)



plt.show()
plt.figure(figsize=(16,7))



deg = ["relationship","marital.status"]



for i in range(2):

    plt.subplot(1,2,i+1)

    sns.countplot(x=deg[i],data=df,hue="income")

    plt.xlabel(deg[i],color="darkorange",fontsize=18)

    plt.ylabel("Count",color="darkorange",fontsize=18)

    plt.xticks(rotation=90,fontsize=15)

    plt.yticks(fontsize=15)



plt.show()    
plt.figure(figsize=(13,10))

sns.countplot(x=df["native.country"],data=df)

plt.xlabel("native.country",color="purple",fontsize=20)

plt.ylabel("Count",color="purple",fontsize=20)

plt.xticks(rotation=90,fontsize=15)

plt.yticks(fontsize=15)

plt.show()
list=['age','education.num',"hours.per.week","fnlwgt"]
plt.figure(figsize=(12,7))

sns.heatmap(df[list].corr(),annot=True, fmt = ".2f", cmap = "YlGnBu")

plt.title("Correlation Matrix",color="darkblue",fontsize=20)

plt.show()
df["woman?"]  = df.sex.replace({"Female":1,"Male":0})

df["income_"] = df.income.replace({"<=50K":0,">50K":1})
df1 = pd.get_dummies(df['workclass'])

df2 = pd.get_dummies(df["education"])

df3 = pd.get_dummies(df["marital.status"])

df4 = pd.get_dummies(df["occupation"])

df5 = pd.get_dummies(df["relationship"])

df6 = pd.get_dummies(df["race"])

df7 = pd.get_dummies(df["native.country"])



df  = pd.concat([df,df1,df2,df3,df4,df5,df6,df7],axis=1)
df.head()
plt.figure(figsize=(7,5))

sns.countplot(df["income_"])

plt.xlabel("İncome Case",fontsize=15)

plt.ylabel("Count",fontsize=15)

print(">50K  rate : %{:.2f}".format(sum(df["income_"])/len(df["income_"])*100))

print("<=50K rate : %{:.2f}".format((len(df["income_"])-sum(df["income_"]))/len(df["income_"])*100))
y = df["income_"]

X = df[['age','hours.per.week',"fnlwgt",

       'woman?','Federal-gov', 'Local-gov',

       'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc',

       'State-gov', 'Without-pay', 'Federal-gov', 'Local-gov', 'Never-worked',

       'Private', 'Self-emp-inc','Self-emp-not-inc', 'State-gov', 'Without-pay', '10th', '11th',

       '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm',

       'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters',

       'Preschool', 'Prof-school', 'Some-college', 'Adm-clerical',

       'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing',

       'Handlers-cleaners', 'Machine-op-inspct', 'Other-service',

       'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales',

       'Tech-support', 'Transport-moving', 'Husband', 'Not-in-family',

       'Other-relative', 'Own-child', 'Unmarried', 'Wife',

       'Amer-Indian-Eskimo','Asian-Pac-Islander', 'Black', 'Other', 'White', 'Cambodia',

       'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador',

       'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala',

       'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India',

       'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico',

       'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines',

       'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan',

       'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam',

       'Yugoslavia']]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score



model_1_predict_model = LogisticRegression(penalty='l2')

model_1_predict_model.fit(X_train,y_train)



predict_train_1 = model_1_predict_model.predict(X_train)

predict_test_1  = model_1_predict_model.predict(X_test)
from sklearn.metrics import classification_report,precision_recall_fscore_support



print("Model's Accuracy values       :",model_1_predict_model.score(X_test,y_test))

print("Model's Train f1_score values :",f1_score(y_train,predict_train_1))

print("Model's Test  f1_score values :",f1_score(y_test,predict_test_1),"\n")



print(classification_report(y_test,predict_test_1),"\n")



metrics_1 =precision_recall_fscore_support(y_test,predict_test_1)



print("Precision:",metrics_1[0])

print("Recall   :",metrics_1[1])

print("F1 Skoru :",metrics_1[2])
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns
model_1_predict_test_proba = model_1_predict_model.predict_proba(X_test)[:,1]



fpr, tpr, thresholds  = roc_curve(y_test,model_1_predict_test_proba )



confusion_matrix_test = confusion_matrix(y_test,predict_test_1)



plt.figure(figsize=(15,7))

plt.subplot(1,2,1)

sns.heatmap(confusion_matrix_test,annot=True,cmap="YlGnBu")

plt.title("Confusion Matrix",fontsize=15)

plt.xlabel("Predicted",fontsize=14)

plt.ylabel("Actual",fontsize=14)





# Plot ROC curve

plt.subplot(1,2,2)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate',fontsize=14)

plt.ylabel('True Positive Rate',fontsize=14)

plt.title('ROC Curve',fontsize=15)

plt.show()

print("\n","\n",'AUC Değeri : ', roc_auc_score(y_test,model_1_predict_test_proba ))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score



model_2_predict_model = LogisticRegression(penalty='l1')

model_2_predict_model.fit(X_train,y_train)



predict_train_2 = model_2_predict_model.predict(X_train)

predict_test_2  = model_2_predict_model.predict(X_test)
from sklearn.metrics import classification_report,precision_recall_fscore_support



print("Model's Accuracy values       :",model_2_predict_model.score(X_test,y_test))

print("Model's Train f1_score values :",f1_score(y_train,predict_train_2))

print("Model's Test  f1_score values :",f1_score(y_test,predict_test_2),"\n")



print(classification_report(y_test,predict_test_2),"\n")



metrics_2 =precision_recall_fscore_support(y_test,predict_test_2)



print("Precision:",metrics_2[0])

print("Racall   :",metrics_2[1])

print("F1 Skoru :",metrics_2[2])
from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns
model_2_predict_test_proba = model_2_predict_model.predict_proba(X_test)[:,1]



fpr, tpr, thresholds  = roc_curve(y_test,model_2_predict_test_proba )



confusion_matrix_test_2 = confusion_matrix(y_test,predict_test_2)



plt.figure(figsize=(15,7))

plt.subplot(1,2,1)

sns.heatmap(confusion_matrix_test,annot=True,cmap="YlGnBu")

plt.title("Confusion Matrix",fontsize=15)

plt.xlabel("Predicted",fontsize=14)

plt.ylabel("Actual",fontsize=14)





# Plot ROC curve

plt.subplot(1,2,2)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()



print('AUC Values : ', roc_auc_score(y_test,model_2_predict_test_proba ))
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import f1_score
def make_model(X,y):

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=111,stratify=y)

    logistic_model = LogisticRegression()

    logistic_model.fit(X_train,y_train)

    

    predict_train = logistic_model.predict(X_train)

    predict_test = logistic_model.predict(X_test)

    confusion_matrix_train = confusion_matrix(y_train,predict_train)

    confusion_matrix_test  = confusion_matrix(y_test,predict_test)

    

    print("Model's Accuracy values       :",logistic_model.score(X_test,y_test))

    print("Model's Train f1_score values :",f1_score(y_train,predict_train))

    print("Model's Test  f1_score values :",f1_score(y_test,predict_test),"\n")

    print("TEST DATA SET")

    print(classification_report(y_test,predict_test))

    

    metrics =precision_recall_fscore_support(y_test,predict_test)



    print("Precision:",metrics[0])

    print("Racall   :",metrics[1])

    print("F1 Skoru :",metrics[2])

    

    return None

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sns
def draw_graphic(X,y):

    

    

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=111,stratify=y)

    logistic_model = LogisticRegression()

    logistic_model.fit(X_train,y_train)

    

    predict_train = logistic_model.predict(X_train)

    predict_test = logistic_model.predict(X_test)

    confusion_matrix_train = confusion_matrix(y_train,predict_train)

    confusion_matrix_test  = confusion_matrix(y_test,predict_test)

    

    logistic_model_predict_test_proba = logistic_model.predict_proba(X_test)[:,1]



    fpr, tpr, thresholds  = roc_curve(y_test,logistic_model_predict_test_proba )



    confusion_matrix_test = confusion_matrix(y_test,predict_test)



    plt.figure(figsize=(15,7))

    plt.subplot(1,2,1)

    sns.heatmap(confusion_matrix_test,annot=True,cmap="YlGnBu")

    plt.title("Confusion Matrix",fontsize=15)

    plt.xlabel("Predicted",fontsize=14)

    plt.ylabel("Actual",fontsize=14)



    plt.subplot(1,2,2)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.plot(fpr, tpr)

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.show()



    print('AUC Values : ', roc_auc_score(y_test,logistic_model_predict_test_proba))
from sklearn.utils import resample
positive = df[df.income_==1]

negative = df[df.income_==0]



positive_increase = resample(positive,

                              replace = True,

                              n_samples = len(negative),

                              random_state = 111)

increase_df = pd.concat([negative,positive_increase])

increase_df.income.value_counts()
X = increase_df[['age','fnlwgt','hours.per.week',

       'woman?','Federal-gov', 'Local-gov',

       'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc',

       'State-gov', 'Without-pay', 'Federal-gov', 'Local-gov', 'Never-worked',

       'Private', 'Self-emp-inc','Self-emp-not-inc', 'State-gov', 'Without-pay', '10th', '11th',

       '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm',

       'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters',

       'Preschool', 'Prof-school', 'Some-college', 'Adm-clerical',

       'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing',

       'Handlers-cleaners', 'Machine-op-inspct', 'Other-service',

       'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales',

       'Tech-support', 'Transport-moving', 'Husband', 'Not-in-family',

       'Other-relative', 'Own-child', 'Unmarried', 'Wife',

       'Amer-Indian-Eskimo','Asian-Pac-Islander', 'Black', 'Other', 'White', 'Cambodia',

       'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador',

       'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala',

       'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India',

       'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico',

       'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines',

       'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan',

       'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam',

       'Yugoslavia']]



y = increase_df["income_"]



make_model(X,y)
draw_graphic(X,y)
from sklearn.utils import resample
positive = df[df.income_==1]

negative = df[df.income_==0]



positive_decrease = resample(negative,

                              replace = True,

                              n_samples = len(positive),

                              random_state = 111)

decrease_df = pd.concat([positive,positive_decrease])

decrease_df.income.value_counts()
X = decrease_df[['age','fnlwgt', 'hours.per.week',

       'woman?','Federal-gov', 'Local-gov',

       'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc',

       'State-gov', 'Without-pay', 'Federal-gov', 'Local-gov', 'Never-worked',

       'Private', 'Self-emp-inc','Self-emp-not-inc', 'State-gov', 'Without-pay', '10th', '11th',

       '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm',

       'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters',

       'Preschool', 'Prof-school', 'Some-college', 'Adm-clerical',

       'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing',

       'Handlers-cleaners', 'Machine-op-inspct', 'Other-service',

       'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales',

       'Tech-support', 'Transport-moving', 'Husband', 'Not-in-family',

       'Other-relative', 'Own-child', 'Unmarried', 'Wife',

       'Amer-Indian-Eskimo','Asian-Pac-Islander', 'Black', 'Other', 'White', 'Cambodia',

       'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador',

       'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala',

       'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India',

       'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico',

       'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines',

       'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan',

       'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam',

       'Yugoslavia']]



y = decrease_df["income_"]

make_model(X,y)
draw_graphic(X,y)
from imblearn.over_sampling import SMOTE
y = df["income_"]

X = df[['age','fnlwgt','hours.per.week',

       'woman?','Federal-gov', 'Local-gov',

       'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc',

       'State-gov', 'Without-pay', 'Federal-gov', 'Local-gov', 'Never-worked',

       'Private', 'Self-emp-inc','Self-emp-not-inc', 'State-gov', 'Without-pay', '10th', '11th',

       '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm',

       'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters',

       'Preschool', 'Prof-school', 'Some-college', 'Adm-clerical',

       'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing',

       'Handlers-cleaners', 'Machine-op-inspct', 'Other-service',

       'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales',

       'Tech-support', 'Transport-moving', 'Husband', 'Not-in-family',

       'Other-relative', 'Own-child', 'Unmarried', 'Wife',

       'Amer-Indian-Eskimo','Asian-Pac-Islander', 'Black', 'Other', 'White', 'Cambodia',

       'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador',

       'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala',

       'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India',

       'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico',

       'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines',

       'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan',

       'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam',

       'Yugoslavia']]



sm = SMOTE(random_state=27,ratio = 1.0)

X_smote, y_smote = sm.fit_sample(X,y)
make_model(X_smote,y_smote)
draw_graphic(X_smote,y_smote)
from imblearn.over_sampling import ADASYN
y = df["income_"]

X = df[['age','fnlwgt','hours.per.week',

       'woman?','Federal-gov', 'Local-gov',

       'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc',

       'State-gov', 'Without-pay', 'Federal-gov', 'Local-gov', 'Never-worked',

       'Private', 'Self-emp-inc','Self-emp-not-inc', 'State-gov', 'Without-pay', '10th', '11th',

       '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm',

       'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters',

       'Preschool', 'Prof-school', 'Some-college', 'Adm-clerical',

       'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing',

       'Handlers-cleaners', 'Machine-op-inspct', 'Other-service',

       'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales',

       'Tech-support', 'Transport-moving', 'Husband', 'Not-in-family',

       'Other-relative', 'Own-child', 'Unmarried', 'Wife',

       'Amer-Indian-Eskimo','Asian-Pac-Islander', 'Black', 'Other', 'White', 'Cambodia',

       'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic', 'Ecuador',

       'El-Salvador', 'England', 'France', 'Germany', 'Greece', 'Guatemala',

       'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary', 'India',

       'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico',

       'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines',

       'Poland', 'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan',

       'Thailand', 'Trinadad&Tobago', 'United-States', 'Vietnam',

       'Yugoslavia']]



ad = ADASYN()

X_adasyn,y_adasyn = ad.fit_sample(X,y)
make_model(X_adasyn,y_adasyn)
draw_graphic(X_adasyn,y_adasyn)
result = pd.DataFrame(columns = ["Models","Train f1 Score","Test f1 Score"])

result["Models"]              = ["Model 1","Model 2","Oversampling","Undersampling","Model Smote","Model Adasyn"]

result["Train f1 Score"]      = [0.0,0.6369504535938589, 0.8191137970688263,0.8098272552783109,

                                 0.88746921182266,0.8692412954234453]

result["Test f1 Score"]       = [0.0,0.6122153957354536,0.8202595390276971,0.823673719717878,

                                 0.881661041219188,0.8633108039947545]

result["Accuracy values"]     = [0.7640104406571473,0.8352525717795178,0.8122977346278317,0.8167038571883966,

                                 0.8829894822006472,0.8629790676509252]

result["AUC Values"]          = [0.5008798249816425,0.8876067714489535,0.887471722122726,0.8921330123827734,

                                 0.9574314032372933,0.9437558082270674]
result
result = pd.DataFrame(columns = ["Models","Train f1 Score","Test f1 Score"])

result["Models"]              = ["Model 1","Model 2","Oversampling","Undersampling","Model Smote","Model Adasyn"]

result["Train f1 Score"]      = [0.0,0.6369504535938589, 0.8191137970688263,0.8098272552783109,

                                 0.88746921182266,0.8692412954234453]

result["Test f1 Score"]       = [0.0,0.6122153957354536,0.8202595390276971,0.823673719717878,

                                 0.881661041219188,0.8633108039947545]



plt.figure(figsize = (12,8))



n_groups = 6

index = np.arange(n_groups)

bar_width = 0.3

opacity = 0.7

 

rects1 = plt.bar(index,result["Train f1 Score"], bar_width,

alpha=opacity,

color='bisque',

label='Train F1 Score')

 

rects2 = plt.bar(index + bar_width,result["Test f1 Score"] , bar_width,

alpha=opacity,

color='navy',

label='Test F1 Score')

 

plt.xlabel('Models',color="navy",fontsize =17)

plt.ylabel('F1 Score Values',color="navy",fontsize =17)

plt.title('Train and Test F1 Score',color="navy",fontsize =18)

plt.xticks(index + bar_width/2, ("Model 1","Model 2","Oversampling","Undersampling","Model Smote","Model Adasyn"),

           rotation=90,fontsize=12)

plt.yticks(fontsize=12)

plt.legend(fontsize='large')



plt.tight_layout()

plt.show()