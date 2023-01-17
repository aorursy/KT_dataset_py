import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
filepath = "/kaggle/input/early-stage-diabetes-risk-prediction-datasets"

def load_dataset(filepath):

    csv_path  = os.path.join(filepath,'diabetes_data_upload.csv')

    return pd.read_csv(csv_path)
df_dataset = load_dataset(filepath)
df_dataset.head()
df_dataset['Gender'].astype('category').value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x = 'Gender',data = df_dataset)

plt.show()
df_dataset['Polyuria'].astype('category').value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x = 'Polyuria',data = df_dataset)

plt.show()
df_dataset['Polydipsia'].astype('category').value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='Polydipsia',data=df_dataset)

plt.show()
df_dataset['sudden weight loss'].astype('category').value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='sudden weight loss',data=df_dataset)

plt.show()
df_dataset['weakness'].astype('category').value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x="weakness",data=df_dataset)

plt.show()
df_dataset['Polyphagia'].astype('category').value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x="Polyphagia",data=df_dataset)

plt.show()

df_dataset['Genital thrush'].astype('category').value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='Genital thrush',data=df_dataset)

plt.show()
df_dataset['visual blurring'].astype('category').value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='visual blurring',data=df_dataset)

plt.show()
df_dataset['Itching'].astype('category').value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='Itching',data=df_dataset)

plt.show()
df_dataset['Irritability'].astype('category').value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='Irritability',data=df_dataset)

plt.show()
df_dataset['delayed healing'].astype('category').value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='delayed healing',data=df_dataset)

plt.show()
df_dataset['partial paresis'].astype('category').value_counts()

f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='partial paresis',data=df_dataset )

plt.show()
df_dataset['muscle stiffness'].astype('category').value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='muscle stiffness',data=df_dataset)

plt.show()
df_dataset['Alopecia'].astype('category').value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='Alopecia',data=df_dataset)

plt.show()
df_dataset['Obesity'].astype('category').value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='Obesity',data=df_dataset)

plt.show()
df_dataset['class'].astype('category').value_counts()

f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='class',data=df_dataset)

plt.show()
sns.distplot(df_dataset['Age'])

plt.show()
df_dataset.groupby('Gender')['class'].value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='Gender',hue='class',data=df_dataset)

plt.show()
df_dataset.groupby('Polyuria')['class'].value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='Polyuria',hue='class',data=df_dataset)

plt.show()
df_dataset.groupby('Polydipsia')['class'].value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='Polydipsia',hue='class',data=df_dataset)

plt.show()
df_dataset.groupby('sudden weight loss')['class'].value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='sudden weight loss',hue='class',data=df_dataset)

plt.show()
df_dataset.groupby('weakness')['class'].value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='weakness',hue='class',data=df_dataset)

plt.show()
df_dataset.groupby('Polyphagia')['class'].value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='Polyphagia',hue='class',data = df_dataset)

plt.show()
df_dataset.groupby('Genital thrush')['class'].value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x = 'Genital thrush',hue='class',data=df_dataset)

plt.show()
df_dataset.groupby('visual blurring')['class'].value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='visual blurring',hue='class',data=df_dataset)

plt.show()
df_dataset.groupby('Itching')['class'].value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='Itching',hue='class',data=df_dataset)

plt.show()
df_dataset.groupby('Irritability')['class'].value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='Irritability',hue='class',data=df_dataset)

plt.show()
df_dataset.groupby('delayed healing')['class'].value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='delayed healing',hue='class',data=df_dataset)

plt.show()
df_dataset.groupby('partial paresis')['class'].value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x = 'partial paresis', hue='class',data=df_dataset)

plt.show()
df_dataset.groupby('muscle stiffness')['class'].value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='muscle stiffness',hue='class',data=df_dataset)

plt.show()
df_dataset.groupby('Alopecia')['class'].value_counts()
f,ax =plt.subplots(figsize = (8,6))

ax = sns.countplot(x='Alopecia',hue='class',data=df_dataset)

plt.show()
df_dataset.groupby('Obesity')['class'].value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x= 'Obesity',hue = 'class',data=df_dataset)

plt.show()


df_dataset['age_bins'] = pd.cut(x=df_dataset['Age'], bins=[20, 29, 39, 49,59,69,79,89,99])

df_dataset



df_dataset['age_bins'].unique()
df_dataset['age_by_decade'] = pd.cut(x=df_dataset['Age'], bins=[20,29, 39, 49,59,69,79,89,99], labels=['20s','30s', '40s', '50s','60s','70s','80s','90s'])
df_dataset
df_dataset.groupby('age_by_decade')['class'].value_counts()
f,ax = plt.subplots(figsize=(8,6))

ax = sns.countplot(x='age_by_decade',hue='class',data=df_dataset)

plt.show()
df_dataset['Gender'] = df_dataset['Gender'].map({'Male': 1, 'Female': 0})
# df_dataset['Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity']=

# df_dataset['Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity'].map({'Yes':1,'No':0})

def convert_toboolean(column_name):

    df_dataset[column_name] = df_dataset[column_name].map({'Yes':1,'No':0})

list_columnname = ['Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity']

for i in list_columnname:

    convert_toboolean(i)
df_dataset
df_datasetfinal = df_dataset
df_datasetfinal
df_datasetfinal = df_datasetfinal.drop('age_bins', 1)
df_datasetfinal = df_datasetfinal.drop('age_by_decade', 1)
df_datasetfinal
df_datasetfinal['class'] = df_datasetfinal['class'].map({'Positive': 1, 'Negative': 0})
df_datasetfinal
from sklearn.model_selection import train_test_split



# Putting feature variable to X

X = df_datasetfinal.drop(['class'],axis=1)



# Putting response variable to y

y = df_datasetfinal['class']



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7,test_size=0.3,random_state=100)



print("Number transactions X_train dataset: ", X_train.shape)

print("Number transactions y_train dataset: ", y_train.shape)

print("Number transactions X_test dataset: ", X_test.shape)

print("Number transactions y_test dataset: ", y_test.shape)
#Training the model on the train data

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



logistic_model = LogisticRegression(class_weight='balanced')

dataset_model = logistic_model.fit(X_train,y_train)


pred_probs_test = dataset_model.predict_proba(X_test)[:,1]

"{:2.2}".format(metrics.roc_auc_score(y_test, pred_probs_test))
#Making prediction on the test data

pred_test = dataset_model.predict_proba(X_test)

y_pred_default = dataset_model.predict(X_test)
# print(classification_report(y_test,y_pred_default))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test,y_pred_default))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred_default))
# Converting y_pred to a dataframe which is an array

y_pred_df = pd.DataFrame(pred_test)

# Converting to column dataframe

y_pred_1 = y_pred_df.iloc[:,[1]]



# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test.reset_index(drop=True, inplace=True)

# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test,y_pred_1],axis=1)



# Renaming the column 

y_pred_final.rename(columns={'class': 'class_new'}, inplace=True)

y_pred_final= y_pred_final.rename(columns={ 1 : 'class_prob'})
y_pred_final
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds =roc_curve(y_pred_final.class_new,y_pred_final.class_prob)

roc_auc = auc(fpr, tpr)

print('ROC_AUC Score: ',roc_auc)
#ROC Curve

def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(6, 6))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return fpr, tpr, thresholds
draw_roc(y_pred_final.class_new, y_pred_final.class_prob)