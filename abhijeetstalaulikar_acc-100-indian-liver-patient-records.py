import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



from sklearn.svm import SVC
raw_data = pd.read_csv('../input/indian-liver-patient-records/indian_liver_patient.csv')

raw_data.head()
raw_data.isnull().sum()
raw_data['Dataset'].value_counts()
sns.lmplot(data=raw_data, x='Age', y='Albumin');

sns.lmplot(data=raw_data, x='Age', y='Total_Protiens');

sns.lmplot(data=raw_data, x='Age', y='Albumin_and_Globulin_Ratio');
sns.countplot(data=raw_data, x='Gender');
g = sns.FacetGrid(raw_data, col="Dataset", row="Gender", margin_titles=True)

g.map(plt.hist, "Age")

plt.subplots_adjust(top=0.9)

g.fig.suptitle('Disease by Gender and Age');
corr = raw_data.drop('Dataset',axis=1).corr()

plt.figure(figsize=(30, 30))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},

           cmap= 'coolwarm');
total_direct_bil = raw_data[["Total_Bilirubin", "Direct_Bilirubin"]]

sns.violinplot(data=total_direct_bil);
aspartate_alamine = raw_data[["Aspartate_Aminotransferase", "Alamine_Aminotransferase"]]

sns.violinplot(data=aspartate_alamine);
Total_Protiens_alb = raw_data[["Albumin", "Total_Protiens"]]

sns.violinplot(data=Total_Protiens_alb);
reduced_data = raw_data[["Age","Gender","Total_Bilirubin","Aspartate_Aminotransferase","Albumin", "Total_Protiens", "Albumin_and_Globulin_Ratio","Dataset"]]

reduced_data.head()
reduced_data[reduced_data['Albumin_and_Globulin_Ratio'].isnull()]
grouped = reduced_data.groupby(["Gender","Dataset"])

reduced_data['Albumin_and_Globulin_Ratio'] = grouped['Albumin_and_Globulin_Ratio'].transform(lambda x: x.fillna(x.mean()))
le = LabelEncoder()

reduced_data.Gender = le.fit_transform(reduced_data.Gender)

reduced_data.head()
x_train, x_test, y_train, y_test = train_test_split(reduced_data, reduced_data.Dataset, test_size=0.2)
scaler = MinMaxScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)

x_test = scaler.transform(x_test)
clf = SVC(gamma='auto')

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)



print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))