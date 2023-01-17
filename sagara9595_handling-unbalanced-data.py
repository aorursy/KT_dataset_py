import warnings 

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split



pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 500)
credit_df = pd.read_csv(r"../input/creditcardfraud/creditcard.csv")

credit_df.head()
counts = credit_df['Class'].value_counts()
counts.plot(kind='bar')

plt.title("Class count")

plt.xlabel("Class")

plt.ylabel("Value Count")

plt.show()
class_0_df = credit_df[credit_df["Class"] == 0]

class_1_df = credit_df[credit_df["Class"] == 1]
count_class_0, count_class_1 = credit_df["Class"].value_counts()
class_0_df_under_samp = class_0_df.sample(count_class_1)

credit_df_under = pd.concat([class_1_df, class_0_df_under_samp], axis=0)

counts = credit_df_under['Class'].value_counts()

counts.plot(kind='bar')

plt.title("Class count")

plt.xlabel("Class")

plt.ylabel("Value Count")

plt.show()
print("We lost {} % of data due to undersampling". format(100 - credit_df_under.shape[0]/credit_df.shape[0]*100))
class_1_df_over_sample = class_1_df.sample(count_class_0, replace=True)

credit_df_over = pd.concat([class_1_df_over_sample, class_0_df], axis=0)

counts = credit_df_over['Class'].value_counts()

counts.plot(kind='bar', )

plt.title("Class count")

plt.xlabel("Class")

plt.ylabel("Value Count")

plt.show()
fig=plt.figure(figsize=(12,6))



plt.subplot(121)

sns.scatterplot(x="V1",y="V2", hue="Class", data=credit_df)

plt.title('Before Sampling')

plt.subplot(122)

sns.scatterplot(x="V1",y="V2", hue="Class", data=credit_df_over)

plt.title('After Sampling')

plt.show()
feature_column = list(credit_df.columns)

feature_column = feature_column[:-1]

class_labels = credit_df.Class

features = credit_df[feature_column]
#train test split

feature_train, feature_test, class_train, class_test = train_test_split(features, class_labels, 

                                                                        test_size = 0.3, random_state=0)
print("Test value counts")

print(class_test.value_counts(),"\n")

print("Train value counts")

print(class_train.value_counts())
feature_train["class"] = class_train
SMOTE_Oversampler=SMOTE(random_state=0)

SOS_features,SOS_labels=SMOTE_Oversampler.fit_sample(feature_train,class_train)
SOS_features['class'] = SOS_labels
fig=plt.figure(figsize=(12,6))



plt.subplot(121)

sns.scatterplot(x="V1",y="V3", hue="class", data=feature_train)

plt.title('Before Sampling')

plt.subplot(122)

sns.scatterplot(x="V1",y="V3", hue="class", data=SOS_features)

plt.title('After Sampling')

plt.show()
from imblearn.over_sampling import ADASYN



Adasyn_Oversampler=ADASYN()

AOS_features,AOS_labels=Adasyn_Oversampler.fit_sample(feature_train,class_train)
AOS_features['class'] = AOS_labels
fig=plt.figure(figsize=(12,6))



plt.subplot(121)

sns.scatterplot(x="V1",y="V3", hue="class", data=feature_train)

plt.title('Before Sampling')

plt.subplot(122)

sns.scatterplot(x="V1",y="V3", hue="class", data=AOS_features)

plt.title('After Sampling')

plt.show()
fig=plt.figure(figsize=(12,6))



plt.subplot(121)

sns.scatterplot(x="V1",y="V3", hue="class", data=SOS_features)

plt.title('SMOTE Oversampling')

plt.subplot(122)

sns.scatterplot(x="V1",y="V3", hue="class", data=AOS_features)

plt.title('AdaSYN Oversampling')

plt.show()