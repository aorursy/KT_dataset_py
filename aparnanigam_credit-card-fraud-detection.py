# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from collections import Counter

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import warnings

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#load using pandas read_csv function and read the first 5 rows of the dataset using head()

df_credit_card = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')

df_credit_card.head()
df_credit_card_copy = df_credit_card.copy()
# get dimension and info

print('Dimension of our dataset rows:',df_credit_card_copy.shape[0],'columns :',df_credit_card_copy.shape[1])

print(df_credit_card_copy.info())
#check any missing values in dataset

df_credit_card_copy.isnull().values.any()
df_credit_card_copy.columns
df_credit_card_copy.describe()

#let's check the distribution of the target variable 'Class'

Counter(df_credit_card_copy['Class'])
count_classes = pd.value_counts(df_credit_card_copy['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")

LABELS = ["Normal", "Fraud"]

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency");
# RobustScaler is less prone to outliers.

from sklearn.preprocessing import StandardScaler, RobustScaler

rob_scaler = RobustScaler()

df_credit_card_copy['Amount_scaled'] = rob_scaler.fit_transform(df_credit_card_copy['Amount'].values.reshape(-1,1))

df_credit_card_copy['Time_scaled']  = rob_scaler.fit_transform(df_credit_card_copy['Time'].values.reshape(-1,1))

#Split the data into x and y variables

x = df_credit_card_copy.drop(['Class'], axis = 1)

y = df_credit_card_copy[['Class']]



x.columns
y.columns
#Split the data into train and test data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 62)
#1. Find the number of the minority class

number_fraud = len(df_credit_card_copy[df_credit_card_copy['Class']==1])

number_non_fraud = len(df_credit_card_copy[df_credit_card_copy['Class']==0])



print('number of frauds:',number_fraud)

print('number of non frauds:',number_non_fraud)


#2. Find the indices of the majority class

index_non_fraud = df_credit_card_copy[df_credit_card_copy['Class']==0].index



#.3 Find the indices of the minority class

index_fraud = df_credit_card_copy[df_credit_card_copy['Class']==1].index



#4. Randomly sample the majority indices with respect to the number of minority classes

random_indices = np.random.choice(index_non_fraud, number_fraud,replace='False')

len(random_indices)

#5. Concat the minority indices with the indices from step 4

under_sample_indices = np.concatenate([index_fraud,random_indices])

#Get the balanced dataframe - This is the final undersampled data

under_sample_df = df_credit_card_copy.iloc[under_sample_indices]

under_sample_df.shape
Counter(under_sample_df['Class'])
under_sample_class_counts = pd.value_counts(under_sample_df['Class'])

under_sample_class_counts.plot(kind='bar')
x_under = under_sample_df.drop(['Class'],axis = 1)

y_under = under_sample_df[['Class']]

cx_under_train,x_under_test,y_under_train,y_under_test = train_test_split(x_under,y_under)

#Run a Logistic Regression Classifer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, recall_score

lm = LogisticRegression()

lm.fit(x_under,y_under)

lm_predict = lm.predict(x_under_test)

lm_accuracy = accuracy_score(lm_predict, y_under_test)

lm_recall = recall_score(lm_predict, y_under_test)

print('Accuracy score :',lm_accuracy)

print('Recall score :',lm_recall)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(x_under, y_under)

rf_predict = rf.predict(x_under_test)

rf_accuracy = accuracy_score(rf_predict, y_under_test)

rf_recall = recall_score(rf_predict, y_under_test)

print('Accuracy score :',rf_accuracy)

print('Recall score :',rf_recall)





#Check accuracy and recall on the train data

rf_predict_x = rf.predict(x_under_train)

rf_predict_y = rf.predict(x_test)



print(accuracy_score(rf_predict_x,y_under_train))

print(recall_score(rf_predict_x,y_under_train))
#Check accuracy and recall on the test data

rf_predict_test = rf.predict(x_under_test)

print(accuracy_score(rf_predict_test,y_under_test))

print(recall_score(rf_predict_test,y_under_test))
# Let's implement simple classifiers

#import Classifier Liabraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier





classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

    "Support Vector Classifier": SVC(probability=True),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report, accuracy_score, make_scorer

#checking accuracy 





for keys, classifier in classifiers.items():

    training_score = cross_val_score(classifier,x_under,y_under,cv = 5)

    print(classifier.__class__.__name__,training_score)

    print("Classifiers : ",classifier.__class__.__name__, " Has a score of",round(training_score.mean(),2)*100,"% accuracy score")

  

from sklearn.metrics import classification_report,confusion_matrix

x_under = under_sample_df.drop(['Class'],axis = 1)

y_under = under_sample_df[['Class']]

x_under_train,x_under_test,y_under_train,y_under_test = train_test_split(x_under,y_under)



classifiers_selected = {

    "LogisiticRegression": LogisticRegression(),

    "DecisionTreeClassifier": DecisionTreeClassifier()

}





for keys, classifier in classifiers_selected.items():

    classifier.fit(x_under_train,y_under_train)

    predicted_test = classifier.predict(x_under_test)

    predicted_train = classifier.predict(x_under_train)

    report_test = classification_report(y_under_test,predicted_test)

    report_train = classification_report(y_under_train,predicted_train)

    print(classifier.__class__.__name__)

    print('Predicting testing data')

    print(report_test)

    print('Predicting training data')

    print(report_train)

