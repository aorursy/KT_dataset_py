import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix



#import the data into a dataframe

income_data=pd.read_csv('../input/income-classification/income_evaluation.csv')
# The target variable has 2 values (>50K and <50K), let us see how they are distributed.

sns.set(style="darkgrid")

sns.countplot(x=" income", data=income_data)

print ('Target Value counts are')

print('\t')

print(income_data[' income'].value_counts())
sns.pairplot(income_data)
plt.matshow(income_data.corr())
# making a copy of master data to do the activity

raw_data=income_data.copy()



# let us understand what are the datatypes of each column

print(raw_data.info())
cat_lst=[' workclass',' education',' marital-status', ' occupation',' relationship',' race',' sex',' native-country',' income']



for i in cat_lst:

    print ('unique values of column',i)

    print('\t')

    print (raw_data[i].unique())

    print('***********************************************')

    print('\t')
raw_data[[' workclass',' occupation',' native-country']]=raw_data[[' workclass',' occupation',' native-country']].replace(' ?',np.NaN)

raw_data = raw_data.dropna(how='any',axis=0)
# separating dependent and Independent Variables (Input and Output)

y=raw_data[' income']

x=raw_data.drop(' income',1)



#applying one hot encoding to the entire independent variables

raw_ohe = OneHotEncoder(categories='auto')

x= raw_ohe.fit_transform(x).toarray()



#applying label encoding to target variable.

le=LabelEncoder()

y=le.fit_transform(y)
#split x and y into Train and Test set

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,stratify=y)
sgdc_clf = linear_model.SGDClassifier(max_iter=10, tol=1e-3)

sgdc_clf.fit(x_train, y_train)



sgdc_clf_predict=sgdc_clf.predict(x_test)



#validating the model

print ('SGDC Accuracy',confusion_matrix(y_test,sgdc_clf_predict))

print ('SGDC Confusion Matrix',accuracy_score(y_test,sgdc_clf_predict))
print('SGDC F1 Score',f1_score(y_test,sgdc_clf_predict, average="macro"))

print('SGDC Precision Score',precision_score(y_test,sgdc_clf_predict, average="macro"))

print('SGDC Recall',recall_score(y_test,sgdc_clf_predict, average="macro")) 
NB_clf = MultinomialNB()

NB_clf.fit(x_train, y_train)



NB_clf_predict=NB_clf.predict(x_test)



#validating the model

print ('NB Accuracy',confusion_matrix(y_test,NB_clf_predict))

print ('NB Confusion Matrix',accuracy_score(y_test,NB_clf_predict))

print('NB F1 Score',f1_score(y_test,NB_clf_predict, average="macro"))

print('NB Precision Score',precision_score(y_test,NB_clf_predict, average="macro"))

print('NB Recall',recall_score(y_test,NB_clf_predict, average="macro")) 
impute_data=income_data.copy()



impute_data[[' workclass',' occupation',' native-country']]=impute_data[[' workclass',' occupation',' native-country']].replace(' ?',np.NaN)

impute_data[' workclass'] = impute_data[' workclass'].fillna(impute_data[' workclass'].mode()[0])

impute_data[' occupation'] = impute_data[' occupation'].fillna(impute_data[' occupation'].mode()[0])

impute_data[' native-country'] = impute_data[' native-country'].fillna(impute_data[' native-country'].mode()[0])



print(impute_data.info())
# seems like there are no more missing data.

# seperating dependent and Independent Variables (Input and OutPut)

impute_data_y=impute_data[' income']

impute_data_x=impute_data.drop(' income',1)



#applying one hot encoding to the entire independent variables

impute_data_ohe = OneHotEncoder(categories='auto')

impute_data_x= impute_data_ohe.fit_transform(impute_data_x).toarray()



#applying label encoding to target variable.

impute_data_le=LabelEncoder()

impute_data_y=impute_data_le.fit_transform(impute_data_y)
#split x and y into Train and Test set

impute_data_x_train,impute_data_x_test,impute_data_y_train,impute_data_y_test=train_test_split(impute_data_x,impute_data_y,test_size=0.25,stratify=impute_data_y)
impute_data_sgdc_clf = linear_model.SGDClassifier(max_iter=10, tol=1e-3)

impute_data_sgdc_clf.fit(impute_data_x_train, impute_data_y_train)



impute_data_sgdc_clf_predict=impute_data_sgdc_clf.predict(impute_data_x_test)



#validating the model

print ('Imputed data SGDC Accuracy',confusion_matrix(impute_data_y_test,impute_data_sgdc_clf_predict))

print ('Imputed data SGDC Confusion Matrix',accuracy_score(impute_data_y_test,impute_data_sgdc_clf_predict))

print('Imputed data SGDC F1 Score',f1_score(impute_data_y_test,impute_data_sgdc_clf_predict, average="macro"))

print('Imputed data SGDC Precision Score',precision_score(impute_data_y_test,impute_data_sgdc_clf_predict, average="macro"))

print('Imputed data SGDC Recall',recall_score(impute_data_y_test,impute_data_sgdc_clf_predict, average="macro")) 
impute_data_NB_clf = MultinomialNB()

impute_data_NB_clf.fit(impute_data_x_train, impute_data_y_train)



impute_data_NB_clf_predict=impute_data_NB_clf.predict(impute_data_x_test)



#validating the model

print ('imputed data MNB Accuracy',confusion_matrix(impute_data_y_test,impute_data_NB_clf_predict))

print ('imputed data MNB Confusion Matrix',accuracy_score(impute_data_y_test,impute_data_NB_clf_predict))

print('Imputed data MNB F1 Score',f1_score(impute_data_y_test,impute_data_NB_clf_predict, average="macro"))

print('Imputed data MNB Precision Score',precision_score(impute_data_y_test,impute_data_NB_clf_predict, average="macro"))

print('Imputed data MNB Recall',recall_score(impute_data_y_test,impute_data_NB_clf_predict, average="macro")) 