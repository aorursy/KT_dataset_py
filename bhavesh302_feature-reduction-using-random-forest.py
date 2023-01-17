# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv(os.path.join(dirname, filename))
print("Head of dataset:",df.head())

print('\n')

print("Tail of dataset:",df.tail())
#------------Null_value_analysis-----------------------

df.isnull().sum()
#----------Count_plot_to_check_classes_are_imbalance_or_not-------------

sns.countplot(df['label'])
#-------------------Correlation_between_independent_variables--------------------

plt.figure(figsize=(20,10))

corr_df=df.corr()

sns.heatmap(corr_df,annot=True)

plt.show()

target=df['label']

encoder.fit(target)

target=encoder.transform(target)

df=df.drop('label',axis=1)
#------------Split_dataset_into_train_and_test_set----------------

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df,target,test_size=0.20,random_state=42)

print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#----------Build_naive_model_with-all_featutres-------------------

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(oob_score=True,random_state=41)

print("These are default model parameters:\n",model)

model.fit(X_train,y_train)
#------------Check_model_performance------------

pred=model.predict(X_test)

from sklearn.metrics import classification_report,accuracy_score

print("The accuracy_score is:",accuracy_score(y_test,pred))

#-----------Calculate_classification_report----------------

print("The classification report is:\n")

print(classification_report(y_test,pred))
#--------------Calculate_ROC_curve-----------

from sklearn.metrics import roc_curve,roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, pred)

plt.figure(figsize=(10,8))

plt.plot(fpr,tpr,linewidth=5)

plt.plot([0.0,0.2,0.4,0.6,0.8,1.0],[0.0,0.2,0.4,0.6,0.8,1.0],'--',color='r')

plt.title('ROC_curve')

plt.xlabel('FalsePositieRate')

plt.ylabel('TruePositiveRate')

plt.xticks
from sklearn.metrics import confusion_matrix

confusion_matrix_val=confusion_matrix(y_test,pred)

print(pd.DataFrame(confusion_matrix_val,columns=['Female','Male']))



# 0 - Female, 1 - Male
sns.heatmap(pd.DataFrame(confusion_matrix_val),annot=True)
#----------------Calculate_feature_importance----------------

feature_imp_df=pd.DataFrame(model.feature_importances_,columns=['Feature_importance'],index=X_train.columns)

feature_imp_df=feature_imp_df.reset_index()

feature_imp_df=feature_imp_df.rename(columns={'index':'Feature_names'})
#-------------Arrange_all_feature_according_to_decreasing_order_of_feature_importance----------------

feature_imp_dec_sort=feature_imp_df.sort_values(by='Feature_importance',ascending=False)

plt.figure(figsize=(20,5))

plt.xticks(rotation=45)

plt.bar('Feature_names','Feature_importance',data=feature_imp_dec_sort)
#--------------Calculate_cumulative_sum_of_all_features---------------------------

cum_sum_df=pd.DataFrame([])

cumsum_imp_features=list(np.cumsum(feature_imp_dec_sort['Feature_importance']))



cum_sum_df['Feature_names'] = list(feature_imp_dec_sort['Feature_names'])

cum_sum_df['Cum_feature_importance'] = cumsum_imp_features
plt.figure(figsize=(25,8))

plt.plot('Feature_names','Cum_feature_importance',data=cum_sum_df,linewidth=5)

plt.axhline(y=0.9,linestyle='--')
#----------Select_features_which_contribute_90%_of_feature_importance--------------

selected_features=list(cum_sum_df[cum_sum_df['Cum_feature_importance']<=0.90]['Feature_names'])
#----------Extract_selected_feature_data------------------

X_train_updated=X_train[selected_features]

X_test_updated=X_test[selected_features]
#----------Build_model_on_selected_features-----------------

updated_model=RandomForestClassifier(oob_score=True,random_state=41)

#----------Train_model_on_selected_features-----------------

updated_model.fit(X_train_updated,y_train)
#-------------Implement_updated_model-------------

updated_pred=updated_model.predict(X_test_updated)
#------------Check_model_performance_of_updated_model--------------

from sklearn.metrics import classification_report,accuracy_score

print("The accuracy_score is:",accuracy_score(y_test,updated_pred))
#-----------Calculate_classification_report----------------

print("The classification report is:\n")

print(classification_report(y_test,updated_pred))
#--------------Calculate_ROC_curve-----------

from sklearn.metrics import roc_curve,roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, updated_pred)

plt.figure(figsize=(10,8))

plt.plot(fpr,tpr,linewidth=5)

plt.plot([0.0,0.2,0.4,0.6,0.8,1.0],[0.0,0.2,0.4,0.6,0.8,1.0],'--',color='r')

plt.title('ROC_curve')

plt.xlabel('FalsePositieRate')

plt.ylabel('TruePositiveRate')

plt.xticks
from sklearn.metrics import confusion_matrix

confusion_matrix_val=confusion_matrix(y_test,updated_pred)

print(pd.DataFrame(confusion_matrix_val,columns=['Female','Male']))



# 0 - Female, 1 - Male
sns.heatmap(pd.DataFrame(confusion_matrix_val),annot=True)