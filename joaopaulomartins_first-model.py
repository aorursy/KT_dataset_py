# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from imblearn.pipeline import Pipeline as imbPipe

from imblearn.over_sampling import SMOTE, ADASYN



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/fasam-churn/train.csv')



data.head()
data = pd.read_csv('../input/fasam-churn/train.csv')



media_monthly = data[(data['SeniorCitizen'] == 0) & (data['StreamingTV'] == 'No') & (data['Churn']==1)]

media_DSL = data[data['InternetService'] == 'DSL'].tenure.mean()



print(media_monthly.tenure.count())

print(media_monthly.tenure.mean())
data = pd.read_csv('../input/fasam-churn/train.csv')



def prepare_dataset(data):

    ''' Prepara o dataset para treino'''

        

    

    data = data.drop(columns=['TotalCharges','OnlineBackup', 'TechSupport', 'gender', 'UserID'], axis = 1)



    

    

    data['Partner'] = data['Partner'].apply(lambda x: 1 if x == 'Yes' else 0)

    #data['gender'] = data['gender'].apply(lambda x: 1 if x=='Male' else 0)

    data['Dependents'] = data['Dependents'].apply(lambda x :1 if x == 'Yes' else 0)

    data['PhoneService'] = data['PhoneService'].apply(lambda x :1 if x == 'Yes' else 0)

    data['PaperlessBilling'] = data['PaperlessBilling'].apply(lambda x: 1 if x == 'Yes' else 0)

    data = pd.get_dummies(data, columns=['MultipleLines'])

    data = pd.get_dummies(data, columns=['InternetService'])

    data = pd.get_dummies(data, columns=['OnlineSecurity'])

    data = pd.get_dummies(data, columns=['PaymentMethod','Contract', 'DeviceProtection','StreamingTV', 'StreamingMovies'])

    

    data['Fiber_NoSec']= data.apply(lambda x: 1 if (x['OnlineSecurity_No'] == 1 and x['InternetService_Fiber optic'] == 1) else 0, axis=1)

    data['noInternet'] = data.apply(lambda x: 0 if (x['InternetService_No']==1 and x['OnlineSecurity_No internet service'] == 1 and  

                                                 x['DeviceProtection_No internet service']==1 and x['StreamingTV_No internet service']==1 and

                                                 x['StreamingMovies_No internet service']==1) else 1, axis=1)

    data['Tenure_TwoYear'] = data.apply(lambda x: 0 if (x['tenure']>=30 and x['Contract_Two year'] ==1) else 1, axis=1)

    data['Tenure_OneYear'] = data.apply(lambda x: 0 if (x['tenure']>=60 and x['Contract_One year'] ==1) else 1, axis=1)

    #data['DSL_MeanTenure'] = data.apply(lambda x: 0 if (x['InternetService_DSL']==1 and x['tenure'] >=33) else 1, axis=1)

    data['DSL_MeanTenure'] = data.apply(lambda x: 1 if (x['InternetService_Fiber optic']==1 and x['StreamingTV_No']==1 and x['tenure']<14) else 0, axis=1)

    data['DSL_MeanTenure0'] = data.apply(lambda x: 0 if (x['InternetService_Fiber optic']==1 and x['StreamingTV_Yes']==1 and x['tenure']>48) else 1, axis=1)

    data['Month_Sitizen'] = data.apply(lambda x: 1 if (x['Contract_Month-to-month'] == 1 and x['SeniorCitizen']==1 and x['tenure']>19) else 0, axis=1)

    data['Month_Sitizen1'] = data.apply(lambda x: 1 if (x['StreamingTV_No'] == 1 and x['SeniorCitizen']==0 and x['tenure']<12) else 0, axis=1)

#     data['Fiber_Tenure']= data.apply(lambda x: 1 if (x['Dependents'] == 0 and x['Contract_Month-to-month']==1 and x['InternetService_DSL'] == 1 

#                                                      and x['PaymentMethod_Electronic check'] == 1 and x['tenure']<9) else 0, axis=1)

    #data['blabla'] = data.apply(lambda x: 0 if (x['MonthlyCharges'] <= data.MonthlyCharges.mean() and x['tenure']>=33) else 1, axis=1)

    #data['blabla2'] = data.apply(lambda x: 1 if (x['MonthlyCharges'] >= data.MonthlyCharges.mean() and x['tenure']<33) else 0, axis=1)

    #data['NoPhone'] = data.apply(lambda x: 1 if (x['PhoneService'] == 0 or x['MultipleLines_No phone service'] == 1) else 0, axis=1)



    

    #data['Automatic'] = data.apply(lambda x: 1 if (x['PaymentMethod_Bank transfer (automatic)'] ==1 or x['PaymentMethod_Credit card (automatic)'] == 1) else 0,axis=1)

    #data['DSL_Automatic'] = data.apply(lambda x: 1 if (x['Automatic'] ==1 or x['InternetService_DSL'] == 1) else 0,axis=1)

    #data['Monthly'] = data.apply(lambda x: 0 if (x['OnlineSecurity_Yes'] == 1 and x['tenure'] >=30 ) else 1, axis=1)

    #data['ElectronicCheckMonthly'] = data.apply(lambda x: 1 if (x['PaymentMethod_Electronic check'] ==1 and x['Contract_Month-to-month'] == 1) else 0,axis=1)

    #data['Fiber_NoSec_Monthly'] = data.apply(lambda x: 1 if (x['Contract_Month-to-month'] == 1 and x['PaperlessBilling'] == 1) else 0, axis=1)

    #data['DSL_Tenure'] = data.apply(lambda x: 1 if (x['InternetService_DSL'] == 1 and x['tenure'] >= 33) else 0, axis=1)

    #data['DSL_Tenure'] = data.apply(lambda x: 1 if (x['InternetService_DSL'] == 1 and x['OnlineSecurity_Yes'] == 1) else 0, axis=1)

    

    

    data = data.drop(columns=['OnlineSecurity_No', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service',

                             'DeviceProtection_No internet service', 'StreamingTV_No internet service', 'StreamingMovies_No internet service', 

                              'Contract_Two year', 'MultipleLines_No phone service', 'Contract_One year', 'PhoneService', 

                              'MultipleLines_Yes', 'OnlineSecurity_No', 'Dependents', 'Partner'], axis = 1)

    

    print('O shape do dataset eh:', data.shape)

    

    return data



data = prepare_dataset(data)

data.head(8)



corr = data.corr()

sns.set(rc={'figure.figsize':(13.7,9.27)})



sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
target = data.Churn



data = data.drop(columns='Churn')



# new = data[['Fiber_NoSec', 'Contract_Month-to-month', 'tenure', 'MonthlyCharges', 'noInternet', 'PaperlessBilling', 

#             'Tenure_TwoYear', 'SeniorCitizen', 'Dependents']]



# new.head()
x_train, x_test, y_train, y_test = train_test_split(data, target,  test_size = 0.15, random_state = 42)

print(x_train.shape, x_test.shape)
from sklearn.ensemble import RandomForestClassifier



# smt = SMOTE(random_state=42)

# ads = ADASYN()



# rf = RandomForestClassifier()





# pipe = imbPipe([#('sc', scaler),

#                     #('balance', ads), 

#                     ('clf', rf)])



# param_grid = [{'clf__n_estimators': [30, 100, 500], 'clf__max_features': [ 14, 16, 20],

#                'clf__max_leaf_nodes': [8,16, 20],'clf__class_weight':[None, 'balanced'], 'clf__criterion': ['gini', 'entropy']}]

# #{'clf__bootstrap': [False], 'clf__n_estimators': [3, 10], 'clf__max_features': [2, 3, 4]},



# grid = GridSearchCV(pipe, cv=5, n_jobs=-1, param_grid=param_grid)



# grid.fit(x_train, y_train)



clf = LogisticRegression(solver='newton-cg')



clf.fit(x_train, y_train)



from sklearn.metrics import classification_report, accuracy_score



y_pred = clf.predict(x_test)



print(classification_report(y_test, y_pred))



print(accuracy_score(y_test, y_pred))
def feature_importance(model, feature_names):

    importances = model.feature_importances_

    df_features = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)

    return df_features[df_features.importance > 0]



def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    

    # Transform target

    le = preprocessing.LabelEncoder()

    le.fit(y_true)

    

    y_true = le.transform(y_true)

    y_pred = le.transform(y_pred)

    

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data

    classes = ('0','1')

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    #print(cm)

    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax
from sklearn.metrics import confusion_matrix

from sklearn import preprocessing



class_names   = np.unique(y_train)

plot_confusion_matrix(y_test, y_pred, class_names)
df_valid = pd.read_csv('../input/fasam-churn/test.csv')



df_valid.head()
df_valid2 = prepare_dataset(df_valid)
preds = clf.predict(df_valid2)

preds

df_final = df_valid[['UserID']]

df_final.head()

df_final['Churn'] = preds



df_final
df_final.to_csv('first_sub21.csv', index=False)