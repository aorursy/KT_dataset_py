# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt





from sklearn.model_selection import train_test_split

from catboost import CatBoostClassifier

from sklearn.model_selection import RandomizedSearchCV



from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import classification_report



import warnings

warnings.filterwarnings("ignore")
# Load data

data = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')

data.head(5)
# Variable target - Frequencies

print(data["Exited"].value_counts())
# Transform Data

del data['RowNumber'], data['CustomerId'], data['Surname']

gender_dummies = pd.get_dummies(data['Gender'])

country_dummies = pd.get_dummies(data['Geography'])

del data['Gender'], data['Geography']

data = pd.concat([data, gender_dummies, country_dummies], axis=1)

data.info()
# Create more bad cases



data_0 = data.loc[data['Exited'] == 0]

data_1 = data.loc[data['Exited'] == 1]

auxiliar=data_1



for i in range(3):

    auxiliar=pd.concat([auxiliar,data_1])



data_df1=pd.concat([data_0,auxiliar])



# Variable target - Frequencies

print(data_df1["Exited"].value_counts())
# Create more good cases



data_0 = data.loc[data['Exited'] == 0]

data_1 = data.loc[data['Exited'] == 1]

auxiliar=data_0



for i in range(2):

    auxiliar=pd.concat([auxiliar,data_0])



data_df0=pd.concat([data_1,auxiliar])



# Variable target - Frequencies

print(data_df0["Exited"].value_counts())
def model(data_df,data):

   

    # Group of variables

    Y = data_df['Exited']

    X = data_df.drop('Exited',axis=1)



    # Split sample

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)



    # Model

    clf = CatBoostClassifier(eval_metric='AUC',

                           random_seed = 1020,

                           bagging_temperature = 0.2,

                           od_type='Iter',

                           metric_period = 50,

                           od_wait=100,

                           learning_rate=0.02,

                           max_depth=16,

                           iterations= 300)



       

    clf.fit(x_train, y_train, verbose=True)



    preds = clf.predict(x_test)



        

    # Confusion Matrix

    cm = pd.crosstab(y_test, preds, rownames=['Real'], colnames=['Predicted'])

    fig, (ax1) = plt.subplots(ncols=1, figsize=(10,10))

    sns.heatmap(cm, 

            xticklabels=['Not Churn', 'Churn'],

            yticklabels=['Not Churn', 'Churn'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkred", cmap="Reds")

    plt.title('Confusion Matrix', fontsize=14)

    plt.show()

    

    # Metric

    print('Roc:',roc_auc_score(y_test, preds))

          

    # Classification Report

    print('\n',classification_report(y_test, preds)) 

    

    score = clf.predict_proba(data.drop(['Exited'],axis=1))

    

    return score
df1=model(data_df1,data)
df0=model(data_df0,data)
def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
import statsmodels.api as sm



Y= data['Exited']



Modelo0=pd.DataFrame(df0)[0]

Modelo1=pd.DataFrame(df1)[0]

X=pd.DataFrame(Modelo0)

X.columns=['Modelo0']

X['Modelo1']=Modelo1



# Split sample

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)



model=sm.Probit(y_train,x_train)

probit_res = model.fit()

print(probit_res.summary())

preds=probit_res.predict(x_test)



print("\nAUC Score: {0}".format(roc_auc_score(y_test, preds)))

fpr, tpr, thresholds = roc_curve(y_test, preds)

plot_roc_curve(fpr, tpr)
# Confusion Matrix

cm = pd.crosstab(y_test, np.where(preds>0.40,1,0), rownames=['Real'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(10,10))

sns.heatmap(cm, 

            xticklabels=['Not Churn', 'Churn'],

            yticklabels=['Not Churn', 'Churn'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Darkred", cmap="Reds")

plt.title('Confusion Matrix', fontsize=14)

plt.show()
print('\n',classification_report(y_test, np.where(preds>0.40,1,0)))