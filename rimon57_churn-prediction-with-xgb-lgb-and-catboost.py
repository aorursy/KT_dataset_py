# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

import io

import plotly.offline as py#visualization

py.init_notebook_mode(connected=True)#visualization

import plotly.graph_objs as go#visualization

import plotly.tools as tls#visualization

import plotly.figure_factory as ff#visualization





from sklearn import metrics

from sklearn.model_selection import GridSearchCV,KFold,train_test_split, StratifiedKFold

from sklearn.preprocessing import LabelEncoder

from category_encoders import CatBoostEncoder, TargetEncoder



from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

import lightgbm as lgb

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier, XGBRFClassifier

from catboost import CatBoostClassifier



from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve,classification_report

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
churn_data = pd.read_csv(r"/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print('''

        Details of the dataset

----------------------------------------



''')

print(churn_data.info(verbose=True, null_counts=True, memory_usage='deep'))
pd.set_option('display.max_colwidth',500)

pd.set_option('display.max_columns',100)

churn_data.head(10)
churn_data.describe(include='all')
label=churn_data['Churn'].value_counts().keys().tolist()

value=churn_data['Churn'].value_counts().tolist()



data = go.Pie(labels = label ,

               values = value ,

               marker = dict(colors =  [ 'lime','red'],

                             line = dict(color = "white",

                                         width =  2.5)

                            ),

               rotation = 90,

               hoverinfo = "label+value+text",

               hole = .5

              )

layout = go.Layout(dict(title = "Customer Churn Proportion",

                        plot_bgcolor  = "rgb(243,243,243)",

                        paper_bgcolor = "rgb(243,243,243)",

                       )

                  )

fig=go.Figure(data=data,layout=layout)

py.iplot(fig)
churn_data['Churn']=np.where(churn_data.Churn =='Yes',1,0)

churn_data.TotalCharges=churn_data.TotalCharges.replace(' ',np.nan)

churn_data.dropna(inplace=True)

churn_data.TotalCharges=churn_data.TotalCharges.astype(float)
#replace 'No internet service' to No for the following columns

replace_cols = [ 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

                'TechSupport','StreamingTV', 'StreamingMovies']

for i in replace_cols : 

    churn_data[i]  = churn_data[i].replace({'No internet service' : 'No'})
%matplotlib notebook

%matplotlib inline

def cor_heat(df):

    cor=df.corr()

    plt.figure(figsize=(20,7),dpi=100)

    sns.heatmap(data=cor,annot=True,square=True,linewidths=0.1,cmap='YlGnBu')

    plt.title("Pearson Co-relation for numerical features: Heat Map")

cor_heat(churn_data.filter(regex='Senior|tenure|Charges|Churn'))
def cor_categorical(col):

    return churn_data.groupby(col)['Churn'].value_counts(normalize=True).unstack()[1].sort_values(ascending=False)    
print('''

Categorical features correlation with predictor

-----------------------------------------------''')

print(cor_categorical('gender'))

print('-'*47)

print(cor_categorical('Partner'))

print('-'*47)

print(cor_categorical('Dependents'))

print('-'*47)

print(cor_categorical('PhoneService'))

print('-'*47)

print(cor_categorical('MultipleLines'))

print('-'*47)

print(cor_categorical('InternetService'))

print('-'*47)

print(cor_categorical('OnlineSecurity'))

print('-'*47)

print(cor_categorical('OnlineBackup'))

print('-'*47)

print(cor_categorical('DeviceProtection'))

print('-'*47)

print(cor_categorical('TechSupport'))

print('-'*47)

print(cor_categorical('StreamingTV'))

print('-'*47)

print(cor_categorical('StreamingMovies'))

print('-'*47)

print(cor_categorical('Contract'))

print('-'*47)

print(cor_categorical('PaperlessBilling'))

print('-'*47)

print(cor_categorical('PaymentMethod'))
def label_encoder(col):

    churn_data[col]=LabelEncoder().fit_transform(churn_data[col])



for cols in churn_data.columns.drop(['customerID','TotalCharges','tenure','MonthlyCharges','Churn']).tolist():

    label_encoder(cols)
X=churn_data.drop(['Churn'],axis=1).set_index('customerID')

y=churn_data[['Churn']]

X_train, X_test, y_train, y_test = train_test_split(X,y, 

                                                    test_size=0.2, 

                                                    random_state=0,

                                                    stratify=y)
def model_building(model):

    your_model=model

    your_model.fit(X_train,y_train)

    pred=your_model.predict(X_test)

    print("Accuracy of {0} : {1}".format(str(model)[:],accuracy_score(y_test,pred)))

    print("AUC :",roc_auc_score(y_test,pred))
model_building(LogisticRegression(solver='newton-cg'))
model_building(LogisticRegression(solver='liblinear'))
model_building(GaussianNB())
model_building(BernoulliNB())
model_building(SVC(kernel='rbf'))
model_building(SVC(kernel='linear'))
model_building(CatBoostClassifier(eval_metric='AUC'))
model_building(XGBClassifier())
model_building(SGDClassifier())
model_building(LGBMClassifier())