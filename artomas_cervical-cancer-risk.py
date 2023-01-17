import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import math

import seaborn as sns
cancer_df = pd.read_csv('../input/cervical-cancer-risk-classification/kag_risk_factors_cervical_cancer.csv')

cancer_df.head()
cancer_df.info()
cancer_df = cancer_df.replace('?', np.NaN)
#missing data

total = cancer_df.isnull().sum().sort_values(ascending=False)

percent = (cancer_df.isnull().sum()/cancer_df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)

## dropping two columns of STDs as it does not give much information because of missing data



cancer_df.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis'],inplace=True,axis=1)

numerical_df = ['Age', 'Number of sexual partners', 'First sexual intercourse','Num of pregnancies', 'Smokes (years)',

                'Smokes (packs/year)','Hormonal Contraceptives (years)','IUD (years)','STDs (number)']

categorical_df = ['Smokes','Hormonal Contraceptives','IUD','STDs','STDs:condylomatosis','STDs:cervical condylomatosis',

                  'STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',

                  'STDs:pelvic inflammatory disease', 'STDs:genital herpes','STDs:molluscum contagiosum', 'STDs:AIDS', 

                  'STDs:HIV','STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis','Dx:Cancer', 'Dx:CIN', 

                  'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller','Citology', 'Biopsy']
cancer_df = cancer_df.replace('?', np.NaN)
### Filling the missing values of numeric data columns with mean of the column data.

for feature in numerical_df:

    print(feature,'',round(pd.to_numeric(cancer_df[feature]).mean(),1))

    feature_mean = round(pd.to_numeric(cancer_df[feature]).mean(),1)

    cancer_df[feature] = cancer_df[feature].fillna(feature_mean)
for feature in categorical_df:

    cancer_df[feature] = pd.to_numeric(cancer_df[feature]).fillna(1.0)
from sklearn.linear_model import LogisticRegression




def rmsle(y_predicted, y_real):

    return np.sqrt(np.mean(np.power(np.log1p(y_predicted)-np.log1p(y_real), 2)))

def procenterror(y_predicted, y_real):

     return np.round( np.mean(np.abs(y_predicted-y_real) )/ np.mean(y_real) *100 ,1)
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures

X = cancer_df.drop(['Dx:Cancer'],axis=1) 

Y=cancer_df['Dx:Cancer']

X=cancer_df.fillna(value=0)

scaler = MinMaxScaler()

scaler.fit(X)

X=scaler.transform(X)
regr=LogisticRegression().fit(X,Y)

    #print( name,'% errors', abs(regr.predict(X)+correct-Y).sum()/(Y.sum())*100)

print('Logistic','%error',procenterror(regr.predict(X),Y),'rmsle',rmsle(regr.predict(X),Y))

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,f1_score, precision_score, recall_score