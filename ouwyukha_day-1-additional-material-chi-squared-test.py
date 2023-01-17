import pandas as pd

import numpy as np



df = pd.read_csv('/kaggle/input/titanic/train.csv',header=0)

colname = ['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Survived']

df = df[colname]

df.head()
print(len(df))

df = df[df['Embarked'].notnull()]

print(len(df))
#Initialize ChiSquare Class

colX = 'Embarked' # Variable to test

colY = 'Survived' # Prediction class



X = df[colX].astype(str)

Y = df[colY].astype(str)
dfObserved = pd.crosstab(Y,X)

dfObserved
import scipy.stats as stats

from scipy.stats import chi2_contingency

from decimal import Decimal

chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
dfExpected = pd.DataFrame(expected, columns=dfObserved.columns, index=dfObserved.index)

dfExpected
print("Chi2 value: " + str(chi2))

print("p value: " + str(Decimal(p))[:13])

print("Degree of freedom value: " + str(dof))
alpha=0.05 #Choose 95% confidence level



if p<alpha:

    result="{0} is IMPORTANT for Prediction".format(colX)

else:

    result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

print(result)
class ChiSquare:

    def __init__(self, dataframe):

        self.df = dataframe

        self.p = None #P-Value

        self.chi2 = None #Chi Test Statistic

        self.dof = None

        

        self.dfObserved = None

        self.dfExpected = None

        

    def _print_chisquare_result(self, colX, alpha):

        result = ""

        if self.p<alpha:

            result="{0} is IMPORTANT for Prediction".format(colX)

        else:

            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)



        print(result)

        

    def TestIndependence(self,colX,colY,alpha=0.05):

        X = self.df[colX].astype(str)

        Y = self.df[colY].astype(str)

        

        self.dfObserved = pd.crosstab(Y,X) 

        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)

        self.p = p

        self.chi2 = chi2

        self.dof = dof 

        

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)

        

        self._print_chisquare_result(colX,alpha)
cT = ChiSquare(df)

#Feature Selection

testColumns = ['Embarked','Cabin','Pclass','Age','Name']

for var in testColumns:

    cT.TestIndependence(colX=var,colY='Survived')