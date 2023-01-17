# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
loan = pd.read_csv("../input/madfhantr.csv")

loan_test= pd.read_csv("../input/madhante.csv")
loan.head(100)
#loan.info()

loan.shape
loan.dtypes
loan.isnull().sum()
loan.dropna(subset = ["Gender"], how = "all", inplace = True)



loan.dropna(subset = ["Married"], how = "all", inplace = True)



loan.dropna(subset = ["Self_Employed"], how = "all", inplace = True)



loan.dropna(subset = ["Dependents"], how = "all", inplace = True)



Avg_Amount=loan.LoanAmount.mean()

loan.LoanAmount=loan.LoanAmount.fillna(value=Avg_Amount)



Avg_CAmount=loan.Loan_Amount_Term.mean()

loan.Loan_Amount_Term=loan.Loan_Amount_Term.fillna(value=Avg_CAmount)



Avg_LoanAmount=loan.Credit_History.mean()

loan.Credit_History=loan.Credit_History.fillna(value=Avg_LoanAmount)
loan.isnull().sum()
loan.describe()
loan.corr()
skew = loan.skew()

print(skew)
loan.Gender[loan.Gender == "Male"] = "1"

loan.Gender[loan.Gender == "Female"] = "2"



loan.Married[loan.Married == "No"] = "0"

loan.Married[loan.Married == "Yes"] = "1"



loan.Education[loan.Education == "Graduate"] = "0"

loan.Education[loan.Education == "Not"] = "1"

loan.Education[loan.Education == "Not Graduate"] = "2"



from sklearn.preprocessing import LabelEncoder

clf = LabelEncoder()

loan["Self_Employed"] = clf.fit_transform(loan["Self_Employed"])

#loan.Self_Employed[loan.Self_Employed == "No"] = "0"

#loan.Self_Employed[loan.Self_Employed == "Yes"] = "1"





from sklearn.preprocessing import LabelEncoder

clf = LabelEncoder()

loan["Dependents"] = clf.fit_transform(loan["Dependents"])



loan.Loan_Status[loan.Loan_Status == "Y"] = "1"

loan.Loan_Status[loan.Loan_Status == "N"] = "0"



loan.Property_Area[loan.Property_Area == "Urban"] = "0"

loan.Property_Area[loan.Property_Area == "Rural"] = "1"

loan.Property_Area[loan.Property_Area == "Semiurban"] = "2"
loan.head()
#plotPerColumnDistribution(loan, 10, 5)

corr = loan.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
%matplotlib inline

loan.hist(figsize = (10,10))

plt.show()
# Density plots:

loan.plot(kind='density',subplots = True , layout = (3,3) , sharex = False , figsize = (20,10))

plt.show()
loan.plot(kind='box',subplots = True , layout = (3,3) , sharex = False , figsize = (20,20))

plt.show()
from matplotlib import pyplot

#from pandas.tools.plotting import radviz

#from pandas.tools.plotting import andrews_curves

#radviz(loan , figsize = (20, 20))

#sns.pairplot(loan ,  size = 3)

#sns.set(style="white", color_codes=True, edgecolor="gray")

#pyplot.show()

sns.pairplot(loan.drop("Loan_ID", axis=1), hue="Loan_Status", size=3)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

X_tra = loan.iloc[:, 1:12]

Y_tra = loan.iloc[:, 12:13]
bestfeature = SelectKBest(score_func = chi2, k=5)

fit = bestfeature.fit(X_tra,Y_tra)
dfscore = pd.DataFrame(fit.scores_)

fcolumns = pd.DataFrame(X_tra.columns)
featurescores = pd.concat([fcolumns, dfscore], axis = 1)

featurescores.columns = ["spec", "scores"]
featurescores
from sklearn.model_selection import train_test_split



x = loan[["Education","ApplicantIncome","CoapplicantIncome","LoanAmount","Credit_History","Property_Area"]]

y = loan[["Loan_Status"]]



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 5)
#venky = pd.DataFrame() 
#comparing classification algorithms:

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from matplotlib import pyplot

models = []

models.append(('LR',LogisticRegression()))

models.append(('LDA',LinearDiscriminantAnalysis()))

models.append(('KNN',KNeighborsClassifier()))

models.append(('CART',DecisionTreeClassifier()))

models.append(('NB',BernoulliNB()))

models.append(('SVM',SVC()))

models.append(("RD",RandomForestClassifier()))

results=[]

loan=[]

for loans,model in models:

    ac= 0.0

    rs =0

    for n in range(1, 100):

        kfold = KFold(n_splits=7,random_state=n)

        result = cross_val_score(model,x,y,cv=kfold,scoring='accuracy')

        #print(result.mean())

        if result.mean() > ac:

            ac = result.mean()

            rs = n

            results.append(result)

            loan.append(loans)

            #print(ac, rs)

    print("%s: %f %.3f (%.3f)" %(loans,rs,ac,result.std()))

        #k= 

   

    #print(k)

       
venky = pd.DataFrame(venky)

venky.head()
fig=pyplot.figure()

fig.suptitle('Classification Algorithms Accuracy Comparision:')

ax=fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(loan)

pyplot.grid()

pyplot.show()
loan_test.head()
loan_test.shape
loan_test.info()
loan_test.isnull().sum()
loan_test.dropna(subset = ["Gender"], how = "all", inplace = True)



loan_test.dropna(subset = ["Dependents"], how = "all", inplace = True)



loan_test.dropna(subset = ["Self_Employed"], how = "all", inplace = True)



avg_depa = loan_test.LoanAmount.mean()

loan_test.LoanAmount = loan_test.LoanAmount.fillna(value = avg_depa)



avg_depa = loan_test.Loan_Amount_Term.mean()

loan_test.Loan_Amount_Term = loan_test.Loan_Amount_Term.fillna(value = avg_depa)



avg_depa = loan_test.Credit_History.mean()

loan_test.Credit_History = loan_test.Credit_History.fillna(value = avg_depa)
loan_test.isnull().sum()
k = loan_test[["Education","ApplicantIncome","CoapplicantIncome","LoanAmount","Credit_History","Property_Area"]]
a = loan_test[["Loan_ID"]]
del loan_test["Gender"]

del loan_test["Married"]

del loan_test["Dependents"]

del loan_test["Self_Employed"]

del loan_test["Loan_Amount_Term"]
k.Education[k.Education == "Graduate"] = "0"

k.Education[k.Education == "Not"] = "1"

k.Education[k.Education == "Not Graduate"] = "2"
k.Property_Area[k.Property_Area == "Urban"] = "0"

k.Property_Area[k.Property_Area == "Rural"] = "1"

k.Property_Area[k.Property_Area == "Semiurban"] = "2"
z = k
z.head()
leg = LogisticRegression(random_state=50)

#leg = LinearDiscriminantAnalysis()



leg = leg.fit(x_train, y_train)

y_pred2 = leg.predict(z)
y_pred2
my_submission = pd.DataFrame({"loan_ID": a.Loan_ID, "Loan_status": y_pred2})
my_submission
my_submission.to_csv("submission.csv", index = False)