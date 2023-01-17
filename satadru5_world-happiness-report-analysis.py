# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.utils import shuffle

from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df1 = pd.read_csv("../input/2015.csv")

df2 = pd.read_csv("../input/2016.csv")
df1.columns
df1.head(2)
sns.regplot(x='Standard Error',y='Happiness Score' ,data=df1)
sns.regplot(x='Economy (GDP per Capita)',y='Happiness Score' ,data=df1)

#so there is a linear relation between GDP & Happiness Score
fr=['Standard Error', 'Economy (GDP per Capita)', 'Family','Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)','Generosity', 'DystopiaResidual']
plt.figure(figsize=(12,28*4))

gs = gridspec.GridSpec(28, 1)

for i, cn in enumerate(fr):

    ax = plt.subplot(gs[i])

    #sns.distplot(df1[cn], bins=50)

    sns.regplot(x=df1[cn],y='Happiness Score' ,data=df1)

    ax.set_xlabel('')

    ax.set_title('Regrassion of feature: ' + str(cn))

plt.show()
#Then it has been shown that feature are mostly co-realted with happiness score.
df1.head(2)
df=df1

cnt=df['Country']

rgn=df['Region']

rnk=df['Happiness Rank']

df1=df1.drop(['Country','Region','Happiness Rank'],axis=1)



corr=df1.corr()

corr = (corr)

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title('Heatmap of Correlation Matrix')
df1.head(2)
#Train-Test split

from sklearn.model_selection import train_test_split

label = df1.pop('Happiness Score')

data_train, data_test, label_train, label_test = train_test_split(df1, label, test_size = 0.2, random_state = 42)
data_train.count(),data_test.count()
#Logistic Regression

from sklearn.linear_model import LinearRegression

linear = LinearRegression()

linear.fit(data_train, label_train)

linear_score_train = linear.score(data_train, label_train)

print("Training score: ",linear_score_train)

linear_score_test = linear.score(data_test, label_test)

print("Testing score: ",linear_score_test)
Predict=linear.predict(data_test)
result_lnr_reg=pd.DataFrame({

    'Actual':label_test,

    'Predict':Predict

})
result_lnr_reg.head(4)
sns.regplot(x='Actual',y='Predict',data=result_lnr_reg)
#decision tree

from sklearn.ensemble import RandomForestRegressor

dt = RandomForestRegressor()

dt.fit(data_train, label_train)

dt_score_train = dt.score(data_train, label_train)

print("Training score: ",dt_score_train)

dt_score_test = dt.score(data_test, label_test)

print("Testing score: ",dt_score_test)
pd.DataFrame({

        'Model'          : ['Logistic Regression', 'SVM', 'kNN', 'Decision Tree', 'Random Forest'],

        'Training_Score' : [logis_score_train, svm_score_train, knn_score_train, dt_score_train, rfc_score_train],

        'Testing_Score'  : [logis_score_test, svm_score_test, knn_score_test, dt_score_test, rfc_score_test]

    })

models.sort_values(by='Testing_Score', ascending=False)
Predict_rf=dt.predict(data_test)

result_rf=pd.DataFrame({

    'Actual':label_test,

    'Predict':Predict_rf,

    'diff':label_test-Predict_rf

})
Predict_rf_train=dt.predict(data_train)

result_rf_train=pd.DataFrame({

    'Actual':label_train,

    'Predict':Predict_rf_train,

    'diff':label_train-Predict_rf_train

})
result_rf.head(4)
sns.pointplot(x='Actual',y='Predict',data=result_rf)
sns.regplot(x='Predict',y='diff',data=result_rf)

#Very minor differnet
plt.scatter(x='Predict',y='diff',data=result_rf,c='b',alpha=0.5)

plt.scatter(x='Actual',y='diff',data=result_rf_train,c='r',alpha=0.5)



#Residial ploting for Rf