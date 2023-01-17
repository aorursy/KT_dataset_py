# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
%matplotlib inline
import numpy as np # linear algebra
import matplotlib as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Read .csv files to specific dataframes
df_B = pd.read_csv("../input/DataSet/Deodorant B.csv")
df_F = pd.read_csv("../input/DataSet/Deodorant F.csv")
df_G = pd.read_csv("../input/DataSet/Deodorant G.csv")
df_H = pd.read_csv("../input/DataSet/Deodorant H.csv")
df_J = pd.read_csv("../input/DataSet/Deodorant J.csv")
test = pd.read_csv("../input/DataSet/test_data.csv")
frames=[df_B,df_F,df_G,df_H,df_J]
df=pd.concat(frames)
print(df.shape)
df.columns
print(test.shape)
test.columns
common_cols = list(set.intersection(*(set(df.columns) for df in frames)))
d = pd.concat([df[common_cols] for df in frames], ignore_index=False)
d.shape
sm = set(test).symmetric_difference(d)
sm
for i in sm :
    d[i] = np.nan
print(d.shape)
d=d[['Respondent.ID', 'Product.ID', 'Product','Instant.Liking',
       'q1_1.personal.opinion.of.this.Deodorant', 'q2_all.words',
       'q3_1.strength.of.the.Deodorant', 'q4_1.artificial.chemical',
       'q4_2.attractive', 'q4_3.bold', 'q4_4.boring', 'q4_5.casual',
       'q4_6.cheap', 'q4_7.clean', 'q4_8.easy.to.wear', 'q4_9.elegant',
       'q4_10.feminine', 'q4_11.for.someone.like.me', 'q4_12.heavy',
       'q4_13.high.quality', 'q4_14.long.lasting', 'q4_15.masculine',
       'q4_16.memorable', 'q4_17.natural', 'q4_18.old.fashioned',
       'q4_19.ordinary', 'q4_20.overpowering', 'q4_21.sharp',
       'q4_22.sophisticated', 'q4_23.upscale', 'q4_24.well.rounded',
       'q5_1.Deodorant.is.addictive', 'q7', 'q8.1', 'q8.2', 'q8.5', 'q8.6',
       'q8.8', 'q8.11', 'q8.12', 'q8.13', 'q8.19', 'q8.20',
       'q9.how.likely.would.you.be.to.purchase.this.Deodorant',
       'q10.prefer.this.Deodorant.or.your.usual.Deodorant',
       'q11.time.of.day.would.this.Deodorant.be.appropriate',
       'q12.which.occasions.would.this.Deodorant.be.appropriate',
       'Q13_Liking.after.30.minutes',
       'q14.Deodorant.overall.on.a.scale.from.1.to.10', 'ValSegb',
       's7.involved.in.the.selection.of.the.cosmetic.products',
       's8.ethnic.background', 's9.education', 's10.income',
       's11.marital.status', 's12.working.status', 's13.2',
       's13a.b.most.often', 's13b.bottles.of.Deodorant.do.you.currently.own']]
d
a=d.copy() # copy dataframe'd' to dataframe 'a'
a.iloc[0:500,3] = df_B['Instant.Liking']
a.iloc[500:1001,3] = df_F['Instant.Liking']
a.iloc[1001:1501,3] = df_G['Instant.Liking']
a.iloc[1501:2001,3] = df_H['Instant.Liking']
a.iloc[2001:2501,3] = df_J['Instant.Liking']
a['Instant.Liking'] = a['Instant.Liking'].astype(int)
a 
#a.iloc[[2000]] To see specific row values
df_J.columns.get_loc('s13a.j.most.often') #For Checking the column index_no.
#Filling appropriate datavalues from dataframes df_B, df_F, df_G, df_H, df_J into the specific locations of 'a'
a.iloc[0:500,39] = df_B['q8.12']
a.iloc[1001:1501,39] = df_G['q8.12']
a.iloc[1501:2001,39] = df_H['q8.12']
a.iloc[2001:2501,39] = df_J['q8.12']


a.iloc[0:500,34] = df_B['q8.2']
a.iloc[1001:1501,34] = df_G['q8.2']
a.iloc[1501:2001,34] = df_H['q8.2']


a.iloc[0:500,42] = df_B['q8.20']
a.iloc[500:1001,42] = df_F['q8.20']
a.iloc[1501:2001,42] = df_H['q8.20']

a.iloc[0:500,37] = df_B['q8.8']

a.iloc[0:500,56] = df_B['s13.2']

a.iloc[0:500,57] = df_B['s13a.b.most.often']
a.iloc[500:1001,57] = df_F['s13a.f.most.often']
a.iloc[1001:1501,57] = df_G['s13a.g.most.often']
a.iloc[1501:2001,57] = df_H['s13a.h.most.often']
a.iloc[2001:2501,57] = df_J['s13a.j.most.often']
a['s13a.b.most.often'] = a['s13a.b.most.often'].astype(int)
#Imputing NAN values in dataframe 'a' with zeros
a.iloc[500:1001,39] = 0
a['q8.12'] = a['q8.12'].astype(int)

a.iloc[500:1001,34] = 0
a.iloc[2001:2501,34]= 0
a['q8.2'] = a['q8.2'].astype(int)

a.iloc[1001:1501,42]= 0
a.iloc[2001:2501,42] = 0
a['q8.20'] = a['q8.20'].astype(int)

a.iloc[500:2501,37] = 0
a['q8.8'] = a['q8.8'].astype(int)

a.iloc[500:2501,56] = 0
a['s13.2'] = a['s13.2'].astype(int)

a.dtypes
a.describe()
train = a.copy() # train dataset for modelling
train.to_csv('train', sep = '\t', encoding = 'utf-8') #Converted df to train.csv File and saved
#Predictions 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
train['Product']= number.fit_transform(train['Product'])
test['Product']= number.fit_transform(test['Product'])

# Linear regression model 
from sklearn import linear_model
y_train = train['Instant.Liking']
X_train = train.drop(['Instant.Liking'], axis=1)
X_test = test
linear = linear_model.LinearRegression()
print(linear.fit(X_train, y_train))
print(linear.score(X_train, y_train))
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
#Predict Output
pred= linear.predict(X_test)
pred= pred.astype(int)

X_test['Product']= number.inverse_transform(test['Product'].astype(int))
a = pd.DataFrame(data=pred,columns=['Instant.Liking'])
b=pd.concat([X_test['Product'].astype('str'),a], axis =1)
pred_final=pd.concat([X_test['Respondent.ID'], b], axis =1)
pred_final.to_csv('Deodorant_predict_LinearRegression', sep ='\t')
#Deodorant_predict_LinearRegression has accuracy score of 67% 
from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
train['Product']= number.fit_transform(train['Product'])
test['Product']= number.fit_transform(test['Product'])
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
fit = lr.fit(X_train, y_train)
print(fit)
score = lr.score(X_train, y_train)
print(score)

#Equation coefficient and Intercept
print('Coefficient: \n', lr.coef_)
print('Intercept: \n', lr.intercept_)
#Predict Output
prediction = lr.predict(X_test)
print(prediction)
pred= prediction.astype(int)

X_test['Product']= number.inverse_transform(test['Product'].astype(int))
a = pd.DataFrame(data=pred,columns=['Instant.Liking'])
b=pd.concat([X_test['Product'].astype('str'),a], axis =1)
pred_final=pd.concat([X_test['Respondent.ID'], b], axis =1)
pred_final.to_csv('Deodorant_predict_LogisticRegression', sep ='\t')

#Deodorant_predict_LogisticRegression has highest accuracy score of 99.9% .So this is the best prediction model for 
# predicting each Deodorant product's Instant.Liking 
#Interesting Insights using seaborn and matplotlib 
# Insights from Predicted data
import seaborn as sns
D_lr = pd.read_csv("../input/Deodorant_predict_LogisticRegression.csv")
D_lr
z = pd.crosstab(index=D_lr["Product"], columns=D_lr["Instant.Liking"])
z.plot(kind="bar", figsize=(16,8), stacked=True)
g = sns.FacetGrid(D_lr, col="Product", size=5, aspect=.9)
g.map(sns.barplot, "Instant.Liking", "Respondent.ID");
from matplotlib import pyplot as plt
sns.violinplot(x='Product', y='Respondent.ID', data=D_lr)
sns.swarmplot(x='Product', y='Respondent.ID', data=D_lr)
sns.countplot(x='Instant.Liking', data=D_lr)
#Insights from train data
print(train.shape)
print(train.dtypes)
sns.lmplot(x='q3_1.strength.of.the.Deodorant', y='Respondent.ID', data=train,
           fit_reg=False, # No regression line
           hue='Instant.Liking')
import matplotlib.pyplot as plt
plt.scatter
g = sns.FacetGrid(train, col= "Product", hue="Instant.Liking")
g.map(plt.scatter, "q3_1.strength.of.the.Deodorant", "Respondent.ID",alpha=.9)
g.add_legend();