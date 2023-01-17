import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.outliers_influence import variance_inflation_factor
#Predict whether the student is placed or not
campus_df = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
campus_df = campus_df.iloc[:,0:-1].copy() #we dnt require salary field as we just have to decide whether the student is placed or not
campus_df.head()
campus_df_eda = campus_df.iloc[:,0:-1].copy()
#Indexing
campus_df.index = campus_df.sl_no
campus_df = campus_df.drop('sl_no',axis = 1)
campus_df.head(2)
campus_df.info() 
X = campus_df.iloc[:,1:-2] #independent variables
X.head()
Y = campus_df.status #dependent valariable
#Y.head()
campus_df.describe() #show details of only numeric data
campus_df.corr() #checking the correlation between the numeric fileds
#Handling the categorical data
#changing the object datypes to category datatypes because the  operations are faster in such cols than object datatype
campus_df.gender = campus_df.gender.astype('category')
campus_df.ssc_b = campus_df.ssc_b.astype('category')
campus_df.hsc_b = campus_df.hsc_b.astype('category')
campus_df.hsc_s = campus_df.hsc_s.astype('category')
campus_df.degree_t = campus_df.degree_t.astype('category')
campus_df.workex = campus_df.workex.astype('category')
campus_df.specialisation = campus_df.specialisation.astype('category')
campus_df.status = campus_df.status.astype('category')
#checking the unique values of each categorical data
print(campus_df.gender.unique())
print(campus_df.ssc_b.unique())
print(campus_df.hsc_b.unique())
print(campus_df.hsc_s.unique())
print(campus_df.degree_t.unique())
print(campus_df.workex.unique())
print(campus_df.specialisation.unique())
print(campus_df.status.unique())
#as there are only two unique values using label encoder
label_enc = LabelEncoder()
campus_df.gender = label_enc.fit_transform(campus_df.gender) #1 for MALE and 0 for FEMALE
campus_df.ssc_b = label_enc.fit_transform(campus_df.ssc_b) #1 for OTHERS and 0 for CENTRAL
campus_df.hsc_b = label_enc.fit_transform(campus_df.hsc_b) #1 for OTHERS and 0 for CENTRAL
campus_df.workex = label_enc.fit_transform(campus_df.workex) #1 for YES and 0 for NO
campus_df.specialisation = label_enc.fit_transform(campus_df.specialisation) #1 for Mkt&HR and 0 for Mkt&Fin
campus_df.status = label_enc.fit_transform(campus_df.status) #1 for PLACED and 0 for NOT PLACED
#campus_df.status = campus_df.status.round()
campus_df.head()

#usning Dummy trap from OneHotEncoding

#for hsc_b
field = pd.get_dummies(campus_df.hsc_s)
campus_df = pd.concat([campus_df,field], axis = 1)
campus_df = campus_df.drop('hsc_s', axis = 1)
#for degree_t
degree = pd.get_dummies(campus_df.degree_t)
campus_df = pd.concat([campus_df,degree],axis = 1)
campus_df = campus_df.drop('degree_t',axis= 1)


campus_df_cat = campus_df.copy()

campus_df.head()
#Now we can perform regression as all the features are converted into numeric values
campus_df.corr()
#FEATURE EXTRACTION
#Now we need to select the features which has highest impact on the target value using SlectKBest
#internal statistics used - chi2
Xnew = campus_df.drop('status',axis = 1)
Xnew.head()
Ynew = campus_df.status
#Ynew.head()
#checking for multicollinearity using  varaince_inflation_factor
model_before = campus_df
series_before = pd.Series([variance_inflation_factor(Xnew.values, i) for i in range(Xnew.shape[1])], index = Xnew.columns)
print('DATA BEFORE')
print('-'*100)
print(series_before)

# all values are less than 5, therefore no multicollinearity
bstfeatures = SelectKBest(score_func = chi2, k = 12)
fit = bstfeatures.fit(Xnew,Ynew)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(Xnew.columns)
fscore = pd.concat([dfcolumns,dfscores], axis = 1)
fscore
#using OLS model
Xnew = sm.add_constant(Xnew)
Xnew.head()
model = sm.OLS(Ynew,Xnew).fit()
model.summary()
#splliting the training and the testing data
xtrain, xtest, ytrain, ytest = train_test_split(Xnew, Ynew, test_size = .20, random_state = 1)
#using LinearRegression method to find the accuracy
#defining the model
linearreg = LinearRegression()
linearreg.fit(xtrain, ytrain)
y_pred = linearreg.predict(xtest)
res = r2_score(ytest,y_pred)
res*100
df = pd.DataFrame({'Actual': ytest, 'Predicted': y_pred.round()})
df
#Findind the accuracy using features with high P value that we got using SelectKBest
x = campus_df[['ssc_p', 'hsc_p', 'degree_p', 'workex', 'etest_p', 'specialisation', 'mba_p']]
y = campus_df.status
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
model.summary()
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .20, random_state = 1)
lr = LinearRegression()
lr.fit(xtrain,ytrain)
y_predict = lr.predict(xtest)
ress = r2_score(ytest,y_predict)
ress*100
#accuracy is reduced
#Doing some EDA
print(campus_df_eda.gender.value_counts())
labels = 'Male', 'Female'
size = [139,76]
plt.pie(size, labels=labels,  autopct='%1.1f%%', shadow=True) #startangle=140)
plt.show()
#From this pie chart we can say that more Males are placed than Females
names = ['ssc_p','hsc_p','mba_p','degree_p']
values = [(campus_df_eda['ssc_p'].mean()),(campus_df_eda['hsc_p'].mean()),(campus_df_eda['mba_p'].mean()),
          (campus_df_eda['degree_p'].mean())]
plt.bar(names, values)
plt.ylabel('Average Percentage')
plt.show()
