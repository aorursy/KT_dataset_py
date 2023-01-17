import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data=pd.read_csv('../input/insurance.csv')
data.head()
data.shape
data.info()
#convert columns to catgorical columns

#Not needed 

#data['sex']=pd.Categorical(data['sex'])

#data['region']=pd.Categorical(data['region'])

#data['smoker']=pd.Categorical(data['smoker'])
data.describe()
sns.pairplot(data)
#create new column based on healthiness



#data=data.drop(columns='healthy')

data['healthy'] =np.nan
def fun(bmi_str):

    if (bmi_str >=18.5 and bmi_str <= 24.9) :

        return 1

    else:

        return 0



data['healthy']=data['bmi'].apply(fun)
data.info()
#only if its in object format it can be processed using one hot encoding

data['healthy']=data['healthy'].astype(object)
data.info()
sns.distplot(data['age'])
sns.distplot(data['bmi'])
sns.distplot(data['children'])
pd.value_counts(data['smoker'])
sns.boxplot(y='charges',x='smoker',data=data)
#This tab is not used. Anova is used while comaring 



data_smoker=data[data['smoker']=='yes']

data_no_smoker=data[data['smoker']=='no']

data_smoker=(data_smoker['charges']).astype(int)

data_no_smoker=(data_no_smoker['charges']).astype(int)





plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

sns.distplot(data_smoker)

plt.subplot(1,2,2)

sns.distplot(data_no_smoker)



#these are independent columns,

from scipy.stats import mannwhitneyu

z_statistic,p_value=mannwhitneyu(data_no_smoker,data_smoker)

print(z_statistic,p_value)



#Since p<0.05 ,null hyp is rejected. Thus both columns are from different population(Different mean)
pd.value_counts(data['sex'])
sns.boxplot(y='charges',x='sex',data=data)
#This tab is not used. Anova is used while comaring 



data_male=data[data['sex']=='male']

data_female=data[data['sex']=='female']

data_male=(data_male['charges']).astype(int)

data_female=(data_female['charges']).astype(int)





plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

sns.distplot(data_male)

plt.subplot(1,2,2)

sns.distplot(data_female)



#these are independent columns,

from scipy.stats import mannwhitneyu,ttest_ind

z_statistic,p_value=mannwhitneyu(data_male,data_female)

z_statistic_ttest,p_value_ttest=mannwhitneyu(data_male,data_female)

print(z_statistic,p_value)

print(z_statistic_ttest,p_value_ttest)



#Since p>0.05 ,null hyp is accepted. Thus both columns are from same population(same mean)
pd.value_counts(data['region'])
sns.boxplot(y='charges',x='region',data=data)
#This tab is not used. Anova is used while comaring 



data_reg1=data[data['region']=='southeast']

data_reg2=data[data['region']=='southwest']

data_reg3=data[data['region']=='northwest']

data_reg4=data[data['region']=='northeast']

data_reg1=(data_reg1['charges']).astype(int)

data_reg2=(data_reg2['charges']).astype(int)

data_reg3=(data_reg3['charges']).astype(int)

data_reg4=(data_reg4['charges']).astype(int)





plt.figure(figsize=(10,5))

plt.subplot(2,2,1)

sns.distplot(data_reg1)

plt.subplot(2,2,2)

sns.distplot(data_reg2)

plt.subplot(2,2,3)

sns.distplot(data_reg3)

plt.subplot(2,2,4)

sns.distplot(data_reg4)



#these are independent columns,Following ANOVA - since there are more than 2 categories(columns)

from statsmodels.formula.api import ols 

from statsmodels.stats.anova import anova_lm

from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison



formula = 'charges ~ C(region)'

model = ols(formula, data).fit()

aov_table = anova_lm(model)

print(aov_table)





#Since p<0.05 ,null hyp is rejected. So we must use multi comparison(Refer one way anova- golf exapmle)



mc = MultiComparison(data['charges'], data['region'])

result = mc.tukeyhsd()

 

print(result)

print(mc.groupsunique)



#for all other case except last one - 0 is in between lower and upper bound values. So accept null hyp(mean is equal) 

#for those and reject for last case. 

pd.value_counts(data['healthy'])
sns.boxplot(y='charges',x='healthy',data=data)
sns.boxplot(y='charges',x='children',data=data)
#find correlation in case of categorical data



from scipy.stats import chisquare,chi2_contingency



print(chisquare(data["sex"].value_counts()))

print(chisquare(data["smoker"].value_counts()))

print(chisquare(data["region"].value_counts()))

print(chisquare(data["healthy"].value_counts()))
# Goodness of Fit Test between 2 categorical variables



# H0: The two categorical variables are independent

# Ha: The two categorical variables are dependent



# Creating contingency table

cont = pd.crosstab(data["sex"],

                   data["smoker"])



print(cont)

print(chi2_contingency(cont))



#The p-value 0.006 < 0.05 hence we conclude that the 2 categorical variables are dependent
cont = pd.crosstab(data["smoker"],

                   data["region"])



print(cont)

print(chi2_contingency(cont))



#The p-value 0.06 > 0.05 hence we conclude that the 2 categorical variables are independent
cont = pd.crosstab(data["smoker"],

                   data["healthy"])



print(cont)

print(chi2_contingency(cont))



#The p-value 0.46 > 0.05 hence we conclude that the 2 categorical variables are independent
cont = pd.crosstab(data["sex"],

                   data["healthy"])



print(cont)

print(chi2_contingency(cont))



#The p-value 0.40 > 0.05 hence we conclude that the 2 categorical variables are independent
correlation=data.corr()

correlation
data.head()

data.info()
# Convert categorical variable into dummy/indicator variables. As many columns will be created as distinct values

# This is also kown as one hot coding. The column names will be America, Europe and Asia... with one hot coding

# Like feature scaling



#Replace 1 and 0 with values in healthy column

data['healthy'] = data['healthy'].replace({0:'not_healthy', 1:'healthy'})
#one hot coding

data=pd.get_dummies(data,columns=['sex','smoker','region','healthy'],drop_first='True')
data.head()
#PREPROCESSING DATA - Standardization and training data split



from sklearn import preprocessing

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



X=data.drop(columns='charges')

y=data['charges']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=1)



# Create the Scaler object

scaler = preprocessing.StandardScaler()



#feature scaling in dependent and independent columns

X_train=scaler.fit_transform(X_train)

X_test=scaler.transform(X_test)



        #Gets some error

#y_train=scaler.fit_transform(y_train)

#y_test=scaler.transform(y_test)



linear_reg=LinearRegression()

linear_reg.fit(X_train,y_train)



linear_reg.score(X_test,y_test)
#Using SKLEARN  - Without Standardized



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



X=data.drop(columns='charges')

y=data['charges']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5,random_state=1)

linear_reg=LinearRegression()

linear_reg.fit(X_train,y_train)



linear_reg.score(X_test,y_test)
#Using OLS method



from statsmodels.formula.api import ols   



formula='charges ~ healthy_not_healthy +  smoker_yes + sex_male + children + age + bmi'

model=ols(formula,data).fit()

model.summary()
#Polynomial regression





from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model



X1=data.drop(columns='charges')

y1=data['charges']



X_poly_train,X_poly_test,y_poly_train,y_poly_test=train_test_split(X1,y1,test_size=0.25,random_state=1)



poly = PolynomialFeatures(degree=2, interaction_only=True)



X1_poly_train=poly.fit_transform(X_poly_train)

X1_poly_test=poly.fit_transform(X_poly_test)



lin=linear_model.LinearRegression()

lin.fit(X1_poly_train,y_poly_train)



y_pred=lin.predict(X1_poly_test)



lin.score(X1_poly_test,y_poly_test)