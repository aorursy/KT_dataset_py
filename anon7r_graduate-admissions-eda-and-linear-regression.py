# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import seaborn as sns
graduate_data = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
graduate_data.shape
## Dropping the Serial No column

graduate_data.drop(columns = 'Serial No.', inplace = True)
## check for missing values

## turns our there aren't any

graduate_data.isnull().sum()
fig, ax = plt.subplots(2,4)

fig.set_size_inches(11,11)

fig.suptitle("Distribution of the different variables", fontsize = 16)

fig.text(0.06, 0.5, 'Count of Occurence', ha='center', va='center', rotation='vertical', fontsize=16)

k = 0

for i in range(0,2):

    for j in range(0,4):

        ax[i,j].hist(graduate_data.iloc[0:,k], bins = 10)

        ax[i,j].set_title(graduate_data.columns[k])

        k += 1

        #print(k)

       

plt.show()
fig, ax = plt.subplots(2,4, sharey = 'row')

fig.set_size_inches(11,11)

fig.suptitle("Relation between Independent variables and Dependent variable", fontsize = 16)

k = 0

for i in range(0,2):

    for j in range(0,4):

        ax[i,j].scatter(graduate_data.iloc[0:,k],graduate_data.iloc[:,7])

        ax[i,j].set_title(graduate_data.columns[k])

        k += 1

        #print(k)

fig.text(0.06, 0.5, 'Chances of Admit', ha='center', va='center', rotation='vertical', fontsize=16)

fig.delaxes(ax[1,3])        

plt.show()
fig = plt.figure(figsize = (9,9))



corr_graduate = graduate_data.corr()

# Generate a mask for the upper right triangle of the square - one half is enough to convey the correlation 

## between the predictors

mask = np.zeros_like(corr_graduate, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Generate the correlation plot 

sns.heatmap(corr_graduate, mask = mask, center=0, annot = True, square=True, linewidths=.5).set_title("Correlation Plot", fontsize = 16)



plt.show()
##Binning the admit chances into four quartiles

#graduate_data['Admit_quartiles'] = pd.qcut(graduate_data['Chance of Admit '], 4, labels = ['low','medium','good','Almost there'])

graduate_data['Admit_quartiles'] = pd.qcut(graduate_data['Chance of Admit '], 4, labels = False)
## Scatter Plot with GRE/ TOEFL scores on axes and the the admit chance quartiles as different quartiles

## The highest admit chance quartile is 3 which is mapped to Green and the lowest is 0 which is mapped to Red. 



color_dict = {0: 'red', 1: 'blue', 2: 'yellow', 3: 'green'}



fig, ax = plt.subplots()

fig.set_size_inches(9,9)

for quartile in graduate_data['Admit_quartiles'].unique():

    index = graduate_data.index[graduate_data['Admit_quartiles']==quartile]

    ax.scatter(graduate_data.loc[index,'GRE Score'], graduate_data.loc[index,'TOEFL Score'], 

               c = color_dict[quartile], label = quartile, s = 50)

ax.legend()

ax.set_xlabel("GRE Score")

ax.set_ylabel("TOEFL Score")

ax.set_title("Relation between Admit chance and GRE/ TOEFL score", fontsize = 16)

plt.show()
## Scatter Plot with University Rating/ CGPA on axes and the the admit chance quartiles as different quartiles

## The highest admit chance quartile is 3 which is mapped to Green and the lowest is 0 which is mapped to Red. 



color_dict = {0: 'red', 1: 'blue', 2: 'yellow', 3: 'green'}



fig, ax = plt.subplots()

fig.set_size_inches(9,9)

for quartile in graduate_data['Admit_quartiles'].unique():

    index = graduate_data.index[graduate_data['Admit_quartiles']==quartile]

    ax.scatter(graduate_data.loc[index,'University Rating'], graduate_data.loc[index,'CGPA'], 

               c = color_dict[quartile], label = quartile, s = 50)

ax.legend()

ax.set_xlabel("University Rating")

ax.set_ylabel("CGPA")

ax.set_title("Relation between Admit chance and University Rating/ CGPA score", fontsize = 16)

plt.show()
min_GREscore_Admit2 = graduate_data[graduate_data['Admit_quartiles']==2]['GRE Score'].min()

min_TOEFLscore_Admit2 = graduate_data[graduate_data['Admit_quartiles']==2]['TOEFL Score'].min()



lower_score_analysis = graduate_data.loc[(graduate_data['GRE Score']<320)&(graduate_data['TOEFL Score']<110)&(graduate_data['GRE Score']>min_GREscore_Admit2)&(graduate_data['TOEFL Score']>min_TOEFLscore_Admit2)]
fig, axes = plt.subplots(2, 3, figsize = (15,15))

axes = axes.flatten()



#2,3,4,5,6 indicates the column indexes of the columns which we are interested in analysing

for i in [2,3,4,5,6]:

    sns.boxplot(x="Admit_quartiles", y=lower_score_analysis.iloc[:,i], data=lower_score_analysis, orient='v', ax=axes[i-2])

fig.delaxes(axes[5])

plt.tight_layout()

plt.show()
from scipy import stats

#from statsmodels.stats import weightstats as stests
def t_test_fn(group1, group2, **kwargs):

    ttest, pval = stats.ttest_ind(group1,group2, **kwargs)

    

    if pval<0.01:

        result = "reject null hypothesis"

    else:

        result = "accept null hypothesis"

    #print(result)

    return(ttest, pval, result)
GRE_scores_group1 = graduate_data.loc[graduate_data['Admit_quartiles']==3,'GRE Score'].tolist()

GRE_scores_group2 = graduate_data.loc[graduate_data['Admit_quartiles']==2,'GRE Score'].tolist()



t_stat, p_value, result = t_test_fn(GRE_scores_group1, GRE_scores_group2, equal_var = False)

print("T statistic is ",t_stat)

print("Probability of getting this T statistic due to random chance ",p_value)

print(result)
Research_group1 = graduate_data.loc[graduate_data['Admit_quartiles']==3,'Research'].tolist()

Research_group2 = graduate_data.loc[graduate_data['Admit_quartiles']==2,'Research'].tolist()



t_stat, p_value, result = t_test_fn(Research_group1, Research_group2, equal_var = False)

print("T statistic is ",t_stat)

print("Probability of getting this T statistic due to random chance ",p_value)

print(result)
Research_group1 = graduate_data.loc[graduate_data['Admit_quartiles']==1,'Research'].tolist()

Research_group2 = graduate_data.loc[graduate_data['Admit_quartiles']==0,'Research'].tolist()



t_stat, p_value, result = t_test_fn(Research_group1, Research_group2, equal_var = False)

print("T statistic is ",t_stat)

print("Probability of getting this T statistic due to random chance ", p_value)

print(result)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

from sklearn.preprocessing import StandardScaler
graduate_data.columns
X = graduate_data.drop(['Chance of Admit ','Admit_quartiles'], axis = 1)

y = graduate_data['Chance of Admit ']
## The random state value helps in selecting the same samples for the train test split - so that we can validate 

## results over multiple runs with the same train/ test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 111)
grad_model = LinearRegression()



##Model training using the training data

grad_model.fit(X_train, y_train)



## using the fitted model to predict on the test data

grad_prediction = grad_model.predict(X_test)



## Getting the co-efficients of the independent variables and the intercept

print("The model coefficients are: ",grad_model.coef_)

print("The model intercept is: %.2f"%grad_model.intercept_)



## Finding how much of the variation in the Target variable is explained by our model

print('Target variation explained by model: %.2f' % r2_score(y_test, grad_prediction))



## Finding the error in the predictions

print('Mean squared error: %.3f' % mean_squared_error(y_test, grad_prediction))
## finding the residuals - the difference between the actual target variable values and the predictions

residuals = y_test - grad_prediction



plt.figure(figsize = (7,7))

plt.scatter(grad_prediction, residuals)

plt.xlabel("Fitted values")

plt.ylabel("Residuals")

plt.title("Residual Plot", fontsize = 16)

plt.show()
import statsmodels.api as sm

X_train = sm.add_constant(X_train)

model = sm.OLS(y_train,X_train).fit()

model.summary()
X = graduate_data.drop(['Chance of Admit ','Admit_quartiles', 'SOP', 'University Rating'], axis = 1)

y = graduate_data['Chance of Admit ']
## The random state value helps in selecting the same samples for the train test split - so that we can validate 

## results over multiple runs with the same train/ test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 111)



grad_model = LinearRegression()



##Model training using the training data

grad_model.fit(X_train, y_train)



## using the fitted model to predict on the test data

grad_prediction = grad_model.predict(X_test)



## Getting the co-efficients of the independent variables and the intercept

print("The model coefficients are: ",grad_model.coef_)

print("The model intercept is: %.2f"%grad_model.intercept_)



## Finding how much of the variation in the Target variable is explained by our model

print('Target variation explained by model: %.2f' % r2_score(y_test, grad_prediction))



## Finding the error in the predictions

print('Mean squared error: %.3f' % mean_squared_error(y_test, grad_prediction))
import statsmodels.api as sm

X_train = sm.add_constant(X_train)

model = sm.OLS(y_train,X_train).fit()

model.summary()