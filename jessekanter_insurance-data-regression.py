import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# load the data

import os

os.chdir("../input")

df=pd.read_csv('insurance.csv')
df.head()
# set up a figure object for Figure 1

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)#fig, axes = plt.subplots(nrows=1, ncols=2)



# Make a plot of charges versus age and bmi using seaborn 

sns.scatterplot(data=df,x='bmi',y='charges',hue=None,ax=ax1,color='red')

sns.scatterplot(data=df,x='age',y='charges',hue=None,ax=ax2)



# add a caption and plot indicators

txt_fig='Figure 1. charges Vs. (a) bmi and (b) age.  ';

fig.text(0,-.1,txt_fig);

fig.text(.15,.8,'(a)');

fig.text(.57,.8,'(b)');

# set up a figure object for Figure 2

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)#fig, axes = plt.subplots(nrows=1, ncols=2)



# Make a plot of charges versus age and bmi using seaborn, 

# with hue set to smoker/non-smoker and a legend to define keys 

sns.scatterplot(data=df,x='bmi',y='charges',hue='smoker',ax=ax1,legend=False)

s=sns.scatterplot(data=df,x='age',y='charges',hue='smoker',ax=ax2)

s.legend(loc='center right', bbox_to_anchor=(1.6, .8), ncol=1,edgecolor=None)



# add a caption and plot indicators

txt_fig='Figure 2. charges Vs. (a) bmi and (b) age with hue set to smoker.  '

fig.text(0,-.1,txt_fig);

fig.text(.15,.8,'(a)');

fig.text(.57,.8,'(b)');
# A function that assigns a value of 1/0 to patients with

# bmi over/under a value n, n is default set to 30.

def overn(bmi,n=30):

    if bmi>n:

        return 'yes'

    else:

        return 'no'

# create new numerical column indicatting a pateitn 

# as having a bmi over or under 30 

df['bmi_over30']=df['bmi'].apply(lambda x: overn(x,30))
# set up a figure object for Figure 3

fig, ax = plt.subplots(ncols=1)



# Make a plot of charges versus age using seaborn, 

# with hue set to smoker/non-smoker, marker set to bmi being over 30

# and a legend to define keys

cVage=sns.scatterplot(data=df ,x='age',y='charges',hue='smoker',style='bmi_over30',ax=ax)

cVage.legend(loc='center right', bbox_to_anchor=(1.35, .8), ncol=1,edgecolor=None)



# add a caption and plot indicators

txt_fig='Figure 3. charges Vs. age with hue set to smoker and marker to bmi_over30.  '

fig.text(0,-.1,txt_fig);
# A function that assigns a value of 1/0 to patients with

# bmi over/under a value n, n is default set to 30.

def overn(bmi,n=30):

    if bmi>n:

        return 1

    else:

        return 0

# Create new numerical column indicating a patient 

# as having a bmi over or under 30 

df['bmi_over30']=df['bmi'].apply(lambda x: overn(x,30))
#A new data frame with just the data from the smokers.

dfn_smoker=df[df['smoker']=='yes'].drop('smoker',axis=1)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
# Create a data frame of features, X_s, and target values, Y_s, 

# for smoker data.

# Split these into a set of traing data and a set of test data.

X_s=dfn_smoker[['age', 'bmi_over30']]

y_s=dfn_smoker['charges']

X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(X_s, y_s, test_size=0.3, random_state=42)



# Create an instance of a linear regression model,

# train it on our training data 

# and create predictions for test data

lm_s=LinearRegression()

lm_s.fit(X_s_train,y_s_train)

pred_s=lm_s.predict(X_s_test)



# set up a figure object for Figure 4

fig, ax = plt.subplots(ncols=1)



# Make a plot of charges versus age using seaborn, 

# with both test data target values (test) and

# and predictions based on test feature data (prediction).

# Add a legend to define keys

sns.scatterplot(X_s_test['age'],y_s_test,ax=ax,label='test')

sns.scatterplot(X_s_test['age'],pred_s,ax=ax,label='prediction')

ax.legend(loc=1, bbox_to_anchor=(1.35, .9))



# Add a caption

txt_fig='Figure 4. Test and predictions for smokers group  '

fig.text(0,-.1,txt_fig);



# Use metrics to evaluate accuracy of predictions

MAE=metrics.mean_absolute_error(y_s_test,pred_s)

MSE=metrics.mean_squared_error(y_s_test,pred_s)

RMSE=np.sqrt(metrics.mean_squared_error(y_s_test,pred_s))

print('MAE: '+str(round(MAE))+'\nRMSE: '+str(round(RMSE)))
# set up a figure object for Figure 5

fig, ax = plt.subplots(ncols=1,figsize=(8,8))





# Make a plot of charges versus age using seaborn, 

# with hue set to number of children

# and marker set to sex

# Add a legend to define keys

cVage_nosmoker=sns.scatterplot(data=df[(df['smoker']=='no')],x='age',y='charges',hue='children',style='sex',palette='coolwarm',ax=ax)



# Draw a lign seperating groups of data.

df['cutoff']=df['age']*261+-1853+2600

ax.plot(df['age'],df['cutoff'],color='black',linestyle='--')



# add a caption and plot indicators

txt_fig='Figure 5. charges Vs. age for non smoker with hue set to children and marker to sex.  '

fig.text(0,0,txt_fig);
# Create a data frame with just the non-smokers 

# in the group below the black line in Figure 5.

df['group 1']=np.greater(df['cutoff'],df['charges'])

df_nonsmoker=df[df['smoker']=='no']

df_nonsmoker=df_nonsmoker[df_nonsmoker['group 1']==True]


# set up a figure object for Figure 6

fig, ax = plt.subplots(ncols=1,figsize=(6,6))



# Make a plot of charges versus age using seaborn, 

# with hue set to number of children

# and marker set to sex

# Add a legend to define keys

cVage_nosmoker=sns.scatterplot(data=df_nonsmoker,x='age',y='charges',hue='children',style='sex',palette='coolwarm')



# add a caption and plot indicators

txt_fig='Figure 6. charges Vs. age for non smoker with hue set to children and marker to sex.  '

fig.text(0,0,txt_fig);





# Create a numerical column for sex 

# (creates a female and male column)

df_nonsmoker[['female','male']]=pd.get_dummies(df_nonsmoker['sex'],drop_first=False)
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline
# Create a data frame of features, X_ns, and target values, Y_ns, 

# for non-smoker data.

# Split these into a set of traing data and a set of test data.

X_ns=df_nonsmoker[['age','female','children']]

y_ns=df_nonsmoker['charges']

X_ns_train, X_ns_test, y_ns_train, y_ns_test = train_test_split(X_ns, y_ns, test_size=0.3, random_state=42)



# Create an pipeline instance of a Quadratic and Linear Regression 

# model, train it on our training data 

# and create predictions for test data

model_ns= make_pipeline(PolynomialFeatures(2), LinearRegression())

model_ns.fit(X_ns_train,y_ns_train)

pred_ns = model_ns.predict(X_ns_test)



# Set up a figure object for Figure 7

fig, ax = plt.subplots(ncols=1)

sns.scatterplot(X_ns_test['age'],y_ns_test,ax=ax,label='Test data')

sns.scatterplot(X_ns_test['age'],pred_ns,ax=ax,label='Prediction')



# Add a caption and plot indicators

txt_fig='Figure 7. charges Vs. age for non smoker, test data and predictions.'

fig.text(0,-.1,txt_fig);



# Use metrics to evaluate accuracy of predictions

MAE=metrics.mean_absolute_error(y_ns_test,pred_ns)

MSE=metrics.mean_squared_error(y_ns_test,pred_ns)

RMSE=np.sqrt(metrics.mean_squared_error(y_ns_test,pred_ns))

print('MAE: '+str(round(MAE))+'\nRMSE: '+str(round(RMSE)))
# Create and show data frame with coeffiecnts from our model's fit 

indexc=['intercept','age','female','children','age^2','age*female','age*children','female^2','female*children','children^2']

coef_poly=pd.DataFrame(model_ns.steps[1][1].coef_.transpose(),index=indexc,columns=['Coefficient'])

coef_poly['Coefficient'][0]=model_ns.steps[1][1].intercept_

coef_poly.apply(lambda s:round(s))
