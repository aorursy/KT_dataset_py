# to predict term_deposit (represented as y column)

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics


bank = pd.read_csv("../input/bank.csv")
bank_df = pd.DataFrame(bank)

#changing yes/no to 1/0

bank['y'] = (bank['y'] == 'yes').astype(int)

#education column have basic.4y basic.y and basic.9y combining all of them in basic

bank['education'] = np.where(bank['education'] == 'basic.4y','basic',bank['education'])
bank['education'] = np.where(bank['education'] == 'basic.6y','basic',bank['education'])
bank['education'] = np.where(bank['education'] == 'basic.9y','basic',bank['education'])

#exploration of y column
print(bank_df['y'].value_counts())
#there are 3668 no and 451 yes for term deposit

#data visualization
fig,ax = plt.subplots(3,2,figsize = (15,10))
plt.subplots_adjust(wspace= .2,hspace=1.7)
edu = pd.crosstab(bank_df['education'],bank['y'])
edu.plot(ax=ax[0,0],kind='bar')
ax[0,0].set_title('Education_level vs Purchase')
ax[0,0].set_xlabel("Education")
ax[0,0].set_ylabel("Purchase")

month = pd.crosstab(bank_df['month'],bank_df['y'])
month.plot(ax=ax[0,1],kind='bar')
ax[0,1].set_title('Month vs Purchase')
ax[0,1].set_xlabel("Month")
ax[0,1].set_ylabel("Purchase")

marital = pd.crosstab(bank_df['marital'],bank_df['y'])
marital.plot(ax=ax[1,0],kind='bar',stacked=True)
ax[1,0].set_title('Marita_Status vs Purchase')
ax[1,0].set_xlabel("Marital Status")
ax[1,0].set_ylabel("Purchase")

poutcome = pd.crosstab(bank_df['poutcome'],bank_df['y'])
poutcome.plot(ax=ax[1,1],kind='bar')
ax[1,1].set_title('Poutcome vs Purchase')
ax[1,1].set_xlabel("P_Outcome")
ax[1,1].set_ylabel("Purchase")

bank['age'].plot(ax=ax[2,0],kind='hist')
ax[2,0].set_title('Age vs Purchase')
ax[2,0].set_xlabel("Age")
ax[2,0].set_ylabel("Purchase")

ax[2,1].set_visible(False)
#dummy variable for categorical variables 

cat_vars = ['job','marital','default','education','housing','contact','loan','month','day_of_week','poutcome']

for v in cat_vars :
    cat_v = v+'_'
    cat_v= pd.get_dummies(bank_df[v],prefix=v)
    bank_v = bank_df.join(cat_v)
    bank_df = bank_v
#filtering the columns
bank_vars = bank_df.columns

final_col = [i for i in bank_vars if i not in cat_vars]

bank_ = bank_df[final_col]

bank_final_col = bank_.columns

Y=['y']
X=[i for i in bank_final_col if i not in Y]

#feature selection

model = LogisticRegression()
selector = RFE(model,10)                      #selecting best 10 variables as predictors
selector = selector.fit(bank_[X],bank_[Y])

print("Selected Predictors : ")

selected_cols = []
for col,rank in zip(X,selector.support_) :
    if rank==1 :
        selected_cols.append(col)
print(selected_cols)
#implementing 

#statsmodel
X_ = bank_[selected_cols]
Y_ = bank_['y']
logistic_model = sm.Logit(Y_,X_)
res = logistic_model.fit()
print(res.summary())
#using scikit-learn
logit_model = LogisticRegression()
logit_model.fit(X_,Y_)

print(logit_model.score(X_,Y_))
#the model in 90% accurate

#model evaluation
scores = cross_val_score(LogisticRegression(),X_,Y_,scoring='accuracy',cv=10)
print("Model Score : ", scores)
print("Model mean score : ",scores.mean())
# we see that the mean of scores do not variate much with respect to individual means, this means model generalizes well

#model validation

#ROC curve

train_x,test_x,train_y,test_y = train_test_split(X_,Y_,test_size=.3)

model_ = LogisticRegression()
model_.fit(train_x,train_y)

probs = model_.predict_proba(test_x)
prob = probs[:,1]
prob_df = pd.DataFrame(prob)
prob_df['actual'] = test_y

threshold_prob = [0.05,0.1,0.13,0.17,0.2,0.2,0.23,0.25]

fpr,senstivity,_ = metrics.roc_curve(test_y,prob)
area = metrics.auc(fpr,senstivity)
print("Area under ROC curve = ", 100*area)
plt.figure(figsize=(15,10))
plt.plot(fpr,senstivity,'r-')
plt.fill_between(fpr,senstivity,color='b')

#the area under the ROC curve is 78.6 % which can be considered good