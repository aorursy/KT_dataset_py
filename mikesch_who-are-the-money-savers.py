import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('fivethirtyeight')
#load data

df1 = pd.read_csv('../input/responses.csv')
df1.head(2)
#create sub-dataset that will be used for predicting perceived ability to save

mov_mus   = df1.iloc[:,[0,19]]

scared    = df1.iloc[:,63:73]

interests = df1.iloc[:,31:63]

demo      = df1.iloc[:,140:150]

spending  = df1.iloc[:,134:140]

predict   = df1.iloc[:,133]



#rolling up fears into one column

scared.fillna(0, inplace=True)

scared = scared.mean(axis=1)



df2 = mov_mus.join([scared, interests, demo, spending, predict])

df2.rename(columns={0:'Scared'}, inplace=True)
df2.info()
#clean missing values

drop_list = ['Gender','Left - right handed','Education','Only child','Village - town','House - block of flats','Finances']

df2.dropna(subset= drop_list, inplace=True)

df2.fillna(0, inplace=True)
#Everyone loves movies & music, right?

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4), sharey=True)



sns.countplot(df2['Movies'], ax=ax1)

ax1.set_xlim(.5,5.5)



sns.countplot(df2['Music'], ax=ax2, palette='hls')

ax2.set_xlim(.5,5.5)

ax2.set_ylabel('')
plt_dict = {}



for i in range(0,len(interests.columns)):

    plt_dict.update({i:interests.columns[i]})

            

fig, ax = plt.subplots(4,8,figsize=(8,5), sharey=True, sharex=True)



initial = 0



for i in range(4):

    for j in range(8):

        sns.countplot(df2[plt_dict[initial]], ax=ax[i,j])

        ax[i,j].set_ylabel('')

        ax[i,j].set_xlabel('')

        ax[i,j].set_xticklabels(labels=np.arange(0,6), fontsize=5)

        ax[i,j].set_yticklabels(labels=np.arange(0,601,100), fontsize=5)

        ax[i,j].set_title(plt_dict[initial], fontsize=5)

        ax[i,j].set_xlim(.5,5.5)

        ax[i,j].set_ylim(0,600)

        initial += 1
#plotting string columns, see what they look like

obj_dict = {0:'Gender', 1:'Left - right handed', 2:'Only child', 3:'Village - town', 4:'House - block of flats', 5:'Education'}

fig, ax = plt.subplots(2,3, figsize=(7,6), sharey=True)



initial = 0



for i in range(2):

    for j in range(3):

        sns.countplot(df2[obj_dict[initial]], ax=ax[i,j])

        ax[i,j].set_title(obj_dict[initial], fontsize=10)

        ax[i,j].set_xlabel('')

        ax[i,j].set_ylabel('')

        ax[i,j].set_xticklabels(labels=df2[obj_dict[initial]].unique(), fontsize=7)

        initial += 1



ax[1,2].set_xticklabels(labels=df2['Education'].unique(), rotation=20, fontsize=6)
#formatting string data for modeling

gender  = pd.get_dummies(df2['Gender'])

handed  = pd.get_dummies(df2['Left - right handed'])

child   = pd.get_dummies(df2['Only child'])

vil_tow = pd.get_dummies(df2['Village - town'])

resid   = pd.get_dummies(df2['House - block of flats'])

educa   = pd.get_dummies(df2['Education'])



df2.drop(['Gender','Left - right handed','Only child','Village - town','House - block of flats','Education'], axis=1, inplace=True)

df2 = df2.join([gender, handed, child, vil_tow, resid, educa])
#plot spending habits by category

spend_aves = pd.Series(df2[spending.columns].mean())

spend_aves = spend_aves.append(pd.Series(df2['Finances'].mean(), index=['Finances']))



spend_aves.plot(

    figsize=(6,5), kind='barh', title='Average Response', 

    color=["#30a2da","#fc4f30","#e5ae38","#6d904f","#8b8b8b",'m', 'r'], xlim=(1,5))



plt.axvline(x=np.mean(spend_aves), color='k', lw=4, ls='dashed')



print('Young folks say they are okay at saving, on 1-5 scale average is {:.2f} - but spending responses are slightly higher at {:.2f}. \n This signals a modest over-confidence in financial discipline'

      .format(df2['Finances'].mean(),np.mean(spend_aves)))
#Instead of doing multi-label prediction, splitting finance into two groups - 3 or less, 4 or more

df2.loc[df2['Finances'] <= 3, 'Finances'] = 0

df2.loc[df2['Finances'] > 3, 'Finances'] = 1
#ML

from sklearn.cross_validation import KFold, train_test_split, cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
#set up data for modeling

x = df2.drop('Finances', axis=1)

y = df2['Finances']



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3)



kf = KFold(len(x_train), n_folds=5)
#Use GridSearchCV for parameter tuning

logreg = LogisticRegression()



param_grid = {'C':[.01,.03,.1,.3,1,3,10]}



gs_logreg = GridSearchCV(logreg, param_grid=param_grid, cv=kf)



gs_logreg.fit(x_train, y_train)

gs_logreg.best_params_
#fit Logistic Regression model, eval scoring

logreg = LogisticRegression(C=.01)

logreg.fit(x_train, y_train)



print('Average accuracy score on cv (KFold) set: {:.3f}'.format(np.mean(cross_val_score(logreg, x_train, y_train, cv=kf))))

print('Accuracy score on test set is: {:.3f}'.format(logreg.score(x_test, y_test)))
#plot feature importance

coeff_df = pd.DataFrame(data=logreg.coef_[0], index=[x_train.columns], columns=['Feature_Import'])

coeff_df = coeff_df.sort_values(by='Feature_Import', ascending=False)



fig, ax1 = plt.subplots(1,1, figsize=(7,6))



sns.barplot(x=coeff_df.index, y=coeff_df['Feature_Import'], ax=ax1)

ax1.set_title('All Features')

ax1.set_xticklabels(labels=coeff_df.index, size=6, rotation=90)

ax1.set_ylabel('Importance')
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(7,10))



sns.barplot(x=coeff_df.index[:10], y=coeff_df['Feature_Import'].head(10), ax=ax1)

ax1.set_title('Top Positive Features')

ax1.set_ylabel('Importance')

ax1.set_xticklabels(labels=coeff_df.index[:10], fontsize=8, rotation=20)





sns.barplot(x=coeff_df.index[-10:], y=coeff_df['Feature_Import'].tail(10), ax=ax2, palette='hls')

ax2.set_title('Top Negative Features')

ax2.set_ylabel('Importance')

ax2.set_xticklabels(labels=coeff_df.index[-10:], fontsize=8, rotation=20)