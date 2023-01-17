# IMPORTS



import pandas as pd

import numpy as np

import re



import seaborn as sns

from matplotlib import pyplot as plt

sns.set_style('whitegrid')



import statsmodels.api as sm



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv('../input/train.csv')

df = pd.DataFrame(train)

test = pd.read_csv('../input/test.csv')

tdf = pd.DataFrame(test)



print('Training data: \n', df.info())

print('Test data: \n', tdf.info())

print('Mean rate of survival: ', df['Survived'].mean())
# DROP PASSENGER ID



df.drop(['PassengerId'], axis = 1, inplace = True)

tdf.drop(['PassengerId'], axis = 1, inplace = True)
# Looking at the Parch feature



parch_s = df.groupby(['Survived','Parch']).size()

parch_s3plus_count = (parch_s[1][3] + parch_s[0][3] + parch_s[1][5] + parch_s[0][5])

parch_s3plus = (parch_s[1][3] + parch_s[1][5]) / parch_s3plus_count



rates_parch = pd.DataFrame([(parch_s[1][0] / (parch_s[1][0] + parch_s[0][0])),

	(parch_s[1][1] / (parch_s[1][1] + parch_s[0][1])),

	(parch_s[1][2] / (parch_s[1][2] + parch_s[0][2])), parch_s3plus],

	index = ['0','1','2','3+'], columns = ['Parch Survival Rates'])



print('Parch Survival Rates\n-----------------\n', rates_parch)



fig = sns.barplot(x = rates_parch.index, y = 'Parch Survival Rates', data = rates_parch)

fig.set_title('Survival Rates by Parent-Child Total')

fig.axes.set_ylim(0,0.6)

fig.axes.set_ylabel('Percent Survived')

fig.axes.set_xlabel('# of Parents/Children')

plt.show()
# Looking at the SibSp feature



sibsp_s = df.groupby(['Survived','SibSp']).size()

sibsp_s3plus_count = sibsp_s[1][3] + sibsp_s[0][3] + sibsp_s[1][4] + sibsp_s[0][4] + sibsp_s[0][5]+ sibsp_s[0][8]

sibsp_s3plus = (sibsp_s[1][3] + sibsp_s[1][4]) / sibsp_s3plus_count



rates_sibsp = pd.DataFrame([(sibsp_s[1][0] / (sibsp_s[1][0] + sibsp_s[0][0])),

	(sibsp_s[1][1] / (sibsp_s[1][1] + sibsp_s[0][1])),

	(sibsp_s[1][2] / (sibsp_s[1][2] + sibsp_s[0][2])), sibsp_s3plus],

	index = ['0','1','2','3+'], columns = ['Sibsp Survival Rates'])



print('Sibsp Survival Rates\n-----------------\n', rates_sibsp)



fig = sns.barplot(x = rates_sibsp.index, y = 'Sibsp Survival Rates', data = rates_sibsp)

fig.set_title('Survival Rates by Sibsp Total')

fig.axes.set_ylim(0,0.6)

fig.axes.set_ylabel('Percent Survived')

fig.axes.set_xlabel('# of Sibsp')

plt.show()
# Combining Parch and SibSp to arrive at a Relatives variable



df['Relatives'] = df['Parch'] + df['SibSp']

tdf['Relatives'] = tdf['Parch'] + tdf['SibSp']



rel_s = df.groupby(['Survived', 'Relatives']).size()



rel_0_sd = [rel_s[1][[0]].sum(), rel_s[0][[0]].sum()]

rel_14_sd = [rel_s[1][[1,2,3,4]].sum(), rel_s[0][[1,2,3,4]].sum()]

rel_5plus_sd = [rel_s[1][[5,6]].sum(), rel_s[0][[5,6,7,10]].sum()]



rates_rel = pd.DataFrame([(rel_0_sd[0] / sum(rel_0_sd)), (rel_14_sd[0] / sum(rel_14_sd)), (rel_5plus_sd[0] / sum(rel_5plus_sd))], index = ['0','1-4','5+'], columns = ['Total Relative Survival Rates'])



print('Total Relative Survival Rates\n-----------------\n', rates_rel)



fig = sns.barplot(x = rates_rel.index, y = 'Total Relative Survival Rates', data = rates_rel)

fig.set_title('Survival Rates by Total Relatives')

fig.axes.set_ylim(0,0.6)

fig.axes.set_ylabel('Percent Survived')

fig.axes.set_xlabel('# of Relatives')

plt.show()
# Resulting Action - Drop Parch, Sibsp, and Relatives columns - Create dummy variable for 1-4 relatives category



df = df.drop(['Parch', 'SibSp'], axis = 1)

tdf = tdf.drop(['Parch', 'SibSp'], axis = 1)



def rel_dummy(num):

	if num in [1,2,3,4]:

		return 1

	else:

		return 0



df['Relatives'] = df['Relatives'].map(rel_dummy)

tdf['Relatives'] = tdf['Relatives'].map(rel_dummy)
embarked_s = df.groupby(['Embarked', 'Survived'])

emb_c_s = embarked_s.get_group(tuple(['C', 1]))['Embarked'].count()

emb_q_s = embarked_s.get_group(tuple(['Q', 1]))['Embarked'].count()

emb_s_s = embarked_s.get_group(tuple(['S', 1]))['Embarked'].count()

emb_c_d = embarked_s.get_group(tuple(['C', 0]))['Embarked'].count()

emb_q_d = embarked_s.get_group(tuple(['Q', 0]))['Embarked'].count()

emb_s_d = embarked_s.get_group(tuple(['S', 0]))['Embarked'].count()



rates_emb = pd.DataFrame([(emb_c_s / (emb_c_s + emb_c_d)), (emb_q_s / (emb_q_s + emb_q_d)),

	(emb_s_s / (emb_s_s + emb_s_d))],	index = ['C','Q','S'], columns = ['Emb Survival Rates'])



print('Embarkment Port Survival Rates\n-----------------\n', rates_emb)



fig = embarked_s['Embarked'].count().unstack('Survived').plot.bar(stacked=True, color=['r','g'])

fig.set_title("Embarkment Location Survival Counts")

fig.axes.set_ylabel('# Survived')

fig.axes.set_xlabel('Port of Embarkment')



class_embarked = df.groupby(['Embarked', 'Pclass'])

fig = class_embarked['Embarked'].count().unstack('Pclass').plot.bar(stacked=True, color=['g','b','y'])

fig.axes.set_ylabel('# Survived')

fig.axes.set_xlabel('Port of Embarkment')

plt.show()
# Resulting Action - Drop embarkment port in favor of economic class variable



df = df.drop(['Embarked'], axis = 1)

tdf = tdf.drop(['Embarked'], axis = 1)
# Looking at name pre-fixes grouped by economic class to see if they can provide any information for filling our missing age values



print('Pre adjustment: ', df['Age'].mean())



df_mr = df[df['Name'].str.contains('Mr.', regex = False)].groupby(['Pclass'])['Age'].mean()

df_mstr = df[df['Name'].str.contains('Master', regex = False)].groupby(['Pclass'])['Age'].mean()

df_mrs = df[df['Name'].str.contains('Mrs.', regex = False)].groupby(['Pclass'])['Age'].mean()

df_miss = df[df['Name'].str.contains('Miss', regex = False)].groupby(['Pclass'])['Age'].mean()



print('\n\nMister prefixed results: \n\n', df_mr)

print('\n\nMaster prefixed results: \n\n', df_mstr)

print('\n\nMrs. prefixed results: \n\n', df_mrs)

print('\n\nMiss prefixed results: \n\n', df_miss)
# Can we use a regression model to use some features from the data set to create a model for age?



age_train = df[df['Age'].notnull()][['Age', 'Relatives', 'Pclass']]



age_train['Relatives'][age_train['Relatives'] < 3] = 1

age_train['Relatives'][age_train['Relatives'] >= 3] = 0

age_train_class_dummies = pd.get_dummies(age_train['Pclass'])

age_train = age_train.join(age_train_class_dummies)

age_train = age_train.drop(['Pclass', 1], axis = 1)

age_train.rename(columns = {'Relatives':'Rel02',2:'Pclass2',3:'Pclass3'}, inplace = True)



Y = age_train['Age']

X = age_train[['Rel02','Pclass2','Pclass3']]

X = sm.add_constant(X)



age_model = sm.OLS(Y,X).fit()

print(age_model.summary())
# Resulting Actions - Clean up Age variable and drop Name variable



for i in [1,2,3]:

	for j in [df, tdf]:

		j['Age'][j['Age'].isnull() & j['Name'].str.contains('Mr.', regex = False) & j['Pclass'] == i] = df_mr[i]

		j['Age'][j['Age'].isnull() & j['Name'].str.contains('Master', regex = False) & j['Pclass'] == i] = df_mstr[i]

		j['Age'][j['Age'].isnull() & j['Name'].str.contains('Mrs.', regex = False) & j['Pclass'] == i] = df_mrs[i]

		j['Age'][j['Age'].isnull() & j['Name'].str.contains('Miss', regex = False) & j['Pclass'] == i] = df_miss[i]



df['Age'][df['Age'].isnull()] = df['Age'].mean()

tdf['Age'][tdf['Age'].isnull()] = tdf['Age'].mean()
print('Post adjustment: ', df['Age'].mean())



age_data, ((pre_munge, post_munge)) = plt.subplots(nrows=1, ncols=2, sharey=True)

pre_munge.hist(age_train['Age'], color='b', label='Pre-Munging', bins = 20)

post_munge.hist(df['Age'], color='g', label='Post-Munge', bins = 20)

pre_munge.set_title('Pre-Estimation')

post_munge.set_title('Post-Estimation')

plt.suptitle('Age Estimation')

plt.show()
# Resulting Actions - Drop Name variable



df = df.drop(['Name'], axis = 1)

tdf = tdf.drop(['Name'], axis = 1)
gender = df.groupby(['Sex'])

gender_class = df.groupby(['Sex','Pclass'])

gender_rel = df.groupby(['Sex','Relatives'])



print(gender.mean())

print(gender_class.mean())

print(gender_rel.mean())



gender_plt, ((mf, mfc, mfr)) = plt.subplots(nrows=1, ncols=3, sharey=True)

gender['Survived'].mean().plot.bar(ax = mf, color = ['g','r'])

gender_class['Survived'].mean().plot.bar(ax = mfc, color = ['g','g','g','r','r','r'])

gender_rel['Survived'].mean().plot.bar(ax = mfr, color = ['g','g','r','r'])

mf.set_title('Gender Survival Rates')

mfc.set_title('Gender-Class Survival Rates')

mfr.set_title('Gender-Relative Survival Rates')

plt.suptitle('Survival Rate by Gender')

plt.show()
# Action - Replace male-female variable with female dummy variable



df_dummies_fem = pd.get_dummies(df['Sex'])

df_dummies_fem = df_dummies_fem.drop(['male'], axis = 1)

df_dummies_fem = df_dummies_fem.rename(columns = {'female':'Female'})

df = df.drop(['Sex'], axis = 1)

df = df.join(df_dummies_fem)



tdf_dummies_fem = pd.get_dummies(tdf['Sex'])

tdf_dummies_fem = tdf_dummies_fem.drop(['male'], axis = 1)

tdf_dummies_fem = tdf_dummies_fem.rename(columns = {'female':'Female'})

tdf = tdf.drop(['Sex'], axis = 1)

tdf = tdf.join(tdf_dummies_fem)
cabin_df = df[df['Cabin'].notnull()]

cabin_df['CabLtr'] = df['Cabin'].str[0]

cabin_s = cabin_df.groupby(['Survived','CabLtr']).size()

cabin_s_rates = dict()



print('Survival Rate', cabin_s[1].sum() / (cabin_s[0].sum() + cabin_s[1].sum()))



for i in ['A','B','C','D','E','F','G']:

	cabin_s_rates[i] = cabin_s[1][i] / (cabin_s[0][i] + cabin_s[1][i])



cabin_df_group = cabin_df.groupby(['CabLtr'])



cabin_plt, ((cbn_surv, cbn_cls, cbn_fare)) = plt.subplots(nrows=1, ncols=3)

cabin_df_group['Survived'].mean().plot.bar(ax = cbn_surv, ylim = (0.3,0.9))

cabin_df_group['Pclass'].mean().plot.bar(ax = cbn_cls, ylim = (0.5,3.5))

cabin_df_group['Fare'].mean().plot.bar(ax = cbn_fare)

cbn_surv.set_title('Survival Rates')

cbn_cls.set_title('Mean Economic Class')

cbn_fare.set_title('Mean Fare')

plt.suptitle('Cabin Statistics')

plt.show()
# Resulting Action - Drop Cabin



df = df.drop(['Cabin'], axis = 1)

tdf = tdf.drop(['Cabin'], axis = 1)
def get_ticket_number(ticket):

	number = re.search('[0-9]{3,7}$', string = ticket)

	# print(number.group(0))

	if hasattr(number, 'group') == False:

		return 987654321

	else:

		return int(number.group(0))



df['TicketNo'] = df['Ticket'].map(get_ticket_number)



group_ticket_fare = df.groupby(pd.cut(df['TicketNo'], np.arange(0, 400001, 80000))).mean()

print(group_ticket_fare)
# Resulting Action - Drop Ticket and Ticket Number variables



df = df.drop(['Ticket'], axis = 1)

tdf = tdf.drop(['Ticket'], axis = 1)

df = df.drop(['TicketNo'], axis = 1)
#Looking at some summary statistics for Fare - We realize there are some with $0 fares



print('\nPre-Min0 Cleaning\n')

print('Fare mean: ', df['Fare'].mean())

print('Fare standard deviation: ', df['Fare'].std())

print('Fare max: ', df['Fare'].max())

print('Fare min: ', df['Fare'].min())
#Let's find these values as we'll likely need to replace them with an estimate



print(df[df['Fare'] == 0])
# Resulting Action - Fill training data where min = 0 with better estimates



df[(df['Fare'] == 0) & (df['Pclass'] == 1)] = df[(df['Pclass'] == 1) & (df['Relatives'] == 0) & (df['Female'] == 0)]['Fare'].mean()

df[(df['Fare'] == 0) & (df['Pclass'] == 2)] = df[(df['Pclass'] == 2) & (df['Relatives'] == 0) & (df['Female'] == 0)]['Fare'].mean()

df[(df['Fare'] == 0) & (df['Pclass'] == 3)] = df[(df['Pclass'] == 3) & (df['Relatives'] == 0) & (df['Female'] == 0)]['Fare'].mean()



print('\nPost-Min0 Cleaning\n')

print('Fare mean: ', df['Fare'].mean())

print('Fare standard deviation: ', df['Fare'].std())

print('Fare max: ', df['Fare'].max())

print('Fare min: ', df['Fare'].min())
# Clean missing fare value in test data



print(tdf[tdf['Fare'].isnull()])

fare_test_val = df[(df['Female'] == 0) & (df['Relatives'] == 0) & (df['Pclass'] == 3)]

tdf[tdf['Fare'].isnull() == True] = fare_test_val['Fare'].mean()
# Scale fare feature



df['ScaledFare'] = df['Fare'] / df['Fare'].mean()

tdf['ScaledFare'] = tdf['Fare'] / tdf['Fare'].mean()

df = df.drop(['Fare'], axis = 1)

tdf = tdf.drop(['Fare'], axis = 1)



print('\nPost-Min0 Cleaning / Fare scaling:\n')

print('Scaled Fare mean: ', df['ScaledFare'].mean())

print('Scaled Fare standard deviation: ', df['ScaledFare'].std())

print('Scaled Fare max: ', df['ScaledFare'].max())

print('Scaled Fare min: ', df['ScaledFare'].min())
# A final look at the structure of our data



print(df.info())

print(tdf.info())
print(df.head(10))

print(tdf.head(10))
# Define train and test data



Y_train = df['Survived'].astype(np.int64)

X_train = df.drop('Survived', axis = 1)

X_test = tdf.copy()
# Logistic Regression



logreg = LogisticRegression()

logreg_fit = logreg.fit(X_train, Y_train)

logreg_score = logreg.score(X_train, Y_train)

print('Logistic Regression Training Score: ', logreg_score)



logreg_coefdf = pd.DataFrame(tdf.columns)

logreg_coefdf.columns = ['Features']

logreg_coefdf['Coefficients'] = pd.Series(logreg_fit.coef_[0])

print(logreg_coefdf)
# Random Forest Model



rf = RandomForestClassifier(n_estimators = 50)

rf_fit = rf.fit(X_train, Y_train)

rf_score = rf.score(X_train, Y_train)

print('Random Forest Training Score: ', rf_score)



rf_featuredf = pd.DataFrame(tdf.columns)

rf_featuredf.columns = ['Features']

rf_featuredf['Importance'] = pd.Series(rf_fit.feature_importances_)

print(rf_featuredf)
# PREDICTION



Ypred = logreg.predict(X_test)

sub = pd.DataFrame({'Survived': Ypred})

sub.to_csv('titanic_preds_JaisonAbraham_011817.csv')