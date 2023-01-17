import pandas as pd
import numpy  as np
from matplotlib import pyplot as plt

np.random.seed(2018)

# load data sets 
predictors = ['Ticket','Pclass']
train = pd.read_csv('../input/train.csv', usecols =['Survived','PassengerId'] + predictors)
test  = pd.read_csv('../input/test.csv' , usecols =['PassengerId'] + predictors)

# combine train and test for joint processing 
test['Survived'] = np.nan
comb = pd.concat([ train, test ])
comb.head()
comb.loc[comb['Ticket']=='LINE']
comb['Ticket'] = comb['Ticket'].replace('LINE','LINE 0')
dup_tickets = comb.groupby('Ticket').size()
comb['DupTickets'] = comb['Ticket'].map(dup_tickets)
plt.xlabel('duplications')
plt.ylabel('frequency')
plt.title('Duplicate Tickets')
comb['DupTickets'].hist(bins=20)
# remove dots and slashes
comb['Ticket'] = comb['Ticket'].apply(lambda x: x.replace('.','').replace('/','').lower())
def get_prefix(ticket):
    lead = ticket.split(' ')[0][0]
    if lead.isalpha():
        return ticket.split(' ')[0]
    else:
        return 'NoPrefix'
comb['Prefix'] = comb['Ticket'].apply(lambda x: get_prefix(x))
comb['TNumeric'] = comb['Ticket'].apply(lambda x: int(x.split(' ')[-1])//1)
comb['TNlen'] = comb['TNumeric'].apply(lambda x : len(str(x)))
comb['LeadingDigit'] = comb['TNumeric'].apply(lambda x : int(str(x)[0]))
comb['TGroup'] = comb['Ticket'].apply(lambda x: str(int(x.split(' ')[-1])//10))
comb.head()
pd.crosstab(comb['Pclass'],comb['LeadingDigit'])
comb = comb.drop(columns=['Ticket','TNumeric','Pclass'])
comb = pd.concat([pd.get_dummies(comb[['Prefix','TGroup']]), comb[['PassengerId','Survived','DupTickets','TNlen','LeadingDigit']]],axis=1)
comb.shape
predictors = sorted(list(set(comb.columns) - set(['PassengerId','Survived'])))
# comb2 now becomes the combined data in numeric with the PassengerId feature removed
comb2 = comb[predictors + ['Survived']]
comb2.head()
df_train = comb2.loc[comb2['Survived'].isin([np.nan]) == False]
df_test  = comb2.loc[comb2['Survived'].isin([np.nan]) == True]

print(df_train.shape)
df_train.head()
print(df_test.shape)
df_test.head()
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
knclass = KNeighborsClassifier(n_neighbors=11, metric = 'manhattan')
param_grid = ({'n_neighbors':[6,7,8,9,11],'metric':['manhattan','minkowski'],'p':[1,2]}) 
grs = GridSearchCV(knclass, param_grid, cv = 28, n_jobs=1, return_train_score = True, iid = False, pre_dispatch=1)
grs.fit(np.array(df_train[predictors]), np.array(df_train['Survived']))
print("Best parameters " + str(grs.best_params_))
gpd = pd.DataFrame(grs.cv_results_)
print("Estimated accuracy of this model for unseen data:{0:1.4f}".format(gpd['mean_test_score'][grs.best_index_]))
pred_knn = grs.predict(np.array(df_test[predictors]))

sub = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pred_knn})
sub.to_csv('ticket_only_knn.csv', index = False, float_format='%1d')
sub.head()