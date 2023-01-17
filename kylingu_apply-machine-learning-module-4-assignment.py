import pandas as pd
import numpy as np
import os
print(os.listdir("../input"))
df_train = pd.read_csv('../input/blight-violations/train.csv', encoding = 'ISO-8859-1', low_memory=False).set_index('ticket_id')
df_address = pd.read_csv('../input/vilation-address/addresses.csv', encoding = 'ISO-8859-1', low_memory=False).set_index('ticket_id')
df_train.head(3)
df_train.describe(include='all').T
df_address.head(3)
city, state= zip(*df_address.address.apply(lambda x: x.split(', ')[1].split(' ')))
violation_address = pd.DataFrame({'vio_city': city, 'vio_state': state}, index=df_address.index)
violation_address.describe(include='all')
df_train.info()
df_train.compliance.value_counts(dropna=False)/len(df_train)
df_train_all = df_train.copy()
df_train = df_train.dropna(subset=['compliance'])
df_train.groupby('agency_name').compliance.agg(['count', 'sum', 'mean', 'std'])
print(df_train_all.disposition.unique(), df_train.disposition.unique())

disposition_replace = {'Responsible by Default': 'By default',
                       'Responsible by Determination': 'By determination', 
                       'Responsible (Fine Waived) by Deter': 'Fine Waived',
                       'Responsible by Admission': 'By admission',
                       'SET-ASIDE (PENDING JUDGMENT)': 'Pending',
                       'PENDING JUDGMENT': 'Pending',
                       'Not responsible by Dismissal': 'Not responsible',
                       'Not responsible by City Dismissal': 'Not responsible',
                       'Not responsible by Determination': 'Not responsible'
                      }

df_train_all.disposition.replace(disposition_replace, inplace=True)
df_train_all.groupby('disposition').compliance.agg(['count', 'sum', 'mean', 'std'])
df_train.disposition.replace(disposition_replace, inplace=True)
df_train.groupby('disposition').compliance.agg(['count', 'sum', 'mean', 'std'])
df_train.groupby('country').compliance.agg(['count', 'sum', 'mean', 'std'])
a = df_train.groupby('state').compliance.agg(['count', 'sum', 'mean', 'std']).sort_values('count', ascending=False)
a['compl_rate'] = a['sum']/a['count']
a['count_rate'] = a['count']/len(df_train)
a
us_statesus_state  = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL',
             'IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT',
             'NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI',
             'SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']

df_train['is_in_state'] = df_train.state.apply(lambda x: x if x in us_statesus_state else 'foreign')
df_train.is_in_state[(df_train.is_in_state != 'foreign') & (df_train.is_in_state != 'MI')] = 'out_of_state'
df_train.groupby('is_in_state').compliance.agg(['count', 'sum', 'mean', 'std']).sort_values('count', ascending=False)
df_train.groupby('inspector_name').compliance.agg(['count', 'sum', 'mean', 'std']).sort_values('count', ascending=False)
df_train.groupby('violation_code').compliance.agg(['count', 'sum', 'mean', 'std']).sort_values('count', ascending=False)
%matplotlib inline
df_train.groupby('discount_amount').compliance.agg(['count', 'sum', 'mean', 'std']).sort_values('count', ascending=False)
df_train['is_discount'] = df_train.discount_amount.apply(lambda x:1 if x > 0 else 0)
# a = pd.cut(df_train.judgment_amount, bins=[0, 50, 100, 150, 200, 250, 300, 350, 12000], labels=list('abcdefgh'))
df_train.groupby('judgment_amount').compliance.agg(['count', 'sum', 'mean', 'std']).sort_index(ascending=False)
# 0	0	140	305	11030

df_train['judgment_level'] = pd.cut(df_train.judgment_amount, bins=[-1, 140, 305, float("inf")])

# df_train.judgment_amount.apply(lambda x: )
df_train.groupby('judgment_level').compliance.agg(['count', 'sum', 'mean', 'std']).sort_index(ascending=False)
selection = ['judgment_level', 'is_discount', 'is_in_state', 'disposition', 'agency_name', 'compliance']
df_selected = df_train[selection]
df_selected.head(3)
df_selected.info()
X = df_selected.drop(columns='compliance')
Y = df_selected.compliance
X.shape, Y.shape
X_encoded = pd.get_dummies(X)
X_encoded.head(3)
X_encoded['disposition_Not responsible'] = np.zeros(len(X))
X_encoded['disposition_Pending'] = np.zeros(len(X))
X_encoded.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X_encoded, Y, random_state = 0)
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier().fit(X_train, y_train)

from sklearn.metrics import roc_auc_score
y_train_pred = clf.predict(X_train)
print('train score ', roc_auc_score(y_train, y_train_pred))

y_pred = clf.predict(X_test)
print('test score ', roc_auc_score(y_test, y_pred))
from sklearn.model_selection import GridSearchCV

params= {'learning_rate': [0.1, 0.3, 1, 3], 'n_estimators':[100], 'max_depth':[3, 5, 8]}

clf = GradientBoostingClassifier(random_state=0)
gscv = GridSearchCV(estimator=clf, param_grid=params, scoring='roc_auc', cv=5, n_jobs=8)
gscv.fit(X_encoded, Y)

# gscv.best_score_, gscv.best_params
gscv.best_score_, gscv.best_params_
y_pred = gscv.predict(X_test)
print('test score ', roc_auc_score(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

params = {'n_estimators':range(1, 50, 5)}
clf = RandomForestClassifier(random_state=0)
gscv_rfc = GridSearchCV(estimator=clf, param_grid=params, scoring='roc_auc', cv=5, n_jobs=8)
gscv_rfc.fit(X_encoded, Y)
gscv_rfc.best_score_, gscv_rfc.best_params_
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1).fit(X_train, y_train)

print('LogisticRegression')
y_train_pred = clf.predict(X_train)
print('train score ', roc_auc_score(y_train, y_train_pred))

y_pred = clf.predict(X_test)
print('test score ', roc_auc_score(y_test, y_pred))

# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_jobs=8, n_neighbors=10)
knn.fit(X_train, y_train)

print('KNeighbours')
y_train_pred = clf.predict(X_train)
print('train score ', roc_auc_score(y_train, y_train_pred))

y_pred = clf.predict(X_test)
print('test score ', roc_auc_score(y_test, y_pred))
df_test = pd.read_csv('../input/blight-violations/test.csv').set_index('ticket_id')
df_test.head(3)
df_test.describe(include='all').T
df_test['judgment_level'] = pd.cut(df_test.judgment_amount, bins=[-1, 140, 305, float("inf")])
df_test['is_discount'] = df_test.discount_amount.apply(lambda x:1 if x > 0 else 0)
us_statesus_state  = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL',
             'IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT',
             'NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI',
             'SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']

df_test['is_in_state'] = df_test.state.apply(lambda x: x if x in us_statesus_state else 'foreign')
df_test.is_in_state[(df_test.is_in_state != 'foreign') & (df_test.is_in_state != 'MI')] = 'out_of_state'

disposition_replace = {'Responsible by Default': 'By default',
                       'Responsible by Determination': 'By determination', 
                       'Responsible (Fine Waived) by Deter': 'Fine Waived',
                       'Responsible by Admission': 'By admission',
                       'SET-ASIDE (PENDING JUDGMENT)': 'Pending',
                       'PENDING JUDGMENT': 'Pending',
                       'Not responsible by Dismissal': 'Not responsible',
                       'Not responsible by City Dismissal': 'Not responsible',
                       'Not responsible by Determination': 'Not responsible',
                       'Responsible (Fine Waived) by Admis': 'Fine Waived',
                       'Responsible - Compl/Adj by Default': 'By default',
                       'Responsible - Compl/Adj by Determi': 'By determination',
                       'Responsible by Dismissal': 'By default'
                      }

df_test.disposition.replace(disposition_replace, inplace=True)
df_test.disposition.value_counts()
selection = ['judgment_level', 'is_discount', 'is_in_state', 'disposition', 'agency_name']
df_test_selected = df_test[selection]
df_test_selected.head(3)
df_test_selected.info()
final_df_test = pd.get_dummies(df_test_selected)
final_df_test.head()
df_test_selected.agency_name.unique()
final_df_test['agency_name_Health Department'] = np.zeros(len(final_df_test), dtype=np.int)
final_df_test['agency_name_Neighborhood City Halls'] = np.zeros(len(final_df_test), dtype=np.int)
final_df_test['disposition_Not responsible'] = np.zeros(len(final_df_test))
final_df_test['disposition_Pending'] = np.zeros(len(final_df_test))
ret = gscv.predict_proba(final_df_test)[:, None, 1]
predict_probs = pd.Series(ret.reshape(len(final_df_test),), index=final_df_test.index)
predict_probs.rename('compliance').astype('float32')
import pandas as pd
import numpy as np

def blight_model():
    
    # Your code here
#   loading data
    print('1. loading data')
    
    df_train = pd.read_csv('../input/blight-violations/train.csv', encoding = 'ISO-8859-1', low_memory=False).set_index('ticket_id')
#   cleaing data and adjust data
    print('2. cleaning data')
    
    df_train = df_train.dropna(subset=['compliance'])
    disposition_replace = {'Responsible by Default': 'By default',
                       'Responsible by Determination': 'By determination', 
                       'Responsible (Fine Waived) by Deter': 'Fine Waived',
                       'Responsible by Admission': 'By admission',
                       'SET-ASIDE (PENDING JUDGMENT)': 'Pending',
                       'PENDING JUDGMENT': 'Pending',
                       'Not responsible by Dismissal': 'Not responsible',
                       'Not responsible by City Dismissal': 'Not responsible',
                       'Not responsible by Determination': 'Not responsible',
                       'Responsible (Fine Waived) by Admis': 'Fine Waived',
                       'Responsible - Compl/Adj by Default': 'By default',
                       'Responsible - Compl/Adj by Determi': 'By determination',
                       'Responsible by Dismissal': 'By default'
                      }

    df_train.disposition.replace(disposition_replace, inplace=True)
    
    print('3. feature engineering')
    
    us_statesus_state  = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL',
             'IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT',
             'NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI',
             'SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']

    df_train['is_in_state'] = df_train.state.apply(lambda x: x if x in us_statesus_state else 'foreign')
    df_train.is_in_state[(df_train.is_in_state != 'foreign') & (df_train.is_in_state != 'MI')] = 'out_of_state'
    df_train['is_discount'] = df_train.discount_amount.apply(lambda x:1 if x > 0 else 0)
    df_train['judgment_level'] = pd.cut(df_train.judgment_amount, bins=[-1, 140, 305, float("inf")])
    selection = ['judgment_level', 'is_discount', 'is_in_state', 'disposition', 'agency_name', 'compliance']
    df_selected = df_train[selection]
    X = df_selected.drop('compliance', axis=1)
    Y = df_selected.compliance
    X_encoded = pd.get_dummies(X)
    X_encoded.head(3)
    X_encoded['disposition_Not responsible'] = np.zeros(len(X))
    X_encoded['disposition_Pending'] = np.zeros(len(X))
    
#     train
    print('4. training')
    
    from sklearn.model_selection import GridSearchCV
#    using GBDT, and best param for GBDT
    from sklearn.ensemble import GradientBoostingClassifier
    params= {'learning_rate': [0.3], 'n_estimators':[100], 'max_depth':[3]}
    clf = GradientBoostingClassifier(random_state=0)
    
#   try SVM, too slow...
#     from sklearn.svm import SVC
#     params = {'gamma':[0.001], 'kernel':['rbf'], }
#     clf = SVC(random_state=0)
    
    gscv = GridSearchCV(estimator=clf, param_grid=params, scoring='roc_auc', cv=5, n_jobs=-1)
    gscv.fit(X_encoded, Y)
    
    print('5. training complete with best score', gscv.best_score_, gscv.best_params_)
#     test 

    print('6. final test')
    
    df_test = pd.read_csv('../input/blight-violations/test.csv').set_index('ticket_id')
    df_test['judgment_level'] = pd.cut(df_test.judgment_amount, bins=[-1, 140, 305, float("inf")])
    df_test['is_discount'] = df_test.discount_amount.apply(lambda x:1 if x > 0 else 0)
    df_test['is_in_state'] = df_test.state.apply(lambda x: x if x in us_statesus_state else 'foreign')
    df_test.is_in_state[(df_test.is_in_state != 'foreign') & (df_test.is_in_state != 'MI')] = 'out_of_state'
    df_test.disposition.replace(disposition_replace, inplace=True)
    
    selection = ['judgment_level', 'is_discount', 'is_in_state', 'disposition', 'agency_name']
    df_test_selected = df_test[selection]
    
    final_df_test = pd.get_dummies(df_test_selected)
    final_df_test['agency_name_Health Department'] = np.zeros(len(final_df_test), dtype=np.int)
    final_df_test['agency_name_Neighborhood City Halls'] = np.zeros(len(final_df_test), dtype=np.int)
    final_df_test['disposition_Not responsible'] = np.zeros(len(final_df_test))
    final_df_test['disposition_Pending'] = np.zeros(len(final_df_test))
    ret = gscv.predict_proba(final_df_test)[:, None, 1]
    predict_probs = pd.Series(ret.reshape(len(final_df_test),), index=final_df_test.index)
    
    return predict_probs.rename('compliance').astype('float32')# Your answer here
# blight_model()
