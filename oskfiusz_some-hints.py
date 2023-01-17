import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
x = train_df.groupby(['Age', 'Rating'], as_index=False)['Id'].count()
plt.rcParams.update({'font.size': 12})
colors=['b','r','y','g', 'Orange', 'Black', 'DarkGreen' , 'DarkBlue', 'DarkRed']
out = []
out.append( pd.cut(train_df[train_df['Rating']==1]['Age'], bins=range(0,90,10), include_lowest=True) )
out[0].value_counts(sort=False).plot.bar(rot=0, color=colors[0], figsize=(12,8))

for r in range(1,9):
    out.append( pd.cut(train_df[train_df['Rating']==r]['Age'], bins=range(0,90,10), include_lowest=True) )
    out[r].value_counts(sort=False).plot.bar(rot=0, color=colors[r], figsize=(12,8), bottom=out[r-1].value_counts(sort=False))

all_df = pd.concat([train_df, test_df], axis=0, sort=False)[['Id','Rating','Age', 'Product_Info_1', 'Product_Info_2', 'Employment_Info_1']]
all_df.head()
from sklearn.preprocessing import LabelEncoder

l_enc = LabelEncoder()
_ = l_enc.fit(all_df['Product_Info_2'])
temp_col = l_enc.transform(all_df['Product_Info_2'])
new_col = pd.DataFrame( temp_col )
new_col.columns = ['Product_Info_2']
all_df = pd.concat([all_df.drop(['Product_Info_2'], axis=1).reset_index(drop=True), new_col], axis=1)

all_df.head()
all_df.isnull().sum()
all_df['Employment_Info_1'].fillna(-1, inplace=True)
all_df.isnull().sum()
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold    
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegression

kf = KFold(n_splits=5) # split data into 5 folds
X = all_df[all_df['Rating'].notnull()].reset_index(drop=True) # select only training data 

columns = ['Model', 'Training_Score', 'Validation_Score']
scores = pd.DataFrame([], columns=columns)

# Train classifier, predict training and validation ratings and return scores 
def score(clf, X_tr, y_tr, X_val, y_val):
    clf.fit(X_tr, y_tr.values.ravel())
    y_res  = clf.predict(X_tr) # predict for training data  
    tr_score = cohen_kappa_score(y_res, y_tr['Rating'], weights="quadratic")
    y_res  = clf.predict(X_val) #predict for validation data
    val_score = cohen_kappa_score(y_res, y_val['Rating'], weights="quadratic")
    return tr_score, val_score

for tr_idx, val_idx in kf.split(X):
    # cut as per indexes of the fold into training and validation
    # for training features we drop Rating, for training labels we use Rating
    X_tr = X.loc[tr_idx].drop(['Rating', 'Id'], axis=1).reset_index(drop=True)
    y_tr = X.loc[tr_idx][['Rating']].reset_index(drop=True)
    X_val = X.loc[val_idx].drop(['Rating', 'Id'], axis=1).reset_index(drop=True)
    y_val = X.loc[val_idx][['Rating']].reset_index(drop=True)

    # Logistic Regression
    tr_score, val_score = score(LogisticRegression(), X_tr, y_tr, X_val, y_val)
    scores = scores.append(pd.DataFrame([['Logistic', tr_score, val_score]], columns=columns))
                              
    # Decision Tree Model
    tr_score, val_score = score(DecisionTreeClassifier(), X_tr, y_tr, X_val, y_val)
    scores = scores.append(pd.DataFrame([['Tree', tr_score, val_score]], columns=columns))

    # Random Forest 
    tr_score, val_score = score(RandomForestClassifier(n_estimators=16, max_depth=4, n_jobs=3, verbose=0), X_tr, y_tr, X_val, y_val)
    scores = scores.append(pd.DataFrame([['Forest', tr_score, val_score]], columns=columns))

scores.groupby(['Model'], as_index=False)[['Training_Score', 'Validation_Score']].mean()
X_tr = all_df[all_df['Rating'].notnull()].drop(['Id', 'Rating'], axis=1).reset_index(drop=True)
y_tr = all_df[all_df['Rating'].notnull()][['Rating']].reset_index(drop=True)
# I'm keeping Id in X_te to produce submission file easier at the very end. 
# Remember to drop it when running predictions
X_te = all_df[all_df['Rating'].isnull()].drop(['Rating'], axis=1).reset_index(drop=True)
clf = DecisionTreeClassifier()
clf.fit(X_tr, y_tr.values.ravel())
y_te = clf.predict(X_te.drop(['Id'], axis=1)) #predict for test data
y_te = pd.DataFrame(y_te)
y_te.columns = ['Rating']
y_te['Rating'] = y_te['Rating'].astype(int)
y_te = pd.concat([X_te[['Id']].reset_index(drop=True),y_te.reset_index(drop=True)], axis=1)                       
y_te.to_csv('./submission_tree.csv', sep=",", index=False)
pd.concat([pd.DataFrame(X_te.drop(['Id'],axis=1).columns, columns=['Feature']), pd.DataFrame(clf.feature_importances_, columns=['Importance'])],axis=1)         
