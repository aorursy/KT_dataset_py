# making imports



import pandas as pd

import matplotlib.pyplot as plt

import sklearn

import numpy as np

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
# Highly imbalanced dataset

train_df['Class'].value_counts()
# Lots of missing values disguised as ?

train_df.head()
train_df = train_df.replace({'?':np.nan})
train_df.info()
train_df.columns
non_null_ints = ['ID', 'Age','IC', 'OC','Timely Income','Gain', 'Loss', 'Stock','Weight',

                'Vet_Benefits', 'Weaks', 'WorkingPeriod',

                 'Class','NOP','Own/Self']



test_cols = ['ID', 'Age','IC', 'OC','Timely Income','Gain', 'Loss', 'Stock','Weight',

                'Vet_Benefits', 'Weaks', 'WorkingPeriod',

                 'NOP','Own/Self']
train_subset = train_df[non_null_ints]
test_subset = test_df[test_cols]
encode = [ 'Schooling', 'Married_Life','Cast','Sex','Full/Part','Tax Status','Summary','Citizen']
from sklearn.preprocessing import LabelEncoder
for col in encode:

    print(col)

    le = LabelEncoder()

    le.fit(train_df[col])

    

    train_subset[col] = le.transform(train_df[col])

    test_subset[col] = le.transform(test_df[col])
train_subset.info()
test_subset.info()
positive_dataset = train_subset[train_subset['Class']==0]

negative_dataset = train_subset[train_subset['Class']==1]
print(len(positive_dataset))

print(len(negative_dataset))
#14 sets?

#left for val

train_dfs = []



for i in range(14):

    pos_sample = positive_dataset[i*6292:(i+1)*6292]

    temp_df = pd.concat([pos_sample]) #can skip appending negative everytie if ram issue

    train_dfs.append(temp_df)
val = positive_dataset[14*6292:]

val_df = pd.concat([val,negative_dataset[6000:]])

val_X = val_df.drop(['Class'],axis=1)

val_y = val_df['Class']
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression
type(train_dfs[0])
clfs = []

results = []

for i in range(14):

    rf = AdaBoostClassifier(n_estimators=200)

    #rf = LogisticRegression()

    train_df = pd.concat([train_dfs[i],negative_dataset[:6000]])

    train_df = train_df.sample(frac = 1,random_state=42)

    X = train_df.drop(['Class'],axis=1)

    y = train_df['Class']

    rf.fit(X,y)

    clfs.append(rf)

    pred = rf.predict(val_X)

    results.append(roc_auc_score(val_y,pred))

    

print(results)

print(np.asarray(results).mean())
#0.88531002291229953
preds = []

for i in range(14):

    clf = clfs[i]

    pred_prob = clf.predict_proba(test_subset)[:,0]

    preds.append(pred_prob)
np_pred = np.asarray(preds)
mean_pred = np.mean(np_pred,axis=0)
final_pred = [0 if i>=0.5 else 1 for i in mean_pred ]
np.asarray(final_pred).mean()
sub = pd.DataFrame(index=test_subset['ID'])
sub['Class'] = final_pred
sub.to_csv('Submission.csv')
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(sub)