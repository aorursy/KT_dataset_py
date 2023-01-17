import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_train=pd.read_csv('/kaggle/input/data-mining-assignment-2/train.csv', low_memory=False)
df_test=pd.read_csv('/kaggle/input/data-mining-assignment-2/test.csv', low_memory=False)
print(df_train['Class'].value_counts())
X_train=df_train.drop('Class', axis=1)
y_train = pd.DataFrame(df_train['Class'], columns = ['Class'])
X_test=df_test

X_train=X_train.drop('ID', axis=1)
X_test=X_test.drop('ID', axis=1)
object_columns_dummies=['col2', 'col11', 'col37', 'col44', 'col56']
# object_columns_label=['col56']
X_train_new=pd.get_dummies(X_train[object_columns_dummies])
X_test_new=pd.get_dummies(X_test[object_columns_dummies])
X_train=X_train.drop(object_columns_dummies, axis=1)
X_test=X_test.drop(object_columns_dummies, axis=1)
X_train=pd.concat([X_train, X_train_new], axis=1)
X_test=pd.concat([X_test, X_test_new], axis=1)
# # Import label encoder 
# from sklearn import preprocessing 
  
# # label_encoder object knows how to understand word labels. 
# label_encoder = preprocessing.LabelEncoder()

# for col in object_columns_label:
#     X_train[col]= label_encoder.fit_transform(X_train[col]) 
#     X_test[col]= label_encoder.fit_transform(X_test[col]) 
from sklearn.model_selection import train_test_split
X_train_, X_test_, y_train_, y_test_ = train_test_split(X_train, y_train, test_size=0.25)
from sklearn.ensemble import RandomForestClassifier
clf_orig = RandomForestClassifier(n_estimators = 2000, random_state=42)
clf_orig.fit(X_train, y_train)
def discard_cols(val):
    cols_to_discard=[]
    ind=0
    for col in X_train.columns: 
#         print(col, clf_orig.feature_importances_[ind])
        
        if clf_orig.feature_importances_[ind]<val:
            cols_to_discard.append(col)
        ind+=1
    return cols_to_discard
vals = np.linspace(0, 0.048, 20)
for val in vals:
    cols_to_discard=discard_cols(val)
    print(len(cols_to_discard))
    X_train_dropped=X_train_.drop(cols_to_discard, axis=1)
    X_test_dropped=X_test_.drop(cols_to_discard, axis=1)
    
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators = 2000, random_state=42)
    clf.fit(X_train_dropped, y_train_)
    y_pred_dropped=clf.predict(X_test_dropped)
    
    from sklearn.metrics import f1_score
    ans=f1_score(y_test_, y_pred_dropped, average='micro')
    print(val, ans)
cols_to_discard=discard_cols(0.01768421052631579)
print(len(cols_to_discard))
X_train_dropped=X_train.drop(cols_to_discard, axis=1)
X_test_dropped=X_test.drop(cols_to_discard, axis=1)
X_train_dropped_normalized=X_train_dropped
X_test_dropped_normalized=X_test_dropped
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=2000,random_state=42)
clf.fit(X_train_dropped_normalized, y_train)
y_pred_dropped_normalised=clf.predict(X_test_dropped_normalized)
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score


scorer_f1 = make_scorer(f1_score, average = 'micro')

cv_results=cross_validate(clf, X_train_dropped_normalized, y_train, cv=10, scoring=(scorer_f1), return_train_score=True)
print("Test Accuracy= ",np.mean(cv_results['test_score']))
y_pred_dropped_normalised= pd.DataFrame(data=y_pred_dropped_normalised)
answer=pd.concat([df_test['ID'], y_pred_dropped_normalised], axis=1)
answer.columns=['ID', 'Class']
answer.to_csv('answer_eval_lab.csv', index=False)
print(answer['Class'].value_counts())


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "results.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(answer)
# answer_f=pd.read_csv('results.csv', low_memory=False)
# print(answer_f['Class'].value_counts())
