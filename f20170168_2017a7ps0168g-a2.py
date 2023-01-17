import numpy as np
import pandas as pd
df=pd.read_csv('../input/dm-assignment2/train(1).csv')
test_data=pd.read_csv('../input/dm-assignment2/test(1).csv')
print(df.columns)
#domain knowledge
df=df.drop(['ID'],axis=1)
test_data=test_data.drop(['ID'],axis=1)
obj_cols=df.columns[df.dtypes=='object']
obj_cols=np.array(obj_cols)
obj_cols
df['col2'].unique()
#assign numerical values to object data types
for i in range(len(obj_cols)):
  dict={}
  for j in  range(len(df[obj_cols[i]].unique())):
    dict[df[obj_cols[i]].unique()[j]]=j+1
  df[obj_cols[i]]=df[obj_cols[i]].map(dict)
  test_data[obj_cols[i]]=test_data[obj_cols[i]].map(dict)

df
test_data.head()
import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(20, 20))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax, annot = True);
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
to_drop
corr_matrix = test_data.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(np.bool))
to_droptest = [column for column in upper.columns if any(upper[column] > 0.95)]
to_droptest #this is a subset of to_drop. So we drop these features from both
for i in to_drop:
  df=df.drop([i],axis=1)
  test_data=test_data.drop([i],axis=1)
df['origin']=0
test_data['origin']=1
df.head()
#sample from training and test data
training = df.sample(300, random_state=7)
testing = test_data.sample(250, random_state=7)

combine=training.append(testing)
y=combine['origin']
combine=combine.drop(['origin','Class'],axis=1)
combine
y=np.array(y).astype('int32')
y
from sklearn.utils import shuffle
combine,y = shuffle(combine, y, random_state=0)
combine
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

rfc1=RandomForestClassifier(random_state=7, max_features='auto', n_estimators= 200, max_depth=11,criterion='gini',min_samples_split=4)
drop_list = []
for i in combine.columns:
    score = cross_val_score(rfc1,pd.DataFrame(combine[i]),y,cv=3,scoring='f1_weighted')
    if (np.mean(score) > 0.70):
        drop_list.append(i)
        print(i,np.mean(score))
y_train=df['Class']
y_train=np.array(y_train).astype('int32')
y_train
for i in drop_list:
  df=df.drop([i],axis=1)
df.head()
df=df.drop(['Class','origin'],axis=1)
df.head()
x_train=df
x_train=np.array(x_train).astype('float32')
x_train
from sklearn import preprocessing
min_max_scaler = preprocessing.StandardScaler()
X_scale = min_max_scaler.fit_transform(x_train)
print(X_scale.shape)
X_scale
from sklearn.utils import shuffle
X_scale,y_train = shuffle(X_scale, y_train, random_state=0)
from sklearn.ensemble import RandomForestClassifier

regr = RandomForestClassifier(random_state=7)
param_grid = { 
    'n_estimators': [245,250,260,270],
    'min_samples_split':[3,4,5,6,7,8,9,10],
    'max_depth' : [8,9,10,11],
    'criterion' :['gini','entropy']
}
from sklearn.model_selection import GridSearchCV
CV_rfc = GridSearchCV(estimator=regr, param_grid=param_grid, cv= 4,scoring='f1_macro')
CV_rfc.fit(X_scale,y_train)
CV_rfc.best_params_
test_data=test_data.drop(['origin'],axis=1)
test_data.head()
for i in drop_list:
  test_data=test_data.drop([i],axis=1)

test_data.head()
test_scale=test_data
test_scale=np.array(test_scale).astype('float32')
test_scale = min_max_scaler.fit_transform(test_scale)
test_scale.shape
rfc1=RandomForestClassifier(random_state=7, max_features='auto', n_estimators= 260, max_depth=8,criterion='gini',min_samples_split=10)
scores = cross_val_score(rfc1, X_scale, y_train, cv=4,scoring='f1_micro')
print(scores)
print(scores.mean())
scores = cross_val_score(rfc1, X_scale, y_train, cv=4,scoring='f1_macro')
print(scores)
print(scores.mean())
scores = cross_val_score(rfc1, X_scale, y_train, cv=4,scoring='f1_weighted')
print(scores)
print(scores.mean())
rfc1.fit(X_scale,y_train)
y_final=rfc1.predict(test_scale)
y_final
y_final.shape
df2=pd.read_csv('../input/dm-assignment2/test(1).csv')
df2.head()
ids=df2['ID']
final_df=pd.DataFrame()
final_df['ID']=ids
final_df['Class']=y_final
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(final_df)