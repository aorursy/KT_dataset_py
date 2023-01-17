import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
%matplotlib inline
df = pd.read_csv("../input/data-mining-assignment-2/train.csv")
test = pd.read_csv("../input/data-mining-assignment-2/test.csv")
df.head()
df.info()
df[['col2','col11','col37', 'col44', 'col56']]
l = ['col2','col11','col37', 'col44', 'col56']
for i in l:
    print(df[i].unique())
df.duplicated().sum()
def encoding(param , df):
    if param =='Label':
        df['col2'] = df['col2'].replace(['Silver', 'Gold', 'Diamond', 'Platinum'], [0,1,2,3])
        df['col11'] = df['col11'].replace(['No', 'Yes'], [0,1])
        df['col37'] = df['col37'].replace(['Male', 'Female'], [0,1])
        df['col44'] = df['col44'].replace(['No' , 'Yes'], [0,1])
        df['col56'] = df['col56'].replace(['Low' , 'Medium' , 'High'] , [0,1,2])
    else:
        df= pd.get_dummies(df, prefix_sep='_')
df_enc = df.copy()
test_enc = test.copy()
encoding(param = 'Label' , df = df_enc)
encoding(param = 'Label', df = test_enc)
df_enc.head()
df_enc.info()
import seaborn as sns
f, ax = plt.subplots(figsize=(100, 100))
corr = df_enc.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax, annot = True);
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
def scaling(param, df):
    if param == 'minmax':
        scaler = MinMaxScaler()
    elif param == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    df = scaler.fit_transform(df)
    return df
        
y = df['Class']
X_df_enc = df_enc.drop(['ID', 'Class'], axis = 1)
X_test_enc = test_enc.drop(['ID'], axis = 1)
X_df_enc.head()
X_full_sca = pd.concat([X_df_enc.copy(), X_test_enc.copy()])
X_full_sca = scaling(param = 'robust', df = X_full_sca)
X_full = pd.DataFrame(X_full_sca)

X_train_full = X_full[0:700]
X_test_full = X_full[700:]

X_train, X_test, y_train, y_test = train_test_split(X_train_full, y, test_size=0.20,random_state=42)

from sklearn.ensemble import RandomForestClassifier
score_train_RF = []
score_test_RF = []
for i in range(5,20,1):
    rf = RandomForestClassifier(n_estimators = 100, max_depth=i, random_state = 42)
    rf.fit(X_train, y_train)
    sc_train = rf.score(X_train,y_train)
    score_train_RF.append(sc_train)
    sc_test = rf.score(X_test,y_test)
    score_test_RF.append(sc_test)
    
plt.figure(figsize=(15,6))
train_score =plt.plot(range(5,20,1),score_train_RF,color='blue', linestyle='dashed', marker='o', markerfacecolor='green', markersize=5)

test_score =plt.plot(range(5,20,1),score_test_RF,color='red',linestyle='dashed', marker='o', markerfacecolor='blue', markersize=5)

plt.legend( [train_score,test_score],["Train Score","Test Score"])
plt.title('Fig4. Score vs. No. of Trees')
plt.xlabel('No. of Trees')
plt.ylabel('Score')
score_test_RF[np.argmax(score_test_RF)]
clf = RandomForestClassifier(random_state = 42, n_estimators = 100)

clf.fit(X_train_full, y)
y_pred = clf.predict(X_test_full)
sub = pd.DataFrame()
sub['ID'] = test['ID']
sub['Class'] = y_pred
#sub.to_csv('sub19.csv', index = False)

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
create_download_link(sub)
