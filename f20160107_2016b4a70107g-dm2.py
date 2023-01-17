import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 100)
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
train=pd.read_csv("../input/data-mining-assignment-2/train.csv")
test=pd.read_csv("../input/data-mining-assignment-2/test.csv")
train.head()
train.info()
print (train.isnull().any(axis = 0))
train.replace('?', np.NaN, inplace = True)
train.fillna(train.mean(), inplace = True)
train.isnull().sum()
#Id is randomly generated so we can remove it
train.drop(['ID'],axis=1,inplace=True)
df_dtype_nunique = pd.concat([train.dtypes, train.nunique()],axis=1)
df_dtype_nunique.columns = ["dtype","unique"]
df_dtype_nunique
categorical_features=['col2','col11','col37','col44','col56']
columns=list(train)
numerical_features=[]
for i in columns:
  numerical_features.append(i)

numerical_features.remove('col2')
numerical_features.remove('col11')
numerical_features.remove('col37')
numerical_features.remove('col44')
numerical_features.remove('col56')
numerical_features.remove('Class')
numerical_features
train=pd.get_dummies(data=train,columns=categorical_features)
X=train.loc[:,train.columns != 'Class']
y=train["Class"]
X.loc[:,"col0":"col56_Medium"]=StandardScaler().fit_transform(X.loc[:,"col0":"col56_Medium"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=4000)
rf.fit(X_train,y_train)
rf.score(X_test, y_test)
test.drop(['ID'],axis=1,inplace=True)
test=pd.get_dummies(data=test,columns=categorical_features)
test.loc[:,"col0":"col56_Medium"]=StandardScaler().fit_transform(test.loc[:,"col0":"col56_Medium"])
test_class=rf.predict(test)
test_class
df=pd.read_csv("../input/data-mining-assignment-2/test.csv")
submission = pd.DataFrame({'ID':df['ID'],'Class':test_class})
submission['Class']=submission['Class'].astype("int")
submission['Class'].dtype
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "answer.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(submission)