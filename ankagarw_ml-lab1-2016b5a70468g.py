import numpy as np

import pandas as pd

data = pd.read_csv("../input/bits-f464-l1/train.csv") 

y=data['label']

x=data.drop(labels=['label','id'],axis=1)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor





X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.1, random_state=42)

regressor= AdaBoostRegressor(base_estimator=RandomForestRegressor(max_depth=30,n_estimators=10),n_estimators=30).fit(X_train,y_train)###Just Change this line to change regressor

y_pred = regressor.predict(X_test)
test = pd.read_csv("../input/bits-f464-l1/test.csv")  #read 

t_ids=test['id']

x_test=test.drop(labels=['id'],axis=1)

y_pred= regressor.predict(x_test)
ans=pd.DataFrame(data={'id':t_ids,'label':y_pred})

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

 csv = ans.to_csv(index=False)

 b64 = base64.b64encode(csv.encode())

 payload = b64.decode()

 html = '<a download="sub.csv" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

 html = html.format(payload=payload,title=title,filename=filename)

 return HTML(html)

create_download_link("sub.csv")