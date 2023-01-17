import pandas as pd
import numpy as np
data = pd.read_csv("../input/Social_Network_Ads.csv")
data.head()
data.drop("User ID",axis=1,inplace=True)
x = data.iloc[:,0:4].values
y = data.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_x=LabelEncoder()
x[:,0]=labelEncoder_x.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()
x = np.delete(x,0,1)
x

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
from sklearn.preprocessing import StandardScaler
s_x = StandardScaler()
s_x.fit(x_train)
x_train = s_x.transform(x_train)
x_test = s_x.transform(x_test)
from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression(random_state=0)
lrc.fit(x_train,y_train)
y_pred = lrc.predict(x_test)
lrc.score(x_test,y_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm





