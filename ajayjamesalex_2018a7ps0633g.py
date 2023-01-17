import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

%matplotlib inline
df_train = pd.read_csv('../input/minor-project-2020/train.csv')
df_test = pd.read_csv('../input/minor-project-2020/test.csv')
#df_train.head()
#Feature Selection:
#import pandas as pd
#import numpy as np
#data = pd.read_csv("../input/ml-minor-project/train.csv")
#X = data.iloc[:,0:88]  #independent columns
#y = data.iloc[:,-1]    #target column i.e price range
#from sklearn.ensemble import ExtraTreesClassifier
#import matplotlib.pyplot as plt
#model = ExtraTreesClassifier()
#model.fit(X,y)
#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
#feat_importances = pd.Series(model.feature_importances_, index=X.columns)
#print(feat_importances.nlargest(50))
X_train=df_train.drop(['id','target'],axis=1)
Y_train=df_train['target']
X_test=df_test.drop(['id'],axis=1)
from sklearn.preprocessing import Normalizer
scalar = Normalizer()
scaled_X_train = scalar.fit_transform(X_train)
scaled_X_test = scalar.transform(X_test)
#from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
#dt = DecisionTreeClassifier()
#dt.fit(scaled_X_train, y_train)
#y_pred = dt.predict(X_test)
#from sklearn.ensemble import RandomForestClassifier
#clf=RandomForestClassifier(n_estimators=200)
#clf.fit(X_train,y_train)
#rf_y_pred=clf.predict(X_test)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(scaled_X_train,Y_train)
y_pred = logreg.predict_proba(scaled_X_test)[:,1]
df=pd.DataFrame()
df['id']=df_test['id']
df['target']=y_pred
df.to_csv('4_file.csv',index=False)