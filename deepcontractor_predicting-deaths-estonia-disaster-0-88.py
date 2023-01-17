import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import pandas as pd
import numpy as np
from colorama import Fore, Back, Style 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import xgboost
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from statsmodels.formula.api import ols
import plotly.graph_objs as gobj

init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
import plotly.figure_factory as ff

%matplotlib inline

from sklearn.ensemble import GradientBoostingClassifier
import lightgbm
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
data.head()
data.describe()
data.isnull().sum()
data.corr()
sns.distplot(data['Age'])
sns.countplot(data['Sex'])
plt.figure(figsize=(10,8))
sns.countplot(data = data, y = 'Country')
fig = px.violin(data, y="Age", x="Sex", color="Survived", box=True, points="all",
          hover_data=data.columns)
fig.update_layout()
fig.show()
from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 
  
data['Sex']= le.fit_transform(data['Sex']) 
data['Category']= le.fit_transform(data['Category']) 
corr = data.corr()
fig = px.imshow(corr)
fig.show()
x=data.loc[:,["Sex","Age","Category"]]
y=data.loc[:,["Survived"]]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2)
# RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=12)
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
clf.score(x_test,y_test)
gradientboost_clf = GradientBoostingClassifier(max_depth=2, random_state=1)
gradientboost_clf.fit(x_train,y_train)
gradientboost_pred = gradientboost_clf.predict(x_test)
gradientboost_clf.score(x_test,y_test)
xgb_clf = xgboost.XGBRFClassifier(max_depth=2, random_state=1)
xgb_clf.fit(x_train,y_train)
xgb_pred = xgb_clf.predict(x_test)
xgb_clf.score(x_test,y_test)
lgb_clf = lightgbm.LGBMClassifier(max_depth=2, random_state=1)
lgb_clf.fit(x_train,y_train)
lgb_pred = lgb_clf.predict(x_test)
lgb_clf.score(x_test,y_test)
