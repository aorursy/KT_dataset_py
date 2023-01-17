import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pandas as pd
df = pd.read_csv('../input/pid-5M.csv')
df.head()
#The id means: positron (-11), pion (211), kaon (321), and proton (2212)
#p is momentum (GeV/c)
#theta and beta are angles (rad)
#nphe is the number of photoelectons
#ein is the inner energy (GeV)
#eout is the outer energy (GeV)
df.describe()
df.shape #number of rows and columns
sns.set(style='darkgrid')
sns.distplot(df['p'], hist=True, kde=True, color='c')
plt.xlabel('Momentum of Particles')
plt.ylabel('Feature Value')
plt.title('Momentum Distribution')
#correlation heat map
sns.set(style='darkgrid')
corr = df[['id', 'p','ein', 'eout','nphe', 'theta', 'beta']].corr()
sns.heatmap(corr)
f1 = df['p'].values
f2 = df['beta'].values
plt.scatter(f1, f2, c='black', s=7)
plt.xlabel('Momentum of the Particle')
plt.ylabel('beta angle')

df.isnull().sum() #there are no null values
features = df.drop('id', axis=1)
labels = df['id']
#test and train split using sklearn.model_selection
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.22, random_state = 1)
y_train.unique()
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(x_train, y_train)
pred_sgd = clf.predict(x_test)
print('accuracy score:', accuracy_score(y_test, pred_sgd))
from sklearn.ensemble import AdaBoostClassifier
clf_abc = AdaBoostClassifier()
clf_abc.fit(x_train, y_train)
pred_abc = clf_abc.predict(x_test)
print('accuracy score:', accuracy_score(y_test, pred_abc))
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
pred_xgb = xgb.predict(x_test)
print('accuracy score:', accuracy_score(y_test, pred_xgb))
from sklearn.ensemble import RandomForestClassifier
clf_rfc = RandomForestClassifier()
clf_rfc.fit(x_train, y_train)
pred_rfc = clf_rfc.predict(x_test)
print('accuracy score:', accuracy_score(y_test, pred_rfc))