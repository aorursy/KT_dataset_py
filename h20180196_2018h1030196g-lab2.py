import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.preprocessing import StandardScaler
df=pd.read_csv("/kaggle/input/eval-lab-2-f464/train.csv")
# df.info()

# plt.plot(df['chem_1'],df['class'])

sns.regplot(x='chem_4', y='class', data=df)
sns.heatmap(df.corr())
X=df[['chem_0','chem_1','chem_4','chem_6','attribute']]

y=df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()

scaled_data = scaler.fit_transform(X_train)

scale_test=scaler.fit_transform(X_test)
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

n = len(X_train)

X_A = X_train[:4*n//5]

y_A = y_train[:4*n//5]

X_B = X_train[4*n//5:]

y_B = y_train[4*n//5:]
# clf_1 = DecisionTreeClassifier().fit(X_A, y_A)

# y_pred_1 = clf_1.predict(X_B)

# clf_2 = RandomForestClassifier(n_estimators=100).fit(X_A, y_A)

# y_pred_2 = clf_2.predict(X_B)

# clf_3 = GradientBoostingClassifier().fit(X_A, y_A)

# y_pred_3 = clf_3.predict(X_B)

# gbc = GradientBoostingClassifier(n_estimators=167)

# gbc.fit(X_train, y_train)



from sklearn.ensemble import VotingClassifier



estimators = [('rf', RandomForestClassifier(n_estimators= 1000,min_samples_split= 4,

 min_samples_leaf= 5,

 max_features= 'sqrt',

 max_depth= 3,

 bootstrap= True,

 n_jobs=-1)), ('bag', BaggingClassifier()), ('gbc', GradientBoostingClassifier(n_estimators=194))

             , ('knn', KNeighborsClassifier(n_neighbors=2))]



soft_voter = VotingClassifier(estimators=estimators, voting='soft').fit(X_train,y_train)

hard_voter = VotingClassifier(estimators=estimators, voting='hard').fit(X_train,y_train)

soft_acc = accuracy_score(y_test,soft_voter.predict(X_test))

hard_acc = accuracy_score(y_test,hard_voter.predict(X_test))



print("Acc of soft voting classifier:{}".format(soft_acc))

print("Acc of hard voting classifier:{}".format(hard_acc))
# y_pred = bag_clf.predict(X_test)

accuracy_score(y_test,y_pred)
df2=pd.read_csv("/kaggle/input/eval-lab-2-f464/test.csv")

X_new=df2[['chem_0','chem_1','chem_4','chem_6','attribute']]

pred=soft_voter.predict(X_new)

# ans=pred.round().astype(int)

pred
final=pd.DataFrame()

final['id']=df2['id']

final['class']=pred

final.head()

final.to_csv('predictions1.csv',index=False)