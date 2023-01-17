import pandas as pd

import seaborn as sns

data = pd.read_csv('../input/lower-back-pain-symptoms-dataset/Dataset_spine.csv')
data.head()
data.tail()
data.info()
data.shape
del data["Unnamed: 13"]

data.describe()

data.rename(columns = {

    "Col1" : "pelvic_incidence", 

    "Col2" : "pelvic_tilt",

    "Col3" : "lumbar_lordosis_angle",

    "Col4" : "sacral_slope", 

    "Col5" : "pelvic_radius",

    "Col6" : "degree_spondylolisthesis", 

    "Col7" : "pelvic_slope",

    "Col8" : "direct_tilt",

    "Col9" : "thoracic_slope", 

    "Col10" :"cervical_tilt", 

    "Col11" : "sacrum_angle",

    "Col12" : "scoliosis_slope", 

    "Class_att" : "class"}, inplace=True)
data.head()

data.shape
data["class"].value_counts().sort_index().plot.barh()

data.corr()
sns.heatmap(data[data.columns[0:13]].corr(),annot=True,cmap='viridis',square=True, vmax=1.0, vmin=-1.0, linewidths=0.2)
X = data.iloc[:,:-1].values

y = data.iloc[:,-1].values
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = ss.fit_transform(X)
from sklearn.model_selection import train_test_split,cross_val_score

X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size=0.2,random_state=47)
print(X_train.shape, y_test.shape)

print(y_train.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,auc



lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test,y_pred)



plt.figure(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=.3)

plt.show()



print(classification_report(y_test,y_pred))