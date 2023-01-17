import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("../input/cancer.csv") 
data.head()
title_mapping={"?":0,"1":1,"2":2,"3":3,"4":4,
               "5":5,"6":6,"7":7,"8":8,"9":9,"10":10}
data['Bare Nuclei']=data['Bare Nuclei'].map(title_mapping)
res = data['diagnosis']

data_new = data.drop(['diagnosis', 'Sample code number'], axis=1)

x_train,x_test,y_train,y_test=train_test_split(data_new,res,
                                                                test_size=0.2,
                                                                random_state=0)
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
pca = PCA(.50)
pca.fit(x_train)

x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
clf=RandomForestClassifier(n_estimators=20,min_samples_split=20,
                           max_depth=5,random_state=28)
clf.fit(x_train,y_train)
pred = clf.predict(x_test)
ac = accuracy_score(y_test,pred)
print("Random Forest Accuracy:",ac)
cm = confusion_matrix(y_test,clf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")