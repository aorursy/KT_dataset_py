import pandas as pd
heart_data = pd.read_csv("../input/heart-disease-uci/heart.csv")
heart_data.head(50)
heart_data.info()
heart_data.describe()
heart_data[heart_data["target"]==0][0:15]
heart_data[heart_data["target"]==1][0:15]
heart_data.shape
heart_data.corr()
heart_data["target"].value_counts()
import matplotlib.pyplot as plt 
%matplotlib inline
have =heart_data[heart_data["target"]==0]

plt.hist(have["age"])
plt.title("person with no heart disease")
plt.xlabel("age")
plt.ylabel("persons")
have1 =heart_data[heart_data["target"]==1]
plt.hist(have1["age"])
plt.title("person have disease")
plt.xlabel("age")
plt.ylabel("persons")
plt.hist(heart_data["sex"])
plt.xticks([0,1])
import seaborn as sns
sns.scatterplot(heart_data["age"],heart_data["chol"],hue=heart_data["target"])
plt.show()
from sklearn.model_selection import train_test_split

train_data =heart_data.drop("target",axis=1)
target_data=heart_data["target"]

train_feature,test_feature,train_target,test_target=train_test_split(train_data,target_data,
                                                                     test_size=0.3, 
                                                                    random_state =0)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=0)
model.fit(train_feature,train_target)
prediction=model.predict(test_feature)
from sklearn.metrics import accuracy_score
accuracy_score(test_target,prediction)
from sklearn.metrics import roc_curve
pred=model.predict_proba(test_feature)[:,1]
fpr,tpr,threholds =roc_curve(test_target,pred)
plt.plot([0,1],[0,1],"k--")
plt.plot(fpr,tpr,label="logistic")
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
