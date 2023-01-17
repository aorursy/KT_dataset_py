import numpy as np # linear algebra
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score

from sklearn.neural_network import MLPClassifier

from sklearn import svm
df = pd.read_csv("../input/crop.csv")
#print(df.head(3))
quality = df["quality"].values
#print(quality)
category = []
for num in quality:
    if num<5:
        category.append("Low")
    elif num>6:
        category.append("High")
    else:
        category.append("Average")
print(category)
category = pd.DataFrame(data=category, columns=["category"])
data = pd.concat([df,category],axis=1)
data.drop(columns="quality",axis=1,inplace=True)

print(data.head(3))
print(data["category"].value_counts())
X= data.iloc[:,:-1].values
y= data.iloc[:,-1].values

print(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=0)
#rfc=MLPClassifier(hidden_layer_sizes=(8, 2))
rfc = svm.LinearSVC(max_iter=200)
#rfc = RandomForestClassifier(n_estimators=250)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
print("our prediction",pred_rfc)
print("Available prediction",y_test)

count=0
for num in range(len(pred_rfc)):
    if(pred_rfc[num]==y_test[num]):
        count=count+1
#print(count)
accuracy=count/len(pred_rfc)
    
print(accuracy*100)

