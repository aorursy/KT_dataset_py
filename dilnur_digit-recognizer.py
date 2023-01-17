import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
x = train.iloc[0 : 5000 , 1:]
y = train.iloc[0 : 5000 , :1]
from sklearn.model_selection import train_test_split
train_x , validate_x , train_y , validate_y = train_test_split(x , y , test_size = 0.3)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(criterion='gini', n_estimators=100, min_samples_split=10, max_features='auto',
                             oob_score=True, random_state=1, n_jobs=-1)
model.fit(train_x , train_y)
from sklearn.metrics import accuracy_score
validate_prediction = model.predict(validate_x)
acc = accuracy_score(validate_y , validate_prediction)
acc
test_pred = model.predict(test)
example = pd.DataFrame({"IdImg" : test.index.values + 1 , 'Predicted_level' : test_pred})
example.to_csv("out.csv" , index = False)
mydata = pd.read_csv("out.csv")
img = test.iloc[21].as_matrix().reshape((28,28))
plt.imshow(img,cmap="binary")
plt.title(mydata.iloc[21,1])


