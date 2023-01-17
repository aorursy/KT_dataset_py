import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

le = LabelEncoder()
mushroom_filepath = "../input/mushroom-classification/mushrooms.csv"
mushroom_data = pd.read_csv(mushroom_filepath)
mushroom_data['class'] = le.fit_transform(mushroom_data['class'])
x = mushroom_data.drop(columns=['class'])
x = pd.get_dummies(x)
y = mushroom_data['class']
train_x, val_x, train_y, val_y = train_test_split(x, y, test_size = 0.2, random_state = 0)
model = RandomForestClassifier()
model.fit(train_x,train_y)
preds = model.predict(val_x)
tn, fp, fn, tp = confusion_matrix(val_y,preds).ravel()
print(tn, fp, fn, tp)
acc = (tn+tp)/(tn+tp+fn+fp)
print(acc)