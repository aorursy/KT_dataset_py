import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
%matplotlib inline
data = pd.read_csv("https://raw.githubusercontent.com/aniket-spidey/bitgrit-webinar/master/code/datasets/Iris.csv")
data = data.dropna(axis=0)
data.head()
y = data['Species']
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = data[features]
def buildModel(train_X, train_y, random):
    model = DecisionTreeClassifier(random_state=random)
    model.fit(train_X, train_y)
    return model
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
dt_model = buildModel(train_X, train_y, 4)
prediction_y = dt_model.predict(val_X)

print(accuracy_score(val_y, prediction_y) * 100)
X_plot = []
y_plot = []
for random in range(1, 101):
    X_plot.append(random)
    
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = random)
    dt_model = buildModel(train_X, train_y, random)
    prediction_y = dt_model.predict(val_X)
    score = accuracy_score(val_y, prediction_y) * 100
    y_plot.append(score)
    
sns.lineplot(X_plot, y_plot)
