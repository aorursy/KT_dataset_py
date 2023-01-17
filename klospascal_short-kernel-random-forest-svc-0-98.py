import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
print("Dataset ready")
for i in df_test.isnull().any():
    if i == True:
        sys.exit("There is unlabeled data")
print("No missing values found")
for i in df_train.isnull().any():
    if i == True:
        sys.exit("There is unlabeled data")
print("No missing values found")
y = df_train["label"]
x = df_train.drop("label", axis=1)
x.shape
df_test.shape
x = x / 255.0
df_test = df_test / 255.0

print("Completed")
plt.imshow(df_test.values.reshape(-1, 28, 28, 1)[40][:,:,0])
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state=2)

print("Completed")
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(criterion = "entropy")
model.fit(x_train, y_train)

print(model.score(x_test, y_test))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

forrest_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(1, 5)],     
    n_estimators = [n for n in range(10, 60, 10)],
)

forrest = RandomForestClassifier(criterion = "entropy")
forest_cv = GridSearchCV(estimator=forrest, param_grid=forrest_params, cv=5) 
forest_cv.fit(x_train.values[: 100, :], y_train.values[: 100])
print(forest_cv.best_score_)
print(forest_cv.best_estimator_)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(criterion = "entropy", max_depth=12, min_samples_leaf=1, min_samples_split=6, n_estimators=50)
model.fit(x_train, y_train.values)

print(model.score(x_test, y_test))
#Learning Curve
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle

x, y = shuffle(x, y)

train_sizes_abs, train_scores, test_scores = learning_curve(RandomForestClassifier(criterion = "entropy", n_estimators = 50), x, y)
plt.plot(train_sizes_abs, np.mean(train_scores, axis = 1))
plt.plot(train_sizes_abs, np.mean(test_scores, axis = 1))
from sklearn.svm import SVC

model = SVC(kernel = "rbf", gamma = 0.021, C = 2.1)
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
pred = model.predict(df_test)

submission = pd.DataFrame({
    "ImageId": list(range(1,len(pred)+1)),
    "Label": pred
})

submission.to_csv("submision.csv", index=False)

print("Prediction Completed")
