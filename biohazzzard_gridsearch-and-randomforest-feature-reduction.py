import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
data = pd.read_csv('../input/fashion-mnist_train.csv')
data.info()
def plot_image(im_num):
    im = data.loc[im_num][1:].values.reshape(28,28)
    plt.imshow(im)
    plt.title("Label:{}".format(data.loc[im_num][0]))


plt.subplots_adjust(hspace=.8, wspace=.4)   
rn = np.random.randint(0,59999, 9)
for i, r in enumerate(rn):
    plt.subplot(3,3,i+1)
    plot_image(r)
sns.countplot(data['label'])
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()
train_x, test_x, train_y, test_y = train_test_split(data.loc[:,data.columns != 'label'], data.loc[:,data.columns=='label'], test_size=0.3)


params = {"criterion":["gini", "entropy"], "max_depth":[None, 1,2,3], "n_estimators":[5,10,20]}
rf_gs = GridSearchCV(rf, params)
rf_gs.fit(train_x, train_y.values.ravel())
rf_best = rf_gs.best_estimator_

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix

cv_predict = cross_val_predict(rf_best, train_x, train_y.values.ravel())
cm = confusion_matrix(train_y, cv_predict)
sns.heatmap(cm)


accuracy = (cv_predict == train_y.values.ravel()).astype('int').mean()
print('cross validation accuracy is: ', accuracy)
sns.heatmap(rf_best.feature_importances_.reshape(28,28))
test_predict = rf_best.predict(test_x)
test_accuracy = (test_predict == test_y.values.ravel()).astype('int').mean()
print('test accuracy is: ', test_accuracy)
from sklearn.base import clone

best_features = (-rf_best.feature_importances_.ravel()).argsort()
acc_list = []
rf = clone(rf_best)

for i in range(1,int((len(best_features)/2))):
    temp_df = train_x.iloc[:, best_features[:(i*2)]]
    rf.fit(temp_df, train_y.values.ravel())
    acc_list.append((rf.predict(test_x.iloc[:, best_features[:(i*2)]]) == test_y.values.ravel()).astype('int').mean())
acc_list = np.array(acc_list)
max_acc = acc_list[acc_list.argmax()]


x_axis = ((np.arange(int(len(best_features)/2)) * 2))[:391]
plt.figure(figsize=(12,8))
plt.plot(x_axis, acc_list)
plt.plot([0,800], [max_acc, max_acc], 'r--')
plt.title("Accuracy as function of number of pixels")
plt.xlabel("Accuracy")
plt.ylabel("Number of most important pixels")


plt.figure(figsize=(12,8))
plt.plot(rf_best.feature_importances_[best_features], 'r.')
plt.xlabel("Feature (pixel)")
plt.ylabel("Feature importance")
plt.title("Importance of Features (pixels)")

