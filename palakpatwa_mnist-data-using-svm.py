import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df
df.shape
df.info()
df.describe()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

x = df.drop('label', 1)
y = df['label']

x = scale.fit_transform(x)
x= pd.DataFrame(x)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.1, test_size = 0.9, random_state= 100)
x_test
x_train
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
sns.distplot(y_train)
y_test.mean()
y_train.mean()
model1 = SVC(kernel= 'rbf')
model1.fit(x_train, y_train)
predictions = model1.predict(x_test)
from sklearn.metrics import accuracy_score 
print(accuracy_score(y_true=y_test, y_pred=predictions))
params = {'C' : [0.1, 1, 10], 'gamma': [1e-2, 1e-3, 1e-4]}

svc = SVC(kernel = 'rbf')

model = GridSearchCV(param_grid= params, estimator= svc, verbose =1,
                    return_train_score = True, scoring = 'accuracy')
model.fit(x_train, y_train)
res = pd.DataFrame(model.cv_results_)
res.head()
plt.figure(figsize= (18,8))

plt.subplot(131)

res['param_C'] = res['param_C'].astype('int')
gamma_01 = res[res['param_gamma']==0.01]

plt.plot(gamma_01.param_C , gamma_01.mean_train_score)
plt.plot(gamma_01.param_C , gamma_01.mean_test_score)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='lower right')
plt.xscale('log')


plt.subplot(132)
gamma_001 = res[res['param_gamma']==0.001]

plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='lower right')
plt.xscale('log')


# subplot 3/3
plt.subplot(133)
gamma_0001 = res[res['param_gamma']==0.0001]

plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])
plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.0001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='lower right')
plt.xscale('log')

plt.show()
model.best_score_
model.best_params_
best_C = 10
best_gamma = 0.001

svm_final = SVC(kernel='rbf', C=best_C, gamma=best_gamma)

svm_final.fit(x_train, y_train)
predictions = svm_final.predict(x_test)
confusion = confusion_matrix(y_true = y_test, y_pred = predictions)
test_accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
confusion
test_accuracy

