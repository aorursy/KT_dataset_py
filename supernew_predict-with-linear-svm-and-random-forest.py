import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('../input/train.csv')
df.head()
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
X.shape
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# linear svm
from sklearn.preprocessing import MinMaxScaler, Binarizer, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
mm = MinMaxScaler()
pipe = Pipeline([('mm', mm),
                ('clf', SVC(kernel='linear',random_state=1))])
pipe.fit(X_train, y_train)
print("score:{}".format(pipe.score(X_test, y_test)))
# random_forest
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
mm = MinMaxScaler()
pipe_forest = Pipeline([('mm', mm),
                        ('clf', RandomForestClassifier(n_estimators=1000,random_state=1,criterion='entropy'))])
pipe_forest.fit(X_train, y_train)
print("score:{}".format(pipe_forest.score(X_test, y_test)))
# save model to local file
import pickle
pickle.dump(pipe_forest, open('random_forest_model', 'wb'), protocol=4)
import pickle
p_fs = pickle.load(open('random_forest_model', 'rb'))
p_fs.score(X_test, y_test)
y_pred = p_fs.predict(X_test)
# plot the figure whose prediction is wrong
X_wrong = X_test[y_pred!=y_test][:25]
y_pred_wrong = y_pred[y_pred!=y_test][:25]
y_pred_true = y_test[y_pred!=y_test][:25]
# plot number with wrong prediction
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = X_wrong[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title('%d) t: %d p: %d' % (i+1, y_pred_true[i], y_pred_wrong[i]))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
# predict the test.csv and save the result to local file
df_x= pd.read_csv('../input/test.csv')
x = df_x.values
y_pred = p_fs.predict(x)
df_result = pd.DataFrame({'ImageId': list(range(1, len(y_pred)+1)), 'Label': y_pred})
df_result.to_csv('../data/result.csv',index=None)