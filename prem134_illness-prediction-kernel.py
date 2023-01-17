import pandas as pd

df = pd.read_csv('../input/toy-dataset/toy_dataset.csv')
df.shape
df.head()
df.drop('Number', axis=1, inplace=True )
df.isna().sum(axis=0)
df['City'].value_counts()
df.loc[df['Gender']=='Male','Gender'] = 1
df.loc[df['Gender']=='Female','Gender'] = 0
from sklearn.preprocessing import LabelBinarizer



lb = LabelBinarizer()

lb_results = lb.fit_transform(df['City'])

new_df = pd.DataFrame(lb_results, columns=lb.classes_)
df = pd.concat([df, new_df], axis = 1)
df.loc[df['Illness'] == 'Yes','Illness'] = 1
df.loc[df['Illness'] == 'No','Illness'] = 0
df.drop(['City'], axis=1, inplace=True)

df.head()
y = df['Illness']

df.drop(['Illness'], axis=1, inplace=True)

X = df
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)

from sklearn.linear_model import LogisticRegression

lm = LogisticRegression()

model = lm.fit(X_train, y_train)

predictions = model.predict(X_test)
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt



conf_mat = confusion_matrix(y_true=y_test, y_pred=predictions)

print('Confusion matrix:\n', conf_mat)



labels = ['Class 0', 'Class 1']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
from sklearn.utils import resample



# setting up testing and training sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)



# concatenate our training data back together

X = pd.concat([X_train, y_train], axis=1)



# separate minority and majority classes

not_ill = X[X.Illness==0]

ill = X[X.Illness==1]

# upsample minority

ill_upsampled = resample(ill,

                          replace=True, # sample with replacement

                          n_samples=len(not_ill), # match number in majority class

                          random_state=27) # reproducible results



# combine majority and upsampled minority

upsampled = pd.concat([not_ill, ill_upsampled])



# check new class counts

upsampled.Illness.value_counts()
df_oversample = upsampled

y_train = upsampled.Illness

X_train=upsampled.drop(['Illness'], axis=1)





from sklearn.ensemble import RandomForestClassifier

lm = RandomForestClassifier(n_estimators = 100, oob_score = True,n_jobs = 1,random_state =50)

model = lm.fit(X_train, y_train)

predictions=model.predict(X_test)



from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score

from matplotlib import pyplot

print(accuracy_score(y_test, predictions))



# predict probabilities

probs = model.predict_proba(X_test)

# keep probabilities for the positive outcome only

probs = probs[:, 1]

# calculate AUC

auc = roc_auc_score(y_test, probs)

print('AUC: %.3f' % auc)

# calculate roc curve

fpr, tpr, thresholds = roc_curve(y_test, probs)

# plot no skill

pyplot.plot([0, 1], [0, 1], linestyle='--')

# plot the roc curve for the model

pyplot.plot(fpr, tpr, marker='.')

# show the plot

pyplot.show()

from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt



conf_mat = confusion_matrix(y_true=y_test, y_pred=predictions)

print('Confusion matrix:\n', conf_mat)



labels = ['Class 0', 'Class 1']

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)

fig.colorbar(cax)

ax.set_xticklabels([''] + labels)

ax.set_yticklabels([''] + labels)

plt.xlabel('Predicted')

plt.ylabel('Expected')

plt.show()
# plot feature importance using built-in function

from numpy import loadtxt

from xgboost import XGBClassifier

from xgboost import plot_importance

from matplotlib import pyplot



xgbmodel = XGBClassifier()

xgbmodel.fit(X_train, y_train)

feat_importances = pd.Series(xgbmodel.feature_importances_, index=X_train.columns)

feat_importances.nlargest(10).plot(kind='barh')