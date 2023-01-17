import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import linear_model, dummy, metrics

from sklearn.linear_model import LogisticRegression

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import *

from sklearn.preprocessing import *



import time

from inspect import signature

from google.colab import files

uploaded = files.upload()
# # Data Exploration

df_test = pd.read_csv('bank-test.csv')

df_train = pd.read_csv('bank-train.csv')



df_test.head()

df_list = [df_train,df_test]

print(df_train.columns)

for df in df_list:

  # df.replace('unknown', np.nan, inplace=True)

  df.replace({'YES': 1, 'NO': 0}, inplace=True)



  # due to irrelevancy

  # df.drop(['pdays','contact','poutcome','previous'],inplace=True, axis=1)

  # due to multicollinearity or no association

  # df.drop(['euribor3m','nr.employed'],inplace=True, axis=1)

  # due to impracticality

  df.drop(['duration','default'],inplace=True, axis=1)

  # df.drop(['month','day_of_week'],inplace=True, axis=1)

  

  # df.dropna(inplace=True, axis=0)

  

# print(df_train.info())

print(df_train.y.value_counts()/len(df_train.y))



print(list(df_train.columns))

# for feature in list(df_train.columns.values):

#   print('\n'+str(feature)+'\n{}'.format(df_train[str(feature)].value_counts()))



# plot corr map

cm = df_train.corr()

plt.figure(figsize=(10,10))

def r_squared(x):

  return x**2



cm2 = cm.apply(r_squared)

cmap = sns.diverging_palette(240, 10, sep=20, as_cmap=True)

sns.heatmap(cm2,vmin=-1,vmax=1,fmt='.3f',cmap=cmap,square=True,annot=True)
# plt.figure(figsize=(20,5))





# ax1 = plt.add_subplot(411)

# month_order = ['jan','feb','mar','apr','may','jun','jul','aug','sept','oct','nov','dec']

# day_order = ['mon','tue','wed','thu','fri']

# sns.catplot(x='month', hue='y', kind="count", order=month_order,data=df_train)



# ax2 = plt.subplot(412)

# sns.catplot(x='day_of_week', hue='y', kind="count", order=day_order,data=df_train)





# edu_order = ['illiterate','basic.4y','basic.6y','basic.9y','high.school','professional.course','university.degree']

# plt.subplot(4,1,3)

# sns.catplot(x='education', hue='y', kind="count", order=edu_order, data=df_train)

# plt.show()



# plt.subplot(4,1,4)

# table=pd.crosstab(df_train.month,df_train.y)

# table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)



sns.distplot(df_train['euribor3m'])

# sns.scatterplot(x='euribor3m',y='y',data=df_train)

print(df.loc[df["euribor3m"] < 3, 'y'].mean())

print(df.loc[df["euribor3m"] > 3, 'y'].mean())



# plt.show()

# encoding vars

y = df_train.y

df_train.drop('y',inplace=True,axis=1)

# df_train.drop('id',inplace=True,axis=1)

# df_test.drop('id',inplace=True,axis=1)



# from sklearn.experimental import enable_iterative_imputer

# from sklearn.impute import IterativeImputer

# from sklearn.impute import SimpleImputer



# imp = SimpleImputer(missing_values = np.nan, strategy='most_frequent')

# df_train = imp.fit_transform(df_train)



for df in df_list:

#   df.loc[df["job"] == 'unknown',"job"] = df["job"].value_counts().index[0]

#   df.loc[df["marital"] == 'unknown',"marital"] = df["marital"].value_counts().index[0]

#   df.loc[df.education == 'unknown',"education"] = df.education.value_counts().index[0]

#   df.loc[df.housing == 'unknown',"housing"] = df.housing.value_counts().index[0]

#   df.loc[df.loan == 'unknown',"loan"] = df.loan.value_counts().index[0]



  edu_order

  edu_dict = dict(zip(edu_order, range(0,7)))

  df['education'] = df['education'].map(edu_dict).fillna(6,inplace=True)  



#   df.loc[df["euribor3m"] < 3, 'euribor3m'] = 0

#   df.loc[df["euribor3m"] > 3, 'euribor3m'] = 1

  

  

X = pd.get_dummies(df_train.iloc[:,1:-1], drop_first=False, dummy_na=False)       # exclues id & y

X_sub = pd.get_dummies(df_test.iloc[:,1:-1], drop_first=False, dummy_na=False)





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



print(X_test.shape)

print(X_sub.shape)



X_train, y_train = resample(method=None)



# checking dimensions

a = list(X.columns.values)

b = list(X_sub.columns.values)

for i in a:

  if i not in b:

    print(i)
def resample(method=None):

  if method == 'oversample':

    from imblearn.over_sampling import SMOTE



    os = SMOTE(random_state=42)

    columns = X_train.columns

    os_data_X,os_data_y=os.fit_resample(X_train, y_train)

    os_data_X = pd.DataFrame(data=os_data_X,columns=columns )

    os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])

    return os_data_x, os_data_y

          

  elif method == 'undersample':

    from imblearn.under_sampling import RandomUnderSampler

  

    us = RandomUnderSampler(random_state=42)

    columns = X_train.columns

    us_data_X,us_data_y=us.fit_resample(X_train, y_train)

    u_data_X = pd.DataFrame(data=us_data_X,columns=columns )

    us_data_y= pd.DataFrame(data=us_data_y,columns=['y'])

    return us_data_x, us_data_y



  elif method == 'combination':

    from imblearn.combine import SMOTEENN

          

    cs = SMOTEENN(random_state=42)

    columns = X_train.columns

    cs_data_X,cs_data_y=os.fit_resample(X_train, y_train)

    cs_data_X = pd.DataFrame(data=cs_data_X,columns=columns )

    cs_data_y= pd.DataFrame(data=cs_data_y,columns=['y'])

    return cs_data_x, cs_data_y

  

  else:

    return X_train, y_train



          

#   bp = sns.countplot(x=cs_data_y.values.ravel())

#   plt.title("Class Distribution of Train Dataset")

#   plt.show()
from sklearn.svm import SVC



start = time.time()

svm = SVC(kernel="rbf", random_state=42)

svm.fit(X_train,y_train)

end = time.time()



print(start-end)

y_hat = svm.predict(X_test)

# y_pred_proba = svm.predict_proba(X_test)



print(classification_report(y_test, y_hat, digits=4))

print(confusion_matrix(y_test, y_hat))

# metrics(y_hat,y_pred_proba, plot=True)

from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(activation= 'tanh', alpha=0.05, solver='adam', random_state=42)

nn.fit(X_train,np.ravel(y_train))



# parameter_space = {

#     'activation': ['tanh', 'relu'],

#     'solver': ['sgd', 'adam'],

#     'alpha': [0.0001, 0.05],

# }



# start = time.time()



# from sklearn.model_selection import GridSearchCV

# clf = GridSearchCV(nn, parameter_space, n_jobs=-1, cv=3)

# clf.fit(X_train, y_train)



# end = time.time()

# print(end - start)





# print(clf.best_params_,clf.best_score_)



y_hat = nn.predict(X_test)

y_pred_proba = nn.predict_proba(X_test)



metrics(y_hat,y_pred_proba, plot=True)





# y_hat = clf.predict(X_test)

# y_pred_proba = clf.predict_proba(X_test)



# metrics(y_hat,y_pred_proba, plot=False)





# means = clf.cv_results_['mean_test_score']

# stds = clf.cv_results_['std_test_score']

# for mean, std, params in zip(means, stds, clf.cv_results_['params']):

#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# print('Best parameters found:\n', clf.best_params_)





from keras import backend as K



def f1(y_true, y_pred):

    def recall(y_true, y_pred):

        """Recall metric.



        Only computes a batch-wise average of recall.



        Computes the recall, a metric for multi-label classification of

        how many relevant items are selected.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = true_positives / (possible_positives + K.epsilon())

        return recall



    def precision(y_true, y_pred):

        """Precision metric.



        Only computes a batch-wise average of precision.



        Computes the precision, a metric for multi-label classification of

        how many selected items are relevant.

        """

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())

        return precision

    precision = precision(y_true, y_pred)

    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))
from keras.models import Sequential

from keras.layers import Dense





model = Sequential()

# model.add(Dense(10, input_dim=14, activation='tanh'))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])



x_train_keras = np.array(X_train)

y_train_keras = np.array(y_train)

#print(x_train_keras.shape)

y_train_keras = y_train_keras.reshape(y_train_keras.shape[0], 1)



model.fit(np.array(x_train_keras), np.array(y_train_keras), epochs=10, batch_size=128, shuffle=True)



y_hat  = model.predict_proba(X_test)

y_pred = (y_hat>0.5)



y_hat.describe()



metrics(y_pred,y_hat, plot=True)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train,y_train)

y_hat = lr.predict(X_test)

y_pred_proba = lr.predict_proba(X_test)



metrics(y_hat,y_pred_proba, plot=False)





from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=21)

rf.fit(X_train,y_train)



important_features = pd.DataFrame({'Importance': rf.feature_importances_}, index = X_train.columns).sort_values('Importance', ascending = False)

important_features = pd.DataFrame({"features" : important_features.index})



imp_fea_array = important_features.values[:10]

imp_fea_array = imp_fea_array.ravel()

print(imp_fea_array)



y_hat = rf.predict(X_test)

y_pred_proba = rf.predict_proba(X_test)





metrics(y_hat,y_pred_proba, plot=True)



from sklearn.ensemble import GradientBoostingClassifier



gb = GradientBoostingClassifier(random_state=21)

gb.fit(X_train,y_train)



# parameter_space = {

#     'n_estimators': range(80,160,10),

#     'max_depth': range(5,16,3),

#     'criterion':['friedman_mse']

# }





# start = time.time()



# from sklearn.model_selection import GridSearchCV

# clf = GridSearchCV(gb, param_grid = parameter_space, scoring='f1_weighted', n_jobs=-1, iid=False, cv=3)

# clf.fit(X_train, y_train)



# end = time.time()

# print(end - start)
#clf.grid_scores_, clf.best_params_, clf.best_score_



y_hat = gb.predict(X_test)

y_pred_proba = gb.predict_proba(X_test)



metrics(y_hat,y_pred_proba)
def metrics(y_hat, y_pred_proba, plot=True):

  report(y_hat)

  if plot:

    plot_cm(y_hat)

  report_ROC(y_pred_proba,plot=plot)

  report_PRC(y_pred_proba,plot=plot)

  

def plot_cm(y_hat):

  cm = confusion_matrix(y_test, y_hat)

  fig, ax = plt.subplots(figsize = (7,7))

  sns.heatmap(pd.DataFrame(cm.T), annot=True, annot_kws={"size": 15}, vmin=0, vmax=2000, cmap="Purples", fmt='.0f', linewidths=1, linecolor="white", cbar=False,

             xticklabels=["0","1"], yticklabels=["0","1"])

  plt.ylabel("Predicted", fontsize=15)

  plt.xlabel("Actual", fontsize=15)

  ax.set_xticklabels(["0","1"], fontsize=13)

  ax.set_yticklabels(["0","1"], fontsize=13)

  plt.title("Confusion Matrix", fontsize=15)

  plt.show()

  

def report(y_hat):

  report = classification_report(y_test,y_hat,digits=4,output_dict=True)

  

  print(classification_report(y_test,y_hat,digits=4))

  print()

  print("Accuracy = {0:0.3f}".format(report["accuracy"]))

  print("Precision = {0:0.3f}".format(report["1"]["precision"]))

  print("Specificity = {0:0.3f}".format(report["0"]["recall"]))

  print("Sensitivity = {0:0.3f}".format(report["1"]["recall"]))

  print("F1-score = {0:0.3f}".format(report["1"]["f1-score"]))

  

def report_ROC(y_pred_proba,plot=True):

  fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])

  

  if plot:

    fig, ax = plt.subplots(figsize = (10,7))

    plt.plot([0, 1], [0, 1], 'k--')

    plt.plot(fpr, tpr)

    plt.fill_between(fpr, tpr, alpha=0.2, color='b')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC curve: AUC={0:0.3f}'.format(roc_auc_score(y_test,y_pred_proba[:,1])))

    plt.show()

  

  print('AUROC={0:0.3f}'.format(roc_auc_score(y_test,y_pred_proba[:,1])))

  

def report_PRC(y_pred_proba, plot=True):

  precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba[:,1])

  average_precision = average_precision_score(y_test, y_pred_proba[:,1])



  if plot:

    #its a step function so plotting is different 

    fig, ax = plt.subplots(figsize = (10,7))

    step_kwargs = ({'step': 'post'}

                   if 'step' in signature(plt.fill_between).parameters

                   else {})

    plt.step(recall, precision, color='orange', alpha=1,

             where='post')

    plt.fill_between(recall, precision, alpha=0.2, color='orange', **step_kwargs)



    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.ylim([0.0, 1.05])

    plt.xlim([0.0, 1.0])

    plt.title('Precision-Recall curve: Average Precision={0:0.3f}'.format(average_precision))

    plt.show()

    

  print('Average Precision={0:0.3f}'.format(average_precision))

gb.fit(X,y)

y_sub_hat = gb.predict(X_sub)



submission = pd.concat([df_test.id, pd.Series(y_sub_hat)], axis = 1)

submission.columns = ['id', 'Predicted']

submission.to_csv('submission2.csv', index=False)



from google.colab import files

files.download("submission2.csv")