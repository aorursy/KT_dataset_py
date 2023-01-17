import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sb
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder,normalize,MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve
import seaborn as sns
import tensorflow as tf

# GPU device Check.
device_name = tf.test.gpu_device_name()
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')
import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # PyTorch use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
# Reading data
train = pd.read_csv('../input/higgs-boson/training.zip')
test = pd.read_csv('../input/higgs-boson/test.zip')
train.head()
test.head()
print(train.columns.values,'\n')
print(test.columns.values)
train = train.drop(['Weight'], axis=1)
print(train['Label'].value_counts())

rcParams['figure.figsize'] = 10,5
sb.barplot(x = train['Label'].value_counts().index, y = train['Label'].value_counts().values)
plt.title('Label counts')
plt.show()
# getting dummy variables column

enc = LabelEncoder()

train['Label'] = enc.fit_transform(train['Label'])
train.head()
y = train["Label"]
X = train
X_test = test
X.set_index(['EventId'],inplace = True)
X_test.set_index(['EventId'],inplace = True)
X = X.drop(['Label'], axis=1)

X.head()
X_test.head()
train.describe()
# #Normalizing

# no = 1

# X["PRI_jet_all_pt"]=((X["PRI_jet_all_pt"]-X["PRI_jet_all_pt"].min())/(X["PRI_jet_all_pt"].max()-X["PRI_jet_all_pt"].min()))*no
# X_test["PRI_jet_all_pt"]=((X_test["PRI_jet_all_pt"]-X_test["PRI_jet_all_pt"].min())/(X_test["PRI_jet_all_pt"].max()-X_test["DER_mass_MMC"].min()))*no

# X["PRI_jet_subleading_pt"]=((X["PRI_jet_subleading_pt"]-X["PRI_jet_subleading_pt"].min())/(X["PRI_jet_subleading_pt"].max()-X["PRI_jet_subleading_pt"].min()))*no
# X_test["PRI_jet_subleading_pt"]=((X_test["PRI_jet_subleading_pt"]-X_test["PRI_jet_subleading_pt"].min())/(X_test["PRI_jet_subleading_pt"].max()-X_test["PRI_jet_subleading_pt"].min()))*no

# X["PRI_jet_leading_pt"]=((X["PRI_jet_leading_pt"]-X["PRI_jet_leading_pt"].min())/(X["PRI_jet_leading_pt"].max()-X["PRI_jet_leading_pt"].min()))*no
# X_test["PRI_jet_leading_pt"]=((X_test["PRI_jet_leading_pt"]-X_test["PRI_jet_leading_pt"].min())/(X_test["PRI_jet_leading_pt"].max()-X_test["PRI_jet_leading_pt"].min()))*no

# X["PRI_met_sumet"]=((X["PRI_met_sumet"]-X["PRI_met_sumet"].min())/(X["PRI_met_sumet"].max()-X["PRI_met_sumet"].min()))*no
# X_test["PRI_met_sumet"]=((X_test["PRI_met_sumet"]-X_test["PRI_met_sumet"].min())/(X_test["PRI_met_sumet"].max()-X_test["PRI_met_sumet"].min()))*no

# X["DER_sum_pt"]=((X["DER_sum_pt"]-X["DER_sum_pt"].min())/(X["DER_sum_pt"].max()-X["DER_sum_pt"].min()))*no
# X_test["DER_sum_pt"]=((X_test["DER_sum_pt"]-X_test["DER_sum_pt"].min())/(X_test["DER_sum_pt"].max()-X_test["DER_sum_pt"].min()))*no

# X["DER_mass_jet_jet"]=((X["DER_mass_jet_jet"]-X["DER_mass_jet_jet"].min())/(X["DER_mass_jet_jet"].max()-X["DER_mass_jet_jet"].min()))*no
# X_test["DER_mass_jet_jet"]=((X_test["DER_mass_jet_jet"]-X_test["DER_mass_jet_jet"].min())/(X_test["DER_mass_jet_jet"].max()-X_test["DER_mass_jet_jet"].min()))*no

# X["DER_pt_h"]=((X["DER_pt_h"]-X["DER_pt_h"].min())/(X["DER_pt_h"].max()-X["DER_pt_h"].min()))*no
# X_test["DER_pt_h"]=((X_test["DER_pt_h"]-X_test["DER_pt_h"].min())/(X_test["DER_pt_h"].max()-X_test["DER_pt_h"].min()))*no

# X["DER_mass_vis"]=((X["DER_mass_vis"]-X["DER_mass_vis"].min())/(X["DER_mass_vis"].max()-X["DER_mass_vis"].min()))*no
# X_test["DER_mass_vis"]=((X_test["DER_mass_vis"]-X_test["DER_mass_vis"].min())/(X_test["DER_mass_vis"].max()-X_test["DER_mass_vis"].min()))*no

# X["DER_mass_transverse_met_lep"]=((X["DER_mass_transverse_met_lep"]-X["DER_mass_transverse_met_lep"].min())/(X["DER_mass_transverse_met_lep"].max()-X["DER_mass_transverse_met_lep"].min()))*no
# X_test["DER_mass_transverse_met_lep"]=((X_test["DER_mass_transverse_met_lep"]-X_test["DER_mass_transverse_met_lep"].min())/(X_test["DER_mass_transverse_met_lep"].max()-X_test["DER_mass_transverse_met_lep"].min()))*no

# X["DER_mass_MMC"]=((X["DER_mass_MMC"]-X["DER_mass_MMC"].min())/(X["DER_mass_MMC"].max()-X["DER_mass_MMC"].min()))*no
# X_test["DER_mass_MMC"]=((X_test["DER_mass_MMC"]-X_test["DER_mass_MMC"].min())/(X_test["DER_mass_MMC"].max()-X_test["DER_mass_MMC"].min()))*no


# X.head()
# # normalize the data attributes
# X = X.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

# X_test = X_test.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))


# X.head()
#Normalizing

from sklearn.preprocessing import normalize

X = normalize(X)
X_test = normalize(X_test)
# print(X.isnull().sum(),'\n')
# print(X_test.isnull().sum())
#X = X.replace(-999.000,np.nan)
#X.head()
#X_test = X_test.replace(-999.000,np.nan)
#X_test.head()
#X = X.replace(-999.000,0)
#X_test = X_test.replace(-999.000,0)
#X.head()
#print(X.isnull().sum(),'\n')
#print(X_test.isnull().sum())
#X.fillna(X.median(), inplace=True)
#X_test.fillna(X_test.median(), inplace=True)

#X.head()
#X.tail(1000)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 10,test_size=0.2,shuffle =True)
logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)
# fit the model on the whole dataset
random_forest = RandomForestClassifier()

random_forest.fit(X_train, y_train)
decisionTreeModel = DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = None, 
                                           splitter='best', 
                                           random_state=10)

decisionTreeModel.fit(X_train,y_train)
# gradientBoostingModel = GradientBoostingClassifier(loss = 'deviance',
#                                                    learning_rate = 0.01,
#                                                    n_estimators = 100,
#                                                    max_depth = 30,
#                                                    random_state=10)

# gradientBoostingModel.fit(X_train,y_train)
KNeighborsModel = KNeighborsClassifier(n_neighbors = 7,
                                       weights = 'distance',
                                      algorithm = 'brute')

KNeighborsModel.fit(X_train,y_train)
# SGDClassifier = SGDClassifier(loss = 'hinge', 
#                               penalty = 'l1',
#                               learning_rate = 'optimal',
#                               random_state = 10, 
#                               max_iter=100)

# SGDClassifier.fit(X_train,y_train)
# SVClassifier = SVC(kernel= 'linear',
#                    degree=3,
#                    max_iter=10000,
#                    C=2, 
#                    random_state = 55)

# SVClassifier.fit(X_train,y_train)
bernoulliNBModel = BernoulliNB(alpha=0.1)
bernoulliNBModel.fit(X_train,y_train)
gaussianNBModel = GaussianNB()
gaussianNBModel.fit(X_train,y_train)
XGB_Classifier = XGBClassifier()
XGB_Classifier.fit(X_train, y_train)
#evaluation Details
models = [logistic_regression, random_forest, decisionTreeModel, KNeighborsModel, 
            bernoulliNBModel, gaussianNBModel, XGB_Classifier]

for model in models:
    print(type(model).__name__,' Train Score is   : ' ,model.score(X_train, y_train))
    print(type(model).__name__,' Test Score is    : ' ,model.score(X_test, y_test))
    
    y_pred = model.predict(X_test)
    print(type(model).__name__,' F1 Score is      : ' ,f1_score(y_test,y_pred))
    print('--------------------------------------------------------------------------')
y_pred = XGB_Classifier.predict(X_test)
import seaborn as sn

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)
from sklearn.metrics import accuracy_score,classification_report

print(accuracy_score(y_test,y_pred).round(4)*100,'\n')

print(pd.crosstab(y_test,y_pred),'\n')

print(classification_report(y_test,y_pred),'\n')
X_test.shape
test.shape
test_to_pred = normalize(test)
test_predict = XGB_Classifier.predict(test_to_pred)
test.reset_index(inplace = True)
test.head()
predict = test['EventId']
test_predict = pd.Series(test_predict)
predict = pd.concat([predict,test_predict], axis=1)
predict.rename(columns={0: "Class"},inplace=True)
predict = predict.replace(1,'s')
predict = predict.replace(0,'b')
predict['RankOrder'] = predict['Class'].argsort().argsort() + 1 # +1 to start at 1
predict = predict[['EventId', 'RankOrder','Class']]
predict.to_csv("submission.csv",index=False)
predict.tail(200)
print(predict.RankOrder.min())
print(predict.RankOrder.max())
sb.countplot(predict.Class)