#Data Wrangling Packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#Preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.base import TransformerMixin

from sklearn.decomposition import PCA



#Model

from sklearn.linear_model import Lasso

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

import xgboost as xgb



#Model Selection and Evaluation

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

%matplotlib inline
train_dta = pd.read_csv('../input/train.csv')

test_dta = pd.read_csv('../input/test.csv')
train_dta.head()
test_dta.head()
print ('Training Data')

train_dta.info()

print ('---------------------------------------')

print ('Test Data')

test_dta.info()
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']

label = ['Survived']
class NumImputer(TransformerMixin):

    def fit(self, x, y=None):

        self.fill = pd.Series([x[c].value_counts().index[0]

            if x[c].dtype == np.dtype('O') else x[c].median() for c in x],

            index=x.columns)

        return self

    def transform(self, x, y=None):

        return x.fillna(self.fill)



train_imputed = NumImputer().fit_transform(train_dta[features]).apply(LabelEncoder().fit_transform)

test_imputed = NumImputer().fit_transform(test_dta[features]).apply(LabelEncoder().fit_transform)
print ("Training Features")

train_imputed.info()

print ('---------------------------------------')

print ("Test Features")

test_imputed.info()
x_train, x_test, y_train, y_test = train_test_split(train_imputed.values,

                                                    train_dta[label].values.ravel(),

                                                    test_size = 0.4, 

                                                    random_state=0)
# Extreme Gradient Boost

xgboost = xgb.XGBClassifier(max_depth=9, 

                            n_estimators=100, 

                            learning_rate=0.1).fit(x_train, y_train)

xgb_predicted = xgboost.predict(x_test)
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)

rf_predicted = random_forest.predict(x_test)
# Gaussian Process Regression 

gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

gs_predicted = gaussian.predict(x_test)
# k-Nearest Neighbours

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train, y_train)

knn_predicted = knn.predict(x_test)
# Support Vector Machine

svc = SVC(kernel='linear', C=1)

svc.fit(x_train, y_train)

svc_predicted = svc.predict(x_test)
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(x_train, y_train)

lgr_predicted = logreg.predict(x_test)
model_score_matrix = {"Model": ["Extreme Gradient Boosting",

                                "Random Forest",

                                "Gaussian Process Regression",

                                "k-Nearest Neighbours",

                                "Support Vector Machine",

                                "Logistic Regression"],

                      "Score": [(xgboost.score(x_train,y_train)),

                               (random_forest.score(x_train, y_train)),

                               (gaussian.score(x_train, y_train)),

                               (knn.score(x_train, y_train)),

                               (svc.score(x_train, y_train)),

                               (logreg.score(x_train, y_train))]}

model_score = pd.DataFrame(model_score_matrix, columns = ["Model","Score"])

model_score
coeff_df = pd.DataFrame(train_dta.columns.delete([0,3,7,8,10]))

coeff_df.columns = ['Features']

coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

coeff_df
cm_xgb = confusion_matrix(y_test, xgb_predicted)

cm_rf = confusion_matrix(y_test, rf_predicted)

cm_gs = confusion_matrix(y_test, gs_predicted)

cm_knn = confusion_matrix(y_test, knn_predicted)

cm_svc = confusion_matrix(y_test, svc_predicted)

cm_lgr = confusion_matrix(y_test, lgr_predicted)



def plot_confusion(cm, target_names = ['Survived', 'Not Survived'],

                   title='Normalized Confusion Matrix'):

    cm.sum(axis=1)

    cm_normalized = cm.astype(np.float64) / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title(title)

    plt.colorbar()



    tick_marks = np.arange(len(target_names))

    plt.xticks(tick_marks, target_names)

    plt.yticks(tick_marks, target_names)

    plt.ylabel('True Label')

    plt.xlabel('Predicted Label')

    plt.tight_layout()
plt.figure(1)



ax1 = plt.subplot(321)

plot_confusion(cm_xgb)

plt.title("XGBoost")



ax2 = plt.subplot(322)

plot_confusion(cm_rf)

plt.title("Random Forest")



plt.subplot(323)

plot_confusion(cm_gs)

plt.title("Gaussian Process Regression")



plt.subplot(324)

plot_confusion(cm_knn)

plt.title("k-Nearest Neighbours")



plt.subplot(325)

plot_confusion(cm_svc)

plt.title("Support Vector Machine")



plt.subplot(326)

plot_confusion(cm_lgr)

plt.title("Logistic Regression")



# Adjust the subplot layout, because the logit one may take more space

# than usual, due to y-tick labels like "1 - 10^{-3}"

plt.subplots_adjust(top=3, bottom=0, left=0, right=1.5, hspace=0.5, wspace=0.5)
X_test = test_imputed.values
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)

rf_predicted = random_forest.predict(X_test)
X_predicted = pd.DataFrame({"PassengerID":test_dta["PassengerId"].values,

                       "Survived":(pd.Series(rf_predicted))})

X_predicted.groupby("Survived").count()