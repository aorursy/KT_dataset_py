#import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import xgboost as xgb
#Import the dataset
churn_data = pd.read_csv('../input/churn_data.csv')
churn_data.head() #Printing first 5 rows of the dataset
# Define X, y, which are IV and DV (Independent variable and dependent variable)
X=churn_data[['title','paymentMethod','couponDiscount','purchaseValue','giftwrapping','throughAffiliate','shippingFees','dvd','blueray','vinyl','videogame','videogameDownload','tvEquiment','prodOthers','prodSecondHand']]
y=churn_data['returnCustomer']

# Sklearn is a very popular package in maching learning, we will often use this
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


#Tell the model there are two categorical variables: title, paymentMethod
X=pd.get_dummies(X)

#Set 50% of the data as test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
#Decision tree
Tree = xgb.XGBClassifier(n_jobs=4,random_state=0,objective='reg:logistic',subsample=0.5,n_estimators=3000,max_depth=2,max_delta_step=0.1,scale_pos_weight=5)
Model = Tree.fit(X_train, y_train)
y_pred = Model.predict(X_test)
#test your model on new test data and show accuracy
print(Model.score(X_test, y_test))
# See accuracy in a table
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
from sklearn.metrics import roc_curve, auc, roc_auc_score
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
auc(false_positive_rate, true_positive_rate)
