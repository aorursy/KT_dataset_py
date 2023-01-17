import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

import pandas as pd
import os

print(os.listdir("../input"))
df = pd.read_csv("../input/mobile-price-classification/train.csv")
df.head()
def initial_observation(df):

    if isinstance(df, pd.DataFrame):

        total_na = df.isna().sum().sum()

        print("Dimensions : %d rows, %d columns" % (df.shape[0], df.shape[1]))

        print("Total NA Values : %d " % (total_na))

        print("%38s %10s     %10s %10s" % ("Column Name", "Data Type", "#Distinct", "NA Values"))

        col_name = df.columns

        dtyp = df.dtypes

        uniq = df.nunique()

        na_val = df.isna().sum()

        for i in range(len(df.columns)):

            print("%38s %10s   %10s %10s" % (col_name[i], dtyp[i], uniq[i], na_val[i]))

        

    else:

        print("Expect a DataFrame but got a %15s" % (type(df)))
initial_observation(df)
df.corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
sns.set()

sns.pairplot(df, size = 5.5)

plt.show();
df["price_range"].value_counts()
sns.catplot(x="ram", y="price_range", kind="swarm", data=df);
sns.catplot(x="battery_power", y="price_range", kind="swarm", data=df);
plt.boxplot(df["fc"])

plt.show()
sorted(df["fc"])
q1, q3= np.percentile(df["fc"],[25,75])
iqr = q3 - q1
lower_bound = q1 -(1.5 * iqr) 

upper_bound = q3 +(1.5 * iqr) 
fc_outlier = df[df["fc"]>16.0]

#df['preTestScore'].where(df['postTestScore'] > 50)
fc_outlier
import numpy as np

import pandas as pd

outliers=[]

def detect_outlier(data_1):

    

    threshold=3

    mean_1 = np.mean(data_1)

    std_1 =np.std(data_1)

    

    

    for y in data_1:

        z_score= (y - mean_1)/std_1 

        if np.abs(z_score) > threshold:

            outliers.append(y)

    return outliers
plt.boxplot(df["px_height"])

plt.show()
outlier_datapoints1 = detect_outlier(df["px_height"])

print(outlier_datapoints1)
plt.boxplot(df["three_g"])

plt.show()
outlier_datapoints2 = detect_outlier(df["three_g"])

print(outlier_datapoints2)
x = df.drop(["price_range"], axis = 1)

y = df["price_range"]
x.head()
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(x,y)
X_train_scaled = X_train.copy()

X_val_scaled = X_val.copy()



from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X_train_scaled)

x_train_scaled_1 = scaler.transform(X_train_scaled)

x_val_scaled_1 = scaler.transform(X_val_scaled)

print("X Train shape:" , X_train.shape)

print("X Validation shape:" ,   X_val.shape)

print("Y Train shape:",     Y_train.shape)

print( "Y Validation Shape:",   Y_val.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
rf_parm = dict(n_estimators = [20, 30, 50, 70, 100, 150], max_features = [0.1, 0.2, 0.6, 0.9], max_depth = [10,20,30],min_samples_leaf=[1,10,100, 400, 500, 600],random_state=[0])
rc = RandomForestClassifier()

rf_grid = GridSearchCV(estimator = rc, param_grid = rf_parm)
rf_grid.fit(X_train,Y_train)
print("RF Best Score:", rf_grid.best_score_)

print("RF Best Parameters:", rf_grid.best_params_)
rc_best = RandomForestClassifier(n_estimators = 150,  max_features = 0.6, min_samples_leaf = 1, max_depth = 20, random_state = 0 )
rc_best.fit(X_train, Y_train)

rc_tr_pred = rc_best.predict(X_train)

rc_val_pred = rc_best.predict(X_val)
print(rc_val_pred)
from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import f1_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score
print("Precision Score : ",precision_score(Y_val, rc_val_pred , 

                                           pos_label='positive',

                                           average='weighted'))

print("Recall Score : ",recall_score(Y_val, rc_val_pred , 

                                           pos_label='positive',

                                           average='weighted'))

print("F1 Score:",  f1_score(Y_val, rc_val_pred , 

                                           pos_label='positive',

                                           average='weighted'))
from sklearn.metrics import classification_report



print(classification_report(Y_val, rc_val_pred))
df_test = pd.read_csv("../input/mobile-price-classification/test.csv")
df_test.head()
x_test = df_test.drop(["id"], axis = 1)
rc_test_pred_unscaled = rc_best.predict(x_test)
rc_test_pred_unscaled
test_pred = pd.DataFrame({"id" : range(1000), "test_price_range" : rc_test_pred_unscaled })
test_pred["test_price_range"].value_counts()
pd.value_counts(test_pred["test_price_range"]).plot.bar()
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,Y_train)
clf_tr_pred = clf.predict(X_train)

clf_val_pred = clf.predict(X_val)
print(clf_val_pred)
print("Precision Score(Decision Tree) : ",precision_score(Y_val, clf_val_pred , 

                                           pos_label='positive',

                                           average='weighted'))

print("Recall Score(Decision Tree) : ",recall_score(Y_val, clf_val_pred , 

                                           pos_label='positive',

                                           average='weighted'))

print("F1 Score(Decision Tree) :",  f1_score(Y_val, clf_val_pred , 

                                           pos_label='positive',

                                           average='weighted'))
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(Y_val, clf_val_pred))
from sklearn.metrics import classification_report



print(classification_report(Y_val, clf_val_pred))
clf_test_pred_unscaled = clf.predict(x_test)
test_pred_tree = pd.DataFrame({"id" : range(1000), "test_price_range" : clf_test_pred_unscaled })
test_pred_tree["test_price_range"].value_counts()
pd.value_counts(test_pred_tree["test_price_range"]).plot.bar()