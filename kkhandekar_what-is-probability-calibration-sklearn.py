# Generic

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, warnings, gc

warnings.filterwarnings("ignore")



# SKLearn

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.calibration import CalibratedClassifierCV, calibration_curve



# Plot

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px

import plotly.graph_objects as go
# Data Load

url = '../input/all-datasets-for-practicing-ml/Class/Class_Ionosphere.csv'

data = pd.read_csv(url, header='infer')



# Total Classes

print("Total Classes: ", data.Class.nunique())



''' Prep '''

encoder = LabelEncoder()

data['Class']= encoder.fit_transform(data['Class']) 



''' Split '''

columns = data.columns

target = ['Class']   

features = columns [:-1]



X = data[features]

y = data[target]



# Training = 90% & Validation = 10% 

sample_weight = np.random.RandomState(42).rand(y.shape[0])

test_size = 0.1

X_train, X_val, y_train, y_val,sw_train, sw_test  = train_test_split(X, y, sample_weight, test_size=test_size, random_state=0, shuffle=True) 





''' Feature Scaling '''

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_val = sc.transform(X_val)
'''Gaussian Naive-Bayes (no calibration)'''

gnb = GaussianNB()

gnb.fit(X_train,y_train)  # GaussianNB itself does not support sample-weights

prob_pos_gnb = gnb.predict_proba(X_val)[:, 1]



'''Gaussian Naive-Bayes (isotonic calibration)'''

gnb_isotonic = CalibratedClassifierCV(gnb, cv='prefit', method="isotonic")

gnb_isotonic.fit(X_train, y_train, sample_weight=sw_train)

prob_pos_gnbiso = gnb_isotonic.predict_proba(X_val)[:, 1]



'''Gaussian Naive-Bayes (sigmoid  calibration)'''

gnb_sigmoid = CalibratedClassifierCV(gnb, cv='prefit', method="sigmoid")

gnb_sigmoid.fit(X_train, y_train, sample_weight=sw_train)

prob_pos_gnbsig = gnb_sigmoid.predict_proba(X_val)[:, 1]



print("Brier scores: (should be low)\n")



gnb_score = brier_score_loss(y_val, prob_pos_gnb, sample_weight=sw_test)

print("GaussianNB Brier Score (no calibration) : %1.3f" % gnb_score)



gnb_iso_score = brier_score_loss(y_val, prob_pos_gnbiso, sample_weight=sw_test)

print("GaussianNB Brier Score (with isotonic) : %1.3f" % gnb_iso_score)



gnb_sig_score = brier_score_loss(y_val, prob_pos_gnbsig, sample_weight=sw_test)

print("GaussianNB Brier Score (with sigmoid) : %1.3f" % gnb_sig_score)
plt.figure(figsize=(15,10))

order = np.lexsort((prob_pos_gnb, ))



plt.plot(prob_pos_gnb[order], 'r', label='No calibration (%1.3f)' % gnb_score)

plt.plot(prob_pos_gnbiso[order], 'g', linewidth=3, label='Isotonic calibration (%1.3f)' % gnb_iso_score)

plt.plot(prob_pos_gnbsig[order], 'b', linewidth=3, label='Sigmoid calibration (%1.3f)' % gnb_sig_score)



plt.ylim([-0.05, 1.05])



plt.xlabel("Instances Sorted according to Predicted Probability " "(uncalibrated GNB)")

plt.ylabel("P(y=1)")

plt.legend(loc="upper left")

plt.title("Gaussian naive Bayes Probabilities", fontsize=20)



plt.show()
fig = plt.figure(figsize=(15, 10))

ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

ax2 = plt.subplot2grid((3, 1), (2, 0))



ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")



frac_of_pos, mean_pred_val = calibration_curve(y_val, prob_pos_gnb, n_bins=10)

frac_of_pos_iso, mean_pred_val_iso = calibration_curve(y_val, prob_pos_gnbiso, n_bins=10)

frac_of_pos_sig, mean_pred_val_sig = calibration_curve(y_val, prob_pos_gnbsig, n_bins=10)



ax1.plot(mean_pred_val, frac_of_pos, "s-", label='No calibration (%1.3f)' % gnb_score)

ax1.plot(mean_pred_val_iso, frac_of_pos_iso, "s-", label='Isotonic calibration (%1.3f)' % gnb_iso_score)

ax1.plot(mean_pred_val_sig, frac_of_pos_sig, "s-", label='Sigmoid calibration (%1.3f)' % gnb_sig_score)



ax2.hist(prob_pos_gnb, range=(0, 1), bins=10, label='No calibration', histtype="step", lw=2)

ax2.hist(prob_pos_gnbiso, range=(0, 1), bins=10, label='Isotonic calibration', histtype="step", lw=2)

ax2.hist(prob_pos_gnbsig, range=(0, 1), bins=10, label='Sigmoid calibration', histtype="step", lw=2)





ax1.set_ylabel("Fraction of positives")

ax1.set_ylim([-0.05, 1.05])

ax1.legend(loc="lower right")

ax1.set_title('GaussianNB Calibration plots  (reliability curve)', fontsize=20)



ax2.set_xlabel("Mean predicted value")

ax2.set_ylabel("Count")

ax2.legend(loc="upper center", ncol=2)



plt.tight_layout()

# Create classifiers

lr = LogisticRegression()

knn = KNeighborsClassifier()

dtc = DecisionTreeClassifier(random_state=42)

rfc = RandomForestClassifier(random_state=42, verbose=0)



'''Plot calibration plots'''



plt.figure(figsize=(15, 10))

ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

ax2 = plt.subplot2grid((3, 1), (2, 0))



ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")



for clf, name in [(lr, 'Logistic'), (knn, 'KNN'), (gnb, 'NaiveBayes'),                   

                  (dtc, 'DecisionTree'), (rfc, 'RandomForest')]:

    

    clf.fit(X_train,y_train)

    prob_pos = clf.predict_proba(X_val)[:, 1]

    fraction_of_positives, mean_predicted_value = calibration_curve(y_val, prob_pos, n_bins=10)



    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s" % (name, ))

    ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    



ax1.set_ylabel("Fraction of positives")

ax1.set_ylim([-0.05, 1.05])

ax1.legend(loc="lower right")

ax1.set_title('Classifier Calibration plots  (reliability curve)', fontsize=20)



ax2.set_xlabel("Mean predicted value")

ax2.set_ylabel("Count")

ax2.legend(loc="upper center", ncol=2)



plt.tight_layout()

plt.show()

    
# Data Load

url = '../input/all-datasets-for-practicing-ml/Class/Class_Abalone.csv'

df = pd.read_csv(url, header='infer')



# Total Classes

print("Total Classes: ", df.Sex.nunique())



''' Prep '''

encoder = LabelEncoder()

df['Sex']= encoder.fit_transform(df['Sex']) 



''' Split '''

columns = df.columns

target = ['Sex']   

features = columns [1:]



X = df[features]

y = df[target]



# Training = 90% & Validation = 10% 

#test_size = 0.1

#X_train, X_val, y_train, y_val  = train_test_split(X, y, test_size=test_size, random_state=0, shuffle=True) 







''' Feature Scaling '''

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_val = sc.transform(X_val)
# Classifier

knn = KNeighborsClassifier()



# Train

knn.fit(X_train,y_train)



#Calc Probability

knn_probs = knn.predict_proba(X_val)

knn_score = log_loss(y_val,knn_probs)



# Probability Calibration (sigmoid method)

sig_knn = CalibratedClassifierCV(knn, method="sigmoid", cv="prefit")

sig_knn.fit(X_train, y_train)

sig_knn_probs = sig_knn.predict_proba(X_val)

sig_knn_score = log_loss(y_val, sig_knn_probs)



# Probability Calibration (sigmoid method)

iso_knn = CalibratedClassifierCV(knn, method="isotonic", cv="prefit")

iso_knn.fit(X_train, y_train)

iso_knn_probs = iso_knn.predict_proba(X_val)

iso_knn_score = log_loss(y_val, iso_knn_probs)





print("KNN Log Loss (no calibration) : %1.3f" % knn_score)

print("KNN Log Loss (sigmoid) : %1.3f" % sig_knn_score)

print("KNN Log Loss (isotonic) : %1.3f" % iso_knn_score)


