import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("../input/creditcardfraud/creditcard.csv")

df.head()
df.describe()
#wow, a complete dataset!

df.isnull().sum()
df["Class"].skew() #of course our target is heavily skewed in favor of the normal cases
df["Class"].value_counts()

df["Class"].value_counts().plot(kind = "bar")
print("Percentage of Frauds:" + str(len(df[df["Class"]==1])/len(df.index)*100))

print("Percentage of Regulars:" + str(len(df[df["Class"]==0])/len(df.index)*100))

# fraud cases make up less than half a percent of all samples
# when scaling we should choose a sclaer which is more robust on outliers

fig, ax = plt.subplots(2,2, figsize = (10,8)) #alternativly fig, (ax1, ax2, ax3, ax4)



ax[0,0].boxplot(df["Time"])

ax[0,1].boxplot(df["Amount"])

ax[0,0].set_title("Time")

ax[0,1].set_title("Amount")



ax[1,0].hist(df["Time"], bins = 200)

ax[1,1].hist(df["Amount"],bins = 200)

ax[1,0].set_title("Time")

ax[1,1].set_title("Amount")
# I settled with the robust scaler

# it is more robust to outliers due to the fact that it uses the median and the IQR rather than the mean/std (Standardscaler)



from sklearn.preprocessing import RobustScaler

rob = RobustScaler()



df["scaled_time"] = rob.fit_transform(df["Time"].values.reshape(-1,1))       #values get directly converted into array

df["scaled_amount"] = rob.fit_transform(df["Amount"].values.reshape(-1,1))   #reshape it into (-1,1) because I have only one feature 

df.drop(columns = {"Time", "Amount"}, inplace = True)
df.columns

df = df[['scaled_time', 'scaled_amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',

       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',

       'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Class', ]]

df.head()
#the ten folds in cv make this whole expeditur computationally very expensive, but for the sake of "comparability" with the following resample-methods..

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

X = df.drop(columns = {"Class"})

y = df["Class"]



lr = LogisticRegression(random_state = 0)

RFC = RandomForestClassifier(random_state = 0)

DTC = DecisionTreeClassifier(random_state = 0)

print("Accuracy LR of original:", cross_val_score(lr, X, y, cv = 10, scoring = "accuracy").mean())

print("Accuracy DTC of original:", cross_val_score(DTC, X, y, cv = 10, scoring = "accuracy").mean())

print("Accuracy RFC of original:", cross_val_score(RFC, X, y, cv = 10, scoring = "accuracy").mean())
# same wiht recall as recall as metric

print("Recall LR of original:", cross_val_score(lr, X, y, cv = 10, scoring = "recall").mean())

print("Recall DTC of original:", cross_val_score(DTC, X, y, cv = 10, scoring = "recall").mean())

print("Recall RFC of original:", cross_val_score(RFC, X, y, cv = 10, scoring = "recall").mean())

#First, I will apply the simple undersampling method:



# determine the len total number of fraud cases

len_fraud = len(df[df["Class"] == 1])



# take the indices of the fraud/normal cases

fraud_indices = df[df["Class"]== 1].index

normal_indices = df[df["Class"] == 0].index



# take a randomly provided set of x samples (= len_fraud) from the indices of the normal cases

rand_norm_ind = np.random.choice(normal_indices ,len_fraud, replace = False)



# concetanate the indices from fraud cases with the randomly selected normal cases

undersample_indices = np.concatenate([rand_norm_ind, fraud_indices])



# create an undersample dataset with the indices

undersample = df.iloc[undersample_indices, :]
y_undersample = undersample["Class"]

X_undersample = undersample.drop(columns = {"Class"})

lr2 = LogisticRegression(random_state = 0)

RFC2 = RandomForestClassifier(random_state = 0)

DTC2 = DecisionTreeClassifier(random_state = 0)

#since I declared that recall is my objective, I will now only calculate recall for the different models
print("Recall LG of undersample:", cross_val_score(lr2, X_undersample, y_undersample, cv = 10, scoring = "recall").mean())

print("Recall DTC of undersample:", cross_val_score(DTC2, X_undersample, y_undersample, cv = 10, scoring = "recall").mean())

print("Recall RFC of undersample:", cross_val_score(RFC2, X_undersample, y_undersample, cv = 10, scoring = "recall").mean())
corr_df = pd.DataFrame()

corr_df["Corr"] = undersample.corr()["Class"].sort_values(ascending = False)

concat = pd.concat([corr_df.head(2), corr_df.tail(7)])

print(concat)

#showing only values with a correlation to "Class" greater than 0.5 or lesser than -0.5

#it is important to use the correlation after balancing the data set
suspect = ["V3", "V4", "V9", "V10", "V11", "V12", "V14", "V16", "V17"]

best_recall = 0

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits = 10, random_state = 0, shuffle = True)



for vars in suspect:

    for e in [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2, 2.5, 3]:

        for m in [lr2, DTC2, RFC2]:

            UQ = np.percentile(undersample[vars], 75)

            LQ = np.percentile(undersample[vars], 25)

            IQR = UQ - LQ

            UW = (UQ + e*IQR)

            LW = (LQ - e*IQR)

            mask = (undersample[[vars]] < LW) | (undersample[[vars]] > UW)

            undersample[mask] = np.nan

            undersample.dropna(inplace = True)

            undersample.shape

            X_undersample_new = undersample.drop(columns = {"Class"})

            y_undersample_new = undersample["Class"]

            recall = cross_val_score(m, X_undersample_new, y_undersample_new, cv = skf, scoring = "recall").mean()

            if recall > best_recall:

                best_recall = recall

                best_factor = {"e": e}

                best_model = {"m": m}

print("Best Recall: {:.3f}".format(best_recall))

print("Best Modell:", best_model)

print("Best factor used:", best_factor)


from imblearn.over_sampling import SMOTE

from sklearn.metrics import recall_score



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.2) #tell python to stratify the target in quotes like in the original



sm = SMOTE(random_state = 10)

X_train_over, y_train_over  = sm.fit_sample(X_train, y_train) #smote creates arrays



# make sure SMOTE worked correctly

np.unique(y_train_over, return_counts = True)
#first with LogisticRegression

lr3 = LogisticRegression(random_state = 10)

RFC3 = RandomForestClassifier(random_state = 0)

DTC3 = DecisionTreeClassifier(random_state = 0)



#first LogisticRegression

lr3.fit(X_train_over, y_train_over)

LR_over_prediction = lr3.predict(X_test)



#then with DecisionTree

DTC3.fit(X_train_over, y_train_over)

DTC_over_prediction = DTC3.predict(X_test)



#then with RandomForestClassifier

RFC3.fit(X_train_over, y_train_over)

RFC_over_prediction = RFC3.predict(X_test)



print("Recall of LG-Model:", recall_score(y_test, LR_over_prediction))

print("Recall of DTC-Model:", recall_score(y_test, DTC_over_prediction))

print("Recall of RFC-Model:", recall_score(y_test, RFC_over_prediction))