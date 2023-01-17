# Imports

import pandas as pd

import numpy as np



# Read data 

train0 = pd.read_csv("bank-train.csv")

test0 = pd.read_csv("bank-test.csv")

training=pd.read_csv('bank-train.csv')

testing=pd.read_csv('bank-test.csv')

id_train = train0.iloc[:, 0]

id_test = test0.iloc[:, 0]



df = pd.concat([train0, test0], sort = False)



df = df.reset_index(drop = True) # Concatenating messed up the index. If you want to concatenate two dataframes with index ranges 1-10 and 1-5, the concatenated dataframe will have 1-10 and 1-5.

X0 = df.iloc[:, :-1]

#X0 = X0.drop(columns = "duration") # duration is not known before a call is made (future leak); this will influence results

Y = df.iloc[:, -1]



del train0, test0

# =============================================================================

# DATA ENCODING

# =============================================================================



# Encode categorical variables (nominal)



def onehotencode(df, field):

    result = []

    grp = list(df.groupby(field).count().iloc[:, 0].index)

    grp1 = []



    for item in grp:

        result.append(list(df.loc[:, field] == item))

        grp1.append(field + " " + item)

    

    result = np.array(result).T

    result = result.astype(int)

    result = pd.DataFrame(result, columns = grp1)

    return result

    

job_encoded = onehotencode(X0, "job")

marital_encoded = onehotencode(X0, "marital")

education_encoded = onehotencode(X0, "education")

default_encoded = onehotencode(X0, "default")

housing_encoded = onehotencode(X0, "housing")

loan_encoded = onehotencode(X0, "loan")

contact_encoded = onehotencode(X0, "contact")

poutcome_encoded = onehotencode(X0, "poutcome")



# Encode categorical variables (ordinal)



def encode_month():

    months_encoded = []

    for i in range(len(X0)):

        month = X0.loc[i, "month"]

        if month == "jan":

            months_encoded.append(1)

        elif month == "feb":

            months_encoded.append(2)

        elif month == "mar":

            months_encoded.append(3)

        elif month == "apr":

            months_encoded.append(4)        

        elif month == "may":

            months_encoded.append(5)

        elif month == "jun":

            months_encoded.append(6)

        elif month == "jul":

            months_encoded.append(7)

        elif month == "aug":

            months_encoded.append(8)

        elif month == "sep":

            months_encoded.append(9)

        elif month == "oct":

            months_encoded.append(10)

        elif month == "nov":

            months_encoded.append(11)

        elif month == "dec":

            months_encoded.append(12)



    months_encoded = pd.DataFrame(months_encoded, columns = ["month num"])

    return months_encoded



def encode_day_of_week():

    days_encoded = []

    for i in range(len(X0)):

        day = X0.loc[i, "day_of_week"]

        if day == "mon":

            days_encoded.append(1)

        elif day == "tue":

            days_encoded.append(2)

        elif day == "wed":

            days_encoded.append(3)

        elif day == "thu":

            days_encoded.append(4)        

        elif day == "fri":

            days_encoded.append(5)



    days_encoded = pd.DataFrame(days_encoded, columns = ["day of week num"])

    return days_encoded



months_encoded = encode_month()

days_encoded = encode_day_of_week()



# Insert all of the encoded variables' values into the dataframe

X = pd.concat([X0, job_encoded, marital_encoded, education_encoded, default_encoded, housing_encoded, loan_encoded, contact_encoded, poutcome_encoded, months_encoded, days_encoded], axis = 1)



# Delete variables

del job_encoded, marital_encoded, education_encoded, default_encoded, housing_encoded, loan_encoded, contact_encoded, poutcome_encoded, months_encoded, days_encoded, X0



# Now that the categorical variables have been encoded, we can remove the original ones

X = X.drop(columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"])



# We need to solve the dummy variable trap (drop some features to prevent perfect multicollinearity)

X = X.drop(columns = ["job unknown", "marital unknown", "education unknown", "default unknown", "housing unknown", "loan unknown", "contact telephone", "poutcome nonexistent"])



# "Rename" variable

X0 = X

del X

# =============================================================================

# FEATURE SCALING

# =============================================================================

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X = pd.DataFrame(sc_X.fit_transform(X0), columns = X0.columns)
# =============================================================================

# TRAIN/TEST SPLIT

# =============================================================================



# For code refining purposes

X = X.iloc[0:32950, :]

Y = Y[0:32950]

from sklearn.model_selection import train_test_split

X_train0, X_test0, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)



# For Kaggle purposes

#X_train0 = X.iloc[0:32950, :]

#X_test0 = X.iloc[32950:, :]

#Y_train = Y[0:32950]

#Y_test = Y[32950:]

# =============================================================================

# FEATURE SELECTION

# =============================================================================

from sklearn.feature_selection import SelectKBest

n = 41

test = SelectKBest(k = n)

fit = test.fit(X_train0, Y_train)

X_train = fit.transform(X_train0)

X_test = fit.transform(X_test0)

# =============================================================================

# RUN MODEL RANDOM FOREST

# =============================================================================



# Load model

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 100, criterion = "entropy", random_state = 0)



# Fit the model to the training set

model.fit(X_train, Y_train)



# Predict the test set results

Y_test_pred = pd.Series(model.predict(X_test).astype(int))



# Predict the training set results

Y_train_pred = pd.Series(model.predict(X_train).astype(int))



# Create Kaggle submission file

#submission = pd.DataFrame()

#submission["id"] = id_test

#submission["Predicted"] = Y_test_pred

#submission.to_csv("submission.csv", index = False)



# F1 score

from sklearn.metrics import f1_score

print("Selected", n, "out of", len(X.T), "features")

print(model.get_params)

print("F1 score:", f1_score(Y_test, Y_test_pred, average = "weighted"))
# =============================================================================

# RUN MODEL SVC 

# =============================================================================



# Load model

from sklearn.svm import SVC

svc = SVC(kernel = 'poly', degree=15, gamma = 'scale')



# Fit the model to the training set

svc.fit(X_train, Y_train)



# Predict the test set results

Y_test_pred = pd.Series(svc.predict(X_test).astype(int))



# Predict the training set results

Y_train_pred = pd.Series(svc.predict(X_train).astype(int))



# Create Kaggle submission file

#submission = pd.DataFrame()

#submission["id"] = id_test

#submission["Predicted"] = Y_test_pred

#submission.to_csv("submission.csv", index = False)



# F1 score

print("Selected", n, "out of", len(X.T), "features")

print(svc.get_params)

print("F1 score:", f1_score(Y_test, Y_test_pred, average = "weighted"))
# =============================================================================

# RUN LOGISTIC REGRESSION MODEL

# =============================================================================

from sklearn.linear_model import LogisticRegression

#takes into account all numerical variables



X_train=training[['age','duration','campaign','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']]

train_y=training.y

#performs logistic regression on training set

LogReg = LogisticRegression()

LogReg.fit(X_train, train_y)



test_x=testing[['age','duration','campaign','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']]

y_test_pred = LogReg.predict(test_x)
