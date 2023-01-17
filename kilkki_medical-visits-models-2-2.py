import numpy as np

import pandas as pd

from pylab import *



# Check data.

df = pd.read_csv(

    '../input/No-show-Issue-Comma-300k.csv', 

    parse_dates=['ApointmentData', 'AppointmentRegistration'])



# Fix some errors

df['AwaitingTime'] *= -1

df = df.rename(columns={

    'Alcoolism': 'Alcoholism', 'ApointmentData': 'AppointmentDate', 'Handcap': 'Handicap'})



df.head()

# Clean-up

df['ShowedUp'] = df['Status']=='Show-Up'

df = df.drop('Status', axis=1)



def dummies(df, column):

    """Get dummy variables for each unique value of df[column]."""

    df = pd.get_dummies(df[column]).astype(float)

    df.columns = ['%s_%s'%(column, s) for s in df.columns]

    return df



# In the data, everyone is marked either male or female

df['IsMale'] = (df['Gender']=='M').astype(int)

df = df.drop('Gender', axis=1)



# Long-term time variable

df['YearFraction'] = df['AppointmentDate'].apply(lambda d: d.year+d.month/12)



# Hour of day

df['RegistrationHour'] = df['AppointmentRegistration'].apply(lambda d: d.hour).astype(float)

df = df.join(dummies(df, 'DayOfTheWeek'))

df = df.drop(['AppointmentDate', 'AppointmentRegistration', 'DayOfTheWeek'], axis=1)



df.info()
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.svm import SVC



classifiers = {

    'regression': LogisticRegression(),

    'GBoost': GradientBoostingClassifier(),

    }



classifiers
# Maybe second-order terms help the linear regression.

df_regression = df.copy()

df_regression['Age^2'] = df_regression['Age']**2

df_regression['RegistrationHour^2'] = df_regression['RegistrationHour']**2

df_regression['AwaitingTime^2'] = df_regression['AwaitingTime']**2



# Attach some attributes to the model objects.



classifiers['regression'].predictors = [c for c in df_regression.columns if c not in ['ShowedUp']]

classifiers['regression'].X = df_regression[classifiers['regression'].predictors]

classifiers['regression'].y = df_regression['ShowedUp']



classifiers['GBoost'].predictors = [c for c in df.columns if c not in ['ShowedUp']]

classifiers['GBoost'].X = df[classifiers['GBoost'].predictors]

classifiers['GBoost'].y = df['ShowedUp']
from sklearn.model_selection import KFold

from sklearn.metrics import f1_score, accuracy_score, log_loss



def KFold_metrics(model):



    X = model.X

    y = model.y



    # Loop over a couple of training/validation dataset pairs

    kf = KFold(n_splits=2, random_state=1)

    for i, (train_idx, cv_idx) in enumerate(kf.split(X)):



        print(type(model), i)



        # Define features and observations

        Xdf_train, Xdf_cv = X.iloc[train_idx], X.iloc[cv_idx]

        y_train, y_cv = y.iloc[train_idx], y.iloc[cv_idx]

    

        # Fit using current training data

        model.fit(Xdf_train, y_train)



        # Predictions for comparison with observations

        cv_prediction = model.predict_proba(Xdf_cv)[:, 1]



        yield {

            # 'accuracy': accuracy_score(y_cv, cv_prediction),

            # 'F1': f1_score(y_cv, cv_prediction),

            'LogLoss': log_loss(y_cv, cv_prediction),

            }



CV_metrics = {

    'regression': pd.DataFrame(list(KFold_metrics(classifiers['regression']))).mean(),

    'GBoost': pd.DataFrame(list(KFold_metrics(classifiers['GBoost']))).mean(),

    }

classifiers['regression'].fit(classifiers['regression'].X, classifiers['regression'].y)

classifiers['GBoost'].fit(classifiers['GBoost'].X, classifiers['GBoost'].y)

print('Done')
np.mean(classifiers['GBoost'].predict(classifiers['GBoost'].X))
np.mean(classifiers['GBoost'].predict_proba(classifiers['GBoost'].X)[:, 1])
# Feature coefficients for the linear (logistic) regression

figure(figsize=(10, 8))

pd.Series(classifiers['regression'].coef_[0], classifiers['regression'].predictors).sort_values().plot(kind='barh')

tight_layout()
# "Feature importances"

figure(figsize=(10, 8))

pd.Series(

    classifiers['GBoost'].feature_importances_, 

    classifiers['GBoost'].predictors).sort_values().plot(kind='barh')


def predicted_series(model):



    X = model.X

    y = model.y



    agedata = X.median()

    ages = range(0, 110)

    age_dataset = pd.DataFrame([agedata]*len(ages))

    age_dataset['Age'] = ages



    predicted = model.predict_proba(age_dataset)

    predicted = pd.Series(predicted[:, 1], ages)

    return predicted



figure()

observed = df['ShowedUp'].groupby(df['Age']).mean()

observed.plot(c='k')

predicted_series(classifiers['regression']).plot()

predicted_series(classifiers['GBoost']).plot()

legend(['Observed', 'Regression', 'Gradient Boost'])

_ = gca().set_ylabel('Probability of showing up')






