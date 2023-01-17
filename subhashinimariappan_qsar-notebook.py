#Import Libraries

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error



#preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures
# column headers 

_headers = ['CIC0', 'SM1', 'GATS1i', 'NdsCH', 'Ndssc', 'MLOGP', 'response'] 

# read in data 

df = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter06/Dataset/qsar_fish_toxicity.csv', names=_headers, sep=';') 
df.head()
# Let's split our data 



features = df.drop('response', axis=1).values 

labels = df[['response']].values 

X_train, X_eval, y_train, y_eval = train_test_split(features, labels, test_size=0.2, random_state=0) 

X_val, X_test, y_val, y_test = train_test_split(X_eval, y_eval, random_state=0) 
model = LinearRegression() 
model.fit(X_train, y_train) 
y_pred = model.predict(X_val) 
r2 = model.score(X_val, y_val) 

print('R^2 score: {}'.format(r2)) 
_ys = pd.DataFrame(dict(actuals=y_val.reshape(-1), predicted=y_pred.reshape(-1))) 

_ys.head() 
# Let's compute our MEAN ABSOLUTE ERROR

mae = mean_absolute_error(y_val, y_pred)

print('MAE: {}'.format(mae))
#Let's get the R2 score

r2 = model.score(X_val, y_val)

print('R^2 score: {}'.format(r2))
#create a pipeline and engineer quadratic features

steps = [

    ('scaler', MinMaxScaler()),

    ('poly', PolynomialFeatures(2)),

    ('model', LinearRegression())

]
#create a Linear Regression model

model = Pipeline(steps)
#train the model

model.fit(X_train, y_train)
#predict on validation dataset

y_pred = model.predict(X_val)
#compute MAE

mae = mean_absolute_error(y_val, y_pred)

print('MAE: {}'.format(mae))
# let's get the R2 score

r2 = model.score(X_val, y_val)

print('R^2 score: {}'.format(r2))
from sklearn.externals import joblib
joblib.dump(model, './model.joblib')
m2 = joblib.load('./model.joblib')
m2_preds = m2.predict(X_val)
ys = pd.DataFrame(dict(predicted=y_pred.reshape(-1), m2=m2_preds.reshape(-1)))

ys.head()
