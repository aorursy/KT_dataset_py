# Loading the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import seaborn as sns
%matplotlib inline

# Loading the DataSet
df = pd.read_csv('../input/glass.csv')

X = df[['Type']]
y = df.drop('Type',1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
### Feature Importance

rf = RandomForestClassifier()
rf.fit(y.values, X.values.ravel())

importance = rf.feature_importances_
importance = pd.DataFrame(importance, index = y.columns, columns=['Importance'])

feats = {}
for feature, importance in zip(y.columns,rf.feature_importances_):
    feats[feature] = importance
    
print (feats)
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=45)
### HeatMap

y_cols = y.columns.tolist()
corr = df[y_cols].corr()

sns.heatmap(corr)
### Prediction

model = PolynomialFeatures(degree= 4)
y_ = model.fit_transform(y)
y_test_ = model.fit_transform(y_test)

lg = LinearRegression()
lg.fit(y_,X)
predicted_data = lg.predict(y_test_)
predicted_data = np.round_(predicted_data)

print ('Mean Square Error:')
print (mean_squared_error(predicted_data, X_test))
print ('Predicted Values:')
print (predicted_data.ravel())

### Correlation Matrix
print ('')
print ('Confusion Matix:')
print (confusion_matrix(X_test, predicted_data))
