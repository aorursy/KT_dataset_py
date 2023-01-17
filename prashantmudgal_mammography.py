import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
# There seem to be some nominal and ordinal variables here
#Ordinal means ranked 1 to 4, 1 being low and 4 high
#Nominal means no integer value, just numbers given to names

##Hmmmm...I'll have to make changes in the dataset now

#Changes done
wisconsin_data = pd.read_csv("breastCancer.csv")
wisconsin_data.head(10)
wisconsin_data.dtypes
wisconsin_data.describe()
missing_data = []
for j in range(len(wisconsin_data)):
    array_of_values = wisconsin_data.iloc[j].values
    if '?' in array_of_values:
        missing_data.append(j)
wisconsin_data = wisconsin_data.drop(missing_data)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in wisconsin_data.columns.values:
    if wisconsin_data[col].dtypes == 'object':
        data = wisconsin_data[col].append(wisconsin_data[col])
        le.fit(data.values)
        wisconsin_data[col] = le.transform(wisconsin_data[col])
target_var_array = wisconsin_data['class'].values

reshaped_data = wisconsin_data.drop(['class', 'id'], axis = 1)

predictors = reshaped_data.values
scaler = StandardScaler()
predictors = scaler.fit_transform(predictors)
from sklearn.model_selection import train_test_split
X_t, X_tt, y_t, y_tt = train_test_split(predictors, target_var_array, test_size = 0.3)
from sklearn.svm import SVC
SVC = SVC()
SVC.fit(X_t, y_t)
prediction_SV = SVC.predict(X_tt)
from sklearn.metrics import accuracy_score 
accuracy_score(y_tt, prediction_SV)
results_from_SV = classification_report(y_tt, prediction_SV)
print(results_from_SV)
#Looks good
#Let's try some more advanced stuff
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(penalty = 'l2', C = 1)
log.fit(X_t , y_t)
accuracy_score(y_tt, log.predict(X_tt))
#Random Forests
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
random.seed(100)
rf = RandomForestClassifier( n_estimators = 500)
rf.fit(X_t , y_t)
accuracy_score(y_tt, rf.predict(X_tt))
results_from_logistic = classification_report(y_tt , log.predict(X_tt))
print(results_from_logistic)
results_from_rf = classification_report(y_tt , rf.predict(X_tt))
print(results_from_rf)
