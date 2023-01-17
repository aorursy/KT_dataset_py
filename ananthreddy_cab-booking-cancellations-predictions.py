import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('../input/Kaggle_YourCabs_training.csv')
print (df.columns)

print (df.shape)
df.isnull().sum()
df['from_area_id'] = df['from_area_id'].fillna(value = np.mean(df['from_area_id']))
df = df.drop(['package_id', 'to_city_id', 'from_city_id', 'from_date', 'to_date', 'from_lat', 
              'from_long', 'to_lat', 'to_long', 'to_area_id', 'id', 'booking_created'], 
             axis = 1)

df.isnull().sum()
X = df[['Car_Cancellation']]
y = df[['vehicle_model_id', 'travel_type_id', 'online_booking', 
        'mobile_site_booking']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
%matplotlib inline

rf = RandomForestClassifier()
rf.fit(y.values, X.values.ravel())

importance = rf.feature_importances_
importance = pd.DataFrame(importance, index = y.columns, columns=['Importance'])

feats = {}
for feature, importance in zip(y.columns,rf.feature_importances_):
    feats[feature] = importance
    
print (feats)
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 
                                                                            'Gini-importance'})
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=45)
y_cols = y.columns.tolist()
corr = df[y_cols].corr()

sns.heatmap(corr)
lr = LogisticRegression()
rf = RandomForestClassifier(n_estimators=10)
bg = BaggingClassifier(DecisionTreeClassifier(), max_samples = 0.5)
clf = SVC(kernel = 'linear')

evc = VotingClassifier(estimators =[('lr', lr),('rf', rf),('bg', bg),('clf', clf)], 
                       voting = 'hard')
evc.fit(y_train, X_train)

predicted_data = evc.predict(y_test)
print ('Score of the Model:')
print (evc.score(y_test, X_test))
print ('Confusion Matrix:')
print (confusion_matrix(X_test, predicted_data))