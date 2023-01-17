import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


files = os.listdir("../input")
input_file = "../input/{}".format(files[0])

# Any results you write to the current directory are saved as output.
# Considering a single region
region_id = 6
df = pd.read_csv(input_file, header = 0,usecols=['iyear', 'imonth', 'iday', 'extended', 'country', 'country_txt', 'region', 'latitude', 'longitude','success', 'suicide','attacktype1','attacktype1_txt', 'targtype1', 'targtype1_txt', 'natlty1','natlty1_txt','weaptype1', 'weaptype1_txt' ,'nkill','multiple', 'individual', 'claimed','nkill','nkillter', 'nwound', 'nwoundte'])
# Filetered dataframe
df_Area = df[df.region == region_id]

df_Area.describe()
df_Area.info()




# Dropping the uneccessary columns
df_Region = df_Area.drop([ 'region', 'claimed', 'nkillter', 'nwound','nwoundte'], axis=1)

# Fill NA
df_Region['nkill'].fillna(df_Region['nkill'].mean(), inplace=True)
df_Region['latitude'].fillna(df_Region['latitude'].mean(), inplace=True)
df_Region['longitude'].fillna(df_Region['longitude'].mean(), inplace=True)
df_Region['natlty1'].fillna(df_Region['natlty1'].mean(), inplace=True)

df_Region.info()
# Kill Plot comparison
df_Region.plot(kind= 'scatter', x='longitude', y='latitude', alpha=1.0,  figsize=(18,6),
                   s=df_Region['nkill']*3, label= 'Casualties', fontsize=1, c='nkill', cmap=plt.get_cmap("jet"), colorbar=True)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)
plt.show()
# Verify Correlation Matrix
corrmat = df_Region.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=1, square=True)
X = df_Region.drop(['iyear', 'success','country', 'country_txt', 'attacktype1_txt','targtype1_txt','natlty1', 'natlty1_txt', 'weaptype1_txt'], axis=1)
y = df_Region['success']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print('Accuracy Score: {}'.format(acc))
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print('Mean: {}'.format(accuracies.mean()))
print('SD: {}'.format(accuracies.std()))
# Pass parameters to predict whether an attack is successful or not
# Date params
month = 6
day = 14
# Boolean 0-No,; 1-Yes
extended = 0
# Location
latitude = 28.585836
longitude = 77.153336
# Attack Params
multiple = 0
suicide = 0
attackType = 3
targetType = 7
individual = 0
weaponType = 6
# Aftermath --> Casuality Number
nkill = 0

attack_params = np.array([[(month),(day),(extended),(latitude),(longitude),(multiple),(suicide),(attackType),(targetType),(individual),(weaponType),(nkill)]])
outcome = classifier.predict(attack_params)
result = outcome[0]
outcome_result_dict = {
    0: 'Failure',
    1: 'Success'
}

print('The attack on Region Id: {} will be a {} based on the given parameters.'.format(region_id, outcome_result_dict[result]))