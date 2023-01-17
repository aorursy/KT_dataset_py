import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#import sci-kit learn libraries

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, RobustScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, confusion_matrix
sdss = pd.read_csv('/kaggle/input/sloan-digital-sky-survey-dr16/Skyserver_12_30_2019 4_49_58 PM.csv')

sdss
sdss.info()
print('Number of NaN values for each feature:\n',sdss.isnull().sum())
print('Number of uniques values for each feature:\n',sdss.nunique())
sdss.head()
sdss_features = sdss.drop(columns=['objid', 'ra','dec', 'run', 'rerun', 'camcol', 'field','specobjid', 'plate', 'mjd', 'fiberid'])

sdss_features
sdss_features.describe()
#Filter each class

stars = sdss_features[sdss_features['class'] == 'STAR']

quasars = sdss_features[sdss_features['class'] == 'QSO']

galaxies = sdss_features[sdss_features['class'] == 'GALAXY']
color_palette = 'GnBu_d'

sns.set()

fig = plt.gcf()

fig.set_size_inches(13,9)

sns.countplot(sdss_features['class'], palette=color_palette)

plt.show()
sns.set(style='darkgrid')

fig, axs = plt.subplots(nrows=3)

fig = plt.gcf()

fig.set_size_inches(13,9)

plt.subplots_adjust(hspace=0.8)

sns.boxplot(stars['redshift'], palette=color_palette, ax=axs[0]).set_title('Stars')

sns.boxplot(galaxies['redshift'], palette=color_palette, ax=axs[1]).set_title('Galaxies')

sns.boxplot(quasars['redshift'], palette=color_palette, ax=axs[2]).set_title('Quasars')

plt.show()
sns.set(style='darkgrid')

sns.pairplot(sdss_features, hue='class')

plt.show()
sdss_features_corr = sdss_features.corr()

fig = plt.gcf()

fig.set_size_inches(13,9)

sns.heatmap(sdss_features_corr, annot=True)

plt.show()
sdss_data = sdss_features[['u','g','r','i','z','redshift']]



#Need to factorize the classes or convert to numerical labels to use in model, returns label array and unique value array, only need the first array

sdss_target = pd.factorize(sdss_features['class'])[0]





#Split data 70/30 and set randomstate to 0 to get the same split every time it is split

x_train, x_test, y_train, y_test = train_test_split(sdss_data, sdss_target, test_size=0.30, random_state=0)
robust_scaler = RobustScaler()



#fit_transform will first perform fit and calculates the parameters, then applies transform 

x_train = robust_scaler.fit_transform(x_train)



#just need to transform since fit was already called

x_test = robust_scaler.transform(x_test)
logRegression = LogisticRegression(max_iter=350)



logRegression.fit(x_train, y_train)

predictions = logRegression.predict(x_test)



accuracy = logRegression.score(x_test, y_test)



print('Classification Test Score:', accuracy ,'\n')

print('Classification Performance:\n', classification_report(y_test, predictions),'\n')

print('Train Score:', logRegression.score(x_train,y_train))



cm = confusion_matrix(y_test, predictions)



fig = plt.gcf()

fig.set_size_inches(13,9)

sns.heatmap(cm, annot=True).set_title('Accuracy Score: {}'.format(accuracy))

plt.xlabel('Actual Class')

plt.ylabel('Predicted Class')



plt.show()