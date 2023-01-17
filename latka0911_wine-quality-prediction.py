# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
wine_df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
print(wine_df.head(10))

print(wine_df.isnull().sum())

print(wine_df.describe())
plt.bar(wine_df['quality'] , wine_df['pH'])

plt.ylabel('pH Values')

plt.xlabel('Quality of the Wine')

plt.show()
sns.boxplot(x='quality' , y='pH' , data=wine_df)

plt.title('pH VS Quality')

plt.show()
plt.figure(figsize=(10,8))

sns.barplot(x='quality' , y='alcohol' , data=wine_df)

plt.xlabel('Quality',fontsize=12)

plt.ylabel('Alcohol',fontsize=12)
plt.figure(figsize=(10,8))

sns.barplot(x='quality' , y='volatile acidity' , data=wine_df)

plt.xlabel('Quality',fontsize=12)

plt.ylabel('Volatile Acidity',fontsize=12)
plt.figure(figsize=(10,8))

sns.barplot(x='quality' , y='citric acid' , data=wine_df)

plt.xlabel('Quality',fontsize=12)

plt.ylabel('Citric Acid',fontsize=12)
wine_df['quality'].unique()
# First Convert The Quality of the wine into Two Categories as good or bad

bins = (3,6.5,8)

labels = ['bad','good']

wine_df['quality'] = pd.cut(wine_df['quality'] , bins=bins , labels=labels)
wine_df['quality']
# Now Converting the Categorical Features into Numbers Using Label Encoder

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



## Sorry But Here The Label Encoder Was Not Working And Due To this I had to use the Pandas Get_dummies Function
wine_df['quality_encoded'] = pd.get_dummies(wine_df['quality'] , drop_first=True)
wine_df.head(10)
wine_df['quality_encoded'].value_counts()
X = wine_df.drop(['quality' , 'quality_encoded'] , axis='columns')

y = wine_df['quality_encoded'].values
# Splitting the data into train_test_split

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.15)
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
X_train = scale.fit_transform(X_train)

X_test = scale.transform(X_test)
from sklearn.linear_model import LogisticRegression

regressor = LogisticRegression(solver='liblinear')
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)

cm

print(accuracy_score(y_test,y_pred))
# Using Heatmap To plot The confusion matrix

sns.heatmap(cm,annot=True,cmap='coolwarm')

plt.xlabel('Truth Value')

plt.ylabel('Predicted Value')
from sklearn.svm import SVC
svc = SVC(C=1,gamma=0.9)
svc.fit(X_train,y_train)
y_pred_svc = svc.predict(X_test)
cm_svc = confusion_matrix(y_test,y_pred_svc)

print(cm_svc)

print(accuracy_score(y_test,y_pred_svc))
'''

'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 

'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 

'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',

'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 

'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 

'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r',

'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r',

'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r',

'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray',

'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 

'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno',

'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma',

'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10',

'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r',

'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'

'''
# Using Heatmap To plot The confusion matrix

sns.heatmap(cm,annot=True,cmap='autumn')

plt.xlabel('Truth Value')

plt.ylabel('Predicted Value')
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=200)
classifier.fit(X_train,y_train)
y_pred_rf = classifier.predict(X_test)
cm_rf = confusion_matrix(y_test,y_pred_rf)

print(cm_rf)

print(accuracy_score(y_test,y_pred_rf))
# Using Heatmap To plot The confusion matrix

sns.heatmap(cm_rf,annot=True,cmap='ocean_r')

plt.xlabel('Truth Value')

plt.ylabel('Predicted Value')
from sklearn.model_selection import GridSearchCV

parameters = {'kernel' : ('linear' , 'rbf') , 'C' : [1,10,15,20] , 'gamma' : [0.3,1.0,0,85,0.6,0.9]}

clf = GridSearchCV(svc , param_grid=parameters , cv=10)
clf.fit(X_train,y_train)
y_pred_gscv = clf.predict(X_test)
clf.best_params_
# Applying it to the SVC Algorithm we get....

svc = SVC(C=10,kernel='rbf',gamma=1.0)

svc.fit(X_train,y_train)
y_new_scv_pred = svc.predict(X_test)
cm_new_svc = confusion_matrix(y_test,y_new_scv_pred)

print(cm_new_svc)

print(accuracy_score(y_test,y_new_scv_pred))
from sklearn.model_selection import cross_val_score

validation = cross_val_score(classifier,X_train,y_train,cv=10)
print(np.argmax(validation))

# So at position 8 we got the maximum accuracy of the model

print(validation[8])
# Using Heatmap To plot The confusion matrix

sns.heatmap(cm_rf,annot=True,cmap='ocean_r')

plt.xlabel('Truth Value')

plt.ylabel('Predicted Value')