import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import missingno as msno 
from sklearn import preprocessing
# bulid model 
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
# cross_val_score
from sklearn.model_selection import cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df= pd.read_csv("../input/mammographic-mass-data-set/Cleaned_data.csv", na_values=['?']) 

df.head()
df.tail()
df.info()
df.describe()
# visualize the location of missing values
msno.matrix(df,color=(0.45, 0.65, 0.65))
#missing amount for data
missing= df.isnull().sum().sort_values(ascending=False)
percentage = (df.isnull().sum()/ df.isnull().count()).sort_values(ascending=False)
missing_recommend_df = pd.concat([missing, percentage], axis=1, keys=['Missing', '%'])
missing_recommend_df.head(6)

#or
# df.loc[(df['BI_RADS'].isnull())|
#               (df['age'].isnull())|
#                 (df['shape'].isnull())|
#                 (df['margin'].isnull())| 
#               (df['density'].isnull()) |
#                  (df['severity'].isnull())]
fig=plt.figure(figsize=(18,10))
ax = fig.gca()
sns.heatmap(df.corr(), annot=True,ax=ax, cmap=plt.cm.YlGnBu)
ax.set_title('The correlations between all features')
palette =sns.diverging_palette(80, 110, n=146)
plt.show()
# correlation with the target
corr_matrix = df.corr()
corr_matrix["Severity"].sort_values(ascending=False)
#array that extracts only the feature data we want to work with (age, shape, margin, and density) 
all_features = df[['Age', 'Shape', 'Margin', 'Density']].values

# array that contains the classes (severity)
all_classes = df['Severity'].values

feature_names = ['Age', 'Shape', 'Margin', 'Density']

print (all_features,'\n')

print (all_classes)


all_features.shape
# normalize the attribute data
scaler = preprocessing.StandardScaler()
scaler_features=scaler.fit_transform(all_features)
scaler_features
# Now set up an actual MLP model using Keras:
def create_model():
    model=Sequential()
    
    # 4 feature inputs going into an 19-unit layer
    model.add(Dense(19,input_dim=4 , kernel_initializer='normal',activation='relu'))
    
    # Output layer with a binary classification (benign or malignant)
    model.add(Dense(1,kernel_initializer='normal', activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    return model
# Wrap our Keras model in an estimator compatible with scikit_learn
estimator= KerasClassifier(build_fn=create_model,nb_epoch=100,verbose=0)
#cross_val_score to evaluate this model 
cv_scores=cross_val_score(estimator,scaler_features,all_classes , cv= 10)
cv_scores.mean() 
