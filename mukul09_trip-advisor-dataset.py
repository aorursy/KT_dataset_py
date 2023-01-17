import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn import gaussian_process
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.feature_selection import RFECV


df = pd.read_csv('../input/trip-advisor.csv')
data1=df.copy(deep=True)
X = df.drop(['Score'],axis=1)
y = df['Score'].values

# check the sum of num values (if any)
df.isnull().sum()

# Features need to be encoded
encode_list=['User country code','Period of stay code','Traveler type code','Swimming Pool code','Exercise Room code','Basketball Court code',
             'Yoga Classes code','Club code','Free Wifi code','Hotel name code','Hotel stars code','Nr. rooms code',
             'User continent code','Review month code','Review weekday code']

#Features not need to be encoded
not_encode_list=['Nr. reviews','Nr. hotel reviews','Helpful votes','Member years']
# Label Encoding of the Categorical data
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data1['User country code'] = label.fit_transform(data1['User country'])
data1['Period of stay code'] = label.fit_transform(data1['Period of stay'])
data1['Traveler type code'] = label.fit_transform(data1['Traveler type'])
data1['Swimming Pool code'] = label.fit_transform(data1['Swimming Pool'])
data1['Exercise Room code'] = label.fit_transform(data1['Exercise Room'])
data1['Basketball Court code'] = label.fit_transform(data1['Basketball Court'])
data1['Yoga Classes code'] = label.fit_transform(data1['Yoga Classes'])
data1['Club code'] = label.fit_transform(data1['Club'])
data1['Free Wifi code'] = label.fit_transform(data1['Free Wifi'])
data1['Hotel name code'] = label.fit_transform(data1['Hotel name'])
data1['Hotel stars code'] = label.fit_transform(data1['Hotel stars'])
data1['Nr. rooms code'] = label.fit_transform(data1['Nr. rooms'])
data1['User continent code'] = label.fit_transform(data1['User continent'])
data1['Review month code'] = label.fit_transform(data1['Review month'])
data1['Review weekday code'] = label.fit_transform(data1['Review weekday'])


X=pd.concat([data1[not_encode_list],data1[encode_list]],axis=1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
names=X.columns
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
scaled_X = pd.DataFrame(X, columns=names)
scaled_X.head()
# Correlation Heatmap
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(scaled_X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#Machine Learning Algorithm (MLA) Selection and Initialization
MLA=[
    GradientBoostingClassifier(),
    RandomForestClassifier(),
    
    LogisticRegressionCV(),
    KNeighborsClassifier(),
        
    DecisionTreeClassifier(),    
    XGBClassifier()
    
]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_X.iloc[:,0:20], y, test_size = 0.2, random_state = 0)

MLA_columns = ['MLA Name', 'MLA Parameters', 'MLA Test Accuracy']
MLA_compare1=pd.DataFrame(columns=MLA_columns)

MLA_predict=y


for count,alg in enumerate(MLA):
    classifier = alg
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test) 
    MLA_name=alg.__class__.__name__
    MLA_compare1.loc[count, 'MLA Name']=MLA_name
    MLA_compare1.loc[count, 'MLA Parameters']=str(alg.get_params())
    MLA_compare1.loc[count, 'MLA Test Accuracy']= accuracy_score(y_test, y_pred)
    
MLA_compare1.sort_values(by=['MLA Test Accuracy'], ascending=False, inplace=True)

MLA_compare1
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegressionCV

MLA_columns = ['MLA Name','MLA Features', 'Number of MLA Features']
MLA_compare2=pd.DataFrame(columns=MLA_columns)

MLA=[
    GradientBoostingClassifier(),
    RandomForestClassifier(),
    LogisticRegressionCV(),   
    XGBClassifier()
]

fig = plt.figure(figsize=(12,7))
plt.xticks([])
plt.ylabel("Cross validation score of number of selected features")
    
for count, alg in enumerate(MLA):
    MLA_name = alg.__class__.__name__
    MLA_compare2.loc[count, 'MLA Name'] = MLA_name
    clf_rf_4 =  alg
    rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation
    rfecv = rfecv.fit(X_train, y_train)
    x=pd.DataFrame(X_train.columns[rfecv.support_],columns=['Models'])
    MLA_compare2.loc[count,'MLA Features'] = list(x['Models'])
    MLA_compare2.loc[count,'Number of MLA Features'] = rfecv.n_features_
    
    
    ax = fig.add_subplot(2, 2, count+1)
    plt.xlabel(MLA_name)

    major_ticks = np.arange(1, 20, 1)
    minor_ticks = np.arange(0, 20, 1)

    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    
    ax.grid(which='both')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    #ax.set_axisbelow(True)

    ax.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.tight_layout()
plt.show()
    

MLA_compare2.sort_values(by=['Number of MLA Features'],inplace=True)
MLA_compare2
from sklearn.cross_validation import train_test_split

MLA=[
    GradientBoostingClassifier(),
    RandomForestClassifier(),
    LogisticRegressionCV(),   
    XGBClassifier()
]
MLA_compare3 = pd.DataFrame(columns=['MLA Name','Accuracy'])
for count,alg in enumerate(MLA):
    features = MLA_compare2.loc[count, 'MLA Features']
    
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(scaled_X.loc[:,features], y, test_size = 0.2, random_state = 0)
    
    MLA_name = alg.__class__.__name__
    MLA_compare3.loc[count, 'MLA Name'] = MLA_name 
    classifier = alg
    classifier = classifier.fit(X_train_final,y_train_final)
    ac = accuracy_score(y_test_final,classifier.predict(X_test_final))
    MLA_compare3.loc[count, 'Accuracy'] = ac
    
MLA_compare3.sort_values(by=['Accuracy'],inplace=True, ascending=False)
MLA_compare3.index = MLA_compare3.index.sort_values()

MLA_compare3

from sklearn import linear_model
clf_rf = LogisticRegressionCV()

for c in range(MLA_compare2.shape[0]):
    if MLA_compare3.loc[0, 'MLA Name'] == MLA_compare2.loc[c,'MLA Name']:
        feat = MLA_compare2.loc[c, 'MLA Features']
        print(len(feat))
X_train_best, X_test_best, y_train_best, y_test_best = train_test_split(scaled_X.loc[:,feat], y, test_size = 0.2, random_state = 0)
clr_rf = clf_rf.fit(X_train_best,y_train_best)

ac = accuracy_score(y_test_best,clf_rf.predict(X_test_best))
print('Accuracy is: ',ac)
cm = confusion_matrix(y_test,clf_rf.predict(X_test_best))
sns.heatmap(cm,annot=True,fmt="d")
