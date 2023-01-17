import pandas as pd
import pandas_profiling as pp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os
import plotly.graph_objects as go
import plotly.io as pio

# Metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Validation
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.pipeline import Pipeline

# Tuning
from sklearn.model_selection import GridSearchCV

# Feature Extraction
from sklearn.feature_selection import RFE

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer, LabelEncoder

# Models
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Ensembles
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')
%matplotlib inline

sns.set_style("whitegrid", {'axes.grid' : False})
pio.templates.default = "plotly_white"

df = pd.read_csv('/kaggle/input/early-stage-diabetes-risk-prediction-datasets/diabetes_data_upload.csv')
df.head()

print("Number of Instances and Attributes:", df.shape)
df.columns
print(df.dtypes)
df.isna().sum()

df.columns = df.columns.str.replace(" ", "_")
df.rename(columns={'weakness':'Weakness', 'visual_blurring':'Visual_blurring', 'delayed_healing':'Delayed_healing', 'partial_paresis':'Partial_paresis','muscle_stiffness':'Muscle_stiffness'}, inplace=True)
labels = ['Male','Female']
values = df.Gender.value_counts()

colors = ['STEELBLUE','crimson']

fig_1 = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', showlegend=False)])

fig_1.update_traces(marker=dict(colors=colors, line=dict(color='#000000', width=0.5)))

fig_1.update_layout( margin={"r":0,"t":100,"l":0,"b":0},
    title_text="<br>Gender Distribution:<br>",
    font=dict(size=15, color='black', family="Arial, Balto, Courier New, Droid Sans"),

)
fig_1.show()
f,ax=plt.subplots(1,2,figsize=(22,10))
#sns.stripplot(x="class", y="Age", data=df, jitter=True, palette="Set1", ax=ax[0])

sns.swarmplot(x="class", y="Age",data=df, palette="Set1", ax=ax[0])
sns.violinplot(x="class", y="Age", data=df, palette="Set1", ax=ax[1])
f.suptitle('Age of Positives vs Negatives', fontweight="bold");


plt.figure(figsize=(12,8))
sns.countplot(x="class", hue="Gender", palette=['STEELBLUE','crimson'],data=df);
plt.title('Gender of Positives vs Negatives',fontweight="bold");
plt.figure(figsize=(10,5))
sns.countplot(x=df['class'], palette='Set1');
plt.title('Class Distribution',fontweight="bold",alpha=0.8);
for c in df.columns:
    if df[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(df[c].values)) 
        df[c] = lbl.transform(list(df[c].values))
        
df.describe().T
colormap = plt.cm.RdBu
plt.figure(figsize=(18,15));
plt.title('Pearson Correlation of Features', y=1.05, size=50);
sns.heatmap(df.corr(),linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True);
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X,
                                                   y,
                                                    test_size=0.2,
                                                   random_state=0)
print('X_train: ',X_train.shape)
print('X_test: ',X_test.shape)
print('y_train: ',y_train.shape)
print('y_test: ',y_test.shape)
from sklearn.model_selection import KFold
models = []
models.append(( ' LR ' , LogisticRegression()))
models.append(( ' LDA ' , LinearDiscriminantAnalysis()))
models.append(( ' KNN ' , KNeighborsClassifier()))
models.append(( ' NB ' , GaussianNB()))
models.append(( ' SVM ' , SVC()))

results = []
names = []

for name, model in models:
    Kfold = KFold(n_splits=10, random_state=0)
    cv_results = cross_val_score(model, X_train, y_train, cv=Kfold, scoring= 'accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std());
    print(msg)
models = []
models.append(( 'Adab' , AdaBoostClassifier()))
models.append(( 'Bagging' , BaggingClassifier()))
models.append(( 'GBC' , GradientBoostingClassifier()))
models.append(( 'RF' , RandomForestClassifier()))


results = []
names = []

for name, model in models:
    Kfold = KFold(n_splits=10, random_state=0)
    cv_results = cross_val_score(model, X_train, y_train, cv=Kfold, scoring= 'accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std());
    print(msg)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)


model = RandomForestClassifier()
param_grid = { 
    'n_estimators': [10,20,50,100],
    'max_features': ['auto', 'sqrt', 'log2']
}

kfold = KFold(n_splits=10, random_state=0)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
grid_result = grid.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
RF = RandomForestClassifier(max_features='auto', n_estimators=50).fit(X_train,y_train)
y_pred = RF.predict(X_test)
print(accuracy_score(y_test, y_pred))
plt.figure(figsize=(10,5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='2.0f');
plt.title("Confusion Matrix",fontweight="bold");
print(classification_report(y_test,y_pred))
