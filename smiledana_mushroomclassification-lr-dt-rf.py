# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
# Import Data
df = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
print('Shape of dataset= ', df.shape) # Numer of rows and columns
df.head(5) # first 5 data
# Data type
df.info()
# Missing value check
df.isna().sum() # Get number of missing value from each column
# Frequency Tables
for col in df.columns:
    print('-' * 40 + col + '-' * 40 )
    #print(pd.DataFrame(df[col].value_counts()))
    print(df[col].value_counts())
# Histogram
df = df.drop('veil-type', axis=1)
df2 = df.melt(value_vars=df.columns)
g = sns.FacetGrid(df2, col="variable", col_wrap=6)
g = g.map(sns.countplot, "value")
from sklearn.preprocessing import LabelEncoder
df3 = pd.DataFrame()
for i in df.columns:
    enc = LabelEncoder()
    df3[i] = enc.fit_transform(df[i]) 
print(df3.head(3))
#Correlation Matrix - Upper Diagonal
import seaborn as sn
corrMatrix = df3.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corrMatrix, dtype=np.bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 20, sep=20,as_cmap=True)

sn.heatmap(corrMatrix, mask=mask, cmap=cmap)
plt.show()
#Split dataset & Standardize data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X = df3.drop('class', axis=1)
y = df3['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#Model Selection Using Grid Search
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([('classifier' , LogisticRegression())])

# Create param grid.
param_grid = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l1', 'l2'],
    'classifier__C' : np.logspace(-4, 4, 20),
    'classifier__solver' : ['liblinear']},
    {'classifier' : [RandomForestClassifier()],
    'classifier__n_estimators' : list(range(10,101,10)),
    'classifier__max_features' : list(range(6,32,5))},
    {'classifier' : [DecisionTreeClassifier()],
    'classifier__criterion' : ['gini','entropy'],
    'classifier__max_depth' : list(range(4,150,20))}
]

# Create grid search object
CV = GridSearchCV(pipe, 
                  param_grid = param_grid, 
                  cv = 5, 
                  #scoring="neg_mean_squared_error",
                  verbose=True, n_jobs=-1)

# Fit on data
best_model= CV.fit(X_train, y_train)
# View best model
print("Best parameters found: ", best_model.best_estimator_.get_params()['classifier'])#best parameters and lowest RMSE
print("Highest Score found: ", round(CV.best_score_,2))
#ROC Curve
from sklearn.metrics import roc_curve, auc
model = RandomForestClassifier(criterion='gini',max_features=6,min_samples_leaf=1, min_samples_split=2,n_estimators=10)
model.fit(X_train,y_train)
probs = model.predict_proba(X_test)
y_prob = probs[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure(figsize=(7,7))
plt.title('Receiver Operating Characteristic(ROC) Curve')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')