%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotly.plotly as py
import plotly.graph_objs as go 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
input_file = "../input//train.csv"

df=pd.read_csv(input_file)
df.head()
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df.head()
df.info()
df['Embarked'].fillna('S', inplace=True) #Just replace the missing 3 'Embarked' with 'S'
df.info()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
sex_cat = df['Sex']
emb_cat = df['Embarked']
sex_cat_encoded = encoder.fit_transform(sex_cat)
emb_cat_encoded = encoder.fit_transform(emb_cat)
df['Sex'] = sex_cat_encoded 
df['Embarked'] = emb_cat_encoded
df.head()
df.describe()
df.corr()
df.hist(bins=50, figsize= (16,10))
plt.show
df['Age'].fillna(29.7, inplace=True)
#PLEASE NOTE! Could use   sklearn.preprocessing.Imputer()   also!
X_train = df.drop(['Survived'], axis=1)
y_train = df['Survived']
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_array_train = X_train.as_matrix() # creating an array
y_array_train = y_train.as_matrix() # creating an array
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)

scaled_data = scaler.transform(X_train) 
import seaborn as sns
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))

scaled_df = scaler.fit_transform(X_train)
scaled_df = pd.DataFrame(scaled_data, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

ax1.set_title('Before Scaling')
sns.kdeplot(df['Pclass'], ax=ax1)
sns.kdeplot(df['Sex'], ax=ax1)
sns.kdeplot(df['Age'], ax=ax1)
sns.kdeplot(df['SibSp'], ax=ax1)
sns.kdeplot(df['Parch'], ax=ax1)
sns.kdeplot(df['Fare'], ax=ax1)
sns.kdeplot(df['Embarked'], ax=ax1)

ax2.set_title('After Standard Scaler')
sns.kdeplot(scaled_df['Pclass'], ax=ax2)
sns.kdeplot(scaled_df['Sex'], ax=ax2)
sns.kdeplot(scaled_df['Age'], ax=ax2)
sns.kdeplot(scaled_df['SibSp'], ax=ax2)
sns.kdeplot(scaled_df['Parch'], ax=ax2)
sns.kdeplot(scaled_df['Fare'], ax=ax2)
sns.kdeplot(scaled_df['Embarked'], ax=ax2)


plt.show()
input_file = "../input//test.csv"

df_test=pd.read_csv(input_file)
df_test.head()
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_test.head()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
sex_cat = df_test['Sex']
emb_cat = df_test['Embarked']
sex_cat_encoded = encoder.fit_transform(sex_cat)
emb_cat_encoded = encoder.fit_transform(emb_cat)
df_test['Sex'] = sex_cat_encoded 
df_test['Embarked'] = emb_cat_encoded
df_test.head()
df_test.info()
df_test.describe()
df_test['Age'].fillna(30.3, inplace=True)
df_test['Fare'].fillna(35.63, inplace=True) 
X_test = df_test.drop(['PassengerId'], axis=1)
features_test = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_array_test = X_test.as_matrix() # creating an array
X_test.head()
from sklearn.svm import SVC
svm_linear = SVC( kernel = 'linear')
svm_linear.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
svm_linear_cv = cross_val_score(svm_linear, X_train, y_train, cv=10)
print(svm_linear_cv.mean())
from sklearn.svm import SVC
svm_rbf = SVC( kernel = 'rbf')
svm_rbf.fit(X_train, y_train) 
svm_rbf_cv = cross_val_score(svm_rbf, X_train, y_train, cv=10)
print(svm_rbf_cv.mean())
from sklearn.neighbors import KNeighborsClassifier
for K in range(50):
    K_value = K+1
    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
    neigh.fit(X_train, y_train)
    mean_k=(cross_val_score(neigh, X_train, y_train, cv=5)*100).mean()
    print ("Accuracy is ", mean_k,"% for K-Value:",K_value)  

k_values = [68.12712449557361,68.1271315883079,70.59719404337987,69.47548875145534,69.70902412132548,68.02739710481679,70.61296473811674,69.47925499337362,69.71152785653668,69.70965892104616,69.71155622747392,69.93627532859753,70.04674466546587,70.71965364759822,69.82327388558073,70.0499044786007,70.04926967888 ,70.83706322498827,70.15909712329335,70.49681057469944,70.38823854425884,71.62043445125465,70.60854241827487,70.04673757273156,70.49808017414082,70.71965364759821,70.60918431072986,70.94375213446973,70.15973192301405,69.93501282189045,69.82201847160796,69.82201847160796,69.59981019842988,68.81139603806955,69.48492563445396,69.59729937048436,68.81011934589387,69.14532196935443,68.92185828220359,68.8088639319211,68.58161272464906,68.02044267882646,69.03423201823401,69.25768861265054,68.92375558863134,68.81266563751093,68.4762076000776,68.25085369923332,68.58919485762577,67.57606159614184]

plt.plot(k_values)
plt.legend(['K-values'], loc=2)
plt.xlabel('K-values in range 1-50')
plt.ylabel('Accuracy score in %')
plt.title('Accuracies with different K-values')

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] =16
fig_size[1] = 4
plt.rcParams["figure.figsize"] = fig_size
plt.grid(True)
plt.show();
neigh = KNeighborsClassifier(n_neighbors=22)
neigh.fit(X_train, y_train) 
neigh_cv = cross_val_score(neigh, X_train, y_train, cv=10)
print(neigh_cv.mean()) 
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
X_minmax = scaler.fit_transform(X_train)

mnb = MultinomialNB()
mnb_cv = cross_val_score(mnb, X_minmax, y_train, cv=10)
print(mnb_cv.mean())
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)
logreg_cv = cross_val_score(logreg, X_train, y_train, cv=10)
print(logreg_cv.mean())
from keras.layers import Dense, Dropout
from keras.models import Sequential

def create_model():
    model = Sequential()
    
    model.add(Dense(64, input_dim=7, kernel_initializer='normal', activation='relu'))
   
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
   
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
   
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

from keras.wrappers.scikit_learn import KerasClassifier

# Wrapping the Keras model in an estimator compatible with scikit_learn
estimator = KerasClassifier(build_fn=create_model, nb_epoch=100, verbose=0)
# Now we can use scikit_learn's cross_val_score to evaluate this model identically to the others
cv_scores = cross_val_score(estimator, X_train, y_train, cv=10)
print(cv_scores.mean()) 
from sklearn import tree
dtc = tree.DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
import pydotplus
from sklearn.externals.six import StringIO  
from IPython.display import Image
dot_data = StringIO()  
tree.export_graphviz(dtc, out_file=dot_data,  
                         feature_names=features)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png()) 
dtc_cv = cross_val_score(dtc, X_train, y_train, cv=10)
print(dtc_cv.mean())
from sklearn.ensemble import RandomForestClassifier 
forest_reg = RandomForestClassifier(random_state=42)
forest_reg.fit(X_train, y_train)
forest_reg_cv = cross_val_score(forest_reg, X_train, y_train, cv=10)
print(forest_reg_cv.mean())
for name, score in zip(X_train[features], forest_reg.feature_importances_):
    print(name, score)
import plotly
plotly.offline.init_notebook_mode()

data = go.Bar(
    y=['Sex','Age','Fare','Pclass','SibSp','Parch','Embarked'],
    x=[0.2805249868635907,0.2660256880057519,0.2602841743055045,0.08185231464807138,0.04327225522263238,0.041154921537327846,0.02688565941712139],   
    orientation = 'h',
    marker = dict(color = 'rgba(255,0,0, 0.6)', line = dict(width = 0.5)))

data = [data]
layout = go.Layout(title = 'Relative Importance of the Features in the Random Forest',
    barmode='group', bargap=0.1, width=800,height=500,)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
print("SVM linear: \t\t", svm_linear_cv.mean())
print("SVM rbf: \t\t", svm_rbf_cv.mean())
print("KNN: \t\t\t", neigh_cv.mean())
print("Naive Bayes: \t\t", mnb_cv.mean())
print("Logistic Regression: \t", logreg_cv.mean())
print("Neural Network: \t", cv_scores.mean())
print("Decision Tree: \t\t", dtc_cv.mean())
print("Random Forest: \t\t", forest_reg_cv.mean())
import plotly
plotly.offline.init_notebook_mode()

data = go.Bar(
    y=['SVM linear:','SVM rbf:','KNN:','Naive Bayes:','Logistic Regression:','Neural Network:','Decision Tree:','Random Forest:'],
    x=[0.7866981613891727,0.7028407672227897,0.7084828623311769,0.6588046192259676,0.7935157757348767,0.6690636708822739,0.7790594143684032,0.8115562932697765],   
    orientation = 'h',
    marker = dict(color = 'rgba(255,0,0, 0.6)', line = dict(width = 0.5)))

data = [data]
layout = go.Layout(title = 'Model Accuracies',
    barmode='group', bargap=0.1, width=800,height=500,)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
X_test.head()
predictions = svm_linear.predict(X_test) #forest_reg
# Create a new dataframe with only the columns Kaggle wants from the data set
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": predictions
    })

submission.to_csv("kaggle.csv", index=False)
#creates a csv-file in the folder with columns PassengerId and Survived