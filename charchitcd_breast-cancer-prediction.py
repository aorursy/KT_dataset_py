import pandas as pd

import numpy as np

import math

import itertools

import seaborn as sns

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d



from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

from sklearn import preprocessing

from sklearn.decomposition import PCA 

from sklearn.manifold import TSNE

from sklearn.feature_selection import SelectKBest,chi2

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

from xgboost import XGBClassifier



%matplotlib inline



# Set Random Seed



np.random.seed(42)

np.random.RandomState(42)
from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
def train(Cancer_models):

  db = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")

  df = pd.DataFrame(db)

  df['Outcome'] = df['Outcome'].apply(lambda x: '1' if x == 'R' else '0')

  df['Outcome'] = pd.to_numeric(df['Outcome'], downcast = "integer")

  X = db.drop(['Outcome', 'ID'], axis = 1)

  y = db['Outcome']

  X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=42)

  df = pd.DataFrame(X_test)

 

  kf = KFold(n_splits = 5, shuffle = True, random_state = 2)

  result = next(kf.split(db), None)

  sc = preprocessing.StandardScaler()

  X_train = sc.fit_transform(X_train)

  X_test = sc.transform(X_test)

  Cancer_models[0].append(LogisticRegression().fit(X_train,y_train))

  Cancer_models[1].append(KNeighborsClassifier().fit(X_train,y_train))

  Cancer_models[2].append(RandomForestClassifier().fit(X_train,y_train))

  return Cancer_models
 
def predict(Cancer_models):

  print('Input: radius_mean	texture_mean	perimeter_mean	area_mean	smoothness_mean	compactness_mean	concavity_mean	concave_points_mean	symmetry_mean	fractal_dimension_mean	radius_std_dev	texture_std_dev	perimeter_std_dev	area_std_dev	smoothness_std_dev	compactness_std_dev	concavity_std_dev	concave_points_std_dev	symmetry_std_dev	fractal_dimension_std_dev	Worst_radius	Worst_texture	Worst_perimeter	Worst_area	Worst_smoothness	Worst_compactness	Worst_concavity	Worst_concave_points	Worst_symmetry	Worst_fractal_dimension	Tumor_Size	Lymph_Node_Status')

  x_pred = list(map(float, input().split()))

  x_pred = np.expand_dims(x_pred, axis = 0)

  predict1 = Cancer_models[0][1].predict(x_pred)

  predict2 = Cancer_models[1][1].predict(x_pred)

  predict3 = Cancer_models[2][1].predict(x_pred)

  print('LR: ', predict1)

  print('KNN: ', predict2)

  print('RF: ', predict3)
def BreastCancerA(f):

  print('1: Train Again, 2: Load Trained Model')

  Cancer_models = [["LogisticRegression"],

                ["KNeighborsClassifier"],

                ["RandomForestClassifier"]]

  # f = int(input())

  if(f == 1):

    Cancer_models = train(Cancer_models)

  elif(f == 2):

    model1 = pickle.load(open('BCModel.sav', 'rb'))

    model2 = pickle.load(open('BCModel.sav', 'rb'))

    model3 = pickle.load(open('BCModel.sav', 'rb'))

    Cancer_models[0].append(model1)

    Cancer_models[1].append(model2)

    Cancer_models[2].append(model3)

  else:

    print('Incorrect input!')

  predict(Cancer_models)
# BreastCancerA(1)

# input: 15.30	25.27	102.4	732.4	0.10820	0.1697	0.16830	0.08751	0.1926	0.06540	0.4390	1.0120	3.498	43.50	0.005233	0.03057	0.03576	0.01083	0.01768	0.002967	20.27	36.71	149.30	1269.0	0.1641	0.6110	0.6335	0.2024	0.4027	0.09876	2.0	0
# def BreastCancerMain():

#   # print('1: Task A, 2: Task B')

#   # f = int(input())

#   # if(f == 1):

#   #   BreastCancerA()

#   # elif(f == 2):

#   #   BreastCancerB()

#   # else:

#   #   print('Incorrect input!')
db= pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")

db.head(6)
db.tail()
db.describe()
db.isnull().sum()
df = pd.DataFrame(db)

# from sklearn.preprocessing import OneHotEncoder

# # creating instance of one-hot-encoder

# enc = OneHotEncoder(handle_unknown='ignore')



# enc_df = pd.DataFrame(enc.fit_transform(df[['Outcome']]).toarray())



# df = df.join(enc_df)

# df
df['diagnosis'] = df['diagnosis'].apply(lambda x: '1' if x == 'M' else '0')

df
df['diagnosis'] = pd.to_numeric(df['diagnosis'], downcast = "integer")
# db['Outcome'] = db['Outcome'].apply(lambda x:'1' if x == 'R' else '0')



# db.head(20)
# df = pd.DataFrame(db)

# df
sns.lineplot(x=db["radius_mean"],y=db["perimeter_mean"], hue=db["diagnosis"])
#Countplot

sns.countplot(db['diagnosis'])
# Barplot: diagnosis vs area_mean.

sns.barplot(db['diagnosis'],db['area_mean'])
sns.scatterplot(x = db['area_mean'],y= db['smoothness_mean'],hue=db['diagnosis'])
X = db.drop(['diagnosis','id','Unnamed: 32'], axis = 1)

y = db['diagnosis']



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=42)

# x_train, _test = train_test_split(a, test_size=0.40, random_state=0)

X_train.head()
X_test.head()

# from sklearn.preprocessing import LabelBinarizer



# encoder = LabelBinarizer()

# Y = encoder.fit_transform(X_train)
# Saving test data

df = pd.DataFrame(X_test)

df.to_excel('testdata.xlsx')

df.dropna()

df
y_test
from sklearn.model_selection import KFold

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)

result = next(kf.split(db), None)

print (result)






sc = preprocessing.StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
Cancer_models = [["LogisticRegression",LogisticRegression()],

                ["KNeighborsClassifier",KNeighborsClassifier()],

                ["RandomForestClassifier",RandomForestClassifier()]]
acc = []

for i in Cancer_models:

  log = i[1]

  log.fit(X_train,y_train)

  predict = log.predict(X_test)

  acc.append([i[0],accuracy_score(predict,y_test)*100.0])

   
main_score = pd.DataFrame(acc)

main_score.columns = ["Model","Score"]
print("Accuracy Scores:")

main_score
conf_mat = confusion_matrix(y_test, predict)

class_label = ["negative", "positive"]

df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)

sns.heatmap(df, annot = True,fmt="d")

plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.show()


df