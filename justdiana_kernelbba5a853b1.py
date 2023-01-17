import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from keras.models import Sequential
from sklearn.datasets import load_iris 
from sklearn.datasets import make_moons 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import os
#print(os.listdir("../input"))

#Dataframe input
df_train = pd.read_csv('../input/horse.csv')
#df_train.dtypes
#print(df_train.isnull().sum())
#df_train.head()

#Dropping columns with NaN data>100 and NaN data after
print('Dropping data')
print('Old size: %d' % len(df_train))
df_train = df_train.drop(columns=['nasogastric_tube', 'nasogastric_reflux','nasogastric_reflux_ph','rectal_exam_feces','rectal_exam_feces','abdomen','abdomo_appearance','abdomo_protein','cp_data'])
df_train = df_train.dropna(how = 'any', axis = 'rows')
print('New size: %d\n' % len(df_train))

#Converting data from String to Int
df_train['surgery'] = np.where(df_train['surgery']=='no',0,1)
df_train['age'] = np.where(df_train['age']=='young',0,1)
df_train['temp_of_extremities'] = np.select([
    df_train['temp_of_extremities']=='normal',
    df_train['temp_of_extremities']=='warm',
    df_train['temp_of_extremities']=='cool',
    df_train['temp_of_extremities']=='cold',
],[0,1,2,3])
df_train['peripheral_pulse'] = np.select([
    df_train['peripheral_pulse']=='normal',
    df_train['peripheral_pulse']=='increased',
    df_train['peripheral_pulse']=='reduced',
    df_train['peripheral_pulse']=='absent',
],[0,1,2,3])
df_train['mucous_membrane'] = np.select([
    df_train['mucous_membrane']=='normal_pink',
    df_train['mucous_membrane']=='bright_pink',
    df_train['mucous_membrane']=='pale_pink',
    df_train['mucous_membrane']=='pale_cyanotic',
    df_train['mucous_membrane']=='bright_red',
    df_train['mucous_membrane']=='dark_cyanotic',
],[0,1,2,3,4,5])
df_train['capillary_refill_time'] = np.where(df_train['capillary_refill_time']=='less_3_sec',0,1)
df_train['pain'] = np.select([
    df_train['pain']=='alert',
    df_train['pain']=='depressed',
    df_train['pain']=='mild_pain',
    df_train['pain']=='severe_pain',
    df_train['pain']=='extreme_pain',
],[0,1,2,3,4])
df_train['peristalsis'] = np.select([
    df_train['peristalsis']=='hypermotile',
    df_train['peristalsis']=='normal',
    df_train['peristalsis']=='hypomotile',
    df_train['peristalsis']=='absent',
],[0,1,2,3])
df_train['abdominal_distention'] = np.select([
    df_train['abdominal_distention']=='none',
    df_train['abdominal_distention']=='slight',
    df_train['abdominal_distention']=='moderate',
    df_train['abdominal_distention']=='severe',
],[0,1,2,3])
df_train['outcome'] = np.select([
    df_train['outcome']=='died',
    df_train['outcome']=='euthanized',
    df_train['outcome']=='lived',
],[0,0,1])
df_train['surgical_lesion'] = np.where(df_train['surgical_lesion']=='no',0,1)
#df_train.head()

x = df_train.loc[:,df_train.columns!='outcome']
y = df_train['outcome']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4, random_state=0)
sns.pairplot(x)

#LogisticRegression
logReg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
logReg = logReg.fit(X_train, Y_train)
Y_pred = logReg.predict(X_test)
log_regr_score2 = logReg.score(X_test, Y_test)
print('log_regr_score ',log_regr_score2)
print(confusion_matrix(Y_test, Y_pred))

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4, random_state=0)

#Decision Tree
decTree = DecisionTreeClassifier(max_depth=10)
decTree = decTree.fit(X_train, Y_train)
Y_pred = decTree.predict(X_test)
dec_tree_score2 = decTree.score(X_test, Y_test)
print('dec_tree_score ',dec_tree_score2)
print(confusion_matrix(Y_test, Y_pred))

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4, random_state=0)

#RandomForestClassifier
ranTree = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
ranTree = ranTree.fit(X_train, Y_train)
Y_pred = ranTree.predict(X_test)
rand_frst_score2 = ranTree.score(X_test, Y_test)
print('ran_dec_tree_score ',rand_frst_score2)
print(confusion_matrix(Y_test, Y_pred))

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4, random_state=0)

#GradientBoostingClassifierx
gradient = RandomForestClassifier()
gradient = gradient.fit(X_train, Y_train)
Y_pred = gradient.predict(X_test)
grand_boost_score2 = gradient.score(X_test, Y_test)
print('dec_tree_score ',grand_boost_score2)
print(confusion_matrix(Y_test, Y_pred))

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4, random_state=0)

#KNeighborsClassifier 
kneib = KNeighborsClassifier(n_neighbors=3)
kneib = kneib.fit(X_train, Y_train)
Y_pred = kneib.predict(X_test)
KNbrs_score2 = kneib.score(X_test, Y_test)
print('KNbrs_score ',KNbrs_score2)
print(confusion_matrix(Y_test, Y_pred))

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4, random_state=0)

#Support vector machine
supVec = SVC(gamma='auto')
supVec = supVec.fit(X_train, Y_train)
Y_pred = supVec.predict(X_test)
supVec_score2 = supVec.score(X_test, Y_test)
print('supVec_score ',supVec_score2)
print(confusion_matrix(Y_test, Y_pred))

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4, random_state=0)

#Naive Bayes
naiveB = GaussianNB()
naiveB = naiveB.fit(X_train, Y_train)
Y_pred = naiveB.predict(X_test)
NB_score2 = naiveB.score(X_test, Y_test)
print('NB_score ',NB_score2)
print(confusion_matrix(Y_test, Y_pred))

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4, random_state=0)

#MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
mlp = mlp.fit(X_train, Y_train)
irisY_pred = mlp.predict(X_test)
mlp_score2 = mlp.score(X_test, Y_test)
print('MLP_score ',mlp_score2)
print(confusion_matrix(Y_test, Y_pred))
