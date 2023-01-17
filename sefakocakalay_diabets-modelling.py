import numpy as np
import pandas as pd 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from warnings import filterwarnings
filterwarnings('ignore')
diabetes = pd.read_csv("../input/diabets/diabetes.csv") #datayı okuma
df = diabetes.copy() #datanın kopyasını alma
df = df.dropna() #boş değerleri çıkartma
df.head() #ilk 5 gözlem
df.info() # genel bilgiler

df["Outcome"].value_counts() # Outcome değerleri
df["Outcome"].value_counts().plot.barh(); #görsel olarak bakma
df.describe().T # istatistiksel bazı değerler
y = df["Outcome"] # bağımlı değişken
X = df.drop(["Outcome"], axis=1) # bağımsız değişken
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42) # datayı test train olarak bölme
svm_model = SVC(random_state = 42).fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
y_pred_train = svm_model.predict(X_train)
print("** Karmaşıklık Matrisi **\n",confusion_matrix(y_test, y_pred))
print("** Test Seti Doğruluk Oranı **\n:",accuracy_score(y_test, y_pred))
print("** Train Seti Doğruluk Oranı **\n:",accuracy_score(y_train, y_pred_train))
print(classification_report(y_test, y_pred)) 
svc_model = SVC(kernel = "linear",random_state = 42).fit(X_train, y_train)
y_pred = svc_model.predict(X_test)
y_pred_train = svc_model.predict(X_train)
print("** Karmaşıklık Matrisi **\n",confusion_matrix(y_test, y_pred))
print("** Test Seti Doğruluk Oranı **\n:",accuracy_score(y_test, y_pred))
print("** Train Seti Doğruluk Oranı **\n:",accuracy_score(y_train, y_pred_train))
print(classification_report(y_test, y_pred)) 
cart = DecisionTreeClassifier(max_depth =8, random_state = 42)
cart_model = cart.fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
y_pred_train = cart_model.predict(X_train)
print("** Karmaşıklık Matrisi **\n",confusion_matrix(y_test, y_pred))
print("** Test Seti Doğruluk Oranı **\n:",accuracy_score(y_test, y_pred))
print("** Train Seti Doğruluk Oranı **\n:",accuracy_score(y_train, y_pred_train))
print(classification_report(y_test, y_pred)) 
knn = KNeighborsClassifier(n_neighbors = 8, weights = 'distance',leaf_size = 39, p = 150)
knn_model = knn.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
y_pred_train = knn_model.predict(X_train)
print("** Karmaşıklık Matrisi **\n",confusion_matrix(y_test, y_pred))
print("** Test Seti Doğruluk Oranı **\n:",accuracy_score(y_test, y_pred))
print("** Train Seti Doğruluk Oranı **\n:",accuracy_score(y_train, y_pred_train))
print(classification_report(y_test, y_pred)) 
rf_model = RandomForestClassifier(random_state = 42, n_estimators=150, max_depth=9, max_leaf_nodes=9).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
y_pred_train = rf_model.predict(X_train)
print("** Karmaşıklık Matrisi **\n",confusion_matrix(y_test, y_pred))
print("** Test Seti Doğruluk Oranı **\n:",accuracy_score(y_test, y_pred))
print("** Train Seti Doğruluk Oranı **\n:",accuracy_score(y_train, y_pred_train))
print(classification_report(y_test, y_pred)) 
svc = SVC(probability=True,kernel='linear')
adb_model = AdaBoostClassifier(base_estimator = svc, random_state = 42).fit(X_train, y_train)
y_pred = adb_model.predict(X_test)
y_pred_train = adb_model.predict(X_train)
print("** Karmaşıklık Matrisi **\n",confusion_matrix(y_test, y_pred))
print("** Test Seti Doğruluk Oranı **\n:",accuracy_score(y_test, y_pred))
print("** Train Seti Doğruluk Oranı **\n:",accuracy_score(y_train, y_pred_train))
print(classification_report(y_test, y_pred)) 
svc = SVC(probability=True, kernel='linear')
bag_model = BaggingClassifier(base_estimator = svc, random_state = 42).fit(X_train, y_train)
y_pred = bag_model.predict(X_test)
y_pred_train = bag_model.predict(X_train)
print("** Karmaşıklık Matrisi **\n",confusion_matrix(y_test, y_pred))
print("** Test Seti Doğruluk Oranı **\n:",accuracy_score(y_test, y_pred))
print("** Train Seti Doğruluk Oranı **\n:",accuracy_score(y_train, y_pred_train))
print(classification_report(y_test, y_pred)) 
Knn_clf = KNeighborsClassifier()
DTree_clf = DecisionTreeClassifier()
SVC_clf = SVC()
voting_clf = VotingClassifier(estimators=[('SVC', SVC_clf), ('DTree', DTree_clf), ('Knn', Knn_clf)], voting='hard')
voting_clf.fit(X_train, y_train)
eclf_model = voting_clf.fit(X_train, y_train)
y_pred = eclf_model.predict(X_test)
y_pred_train = eclf_model.predict(X_train)
print("** Karmaşıklık Matrisi **\n",confusion_matrix(y_test, y_pred))
print("** Test Seti Doğruluk Oranı **\n:",accuracy_score(y_test, y_pred))
print("** Train Seti Doğruluk Oranı **\n:",accuracy_score(y_train, y_pred_train))
print(classification_report(y_test, y_pred)) 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
mlpc_model = MLPClassifier(random_state = 42).fit(X_train_scaled, y_train)
y_pred = mlpc_model.predict(X_test_scaled)
y_pred_train = mlpc_model.predict(X_train)
print("** Karmaşıklık Matrisi **\n",confusion_matrix(y_test, y_pred))
print("** Test Seti Doğruluk Oranı **\n:",accuracy_score(y_test, y_pred))
print("** Train Seti Doğruluk Oranı **\n:",accuracy_score(y_train, y_pred_train))
print(classification_report(y_test, y_pred)) 
cat_model = CatBoostClassifier(random_state = 42).fit(X_train, y_train)
y_pred = cat_model.predict(X_test)
y_pred_train = cat_model.predict(X_train)
print("** Karmaşıklık Matrisi **\n",confusion_matrix(y_test, y_pred))
print("** Test Seti Doğruluk Oranı **\n:",accuracy_score(y_test, y_pred))
print("** Train Seti Doğruluk Oranı **\n:",accuracy_score(y_train, y_pred_train))
print(classification_report(y_test, y_pred)) 
lgbm_model = LGBMClassifier(random_state = 42).fit(X_train, y_train)
y_pred = lgbm_model.predict(X_test)
y_pred_train = lgbm_model.predict(X_train)
print("** Karmaşıklık Matrisi **\n",confusion_matrix(y_test, y_pred))
print("** Test Seti Doğruluk Oranı **\n:",accuracy_score(y_test, y_pred))
print("** Train Seti Doğruluk Oranı **\n:",accuracy_score(y_train, y_pred_train))
print(classification_report(y_test, y_pred)) 
xgb_model = XGBClassifier(random_state = 42).fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
y_pred_train = xgb_model.predict(X_train)
print("** Karmaşıklık Matrisi **\n",confusion_matrix(y_test, y_pred))
print("** Test Seti Doğruluk Oranı **\n:",accuracy_score(y_test, y_pred))
print("** Train Seti Doğruluk Oranı **\n:",accuracy_score(y_train, y_pred_train))
print(classification_report(y_test, y_pred)) 
nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
y_pred_train = nb_model.predict(X_train)
print("** Karmaşıklık Matrisi **\n",confusion_matrix(y_test, y_pred))
print("** Test Seti Doğruluk Oranı **\n:",accuracy_score(y_test, y_pred))
print("** Train Seti Doğruluk Oranı **\n:",accuracy_score(y_train, y_pred_train))
print(classification_report(y_test, y_pred)) 
gbm_model = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
y_pred = gbm_model.predict(X_test)
y_pred_train = gbm_model.predict(X_train)
print("** Karmaşıklık Matrisi **\n",confusion_matrix(y_test, y_pred))
print("** Test Seti Doğruluk Oranı **\n:",accuracy_score(y_test, y_pred))
print("** Train Seti Doğruluk Oranı **\n:",accuracy_score(y_train, y_pred_train))
print(classification_report(y_test, y_pred)) 
modeller = [
    knn_model,
    svc_model,
    nb_model,
    mlpc_model,
    cart_model,
    rf_model,
    gbm_model,
    cat_model,
    lgbm_model,
]


for model in modeller:
    isimler = model.__class__.__name__
    y_pred = model.predict(X_test)
    dogruluk = accuracy_score(y_test, y_pred)
    print("-"*28)
    print(isimler + ":" )
    print("Accuracy: {:.4%}".format(dogruluk))
sonuc = []

sonuclar = pd.DataFrame(columns= ["Modeller","Accuracy"])

for model in modeller:
    isimler = model.__class__.__name__
    y_pred = model.predict(X_test)
    dogruluk = accuracy_score(y_test, y_pred)    
    sonuc = pd.DataFrame([[isimler, dogruluk*100]], columns= ["Modeller","Accuracy"])
    sonuclar = sonuclar.append(sonuc)
    
    
sns.barplot(x= 'Accuracy', y = 'Modeller', data=sonuclar, color="r")
plt.xlabel('Accuracy %')
plt.title('Modellerin Doğruluk Oranları');    