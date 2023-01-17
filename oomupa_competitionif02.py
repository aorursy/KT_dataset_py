import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing
train = pd.read_csv("../input/train.csv")

teste = pd.read_csv("../input/test.csv")
train.Churn.value_counts()
train.head()
teste.head()
num_classes = 14

print("Numero de classes: ", num_classes ,"\n\n")



# tenure

amplitude_total_1 = train.tenure.max() - train.tenure.min()

saltos_1 = amplitude_total_1 / num_classes

print("Tenure saltos: ", saltos_1)

print("Tenure min: ", train.tenure.min())

print("Tenure max: ", train.tenure.max(), "\n\n")



# MonthlyCharges

amplitude_total_2 = train.MonthlyCharges.max() - train.MonthlyCharges.min()

saltos_2 = amplitude_total_2 / num_classes

print("MonthlyCharges saltos: ", saltos_2)

print("MonthlyCharges min: ", train.MonthlyCharges.min())

print("MonthlyCharges max: ", train.MonthlyCharges.max(), "\n\n")



# TotalCharges

amplitude_total = train.TotalCharges.max() - train.TotalCharges.min()

saltos = amplitude_total / num_classes

print("TotalCharges saltos: ", saltos)

print("TotalCharges min: ", train.TotalCharges.min())

print("TotalCharges max: ", train.TotalCharges.max())
def feature_engineer(churn):

    le_dep = preprocessing.LabelEncoder()

    #to convert into numbers

    churn['Churn'] = le_dep.fit_transform(churn['Churn'])

    churn['gender'] = le_dep.fit_transform(churn['gender'])

    churn['Partner'] = le_dep.fit_transform(churn['Partner'])

    churn['Dependents'] = le_dep.fit_transform(churn['Dependents'])

    churn['PhoneService'] = le_dep.fit_transform(churn['PhoneService'])

    churn['MultipleLines'] = le_dep.fit_transform(churn['MultipleLines'])

    churn['InternetService'] = le_dep.fit_transform(churn['InternetService'])

    churn['OnlineSecurity'] = le_dep.fit_transform(churn['OnlineSecurity'])

    churn['OnlineBackup'] = le_dep.fit_transform(churn['OnlineBackup'])

    churn['DeviceProtection'] = le_dep.fit_transform(churn['DeviceProtection'])

    churn['TechSupport'] = le_dep.fit_transform(churn['TechSupport'])

    churn['StreamingTV'] = le_dep.fit_transform(churn['StreamingTV'])

    churn['StreamingMovies'] = le_dep.fit_transform(churn['StreamingMovies'])

    churn['Contract'] = le_dep.fit_transform(churn['Contract'])

    churn['PaperlessBilling'] = le_dep.fit_transform(churn['PaperlessBilling'])

    churn['PaymentMethod'] = le_dep.fit_transform(churn['PaymentMethod'])

    

    #tenure classes

    churn.loc[(churn.tenure > 0) & (churn.tenure <= 6), 'tenure'] = 1

    churn.loc[(churn.tenure > 6) & (churn.tenure <= 12), 'tenure'] = 2

    churn.loc[(churn.tenure > 12) & (churn.tenure <= 18), 'tenure'] = 3

    churn.loc[(churn.tenure > 18) & (churn.tenure <= 24), 'tenure'] = 4

    churn.loc[(churn.tenure > 24) & (churn.tenure <= 30), 'tenure'] = 5

    churn.loc[(churn.tenure > 30) & (churn.tenure <= 36), 'tenure'] = 6

    churn.loc[(churn.tenure > 36) & (churn.tenure <= 42), 'tenure'] = 7

    churn.loc[(churn.tenure > 42) & (churn.tenure <= 48), 'tenure'] = 8

    churn.loc[(churn.tenure > 48) & (churn.tenure <= 54), 'tenure'] = 9

    churn.loc[(churn.tenure > 54) & (churn.tenure <= 60), 'tenure'] = 10

    churn.loc[(churn.tenure > 60) & (churn.tenure <= 66), 'tenure'] = 11

    churn.loc[(churn.tenure > 66) & (churn.tenure <= 72), 'tenure'] = 12

    churn.loc[(churn.tenure > 72) & (churn.tenure <= 78), 'tenure'] = 13

    churn.loc[churn.tenure > 78, 'tenure'] = 14

    churn['tenure'] = le_dep.fit_transform(churn['tenure'])

    

    # MonthlyCharges classes

    churn.loc[(churn.MonthlyCharges > 18) & (churn.MonthlyCharges <= 26), 'MonthlyCharges'] = 1

    churn.loc[(churn.MonthlyCharges > 26) & (churn.MonthlyCharges <= 34), 'MonthlyCharges'] = 2

    churn.loc[(churn.MonthlyCharges > 34) & (churn.MonthlyCharges <= 42), 'MonthlyCharges'] = 3

    churn.loc[(churn.MonthlyCharges > 42) & (churn.MonthlyCharges <= 50), 'MonthlyCharges'] = 4

    churn.loc[(churn.MonthlyCharges > 50) & (churn.MonthlyCharges <= 58), 'MonthlyCharges'] = 5

    churn.loc[(churn.MonthlyCharges > 58) & (churn.MonthlyCharges <= 66), 'MonthlyCharges'] = 6

    churn.loc[(churn.MonthlyCharges > 66) & (churn.MonthlyCharges <= 74), 'MonthlyCharges'] = 7

    churn.loc[(churn.MonthlyCharges > 74) & (churn.MonthlyCharges <= 82), 'MonthlyCharges'] = 8

    churn.loc[(churn.MonthlyCharges > 82) & (churn.MonthlyCharges <= 90), 'MonthlyCharges'] = 9

    churn.loc[(churn.MonthlyCharges > 90) & (churn.MonthlyCharges <= 98), 'MonthlyCharges'] = 10

    churn.loc[(churn.MonthlyCharges > 98) & (churn.MonthlyCharges <= 106), 'MonthlyCharges'] = 11

    churn.loc[(churn.MonthlyCharges > 106) & (churn.MonthlyCharges <= 114), 'MonthlyCharges'] = 12

    churn.loc[(churn.MonthlyCharges > 114) & (churn.MonthlyCharges <= 122), 'MonthlyCharges'] = 13

    churn.loc[churn.MonthlyCharges > 122, 'MonthlyCharges'] = 14

    churn['MonthlyCharges'] = le_dep.fit_transform(churn['MonthlyCharges'])

    

    # TotalCharges classes

    churn.loc[churn.TotalCharges.isna(), 'TotalCharges'] = churn[~churn.TotalCharges.isna()].TotalCharges.mean()

    churn.loc[(churn.TotalCharges > 18) & (churn.TotalCharges <= 637), 'TotalCharges'] = 1

    churn.loc[(churn.TotalCharges > 637) & (churn.TotalCharges <= 1256), 'TotalCharges'] = 2

    churn.loc[(churn.TotalCharges > 1256) & (churn.TotalCharges <= 1875), 'TotalCharges'] = 3

    churn.loc[(churn.TotalCharges > 1875) & (churn.TotalCharges <= 2494), 'TotalCharges'] = 4

    churn.loc[(churn.TotalCharges > 2494) & (churn.TotalCharges <= 3113), 'TotalCharges'] = 5

    churn.loc[(churn.TotalCharges > 3113) & (churn.TotalCharges <= 3732), 'TotalCharges'] = 6

    churn.loc[(churn.TotalCharges > 3732) & (churn.TotalCharges <= 4351), 'TotalCharges'] = 7

    churn.loc[(churn.TotalCharges > 4351) & (churn.TotalCharges <= 4970), 'TotalCharges'] = 8

    churn.loc[(churn.TotalCharges > 4970) & (churn.TotalCharges <= 5589), 'TotalCharges'] = 9

    churn.loc[(churn.TotalCharges > 5589) & (churn.TotalCharges <= 6208), 'TotalCharges'] = 10

    churn.loc[(churn.TotalCharges > 6208) & (churn.TotalCharges <= 6827), 'TotalCharges'] = 11

    churn.loc[(churn.TotalCharges > 6827) & (churn.TotalCharges <= 7446), 'TotalCharges'] = 12

    churn.loc[(churn.TotalCharges > 7446) & (churn.TotalCharges <= 8065), 'TotalCharges'] = 13

    churn.loc[churn.TotalCharges > 8065, 'TotalCharges'] = 14

    churn['TotalCharges'] = le_dep.fit_transform(churn['TotalCharges'])

        

    return churn



def feature_engineer_test(churn):

    le_dep = preprocessing.LabelEncoder()

    #to convert into numbers

    churn['gender'] = le_dep.fit_transform(churn['gender'])

    churn['Partner'] = le_dep.fit_transform(churn['Partner'])

    churn['Dependents'] = le_dep.fit_transform(churn['Dependents'])

    churn['PhoneService'] = le_dep.fit_transform(churn['PhoneService'])

    churn['MultipleLines'] = le_dep.fit_transform(churn['MultipleLines'])

    churn['InternetService'] = le_dep.fit_transform(churn['InternetService'])

    churn['OnlineSecurity'] = le_dep.fit_transform(churn['OnlineSecurity'])

    churn['OnlineBackup'] = le_dep.fit_transform(churn['OnlineBackup'])

    churn['DeviceProtection'] = le_dep.fit_transform(churn['DeviceProtection'])

    churn['TechSupport'] = le_dep.fit_transform(churn['TechSupport'])

    churn['StreamingTV'] = le_dep.fit_transform(churn['StreamingTV'])

    churn['StreamingMovies'] = le_dep.fit_transform(churn['StreamingMovies'])

    churn['Contract'] = le_dep.fit_transform(churn['Contract'])

    churn['PaperlessBilling'] = le_dep.fit_transform(churn['PaperlessBilling'])

    churn['PaymentMethod'] = le_dep.fit_transform(churn['PaymentMethod'])

    

    #tenure classes

    churn.loc[(churn.tenure > 0) & (churn.tenure <= 6), 'tenure'] = 1

    churn.loc[(churn.tenure > 6) & (churn.tenure <= 12), 'tenure'] = 2

    churn.loc[(churn.tenure > 12) & (churn.tenure <= 18), 'tenure'] = 3

    churn.loc[(churn.tenure > 18) & (churn.tenure <= 24), 'tenure'] = 4

    churn.loc[(churn.tenure > 24) & (churn.tenure <= 30), 'tenure'] = 5

    churn.loc[(churn.tenure > 30) & (churn.tenure <= 36), 'tenure'] = 6

    churn.loc[(churn.tenure > 36) & (churn.tenure <= 42), 'tenure'] = 7

    churn.loc[(churn.tenure > 42) & (churn.tenure <= 48), 'tenure'] = 8

    churn.loc[(churn.tenure > 48) & (churn.tenure <= 54), 'tenure'] = 9

    churn.loc[(churn.tenure > 54) & (churn.tenure <= 60), 'tenure'] = 10

    churn.loc[(churn.tenure > 60) & (churn.tenure <= 66), 'tenure'] = 11

    churn.loc[(churn.tenure > 66) & (churn.tenure <= 72), 'tenure'] = 12

    churn.loc[(churn.tenure > 72) & (churn.tenure <= 78), 'tenure'] = 13

    churn.loc[churn.tenure > 78, 'tenure'] = 14

    churn['tenure'] = le_dep.fit_transform(churn['tenure'])

    

    # MonthlyCharges classes

    churn.loc[(churn.MonthlyCharges > 18) & (churn.MonthlyCharges <= 26), 'MonthlyCharges'] = 1

    churn.loc[(churn.MonthlyCharges > 26) & (churn.MonthlyCharges <= 34), 'MonthlyCharges'] = 2

    churn.loc[(churn.MonthlyCharges > 34) & (churn.MonthlyCharges <= 42), 'MonthlyCharges'] = 3

    churn.loc[(churn.MonthlyCharges > 42) & (churn.MonthlyCharges <= 50), 'MonthlyCharges'] = 4

    churn.loc[(churn.MonthlyCharges > 50) & (churn.MonthlyCharges <= 58), 'MonthlyCharges'] = 5

    churn.loc[(churn.MonthlyCharges > 58) & (churn.MonthlyCharges <= 66), 'MonthlyCharges'] = 6

    churn.loc[(churn.MonthlyCharges > 66) & (churn.MonthlyCharges <= 74), 'MonthlyCharges'] = 7

    churn.loc[(churn.MonthlyCharges > 74) & (churn.MonthlyCharges <= 82), 'MonthlyCharges'] = 8

    churn.loc[(churn.MonthlyCharges > 82) & (churn.MonthlyCharges <= 90), 'MonthlyCharges'] = 9

    churn.loc[(churn.MonthlyCharges > 90) & (churn.MonthlyCharges <= 98), 'MonthlyCharges'] = 10

    churn.loc[(churn.MonthlyCharges > 98) & (churn.MonthlyCharges <= 106), 'MonthlyCharges'] = 11

    churn.loc[(churn.MonthlyCharges > 106) & (churn.MonthlyCharges <= 114), 'MonthlyCharges'] = 12

    churn.loc[(churn.MonthlyCharges > 114) & (churn.MonthlyCharges <= 122), 'MonthlyCharges'] = 13

    churn.loc[churn.MonthlyCharges > 122, 'MonthlyCharges'] = 14

    churn['MonthlyCharges'] = le_dep.fit_transform(churn['MonthlyCharges'])

    

    # TotalCharges classes

    churn.loc[churn.TotalCharges.isna(), 'TotalCharges'] = churn[~churn.TotalCharges.isna()].TotalCharges.mean()

    churn.loc[(churn.TotalCharges > 18) & (churn.TotalCharges <= 637), 'TotalCharges'] = 1

    churn.loc[(churn.TotalCharges > 637) & (churn.TotalCharges <= 1256), 'TotalCharges'] = 2

    churn.loc[(churn.TotalCharges > 1256) & (churn.TotalCharges <= 1875), 'TotalCharges'] = 3

    churn.loc[(churn.TotalCharges > 1875) & (churn.TotalCharges <= 2494), 'TotalCharges'] = 4

    churn.loc[(churn.TotalCharges > 2494) & (churn.TotalCharges <= 3113), 'TotalCharges'] = 5

    churn.loc[(churn.TotalCharges > 3113) & (churn.TotalCharges <= 3732), 'TotalCharges'] = 6

    churn.loc[(churn.TotalCharges > 3732) & (churn.TotalCharges <= 4351), 'TotalCharges'] = 7

    churn.loc[(churn.TotalCharges > 4351) & (churn.TotalCharges <= 4970), 'TotalCharges'] = 8

    churn.loc[(churn.TotalCharges > 4970) & (churn.TotalCharges <= 5589), 'TotalCharges'] = 9

    churn.loc[(churn.TotalCharges > 5589) & (churn.TotalCharges <= 6208), 'TotalCharges'] = 10

    churn.loc[(churn.TotalCharges > 6208) & (churn.TotalCharges <= 6827), 'TotalCharges'] = 11

    churn.loc[(churn.TotalCharges > 6827) & (churn.TotalCharges <= 7446), 'TotalCharges'] = 12

    churn.loc[(churn.TotalCharges > 7446) & (churn.TotalCharges <= 8065), 'TotalCharges'] = 13

    churn.loc[churn.TotalCharges > 8065, 'TotalCharges'] = 14

    churn['TotalCharges'] = le_dep.fit_transform(churn['TotalCharges'])

    return churn
train = pd.read_csv("../input/train.csv")

teste = pd.read_csv("../input/test.csv")

train = feature_engineer(train)

teste = feature_engineer_test(teste)
train.head()
teste.head()
train.info()
teste.info()
train.isna().sum()
teste.isna().sum()
fig,axes = plt.subplots(1,2,figsize=(25,8))

print(axes)



sns.boxplot(x='Churn', y='TotalCharges', data=train, palette='viridis', ax=axes[0])



sns.boxplot(x='Churn', y='tenure', data=train, palette='viridis', ax=axes[1])



plt.show()
numerical_column = ['int64']

plt.figure(figsize=(10,10))

sns.pairplot(train.select_dtypes(include=numerical_column), hue = 'Churn')
# Mapa de correlação

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
X_train = train.drop('Churn',axis=1).select_dtypes(include=['int64','float64'])

y_train = train['Churn']

X_test = teste.select_dtypes(include=['int64','float64'])

test_id = teste.customerID
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

X_train_ohe = ohe.fit_transform(X_train)

X_test_ohe = ohe.fit_transform(X_test)
X_train.head()
X_test.head()
# This is important

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_SS = scaler.fit_transform(X_train)

X_test_SS = scaler.fit_transform(X_test)
from sklearn import svm

lsvm_model = svm.LinearSVC()

lsvm_model.fit(X_train_ohe, y_train) 
print("OHE better\n")

scoreLSVM = lsvm_model.score(X_train_ohe, y_train)

print(scoreLSVM)



print("\n\n")

print("Kfold on LinearSVM: %0.4f (+/- %0.4f)" % (scoreLSVM.mean(), scoreLSVM.std()))





predicao_lsvm = lsvm_model.predict(X_test_ohe)

submissao_lsvm = pd.DataFrame({

    "Id": test_id, 

    "Expected": predicao_lsvm

})

submissao_lsvm.to_csv('LinearSVM.csv', index=False)
from sklearn import svm

svm_model = svm.SVC()

svm_model.fit(X_train, y_train) 
print("LE better")



scoreSVM = svm_model.score(X_train, y_train)

print(scoreSVM)



print("\n\n")

print("Kfold on SVM: %0.4f (+/- %0.4f)" % (scoreSVM.mean(), scoreSVM.std()))





predicao_svm = svm_model.predict(X_test)

submissao_svm = pd.DataFrame({

    "Id": test_id, 

    "Expected": predicao_svm

})

submissao_svm.to_csv('SVM.csv', index=False)
from sklearn.neighbors import KNeighborsClassifier

kn_model = KNeighborsClassifier(n_neighbors=3)

kn_model.fit(X_train, y_train) 
print("LE better")



scoreKN = kn_model.score(X_train, y_train)

print(scoreKN)



print("\n\n")

print("Kfold on KNeighborsClassifier: %0.4f (+/- %0.4f)" % (scoreKN.mean(), scoreKN.std()))





predicao_kn = kn_model.predict(X_test)

submissao_kn = pd.DataFrame({

    "Id": test_id, 

    "Expected": predicao_kn

})

submissao_kn.to_csv('KNeighborsClassifier.csv', index=False)
from sklearn.ensemble import RandomForestClassifier



random_forest_model = RandomForestClassifier(n_estimators=100)

random_forest_model.fit(X_train, y_train)
print("Both the same\n")



scoreRF = random_forest_model.score(X_train, y_train)

print(scoreRF)



print("\n\n")

print("Kfold on Random Forest: %0.4f (+/- %0.4f)" % (scoreRF.mean(), scoreRF.std()))



predicao_random = random_forest_model.predict(X_test)

submissao_random = pd.DataFrame({

    "Id": test_id, 

    "Expected": predicao_random

})

submissao_random.to_csv('RandomForestClassifier.csv', index=False)
from sklearn.ensemble import GradientBoostingClassifier



GBR_model = GradientBoostingClassifier()      

GBR_model.fit(X_train_ohe, y_train)
print("OHE little better\n")

scoreGBR = GBR_model.score(X_train_ohe, y_train)

print(scoreRF)



print("\n\n")

print("Kfold on GradientBoostingClassifier: %0.4f (+/- %0.4f)" % (scoreGBR.mean(), scoreGBR.std()))



predicao_GBR = GBR_model.predict(X_test_ohe)

submissao_GBR = pd.DataFrame({

    "Id": test_id, 

    "Expected": predicao_GBR

})

submissao_GBR.to_csv('GradientBoostingClassifier.csv', index=False)
from sklearn.ensemble import AdaBoostClassifier

ADA_model = AdaBoostClassifier()      

ADA_model.fit(X_train_ohe, y_train)
print("OHE better\n")

scoreADA = ADA_model.score(X_train_ohe, y_train)

print(scoreADA)



print("\n\n")

print("Kfold on AdaBoostClassifier: %0.4f (+/- %0.4f)" % (scoreADA.mean(), scoreADA.std()))



predicao_ADA = ADA_model.predict(X_test_ohe)

submissao_ADA = pd.DataFrame({

    "Id": test_id, 

    "Expected": predicao_ADA

})

submissao_ADA.to_csv('AdaBoostClassifier.csv', index=False)
from sklearn.ensemble import BaggingClassifier

BC_model = BaggingClassifier()      

BC_model.fit(X_train, y_train)
print("LE better")

scoreBC = BC_model.score(X_train, y_train)

print(scoreBC)



print("\n\n")

print("Kfold on BaggingClassifier: %0.4f (+/- %0.4f)" % (scoreBC.mean(), scoreBC.std()))



predicao_BC = BC_model.predict(X_test)

submissao_BC = pd.DataFrame({

    "Id": test_id, 

    "Expected": predicao_BC

})

submissao_BC.to_csv('BaggingClassifier.csv', index=False)
import xgboost as xgb

from sklearn import model_selection



xg_boost_model = xgb.XGBClassifier(learning_rate=0.3, base_score=0.5, n_estimators=50) #1.5

xg_boost_model.fit(X_train, y_train)
print("Acuracia normal: ",xg_boost_model.score(X_train, y_train))



print("\n\n")

scoreXG = model_selection.cross_val_score(xg_boost_model, X_train, y_train, cv=5, scoring='accuracy')

print(scoreXG)

print("Kfold on XGBClassifier: %0.4f (+/- %0.4f)" % (scoreXG.mean(), scoreXG.std()))



predicao_xg = xg_boost_model.predict(X_test)



submissao_xg = pd.DataFrame({

    "Id": test_id, 

    "Expected": predicao_xg 

})

#submissao_xg.head(10)



submissao_xg.to_csv('XGBoost.csv', index=False)
from sklearn.linear_model import SGDClassifier



stoc_Grad_Desc_model = SGDClassifier()

stoc_Grad_Desc_model.fit(X_train_ohe, y_train)
print("OHE better")



scoreSGD = stoc_Grad_Desc_model.score(X_train_ohe, y_train)

print(scoreSGD)



print("\n\n")

print("Kfold on Stochastic Gradient Descent: %0.4f (+/- %0.4f)" % (scoreSGD.mean(), scoreSGD.std()))



predicao_sgd = stoc_Grad_Desc_model.predict(X_test_ohe)

submissao_sgd = pd.DataFrame({

    "Id": test_id, 

    "Expected": predicao_sgd

})



submissao_sgd.to_csv('StochasticGradientDescent.csv', index=False)
from sklearn.naive_bayes import GaussianNB

gnb_model = GaussianNB()

gnb_model.fit(X_train, y_train)
print("LE better\n")



scoreGNB = gnb_model.score(X_train, y_train)

print(scoreGNB)



print("\n\n")

print("Kfold on Gaussian Naive Bayes: %0.4f (+/- %0.4f)" % (scoreGNB.mean(), scoreGNB.std()))





# Send to .csv

predicao_gnb = gnb_model.predict(X_test)



submissao_gnb = pd.DataFrame({

    "Id": test_id, 

    "Expected": predicao_gnb

})



submissao_gnb.to_csv('GaussianNaiveBayes.csv', index=False)
from sklearn import tree

tree_model = tree.DecisionTreeClassifier()

tree_model.fit(X_train, y_train)
print("Both the same\n")



scoreTREE = tree_model.score(X_train, y_train)

print(scoreTREE)



print("\n\n")

print("Kfold on DecisionTree: %0.4f (+/- %0.4f)" % (scoreTREE.mean(), scoreTREE.std()))





# Send to .csv

predicao_tree = tree_model.predict(X_test)



submissao_tree = pd.DataFrame({

    "Id": test_id, 

    "Expected": predicao_tree

})



submissao_tree.to_csv('DecisionTree.csv', index=False)
from sklearn.linear_model import LogisticRegression

log_reg_model = LogisticRegression()

log_reg_model.fit(X_train_ohe, y_train)
print("OHE better")



scoreRL = log_reg_model.score(X_train_ohe, y_train)

print(scoreRL)



print("\n\n")

print("Kfold on LogisticRegression: %0.4f (+/- %0.4f)" % (scoreRL.mean(), scoreRL.std()))





# Send to .csv

predicao_rl = log_reg_model.predict(X_test_ohe)



submissao_rl = pd.DataFrame({

    "Id": test_id, 

    "Expected": predicao_rl

})



submissao_rl.to_csv('LogisticRegression.csv', index=False)
from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(batch_size=32, max_iter=2000, learning_rate_init=0.5)

mlp_model.fit(X_train, y_train)
print("Both the same")



scoreMLP = mlp_model.score(X_train, y_train)

print(scoreMLP)



print("\n\n")

print("Kfold on Multi Layer Perceptron: %0.4f (+/- %0.4f)" % (scoreMLP.mean(), scoreMLP.std()))





# Send to .csv

predicao_mlp = mlp_model.predict(X_test)



submissao_mlp = pd.DataFrame({

    "Id": test_id, 

    "Expected": predicao_mlp

})



submissao_mlp.to_csv('MultiLayerPerceptron.csv', index=False)
from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC



one_rest_model = OneVsRestClassifier(SVC(max_iter=10000))

one_rest_model.fit(X_train, y_train)
print("LE better")



scoreONEREST = one_rest_model.score(X_train, y_train)

print(scoreONEREST)



print("\n\n")

print("Kfold on One vs Rest: %0.4f (+/- %0.4f)" % (scoreONEREST.mean(), scoreONEREST.std()))





# Send to .csv

predicao_ONEREST = one_rest_model.predict(X_test)



submissao_ONEREST = pd.DataFrame({

    "Id": test_id, 

    "Expected": predicao_ONEREST

})



submissao_ONEREST.to_csv('OneVsRestClassifier_SVM.csv', index=False)
print("Bagging score: %0.4f" % scoreBC)

print("Random score: %0.4f" %scoreRF)

print("Decision score: %0.4f" %scoreTREE)