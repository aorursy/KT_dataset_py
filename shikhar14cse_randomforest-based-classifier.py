import pandas as pd
import seaborn as sb
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
# Reading wine quality dataset
df = pd.read_csv('../input/winequality-red.csv', engine = 'python', error_bad_lines = False)
width = len(df.columns)
print(df.columns)
# Binning values of quality attribute
df['quality'] = pd.cut(df['quality'], (2, 6.5, 8), labels = [0, 1])
# Dividing dataframe to data and target labels
data = df.iloc[:, [1, 2, 4, 6, 9, 10, 11]] # Features selected after visualising trends in data
data = data.sample(frac = 1)
X = data.iloc[:, :6]
Y = data.iloc[:, 6]
sc = StandardScaler()
print('Bad wine : %d, Good wine : %d'%(Counter(Y)[0], Counter(Y)[1]))
# Training and evaluating Random Forest classifier
X = sc.fit_transform(X)    
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size = .33,
                                                        random_state = 42)
rfc = RandomForestClassifier(n_estimators=20)
rfc.fit(X_train, Y_train)
pred_rfc = rfc.predict(X_test)
print(classification_report(Y_test, pred_rfc))
# Measuring the cross validation score for the model
rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = Y_train, cv = 10)
print(rfc_eval.mean())
# checking performance after balancing the data
# Undersampling
good = data[data.quality == 1]
bad = data[data.quality == 0]
while(len(bad) > 0):
    siz = min(len(good), len(bad))
    part = bad.sample(siz, random_state = 32)
    bad = bad.drop(part.index)
    demo = pd.concat([part, good])
    demo = demo.sample(frac = 1, random_state = 200)
    X = demo.iloc[:, :6]
    Y = demo.iloc[:, 6]
    X = sc.fit_transform(X)    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size = .2,
                                                        random_state = 42)
    pred_rfc = rfc.predict(X_test)
    rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = Y_train, cv = 10)
    print(rfc_eval.mean())
feat = pd.DataFrame({'Feat':[1, 2, 4, 6, 9, 10],
                   'Imp':rfc.feature_importances_.tolist()})
print(feat.head())
sb.barplot(x = feat['Feat'], y = feat['Imp'])
plt.show()
#Going with all columns
X = np.asarray(df.loc[:, df.columns != 'quality'])
Y = np.asarray(df.loc[:, df.columns == 'quality']).ravel()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size = .3,
                                                        random_state = 42)
rfc.fit(X_train, Y_train)
print(classification_report(Y_test, rfc.predict(X_test), target_names = ['bad', 'good']))
feat = pd.DataFrame({'feat': df.loc[:, df.columns != 'quality'].columns.tolist(),
                    'imp':rfc.feature_importances_.tolist()})
g = sb.barplot(x = feat['feat'], y = feat['imp'])

labels = g.get_xticklabels()
g.set_xticklabels(labels,rotation=50)

plt.show(g)