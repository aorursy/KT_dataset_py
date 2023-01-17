import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import warnings
#Reading the dataset
df = pd.read_csv('../input/dataset.csv', engine = 'python', error_bad_lines = False)
print(df.isnull().sum().sum())
#Imputing the null values in CONTENT_LENGTH attribute
no_null = 812
avg = df.CONTENT_LENGTH.sum() / (len(df) - 812)
df.loc[df['CONTENT_LENGTH'].isnull(), 'CONTENT_LENGTH'] = avg
df = df.dropna()
print(df.isnull().any())
#Encoding the columns having string data
cols = ['URL', 'SERVER', 'CHARSET',
       'WHOIS_COUNTRY', 'WHOIS_STATEPRO', 'WHOIS_REGDATE',
       'WHOIS_UPDATED_DATE']
le = LabelEncoder()
for c in cols:    
    df[c] = le.fit_transform(df[c])
#Segregating the classes 
yes = df[df.Type == 1]
no = df[df.Type == 0]
print('YES : %d  No: %d'%(len(yes), len(no)))
# Under sampling data for better training
while(len(no) > 0):
    size = min(len(yes), len(no))
    t = no.sample(size, random_state = 200)
    no = no.drop(t.index)
    fr = pd.concat([t, yes])
    fr = fr.sample(frac = 1, random_state = 42)
    X = np.asarray(fr.iloc[:, :20])
    Y = np.asarray(fr['Type'])
    print(Counter(Y))    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size = .3,
                                                            random_state = 42)
    rfc = RandomForestClassifier(n_estimators = 20)
    rfc.fit(X_train, Y_train)
    pred_rfc = rfc.predict(X_test)
    print(classification_report(Y_test, pred_rfc))
# feature importance
import seaborn as sb
import matplotlib.pyplot as plt
g = sb.barplot(x = df.loc[:, df.columns != 'Type'].columns.tolist(), y =  rfc.feature_importances_.tolist() )
labels = g.get_xticklabels()
g.set_xticklabels(labels,rotation=50)
plt.show(g)