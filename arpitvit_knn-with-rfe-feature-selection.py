import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import math
import seaborn as sns
cancer_ds = pd.read_csv('../input/kag_risk_factors_cervical_cancer.csv')
cancer_ds.head()
cancer_ds.info()
cancer_ds.describe()
numerical_ds = ['Age', 'Number of sexual partners', 'First sexual intercourse','Num of pregnancies', 'Smokes (years)',
                'Smokes (packs/year)','Hormonal Contraceptives (years)','IUD (years)','STDs (number)','STDs: Time since first diagnosis'
               ,'STDs: Time since last diagnosis']
categorical_ds = ['Smokes','Hormonal Contraceptives','IUD','STDs','STDs:condylomatosis','STDs:cervical condylomatosis',
                  'STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
                  'STDs:pelvic inflammatory disease', 'STDs:genital herpes','STDs:molluscum contagiosum', 'STDs:AIDS', 
                  'STDs:HIV','STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis','Dx:Cancer', 'Dx:CIN', 
                  'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller','Citology', 'Biopsy']
cancer_ds = cancer_ds.replace('?', np.NaN)
for feature in numerical_ds:
    print(feature,'',cancer_ds[feature].convert_objects(convert_numeric=True).mean())
    feature_mean = round(cancer_ds[feature].convert_objects(convert_numeric=True).mean(),1)
    cancer_ds[feature] = cancer_ds[feature].fillna(feature_mean)
for feature in categorical_ds:
    cancer_ds[feature] = cancer_ds[feature].convert_objects(convert_numeric=True).fillna(1.0)
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
y=cancer_ds.Biopsy.values
cancer_ds=cancer_ds.drop(['Biopsy'],axis=1)
x=cancer_ds.as_matrix()
colnames=cancer_ds.columns
model= RandomForestClassifier()
model.fit(x,y)
rfe=RFE(model,n_features_to_select=1,verbose=2)
fit=rfe.fit(x,y)
from sklearn.preprocessing import MinMaxScaler
ranks={}
def ranking (ranks,names,order=1):
    minmax=MinMaxScaler()
    ranks=minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks=map(lambda x: round(x,2),ranks)
    return dict(zip(names,ranks))
ranks["RFE"]=ranking(list(map(float,rfe.ranking_)),colnames,order=-1)
ranking_ds=pd.DataFrame({"Attribute Name":cancer_ds.columns})
ref_ranking=rfe.ranking_
ranking_ds['Ranking']=ref_ranking
ranking_ds
cancer_ds = pd.read_csv('../input/kag_risk_factors_cervical_cancer.csv')
numerical_ds = ['Age', 'Number of sexual partners', 'First sexual intercourse','Num of pregnancies', 'Smokes (years)',
                'Smokes (packs/year)','Hormonal Contraceptives (years)','IUD (years)','STDs (number)','STDs: Time since first diagnosis'
               ,'STDs: Time since last diagnosis']
categorical_ds = ['Smokes','Hormonal Contraceptives','IUD','STDs','STDs:condylomatosis','STDs:cervical condylomatosis',
                  'STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
                  'STDs:pelvic inflammatory disease', 'STDs:genital herpes','STDs:molluscum contagiosum', 'STDs:AIDS', 
                  'STDs:HIV','STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis','Dx:Cancer', 'Dx:CIN', 
                  'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller','Citology', 'Biopsy']
cancer_ds = cancer_ds.replace('?', np.NaN)
for feature in numerical_ds:
    print(feature,'',cancer_ds[feature].convert_objects(convert_numeric=True).mean())
    feature_mean = round(cancer_ds[feature].convert_objects(convert_numeric=True).mean(),1)
    cancer_ds[feature] = cancer_ds[feature].fillna(feature_mean)
for feature in categorical_ds:
    cancer_ds[feature] = cancer_ds[feature].convert_objects(convert_numeric=True).fillna(1.0)
cancer_ds=cancer_ds.drop(['STDs:Hepatitis B'],axis=1)
cancer_ds=cancer_ds.drop(['Smokes'],axis=1)
print(cancer_ds.columns)
cancer_ds=cancer_ds.drop(['Smokes (years)', 'Smokes (packs/year)',
       'Hormonal Contraceptives', 'IUD', 'STDs', 'STDs:condylomatosis',
       'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
       'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
       'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
       'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV', 'STDs:HPV',
       'STDs: Number of diagnosis', 'STDs: Time since first diagnosis',
       'STDs: Time since last diagnosis', 'Dx:Cancer', 'Dx:CIN', 'Dx:HPV',
       'Dx'],axis=1)
print(cancer_ds.columns)
cancer_ds.info()
cancer_ds.describe()
import numpy as np
np.random.seed(42)
df_data_shuffle = cancer_ds.iloc[np.random.permutation(len(cancer_ds))]

df_train = df_data_shuffle.iloc[1:686, :]
df_test = df_data_shuffle.iloc[686: , :]
df_train_feature = df_train[['Age', 'Number of sexual partners', 'First sexual intercourse',
       'Num of pregnancies', 'Hormonal Contraceptives (years)', 'IUD (years)',
       'STDs (number)', 'Hinselmann', 'Schiller', 'Citology']]

train_label = np.array(df_train['Biopsy'])

df_test_feature = df_test[['Age', 'Number of sexual partners', 'First sexual intercourse',
       'Num of pregnancies', 'Hormonal Contraceptives (years)', 'IUD (years)',
       'STDs (number)', 'Hinselmann', 'Schiller', 'Citology']]

test_label = np.array(df_test['Biopsy'])
from sklearn import preprocessing
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
train_feature = minmax_scale.fit_transform(df_train_feature)
test_feature = minmax_scale.fit_transform(df_test_feature)
train_feature.shape
import matplotlib
import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import cross_val_score, train_test_split
X = np.array(cancer_ds.drop('Biopsy',1))
X = preprocessing.scale(X)
y = np.array(cancer_ds['Biopsy'])
accuracy = []
x_range = []
for j in range(1000):
    x_range.append(j)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = neighbors.KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train,y_train)
    acc = clf.score(X_test,y_test)
    accuracy.append(acc)
plt.title(str(5) + ' nearest neighbors')
plt.plot(x_range, accuracy)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()
print(acc)
test_label = np.array(cancer_ds['Biopsy'])

clf = neighbors.KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train,y_train)

predictions = clf.predict(X_test)
print(predictions)
print(y_test)

df_ans = pd.DataFrame({'Biopsy' :y_test})
df_ans['predictions'] = predictions
df_ans
df_ans[ df_ans['Biopsy'] != df_ans['predictions'] ]
cols = ['Biopsy_1','Biopsy_0']  #Gold standard
rows = ['Prediction_1','Prediction_0'] #diagnostic tool (our prediction)

B1P1 = len(df_ans[(df_ans['predictions'] == df_ans['Biopsy']) & (df_ans['Biopsy'] == 1)])
B1P0 = len(df_ans[(df_ans['predictions'] != df_ans['Biopsy']) & (df_ans['Biopsy'] == 1)])
B0P1 = len(df_ans[(df_ans['predictions'] != df_ans['Biopsy']) & (df_ans['Biopsy'] == 0)])
B0P0 = len(df_ans[(df_ans['predictions'] == df_ans['Biopsy']) & (df_ans['Biopsy'] == 0)])

conf = np.array([[B1P1,B0P1],[B1P0,B0P0]])
df_cm = pd.DataFrame(conf, columns = [i for i in cols], index = [i for i in rows])

f, ax= plt.subplots(figsize = (5, 5))
sns.heatmap(df_cm, annot=True, ax=ax) 
ax.xaxis.set_ticks_position('top') #Making x label be on top is common in textbooks.

print('total test case number: ', np.sum(conf))
def model_efficacy(conf):
    total_num = np.sum(conf)
    sen = conf[0][0]/(conf[0][0]+conf[1][0])
    spe = conf[1][1]/(conf[1][0]+conf[1][1])
    false_positive_rate = conf[0][1]/(conf[0][1]+conf[1][1])
    false_negative_rate = conf[1][0]/(conf[0][0]+conf[1][0])
    
    print('total_num: ',total_num)
    print('G1P1: ',conf[0][0]) 
    print('G0P1: ',conf[0][1])
    print('G1P0: ',conf[1][0])
    print('G0P0: ',conf[1][1])
    print('##########################')
    print('sensitivity: ',sen)
    print('specificity: ',spe)
    print('false_positive_rate: ',false_positive_rate)
    print('false_negative_rate: ',false_negative_rate)
    
    return total_num, sen, spe, false_positive_rate, false_negative_rate

model_efficacy(conf)