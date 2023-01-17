import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import math
import seaborn as sns
cancer_ds = pd.read_csv('C:/Users/ARPIT/Desktop/Notes 2/PROJECT SEM 2/DM PROJECT/data/kag_risk_factors_cervical_cancer.csv')
cancer_ds.head()
cancer_ds.info()
cancer_ds.drop(['STDs: Time since first diagnosis','STDs: Time since last diagnosis'],inplace=True,axis=1)
cancer_ds.describe()
numerical_ds = ['Age', 'Number of sexual partners', 'First sexual intercourse','Num of pregnancies', 'Smokes (years)',
                'Smokes (packs/year)','Hormonal Contraceptives (years)','IUD (years)','STDs (number)']
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
category_ds = ['Hinselmann', 'Schiller','Citology', 'Biopsy']
for feature in categorical_ds:
   sns.factorplot(feature,data=cancer_ds,kind='count')
cancer_ds['Number of sexual partners'] = round(cancer_ds['Number of sexual partners'].convert_objects(convert_numeric=True))
cancer_ds['First sexual intercourse'] = cancer_ds['First sexual intercourse'].convert_objects(convert_numeric=True)
cancer_ds['Num of pregnancies']=round(cancer_ds['Num of pregnancies'].convert_objects(convert_numeric=True))
cancer_ds['Smokes'] = cancer_ds['Smokes'].convert_objects(convert_numeric=True)
cancer_ds['Smokes (years)'] = cancer_ds['Smokes (years)'].convert_objects(convert_numeric=True)
cancer_ds['Hormonal Contraceptives'] = cancer_ds['Hormonal Contraceptives'].convert_objects(convert_numeric=True)
cancer_ds['Hormonal Contraceptives (years)'] = cancer_ds['Hormonal Contraceptives (years)'].convert_objects(convert_numeric=True)
cancer_ds['IUD (years)'] = cancer_ds['IUD (years)'].convert_objects(convert_numeric=True)

print('minimum:',min(cancer_ds['Hormonal Contraceptives (years)']))
print('maximum:',max(cancer_ds['Hormonal Contraceptives (years)']))
g = sns.PairGrid(cancer_ds,
                 y_vars=['Hormonal Contraceptives'],
                 x_vars= category_ds,
                 aspect=.75, size=3.5)
g.map(sns.barplot, palette="pastel");
cancer_ds['Age'].hist(bins=70)
plt.xlabel('Age')
plt.ylabel('Count')
print('Mean age of the Women facing the risk of Cervical cancer',cancer_ds['Age'].mean())
for feature in category_ds:

 as_fig = sns.FacetGrid(cancer_ds,hue=feature,aspect=5)

 as_fig.map(sns.kdeplot,'Age',shade=True)

 oldest = cancer_ds['Age'].max()

 as_fig.set(xlim=(0,oldest))

 as_fig.add_legend()
for feature in category_ds:
  
  sns.factorplot(x='Number of sexual partners',y='Age',hue=feature,data=cancer_ds,aspect=1.95,kind='bar');
sns.distplot(cancer_ds['First sexual intercourse'].convert_objects(convert_numeric=True))

for feature in category_ds:

 as_fig = sns.FacetGrid(cancer_ds,hue=feature,aspect=5)

 as_fig.map(sns.kdeplot,'Age',shade=True)

 oldest = cancer_ds['Age'].max()

 as_fig.set(xlim=(0,oldest))

 as_fig.add_legend()
cancer_ds['Number of sexual partners'].corr(cancer_ds['Num of pregnancies'])
cancer_ds['Smokes (packs/year)'] = cancer_ds['Smokes (packs/year)'].convert_objects(convert_numeric=True)
print('Correlation between Smokes and Smokes (years) feature:',cancer_ds['Smokes'].corr(cancer_ds['Smokes (years)']))
print('Correlation between Smokes and Smokes (packs/year) feature:',cancer_ds['Smokes'].corr(cancer_ds['Smokes (packs/year)']))
cancer_ds.drop('Smokes',axis=1,inplace=True)
smokes_table = pd.crosstab(index=cancer_ds["Schiller"], columns=(cancer_ds["Smokes (years)"]))
g = sns.PairGrid(cancer_ds,
                 y_vars=['Smokes (years)'],
                 x_vars= category_ds,
                 aspect=.75, size=3.5)
g.map(sns.stripplot, palette="winter");
cancer_ds.drop('Hormonal Contraceptives',axis=1,inplace=True)
harmones_table = pd.crosstab(index=cancer_ds["Schiller"], columns=(cancer_ds["Hormonal Contraceptives (years)"]))
for feature in category_ds:

 as_fig = sns.FacetGrid(cancer_ds,hue=feature,aspect=5)

 as_fig.map(sns.kdeplot,'Hormonal Contraceptives (years)',shade=True)

 oldest = cancer_ds['Hormonal Contraceptives (years)'].max()

 as_fig.set(xlim=(0,oldest))

 as_fig.add_legend()
cancer_ds.drop('IUD',axis=1,inplace=True)
sns.factorplot('IUD (years)',data=cancer_ds,kind='count',aspect=5)
g = sns.PairGrid(cancer_ds,
                 y_vars=['IUD (years)'],
                 x_vars= category_ds,
                 aspect=.75, size=5.5)
g.map(sns.violinplot, palette="Accent",inner='stick');
HU_table = pd.crosstab(index=cancer_ds["Biopsy"], columns=(cancer_ds["IUD (years)"]))
cancer_ds['STDs (number)'] = round(cancer_ds['STDs (number)'].convert_objects(convert_numeric=True))
sns.countplot('STDs (number)',data=cancer_ds)
std_table = pd.crosstab(index=cancer_ds["Hinselmann"], 
                          columns=cancer_ds["STDs (number)"])

std_table
cancer_ds.drop('Dx',axis=1,inplace=True)
cancer_ds.info()
cancer_ds_features = cancer_ds.drop(['Hinselmann', 'Schiller', 'Citology','Biopsy'],axis=1)
cancer_ds_label = pd.DataFrame(data=cancer_ds['Hinselmann'])
cancer_ds.info()
cancer_ds_label['Schiller'] = cancer_ds['Schiller']
cancer_ds_label['Citology'] = cancer_ds['Citology']
cancer_ds_label['Biopsy'] = cancer_ds['Biopsy']
def cervical_cancer(cancer_label):
    
    hil, sch, cit, bio = cancer_label
    
    return hil+sch+cit+bio

cancer_ds_label['cervical_cancer'] = cancer_ds_label[['Hinselmann', 'Schiller', 'Citology','Biopsy']].apply(cervical_cancer,axis=1)
cancer_ds_label.drop(['Hinselmann', 'Schiller', 'Citology','Biopsy'],axis=1,inplace=True)
print('Value counts of each target variable:',cancer_ds_label['cervical_cancer'].value_counts())
cancer_ds_label = cancer_ds_label.astype(int)
cancer_ds_label = cancer_ds_label.values.ravel()

print('Final feature vector shape:',cancer_ds_features.shape)
print('Final target vector shape',cancer_ds_label.shape)
import numpy as np
np.random.seed(42)
df_data_shuffle = cancer_ds.iloc[np.random.permutation(len(cancer_ds))]

df_train = df_data_shuffle.iloc[1:686, :]
df_test = df_data_shuffle.iloc[686: , :]
df_train_feature = df_train[['Age', 'Number of sexual partners', 'First sexual intercourse',
       'Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)',
       'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)',
       'STDs:condylomatosis', 'STDs:cervical condylomatosis',
       'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis',
       'STDs:syphilis', 'STDs:pelvic inflammatory disease',
       'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS',
       'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
       'STDs','Dx:Cancer','Dx:CIN', 'Dx:HPV','Hinselmann', 
       'Citology','Schiller']]

train_label = np.array(df_train['Biopsy'])

df_test_feature = df_test[['Age', 'Number of sexual partners', 'First sexual intercourse',
       'Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)',
       'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)',
       'STDs:condylomatosis', 'STDs:cervical condylomatosis',
       'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis',
       'STDs:syphilis', 'STDs:pelvic inflammatory disease',
       'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS',
       'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
       'STDs','Dx:Cancer','Dx:CIN', 'Dx:HPV','Hinselmann', 
       'Citology','Schiller']]

test_label = np.array(df_test['Biopsy'])
from sklearn import preprocessing
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
train_feature = minmax_scale.fit_transform(df_train_feature)
test_feature = minmax_scale.fit_transform(df_test_feature)
print(train_feature[0])
print(train_label[0])
print(test_feature[0])
print(test_label[0])
train_feature.shape
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
X = np.array(cancer_ds.drop('Biopsy',1))
X = preprocessing.scale(X)
y = np.array(cancer_ds['Biopsy'])
accuracy = []
x_range = []
for j in range(1000):
    x_range.append(j)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=20, criterion='gini', min_samples_split=2,max_features='auto')
    clf.fit(X_train,y_train)
    acc = clf.score(X_test,y_test)
    accuracy.append(acc)
plt.title(' Random Forest')
plt.plot(x_range, accuracy)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()
print(acc)

test_label = np.array(cancer_ds['Biopsy'])

clf = RandomForestClassifier(n_estimators=20, criterion='gini', min_samples_split=2,max_features='auto')
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

print(B1P1)
print(B1P0)
print(B0P1)
print(B0P0)


conf = np.array([[B1P1,B0P1],[B1P0,B0P0]])
ds_cm = pd.DataFrame(conf, columns = [i for i in cols], index = [i for i in rows])

f, ax= plt.subplots(figsize = (5, 5))
sns.heatmap(ds_cm, annot=True, ax=ax) 
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
