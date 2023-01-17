import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

plt.style.use('fivethirtyeight')
df = pd.read_csv('../input/responses.csv')



music = df.iloc[:,1:19]

movies = df.iloc[:,19:31]

interests = df.iloc[:,31:46]

hobbies = df.iloc[:,46:63]

phobias = df.iloc[:,63:73]

health = df.iloc[:,73:76]

traits = df.iloc[:,76:133]

spending = df.iloc[:,133:140]

demographics = df.iloc[:,140:150]



feature = interests.join([music.iloc[:,0],movies.iloc[:,0],hobbies,phobias,health.iloc[:,-1]])

spending.fillna(spending.mean(),inplace=True)

# Owing to using machine learning, I used fillna method instead of dropping the missing data.  

feature.fillna(feature.mean(),inplace=True).head()
plt.figure(figsize=(15,6))

sns.barplot(x = spending.describe().columns, y=spending.describe().loc['mean'])
sns.heatmap(spending.corr(),annot=True)
demo1 = ['Gender','Left - right handed','Education','Only child','Village - town','House - block of flats']



dic = dict()

for a,b in enumerate(demo1):

    dic[a] = b



fig, axes = plt.subplots(2,3,figsize=(8,8))

num = 0

for i in range(2):

    for j in range(3):

        sns.countplot(demographics[dic[num]],ax = axes[i,j])

        axes[i,j].set_title(dic[num],fontsize=10)

        axes[i,j].set_xlabel('')

        axes[i,j].set_ylabel('')

        axes[i,j].set_xticklabels(labels=demographics[dic[num]].unique(), fontsize=7)

        num+=1

        

axes[0,2].set_xticklabels(labels=demographics['Education'],rotation=20, fontsize=6)



print('Female/Male:\t\t{}/{}({:.2f})'.format(demographics.groupby('Gender').count()['Age'][0],

                                          demographics.groupby('Gender').count()['Age'][1],

                                          demographics.groupby('Gender').count()['Age'][0]/

                                          demographics.groupby('Gender').count()['Age'][1]))

print('Left/Right:\t\t{}/{}({:.2f})'.format(demographics.groupby('Left - right handed').count()['Age'][0],

                                          demographics.groupby('Left - right handed').count()['Age'][1],

                                          demographics.groupby('Left - right handed').count()['Age'][0]/

                                          demographics.groupby('Left - right handed').count()['Age'][1]))

print('Only child No/Yes:\t{}/{}({:.2f})'.format(demographics.groupby('Only child').count()['Age'][0],

                                          demographics.groupby('Only child').count()['Age'][1],

                                          demographics.groupby('Only child').count()['Age'][0]/

                                          demographics.groupby('Only child').count()['Age'][1]))

print('City/Village:\t\t{}/{}({:.2f})'.format(demographics.groupby('Village - town').count()['Age'][0],

                                          demographics.groupby('Village - town').count()['Age'][1],

                                          demographics.groupby('Village - town').count()['Age'][0]/

                                          demographics.groupby('Village - town').count()['Age'][1]))

print('Block of flats/House:\t{}/{}({:.2f})'.format(demographics.groupby('House - block of flats').count()['Age'][0],

                                          demographics.groupby('House - block of flats').count()['Age'][1],

                                          demographics.groupby('House - block of flats').count()['Age'][0]/

                                          demographics.groupby('House - block of flats').count()['Age'][1]))
demo2 = ['Age','Height','Weight','Number of siblings']



demographics[demo2].dropna().describe()
like = {}

for i in spending.columns:

    df_temp = demographics[spending[i]>=4]

    like[i] = df_temp

    

temp = []

for i in spending.columns:

    temp.append(like[i].groupby('Gender').count()['Age'][0])

plt.figure(figsize=(14,6))

plt.title('Female')

plt.axhline(y=np.mean(temp), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,temp)

    #print('{}'.format(i),like[i].groupby('Gender').count()['Age'])

    #print('{}'.format(i),notlike[i].groupby('Gender').count()['Age'])

#demographics

temp = []

for i in spending.columns:

    temp.append(like[i].groupby('Gender').count()['Age'][1])

plt.figure(figsize=(14,6))

plt.title('Male')

plt.axhline(y=np.mean(temp), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,temp)



temp = []

for i in spending.columns:

    temp.append(like[i].groupby('Gender').count()['Age'][0]/like[i].groupby('Gender').count()['Age'][1])

plt.figure(figsize=(14,6))

plt.title('Female/Male Ratio')

plt.axhline(y=np.mean(temp), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,temp)
temp = []

for i in spending.columns:

    temp.append(like[i].groupby('Left - right handed').count()['Age'][0])

plt.figure(figsize=(14,6))

plt.title('Left-hand')

plt.axhline(y=np.mean(temp), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,temp)

    

temp = []

for i in spending.columns:

    temp.append(like[i].groupby('Left - right handed').count()['Age'][1])

plt.figure(figsize=(14,6))

plt.title('Right-hand')

plt.axhline(y=np.mean(temp), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,temp)



temp = []

for i in spending.columns:

    temp.append(like[i].groupby('Left - right handed').count()['Age'][0]/

                like[i].groupby('Left - right handed').count()['Age'][1])

plt.figure(figsize=(14,6))

plt.title('Left/Right-hand Ratio')

plt.axhline(y=np.mean(temp), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,temp)
h_ed = []

for i in spending.columns:

    

    h_ed.append(like[i].groupby('Education').count()['Age'][0]+

                like[i].groupby('Education').count()['Age'][2]+

                like[i].groupby('Education').count()['Age'][3])

plt.figure(figsize=(14,6))

plt.title('High Education Level')

plt.axhline(y=np.mean(h_ed), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,h_ed)



l_ed = []

for i in spending.columns:

    

    l_ed.append(like[i].groupby('Education').count()['Age'][1]+

                like[i].groupby('Education').count()['Age'][4]+

                like[i].groupby('Education').count()['Age'][5])

plt.figure(figsize=(14,6))

plt.title('Low Education Level')

plt.axhline(y=np.mean(l_ed), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,l_ed)



temp = []

for i in range(len(spending.columns)):

    temp.append(h_ed[i]/l_ed[i])



plt.figure(figsize=(14,6))

plt.title('High/Low Level')

plt.axhline(y=np.mean(temp), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,temp)
temp = []

for i in spending.columns:

    temp.append(like[i].groupby('Only child').count()['Age'][0])

plt.figure(figsize=(14,6))

plt.title('Only child: {}'.format(like[i].groupby('Only child').count()['Age'].index[0]))

plt.axhline(y=np.mean(temp), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,temp)



temp = []

for i in spending.columns:

    temp.append(like[i].groupby('Only child').count()['Age'][1])

plt.figure(figsize=(14,6))

plt.title('Only child: {}'.format(like[i].groupby('Only child').count()['Age'].index[1]))

plt.axhline(y=np.mean(temp), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,temp)



temp = []

for i in spending.columns:

    temp.append(like[i].groupby('Only child').count()['Age'][0]/

                like[i].groupby('Only child').count()['Age'][1])

plt.figure(figsize=(14,6))

plt.title('Only child No/Yes')

plt.axhline(y=np.mean(temp), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,temp)
temp = []

for i in spending.columns:

    temp.append(like[i].groupby('Village - town').count()['Age'][0])

plt.figure(figsize=(14,6))

plt.title('{}'.format(like[i].groupby('Village - town').count()['Age'].index[0]))

plt.axhline(y=np.mean(temp), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,temp)



temp = []

for i in spending.columns:

    temp.append(like[i].groupby('Village - town').count()['Age'][1])

plt.figure(figsize=(14,6))

plt.title('{}'.format(like[i].groupby('Village - town').count()['Age'].index[1]))

plt.axhline(y=np.mean(temp), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,temp)



temp = []

for i in spending.columns:

    temp.append(like[i].groupby('Village - town').count()['Age'][0]/

                like[i].groupby('Village - town').count()['Age'][1])

plt.figure(figsize=(14,6))

plt.title('City/Village Ratio')

plt.axhline(y=np.mean(temp), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,temp)
temp = []

for i in spending.columns:

    temp.append(like[i].groupby('House - block of flats').count()['Age'][0])

plt.figure(figsize=(14,6))

plt.title('{}'.format(like[i].groupby('House - block of flats').count()['Age'].index[0]))

plt.axhline(y=np.mean(temp), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,temp)



temp = []

for i in spending.columns:

    temp.append(like[i].groupby('House - block of flats').count()['Age'][1])

plt.figure(figsize=(14,6))

plt.title('{}'.format(like[i].groupby('House - block of flats').count()['Age'].index[1]))

plt.axhline(y=np.mean(temp), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,temp)



temp = []

for i in spending.columns:

    temp.append(like[i].groupby('House - block of flats').count()['Age'][0]/

                like[i].groupby('House - block of flats').count()['Age'][1])

plt.figure(figsize=(14,6))

plt.title('Block of flats/House Ratio')

plt.axhline(y=np.mean(temp), color='k', lw=4, ls='dashed')

sns.barplot(spending.columns,temp)
for i in spending.columns:

    print('TITLE: {}'.format(i))

    print(like[i][demo2].dropna().describe())

    print('\n')
## Get Dummies

smoking = pd.get_dummies(health['Smoking'],drop_first=True)

alcohol = pd.get_dummies(health['Alcohol'],drop_first=True)

gender  = pd.get_dummies(demographics['Gender'],drop_first=True)

handed  = pd.get_dummies(demographics['Left - right handed'],drop_first=True)

child   = pd.get_dummies(demographics['Only child'],drop_first=True)

vil_tow = pd.get_dummies(demographics['Village - town'],drop_first=True)

resid   = pd.get_dummies(demographics['House - block of flats'],drop_first=True)

edu   = pd.get_dummies(demographics['Education'],drop_first=True)



features = feature.join([smoking,alcohol,gender,handed,child,vil_tow,resid,edu])

features.fillna(feature.mean(),inplace=True).head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix



def binary(num):

    if num >= 4:

        return 1

    else:

        return 0



X = features

count = 1    

for i in spending.columns:



    y = spending.fillna(spending.mean(),inplace=True)[i].apply(binary)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=23)



    lr = LogisticRegression()

    lr.fit(X_train,y_train)

    lr_pred = lr.predict(X_test)

    

    print('{}. {}'.format(count,i.upper()))

    print('  A. Confusion matrix:\n',confusion_matrix(y_test,lr_pred),'\n')

    print('  B. Classification Report:\n',classification_report(y_test,lr_pred),'\n')

    count += 1

    
from sklearn.ensemble import RandomForestClassifier



X = features

count = 1

for i in spending.columns:



    y = spending.fillna(spending.mean(),inplace=True)[i].apply(binary)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=23)



    rfc = RandomForestClassifier(n_estimators=500)

    rfc.fit(X_train, y_train)

    rfc_pred = rfc.predict(X_test)



    print('{}. {}'.format(count,i.upper()))

    print('  A. Confusion matrix:\n',confusion_matrix(y_test,rfc_pred),'\n')

    print('  B. Classification Report:\n',classification_report(y_test,rfc_pred),'\n')

    count += 1
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



X = features

scaler = StandardScaler()

scaler.fit(X)

scaled_data = scaler.transform(X)

pca = PCA(n_components=2)

pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)

    

fig, axes = plt.subplots(3,3,figsize=(10,8),sharex=True,sharey=True)



num = 0

for i in range(3):

    for j in range(3):

        y = spending.fillna(spending.mean(),inplace=True)[spending.columns[num]].apply(binary)

        axes[i,j].scatter(x_pca[:,0],x_pca[:,1],c=y,cmap='plasma')

        axes[i,j].set_title('{}'.format(spending.columns[num]),fontsize=10)



        num+=1

        if num == 7:break
df_comp = pd.DataFrame(pca.components_,columns=X.columns)

plt.figure(figsize=(12,6))

sns.heatmap(df_comp,cmap='plasma',)
from xgboost import XGBClassifier

import xgboost as xgb



X = features

count = 1



for i in spending.columns:



    X = features

    y = spending.fillna(spending.mean(),inplace=True)[i].apply(binary)



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=23)



    xgbc = XGBClassifier()

    xgbc.fit(X_train, y_train)

    xgbc_pred = xgbc.predict(X_test)

    

    fig = plt.figure(figsize=(12,12))

    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    xgb.plot_importance(xgbc,height=0.5,ax=axes,title='{}'.format(i.upper()))



    print('{}. {}'.format(count,i.upper()))

    print('  A. Confusion matrix:\n',confusion_matrix(y_test,xgbc_pred),'\n')

    print('  B. Classification Report:\n',classification_report(y_test,xgbc_pred),'\n')

    count += 1
#from numpy import sort

#from sklearn.feature_selection import SelectFromModel

#from sklearn.metrics import accuracy_score
'''

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Fit model using each importance as a threshold

thresholds = sort(clf.feature_importances_)

for thresh in thresholds:

    # select features using threshold

    selection = SelectFromModel(clf, threshold=thresh, prefit=True)

    select_X_train = selection.transform(X_train)

    # train model

    selection_model = XGBClassifier()

    selection_model.fit(select_X_train, y_train)

    # eval model

    select_X_test = selection.transform(X_test)

    y_pred = selection_model.predict(select_X_test)

    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)

    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

    

'''