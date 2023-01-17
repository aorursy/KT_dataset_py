import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

cols = list(df.columns)

df.head()
df.info()

df.describe()
fig, ax = plt.subplots(3,4)

k = 0

for i in range(3):

    for j in range(4):

        ax[i,j].hist(cols[k], data = df, edgecolor= 'black', linewidth= 1, color= 'red')

        plt.subplots_adjust(left=1, bottom=1, right=2, top=2, wspace=1, hspace=1)

        ax[i,j].set_title(str(cols[k]))

        k+=1

fig.set_size_inches(15,10)

plt.gcf()

fig.tight_layout()
corr_mat = df.corr()

mask = np.array(corr_mat)

mask[np.tril_indices_from(mask)] = False

fig = plt.gcf()

fig.set_size_inches(40,10)

sns.heatmap(data=corr_mat,mask=mask,square=True,annot=True,cbar=True)
def single_feature(target='quality',feature_x=cols[0], data=df, kind='bar'):

    sns.factorplot(x=target, y=feature_x, data=data, kind=kind)

single_feature(feature_x='citric acid', kind='swarm')
fig, ax = plt.subplots()

fig.subplots_adjust(wspace = 0.4)

for i in range(0,12):

    plt.title(str(df.columns[i]) + ' vs quality')

    sns.barplot(x = 'quality', y = str(cols[i]), data = df)

    fig.tight_layout()

    plt.show()
sns.pairplot(df)
from collections import Counter

Counter(df['quality'])

sns.countplot(x = 'quality', data = df)
fig, ax = plt.subplots()

fig.subplots_adjust(wspace = 0.4)

for i in range(0,12):

    plt.title(str(df.columns[i]) + ' vs quality')

    sns.boxplot(x = 'quality', y = str(cols[i]), data = df)

    fig.tight_layout()

    plt.show()
score = []

for i in df['quality']:

    if i>=1 and i<4:

        score.append(1)

    elif i>=4 and i<8:

        score.append(2)

    elif i>=8 and i<10:

        score.append(3)

df['score'] = score
Counter(df['score'])
X = df.iloc[:, :11]

y = df['score']
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit_transform(X)
from sklearn.decomposition import PCA

pca = PCA()

pca.fit_transform(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.grid()
pca_new = PCA(n_components=8)

X_new = pca_new.fit_transform(X)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.25)
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import LinearSVC,SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score



from sklearn.metrics import confusion_matrix, accuracy_score
models=[LogisticRegression(),LinearSVC(),SVC(kernel='rbf'),KNeighborsClassifier(),RandomForestClassifier(),

        DecisionTreeClassifier(),GradientBoostingClassifier(),GaussianNB()]

model_names=['LogisticRegression','LinearSVM','rbfSVM','KNearestNeighbors','RandomForestClassifier','DecisionTree',

             'GradientBoostingClassifier','GaussianNB']

acc_score=[]



for model in range(len(models)):

    clf=models[model]

    clf.fit(X_train,y_train)

    pred=clf.predict(X_test)

    acc_score.append(accuracy_score(pred,y_test))

     

d={'Modelling Algo':model_names,'Accuracy':acc_score}



acc_table=pd.DataFrame(d)

acc_table
sns.barplot(y='Modelling Algo',x='Accuracy',data=acc_table)

plt.xlabel('Learning Models')

plt.ylabel('Accuracy scores')

plt.title('Accuracy levels of different classification models')



single_feature(target='Accuracy',feature_x='Modelling Algo',data = acc_table,kind= 'point')