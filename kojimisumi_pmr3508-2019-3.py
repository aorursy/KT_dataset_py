import pandas as pd
adults = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv',names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?") 
adults = adults.drop(adults.index[0])
adults.head()
adults.shape
adults.isnull().values.any()
adults['Occupation'].value_counts()
import matplotlib.pyplot as plt
adults['Age'].value_counts().plot(kind='pie')
import pandas_profiling as pp
pp.ProfileReport(adults)
adults['Country'].value_counts().plot(kind='bar')
testadults = pd.read_csv('/kaggle/input/adult-pmr3508/test_data.csv',names=["Id",

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
testadults
Xadults = adults[["Age","Relationship","Hours per week","Education-Num","Country","Target"]]
Xadults.isnull().values.any()
Xadults = Xadults.dropna()
Xadults.shape
Yadults = Xadults[['Target']]
Xadults = Xadults.drop(['Target'], axis=1)
Xadults
Yadults
for i in Xadults.index:

    if Xadults.at[i,'Country'] == 'United-States':

        Xadults.at[i,'Country'] = 1

    else: 

        Xadults.at[i,'Country'] = 0
Xadults.head()
Xadults['Relationship'].value_counts()
Xadults = Xadults.replace(['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'], [1, 2, 2, 2, 1, 2])
Xadults.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
#calculando melhor n



from tqdm import tqdm



y = []

x = range(3, 30)

n = 0

s = 0.0

for i in tqdm(x):

    knn = KNeighborsClassifier(n_neighbors=i)

    scores = cross_val_score(knn, Xadults, Yadults, cv=5)

    if scores.mean()>s:

        n=i

        s=scores.mean()

    y.append(scores.mean())

plt.scatter(x, y)



#selecionando o melhor modelo

knn = KNeighborsClassifier(n_neighbors=n)

scores = cross_val_score(knn, Xadults, Yadults, cv=10)

print('melhor n:', n, 'score:', scores.mean())
knn = KNeighborsClassifier(n_neighbors=16)
scores
scores = cross_val_score(knn, Xadults, Yadults, cv=10)
scores
Xadults[["Capital Gain", "Capital Loss"]] = adults[["Capital Gain", "Capital Loss"]]

Xadults = Xadults.dropna()

scores = cross_val_score(knn, Xadults, Yadults, cv=10)

scores.mean()
ntestadults = testadults.drop(testadults.index[0])
ntestadults
Xtest = ntestadults[["Age","Relationship","Hours per week","Education-Num","Country","Capital Gain", "Capital Loss"]]
Ytest = ntestadults.Target
for i in Xtest.index:

    if Xtest.at[i,'Country'] == 'United-States':

        Xtest.at[i,'Country'] = 1

    else: 

        Xtest.at[i,'Country'] = 0
Xtest = Xtest.replace(['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative'], [1, 2, 2, 2, 1, 2])
knn.fit(Xadults,Yadults)
scores = cross_val_score(knn, Xadults, Yadults, cv=10)
scores
YtestPred = knn.predict(Xtest)
Xadults.head()
from sklearn.metrics import accuracy_score

YtestPred.shape
Id = [i for i in range(len(YtestPred))]



d = {'Id' : Id, 'income' : YtestPred}

myDf = pd.DataFrame(d) 

myDf.to_csv('bestPrediction.csv',

             index=False, sep=',', line_terminator = '\n', header = ["Id", "income"])
YtestPred