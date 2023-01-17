import statistics as stt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import linear_model as lm
from sklearn.neighbors import KNeighborsRegressor
test = pd.read_csv('../input/test.csv',
        sep = r'\s*,\s*',
        engine = 'python',
        na_values = "?")
test.rename(columns={'median_house_value':'target'}, inplace=True)
train = pd.read_csv('../input/train.csv',
        sep = r'\s*,\s*',
        engine = 'python',
        na_values = "?")
train.rename(columns={'median_house_value':'target'}, inplace=True)
names = list(train.columns)
names.remove('target')
names.remove('Id')
names
train.shape
train.head()
plt.hist(train.target)
plt.title("House Values")
plt.show()
plt.close()
plt.boxplot(train.target)
plt.title("House Values")
plt.show()
plt.close()
direct = train.filter(items=names)
mcorr = direct.corr()
mcorr
plt.matshow(mcorr)
amostra = train.sample(n=500)

plt.scatter(amostra['median_income'],amostra['target'], c='b')
plt.xlabel('Income')
plt.ylabel('Target')
plt.show()
plt.close()
# longitude e latitude Los Angeles
longLA = -118.25
latLA = 34.05
# longitude e latitude São Francisco
longSF = -122.42
latSF = 37.78
# longitude e latitude São Diego
longSD = -117.15
latSD = 32.78
# longitude e latitude Sacramento
longSC = -121.47
latSC = 38.56
train['LA'] = np.sqrt((train.longitude - longLA)**2 + (train.latitude - latLA)**2)
train['SF'] = np.sqrt((train.longitude - longSF)**2 + (train.latitude - latSF)**2)
train['SD'] = np.sqrt((train.longitude - longSD)**2 + (train.latitude - latSD)**2)
train['SC'] = np.sqrt((train.longitude - longSC)**2 + (train.latitude - latSC)**2)
names.append('LA'); names.append('SF'); names.append('SD'); names.append('SC')
train.head()
direct = train.filter(items=['longitude','latitude','LA','SF','SD','SC','target'])
mcorr = direct.corr()
mcorr
Xtrain = train[names]
Ytrain = train.target
reg = lm.LinearRegression()
reg.fit(Xtrain, Ytrain)
print(reg.intercept_)
coef = pd.DataFrame(list(zip(Xtrain.columns, reg.coef_)), columns= ['Features', 'Coef'])
coef
scores = cross_val_score(reg, Xtrain, Ytrain, cv=10)
print('Media:',stt.mean(scores))
print('Desvio:',stt.pstdev(scores))
print('R²:',reg.score(Xtrain, Ytrain))
print(stt.mean(scores)*reg.score(Xtrain, Ytrain))
nota = []
for n in range(1,51):
    knn = KNeighborsRegressor(n_neighbors = n)
    knn.fit(Xtrain, Ytrain)
    cvs = cross_val_score(knn, Xtrain, Ytrain, cv=10)
    media = stt.mean(cvs)
    scores = knn.score(Xtrain,Ytrain)
    if scores != 1.0:
        nota.append(media*scores)
    else:
        nota.append(0)
bestn = nota.index(max(nota))+1
print(bestn)
knn = KNeighborsRegressor(n_neighbors = bestn)
knn.fit(Xtrain, Ytrain)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
print('Media:',stt.mean(scores))
print('Desvio:',stt.pstdev(scores))
print('R²:',knn.score(Xtrain, Ytrain))
print(stt.mean(scores)*knn.score(Xtrain, Ytrain))
net = lm.ElasticNet(alpha=1, l1_ratio=.9)
net.fit(Xtrain, Ytrain)
scores = cross_val_score(net, Xtrain, Ytrain, cv=10)
print('Media:',stt.mean(scores))
print('Desvio:',stt.pstdev(scores))
print('R²:',net.score(Xtrain, Ytrain))
print(stt.mean(scores)*net.score(Xtrain, Ytrain))
test.head()
test['LA'] = np.sqrt((test.longitude - longLA)**2 + (test.latitude - latLA)**2)
test['SF'] = np.sqrt((test.longitude - longSF)**2 + (test.latitude - latSF)**2)
test['SD'] = np.sqrt((test.longitude - longSD)**2 + (test.latitude - latSD)**2)
test['SC'] = np.sqrt((test.longitude - longSC)**2 + (test.latitude - latSC)**2)
test.head()
Xtest = test.filter(items=names)
Y = reg.predict(Xtest)
Y = np.abs(Y)
answer = pd.DataFrame(list(zip(test.Id, Y)), columns= ['Id','median_house_value'])
answer.head()
#answer.to_csv(r"C:\Users\User\Desktop\sem_4\PMR3508-2018\california\california.csv",index=False)