#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import sklearn
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn import linear_model
#import data

data = pd.read_csv('../input/student-mat.csv', sep=';')

data.head()
data.info()
data.describe()
dataColumnsValues = data.columns.values; dataColumnsValues
#label
label = 'G3'
#Selectiong columns dtype int64

dataColumns_int64 = data.select_dtypes(include='int64'); dataColumns_int64

#selection names of columns with dtype int64

namesColInt64 = dataColumns_int64.columns.values; namesColInt64
#Plotagens scatter column G3 vs columns dtype int64 

numPlots = len(dataColumns_int64.columns) # num columns dtype int64

fig, ax = plt.subplots(round(numPlots/4),round(numPlots/4), figsize=(10,10)) # subplots(4,4) # Alterando os valores em figsize é possível melhorar os detalhes das plotagens.
fig.subplots_adjust(wspace=0.6, hspace=0.8) # space with and height of plots
fig.suptitle('Plot scatter G3 vs columns int64')

n = 0 # control, contador index column 
for i in range(round(numPlots/4)):
    for j in range(round(numPlots/4)):
        col = namesColInt64[n]; n += 1
        ax[i][j].scatter(dataColumns_int64[col],dataColumns_int64[label], color='r', alpha=0.5) #label = G3
        ax[i][j].set_title(label + ' vs '+ col)
        ax[i][j].set_xlabel(col)
        ax[i][j].set_ylabel(label)
        
plt.show()

# value_counts of columns dtype int64

for i in range(len(namesColInt64)):
    col = namesColInt64[i]
    print('Value counts column '+ col)
    print(dataColumns_int64[col].value_counts()); print(40*'-')

# histrogram, frequency columns dtype int64

fig, ax = plt.subplots(round(numPlots/4),round(numPlots/4), figsize=(14,10))
fig.subplots_adjust(wspace=0.6, hspace=0.8)
fig.suptitle('Histograms columns dtype int64')

n = 0
for i in range(round(numPlots/4)):
    for j in range(round(numPlots/4)):
        col = namesColInt64[n]; n +=1
        ax[i][j].hist(dataColumns_int64[col], color='r', edgecolor='black')
        ax[i][j].set_title('Frequency '+col)
        ax[i][j].set_xlabel(col)
        ax[i][j].set_ylabel('Frequency')

plt.show()


#Plots scatter columns dtype int64 vs columns dtype int64
#Plot scatter of all columns vs all columns with dtype int64
#This process can take a big time
#Each int64 columns's of data it is ploted here using plot kind scatter, each column it is ploted in two lines 
#of 8 columns in a total of 16 subplots to each column in axis y. Repare that are 16 int64 columns's.

k = 0 # control the column in axis y to each set of subplots.
#Create 16 subplots em format 2 lines and 8 columns
for m in range(2):
    for n in range(8):
        col = namesColInt64[k]; k +=1 # select the name of column to be ploted in axis y
        #create objet Figure and Axe with 2 lines and 8 columns.
        fig, ax = plt.subplots(2, 8, figsize=(16,4)) #Sharey=True, compartilhar eixo y.
        fig.subplots_adjust(wspace=0.4, hspace=0.7) # Some adjusts in space between subplots
        fig.suptitle(col+' vs columns dtype int64') # title figure

        # To each subplot create above it is fit with data. 
        #Esses laços são criados para preencher cada subplot criado acima, ou seja, preencher os 16 subplots
        # em formato de 2 linhas e 8 colunas.
        c = 0 #Control axis x to be ploted with axis y.
        for i in range(2):
            for j in range(8):
                linha = namesColInt64[c]; c +=1 #Control the axis x.
                ax[i][j].scatter(dataColumns_int64[linha], dataColumns_int64[col], color='r', alpha=0.5)
                ax[i][j].set_title('y vs '+linha)
                ax[i][j].set_xlabel(linha)
                ax[i][j].set_ylabel(col)

# Repare that it is created many Figure and Axes objets.
# It is create a object Figure and Axes to each column in axis y to be ploted.
# Each object Figure and Axes have much subplots that represent the other columns, represent in axis x, to be 
#ploted.

#Plot (scatter and histogram) to compare, more details about some axis/column

# select here the axis/columns
axiY = 'absences' # Axi Y
axiX = 'age' # Axi X



#----------------------------------- plot

fig, ax1 = plt.subplots(1,1, figsize=(12,6))

#Plot scatter
ax1.scatter(dataColumns_int64[axiX], dataColumns_int64[axiY], color='r', alpha=0.6)
ax1.set_title(axiY + ' vs ' + axiX)
ax1.set_xlabel(axiX)
ax1.set_ylabel(axiY)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(color='grey', alpha=0.5)
plt.show()

# Plots histograms
fig2, (ax21, ax22) = plt.subplots(1,2, figsize=(12,3))
fig2.subplots_adjust(wspace=0.3)

ax21.hist(dataColumns_int64[axiX], label=axiX, edgecolor='black') #density=True para frequência relativa
ax21.set_title('Histogram '+axiX)
ax21.set_xlabel(axiX)
ax21.set_ylabel('Frequency')
ax21.spines['top'].set_visible(False)
ax21.spines['right'].set_visible(False)
ax21.legend()

ax22.hist(dataColumns_int64[axiY], label=axiY,color='g', edgecolor='black') #density=True para frequência relativa
ax22.set_title('Histogram '+axiY)
ax22.set_xlabel(axiY)
ax22.set_ylabel('Frequency')
ax22.spines['top'].set_visible(False)
ax22.spines['right'].set_visible(False)
ax22.legend()

plt.show()
#selection columns dtype object

dataColumns_object = data.select_dtypes(include='object'); dataColumns_object
#selection names of columns dtype object
namesColObject = dataColumns_object.columns.values; namesColObject
# Example values and value_counts applying about columns school

print('Values column school: \n\n',data['school'].values); print(50*'-')
print('Value counts column school: \n\n',data['school'].value_counts())
# print out columns dtype object with yours value_counts

for i in range(len(namesColObject)):
    col = namesColObject[i]
    print('value counts column '+col+': ')
    print(dataColumns_object[col].value_counts()); print(40*'-')
    
# Having some problem

#numColObject = len(namesColObject) # 17

#fig, ax = plt.subplots(round(numColObject/3),round(numColObject/6), figsize=(13,14))
#fig.subplots_adjust(hspace=0.8,wspace=0.4)
#fig.suptitle('Histograms columns dtype object')

#n = 0
#for i in range(round(numColObject/3)):
#    for j in range(round(numColObject/6)):
#        col = namesColObject[n]; n +=1
#        ax[i][j].hist(dataColumns_object[col], edgecolor='black')
#        ax[i][j].set_title('histogram: Frequency '+col)
#        ax[i][j].set_xlabel(col)
#        ax[i][j].set_ylabel('frequency')
        
#        if(n==17): # Pois existem somentes 17 plots e os laços vão até 18
#            break
        
#plt.show()


# Preprocessing data dtype object using label encoder sklearn 

labEncod = preprocessing.LabelEncoder()

newColumns = pd.DataFrame()

for i in range(len(namesColObject)):
    for j in range(len(dataColumns_object)):
        newCol = labEncod.fit_transform(list(dataColumns_object[namesColObject[i]]))
        newColumns[namesColObject[i]] = newCol
    
newColumns # DataFrame with object dtype columns with new labels.
# Gerating correlation matriz to object dtype of dada

matrixCorrelation = newColumns.corr(); matrixCorrelation
#Ploting heatMap of correlation Matriz above

fig, ax = plt.subplots(figsize=(15,8))
sn.heatmap(matrixCorrelation, annot=True)
plt.plot()



# concatenando dataFrames with int64 and object dtypes.
frames = [dataColumns_int64, newColumns]
#data = pd.concat(frames)
nData = pd.concat(frames, axis=1) # 1 indica que a concatenação é feita pelas colunas.

#nData.info()

matCorrelat = nData.corr()
fig, ax = plt.subplots(figsize=(8,10))

sn.heatmap(matCorrelat[['G3']], annot=True) # Plotando correlação de todas as features com apenas a variavel alvo.
plt.plot()

# Repare pela plotagem do heatmap que para as feature categoricas, a varivale alvo (G3, que é uma variavel numerica float),
# apresenta um espaço em branco, print out the matCorrelat, a matrix de correlação, pode perceber-se que 
# esses valores são nan.

#label and features

#x = nData[['Medu']]
#x = np.array(nData.drop([label,'G1','G2'], 1)) # features
x = np.array(nData.drop([label], 1)) # features
y = np.array(nData[label]) # label


# split data in train and test

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
# 10% of data to test and 90% to train


# Building model, accuracy

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)

print('Accuracy: ',accuracy)

# Compare predicted, features, realValue

pred = []

predicted = linear.predict(x_test)

for i in range(len(x_test)):
    pred.append(['{0:.3f}'.format(predicted[i]), x_test[i], y_test[i]])

dfPred = pd.DataFrame(pred, columns=['Predicted', 'features', 'realValue']); dfPred