import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns

from sklearn.model_selection import train_test_split

from scipy import stats

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn import tree

#from dtreeplt import dtreeplt

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier

import seaborn as sns

from sklearn.decomposition import PCA

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import plot_model

#from sklearn.preprocessing import StandardScaler

#from keras.utils import plot_model

#from IPython.display import Image

#from graphviz import *
data = pd.read_csv('../input/mobile-price-classification/train.csv')
data.head(5)
data.shape
data.columns
# train = data.iloc[ :1600]

# test = data.iloc[1600:]

train = data.sample(frac=0.8, random_state=5)

test = data.loc[~data.index.isin(train.index),:]
train.sample(5)
train.head(5)
train.shape
test.sample(5)
test.shape
train.isnull().sum()
train.describe()
sns_plot = sns.pairplot(train,hue='price_range')

sns_plot.savefig("pairplot_raw.png", format='png')

sns_plot
train['px_total'] = train['px_height'] * train['px_width']

train['3&4g'] = train['three_g'] * train['four_g']

train['sc_total'] = train['sc_w']*train['sc_h']

train['volume'] = train['sc_w']*train['sc_h'] * train['m_dep']



train['battry'] = train['battery_power'] * train['talk_time']

train['battry'] = train['battry'].round(2)

                         

train['battry_per_talk'] = train['battery_power'] / train['talk_time']

train['battry_per_talk'] = train['battry_per_talk'].round(2)

                                  

train['cal'] = train['clock_speed'] * train['ram']

train['cal'] = train['cal'].round(2)

                                  

train['ram_per_clock'] = train['ram'] / train['clock_speed']

train['ram_per_clock'] = train['ram_per_clock'].round(2)

                                  

train['clock_per_ram'] = train['clock_speed'] / train['ram'] 

train['clock_per_ram'] = train['clock_per_ram'].round(5)

                                  

train['memory_per_ram'] = train['int_memory'] / train['ram'] 

train['memory_per_ram'] = train['memory_per_ram'].round(5)

                                  

train['ram_per_memory'] = train['ram'] / train['int_memory'] 

train['ram_per_memory'] = train['ram_per_memory'].round(2)                               
train.sample(5)
sns_plot = sns.pairplot(train,hue='price_range')

sns_plot.savefig("pairplot_add_feature.png", format='png')

sns_plot
test['px_total'] = test['px_height'] * test['px_width']

test['3&4g'] = test['three_g'] * test['four_g']

test['sc_total'] = test['sc_w']*test['sc_h']

test['volume'] = test['sc_w']*test['sc_h'] * test['m_dep']



test['battry'] = test['battery_power'] * test['talk_time']

test['battry'] = test['battry'].round(2)

                         

test['battry_per_talk'] = test['battery_power'] / test['talk_time']

test['battry_per_talk'] = test['battry_per_talk'].round(2)

                                  

test['cal'] = test['clock_speed'] * test['ram']

test['cal'] = test['cal'].round(2)

                                  

test['ram_per_clock'] = test['ram'] / test['clock_speed']

test['ram_per_clock'] = test['ram_per_clock'].round(2)

                                  

test['clock_per_ram'] = test['clock_speed'] / test['ram'] 

test['clock_per_ram'] = test['clock_per_ram'].round(5)

                                  

test['memory_per_ram'] = test['int_memory'] / test['ram'] 

test['memory_per_ram'] = test['memory_per_ram'].round(5)

                                  

test['ram_per_memory'] = test['ram'] / test['int_memory'] 

test['ram_per_memory'] = test['ram_per_memory'].round(2)       
train.columns
train =train[['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',

       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',

       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',

       'touch_screen', 'wifi','px_total', '3&4g', 'sc_total',

       'volume','battry','battry_per_talk','cal','ram_per_clock','clock_per_ram',

       'memory_per_ram','ram_per_memory','price_range']] 



test = test[['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',

       'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',

       'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',

       'touch_screen', 'wifi','px_total', '3&4g', 'sc_total',

       'volume','battry','battry_per_talk','cal','ram_per_clock','clock_per_ram',

       'memory_per_ram','ram_per_memory','price_range']]
train.columns
len(train.columns)
train.head(5)
sns.countplot(x='price_range',data=train)
plt.figure(figsize=(15,8))

sns.violinplot("price_range", "battery_power", data=train)
plt.figure(figsize=(15,8))

sns.boxplot("price_range", "battery_power", data=train)
plt.figure(figsize=(15,8))

sns.violinplot("price_range", "ram", data=train)
plt.figure(figsize=(15,8))

sns.boxplot("price_range", "ram", data=train)
fstat, pval = stats.f_oneway(*[train.ram[data.price_range == s]

for s in train.price_range.unique()])

print("Oneway Anova ram ~ price_range F=%.2f, p-value=%E" % (fstat, pval))
plt.figure(figsize=(20,15))

correlation_matrix = train.corr().round(2)

sns.heatmap(data=correlation_matrix ,center=0 , linewidths=.7, annot=True)
pca = PCA(n_components=10)
pca.fit(train.iloc[:,:20])
# principalComponentsTrain = pca.fit_transform(train.iloc[:,:20])

# principalComponentsTrain
pca_train10 = pd.DataFrame()

pca_train10 = pd.DataFrame(data = pca.transform(train.iloc[:,:20]), columns =['pca1','pca2','pca3','pca4','pca5','pca6','pca7'

                                                               ,'pca8','pca9','pca10'])

pca_train10 = pca_train10.merge(train.iloc[:,31:], left_index=True, right_index=True)
pca_train10.sample(5)
# principalComponentsTest = pca.fit_transform(test.iloc[:,:20])

# principalComponentsTest
pca_test10 = pd.DataFrame()

pca_test10 = pd.DataFrame(data = pca.transform(test.iloc[:,:20]), columns =['pca1','pca2','pca3','pca4','pca5','pca6','pca7'

                                                               ,'pca8','pca9','pca10'])

pca_test10 = pca_test10.merge(test.iloc[:,31:], left_index=True, right_index=True)
pca_test10.sample(5)
sns_plot = sns.pairplot(pca_train10,hue='price_range')

sns_plot.savefig("pairplot_pca10.png", format='png')

sns_plot
pca = PCA(n_components=15)
pca.fit(train.iloc[:,:20])
# principalComponents = pca.fit_transform(train.iloc[:,:20])

# principalComponents
pca_train15 = pd.DataFrame(data = pca.transform(train.iloc[:,:20]), columns =['pca1','pca2','pca3','pca4','pca5','pca6','pca7'

                                                               ,'pca8','pca9','pca10','pca11','pca12','pca13'

                                                                ,'pca14','pca15'])

pca_train15 = pca_train15.merge(train.iloc[:,31:], left_index=True, right_index=True)
pca_train15.sample(5)
# principalComponentsTest = pca.fit_transform(test.iloc[:,:20])

# principalComponentsTest
pca_test15 = pd.DataFrame()

pca_test15 = pd.DataFrame(data =  pca.transform(test.iloc[:,:20]), columns =['pca1','pca2','pca3','pca4','pca5','pca6','pca7'

                                                               ,'pca8','pca9','pca10','pca11','pca12','pca13'

                                                                   ,'pca14','pca15'])

pca_test15 = pca_test15.merge(test.iloc[:,31:], left_index=True, right_index=True)
pca_test15.sample(5)
sns_plot = sns.pairplot(pca_train15,hue='price_range')

sns_plot.savefig("pairplot_pca15.png", format='png')

sns_plot
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,13], train.iloc[:,31:], 

                                                    test_size=0.3, random_state=0)
X_train = pd.DataFrame(X_train,columns=['ram'])

X_train.head(5)
X_test = pd.DataFrame(X_test,columns=['ram'])
y_train = pd.DataFrame(y_train,columns=['price_range'])

y_train.head(5)
y_test = pd.DataFrame(y_test,columns=['price_range'])
model = LogisticRegression(multi_class='auto')

model.fit(X_train, y_train)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_test = pd.DataFrame(test.iloc[:,13],columns=['ram'])

x_test.head(5)
y_pred = model.predict(x_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:20], train.iloc[:,31:], test_size=0.3, random_state=0)
X_train.head(5)
len(X_train)
y_train.head(5)
len(y_train)
model = LogisticRegression(multi_class='auto')

model.fit(X_train, y_train)
y_test.head(5)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
test.iloc[:,:20].head(5)
y_pred = model.predict(test.iloc[:,:20])
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:31], train.iloc[:,31:],

                                                    test_size=0.3, random_state=0)
X_train.head(5)
y_train.head(5)
model = LogisticRegression(multi_class='auto')

model.fit(X_train, y_train)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
y_pred = model.predict(test.iloc[:,:31])
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(pca_train10.iloc[:,:10], pca_train10.iloc[:,10:],

                                                    test_size=0.3, random_state=0)
X_train.head(5)
y_train.head(5)
model = LogisticRegression(multi_class='auto')

model.fit(X_train, y_train)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
y_pred = model.predict(pca_test10.iloc[:,:10])
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = pca_test10.iloc[:,10:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(pca_train15.iloc[:,:15], pca_train15.iloc[:,15:],

                                                    test_size=0.3, random_state=0)
X_train.head(5)
y_train.head(5)
model = LogisticRegression(multi_class='auto')
model.fit(X_train, y_train)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
y_pred = model.predict(pca_test15.iloc[:,:15])
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = pca_test15.iloc[:,15:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,13], train.iloc[:,31:], 

                                                    test_size=0.3, random_state=0)
X_train = pd.DataFrame(X_train,columns=['ram'])

X_train.head(5)
X_test = pd.DataFrame(X_test,columns=['ram'])
y_train = pd.DataFrame(y_train,columns=['price_range'])

y_train.head(5)
y_test = pd.DataFrame(y_test,columns=['price_range'])
y_test = y_test.reset_index(drop=True)

y_test.head(5)
gnb = GaussianNB()

gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,13]

x_train = pd.DataFrame(x_train,columns=['ram'])

x_train.sample(5)
y_pred = gnb.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:20], train.iloc[:,31:], test_size=0.3, random_state=0)
X_train.head(5)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
gnb = GaussianNB()

gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:20]

x_train.sample(5)
y_pred = gnb.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:31], train.iloc[:,31:], test_size=0.3, random_state=0)
X_train.head(5)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
gnb = GaussianNB()

gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:31]

x_train.sample(5)
y_pred = gnb.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:20], train.iloc[:,31:], test_size=0.3, random_state=0)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:20]

x_train.sample(5)
y_pred = knn.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:20], train.iloc[:,31:], test_size=0.3, random_state=0)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:20]

x_train.sample(5)
y_pred = knn.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:31], train.iloc[:,31:], test_size=0.3, random_state=0)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:31]

x_train.sample(5)
y_pred = knn.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:31], train.iloc[:,31:], test_size=0.3, random_state=0)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
knn = KNeighborsClassifier(n_neighbors=10, metric='euclidean')

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:31]

x_train.sample(5)
y_pred = knn.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:20], train.iloc[:,31:], test_size=0.3, random_state=0)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
svm_linear = SVC(kernel='linear',cache_size = 15)

svm_linear.fit(X_train, y_train)
y_pred = svm_linear.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:20]

x_train.sample(5)
y_pred = svm_linear.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:20], train.iloc[:,31:], test_size=0.3, random_state=0)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
svm_polynomial = SVC(kernel='poly',cache_size = 15)

svm_polynomial.fit(X_train, y_train)
y_pred = svm_polynomial.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:20]

x_train.sample(5)
y_pred = svm_polynomial.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:20], train.iloc[:,31:], test_size=0.3, random_state=0)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
DT = tree.DecisionTreeClassifier(min_samples_leaf= 10)

DT.fit(X_train, y_train)
# %matplotlib inline

# plt.figure(figsize=(50,25))

# dtree = dtreeplt(model=DT,

#     feature_names=train.iloc[:,:20].columns,

#     target_names=['0','1','2','3'], X=train.iloc[:,:20], y=train.iloc[:,31:], eval=True)

# fig = dtree.view()

# fig.savefig('dtree_20feature_leaf_size_10.pdf')  
y_pred = DT.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)

common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:20]

x_train.sample(5)
y_pred = DT.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:20], train.iloc[:,31:], test_size=0.3, random_state=0)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
DT = tree.DecisionTreeClassifier()

DT.fit(X_train, y_train)
# tree.plot_tree(DT, filled=True) 
# %matplotlib inline

# import matplotlib.pyplot as plt



# fig = DT.fit(X_train, y_train)

# tree.plot_tree(fig,filled=True)

# plt.figure(figsize=(50,1))

# plt.show()
# %matplotlib inline

# plt.figure(figsize=(50,25))

# dtree = dtreeplt(model=DT,

#     feature_names=train.iloc[:,:20].columns,

#     target_names=['0','1','2','3'], X=train.iloc[:,:20], y=train.iloc[:,31:], eval=True)

# fig = dtree.view()

# fig.savefig('dtree_20feature_gini.pdf')  
y_pred = DT.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:20]

x_train.sample(5)
y_pred = DT.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:20], train.iloc[:,31:], test_size=0.3, random_state=0)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
DT = tree.DecisionTreeClassifier(criterion = 'entropy')

DT.fit(X_train, y_train)
# %matplotlib inline

# plt.figure(figsize=(50,25))

# dtree = dtreeplt(model=DT,

#     feature_names=train.iloc[:,:20].columns,

#     target_names=['0','1','2','3'], X=train.iloc[:,:20], y=train.iloc[:,24:], eval=True)

# fig = dtree.view()

# fig.savefig('dtree_20feature_entropy.pdf')  
y_pred = DT.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:20]

x_train.sample(5)
y_pred = DT.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:31], train.iloc[:,31:], test_size=0.3, random_state=0)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
DT = tree.DecisionTreeClassifier()

DT.fit(X_train, y_train)
# %matplotlib inline

# plt.figure(figsize=(50,25))

# dtree = dtreeplt(model=DT,

#     feature_names=train.iloc[:,:31].columns,

#     target_names=['0','1','2','3'], X=train.iloc[:,:31], y=train.iloc[:,31:], eval=True)

# fig = dtree.view()

# fig.savefig('dtree_31feature.pdf')  
y_pred = DT.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:31]

x_train.sample(5)
y_pred = DT.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:20], train.iloc[:,31:], test_size=0.3, random_state=0)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred.describe()
for i in range(len(y_pred)):

    if 0 <= y_pred['y_pred'][i]<=0.5:

        y_pred['y_pred'][i] = 0

    elif 0.5 <= y_pred['y_pred'][i] <= 1.5:

        y_pred['y_pred'][i] = 1

    elif 1.5 <= y_pred['y_pred'][i] <= 2.5:

        y_pred['y_pred'][i] = 2

    elif 2.5 <= y_pred['y_pred'][i]:

        y_pred['y_pred'][i] = 3
y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:20]

x_train.sample(5)
y_pred = rf.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
for i in range(len(y_pred)):

    if 0 <= y_pred['y_pred'][i]<=0.5:

        y_pred['y_pred'][i] = 0

    elif 0.5 <= y_pred['y_pred'][i] <= 1.5:

        y_pred['y_pred'][i] = 1

    elif 1.5 <= y_pred['y_pred'][i] <= 2.5:

        y_pred['y_pred'][i] = 2

    elif 2.5 <= y_pred['y_pred'][i]:

        y_pred['y_pred'][i] = 3
y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:31], train.iloc[:,31:], test_size=0.3, random_state=0)
y_test = y_test.reset_index(drop=True)

y_test.head(5)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
for i in range(len(y_pred)):

    if 0 <= y_pred['y_pred'][i]<=0.5:

        y_pred['y_pred'][i] = 0

    elif 0.5 <= y_pred['y_pred'][i] <= 1.5:

        y_pred['y_pred'][i] = 1

    elif 1.5 <= y_pred['y_pred'][i] <= 2.5:

        y_pred['y_pred'][i] = 2

    elif 2.5 <= y_pred['y_pred'][i]:

        y_pred['y_pred'][i] = 3
y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:31]

x_train.sample(5)
y_pred = rf.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
for i in range(len(y_pred)):

    if 0 <= y_pred['y_pred'][i]<=0.5:

        y_pred['y_pred'][i] = 0

    elif 0.5 <= y_pred['y_pred'][i] <= 1.5:

        y_pred['y_pred'][i] = 1

    elif 1.5 <= y_pred['y_pred'][i] <= 2.5:

        y_pred['y_pred'][i] = 2

    elif 2.5 <= y_pred['y_pred'][i]:

        y_pred['y_pred'][i] = 3
y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:20], train.iloc[:,31:], test_size=0.3, random_state=0)
X_train.sample(5)
y_train.sample(5)
model = Sequential()

model.add(Dense(20, activation='linear', kernel_initializer='random_normal', input_dim=20))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(1, activation='linear', kernel_initializer='random_normal'))
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
print(model.summary())
#plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

#Image(retina=True, filename='model.png')



plot_model(model, to_file='NN1.png')
history = model.fit(X_train, y_train, epochs=2000, batch_size=10)
# plt.plot(history.history['acc'])

# #plt.plot(history.history['val_acc'])

# plt.title('Model accuracy')

# plt.ylabel('Accuracy')

# plt.xlabel('Epoch')

# plt.legend(['Train', 'Test'], loc='upper left')

# plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:20]

x_train.sample(5)
y_pred = model.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
train.head(5)
train['class0'] = 0

train['class1'] = 0

train['class2'] = 0

train['class3'] = 0

test['class0'] = 0

test['class1'] = 0

test['class2'] = 0

test['class3'] = 0
train.head(5)
train.loc[train['price_range'] == 0, 'class0'] = 1

train.loc[train['price_range'] == 1, 'class1'] = 1

train.loc[train['price_range'] == 2, 'class2'] = 1

train.loc[train['price_range'] == 3, 'class3'] = 1
train.head(5)
test.loc[test['price_range'] == 0, 'class0'] = 1

test.loc[test['price_range'] == 1, 'class1'] = 1

test.loc[test['price_range'] == 2, 'class2'] = 1

test.loc[test['price_range'] == 3, 'class3'] = 1
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:20], train.iloc[:,32:],

                                                    test_size=0.3, random_state=0)
X_train.head(5)
y_train.head(5)
y_test.head(5)
model = Sequential()

model.add(Dense(20, activation='sigmoid', kernel_initializer='random_normal', input_dim=20))

model.add(Dense(20, activation='relu', kernel_initializer='random_normal'))

model.add(Dense(20, activation='relu', kernel_initializer='random_normal'))

model.add(Dense(4, activation='softmax', kernel_initializer='random_normal'))
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='NN2.png')
history = model.fit(X_train, y_train, epochs=5000, batch_size=15)
plt.plot(history.history['accuracy'])

#plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
y_pred = model.predict(X_test)

y_pred
y_pred = pd.DataFrame(y_pred,columns=['0','1','2','3'])

y_pred.head(5)
y_pred['y_pred'] = 0
y_pred['max_value'] = y_pred.max(axis=1)
y_pred.head(5)
y_pred.columns
for i in range(len(y_pred)):

    for j in range(0,4):

        if y_pred[str(j)][i] == y_pred['max_value'][i]:

            y_pred['y_pred'][i] = j           
y_pred.head(5)
y_pred = y_pred[['y_pred']]
y_test['price_range'] = 0
y_test = y_test.reset_index(drop=True)
y_test.head(5)
for i in range(len(y_test)):

    if y_test['class0'][i] == 1:

        y_test['price_range'][i] = 0

    elif y_test['class1'][i] == 1:

        y_test['price_range'][i] = 1

    elif y_test['class2'][i] == 1:

        y_test['price_range'][i] = 2

    elif y_test['class3'][i] == 1:

        y_test['price_range'][i] = 3
y_test = y_test[["price_range"]]
y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
test.head(5)
X_test = test.iloc[:,:20]

X_test.sample(5)
y_pred = model.predict(X_test)

y_pred
y_pred = pd.DataFrame(y_pred,columns=['0','1','2','3'])

y_pred.head(5)
y_pred['y_pred'] = 0
y_pred['max_value'] = y_pred.max(axis=1)
y_pred.head(5)
for i in range(len(y_pred)):

    for j in range(0,4):

        if y_pred[str(j)][i] == y_pred['max_value'][i]:

            y_pred['y_pred'][i] = j   
y_pred.head(5)
y_pred = y_pred[['y_pred']]
y_test = test.iloc[:,32:]
y_test['price_range'] = 0
y_test.head(5)
y_test = y_test.reset_index(drop=True)
for i in range(len(y_test)):

    if y_test['class0'][i] == 1:

        y_test['price_range'][i] = 0

    elif y_test['class1'][i] == 1:

        y_test['price_range'][i] = 1

    elif y_test['class2'][i] == 1:

        y_test['price_range'][i] = 2

    elif y_test['class3'][i] == 1:

        y_test['price_range'][i] = 3
y_test = y_test[["price_range"]]
y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:20], train.iloc[:,31:],

                                                    test_size=0.3, random_state=0)
X_train.sample(5)
y_train.sample(5)
model = Sequential()

model.add(Dense(20, activation='linear', kernel_initializer='random_normal', input_dim=20))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(1, activation='linear', kernel_initializer='random_normal'))
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='NN3.png')
history = model.fit(X_train, y_train, epochs=500, batch_size=10)
plt.plot(history.history['acc'])

#plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:20]

x_train.sample(5)
y_pred = model.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:20], train.iloc[:,31:],

                                                    test_size=0.3, random_state=0)
X_train.sample(5)
y_train.sample(5)
model = Sequential()

model.add(Dense(20, activation='linear', kernel_initializer='random_normal', input_dim=20))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(1, activation='linear', kernel_initializer='random_normal'))
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='NN4.png')
history = model.fit(X_train, y_train, epochs=500, batch_size=10)
plt.plot(history.history['acc'])

#plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:20]

x_train.sample(5)
y_pred = model.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(pca_train10.iloc[:,:10], pca_train10.iloc[:,10:],

                                                    test_size=0.3, random_state=0)
X_train.sample(5)
y_train.sample(5)
model = Sequential()

model.add(Dense(10, activation='linear', kernel_initializer='random_normal', input_dim=10))

model.add(Dense(10, activation='relu', kernel_initializer='random_normal'))

model.add(Dense(10, activation='relu', kernel_initializer='random_normal'))

model.add(Dense(10, activation='relu', kernel_initializer='random_normal'))

model.add(Dense(10, activation='relu', kernel_initializer='random_normal'))

model.add(Dense(10, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(10, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(10, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(10, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(10, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(1, activation='linear', kernel_initializer='random_normal'))
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='NN5.png')
history = model.fit(X_train, y_train, epochs=5000, batch_size=15)
plt.plot(history.history['acc'])

#plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
y_pred = model.predict(pca_test10.iloc[:,:10])
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
y_test = pca_test10.iloc[:,10:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:20], train.iloc[:,31:],

                                                    test_size=0.3, random_state=0)
X_train.sample(5)
y_train.sample(5)
model = Sequential()

model.add(Dense(20, activation='linear', kernel_initializer='random_normal', input_dim=20))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(1, activation='linear', kernel_initializer='random_normal'))
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='NN6.png')
history = model.fit(X_train, y_train, epochs=500, batch_size=10)
plt.plot(history.history['acc'])

#plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:20]

x_train.sample(5)
y_pred = model.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:31], train.iloc[:,31:],

                                                    test_size=0.3, random_state=0)
X_train.sample(5)
y_train.sample(5)
model = Sequential()

model.add(Dense(31, activation='linear', kernel_initializer='random_normal', input_dim=31))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(1, activation='linear', kernel_initializer='random_normal'))
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='NN7.png')
history = model.fit(X_train, y_train, epochs=500, batch_size=10)
plt.plot(history.history['acc'])

#plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:31]

x_train.sample(5)
y_pred = model.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:31], train.iloc[:,31:],

                                                    test_size=0.3, random_state=0)
X_train.sample(5)
model = Sequential()

model.add(Dense(31, activation='linear', kernel_initializer='random_normal', input_dim=31))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(1, activation='linear', kernel_initializer='random_normal'))
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='NN8.png')
history = model.fit(X_train, y_train, epochs=500, batch_size=10)
plt.plot(history.history['acc'])

#plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:31]

x_train.sample(5)
y_pred = model.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:31], train.iloc[:,31:],

                                                    test_size=0.3, random_state=0)
X_train.sample(5)
model = Sequential()

model.add(Dense(31, activation='linear', kernel_initializer='random_normal', input_dim=31))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(20, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(1, activation='linear', kernel_initializer='random_normal'))
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='NN9.png')
history = model.fit(X_train, y_train, epochs=500, batch_size=10)
plt.plot(history.history['acc'])

#plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:31]

x_train.sample(5)
y_pred = model.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:,:31], train.iloc[:,31:],

                                                    test_size=0.3, random_state=0)
X_train.sample(5)
model = Sequential()

model.add(Dense(31, activation='linear', kernel_initializer='random_normal', input_dim=31))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(31, activation='linear', kernel_initializer='random_normal'))

model.add(Dense(1, activation='linear', kernel_initializer='random_normal'))
model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='NN10.png')
history = model.fit(X_train, y_train, epochs=1000, batch_size=10)
plt.plot(history.history['acc'])

#plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
# Plot training & validation loss values

plt.plot(history.history['loss'])

#plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
x_train = test.iloc[:,:31]

x_train.sample(5)
y_pred = model.predict(x_train)
y_pred = pd.DataFrame(y_pred,columns=['y_pred'])

y_pred.head(5)
y_pred = y_pred.round(0)

y_pred.head(5)
y_test = test.iloc[:,31:]

y_test = y_test.reset_index(drop=True)

y_test.head(5)
common = y_test.merge(y_pred, left_index=True, right_index=True)

common.sample(5)
mat = confusion_matrix(y_test,y_pred)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

plt.xlabel('true label')

plt.ylabel('predicted label')
print(classification_report(y_test, y_pred))
count_misclassified = (common.price_range != common.y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))