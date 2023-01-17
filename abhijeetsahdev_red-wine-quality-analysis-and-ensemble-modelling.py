%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder

outlier_circle = dict(markerfacecolor='#081d58', marker='.')
meanlineprops = dict(linestyle='--', linewidth=2, color='#7fcdbb')
medianprops = dict(linestyle='dotted', linewidth=2, color='#1d91c0')
boxprops=dict(facecolor='#ffffd9', color='#ffffd9')
required_df = pd.read_csv("../input/red-wine-quality/winequality-red.csv")
required_df.head()

z = np.abs(stats.zscore(required_df))

required_df.dropna()

fixed_acidity = required_df["fixed acidity"];
x_label = "Fixed Acidity in  g / dm^3"
y_label = "Counts"
plt.figure(figsize=(10,10),facecolor="w")
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2,sharex = ax1)
ax1.boxplot(fixed_acidity,vert = False,notch = True, meanline = True,showmeans = True, flierprops=outlier_circle,whis=0.5,meanprops = meanlineprops, 
            medianprops =medianprops, boxprops = boxprops, patch_artist = True )
ax2.hist(fixed_acidity,bins = range(4,16),facecolor = '#ffffd9',edgecolor = "gray")
ax2.set_ylabel(y_label, fontsize = 15)
ax2.set_xlabel(x_label, fontsize = 15)
ax2.yaxis.set_label_coords(-0.1,0.5)
plt.setp(ax1.get_xticklabels(),visible=False)

print("Consider Fixed Acidity,")
print("\t\t Minimum Value :",fixed_acidity.min())
print("\t\t Maximum Value :",fixed_acidity.max())
print("\t\t Mean          :",round(fixed_acidity.mean(),2))
print("\t\t Median        :",fixed_acidity.median())
print("\t\t Std. Deviation:",round(fixed_acidity.std(),2))

residual_sugar = required_df["residual sugar"];
x_label = "Residual Sugar in  g / dm^3"
y_label = "Count"

plt.figure(figsize=(10,10),facecolor="w")
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2,sharex = ax1)
ax1.boxplot(residual_sugar,vert = False,meanline = True,boxprops = boxprops, patch_artist = True,showmeans = True, flierprops=outlier_circle,meanprops = meanlineprops, medianprops =medianprops, widths = 1)
ax2.hist(residual_sugar,bins = range(0,16),facecolor = '#ffffd9',edgecolor = "gray")
ax2.set_ylabel(y_label, fontsize = 15)
ax2.set_xlabel(x_label, fontsize = 15)
ax2.yaxis.set_label_coords(-0.1,0.5)
plt.setp(ax1.get_xticklabels(),visible=False)

print("Consider Residual Sugar,")
print("\t\t Minimum Value :",residual_sugar.min())
print("\t\t Maximum Value :",residual_sugar.max())
print("\t\t Mean          :",round(residual_sugar.mean(),2))
print("\t\t Median        :",residual_sugar.median())
print("\t\t Std. Deviation:",round(residual_sugar.std(),2))

chlorides = required_df["chlorides"]
x_label = "Chlorides in  g / dm^3"
y_label = "Count"

plt.figure(figsize=(10,10),facecolor="w")
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2,sharex = ax1)
ax1.boxplot(chlorides,vert = False,meanline = True,boxprops = boxprops, patch_artist = True,showmeans = True, flierprops=outlier_circle,meanprops = meanlineprops, medianprops =medianprops, widths = 1)
ax2.hist(chlorides,bins = 7,facecolor = '#ffffd9',edgecolor = "gray")
ax2.set_ylabel(y_label, fontsize = 15)
ax2.set_xlabel(x_label, fontsize = 15)
ax2.yaxis.set_label_coords(-0.1,0.5)
plt.setp(ax1.get_xticklabels(),visible=False)
print("Consider Chlorides,")
print("\t\t Minimum Value :",chlorides.min())
print("\t\t Maximum Value :",chlorides.max())
print("\t\t Mean          :",round(chlorides.mean(),2))
print("\t\t Median        :",chlorides.median())
print("\t\t Std. Deviation:",round(chlorides.std(),2))

citric_acid = required_df["citric acid"];
x_label = "Citric Acid in  g / dm^3"
y_label = "Count"
plt.figure(figsize=(10,10),facecolor="w")
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2,sharex = ax1)
ax1.boxplot(citric_acid,vert = False,notch = True,boxprops = boxprops, patch_artist = True ,meanline = True,showmeans = True, flierprops=outlier_circle,whis=0.5,meanprops = meanlineprops, 
            medianprops =medianprops)
ax2.hist(citric_acid,facecolor = '#ffffd9',edgecolor = "gray")
ax2.set_ylabel(y_label, fontsize = 15)
ax2.set_xlabel(x_label, fontsize = 15)
ax2.yaxis.set_label_coords(-0.1,0.5)
plt.setp(ax1.get_xticklabels(),visible=False)
print("Consider Citric Acid,")
print("\t\t Minimum Value :",citric_acid.min())
print("\t\t Maximum Value :",citric_acid.max())
print("\t\t Mean          :",round(citric_acid.mean(),2))
print("\t\t Median        :",citric_acid.median())
print("\t\t Std. Deviation:",round(citric_acid.std(),2))

quality = required_df["quality"]
x_label = "Quality (1-10)"
y_label = "Count"

fig = sns.countplot(x = quality,data= required_df)
fig.set(xlabel=x_label,ylabel=y_label)
print("Consider Quality,")
print("\t\t Minimum Value :",quality.min())
print("\t\t Maximum Value :",quality.max())
print("\t\t Mean          :",round(quality.mean(),2))
print("\t\t Median        :",quality.median())
print("\t\t Std. Deviation:",round(quality.std(),2))

alcohol = required_df["alcohol"];
x_label = "Alcohol"
y_label = "Count"
plt.figure(figsize=(10,10),facecolor="w")
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2,sharex = ax1)
ax1.boxplot(alcohol,vert = False,notch = True,boxprops = boxprops, patch_artist = True ,meanline = True,showmeans = True, flierprops=outlier_circle,whis=0.5,meanprops = meanlineprops, 
            medianprops =medianprops)
ax2.hist(alcohol,facecolor = '#ffffd9',edgecolor = "gray", range =(8,15))
ax2.set_ylabel(y_label, fontsize = 15)
ax2.set_xlabel(x_label, fontsize = 15)
ax2.yaxis.set_label_coords(-0.1,0.5)
plt.setp(ax1.get_xticklabels(),visible=False)
print("Consider Alcohol,")
print("\t\t Minimum Value :",alcohol.min())
print("\t\t Maximum Value :",alcohol.max())
print("\t\t Mean          :",round(alcohol.mean(),2))
print("\t\t Median        :",alcohol.median())
print("\t\t Std. Deviation:",round(alcohol.std(),2))

free_so2 = required_df["free sulfur dioxide"];
x_label = "Free Sulfur Dioxide in  g / dm^3"
y_label = "Count"
plt.figure(figsize=(10,10),facecolor="w")
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2,sharex = ax1)
ax1.boxplot(free_so2,vert = False,notch = True,boxprops = boxprops, patch_artist = True ,meanline = True,showmeans = True, flierprops=outlier_circle,whis=0.5,meanprops = meanlineprops, 
            medianprops =medianprops)
ax2.hist(free_so2,facecolor = '#ffffd9',edgecolor = "gray", bins = 20)
ax2.set_ylabel(y_label, fontsize = 15)
ax2.set_xlabel(x_label, fontsize = 15)
ax2.yaxis.set_label_coords(-0.1,0.5)
plt.setp(ax1.get_xticklabels(),visible=False)
print("Consider Free Sulfur Dioxide,")
print("\t\t Minimum Value :",free_so2.min())
print("\t\t Maximum Value :",free_so2.max())
print("\t\t Mean          :",round(free_so2.mean(),2))
print("\t\t Median        :",free_so2.median())
print("\t\t Std. Deviation:",round(free_so2.std(),2))

total_so2 = required_df["total sulfur dioxide"];
x_label = "Total Sulfur Dioxide in  g / dm^3"
y_label = "Count"
plt.figure(figsize=(10,10),facecolor="w")
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2,sharex = ax1)
ax1.boxplot(total_so2,vert = False,notch = True,boxprops = boxprops, patch_artist = True ,meanline = True,showmeans = True, flierprops=outlier_circle,whis=0.5,meanprops = meanlineprops, 
            medianprops =medianprops, )
ax2.hist(total_so2,facecolor = '#ffffd9',edgecolor = "gray", bins = 20)
ax2.set_ylabel(y_label, fontsize = 15)
ax2.set_xlabel(x_label, fontsize = 15)
ax2.yaxis.set_label_coords(-0.1,0.5)
plt.setp(ax1.get_xticklabels(),visible=False)
print("Consider Total Sulfur Dioxide,")
print("\t\t Minimum Value :",total_so2.min())
print("\t\t Maximum Value :",total_so2.max())
print("\t\t Mean          :",round(total_so2.mean(),2))
print("\t\t Median        :",total_so2.median())
print("\t\t Std. Deviation:",round(total_so2.std(),2))

density = required_df["density"];
x_label = "Density in  g / cm^3"
y_label = "Count"
plt.figure(figsize=(10,10),facecolor="w")
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2,sharex = ax1)
ax1.boxplot(density,vert = False,notch = True,boxprops = boxprops, patch_artist = True ,meanline = True,showmeans = True, flierprops=outlier_circle,whis=0.5,meanprops = meanlineprops, 
            medianprops =medianprops, )
ax2.hist(density,facecolor = '#ffffd9',edgecolor = "gray",bins = 15)
ax2.set_ylabel(y_label, fontsize = 15)
ax2.set_xlabel(x_label, fontsize = 15)
ax1.yaxis.set_label_coords(-0.1,0.5)
ax2.yaxis.set_label_coords(-0.1,0.5)
plt.setp(ax1.get_xticklabels(),visible=False)
print("Consider Density,")
print("\t\t Minimum Value :",density.min())
print("\t\t Maximum Value :",density.max())
print("\t\t Mean          :",round(density.mean(),2))
print("\t\t Median        :",density.median())
print("\t\t Std. Deviation:",round(density.std(),2))

ph = required_df["pH"];
x_label = "pH "
y_label = "Count"
plt.figure(figsize=(10,10),facecolor="w")
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2,sharex = ax1)
ax1.boxplot(ph,vert = False,notch = True,boxprops = boxprops, patch_artist = True ,meanline = True,showmeans = True, flierprops=outlier_circle,whis=0.5,meanprops = meanlineprops, 
            medianprops =medianprops, )
ax2.hist(ph,facecolor = '#ffffd9',edgecolor = "gray")
ax2.set_ylabel(y_label, fontsize = 15)
ax2.set_xlabel(x_label, fontsize = 15)
ax2.yaxis.set_label_coords(-0.1,0.5)
plt.setp(ax1.get_xticklabels(),visible=False)
print("Consider pH,")
print("\t\t Minimum Value :",ph.min())
print("\t\t Maximum Value :",ph.max())
print("\t\t Mean          :",round(ph.mean(),2))
print("\t\t Median        :",ph.median())
print("\t\t Std. Deviation:",round(ph.std(),2))

sulphates = required_df["sulphates"];
x_label = "Sulphates in g/dm^3"
y_label = "Count"
plt.figure(figsize=(10,10),facecolor="w")
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2,sharex = ax1)
ax1.boxplot(sulphates,vert = False,notch = True,boxprops = boxprops, patch_artist = True ,meanline = True,showmeans = True, flierprops=outlier_circle,whis=0.5,meanprops = meanlineprops, 
            medianprops =medianprops, )
ax2.hist(sulphates,facecolor = '#ffffd9',edgecolor = "gray", bins = 20)
ax2.set_ylabel(y_label, fontsize = 15)
ax2.set_xlabel(x_label, fontsize = 15)
ax2.yaxis.set_label_coords(-0.1,0.5)
plt.setp(ax1.get_xticklabels(),visible=False)
print("Consider Sulphates,")
print("\t\t Minimum Value :",sulphates.min())
print("\t\t Maximum Value :",sulphates.max())
print("\t\t Mean          :",round(sulphates.mean(),2))
print("\t\t Median        :",sulphates.median())
print("\t\t Std. Deviation:",round(sulphates.std(),2))

volatile_acidity = required_df["volatile acidity"];
x_label = "Volatile Acidity in g/dm^3 "
y_label = "Count"
plt.figure(figsize=(10,10),facecolor="w")
ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2,sharex = ax1)
ax1.boxplot(volatile_acidity,vert = False,notch = True,boxprops = boxprops, patch_artist = True ,meanline = True,showmeans = True, flierprops=outlier_circle,whis=0.5,meanprops = meanlineprops, 
            medianprops =medianprops, )
ax2.hist(volatile_acidity,facecolor = '#ffffd9',edgecolor = "gray", bins = 20)
ax2.set_ylabel(y_label, fontsize = 15)
ax2.set_xlabel(x_label, fontsize = 15)
ax2.yaxis.set_label_coords(-0.1,0.5)
plt.setp(ax1.get_xticklabels(),visible=False)
print("Consider Volatile Acidity in g/dm^3,")
print("\t\t Minimum Value :",volatile_acidity.min())
print("\t\t Maximum Value :",volatile_acidity.max())
print("\t\t Mean          :",round(volatile_acidity.mean(),2))
print("\t\t Median        :",volatile_acidity.median())
print("\t\t Std. Deviation:",round(volatile_acidity.std(),2))

correlation = required_df.corr()
mask = np.zeros_like(correlation)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(10, 7))
    ax = sns.heatmap(correlation, mask=mask,linewidths = 0.5, annot = True, cmap="YlGnBu")

plt.scatter(ph,fixed_acidity, alpha = 0.5)
plt.xlabel('pH', fontsize = 15)
plt.ylabel('Fixed Acidity',fontsize = 15 )

plt.scatter(density,fixed_acidity,alpha = 0.5)
plt.xlabel('Density', fontsize = 15)
plt.ylabel('Fixed Acidity',fontsize = 15 )

plt.scatter(residual_sugar,chlorides, alpha = 0.5)
plt.xlabel('Residual Sugar', fontsize = 15)
plt.ylabel('Chlorides',fontsize = 15)

plt.scatter(citric_acid,chlorides, alpha = 0.5)
plt.xlabel('Citric Acid', fontsize = 15)
plt.ylabel('Chlorides',fontsize = 15 )

plt.scatter(density,alcohol, alpha = 0.5)
plt.xlabel('Desnsity', fontsize = 15)
plt.ylabel('Fixed Acidity',fontsize = 15 )

correlation["quality"].sort_values(ascending = False)
required_df = required_df[(z < 3).all(axis=1)]
required_df.shape

bins = (2,6.5,8)
labels = ['bad','good']
required_df['quality'] = pd.cut(required_df['quality'],bins=bins,labels=labels)
le = LabelEncoder()
required_df['quality'] = le.fit_transform(required_df['quality'])
required_df['quality']
X = np.asarray(required_df.loc[:,["alcohol","pH","citric acid","volatile acidity"]])
X
y = np.asarray(required_df["quality"])
y
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify = y)
print ('Train set:', X_train.shape, y_train.shape)
print ('Test set:', X_test.shape, y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
k_range = range(1, 26)
k_scores = []
# Calculate cross validation score for every k number from 1 to 26
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy') 
    k_scores.append(scores.mean())
# Plot accuracy for every k number between 1 and 26
print(k_scores)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report

knn_confusionmatrix = confusion_matrix(knn_pred,y_test)
ax = sns.heatmap(knn_confusionmatrix,annot=True,cmap="YlGnBu",)
ax.set(xlabel='Predict', ylabel='true')
knn_accuracyscore = accuracy_score(knn_pred,y_test)
print("KNearest neighbors  with k  = 20, accuracy score: ",knn_accuracyscore,)
print(classification_report(y_test,knn_pred))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
print ('Train set:', X_train.shape, y_train.shape)
print ('Test set:', X_test.shape, y_test.shape)
from sklearn.linear_model import LogisticRegression

l = LogisticRegression()
log_reg = l.fit(X_train,y_train)
logreg_pred = log_reg.predict(X_test)
logreg_accuracyscore = accuracy_score(logreg_pred,y_test)
print("Logistic Regression, accuracy score = ",logreg_accuracyscore)
print(classification_report(y_test,logreg_pred))