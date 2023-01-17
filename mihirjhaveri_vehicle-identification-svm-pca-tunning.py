# importing the necessary initial libraries.



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
# import the data



vechicle_raw_data = pd.read_csv("../input/vechile-dataset/vehicle.csv")



vechicle_raw_data.shape
vechicle_raw_data.head(10)
vechicle_raw_data.info()
vechicle_raw_data.describe().T
dups = vechicle_raw_data.duplicated()

print('Number of duplicate rows = %d' % (dups.sum()))
### Visuvalizing Boxplot of each columns

plt.figure(figsize=(30,6))



#Subplot 1- Boxplot

plt.subplot(1,3,1)

plt.title('compactness')

sns.boxplot(vechicle_raw_data['compactness'],orient='horizondal',color='red')



#Subplot 2

plt.subplot(1,3,2)

plt.title('circularity')

sns.boxplot(vechicle_raw_data['circularity'],orient='horizondal',color='blue')



#Subplot 3

plt.subplot(1,3,3)

plt.title('distance_circularity')

sns.boxplot(vechicle_raw_data['distance_circularity'],orient='horizondal',color='green')



plt.figure(figsize=(30,6))



#Subplot 1

plt.subplot(1,3,1)

plt.title('radius_ratio')

sns.boxplot(vechicle_raw_data['radius_ratio'],orient='horizondal',color='purple')



#Subplot 2

plt.subplot(1,3,2)

plt.title('pr.axis_aspect_ratio')

sns.boxplot(vechicle_raw_data['pr.axis_aspect_ratio'],orient='horizondal',color='black')

  



#Subplot 3

plt.subplot(1,3,3)

plt.title('max.length_aspect_ratio')

sns.boxplot(vechicle_raw_data['max.length_aspect_ratio'],orient='horizondal',color='yellow')
plt.figure(figsize=(30,6))



#Subplot 1- Boxplot

plt.subplot(1,3,1)

plt.title('scatter_ratio')

sns.boxplot(vechicle_raw_data['scatter_ratio'],orient='horizondal',color='red')



#Subplot 2

plt.subplot(1,3,2)

plt.title('elongatedness')

sns.boxplot(vechicle_raw_data['elongatedness'],orient='horizondal',color='blue')



#Subplot 3

plt.subplot(1,3,3)

plt.title('pr.axis_rectangularity')

sns.boxplot(vechicle_raw_data['pr.axis_rectangularity'],orient='horizondal',color='green')



plt.figure(figsize=(30,6))



#Subplot 1

plt.subplot(1,3,1)

plt.title('max.length_rectangularity')

sns.boxplot(vechicle_raw_data['max.length_rectangularity'],orient='horizondal',color='purple')



#Subplot 2

plt.subplot(1,3,2)

plt.title('scaled_variance')

sns.boxplot(vechicle_raw_data['scaled_variance'],orient='horizondal',color='yellow')

  



#Subplot 3

plt.subplot(1,3,3)

plt.title('scaled_variance.1')

sns.boxplot(vechicle_raw_data['scaled_variance.1'],orient='horizondal',color='black')
plt.figure(figsize=(30,6))



#Subplot 1- Boxplot

plt.subplot(1,3,1)

plt.title('scaled_radius_of_gyration')

sns.boxplot(vechicle_raw_data['scaled_radius_of_gyration'],orient='horizondal',color='red')



#Subplot 2

plt.subplot(1,3,2)

plt.title('scaled_radius_of_gyration.1')

sns.boxplot(vechicle_raw_data['scaled_radius_of_gyration.1'],orient='horizondal',color='blue')



#Subplot 3

plt.subplot(1,3,3)

plt.title('skewness_about')

sns.boxplot(vechicle_raw_data['skewness_about'],orient='horizondal',color='green')



plt.figure(figsize=(30,6))



#Subplot 1

plt.subplot(1,3,1)

plt.title('skewness_about.1')

sns.boxplot(vechicle_raw_data['skewness_about.1'],orient='horizondal',color='purple')



#Subplot 2

plt.subplot(1,3,2)

plt.title('skewness_about.2')

sns.boxplot(vechicle_raw_data['skewness_about.2'],orient='horizondal',color='yellow')

  



#Subplot 3

plt.subplot(1,3,3)

plt.title('hollows_ratio')

sns.boxplot(vechicle_raw_data['hollows_ratio'],orient='horizondal',color='black')
# Plot the central tendency of the dataset

_, bp = vechicle_raw_data.boxplot(return_type='both', figsize=(20,10), rot='vertical')



fliers = [flier.get_ydata() for flier in bp["fliers"]]

boxes = [box.get_ydata() for box in bp["boxes"]]

caps = [cap.get_ydata() for cap in bp['caps']]

whiskers = [whiskers.get_ydata() for whiskers in bp["whiskers"]]
# we define outliers by using Inter Quantile range. 

# Data_point > (Q3 * 1.5) is said to be outlier where Q3 is 75% Quantile !



# finding the IQR for each of the numerical columns

def check_outliers(data):

    vData_num = data.loc[:,data.columns != 'class']

    Q1 = vData_num.quantile(0.25)

    Q3 = vData_num.quantile(0.75)

    IQR = Q3 - Q1

    count = 0

    # checking for outliers, True represents outlier

    vData_num_mod = ((vData_num < (Q1 - 1.5 * IQR)) |(vData_num > (Q3 + 1.5 * IQR)))

    #iterating over columns to check for no.of outliers in each of the numerical attributes.

    for col in vData_num_mod:

        if(1 in vData_num_mod[col].value_counts().index):

            print("No. of outliers in %s: %d" %( col, vData_num_mod[col].value_counts().iloc[1]))

            count += 1

    print("\n\nNo of attributes with outliers are :", count)

    

check_outliers(vechicle_raw_data)
# creating a datacopy for cleaning the records 



vechicle_clean_data = vechicle_raw_data.copy()
# to replace with median we will loop through each column in the dataframe



for col in vechicle_clean_data.columns[:-1]:

    Q1 = vechicle_clean_data[col].quantile(0.25)

    Q3 = vechicle_clean_data[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_value = Q1 - (1.5 * IQR)

    upper_value = Q3 + (1.5 * IQR)

    

    vechicle_clean_data.loc[(vechicle_clean_data[col]< lower_value) | ( vechicle_clean_data[col] > upper_value), col] = vechicle_clean_data[col].median()



# check for outliers

check_outliers(vechicle_clean_data)
# Check the dataset after Outlier treatment

sns.set_style('darkgrid')

plt.figure(figsize=(30, 30))

index = 1

for col in vechicle_clean_data.columns[:-1]:

    plt.subplot(1, len(vechicle_clean_data.columns[:-1]), index)

    sns.boxplot(y=vechicle_clean_data[col], palette='inferno', fliersize=12)

    index += 1

plt.tight_layout()
print("Missing values if any (True/False)? :", vechicle_raw_data.isnull().values.any())

vechicle_raw_data.isna().apply(pd.value_counts).T
#lets visualize missing values in heatmap



sns.heatmap(vechicle_raw_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
vechicle_raw_data[vechicle_raw_data.isnull().any(axis=1)]
# creating a meanfiller to replace all column missing values with mean

meanFiller = lambda x: x.fillna(x.mean())



vData_num = vechicle_clean_data.loc[:,vechicle_clean_data.columns != 'class']

vData_cat = vechicle_clean_data.loc[:,vechicle_clean_data.columns == 'class']

vData_num = vData_num.apply(meanFiller,axis=0)



vechicle_clean_data = pd.concat([vData_num, vData_cat], axis = 1)

vechicle_clean_data.info()
#lets visualize missing values in heatmap



sns.heatmap(vechicle_clean_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
print("Types of class:", vechicle_clean_data['class'].unique())



print("\nValue Counts:\n",vechicle_clean_data['class'].value_counts())



sns.countplot(vechicle_clean_data['class'])
# we can vsualise target class and see that car class is dominating data set by 50%,shows imabalance

labels = ['car','bus','van']

size = vechicle_clean_data['class'].value_counts()

colors = ['blue', 'orange','green']

explode = [0.1, 0.1,0.1]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')

plt.title('Class distribution', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
vechicle_clean_data.groupby(["class"]).count() #lets group the classes and we can see car class dominates half of data set
# we will start with understanding the distrubtion of each attributes using "hist".

plt.style.use('seaborn-whitegrid')



vechicle_clean_data.hist(bins=20, figsize=(60,40), color='lightblue', edgecolor = 'red')

plt.show()
# understanding the distrubtion of each attributes using distplot.



fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(25, 25))

for i, column in enumerate(vechicle_clean_data.columns[:-1]):

    sns.distplot(vechicle_clean_data[column],ax=axes[i//3,i%3])
plt.figure(figsize= (15,15))

vechicle_clean_data.boxplot()

plt.xticks(rotation = 90)
skewValue = vechicle_clean_data.skew()

print("skewValue of dataframe attributes: ", skewValue)
sns.boxplot(x="class", y="compactness", palette="ch:r=-.5,l=.75", data=vechicle_clean_data); #box plot of target clasess
# for the purpose of readability let us set visiblity only for lower triangle.

g = sns.pairplot(vechicle_clean_data, hue='class')

for i, j in zip(*np.triu_indices_from(g.axes, 1)):

    g.axes[i, j].set_visible(False)
# Heatmap

#Correlation Matrix

corr = vechicle_clean_data.corr() # correlation matrix

lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix

mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap



plt.figure(figsize = (15,15))  # setting the figure size

sns.set_style(style = 'white')  # Setting it to white so that we do not see the grid lines

sns.heatmap(lower_triangle, annot= True,cmap='viridis', xticklabels = corr.index,yticklabels = corr.columns,linewidths= 1,mask = mask)   # Da Heatmap

plt.xticks(rotation = 90)   # Aesthetic purposes

plt.show()
#splitting dependant and independant attributes.



X = vechicle_clean_data.drop(['class'],axis = 1)

y = vechicle_clean_data[['class']]



print("shape of independant data: ", X.shape)

print("shape of dependant data: ", y.shape)

# encoding the class attribute.

y.replace({'car':0,'bus':1,'van':2},inplace=True)

# prior to scaling 

plt.plot(X)

plt.show()
# Scaling the attributes.



from scipy.stats import zscore

XScaled=X.apply(zscore)

XScaled.head()
#after scaling

plt.plot(XScaled)

plt.show()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(XScaled, y, test_size=0.30, random_state=3)



print("Shape of X train : ",X_train.shape)

print("Shape of X_test  : ",X_test.shape)



print("Shape of y_train : ",y_train.shape)

print("Shape of y_test  : ",y_test.shape)



# building an svm model

from sklearn import svm

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



svm_model_1 = svm.SVC(gamma=0.025, C=3, kernel= 'linear')

svm_model_1.fit(X_train , y_train)
# getting model accuracies.



y_predict_1 = svm_model_1.predict(X_test)



train_score_1 = svm_model_1.score(X_train,y_train)

test_score_1 = svm_model_1.score(X_test, y_test)



print("SVM_model_1 score for train set:", train_score_1*100)

print("SVM_model_1 score for test set:", test_score_1*100)



#Store the accuracy results for each model in a dataframe for final comparison

resultsDf = pd.DataFrame({'Method':['SVM (kernel: Linear)'], 'accuracy':test_score_1 })

resultsDf = resultsDf[['Method', 'accuracy']]



confusion_matrix_1 = confusion_matrix(y_test,y_predict_1)

print("\n\nConfusion Matrix:\n   car  bus van\n",confusion_matrix_1)

target_names = ['car', 'bus', 'van']

print("\nClassification Report:\n", classification_report(y_test, y_predict_1, target_names=target_names))

#metrics.classification_report(y_test, y_predict_1, target_names=target_names)

resultsDf
# lets try out with a different kernel

# we will use the same gamma and C value throught. We will consider hyper parameter tuning later.



svm_model_2 = svm.SVC(gamma=0.025, C=3, kernel= 'rbf')

svm_model_2.fit(X_train , y_train)
y_predict_2 = svm_model_2.predict(X_test)



train_score_2 = svm_model_2.score(X_train,y_train)

test_score_2 = svm_model_2.score(X_test, y_test)



print("SVM_model_2 score for train set:", train_score_2*100)

print("SVM_model_2 score for test set:", test_score_2*100)



tempResultsDf = pd.DataFrame({'Method':['SVM (kernel: rbf)'], 'accuracy':test_score_2 })

resultsDf = pd.concat([resultsDf, tempResultsDf])

resultsDf = resultsDf[['Method', 'accuracy']]



confusion_matrix_2 = confusion_matrix(y_test,y_predict_2)



print("\nConfusion Matrix:\n   car bus van\n",confusion_matrix_2)

target_names = ['car', 'bus', 'van']

print("Classification report:\n", classification_report(y_test, y_predict_2, target_names=target_names))

#metrics.classification_report(y_test, y_predict_1, target_names=target_names)





resultsDf
#Grid search to tune model parameters for SVC

from sklearn.model_selection import GridSearchCV



c_range = range(5,15)

gamma_range = [0.001,0.025,0.05,0.04,0.03,0.1,0.5,1,10]

params = dict(C=c_range, gamma=gamma_range,kernel=['linear', 'rbf'])

model = GridSearchCV(svm.SVC(), param_grid=params, verbose=1)

model.fit(X_train, y_train)

print("Best Hyper Parameters:\n", model.best_params_)
# Build svm model with C=5 and gamma=0.1 using rbf kernel



svm_model_3 = svm.SVC(gamma=0.1, C=5, kernel= 'rbf')

svm_model_3.fit(X_train , y_train)
y_predict_3 = svm_model_3.predict(X_test)



train_score_3 = svm_model_3.score(X_train,y_train)

test_score_3 = svm_model_3.score(X_test, y_test)



print("SVM_model_3 score for train set:", train_score_3*100)

print("SVM_model_3 score for test set:", test_score_3*100)



tempResultsDf = pd.DataFrame({'Method':['SVM - Tuned'], 'accuracy':test_score_3 })

resultsDf = pd.concat([resultsDf, tempResultsDf])

resultsDf = resultsDf[['Method', 'accuracy']]



confusion_matrix_3 = confusion_matrix(y_test,y_predict_3)



print("\nConfusion Matrix:\n   car bus van\n",confusion_matrix_3)

target_names = ['car', 'bus', 'van']

print("Classification report:\n", classification_report(y_test, y_predict_3, target_names=target_names))

#metrics.classification_report(y_test, y_predict_1, target_names=target_names)





resultsDf
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score





num_folds = 50

seed = 7



kfold = KFold(n_splits=num_folds, random_state=seed)

model = svm.SVC(gamma=0.05, C=10, kernel= 'rbf')

results = cross_val_score(model, XScaled, y, cv=kfold)

print(results)

print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))



CVScores_Df = pd.DataFrame({'Method':['SVM - Tuned'], 'accuracy':results.mean()*100.0, 'std(+/-)':results.std()*100.0})

CVScores_Df = CVScores_Df[['Method', 'accuracy', 'std(+/-)']]

CVScores_Df
sns.distplot(results,kde=True,bins=5)

plt.xlabel("Accuracy")

plt.show()













# confidence intervals

alpha = 0.95                             # for 95% confidence 



p = ((1.0-alpha)/2.0) * 100              

lower = max(0.0, np.percentile(results, p))  



p = (alpha+((1.0-alpha)/2.0)) * 100

upper = min(1.0, np.percentile(results, p))





print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
# we have deduced in section 2 that some attributes have very low correlation. 

# Hence we will not perform PCA for those attibutes



# So, we drop the columns 'pr.axis_aspect_ratio','max.length_aspect_ratio', 'skewness_about' and 'skewness_about.1'

# from PCA analysis.



XScaled_cp = XScaled.drop(['pr.axis_aspect_ratio','max.length_aspect_ratio', 'skewness_about', 'skewness_about.1'],axis=1)

XScaled_cp.shape
covMatrix = np.cov(XScaled_cp,rowvar=False)

print(covMatrix)
from sklearn.decomposition import PCA



pca = PCA(n_components=len(XScaled_cp.columns), whiten=False)

pca.fit(XScaled_cp)
print('Features Started with:', len(XScaled_cp.columns))

print()



print('Eigen Values: \n', pca.explained_variance_)

print()



print('Eigen Vector: \n', pca.components_)

print()



percent_variance = np.asarray([float(format(num, '.3f')) for num in pca.explained_variance_ratio_])

percent_variance = np.round(np.asarray(percent_variance) * 100, decimals =2)

print('Percentage variance explained:\n ', percent_variance)
plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

plt.bar(list(range(1,15)),percent_variance,alpha=0.5, align='center')

plt.ylabel('Variation explained')

plt.xlabel('eigen Value')



plt.subplot(1,2,2)

plt.step(list(range(1,15)),np.cumsum(percent_variance), where='mid')

plt.axhline(y=95, color='r', linestyle='-')

plt.ylabel('Cummulative variance explained')

plt.xlabel('eigen Value')

plt.show()



# 5 features explain 95 % of the variance

pca_rd = PCA(n_components=5)

pca_rd.fit(XScaled_cp)
print(pca_rd.components_)

print()



print(pca_rd.explained_variance_ratio_)

print()
#print(XScaled_cp.columns)

XScaled_cp_pca = pd.DataFrame(pca_rd.transform(XScaled_cp))



print("shape after dimenationality reduction:", XScaled_cp_pca.shape)



XScaled_cp_pca.head()
sns.pairplot(XScaled_cp_pca, diag_kind='kde') 
X_non_pca = XScaled[['pr.axis_aspect_ratio','max.length_aspect_ratio', 'skewness_about', 'skewness_about.1']]



XScaled_pca = pd.merge(XScaled_cp_pca, X_non_pca, right_index=True, left_index=True)



print("Final shape of the data: ", XScaled_pca.shape)



XScaled_pca.head()
# for data split we will use the same random state as above, to ensure the accuracy can be compared on same rows.

# so we give the same random state.



print("shape after pca:", XScaled_pca.shape)

X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(XScaled_pca, y, test_size=0.30, random_state=3)



print("Shape of X train : ",X_pca_train.shape)

print("Shape of X_test  : ",X_pca_test.shape)

print("Shape of y train : ",y_pca_train.shape)

print("Shape of y_test  : ",y_pca_test.shape)
svm_model_4 = svm.SVC(gamma=0.025, C=3, kernel= 'rbf')

svm_model_4.fit(X_pca_train , y_pca_train)



y_predict_4 = svm_model_4.predict(X_pca_test)



train_score_4 = svm_model_4.score(X_pca_train,y_pca_train)

test_score_4 = svm_model_4.score(X_pca_test, y_pca_test)



print("SVM_model_4 score for train set:", train_score_4*100)

print("SVM_model_4 score for test set:", test_score_4*100)



#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['SVM with PCA (kernel: rbf)'], 'accuracy':test_score_4 })

resultsDf = pd.concat([resultsDf, tempResultsDf])

resultsDf = resultsDf[['Method', 'accuracy']]



confusion_matrix_4 = confusion_matrix(y_pca_test,y_predict_4)



print("\n\nConfusion Matrix:\n   car bus van\n",confusion_matrix_4)

target_names = ['car', 'bus', 'van']

print("\nClassification Report:\n", classification_report(y_pca_test, y_predict_4, target_names=target_names))



resultsDf
# with tuned parameters

svm_model_5 = svm.SVC(gamma=0.05, C=10, kernel= 'rbf')

svm_model_5.fit(X_pca_train , y_pca_train)
y_predict_5 = svm_model_5.predict(X_pca_test)



train_score_5 = svm_model_5.score(X_pca_train,y_pca_train)

test_score_5 = svm_model_5.score(X_pca_test, y_pca_test)



print("SVM_model_5 score for train set:", train_score_5*100)

print("SVM_model_5 score for test set:", test_score_5*100)



#Store the accuracy results for each model in a dataframe for final comparison

tempResultsDf = pd.DataFrame({'Method':['SVM with PCA -Tuned'], 'accuracy':test_score_5 })

resultsDf = pd.concat([resultsDf, tempResultsDf])

resultsDf = resultsDf[['Method', 'accuracy']]



confusion_matrix_5 = confusion_matrix(y_pca_test,y_predict_5)



print("\n\nConfusion Matrix:\n   car bus van\n",confusion_matrix_5)

target_names = ['car', 'bus', 'van']

print("\nClassification Report:\n", classification_report(y_pca_test, y_predict_5, target_names=target_names))



resultsDf
sns.set_style('whitegrid')

plt.figure(figsize = (12,10))

plt.yticks(np.arange(0,100,10))

sns.barplot(x = list(resultsDf.Method), y = list(resultsDf.accuracy))
num_folds = 50

seed = 7



kfold = KFold(n_splits=num_folds, random_state=seed)

model = svm.SVC(gamma=0.05, C=10, kernel= 'rbf')

results = cross_val_score(model, XScaled_pca, y, cv=kfold)

print(results)

print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))



tempCVScores_Df = pd.DataFrame({'Method':['SVM with PCA -tuned'], 'accuracy':results.mean()*100.0, 'std(+/-)':results.std()*100.0})

CVScores_Df = pd.concat([CVScores_Df, tempCVScores_Df])

CVScores_Df = CVScores_Df[['Method', 'accuracy', 'std(+/-)']]

CVScores_Df
sns.distplot(results,kde=True,bins=5)

plt.xlabel("Accuracy")

plt.show()



# confidence intervals

alpha = 0.95                             # for 95% confidence 



p = ((1.0-alpha)/2.0) * 100              

lower = max(0.0, np.percentile(results, p))  



p = (alpha+((1.0-alpha)/2.0)) * 100

upper = min(1.0, np.percentile(results, p))





print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
# accuracy scores: 



resultsDf
plt.figure(figsize = (16,4))

plt.suptitle("Confusion Matrices",fontsize=12)

plt.subplots_adjust(wspace = 0.8, hspace = 0.8)



plt.subplot(1,3,1)

plt.title("SVM (kernel: Linear) Confusion Matrix")

sns.heatmap(confusion_matrix_1, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})





plt.subplot(1,3,2)

plt.title("SVM (kernel: rbf) Confusion Matrix")

sns.heatmap(confusion_matrix_2, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})



plt.subplot(1,3,3)

plt.title("SVM - Tuned Confusion Matrix")

sns.heatmap(confusion_matrix_3, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})

plt.figure(figsize = (16,4))

plt.suptitle("Confusion Matrices",fontsize=12)

plt.subplots_adjust(wspace = 0.8, hspace = 0.8)





plt.subplot(1,2,1)

plt.title("SVM with PCA (kernel: rbf) Confusion Matrix")

sns.heatmap(confusion_matrix_4, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})



plt.subplot(1,2,2)

plt.title("SVM with PCA -Tuned Confusion Matrix")

sns.heatmap(confusion_matrix_5, annot = True, cmap = "Blues", fmt = 'd', cbar = False, annot_kws = {"size": 12})

# comparing cross validation scores of raw data and with PCA



CVScores_Df
## Confusion matrix



plt.figure(figsize=(10,5))

class_label = ["car", "bus", "van"]

plt.xlabel("Predicted Class")

plt.ylabel("True Class")





plt.subplot(1,2,1)

svm_cm = confusion_matrix(y_test, y_predict_3)

df_cm = pd.DataFrame(svm_cm, index = class_label, columns = class_label)

sns.heatmap(df_cm, annot = True, fmt='d', cbar= False, cmap="Blues")

plt.title("Confusion Matrix -- SVM Tuned (Raw data)")



plt.subplot(1,2,2)

pca_cm = confusion_matrix(y_pca_test,y_predict_5)

df_cm1 = pd.DataFrame(pca_cm, index = class_label, columns = class_label)

sns.heatmap(df_cm1, annot = True,fmt = "d",cbar= False, cmap="Blues")

plt.title("Confusion Matrix -- SVM Tuned (PCA)")



plt.show()
target_names = ['car', 'bus', 'van']

print("\nClassification Report for SVM -Tuned (RAW data):\n", classification_report(y_test, y_predict_3, target_names=target_names))

print()

print()

print("\nClassification Report for SVM -Tuned (PCA):\n", classification_report(y_pca_test, y_predict_5, target_names=target_names))
#Finding optimal no. of clusters

from scipy.spatial.distance import cdist



from sklearn.cluster import KMeans



clusters = range(1,10)

meanDistortions = []



# creating a datacopy for PCA with groups (XScaled_pca)



XScaled_gr_pca = XScaled_pca.copy()



for k in clusters:

    km_model= KMeans(n_clusters = k)

    km_model.fit(XScaled_gr_pca)

    km_prediction = km_model.predict(XScaled_gr_pca)

    meanDistortions.append(sum(np.min(cdist(XScaled_gr_pca, km_model.cluster_centers_, 'euclidean'), axis=1)) / XScaled_gr_pca.shape[0])





plt.plot(clusters, meanDistortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Average distortion')

plt.title('Selecting k with the Elbow Method')
# Model fitting

kmeansmodel=KMeans(3)

kmeansmodel.fit(XScaled_gr_pca)

km_prediction=kmeansmodel.predict(XScaled_gr_pca)



#Append the prediction in the group

XScaled_gr_pca["GROUP"] = km_prediction

XScaled_gr_pca["GROUP"] = km_prediction

print("Groups Assigned : \n")

XScaled_gr_pca.head()
XScaled_gr_pca.groupby(['GROUP'])

XScaled_gr_pca.boxplot(by='GROUP', layout = (3,3),figsize=(15,10))
from sklearn import metrics



score_km = metrics.silhouette_score(XScaled_gr_pca, kmeansmodel.labels_, metric='euclidean')

print("The silhouette scores is "+str(score_km*100))
from sklearn.linear_model import LogisticRegression



model = LogisticRegression()



model.fit(X_train, y_train)

print (' Logistic Regression - Before PCA score', model.score(X_test, y_test))



model.fit(X_pca_train, y_pca_train)

print (' Logistic Regression - After PCA score', model.score(X_pca_test, y_pca_test))



num_folds = 10

seed = 1



kfold = KFold(n_splits=num_folds, random_state=seed)

model_scaled = LogisticRegression(solver='lbfgs', multi_class='auto')

results = cross_val_score(model_scaled, XScaled, y, cv=kfold)

print(results)

print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
kfold = KFold(n_splits=num_folds, random_state=seed)

model_pca = LogisticRegression(solver='lbfgs', multi_class='auto')

results = cross_val_score(model_pca, XScaled_pca, y, cv=kfold)

print(results)

print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()



nb.fit(X_train, y_train)

print (' Naive Bayes - Before PCA score', nb.score(X_test, y_test))



nb.fit(X_pca_train, y_pca_train)

print (' Naive Bayes - After PCA score', nb.score(X_pca_test, y_pca_test))
from sklearn.tree import DecisionTreeClassifier



dt_model = DecisionTreeClassifier(criterion = 'entropy' )



dt_model.fit(X_train, y_train)

print (' Decisiontree Classifier - Before PCA score', dt_model.score(X_test, y_test))



dt_model.fit(X_pca_train, y_pca_train)

print (' Decisiontree Classifier - After PCA score', dt_model.score(X_pca_test, y_pca_test))