import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')
from subprocess import check_output



print(check_output(["ls", "../input/wine-quality"]).decode("utf8"))
df = pd.read_csv('../input/wine-quality/winequalityN.csv')
df.info()
print(*df.columns, sep='\n')
df.columns = ('type', 'fixed_acidity', 'volatile_acidity', 'citric_acid',

       'residual_sugar', 'chlorides', 'free_sulfur_dioxide',

       'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol',

       'quality')
df.head()
import seaborn as sns
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
Sum = df.isnull().sum()

Percentage = ( df.isnull().sum()/df.isnull().count())



pd.concat([Sum,Percentage], axis =1, keys= ['Sum', 'Percentage'])
def null_cell(df): 

    total_missing_values = df.isnull().sum() 

    missing_values_per = df.isnull().sum()/df.isnull().count() 

    null_values = pd.concat([total_missing_values, missing_values_per], axis=1, keys=['total_null', 'total_null_perc']) 

    null_values = null_values.sort_values('total_null', ascending=False) 

    return null_values[null_values['total_null'] > 0] 
fill_list = (null_cell(df)).index
df_mean = df.copy()



for col in fill_list:

    df_mean.loc[:, col].fillna(df_mean.loc[:, col].mean(), inplace=True)
sns.heatmap(df_mean.isnull(),yticklabels=False,cbar=False,cmap='viridis')
corr_matrix = df_mean.corr()

corr_list = corr_matrix.quality.abs().sort_values(ascending=False).index[0:]
corr_list
plt.figure(figsize=(11,9))

dropSelf = np.zeros_like(corr_matrix)

dropSelf[np.triu_indices_from(dropSelf)] = True



sns.heatmap(corr_matrix, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f", mask=dropSelf)



sns.set(font_scale=1.5)
from scipy.stats import norm 
plt.figure(figsize = (20,22))



for i in range(1,13):

    plt.subplot(5,4,i)

    sns.distplot(df_mean[df_mean.columns[i]], fit=norm)

    
df_bins= df_mean.copy()
bins = [0,5,10]





labels = [0, 1] # 'low'=0, 'high'=1

df_bins['quality_range']= pd.cut(x=df_bins['quality'], bins=bins, labels=labels)



print(df_bins[['quality_range','quality']].head(5))



df_bins = df_bins.drop('quality', axis=1) 
plt.figure(figsize=(8,5))



sns.countplot(x = 'type', hue = 'quality_range', data = df_bins)

plt.show()

# 'low'=0, 'high'=1
plt.figure(figsize=(8,7))

sns.scatterplot(x='quality_range', 

                y='alcohol', 

                hue='type',

                data=df_bins);

plt.xlabel('Quality',size=15)

plt.ylabel('Alcohol', size =15)

plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

f.suptitle('Wine Types by Quality & Acidity', fontsize=14)



sns.violinplot(x='quality_range', y='volatile_acidity', hue='type', data=df_bins, split=True, inner='quart', linewidth=1.3,

               palette={'red': 'red', 'white': 'white'}, ax=ax1)

ax1.set_xlabel("Wine Quality Class ",size = 15,alpha=0.8)

ax1.set_ylabel("Wine Fixed Acidity",size = 15,alpha=0.8)



sns.violinplot(x='quality_range', y='alcohol', hue='type', data=df_bins, split=True, inner='quart', linewidth=1.3,

               palette={'red': 'darkred', 'white': 'white'}, ax=ax2)

ax2.set_xlabel("Wine Quality Class",size = 15,alpha=0.8)

ax2.set_ylabel("Wine Fixed Alcohol",size = 15,alpha=0.8)

plt.show()
plt.figure(figsize= (6,4))



low_quality = df_bins [df_bins['quality_range']== 0]['chlorides']

high_quality   = df_bins [df_bins['quality_range']== 1][ 'chlorides']

ax = sns.kdeplot(data= low_quality, label= 'low_quality', shade=True, color=None)

ax = sns.kdeplot(data= high_quality,label= 'high_quality',shade=True, color= "r")



plt.title("Chloride Level in Wine Classes")

plt.xlim(0.0,0.3)

plt.legend()

plt.show()
f, (ax1, ax2, ax3) = plt.subplots(3, figsize = (10,10))



f.suptitle('Wine Quality - Acidity Levels', fontsize=14)





fixed_acidity_low_quality    = df_bins [df_bins['quality_range']== 0]['fixed_acidity']

fixed_acidity_high_quality   = df_bins [df_bins['quality_range']== 1]['fixed_acidity']





volatile_acidity_low_quality = df_bins [df_bins['quality_range']== 0]['volatile_acidity']

volatile_acidity_high_quality= df_bins [df_bins['quality_range']== 1]['volatile_acidity']



citric_acid_low_quality      = df_bins [df_bins['quality_range']== 0]['citric_acid']

citric_acid_high_quality     = df_bins [df_bins['quality_range']== 1]['citric_acid']





sns.kdeplot(data=fixed_acidity_low_quality, label="low_quality", shade=True,ax=ax1)

sns.kdeplot(data=fixed_acidity_high_quality, label="high_quality", shade=True, ax=ax1)

ax1.set_xlabel("fixed_acidity",size = 15,alpha=0.8)

ax1.set_ylabel("Wine Quality",size = 15,alpha=0.8)





sns.kdeplot(data=volatile_acidity_low_quality, label="low_quality", shade=True,ax=ax2)

sns.kdeplot(data=volatile_acidity_high_quality, label="high_quality", shade=True, ax=ax2)

ax2.set_xlabel("volatile_acidity",size = 15,alpha=0.8)

ax2.set_ylabel("Wine Quality",size = 15,alpha=0.8)





sns.kdeplot(data=citric_acid_low_quality, label="low_quality", shade=True,ax=ax3)

sns.kdeplot(data=citric_acid_high_quality, label="high_quality", shade=True, ax=ax3)

ax3.set_xlabel("citric_acid",size = 15,alpha=0.8)

ax3.set_ylabel("Wine Quality",size = 15,alpha=0.8)





plt.legend()

plt.show()
plt.figure(figsize=(8,5))



residual_sugar_low   = df_bins [df_bins['quality_range']== 0]['residual_sugar']

residual_sugar_high  = df_bins [df_bins['quality_range']== 1]['residual_sugar'] 

ax = sns.kdeplot(data= residual_sugar_low, label= 'low quality', shade=True)

ax = sns.kdeplot(data= residual_sugar_high,   label= 'high quality',   shade=True)



plt.title("Distributions of Residual Sugar by Wine Qualities")

plt.legend()

plt.show()
plt.figure(figsize=(12,8))

sns.scatterplot(x='total_sulfur_dioxide', y='free_sulfur_dioxide', hue='quality_range',data=df_bins);

plt.xlabel('total_sulfur_dioxide',size=15)

plt.ylabel('free_sulfur_dioxide', size =15)
plt.figure(figsize=(8,7))



pH_low_quality  = df_bins [df_bins['quality_range']== 0]['pH']

pH_high_quality = df_bins [df_bins['quality_range']== 1][ 'pH']

ax = sns.kdeplot(data= pH_low_quality, label= 'low_quality', shade=True) 

ax = sns.kdeplot(data= pH_high_quality,label= 'high_quality',   shade=True)



plt.title("pH Levels in Low/High Quality Wines")

plt.xlabel('pH')

plt.legend()

plt.show()
plt.figure(figsize=(8,5))



density_low_quality  = df_bins [df_bins['quality_range']== 0]['density']

density_high_quality = df_bins [df_bins['quality_range']== 1][ 'density']

ax = sns.kdeplot(data= density_low_quality, label= 'low_quality', shade=True) 

ax = sns.kdeplot(data= density_high_quality,label= 'high_quality', shade=True)



plt.title("Density Levels in Low/High Quality of Wines")

plt.xlabel('density')

plt.legend()

plt.show()
plt.figure(figsize=(8,5))



sulphates_low_quality    = df_mean [df_bins['quality_range']== 0]['sulphates']

sulphates_high_quality   = df_mean [df_bins['quality_range']== 1][ 'sulphates']

ax = sns.kdeplot(data= sulphates_low_quality, label= 'low_quality',  shade=True) 

ax = sns.kdeplot(data= sulphates_high_quality,label= 'high_quality', shade=True)



plt.title("Sulphates Levels in Low/High Quality of Wines")

plt.xlabel('sulphates')

plt.legend()

plt.show()
outliers_by_12_variables = ['fixed_acidity', 'volatile_acidity', 'citric_acid',

                            'residual_sugar', 'chlorides', 'free_sulfur_dioxide',

                            'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'] 

plt.figure(figsize=(22,20))



for i in range(0,11):

    plt.subplot(5, 4, i+1)

    plt.boxplot(df_bins[outliers_by_12_variables[i]])

    plt.title(outliers_by_12_variables[i])
def winsor(x, multiplier=3): 

    upper= x.median() + x.std()*multiplier

    for limit in np.arange(0.001, 0.20, 0.001):

        if np.max(winsorize(x,(0,limit))) < upper:

            return limit

    return None 
from scipy.stats.mstats import winsorize



kolon_isimleri = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',

                                  'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']



for i in range(1,len(kolon_isimleri)):



    df_bins[kolon_isimleri[i]] = winsorize(df_bins[kolon_isimleri[i]], (0, winsor(df_bins[kolon_isimleri[i]])))
df_bins.type = df_bins.type.map({'white':0, 'red':1})
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score 
X = df_bins[['type', 'alcohol', 'density', 'volatile_acidity', 'chlorides',

       'citric_acid', 'fixed_acidity', 'free_sulfur_dioxide',

       'total_sulfur_dioxide', 'sulphates', 'residual_sugar', 'pH']] 

y = df_bins.quality_range



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=40)

lr = LogisticRegression(random_state=40)

lr.fit(X_train, y_train)
train_accuracy = lr.score(X_train, y_train)

test_accuracy = lr.score(X_test, y_test)

print('One-vs-rest', '-'*35, 

      'Accuracy in Train Group   : {:.2f}'.format(train_accuracy), 

      'Accuracy in Test  Group   : {:.2f}'.format(test_accuracy), sep='\n')
from sklearn.metrics import confusion_matrix as cm



predictions = lr.predict(X_test)

score = round(accuracy_score(y_test, predictions), 3)

cm1 = cm(y_test, predictions)

sns.heatmap(cm1, annot=True, fmt=".0f")

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Accuracy Score: {0}'.format(score), size = 15)

plt.show()
pred_test  = lr.predict(X_test)

pred_train = lr.predict(X_train)
from sklearn.metrics import confusion_matrix 





cm = confusion_matrix(y_test,pred_test)

cm
quality_pred = LogisticRegression(random_state=40)

quality_pred.fit(X_train,y_train)
confusion_matrix_train = confusion_matrix(y_train,pred_train)

confusion_matrix_test = confusion_matrix(y_test,pred_test)



print('Confusion Matrix Train Data', '--'*20, confusion_matrix_train, sep='\n')

print('Confusion Matrix Test Data', '--'*20, confusion_matrix_test, sep='\n')
TN = confusion_matrix_test[0][0]

TP = confusion_matrix_test[1][1]

FP = confusion_matrix_test[0][1]

FN = confusion_matrix_test[1][0]



print("(Total) True Negative       :", TN)

print("(Total) True Positive       :", TP)

print("(Total) Negative Positive   :", FP)

print("(Total) Negative Negative   :", FN)
FP+FN 
from sklearn.metrics import accuracy_score



print("Accuracy Score of Our Model     : ",  quality_pred.score(X_test, y_test))

#print("Accuracy Score of Our Model     : ",  accuracy_score(y_test, pred_test)) # same 
Error_Rate = 1- (accuracy_score(y_test, pred_test))  

Error_Rate
from sklearn.metrics import precision_score



print("precision_score()         : ",  precision_score(y_test, pred_test, average='micro'))
from sklearn.metrics import recall_score



print("recall_score()            : ",  recall_score(y_test, pred_test, average='micro'))
print(" Specificity Score   : ",  (TN)/(TN + FP)) 
from sklearn.metrics import f1_score



precision_s = precision_score(y_test, pred_test,average='micro')

recall_s    = recall_score(y_test, pred_test, average='micro')





print("F1_score     : ",  2*((precision_s*recall_s)/(precision_s + recall_s)))

#print("F1_score     : ",  f1_score(y_test, pred_test,average='micro')) #By formula
from sklearn.metrics import classification_report, precision_recall_fscore_support



print(classification_report(y_test,pred_test))



print("f1_score        : {:.2f}".format(f1_score(y_test, pred_test, average='micro')))

print("recall_score    : {:.2f}".format(recall_score(y_test, pred_test, average='micro')))

print("precision_score : {:.2f}".format(precision_score(y_test, pred_test, average='micro')))



print('\n')

metrics =  precision_recall_fscore_support(y_test, pred_test)

print("Precision       :" , metrics[0]) 

#print("Recall          :" , metrics[1]) 

print("F1 Score        :" , metrics[2]) 
probs = quality_pred.predict_proba(X_test)[:,1]  #Predict probabilities for the test data



from sklearn.metrics import roc_curve, roc_auc_score



fpr, tpr, thresholds  = roc_curve(y_test, probs) #Get the ROC Curve





import matplotlib.pyplot as plt





plt.figure(figsize=(8,5))

# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate = 1 - Specificity Score')

plt.ylabel('True Positive Rate  = Recall Score')

plt.title('ROC Curve')

plt.show()
print('AUC Değeri : ', roc_auc_score(y_test.values, probs))
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, pred_test)



plt.plot(recall, precision)

plt.show()
from sklearn.metrics import log_loss



print("Log-Loss)    : " , log_loss(y_test.values, probs))

print("Error Rate   : " , 1- accuracy_score(y_test.values, pred_test))
C_values = [0.001,0.01,0.1,1,10,100, 1000]

accuracy_df = pd.DataFrame(columns = ['C_values','Accuracy'])



accuracy_values = pd.DataFrame(columns=['C Value', 'Accuracy Train', 'Accuracy Test'])



for c in C_values:

    

    # Apply logistic regression model to training data

    lr = LogisticRegression(penalty = 'l2', C = c, random_state = 0)

    lr.fit(X_train,y_train)

    accuracy_values = accuracy_values.append({'C Value': c,

                                                    'Accuracy Train' : lr.score(X_train, y_train),

                                                    'Accuracy Test': lr.score(X_test, y_test)

                                                    }, ignore_index=True)

display(accuracy_values)
df_mean.head(1)
df_bins3= df_mean.copy()
df_bins3.type = df_bins3.type.map({'white':0, 'red':1})
bins = [0,4,7,10]



labels = [0,1,2] # 'low'=0,'average'=1, 'high'=2



df_bins3['quality_range']= pd.cut(x=df_bins3['quality'], bins=bins, labels=labels)



#df_bins3.type = df_bins3.type.map({'white':0, 'red':1})



print(df_bins3[['quality_range','quality']].head(5))

X = df_bins3[['type', 'alcohol', 'density', 'volatile_acidity', 'chlorides',

       'citric_acid', 'fixed_acidity', 'free_sulfur_dioxide',

       'total_sulfur_dioxide', 'sulphates', 'residual_sugar', 'pH']]

y = df_bins3.quality_range



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=40)

X_test.head()
lr    = LogisticRegression(random_state=40)

lr.fit(X_train, y_train)
train_accuracy = lr.score(X_train, y_train)

test_accuracy = lr.score(X_test, y_test)

print('One-vs-rest', '-'*35, 

      'Accuracy Score of Train Model : {:.2f}'.format(train_accuracy), 

      'Accuracy Score of Test  Model : {:.2f}'.format(test_accuracy), sep='\n')
from sklearn.metrics import confusion_matrix as cm



predictions = lr.predict(X_test)

score = round(accuracy_score(y_test, predictions), 3)

cm1 = cm(y_test, predictions)

sns.heatmap(cm1, annot=True, fmt=".0f")

plt.xlabel('Predicted Values')

plt.ylabel('Actual Values')

plt.title('Accuracy Score: {0}'.format(score), size = 15)

plt.show()
y_pred = lr.predict(X_test)

y_pred[y_pred == 2]
cm = confusion_matrix(y_test,y_pred)

cm
quality_pred = LogisticRegression(random_state=40)

quality_pred.fit(X_train,y_train)
pred_train = lr.predict(X_train)

pred_test  = lr.predict(X_test)
confusion_matrix_train = confusion_matrix(y_train,pred_train)

confusion_matrix_test = confusion_matrix(y_test,pred_test)



print('Confusion Matrix Train Data', '--'*20, confusion_matrix_train, sep='\n')

print('Confusion Matrix Test  Data ', '--'*20, confusion_matrix_test, sep='\n')
#TN = confusion_matrix_test[0][0]

#TP = confusion_matrix_test[1][1]

#FP = confusion_matrix_test[0][1]

#FN = confusion_matrix_test[1][0]



print("(Total) True Negative       :", TN)

print("(Total) True Positive       :", TP)

print("(Total) Negative Positive   :", FP)

print("(Total) Negative Negative   :", FN)
from sklearn.metrics import accuracy_score



print("Accuracy Score of Test Model : ",  quality_pred.score(X_test, y_test))
Error_Rate = 1 - (accuracy_score(y_test, pred_test))

Error_Rate
from sklearn.metrics import precision_score



print("precision_score        : ",  precision_score(y_test, pred_test, average='micro'))
from sklearn.metrics import recall_score



print("recall_score        : ",  recall_score(y_test, pred_test, average='micro'))
from sklearn.metrics import f1_score



precision_s = precision_score(y_test, pred_test,average='micro')

recall_s    = recall_score(y_test, pred_test, average='micro')





print("F1_score     : ",  2*((precision_s*recall_s)/(precision_s + recall_s)))# by mathematical formula

print("f1_score()   : ",  f1_score(y_test, pred_test,average='micro'))  #By formula
from sklearn.metrics import classification_report, precision_recall_fscore_support



print(classification_report(y_test,pred_test) )



print("f1_score()         : {:.2f}".format(f1_score(y_test, pred_test, average='micro')))

print("recall_score()     : {:.2f}".format(recall_score(y_test, pred_test, average='micro')))

print("precision_score()  : {:.2f}".format(precision_score(y_test, pred_test, average='micro')))



print('\n')

metrikler =  precision_recall_fscore_support(y_test, pred_test)

print("Precision   :" , metrics[0]) 

print("Recall      :" , metrics[1]) 

print("F1 Score    :" , metrics[2]) 



warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelBinarizer
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):

    lb = LabelBinarizer()

    lb.fit(y_test)

    y_test = lb.transform(y_test)

    y_pred = lb.transform(y_pred)

    return roc_auc_score(y_test, y_pred, average=average)
print('AUC Değeri : ', multiclass_roc_auc_score(y_test.values, y_pred))
probs = quality_pred.predict_proba(X_test)[:,1]



from sklearn.metrics import roc_curve, roc_auc_score



fpr, tpr, thresholds  = roc_curve(y_test, probs, pos_label=1)





# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()
from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(y_test, probs, pos_label=1)



plt.plot(precision, recall)

plt.show()
C_values = [0.001,0.01,0.1,1,10,100, 1000]

accuracy_df = pd.DataFrame(columns = ['C_values','Accuracy'])



accuracy_values = pd.DataFrame(columns=['C Value', 'Accuracy Train', 'Accuracy Test'])



for c in C_values: 

    

    # Apply logistic regression model to training data

    lr = LogisticRegression(penalty = 'l2', C = c, random_state = 0)

    lr.fit(X_train,y_train)

    accuracy_values = accuracy_values.append({'C Value': c,

                                                    'Accuracy Train' : lr.score(X_train, y_train),

                                                    'Accuracy Test': lr.score(X_test, y_test)

                                                    }, ignore_index=True)

display(accuracy_values)
df_mean_imb = df_mean.copy() 
bins = [0,4,10] 





labels = [0, 1] # 'low'=0, 'high'=1 

df_mean_imb['quality_range']= pd.cut(x=df_mean_imb['quality'], bins=bins, labels=labels) 



print(df_mean_imb[['quality_range','quality']].head(5)) 



df_mean_imb = df_mean_imb.drop('quality', axis=1) #
sns.countplot(df_mean_imb.quality_range)

 #'low'=0, 'high'=1

    

print("Low Quality  0   : %{:.2f}".format(sum(df_mean_imb.quality_range)/len(df_mean_imb.quality_range)*100))

print("High Quality 1   : %{:.2f}".format((len(df_mean_imb.quality_range)-sum(df_mean_imb.quality_range))/len(df_mean_imb.quality_range)*100))
balance = (df_mean_imb.quality_range.value_counts()[1]/df_mean_imb.quality_range.shape[0])*100

print('Data Quality Percentage:\n', balance,'%')
from sklearn.utils import resample 

from imblearn.over_sampling import SMOTE 

smote = SMOTE() 
df_mean_imb.type = df_mean_imb.type.map({'white':0, 'red':1}) 
X =  df_mean_imb.drop(['quality_range'], axis=1) 

y =  df_mean_imb.quality_range 



X_sm, y_sm =smote.fit_resample(X,y) 



print(X.shape, y.shape) 

print(X_sm.shape, y_sm.shape) 

sns.countplot(y_sm) 
def create_model(X, y): 

    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.20, random_state=40, stratify = y) 

    logreg_model = LogisticRegression() 

    logreg_model.fit(X_train, y_train) 



    pred_train = logreg_model.predict(X_train) 

    pred_test = logreg_model.predict(X_test) 

    confusion_matrix_train = confusion_matrix(y_train, pred_train) 

    confusion_matrix_test = confusion_matrix(y_test, pred_test) 

    print("Accuracy of Test Model : ",  logreg_model.score(X_test, y_test)) 

    print("Train Data Set") 

    print(classification_report(y_train,pred_train) ) 

    print("Test Data Set ") 

    print(classification_report(y_test,pred_test) ) 

    return  None 
create_model(X_sm,y_sm) 

warnings.filterwarnings('ignore')
df_bins.head()
X = df_bins.drop(['quality_range'], axis=1)

y = df_bins.quality_range

y = np.array(y)
plt.style.use('fivethirtyeight')



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

print("Number of Rows in    Training dataset :  {} ".format(len(X_train)))

print("Number of Targets in Training dataset :  {} ".format(len(y_train)))

print("Number of Rows in    Test dataset :  {} ".format(len(X_test)))

print("Number of Targets in Test dataset :  {} ".format(len(y_test)))
sns.countplot(y_test)

plt.ylim((0,1000))
plt.figure(figsize=(15,9))

y_list = [y, y_train, y_test]

titles = ['All Data','Train Data', 'Test Data']



for i in range(1,4):

    plt.subplot(1,3,i)

    sns.countplot(y_list[i-1])

    plt.title(titles[i-1])

    

print("Tüm veri kümesi '0' yüzdesi : %{:.0f} ".format(len(y[y==0])/len(y)*100))

print("Test verisi '0' yüzdesi     : %{:.0f} ".format(len(y_test[y_test==0])/len(y_test)*100))

print("Eğitim verisi '0' yüzdesi   : %{:.0f} ".format(len(y_train[y_train==0])/len(y_train)*100))
LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

tahmin_eğitim = model.predict(X_train)

tahmin_test = model.predict(X_test)

model.score(X_test, y_test)
from sklearn.model_selection import KFold 

kf = KFold(n_splits=5, shuffle=True, random_state=40) 
X.loc[[3,5]] 

parcalar = kf.split(X)

for num, (train_index, test_index) in enumerate(parcalar): 

    print("{}.Training Set Size : {}".format(num+1,len(train_index)))  

    print("{}.Test Set Size     : {}".format(num+1,len(test_index))) 

    print('-'*26)
from sklearn.metrics import mean_squared_error 



model2 = LogisticRegression()

pieces = kf.split(X)

accuracy_list = []



for i, (egitim_indeks, test_indeks) in enumerate(pieces):

    

    X_train, y_train = X.loc[train_index], y[train_index]

    X_test, y_test = X.loc[test_indeks], y[test_indeks]

    

    model2.fit(X_train, y_train)

    tahmin = model2.predict(X_test)

    accuracy_value = model2.score(X_test, y_test)  

    

    accuracy_list.append(accuracy_value)

    

    print("{}.Accuracy Value of Pieces: {:.3f}".format(i+1, accuracy_value))

    print("-"*30)
print("Avarage Accuracy Value : {:.2f}".format(np.mean(accuracy_list)))
from sklearn.model_selection import cross_validate, cross_val_score
lrm = LogisticRegression()

cv = cross_validate(estimator=lrm,

                     X=X,

                     y=y,

                     cv=10,return_train_score=True

                    )

print('Test Scores            : ', cv['test_score'], sep = '\n')

print("-"*50)

print('Train Scores           : ', cv['train_score'], sep = '\n')
print('Mean of Test Set  : ', cv['test_score'].mean())

print('Mean of Train Set : ', cv['train_score'].mean())
cv = cross_validate(estimator=lrm, 

                     X=X,

                     y=y,

                     cv=10,return_train_score=True,

                     scoring = ['accuracy', 'r2', 'precision']

                    )
print('Test Set Accuracy   Mean      : {:.2f}'.format(cv['test_accuracy'].mean()))

print('Test Set R Square   Mean      : {:.2f}'.format(cv['test_r2'].mean()))

print('Test Set Precision  Mean      : {:.2f}'.format(cv['test_precision'].mean()))

print('Train Set Accuracy  Mean      : {:.2f}'.format(cv['train_accuracy'].mean()))

print('Train Set R Square  Mean      : {:.2f}'.format(cv['train_r2'].mean()))

print('Train Set Precision Mean      : {:.2f}'.format(cv['train_precision'].mean()))
cv = cross_val_score(estimator=lrm,

                     X=X,

                     y=y,

                     cv=10                    

                    )

print('Model Scores           : ', cv, sep = '\n')
from sklearn.model_selection import cross_val_predict 
y_pred = cross_val_predict(estimator=lrm, X=X, y=y, cv=10)

print(y_pred[0:10])
logreg = LogisticRegression()

print(logreg.get_params())
parameters = {"C": [10 ** x for x in range (-5, 5, 1)],

                "penalty": ['l1', 'l2']

                }
parameters
from sklearn.model_selection import GridSearchCV





grid_cv = GridSearchCV(estimator=logreg,

                       param_grid = parameters,

                       cv = 10

                      )

grid_cv.fit(X, y)
print("The Best Parametre : ", grid_cv.best_params_)

print("The Best Score     : ", grid_cv.best_score_)
results = grid_cv.cv_results_

df = pd.DataFrame(results)

df.head()
df = df[['param_penalty','param_C', 'mean_test_score']]

df = df.sort_values(by='mean_test_score', ascending = False)

df
#The most successful 10 parametres on a chart.

plt.style.use('fivethirtyeight')



plt.figure(figsize=(12,12))



sns.scatterplot(x = 'param_C', y = 'mean_test_score', hue = 'param_penalty', data = df[0:10], s=150)



plt.xscale('symlog')

#plt.ylim((0.9,1))

plt.show()
parametres = {"C": [10 ** x for x in range (-5, 5, 1)],

                "penalty": ['l1', 'l2']

                }
from sklearn.model_selection import RandomizedSearchCV



import warnings

warnings.filterwarnings('ignore')





rs_cv = RandomizedSearchCV(estimator=logreg,

                           param_distributions = parametres,

                           cv = 10,

                           n_iter = 10,

                           random_state = 111,

                           scoring = 'precision'

                      )

rs_cv.fit(X, y)
print("The Best Parametres        : ", rs_cv.best_params_)

print("All Precisions Values      : ", rs_cv.cv_results_['mean_test_score'])

print("The Best Precision Value   : ", rs_cv.best_score_)
results_rs = rs_cv.cv_results_

df_rs = pd.DataFrame(results_rs)
results_rs = rs_cv.cv_results_

df_rs = pd.DataFrame(results_rs)

df_rs = df_rs[['param_penalty','param_C', 'mean_test_score']]

df_rs = df_rs.sort_values(by='mean_test_score', ascending = False)

df_rs
plt.style.use('fivethirtyeight')

plt.figure(figsize=(12,12))

sns.scatterplot(x = 'param_C', y = 'mean_test_score', hue = 'param_penalty', data = df_rs, s=200)

plt.xscale('symlog')

plt.ylim((0.0,1))

plt.show()
