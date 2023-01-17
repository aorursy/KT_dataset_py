import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sample_submission = pd.read_csv("../input/cat-in-the-dat/sample_submission.csv")

test = pd.read_csv("../input/cat-in-the-dat/test.csv")

train = pd.read_csv("../input/cat-in-the-dat/train.csv")



# iterating the columns 

list(train.columns.values) 
train.head(10)


print (train["bin_0"].agg(['nunique']),'\n', train["bin_0"].value_counts(),'\n------\n')

print (train["bin_1"].agg(['nunique']),'\n', train["bin_1"].value_counts(),'\n------\n')

print (train["bin_2"].agg(['nunique']),'\n', train["bin_2"].value_counts(),'\n------\n')

print (train["bin_3"].agg(['nunique']),'\n', train["bin_3"].value_counts(),'\n------\n')

print (train["bin_4"].agg(['nunique']),'\n', train["bin_4"].value_counts(),'\n------\n')
print (train["nom_0"].agg(['nunique']),'\n', train["nom_0"].value_counts(),'\n------\n')

print (train["nom_1"].agg(['nunique']),'\n', train["nom_1"].value_counts(),'\n------\n')

print (train["nom_2"].agg(['nunique']),'\n', train["nom_2"].value_counts(),'\n------\n')

print (train["nom_3"].agg(['nunique']),'\n', train["nom_3"].value_counts(),'\n------\n')

print (train["nom_4"].agg(['nunique']),'\n', train["nom_4"].value_counts(),'\n------\n')

print (train["nom_5"].agg(['nunique']),'\n', train["nom_5"].value_counts(),'\n------\n')

print (train["nom_6"].agg(['nunique']),'\n', train["nom_6"].value_counts(),'\n------\n')

print (train["nom_7"].agg(['nunique']),'\n', train["nom_7"].value_counts(),'\n------\n')

print (train["nom_8"].agg(['nunique']),'\n', train["nom_8"].value_counts(),'\n------\n')

print (train["nom_9"].agg(['nunique']),'\n', train["nom_9"].value_counts(),'\n------\n')

print (train["ord_0"].agg(['nunique']),'\n', train["ord_0"].value_counts(),'\n------\n')

print (train["ord_1"].agg(['nunique']),'\n', train["ord_1"].value_counts(),'\n------\n')

print (train["ord_2"].agg(['nunique']),'\n', train["ord_2"].value_counts(),'\n------\n')

print (train["ord_3"].agg(['nunique']),'\n', train["ord_3"].value_counts(),'\n------\n')

print (train["ord_4"].agg(['nunique']),'\n', train["ord_4"].value_counts(),'\n------\n')

print (train["ord_5"].agg(['nunique']),'\n', train["ord_5"].value_counts(),'\n------\n')
print (train["day"].agg(['nunique']),'\n', train["day"].value_counts(),'\n------\n')

print (train["month"].agg(['nunique']),'\n', train["month"].value_counts(),'\n------\n')
# Create the result using the test set
def calc_smooth_mean(df, by, on, m):

    # Compute the global mean

    mean = df[on].mean()



    # Compute the number of values and the mean of each group

    agg = df.groupby(by)[on].agg(['count', 'mean'])

    counts = agg['count']

    means = agg['mean']



    # Compute the "smoothed" means

    smooth = (counts * means + m * mean) / (counts + m)



    # Replace each value by the according smoothed mean

    #return df[by].map(smooth)

    return smooth
import scipy.stats as ss



def cramers_v(x, y):

    confusion_matrix = pd.crosstab(x,y)

    chi2 = ss.chi2_contingency(confusion_matrix)[0]

    n = confusion_matrix.sum().sum()

    phi2 = chi2/n

    r,k = confusion_matrix.shape

    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))

    rcorr = r-((r-1)**2)/(n-1)

    kcorr = k-((k-1)**2)/(n-1)

    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))



def getMapforTargetEncoding(df_preprocess):

    stack=[]

    # Using Smooth means to perform the target encoding

    m=10

    means5 = calc_smooth_mean(df_preprocess, by='nom_5', on='target', m=m)

    means6 = calc_smooth_mean(df_preprocess, by='nom_6', on='target', m=m)

    means7 = calc_smooth_mean(df_preprocess, by='nom_7', on='target', m=m)    

    means8 = calc_smooth_mean(df_preprocess, by='nom_8', on='target', m=m)

    means9 = calc_smooth_mean(df_preprocess, by='nom_9', on='target', m=m)

    

    stack.append(means5)

    stack.append(means6)

    stack.append(means7)

    stack.append(means8)

    stack.append(means9)

    

    return stack
# perform the Cramers V correlation

# print out the result

for col in train.columns:

    print ("column: [", col ,"] --> %.2f" % cramers_v(train[col],train['target']))

# Function to preprocess the data

def preprocess(df_preprocess, TargetEncodingMap):

    

    # Binary Encoding

    # ---------------------------------------------------------------

    #The first 5 columns,no preprocessing is required

    # copy all the transform cols to df_analyse

    # 

    df_analyse = df_preprocess[['bin_1','bin_4']]

    

    # transform bin_4

    # bin_4: Y=1, N=0

    mapping = {'T': 1, 'F': 0,'Y':1,'N':0}

    df_analyse = df_analyse.replace({'bin_4': mapping})

    



    # Nominal value encoding

    # ---------------------------------------------------------------

    # one hot encoding for nom_0, nom_1, nom_2, nom_3, nom_4

    

    #nom_0 has 3 values

    #nom_1 has 6 values

    #nom_2 has 6 values

    #nom_3 has 6 values

    #nom_4 has 4 values



    df_nom0=pd.get_dummies(df_preprocess['nom_0'],prefix='nom_0')

    df_nom1=pd.get_dummies(df_preprocess['nom_1'],prefix='nom_1')

    df_nom2=pd.get_dummies(df_preprocess['nom_2'],prefix='nom_2')

    df_nom3=pd.get_dummies(df_preprocess['nom_3'],prefix='nom_3')

    df_nom4=pd.get_dummies(df_preprocess['nom_4'],prefix='nom_4')

    



    # concate back to df_analyse

    df_analyse = pd.concat([df_analyse, df_nom0], axis=1) 

    df_analyse = pd.concat([df_analyse, df_nom1], axis=1) 

    df_analyse = pd.concat([df_analyse, df_nom2], axis=1) 

    df_analyse = pd.concat([df_analyse, df_nom3], axis=1) 

    df_analyse = pd.concat([df_analyse, df_nom4], axis=1) 



    # Target encoding for nom_5, nom_6, nom_7, nom_8, nom_9

    mean9 = TargetEncodingMap[4]

    mean8 = TargetEncodingMap[3]

    mean7 = TargetEncodingMap[2]

    mean6 = TargetEncodingMap[1]

    mean5 = TargetEncodingMap[0]

    

    

    df_nom5 = df_preprocess[['nom_5']]

    df_nom5['nom_5'] = df_nom5['nom_5'].map(mean5)

    df_nom6 = df_preprocess[['nom_6']]

    df_nom6['nom_6'] = df_nom6['nom_6'].map(mean6)

    df_nom7 = df_preprocess[['nom_7']]

    df_nom7['nom_7'] = df_nom7['nom_7'].map(mean7)

    df_nom8 = df_preprocess[['nom_8']]

    df_nom8['nom_8'] = df_nom8['nom_8'].map(mean8)

    df_nom9 = df_preprocess[['nom_9']]

    df_nom9['nom_9'] = df_nom9['nom_9'].map(mean9)



    #concate back to df_analyse

    df_analyse = pd.concat([df_analyse, df_nom5], axis=1) 

    df_analyse = pd.concat([df_analyse, df_nom6], axis=1) 

    df_analyse = pd.concat([df_analyse, df_nom7], axis=1) 

    df_analyse = pd.concat([df_analyse, df_nom8], axis=1) 

    df_analyse = pd.concat([df_analyse, df_nom9], axis=1) 



    # Ordinal value encoding

    # ---------------------------------------------------------------

    # Remapping for the ordinal variables

    # reorder ord_0

    ord0_mapper = {3:1, 

                    2:2,

                    1:3}

    df_ord0 = df_preprocess['ord_0'].replace(ord0_mapper)

    

    # reorder ord_1

    ord1_mapper = {'Novice':1, 

                   'Contributor':2,

                   'Expert':3,

                   'Master':4,

                   'Grandmaster':5

                  }

    df_ord1 = df_preprocess['ord_1'].replace(ord1_mapper)



    # reorder ord_2

    ord2_mapper = {'Freezing':1,    

                   'Cold':2, 

                   'Warm':3,

                   'Hot':4,

                   'Boiling Hot':5,

                   'Lava Hot':6

                  }

    df_ord2= df_preprocess['ord_2'].replace(ord2_mapper)



    # reorder ord_3 and remap ord_3

    dict= df_preprocess.groupby(['ord_3']).groups.keys()

    ord3mapper={}

    for i, val in enumerate(dict): 

        ord3mapper[val]=i+1

    df_ord3= df_preprocess['ord_3'].replace(ord3mapper)



    # reorder ord_4 and remap ord_4

    dict= df_preprocess.groupby(['ord_4']).groups.keys()

    ord4_mapper={}

    for i, val in enumerate(dict): 

        ord4_mapper[val]=i+1

    df_ord4= df_preprocess['ord_4'].replace(ord4_mapper)



    # reorder ord_5 and remap ord_5

    dict= df_preprocess.groupby(['ord_5']).groups.keys()

    ord5_mapper={}

    for i, val in enumerate(dict): 

        ord5_mapper[val]=i+1

    df_ord5= df_preprocess['ord_5'].replace(ord5_mapper)



    #concate back to df_analyse

    df_analyse = pd.concat([df_analyse, df_ord1], axis=1) 

    df_analyse = pd.concat([df_analyse, df_ord2], axis=1) 

    df_analyse = pd.concat([df_analyse, df_ord3], axis=1) 

    df_analyse = pd.concat([df_analyse, df_ord4], axis=1) 

    df_analyse = pd.concat([df_analyse, df_ord5], axis=1) 



    # Cyclic value encoding

    # https://towardsdatascience.com/ml-intro-5-one-hot-encoding-cyclic-representations-normalization-6f6e2f4ec001

    # ---------------------------------------------------------------

    np_dayofweek_sin = np.sin((df_preprocess['day']-1)*(2.*np.pi/7))

    np_dayofweek_cos = np.cos((df_preprocess['day']-1)*(2.*np.pi/7))

    np_month_sin = np.sin((df_preprocess['month']-1)*(2.*np.pi/12))

    np_month_cos = np.cos((df_preprocess['month']-1)*(2.*np.pi/12))



    df_cyclic = pd.DataFrame()

    df_cyclic['dayofweek_sin'] = np_dayofweek_sin

    df_cyclic['dayofweek_cos'] = np_dayofweek_cos

    df_cyclic['month_sin'] = np_month_sin

    df_cyclic['month_cos'] = np_month_cos

    

    df_analyse = pd.concat([df_analyse, df_cyclic], axis=1) 

   

    return df_analyse



myTargetEncodingstack = getMapforTargetEncoding(train)



df_train = preprocess(train, myTargetEncodingstack)

print(df_train)



from sklearn.model_selection import train_test_split



y=train['target']

X=df_train



# Split the training and test set to 80/20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, y_train.shape, X_test.shape)
# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics



randomForestClassifier= RandomForestClassifier(bootstrap=False, class_weight={0: 1, 1: 1.4}, criterion='gini', 

                                               max_depth=None, max_features='auto', max_leaf_nodes=None,min_samples_leaf=1,

                                               min_impurity_split=None, min_samples_split=2, 

                                               min_weight_fraction_leaf=0.0, oob_score=False, n_estimators=180)

randomForestClassifier.fit(X_train,y_train)

y_pred = randomForestClassifier.predict(X_test)



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

result = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")

print(result)

result1 = classification_report(y_test, y_pred)

print("Classification Report:",)

print (result1)

result2 = accuracy_score(y_test,y_pred)

print("Accuracy:",result2)
from sklearn.linear_model import LogisticRegression



# hyper parameters were obtained from Kaggle forum

lr = LogisticRegression(C=0.095, class_weight={0: 1, 1: 1.4}, tol=0.00001,solver='liblinear', penalty='l2')



lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score

result = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")

print(result)

result1 = classification_report(y_test, y_pred)

print("Classification Report:",)

print (result1)

result2 = accuracy_score(y_test,y_pred)

print("Accuracy:",result2)



# Encode the Kaggle test data

df_test = preprocess(test, myTargetEncodingstack)



# There are some unseen values in the test set. Target encoding results in some of the field values is NAN

# To fill the values with the mean

print(df_test.isnull().sum())

df_test.fillna(df_test['nom_5'].mean(), inplace=True)
y_pred_test_RFC = randomForestClassifier.predict(df_test)

# Write the result to the a file

df=pd.DataFrame(y_pred_test_RFC, columns=['target']) 

df.insert(0, 'id', range(300000, 300000 + len(df)))

df.to_csv('result_rf.csv', index=False)
y_pred_test_lr = lr.predict(df_test)

# Write the result to the a file

df=pd.DataFrame(y_pred_test_lr, columns=['target']) 

df.insert(0, 'id', range(300000, 300000 + len(df)))

df.to_csv('result_lr.csv', index=False)