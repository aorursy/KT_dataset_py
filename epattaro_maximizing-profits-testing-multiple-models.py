import pandas as pd

import numpy as np



%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib as mpl

from matplotlib import font_manager as fm

from matplotlib import gridspec



import seaborn as sns



from sklearn.model_selection import train_test_split



from xgboost import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
df = pd.read_csv('../input/creditcard.csv', encoding='utf-8')

print (df.info())
count = df['Class'].value_counts()



plt.bar([0,1], count, align='center', color=['green','red'], width=0.5)



plt.yticks(fontsize=20, fontweight='bold')

plt.xticks([0,1],['Regular','Fraud'],fontsize=20, fontweight='bold')



plt.ylabel('Count', fontsize=20, fontweight='bold')

plt.xlabel('Class', fontsize=20, fontweight='bold')



plt.ylim((0,350000))

plt.title('Transaction Count\nby type\n', fontsize=20, fontweight='bold')



######################

######################



cc_rev = df.groupby('Class')['Amount'].sum()

cc_rev.loc[0] = cc_rev.loc[0]*0.01 

cc_rev.loc[len(cc_rev)] = cc_rev.loc[0] - cc_rev.loc[1]



cc_rev.index = ['IF','DL','Profit']



plt.figure()

plt.bar([1,2,3], cc_rev, align='center', color=['blue','red','green'], width=0.5)



plt.yticks(fontsize=20, fontweight='bold')

plt.xticks(fontsize=20, fontweight='bold')



plt.xticks([1,2,3], cc_rev.index)



plt.ylabel('Amount ($)', fontsize=20, fontweight='bold')

plt.xlabel('Type', fontsize=20, fontweight='bold')



plt.ylim((0,350000))



plt.title('Balance from Transactions\n', fontsize=20, fontweight='bold')



plt.tight_layout()
X_train, X_test, y_train, y_test = train_test_split(df.drop(u'Class',1), df[u'Class'], test_size=0.20, random_state=42)



classifiers = [

    KNeighborsClassifier(3),

    XGBClassifier(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis()]



# Logging for Visual Comparison

log_cols=["Classifier", "Precision","DL Prevented", "Sensitivity","Revenue Losses"]

log = pd.DataFrame(columns=log_cols)



for clf in classifiers:

    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    

    predictions = clf.predict(X_test)

    

    DF = pd.DataFrame([predictions,y_test], index=['prediction','true'], columns=X_test.index).T

    DF['Amount'] = X_test['Amount']

    DLP = DF[(DF['prediction']==1)&(DF['true']==1)]['Amount'].sum()

    RL = DF[(DF['prediction']==1)&(DF['true']==0)]['Amount'].sum()*0.01

    

    ps = len (DF[(DF['prediction']==1)&(DF['true']==1)])/ float ( len (DF[DF['true']==1]))

    ss = len (DF[(DF['prediction']==0)&(DF['true']==0)])/ float ( len (DF[DF['true']==0]))

    

    print("="*30)

    print(name)

    

    print('****Results****')

    print("Precision: {:.4%}".format(ps))

    print("DL Prevented: %d" %(DLP))

    

    print("Sensitivity: {:.4%}".format(ss))

    print("Revenues Loss: %d"%(RL))

    

    log_entry = pd.DataFrame([[name, ps, DLP ,ss, RL]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)

log['Net'] = log['DL Prevented']-log['Revenue Losses']
plt.figure(figsize=(15,15))



gs = gridspec.GridSpec(3, 5)

ax1 = plt.subplot(gs[0, :2])



string = log['Classifier'].map(lambda x: x.replace('Classifier','').replace('DiscriminantAnalysis','DA'))



plt.bar(np.arange (len (log)), log['DL Prevented'].values, align='center')

plt.xticks(np.arange(len(log)), string, fontsize=20,fontweight='bold', rotation='vertical')

plt.yticks(fontsize=20,fontweight='bold')

plt.title('Default Losses Prevented\n', fontsize=20, fontweight='bold')



ax2 = plt.subplot(gs[0,3:])

# plt.subplot(1,2,2)

plt.bar(np.arange (len (log)), log['Revenue Losses'].values, align='center')

plt.xticks(np.arange(len(log)), string, fontsize=20,fontweight='bold', rotation='vertical')

plt.yticks(fontsize=20,fontweight='bold')

plt.title('Interchange Fee Gains Lost\n', fontsize=20, fontweight='bold')



ax3 = plt.subplot(gs[2,:])

# plt.subplot(1,2,2)

plt.bar(np.arange (len (log)), log['Net'].values, align='center')

plt.xticks(np.arange(len(log)), string, fontsize=20,fontweight='bold', rotation='vertical')

plt.yticks(fontsize=20,fontweight='bold')

plt.title('Total Profit Generated\n', fontsize=20, fontweight='bold')



# plt.tight_layout()