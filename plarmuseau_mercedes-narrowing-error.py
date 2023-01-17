import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



pal = sns.color_palette()



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print(train.shape,test.shape)

train['tt']=0

train['y']=np.log(np.log(train['y']))

test['tt']=1

total=train.append(test)

total=total.sort_values('ID')

total['group']=total['ID']/426

total=total.set_index('ID')



total['group']=total['group'].round(0)

total=total.drop(['X214', 'X239', 'X53', 'X199', 'X134', 'X147', 'X222', 'X48', 'X119', 'X227', 'X146', 'X226', 'X326', 'X360', 'X382', 'X216', 'X62', 'X262', 'X67', 'X254', 'X279', 'X364', 'X71', 'X84', 'X385', 'X60', 'X293', 'X330', 'X296', 'X299', 'X44', 'X35', 'X37', 'X58', 'X39', 'X76'],axis=1)
groepX0=total.groupby(['X0','group'])['y'].describe().fillna(method='bfill')

groepX0['eff']=groepX0['std']/groepX0['mean']

groepX0['eff2']=groepX0['eff']*groepX0['std']





def clust(x):

    kl=0

    if x<0.00024:   # low variability cluster

        kl=1

    if x>0.000239 and x<0.000438: # moderate variability cluster, process with short adjustments

        kl=2

    if x>0.0004379: # high variability class, process times with long outages, failures of tests

        kl=4

    return kl

groepX0['clust']=groepX0['eff2'].map(clust)

groepX0

                                   
#compare cluster 4 with 1   

#first merge data

total=pd.merge(total,groepX0[['mean','min','25%','50%','75%','std','clust']], how='outer', left_on=['X0','group'],suffixes=('', '_X0'), right_index=True)





from collections import Counter

def todrop_col(df,tohold):

    # use todrop_col(dataframe,['listtohold'])

    # Categorical features

    df.replace([np.inf, -np.inf], np.nan).fillna(value=-1)

    

    cat_cols = []

    for c in df.columns:

        if df[c].dtype == 'object':

            cat_cols.append(c)

    #print('Categorical columns:', cat_cols)

    

    

    # Constant columns

    cols = df.columns.values    

    const_cols = []

    for c in cols:   

        if len(df[c].unique()) == 1:

            const_cols.append(c)

    #print('Constant cols:', const_cols)

    

    

    # Dublicate features

    d = {}; done = []

    cols = df.columns.values

    for c in cols:

        d[c]=[]

    for i in range(len(cols)):

        if i not in done:

            for j in range(i+1, len(cols)):

                if all(df[cols[i]] == df[cols[j]]):

                    done.append(j)

                    d[cols[i]].append(cols[j])

    dub_cols = []

    for k in d.keys():

        if len(d[k]) > 0: 

            # print k, d[k]

            dub_cols += d[k]        

    #print('Dublicates:', dub_cols)

    

    kolom=list(set(dub_cols+const_cols+cat_cols))

    kolom=[k for k in kolom if k not in tohold]

    

    return kolom



def tree_col(df,splitcol,splitval,groupcol):

    #use tree_col(dataframe,column that splits,vale to split, column that groups)

    #sklear feature selection

    import sklearn    

    from sklearn.svm import LinearSVC

    from sklearn.feature_selection import SelectFromModel

    from sklearn.ensemble import ExtraTreesClassifier

    

    tabel = df[df[splitcol]==splitval]

    label = tabel[groupcol].round(0)

    feat = df.columns  

    clf = ExtraTreesClassifier()

    clf = clf.fit(tabel[feat], label)

    model = SelectFromModel(clf, prefit=True)

    interesting_cols = model.transform(tabel[feat])

    #print('Treeclassifier cols',interesting_cols.shape)

    tabel2=pd.DataFrame(interesting_cols,index=tabel.index)

    feat2=tabel2.columns

    feat3=[]

    for ci in feat:

        for cj in feat2:

            if all(tabel[ci] == tabel2[cj]):

                feat3.append(ci) 

    #print('interesting Treecolumns',feat3)

    return feat3



def corr_col(df,explvar):

    

    corr4=df.corr()

    explstd=corr4[explvar]

    absexplstd=explstd.abs()

    expl_std4=[k for k in corr4.columns if absexplstd.loc[k]>0.1]

    return expl_std4



def plotxy(titel,xlabel,ylabel,toty,y_pred,y_test,y_pred2,y_test2):

    plt.figure(figsize=(20,5))



    plt.subplot(1,5,1)



    plt.title(titel+xlabel+ylabel)

    plt.plot([1.4,1.8], [1.4,1.8], color='g', alpha=0.3)

    plt.scatter(x=toty, y=y_pred, marker='.', alpha=0.5)

    plt.scatter(x=toty, y=y_pred2, marker='.', alpha=0.5,color='g')

    plt.scatter(x=[np.mean(toty)], y=[np.mean(y_pred)], marker='o', color='red')

    plt.xlabel(xlabel); plt.ylabel(ylabel)

    

    plt.subplot(1,5,2)

    sns.distplot(toty, kde=False, color='g')

    sns.distplot(y_pred, kde=False, color='r')

    plt.title('Distr.'+xlabel )



    plt.subplot(1,5,3)

    sns.distplot(toty, kde=False, color='g')

    sns.distplot(y_test, kde=False, color='b')

    plt.title('Distr'+ylabel)



    plt.subplot(1,5,4)

    sns.distplot(toty, kde=False, color='g')

    sns.distplot(y_pred2, kde=False, color='r')

    plt.title(' Distr. 2'+xlabel)



    plt.subplot(1,5,5)

    sns.distplot(toty, kde=False, color='g')

    sns.distplot(y_test2, kde=False, color='b')

    plt.title('Distr2'+ylabel)

    



y_pred_tot=pd.DataFrame()

#select cluster 4  merge ,split, search constant columns

# select columns correlating with variability

for clustval in [4,2,1]:

    total4=total[total['clust']==clustval]



    dropcol=todrop_col(total4,['clust','tt'])

    total4=total4.drop(dropcol,axis=1)

    total4['intercept']=1

    total41=total4[total4['tt']==1]

    total40=total4[total4['tt']==0] 



    #tree classifier columns tree related with group

    treeko4=tree_col(total4,'tt',0,'group')



    #columns correlated with standarddeviation

    expl_std4=corr_col(total40,'std')



    expl_std = list(set(expl_std4+treeko4))

    expl_std=[k for k in expl_std if k not in ['y','mean','min','std','25%','50%','75%']]

    import statsmodels.formula.api as sm

    #ols

    

    res = sm.OLS(total40.y,total40[expl_std]).fit()

    print('Columns explaining variance ',clustval)

    tval=pd.DataFrame(res.tvalues)

    ttval=tval[tval[0]>2].append(tval[tval[0]<-2])

    print('signific columns',[kt for kt in ttval.index])

    y_pred = res.predict(total40[expl_std])

    y_test = res.predict(total41[expl_std])

    #print(pd.DataFrame(y_test ).isnull())

    #print(y_test)

    if clustval==4:

        submis=pd.DataFrame(y_test)

        y_pred_tot=y_pred_tot.append(submis)

    if clustval==2:

        submis=pd.DataFrame(y_test)

        y_pred_tot=y_pred_tot.append(submis)

    if clustval==1:

        submis=pd.DataFrame(y_test)

        y_pred_tot2=y_pred_tot.append(submis)

        #print(y_pred_tot)

        y_pred_tot2=np.exp(np.exp(y_pred_tot2))

        y_pred_tot2.to_csv('stacked-models2.csv')    #0.48

    res2 = sm.OLS(total40.y,total40[['intercept','min','std'] ]).fit()

    print(res2.summary())

    

    y_pred2 = res2.predict(total40[['intercept','min','std']])

    y_test2 = res2.predict(total41[['intercept','min','std']])

    if clustval==1:

        submis=pd.DataFrame(y_test2)

        y_pred_tot=y_pred_tot.append(submis)

        #print(y_pred_tot)

        y_pred_tot=np.exp(np.exp(y_pred_tot))        

        y_pred_tot.to_csv('stacked-models1.csv')  #0.39 the mix is worse althoug graph is better







    plotxy(' fit ',' train predicted',' test predicted',total40.y,y_pred,y_test,y_pred2,y_test2)



#print(y_pred_tot)
import statsmodels.formula.api as sm



#split data again

total_0=total[total['tt']==0] # train data

total_1=total[total['tt']==1] #test data

total_0['tt']=1



#res = sm.ols(formula="y ~ X115+X116+X144+X157+X220+X27+X301+X313+X315+X334+min+std+clust",data=total).fit()

res = sm.ols(formula="y ~ X186+X187+X204+X205+X142+X263+X156+X157+X158+X51+X168+X171+X136",data=total_0).fit()

print(res.summary())

  



y_pred = res.predict(total_0)

print('check size pred',y_pred.shape,total_0.shape)



y_pred1 = res.predict(total_1)

print('chekc size pred1',y_pred1.shape,total_1.shape)

print(y_pred1.head())



sub = pd.DataFrame()

sub['ID'] = y_pred1.index

sub['y'] = y_pred1

sub.to_csv('submission.csv', index=False)





plt.figure(figsize=(16,4))







plt.subplot(1,4,2)

sns.distplot(total_0.y, kde=False, color='g')

sns.distplot(y_pred, kde=False, color='r')

plt.title('Distr. of train and pred. train')



plt.subplot(1,4,3)

sns.distplot(total_0.y, kde=False, color='g')

sns.distplot(y_pred1, kde=False, color='b')

plt.title('Distr. of train and pred. test')



#kolom4=['X215', 'X187', 'X205', 'X186', 'X118', 'X157', 'X156', 'X275', 'X204', 'X51']

#kolom1=['tt','X313', 'X157', 'X316', 'X156', 'X301', 'X158', 'X286', 'X118', 'X142', 'X263', 'X54', 'X315', 'X18', 'X314', 'X29', 'X125', 'X351', 'group', 'X275', 'X232']

#kolom2=['X187', 'X186', 'X118', 'X272', 'X171', 'X314', 'X194', 'X276', 'X232', 'X311', 'X157', 'X136', 'X168', 'X156', 'X54', 'X29', 'X162', 'X313', 'X127', 'X352', 'X148', 'X261', 'X316']

kolom4=['X168', 'X205', 'X171', 'X204', 'X130', 'X18', 'X275', 'X156', 'X128', 'X157', 'X229']

kolom2=['X358', 'X187', 'X186', 'X194', 'X127', 'X313', 'X316']

kolom1=['X156', 'X313', 'X314', 'X118', 'X316', 'X136', 'X315', 'X157', 'X142', 'X18', 'X158', 'X54', 'X301', 'group', 'X125', 'X351', 'X225']

kolom=list(set(kolom1+kolom2+kolom4))

res4_mod=sm.OLS(total_0.y,total_0[kolom],1).fit()

print(res4_mod.summary())

y_pred2 = res4_mod.predict(total_1[kolom])

y_pred2t = res4_mod.predict(total_0[kolom])



print(y_pred2.head())

#prediction



plt.subplot(1,4,4)

sns.distplot(total_0.y, kde=False, color='g')

sns.distplot(y_pred2, kde=False, color='b')

plt.title('Distr. of train and pred. test')



plt.subplot(1,4,1)

plt.title('True vs. Pred. train')

plt.plot([80,265], [80,265], color='g', alpha=0.3)

plt.scatter(x=total_0.y, y=y_pred2t, marker='.', alpha=0.5)

plt.scatter(x=[np.mean(train.y)], y=[np.mean(y_pred2t)], marker='o', color='red')

plt.xlabel('Real train'); plt.ylabel('Pred. train')





print(total_1.shape)





plt.figure(figsize=(18,1))

plt.plot( total_0.y[:200], color='r', linewidth=0.7)

plt.plot(y_pred[:200], color='g', linewidth=0.7)

plt.title('First 200 true and pred. trains')



from sklearn.metrics import r2_score, mean_squared_error

print('Mean error =', np.mean(total_0.y - y_pred1))

print('Train r2 =', r2_score(total_0.y, y_pred1))

print('Train r2 =', r2_score(total_0.y, y_pred2))

print('Mean error =', np.mean(total_0.y - y_pred2))

y_pred2.to_csv('stacked-models3.csv')



#res = sm.ols(formula="y ~ X3+X6+mean", data=total0).fit()  188 R2=1

#res = sm.ols(formula="y ~ X3+X6+mean", data=total1).fit()  2000 R2=0.79  X3,X6 twee groepjes relevant

#res = sm.ols(formula="y ~ X3+X6+mean", data=total2).fit()  639 R2=0.495

#res = sm.ols(formula="y ~ X3+X6+mean", data=total4).fit()  400 R2=0.29

#print(res.summary())

y_pred2.columns=['y']

y_pred2=np.exp(np.exp(y_pred2))

y_pred2
y_pred2.to_csv('stacked-models3.csv')


#analyse table

from collections import Counter



def detect_outliers(df,n,features):

    # Categorical features

    cat_cols = []

    for c in df.columns:

        if df[c].dtype == 'object':

            cat_cols.append(c)

    print('Categorical columns:', cat_cols)

    

    

    # Constant columns

    cols = df.columns.values    

    const_cols = []

    for c in cols:

        if len(df[c].unique()) == 1:

            const_cols.append(c)

    print('Constant cols:', const_cols)

    

    

    # Dublicate features

    d = {}; done = []

    cols = df.columns.values

    for c in cols:

        d[c]=[]

    for i in range(len(cols)):

        if i not in done:

            for j in range(i+1, len(cols)):

                if all(df[cols[i]] == df[cols[j]]):

                    done.append(j)

                    d[cols[i]].append(cols[j])

    dub_cols = []

    for k in d.keys():

        if len(d[k]) > 0: 

            # print k, d[k]

            dub_cols += d[k]        

    print('Dublicates:', dub_cols)



    #sklear feature selection

    import sklearn    

    from sklearn.svm import LinearSVC

    from sklearn.feature_selection import SelectFromModel

    from sklearn.ensemble import ExtraTreesClassifier



    feat=[x for x in features if x not in cat_cols]

    feat=[x for x in feat if x not in const_cols]

    feat=[x for x in feat if x not in dub_cols]



    print(feat)

    

    

    tabel = df[df['tt']==0]

    label = tabel['group'].round(0)

    clf = ExtraTreesClassifier()

    clf = clf.fit(tabel[feat], label)

    model = SelectFromModel(clf, prefit=True)

    interesting_cols = model.transform(tabel[feat])

    print('Treeclassifier cols',interesting_cols.shape)

    tabel2=pd.DataFrame(interesting_cols,index=tabel.index)

    feat2=tabel2.columns

    feat3=[]

    for ci in feat:

        for cj in feat2:

            if all(tabel[ci] == tabel2[cj]):

                feat3.append(ci) 

    print('interesting Treecolumns',feat3)

    

    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(tabel[feat],label)

    model_2 = SelectFromModel(lsvc, prefit=True)

    interesting_cols = model_2.transform(tabel[feat])

    print('LinearSVC cols',interesting_cols.shape)

    tabel2=pd.DataFrame(interesting_cols,index=tabel.index)

    feat2=tabel2.columns

    feat4=[]

    for ci in feat:

        for cj in feat2:

            if all(tabel[ci] == tabel2[cj]):

                feat4.append(ci)        

    print('interesting SVCcolumns',feat4)

    print('mixed',list(set(feat3+feat4)))    





    # Outlier detection     

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in feat:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



kolom=[x for x in total.columns if x not in ['ID']]

# detect outliers from Age, SibSp , Parch and Fare





#Outliers = detect_outliers(total0.replace([np.inf, -np.inf], np.nan).fillna(value=-1),2,kolom)

Outliers = detect_outliers(total1.replace([np.inf, -np.inf], np.nan).fillna(value=-1),2,kolom)

Outliers = detect_outliers(total2.replace([np.inf, -np.inf], np.nan).fillna(value=-1),2,kolom)

Outliers = detect_outliers(total4.replace([np.inf, -np.inf], np.nan).fillna(value=-1),2,kolom)



total.loc[Outliers_to_drop] # Show the outliers rows


