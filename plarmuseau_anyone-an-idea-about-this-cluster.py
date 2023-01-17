import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
samp = pd.read_csv('../input/sample_submission.csv')

ytrain=train['Cover_Type']  #first cop
#train=train/10*10
train=train.drop('Cover_Type',axis=1)

totaal=train.append(test)
train.head()
totaal['HF1'] = totaal['Horizontal_Distance_To_Hydrology']+totaal['Horizontal_Distance_To_Fire_Points']
totaal['HF2'] = abs(totaal['Horizontal_Distance_To_Hydrology']-totaal['Horizontal_Distance_To_Fire_Points'])
totaal['HR1'] = abs(totaal['Horizontal_Distance_To_Hydrology']+totaal['Horizontal_Distance_To_Roadways'])
totaal['HR2'] = abs(totaal['Horizontal_Distance_To_Hydrology']-totaal['Horizontal_Distance_To_Roadways'])
totaal['FR1'] = abs(totaal['Horizontal_Distance_To_Fire_Points']+totaal['Horizontal_Distance_To_Roadways'])
totaal['FR2'] = abs(totaal['Horizontal_Distance_To_Fire_Points']-totaal['Horizontal_Distance_To_Roadways'])
totaal['ele_vert'] = totaal.Elevation-totaal.Vertical_Distance_To_Hydrology

totaal['slope_hyd'] = (totaal['Horizontal_Distance_To_Hydrology']**2+totaal['Vertical_Distance_To_Hydrology']**2)**0.5
#Mean distance to Amenities 
totaal['Mean_Amenities']=(totaal.Horizontal_Distance_To_Fire_Points + totaal.Horizontal_Distance_To_Hydrology + totaal.Horizontal_Distance_To_Roadways) / 3 
#Mean Distance to Fire and Water 
totaal['Mean_Fire_Hyd']=(totaal.Horizontal_Distance_To_Fire_Points + totaal.Horizontal_Distance_To_Hydrology) / 2 
totaal['W3S3839']=(totaal['Soil_Type38']+totaal['Soil_Type39'])*totaal['Wilderness_Area3']
totaal['W1S2922']=(totaal['Soil_Type29']+totaal['Soil_Type22'])*totaal['Wilderness_Area1']
def cohen_effect_size(X, y):
    """Calculates the Cohen effect size of each feature.
    
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target vector relative to X
        Returns
        -------
        cohen_effect_size : array, shape = [n_features,]
            The set of Cohen effect values.
        Notes
        -----
        Based on https://github.com/AllenDowney/CompStats/blob/master/effect_size.ipynb
    """
    print(X.shape,y.shape)
    group1, group2 = X[y<y.median()], X[y>y.median()]
    diff = group1.mean() - group2.mean()
    var1, var2 = group1.var(), group2.var()
    n1, n2 = group1.shape[0], group2.shape[0]
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d

excluded_feats = [] #['SK_ID_CURR']

features = [f_ for f_ in totaal.columns if f_ not in excluded_feats]
print('Number of features %d' % len(features),totaal.shape,ytrain.shape)
effect_sizes = cohen_effect_size(totaal[:len(train)], ytrain.fillna(0))
effect_sizes.reindex(effect_sizes.abs().sort_values(ascending=False).nlargest(50).index)[::-1].plot.barh(figsize=(6, 10));
print('Features with the 30 largest effect sizes')
significant_features = [f for f in features if np.abs(effect_sizes.loc[f]) > 0.02]
print('Significant features %d: %s' % (len(significant_features), significant_features))

def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    a=a.values  #if panda dataframe
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def mapping6(data1,data2,data3,y1,k,sigfeat):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier,VotingClassifier, RandomForestClassifier
    from sklearn.linear_model import PassiveAggressiveClassifier,Perceptron,SGDClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    #data1 the unknown start data
    #data2 the database you want to link with and classify with

    datatot=(data3)[significant_features]
    print(data1.shape,data2.shape,k,datatot.shape)    
    

    #svd
    if k>1:
        svd = TruncatedSVD(n_components=k, n_iter=7, random_state=42)
        U123=svd.fit_transform(normalized(datatot))
        kleur=['b']*len(data1)+['r']*len(data2)
        pd.DataFrame(U123[:,:2]).plot.scatter(x=0,y=1,c=kleur)        
        Xr=svd.inverse_transform(U123)
        #U123,s123,V123=svds(vect.fit_transform(data1[veld1].append(data2[veld2])),k=k) #.append(data3[veld3])
        print("datasvd",U123.shape)
        #temp=np.concatenate( (U123[len(data1):len(data1)+len(data2)]*s123[:k]   , dwm[len(data1):len(data1)+len(data2)]), axis=1 )
        temp=Xr[len(data1):len(data1)+len(data2)] #*s123[:k]
        U2=pd.DataFrame( temp  , index= data2.index)
    else:
        U2=pd.DataFrame(datatot.values[len(data1):len(data1)+len(data2)], index= data2.index)
        
    
    if k>1:
        temp=Xr[:len(data1)] #*s123[:k]
        U1=pd.DataFrame( temp , index=data1.index )
    else:
        U1=pd.DataFrame( datatot.values[:len(data1)], index=data1.index )
    
    et = ExtraTreesClassifier(n_estimators=25, max_depth=300, min_samples_split=5, min_samples_leaf=1, random_state=None, min_impurity_decrease=1e-7)
    
    clf1 = SVC()
    clf2 = KNeighborsClassifier()
    clf3 = GradientBoostingClassifier()
    clf4 = XGBClassifier()
    #et = SVC()
    clf5 = RandomForestClassifier()
    model = VotingClassifier(estimators=[('svc', clf1), ('knn', clf2), ('gbc', clf3), ('xgbc', clf4), ('rf', clf5)], voting='hard')

    #y_pred = eclf1.predict(x)
    
    # TRAINING
    #et = SGDClassifier(n_jobs=4,max_iter=100)
    model = OneVsRestClassifier(et)  
    #temp=pd.DataFrame( dwm[ len(data1):len(data1)+len(data2) ] )
    temp=U2 #.T.append(temp.T)   #U2  append word tfidf vector
    print('U2',temp.shape)
    model.fit(U1,y1)
    
    print( (model.predict(U1)==y1).mean()*100 ) 
    
    #PREDICTING
    #temp=pd.DataFrame(dwm[ :len(data1)] )
    temp=U2 #.T.append(temp.T)
    data2['pre']=model.predict(U2)
    
    return data2
test=mapping6(train,test,totaal,ytrain,2,significant_features)
sub_eclf = pd.DataFrame()
sub_eclf['Id'] = test['Id']
sub_eclf['Cover_Type'] = test['pre']

sub_eclf.to_csv('submission_eclf.csv', index=False)
test