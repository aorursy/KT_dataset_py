import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sn



train = pd.read_csv("../input/mitbih-arrhythmia-database-de-chazal-class-labels/DS1_signals.csv", header=None)

labels = pd.read_csv("../input/mitbih-arrhythmia-database-de-chazal-class-labels//DS1_labels.csv", header=None)

test = pd.read_csv("../input/mitbih-arrhythmia-database-de-chazal-class-labels/DS2_signals.csv", header=None)

labels2 = pd.read_csv("../input/mitbih-arrhythmia-database-de-chazal-class-labels//DS2_labels.csv", header=None)



train['arrhytmia']=labels[0]

test['arrhytmia']=labels2[0]

total=train.append(test)
total
def clusterembed(dtrain,label,indexv):

    print('#encodings',dtrain.shape)

    dtest=dtrain[dtrain[label].isnull()==True][[indexv,label]]



    #split data or use splitted data

    X_train=dtrain[dtrain[label].isnull()==False].drop([indexv,label],axis=1).fillna(0)

    Y_train=dtrain[dtrain[label].isnull()==False][label].values

    X_test=dtrain[dtrain[label].isnull()==True].drop([indexv,label],axis=1).fillna(0)

    Y_test=np.random.random((X_test.shape[0],1)).astype('int')

    if len(X_test)==0:

        from sklearn.model_selection import train_test_split

        X_train,X_test,Y_train,Y_test = train_test_split(dtrain.drop([indexv,label],axis=1).fillna(0),dtrain[label].values,test_size=0.25,random_state=0)

    lenxtr=len(X_train)

    y=list(Y_train)+list(Y_test)

    print('splitting data train test X-y',X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)





    import matplotlib.pyplot as plt

    from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

    

    from sklearn import manifold,ensemble,decomposition

    from sklearn.model_selection import train_test_split

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis

    from sklearn.pipeline import make_pipeline

    from sklearn.preprocessing import StandardScaler

    from sklearn.decomposition import PCA,TruncatedSVD,NMF,FastICA

    from sklearn.random_projection import SparseRandomProjection

    from sklearn.cluster import AffinityPropagation

    import xgboost as xgb

    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

    

    n_neighbors = 4

    random_state = 0

    dim = len(X_train.iloc[0])

    n_classes = len(np.unique(Y_train))

    for nco in range(2,20,6):

        svd = decomposition.TruncatedSVD(n_components=2*nco)        

        pca = make_pipeline(StandardScaler(),PCA(n_components=nco, random_state=random_state))

        # Use a classifier to evaluate the methods

        knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        knn = RandomForestClassifier()

        #knn= AffinityPropagation(damping=0.9, preference=-200)

        #knn = xgb.XGBClassifier(n_estimators=50, max_depth = 9, learning_rate=0.01, subsample=0.75, random_state=11)



        # Make a list of the methods to be compared

        dim_reduction_methods = [('PCA', pca),('SVD2',svd)]

        # plt.figure()

        for i, (name, model) in enumerate(dim_reduction_methods):

            # Fit the method's model

            model.fit(X_train, Y_train) #print(model.transform(X_train)[:10])

            # Fit a nearest neighbor classifier on the embedded training set

            knn.fit(model.transform(X_train), Y_train)

            # Compute the nearest neighbor accuracy on the embedded test set

            print(classification_report(knn.predict(model.transform(X_train)),Y_train))

            if len(dtest)==0:

                acc_knn = knn.score(model.transform(X_test), Y_test)

                print(name,acc_knn)

            else:

                

                testpredi=knn.predict(model.transform(X_test))

                submit = pd.DataFrame({indexv: dtest[indexv],label: testpredi})

                submit[label]=submit[label].astype('int')

                filenaam='subm_emb'+name+'_'+str(nco)+'.csv'

                submit.to_csv(path_or_buf =filenaam, index=False)

                acc_knn = knn.score(model.transform(X_train), Y_train)

            

            # Embed the data set in 2 dimensions using the fitted model

            X_embedded = model.transform(dtrain.drop([indexv,label],axis=1).fillna(0).values) #print(X_embedded[:10])

            # Plot the projected points and show the evaluation score

            plt.figure()



            plt.scatter(X_embedded[:, 0], X_embedded[:, 1],c=y, s=30, cmap='Set1')

            plt.title("{}, rFor (k={})\nTest accuracy = {:.3f}".format(name,

                                                              nco,

                                                              acc_knn))

            plt.show()

    return







clusterembed(total.reset_index(),'arrhytmia','index') 